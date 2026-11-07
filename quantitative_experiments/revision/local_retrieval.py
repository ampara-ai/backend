"""Recuperación local con búsqueda exacta sobre los 893 chunks (sin MongoDB ni model service).

Escribe un ranking de profundidad --depth en el formato de results/runs/<config>.json ({dataset_id: [chunk_id, ...]}),
de modo que reproduce_tables, compare, el pooling y ragas_final lo leen igual que las corridas publicadas. Junto al ranking
guarda <name>.meta.json: especificación del modelo, dispositivo, precisión, parámetros, prefijos de consulta y tiempos.

Los embeddings del corpus se guardan en results/revision/embeddings/ y se reutilizan entre conjuntos de preguntas
(Test A, Test B, consultas reescritas). Con --latency-sample N mide además la latencia de codificar UNA consulta en CPU
(mediana de N), la columna de eficiencia de la tabla de líneas de base.

Uso (ejemplos):
    # FP16 base y fine-tuned (MAJOR #36: comparación primaria fuera de INT8)
    uv run python -m quantitative_experiments.revision.local_retrieval --model cls:BAAI/bge-m3 --name dense_base_fp16
    uv run python -m quantitative_experiments.revision.local_retrieval --model cls:<checkpoint-256> --name dense_finetuned_fp16
    # Reproducir la corrida INT8 publicada sin MongoDB
    uv run python -m quantitative_experiments.revision.local_retrieval --model onnx:<model-service>/modelo_onnx_int8 --name dense_finetuned_int8_local
    # Líneas de base zero-shot y modos de BGE-M3
    uv run python -m quantitative_experiments.revision.local_retrieval --model st:Qwen/Qwen3-Embedding-0.6B --name qwen3_0.6b
    uv run python -m quantitative_experiments.revision.local_retrieval --model bgem3-sparse:BAAI/bge-m3 --name bgem3_sparse
    # Otro conjunto de preguntas (Test B) o consultas reescritas
    ... --dataset results/revision/test_b/eval_test_b_dataset.jsonl --qrels results/revision/test_b/eval_test_b_qrels.jsonl
    ... --dataset results/revision/rewrite/<modelo>.jsonl --query-field query_rewritten
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    CORPUS_PATH, OUT, STRICT_QRELS, TEST_DATASET, corpus, fmt_ci, load_dataset, load_qrels, per_query, summary,
)
from quantitative_experiments.revision.embed import make_encoder, timed


def slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in text)[:120]


def topk(scores: np.ndarray, ids: list[str], depth: int) -> list[list[str]]:
    """Orden por puntaje descendente; empates por chunk_id para que el ranking sea determinista."""
    order = np.lexsort((np.array(ids)[None, :].repeat(len(scores), 0), -scores), axis=1)
    return [[ids[j] for j in row[:depth]] for row in order]


def model_stamp(spec: str) -> str:
    """Para modelos en una carpeta local, la fecha de sus archivos: si se reentrena en la misma ruta, la caché no se reutiliza."""
    folder = Path(spec.split(":", 1)[1])
    if not folder.is_dir():
        return ""
    return str(max((f.stat().st_mtime_ns for f in folder.rglob("*") if f.is_file()), default=0))


def corpus_embeddings(enc, chunk_ids: list[str], texts: list[str], batch_size: int, cache_dir: Path):
    if enc.kind != "dense":  # sparse y ColBERT: en memoria (los vectores multi-token ocupan ~1 GB)
        return timed(enc.encode_documents, texts, batch_size)
    key = hashlib.sha256(f"{enc.spec}|{model_stamp(enc.spec)}|{enc.max_length}|{enc.dtype}|{CORPUS_PATH.stat().st_size}"
                         .encode()).hexdigest()[:16]
    path = cache_dir / slug(enc.spec) / f"corpus_{key}.npy"
    if path.exists():
        return np.load(path), 0.0
    emb, secs = timed(enc.encode_documents, texts, batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    (path.parent / f"corpus_{key}.ids.json").write_text(json.dumps(chunk_ids), encoding="utf-8")
    return emb, secs


def retrieve(model: str, dataset: list[dict], query_field: str = "query", depth: int = 100, device: str = "auto",
             max_length: int = 2048, fp16: bool = True, doc_batch: int = 4, query_batch: int = 32, encoder=None):
    """Ranking exacto de las preguntas de `dataset` sobre el corpus; devuelve (corrida, codificador, segundos por etapa)."""
    C = corpus()
    chunk_ids = sorted(C)
    enc = encoder or make_encoder(model, device, max_length, fp16=fp16)
    doc_emb, doc_secs = corpus_embeddings(enc, chunk_ids, [C[c]["text"] for c in chunk_ids], doc_batch, OUT / "embeddings")
    queries = [d.get(query_field) or d["query"] for d in dataset]
    q_emb, q_secs = timed(enc.encode_queries, queries, query_batch)
    t0 = time.perf_counter()
    ranked = topk(np.asarray(enc.scores(q_emb, doc_emb)), chunk_ids, depth)
    secs = {"corpus_encode": doc_secs, "queries_encode": q_secs, "search": time.perf_counter() - t0}
    return {d["dataset_id"]: r for d, r in zip(dataset, ranked)}, enc, secs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="nombre de la configuración (archivo de salida)")
    ap.add_argument("--dataset", type=Path, default=TEST_DATASET)
    ap.add_argument("--query-field", default="query")
    ap.add_argument("--qrels", type=Path, default=None, help="para imprimir métricas; por defecto los estrictos si --dataset es Test A")
    ap.add_argument("--out-dir", type=Path, default=None, help="por defecto results/revision/runs/<nombre del dataset>")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--doc-batch", type=int, default=4)
    ap.add_argument("--query-batch", type=int, default=32)
    ap.add_argument("--latency-sample", type=int, default=0, help="mide la codificación de N consultas de a una en CPU")
    args = ap.parse_args()

    dataset = load_dataset(args.dataset)
    out_dir = args.out_dir or OUT / "runs" / args.dataset.stem
    run, enc, secs = retrieve(args.model, dataset, args.query_field, args.depth, args.device, args.max_length,
                              not args.fp32, args.doc_batch, args.query_batch)
    print(f"{args.model}: {enc.info()}")
    queries = [d.get(args.query_field) or d["query"] for d in dataset]

    latency = None
    if args.latency_sample:
        cpu = make_encoder(args.model, "cpu", args.max_length, fp16=False) if enc.device != "cpu" else enc
        times = [timed(cpu.encode_queries, [q], 1)[1] for q in queries[: args.latency_sample]]
        latency = {"cpu_query_encode_ms_median": 1000 * statistics.median(times), "n": len(times)}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.name}.json").write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
    meta = {"name": args.name, **enc.info().__dict__, "dataset": str(args.dataset), "query_field": args.query_field,
            "depth": args.depth, "search": "exact", "corpus_sha256": hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest(),
            "seconds": secs, "latency": latency,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out_dir / f"{args.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out_dir / args.name}.json")

    qrels_path = args.qrels or (STRICT_QRELS if args.dataset.resolve() == TEST_DATASET.resolve() else None)
    if qrels_path:
        qrels = load_qrels(qrels_path)
        qids = [d["dataset_id"] for d in dataset]
        s = summary(run, qrels, qids)
        print("  ".join(f"{k}={v:.3f}" for k, v in s.items()))


if __name__ == "__main__":
    main()
