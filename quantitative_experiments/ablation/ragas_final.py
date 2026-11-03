"""RAGAS (Faithfulness y Answer Relevancy) y latencia del pipeline completo con la arquitectura final (denso entrenado).

Lee la traza que escribe full_pipeline.py con `--with-generation --concurrency 1 --no-filter-cache` (una fila por pregunta, con la respuesta,
la decisión del filtro de dominio, si el umbral dejó sin fuentes, y la latencia total y por etapa) y calcula:
  * RAGAS reference-free sobre las preguntas que el pipeline respondió (pasaron el filtro y tuvieron al menos un chunk sobre el umbral):
    Faithfulness y Answer Relevancy, con los contextos que realmente recibió el generador (los chunks del ranking sobre el umbral, en orden).
    No se calculan Context Recall ni Context Precision (requieren respuestas de referencia o juzgan contra la respuesta generada).
  * Resumen de cada métrica: media, desviación, mediana e IC95 % por bootstrap sobre los chunks de origen (clusters).
  * Latencia por respuesta (total y por etapa): media, mediana y percentil 95, sobre todas las preguntas y sobre las respondidas.

Modelo juez: `--judge-model` (por defecto openai/gpt-5.6-luna, el que cita el manuscrito) y embeddings de Answer Relevancy: settings.ragas_realtime_embeddings_model
(qwen/qwen3-embedding-8b); ambos por OpenRouter y quedan registrados en el resumen. El script es REANUDABLE: cada pregunta evaluada se guarda en
ragas_parcial.jsonl y al volver a ejecutarlo se omiten las ya hechas (útil si se agota el crédito). Con `--sample N` evalúa una muestra aleatoria reproducible.

Uso:
    uv run python -m quantitative_experiments.ablation.ragas_final --trace <dense_finetuned.trace.jsonl> --out-dir <carpeta>
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ragas.embeddings import OpenAIEmbeddings  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import AnswerRelevancy, Faithfulness  # noqa: E402

from app.core.config import settings  # noqa: E402

EXP = Path(__file__).resolve().parents[1]


def boot_ci(values: np.ndarray, clusters: np.ndarray, n: int = 10_000, seed: int = 0) -> list[float]:
    """IC95 % de la media remuestreando clusters (chunks de origen) completos."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    by = {c: values[clusters == c] for c in ids}
    means = []
    for _ in range(n):
        pick = rng.choice(ids, size=len(ids), replace=True)
        means.append(np.concatenate([by[c] for c in pick]).mean())
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    return {"n": int(len(x)), "media": float(x.mean()), "desv": float(x.std(ddof=1)) if len(x) > 1 else 0.0, "mediana": float(x.median()),
            "p95": float(np.percentile(x, 95)), "min": float(x.min()), "max": float(x.max())}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--dataset-path", type=Path, default=EXP / "test_set" / "eval_test_dataset.jsonl")
    ap.add_argument("--corpus", type=Path, default=EXP / "data" / "inputs" / "document_chunks_all.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--judge-model", default="openai/gpt-5.6-luna", help="juez LLM de RAGAS (id de OpenRouter)")
    ap.add_argument("--max-tokens", type=int, default=20_000, help="tope de tokens de salida del juez")
    ap.add_argument("--sample", type=int, default=0, help="evalúa solo N preguntas respondidas elegidas al azar (semilla --seed); 0 = todas")
    ap.add_argument("--seed", type=int, default=520)
    ap.add_argument("--retry-missing", action="store_true", help="solo reevalúa las preguntas sin Faithfulness en el CSV existente y las fusiona")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    corpus = {c["chunk_id"]: c for c in map(json.loads, args.corpus.read_text(encoding="utf-8").splitlines()) if c}
    dataset = {d["dataset_id"]: d for d in map(json.loads, args.dataset_path.read_text(encoding="utf-8").splitlines()) if d}
    rows = [json.loads(l) for l in args.trace.read_text(encoding="utf-8").splitlines() if l.strip()][: args.limit]

    answered = [r for r in rows if r["is_relevant"] and not r["no_results"] and r.get("answer")]
    if args.sample and args.sample < len(answered):
        rng = np.random.default_rng(args.seed)
        answered = [answered[i] for i in sorted(rng.choice(len(answered), size=args.sample, replace=False))]
    print(f"preguntas: {len(rows)} | respondidas: {len(answered)} | rechazadas por el filtro: {sum(not r['is_relevant'] for r in rows)} "
          f"| sin fuentes sobre el umbral: {sum(r['no_results'] for r in rows)}")

    client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)
    llm = llm_factory(args.judge_model, provider="openai", client=client, max_tokens=args.max_tokens)
    emb = OpenAIEmbeddings(client=client, model=settings.ragas_realtime_embeddings_model)
    faith, rel = Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=emb)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(r: dict) -> dict:
        q = dataset[r["dataset_id"]]["query"]
        n_ctx = r["n_above_threshold"] or len(r["ranked"][:10])
        contexts = [corpus[c]["text"] for c in r["ranked"][:n_ctx]]
        out = {"dataset_id": r["dataset_id"], "faithfulness": None, "answer_relevancy": None}
        async with sem:
            try:
                out["faithfulness"] = (await faith.ascore(user_input=q, response=r["answer"], retrieved_contexts=contexts)).value
            except Exception as exc:  # noqa: BLE001
                print(f"  faithfulness error {r['dataset_id']}: {exc}")
            try:
                out["answer_relevancy"] = (await rel.ascore(user_input=q, response=r["answer"])).value
            except Exception as exc:  # noqa: BLE001
                print(f"  answer_relevancy error {r['dataset_id']}: {exc}")
        return out

    partial_path = args.out_dir / "ragas_parcial.jsonl"
    done = {}
    if partial_path.exists():
        for l in partial_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                row = json.loads(l)
                done[row["dataset_id"]] = row
    todo = [r for r in answered if r["dataset_id"] not in done
            or (args.retry_missing and done[r["dataset_id"]].get("faithfulness") is None)]
    print(f"ya evaluadas: {len(answered) - len(todo)} | por evaluar: {len(todo)} | juez: {args.judge_model}")

    async def one_saved(r: dict) -> dict:
        row = await one(r)
        with open(partial_path, "a", encoding="utf-8") as f:   # se guarda al terminar cada pregunta (reanudable)
            f.write(json.dumps(row) + "\n")
        return row

    for row in await asyncio.gather(*(one_saved(r) for r in todo)):
        done[row["dataset_id"]] = row
    scores = pd.DataFrame([done[r["dataset_id"]] for r in answered if r["dataset_id"] in done])
    lat = pd.DataFrame([{"dataset_id": r["dataset_id"], "answered": r in answered, "latency_ms": r["latency_ms"], **{f"ms_{k}": v for k, v in r["node_ms"].items()}} for r in rows])
    full = lat.merge(scores, on="dataset_id", how="left")
    full["source_chunk"] = full["dataset_id"].map(lambda d: dataset[d]["positive_chunk_ids"][0])
    full.to_csv(args.out_dir / "ragas_y_latencia_por_pregunta.csv", index=False)

    summary = {"juez_llm": args.judge_model, "embeddings_answer_relevancy": settings.ragas_realtime_embeddings_model,
               "preguntas": len(rows), "respondidas": len(answered), "rechazadas_filtro": int(sum(not r["is_relevant"] for r in rows)),
               "sin_fuentes_umbral": int(sum(r["no_results"] for r in rows))}
    for m in ("faithfulness", "answer_relevancy"):
        s = full.loc[full["answered"], m].dropna()
        if len(s):
            c = full.loc[s.index, "source_chunk"].to_numpy()
            summary[m] = {**stats(s), "ic95_bootstrap_por_chunk": boot_ci(s.to_numpy(), c)}
    ms_cols = [c for c in full.columns if c.startswith("ms_")]
    summary["latencia_ms_todas"] = {"total": stats(full["latency_ms"]), **{c[3:]: stats(full[c]) for c in ms_cols}}
    summary["latencia_ms_respondidas"] = {"total": stats(full.loc[full["answered"], "latency_ms"]),
                                          **{c[3:]: stats(full.loc[full["answered"], c]) for c in ms_cols}}
    (args.out_dir / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if not k.startswith("latencia")}, ensure_ascii=False, indent=1))
    t = summary["latencia_ms_respondidas"]["total"]
    print(f"latencia total (respondidas): media {t['media']/1000:.2f} s | mediana {t['mediana']/1000:.2f} s | p95 {t['p95']/1000:.2f} s")


if __name__ == "__main__":
    asyncio.run(main())
