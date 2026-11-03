"""Genera los "runs" de recuperación (ranking top-N por pregunta y por configuración) que consume el
pooling de qrels (unseen_dataset/05_pool_qrels.py).

Formato de salida, un JSON por configuración: {"<dataset_id>": ["chunk_id_1", "chunk_id_2", ...]} (ordenado).
Usa exactamente las mismas funciones de recuperación que eval_ablation.py (misma lista "comparable").

Requiere el model service y MongoDB Atlas disponibles (ver .env). No llama a ningún LLM.

Uso:
    uv run python -m quantitative_experiments.ablation.build_runs \
        --dataset-path quantitative_experiments/unseen_dataset/outputs/unseen_eval_dataset.jsonl \
        --out-dir quantitative_experiments/results/runs --depth 100 --reranker-backend openrouter
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import settings  # noqa: E402
from app.services import mongodb  # noqa: E402
from quantitative_experiments.ablation.eval_ablation import (  # noqa: E402
    CORPUS_PATH,
    EMBED_ENDPOINTS,
    HYBRID_FUSION,
    config_arg,
    get_config_docs,
    parse_config,
)
from scripts.eval_dataset_ragas import load_dataset  # noqa: E402

MODEL_DIRS = {"base": "modelo_base_onnx_int8", "finetuned": "modelo_onnx_int8"}
PROBE_CHUNKS = 4
MIN_COSINE = 0.999


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


async def verify_index(variant: str) -> None:
    """Aborta si los vectores guardados en Mongo no son los que sirve hoy el endpoint de la variante.
    Detecta un reindexado con otro modelo, un modelo cambiado en el model service o un índice sin sincronizar."""
    with open(CORPUS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    probe = [chunks[i] for i in np.linspace(0, len(chunks) - 1, PROBE_CHUNKS, dtype=int)]
    col = mongodb.get_chunks_collection()
    url = settings.model_service_url.rstrip("/") + EMBED_ENDPOINTS[variant]
    async with httpx.AsyncClient(timeout=60) as client:
        for c in probe:
            stored = np.array((await col.find_one({"chunk_id": c["chunk_id"]}, {"embedding": 1}))["embedding"])
            live = np.array((await client.post(url, json={"text": c["text"]})).json()["embedding"])
            cos = float(stored @ live / np.linalg.norm(stored) / np.linalg.norm(live))
            if cos < MIN_COSINE:
                raise SystemExit(f"Mongo no tiene embeddings '{variant}' vigentes: {c['chunk_id']} cos={cos:.4f} < {MIN_COSINE}. "
                                 f"Reindexa con scripts/ingest_chunks.py y reintenta.")


async def model_meta(variant: str, model_service_dir: Path) -> dict:
    async with httpx.AsyncClient(timeout=30) as client:
        health = (await client.get(settings.model_service_url.rstrip("/") + "/health")).json()
    model_dir = model_service_dir / MODEL_DIRS[variant]
    onnx = next((model_dir / n for n in ("model.onnx", "model_quantized.onnx") if (model_dir / n).exists()), None)
    files = [onnx] + ([onnx.with_name(onnx.name + ".data")] if onnx and onnx.with_name(onnx.name + ".data").exists() else [])
    return {"variant": variant, "endpoint": EMBED_ENDPOINTS[variant], "model_files": [str(f) for f in files if f],
            "model_sha256": {f.name: _sha256(f) for f in files if f}, "service_health": health}


async def build_run(config: str, dataset: list[dict], depth: int, backend: str, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> tuple[str, list[str]]:
        async with sem:
            query = item.get("normalized_query") or item["query"]
            docs = await get_config_docs(config, query, reranker_backend=backend)
            return item["dataset_id"], [d["chunk_id"] for d in docs["comparable"][:depth]]

    return dict(await asyncio.gather(*(one(it) for it in dataset)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument("--samples", type=int, default=10_000)
    ap.add_argument("--configs", nargs="+", required=True, type=config_arg,
                    help="p. ej. bm25_only dense_finetuned hybrid_finetuned hybrid_finetuned_rerank50 (ver eval_ablation.py)")
    ap.add_argument("--depth", type=int, default=100, help="Longitud del ranking guardado por pregunta")
    ap.add_argument("--reranker-backend", choices=["local", "openrouter"], default="local",
                    help="local = reranker ONNX del model service (gratis); openrouter = Voyage (de pago)")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-service-dir", type=Path, default=Path(os.environ.get("MODEL_SERVICE_DIR", "../model-service")),
                    help="Carpeta del model service (para registrar el hash del modelo usado)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_path, args.samples)
    meta_cache: dict[str, dict] = {}
    for config in args.configs:
        spec = parse_config(config)
        variant = spec["variant"]
        if variant is not None:
            await verify_index(variant)
            if variant not in meta_cache:
                meta_cache[variant] = await model_meta(variant, args.model_service_dir)
        run = await build_run(config, dataset, args.depth, args.reranker_backend, args.concurrency)
        path = args.out_dir / f"{config}.json"
        path.write_text(json.dumps(run, ensure_ascii=False), encoding="utf-8")
        if variant is not None:
            await verify_index(variant)  # el índice no cambió durante la corrida
            (args.out_dir / f"{config}.meta.json").write_text(
                json.dumps({"config": config, **meta_cache[variant],
                            "rrf": HYBRID_FUSION if spec["kind"] == "hybrid" else None,
                            "rerank_k": spec["rerank_k"], "reranker_backend": args.reranker_backend if spec["rerank_k"] else None},
                           ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{config}: {len(run)} preguntas -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
