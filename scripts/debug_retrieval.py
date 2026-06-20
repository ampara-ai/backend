"""
Diagnóstico de retrieval para las primeras 10 preguntas del dataset.
Muestra chunks recuperados antes y después del reranker, marcando cuáles
coinciden con los positive_chunk_ids del dataset.

Uso:
    uv run python scripts/debug_retrieval.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import settings
from app.graph.nodes.reranker import rerank_node
from app.graph.nodes.retrieval import bm25_search_node, semantic_search_node
from app.graph.nodes.rrf_fusion import rrf_fusion_node
from app.services import model_service

DATASET_PATH = Path(__file__).resolve().parents[1] / "questions_with_positive_context.jsonl"
N = 10
HIT  = "✓"
MISS = "✗"
SEP  = "─" * 80


def load_dataset(path: Path, n: int) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def fmt_chunk(doc: dict, positives: set[str], extra: str = "") -> str:
    cid = doc.get("chunk_id", "?")
    title = doc.get("title", "")[:55]
    hit = HIT if cid in positives else MISS
    return f"  {hit} {cid:<45} {title:<55} {extra}"


async def run_query(item: dict) -> None:
    query: str = item["normalized_query"]
    positives: set[str] = set(item["positive_chunk_ids"])
    dataset_id: str = item.get("dataset_id", "?")

    print(f"\n{'═' * 80}")
    print(f"  {dataset_id}")
    print(f"  Q: {query[:100]}")
    print(f"  Positivos esperados ({len(positives)}): {', '.join(sorted(positives))}")
    print(SEP)

    # 1. Embedding
    embedding = await model_service.get_embedding(query, use_base=False)

    # 2. BM25 + semántica en paralelo
    bm25_state, semantic_state = await asyncio.gather(
        bm25_search_node({"synthesized_query": query}),
        semantic_search_node({"synthesized_query": query, "query_embedding": embedding}),
    )

    bm25_docs = bm25_state["bm25_results"]
    semantic_docs = semantic_state["semantic_results"]

    # 3. RRF fusion
    fused_state = rrf_fusion_node({
        "bm25_results": bm25_docs,
        "semantic_results": semantic_docs,
    })
    fused = fused_state["fused_results"]

    # ── PRE-RERANKER ──────────────────────────────────────────────────────────
    print(f"\n  PRE-RERANKER (RRF top-{len(fused)}):")
    print(f"  {'':2} {'chunk_id':<45} {'title':<55} {'rrf_score':>9}")
    for doc in fused:
        extra = f"{doc.get('rrf_score', 0):.5f}"
        print(fmt_chunk(doc, positives, extra))

    pre_hits = sum(1 for d in fused if d.get("chunk_id") in positives)
    print(f"\n  Hits pre-reranker: {pre_hits}/{len(positives)} positivos encontrados")

    # 4. Reranker
    reranked_state = await rerank_node({
        "fused_results": fused,
        "synthesized_query": query,
    })
    reranked = reranked_state.get("reranked_docs") or []
    no_results = reranked_state.get("no_results", False)
    top_score = reranked_state.get("rerank_score", 0.0)

    # ── POST-RERANKER ─────────────────────────────────────────────────────────
    print(f"\n  POST-RERANKER (top-{len(reranked)})  "
          f"[threshold={settings.reranker_score_threshold}  top_score={top_score:.4f}  "
          f"no_results={no_results}]:")
    print(f"  {'':2} {'chunk_id':<45} {'title':<55} {'rerank_score':>12}")
    for doc in reranked:
        extra = f"{doc.get('rerank_score', 0):.4f}"
        print(fmt_chunk(doc, positives, extra))

    post_hits = sum(1 for d in reranked if d.get("chunk_id") in positives)
    print(f"\n  Hits post-reranker: {post_hits}/{len(positives)} positivos encontrados")


async def main() -> None:
    print(f"Cargando primeras {N} preguntas de {DATASET_PATH.name}...")
    dataset = load_dataset(DATASET_PATH, N)
    print(f"Loaded {len(dataset)} items. MODEL_SERVICE_URL={settings.model_service_url}\n")

    for item in dataset:
        await run_query(item)

    print(f"\n{'═' * 80}")
    print("Diagnóstico completado.")


if __name__ == "__main__":
    asyncio.run(main())
