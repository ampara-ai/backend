_RRF_K = 60
_TOP_N = 10


# No se usa en producción (app/graph/rag_graph.py) — solo recupera con el
# embedding finetuneado, sin BM25 que fusionar. Se mantiene como punto de
# comparación del experimento de ablación (scripts/eval_ablation.py, configs
# hybrid_no_rerank/hybrid_rerank) y de scripts/eval_dataset_ragas.py.
def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def rrf_fusion_node(state: dict) -> dict:
    bm25 = state.get("bm25_results") or []
    semantic = state.get("semantic_results") or []

    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for rank, doc in enumerate(bm25):
        doc_id = doc["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        docs[doc_id] = doc

    for rank, doc in enumerate(semantic):
        doc_id = doc["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        docs[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:_TOP_N]
    fused = [{**docs[doc_id], "rrf_score": score} for doc_id, score in ranked]
    return {"fused_results": fused}
