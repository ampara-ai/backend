from app.core.config import settings
from app.graph.state import RAGState
from app.services import model_service

_TOP_N = 5


# No se usa en producción (app/graph/rag_graph.py) — el experimento de
# ablación (scripts/eval_ablation.py, config hybrid_rerank) mostró que el
# reranker empeora el retrieval del embedding finetuneado en este dominio,
# incluso sin BM25 de por medio. Se mantiene solo como punto de comparación.
async def rerank_node(state: RAGState) -> dict:
    fused = state.get("fused_results") or []

    if not fused:
        return {"reranked_docs": [], "rerank_score": 0.0, "no_results": True}

    texts = [doc["text"] for doc in fused]
    ranked = await model_service.rerank(state["synthesized_query"], texts)

    # Map reranker results back to original doc dicts
    reranked = []
    for item in ranked[:_TOP_N]:
        original_doc = fused[item["index"]]
        reranked.append({**original_doc, "rerank_score": item["score"]})

    top_score = reranked[0]["rerank_score"] if reranked else 0.0
    no_results = top_score < settings.reranker_score_threshold

    return {
        "reranked_docs": reranked,
        "rerank_score": top_score,
        "no_results": no_results,
    }
