from app.core.config import settings
from app.graph.state import RAGState
from app.services import mongodb


def atlas_score_to_cosine(score: float) -> float:
    """Atlas Vector Search normaliza la similitud coseno a (1 + coseno) / 2."""
    return 2.0 * score - 1.0


def apply_score_threshold(docs: list[dict], threshold: float) -> list[dict]:
    """Conserva, en su orden, los chunks cuyo puntaje de Atlas supera el umbral estrictamente."""
    return [d for d in docs if d.get("score", 0.0) > threshold]


async def bm25_search_node(state: RAGState) -> dict:
    # No se usa en producción (app/graph/rag_graph.py) — solo lo llaman
    # scripts/eval_dataset_ragas.py y tests/test_rrf_fusion.py como punto de
    # comparación del experimento de ablación ya descartado (hybrid_*).
    results = await mongodb.bm25_search(state["synthesized_query"], k=20)
    return {"bm25_results": results}


async def semantic_search_node(state: RAGState) -> dict:
    # Único paso de recuperación en producción (sin BM25/RRF/reranker, decisión
    # tomada tras el experimento de ablación — el embedding finetuneado solo
    # rindió mejor que cualquier combinación probada). k=10 (settings.retrieval_k): el corte
    # validado en la ablación (config dense_finetuned). Solo se pasan al generador los chunks cuyo
    # puntaje supera settings.retrieval_score_threshold; si no queda ninguno, el grafo responde con
    # el mensaje fijo de "sin fuentes suficientes" (route_retrieval -> no_results).
    # "reranked_docs" se mantiene como nombre de la clave que espera
    # generate_node — ya no hay reranker, pero renombrarla implicaría tocar
    # generate_node/source_grounding_node y los scripts de evaluación que
    # también usan esa misma clave como interchange genérico.
    results = await mongodb.vector_search(state["query_embedding"], k=settings.retrieval_k)
    return {
        "semantic_results": results,
        "reranked_docs": apply_score_threshold(results, settings.retrieval_score_threshold),
    }
