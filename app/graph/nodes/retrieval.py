from app.graph.state import RAGState
from app.services import mongodb


async def bm25_search_node(state: RAGState) -> dict:
    # No se usa en producción (app/graph/rag_graph.py) — solo lo llaman
    # scripts/eval_dataset_ragas.py y tests/test_rrf_fusion.py como punto de
    # comparación del experimento de ablación ya descartado (hybrid_*).
    results = await mongodb.bm25_search(state["synthesized_query"], k=20)
    return {"bm25_results": results}


async def semantic_search_node(state: RAGState) -> dict:
    # Único paso de recuperación en producción (sin BM25/RRF/reranker, decisión
    # tomada tras el experimento de ablación — el embedding finetuneado solo
    # rindió mejor que cualquier combinación probada). k=10: el corte validado
    # en scripts/eval_ablation.py (config dense_finetuned).
    # "reranked_docs" se mantiene como nombre de la clave que espera
    # generate_node — ya no hay reranker, pero renombrarla implicaría tocar
    # generate_node/source_grounding_node y los scripts de evaluación que
    # también usan esa misma clave como interchange genérico.
    results = await mongodb.vector_search(state["query_embedding"], k=10)
    return {"semantic_results": results, "reranked_docs": results}
