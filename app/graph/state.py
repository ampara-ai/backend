import operator
from typing import Annotated

from typing_extensions import TypedDict


class RAGState(TypedDict):
    # --- Input ---
    session_id: str
    user_query: str
    # All conversation turns; reducer appends each new message
    messages: Annotated[list[dict], operator.add]

    # --- Routing flags ---
    is_relevant: bool
    # Peligro actual o inminente según el detector de crisis (corre en paralelo con el filtro de dominio)
    in_crisis: bool
    context_sufficient: bool
    # True cuando ningún chunk recuperado supera el umbral de similitud
    # (settings.retrieval_score_threshold) y el grafo responde con el mensaje fijo de
    # "sin fuentes suficientes" (app/graph/rag_graph.py). La API
    # (app/api/v1/chat.py, schemas.py) lo expone en la respuesta.
    no_results: bool

    # --- Retrieval pipeline ---
    # Solo recuperación con embedding finetuneado (sin BM25/RRF/reranker/
    # reformulación de consulta — decisión tras el experimento de ablación en
    # quantitative_experiments/ablation/eval_ablation.py). "reranked_docs" es el nombre histórico de la
    # clave que espera generate_node; ya no hay reranker, la puebla
    # directamente semantic_search_node.
    query_embedding: list[float]
    semantic_results: list[dict]
    reranked_docs: list[dict]

    # --- Output ---
    answer: str
    sources: list[dict]
    out_of_domain: bool
    # True cuando la respuesta lleva las indicaciones de seguridad del detector de crisis
    crisis: bool
    from_context: bool
