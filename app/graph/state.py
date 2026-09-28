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
    context_sufficient: bool
    # Siempre False en producción desde que se quitó el reranker (no hay señal
    # de retrieval confiable para gatear "no encontré nada" — ver
    # app/graph/rag_graph.py). Se mantiene el campo porque la API
    # (app/api/v1/chat.py, schemas.py) sigue exponiéndolo en la respuesta.
    no_results: bool

    # --- Retrieval pipeline ---
    # Solo recuperación con embedding finetuneado (sin BM25/RRF/reranker/
    # reformulación de consulta — decisión tras el experimento de ablación en
    # scripts/eval_ablation.py). "reranked_docs" es el nombre histórico de la
    # clave que espera generate_node; ya no hay reranker, la puebla
    # directamente semantic_search_node.
    query_embedding: list[float]
    semantic_results: list[dict]
    reranked_docs: list[dict]

    # --- Output ---
    answer: str
    sources: list[dict]
    out_of_domain: bool
    from_context: bool
