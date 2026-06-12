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
    no_results: bool

    # --- Retrieval pipeline ---
    synthesized_query: str
    query_embedding: list[float]
    bm25_results: list[dict]
    semantic_results: list[dict]
    fused_results: list[dict]
    reranked_docs: list[dict]
    rerank_score: float

    # --- Output ---
    answer: str
    sources: list[dict]
    out_of_domain: bool
    from_context: bool
