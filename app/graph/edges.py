from typing import Literal

from langgraph.types import Send

from app.graph.state import RAGState


def route_relevance(state: RAGState) -> Literal["context_check", "out_of_domain"]:
    return "context_check" if state.get("is_relevant") else "out_of_domain"


def route_context(state: RAGState) -> Literal["query_synthesis", "generate"]:
    return "generate" if state.get("context_sufficient") else "query_synthesis"


def route_after_embed(state: RAGState) -> list[Send]:
    return [
        Send("bm25_search", state),
        Send("semantic_search", state),
    ]


def route_rerank(state: RAGState) -> Literal["generate", "no_results"]:
    return "no_results" if state.get("no_results") else "generate"
