from typing import Literal

from app.graph.state import RAGState


def route_relevance(state: RAGState) -> Literal["context_check", "out_of_domain"]:
    return "context_check" if state.get("is_relevant") else "out_of_domain"


def route_context(state: RAGState) -> Literal["embed_query", "generate"]:
    return "generate" if state.get("context_sufficient") else "embed_query"
