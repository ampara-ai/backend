from typing import Literal

from app.graph.state import RAGState


def route_relevance(state: RAGState) -> Literal["context_check", "out_of_domain"]:
    return "context_check" if state.get("is_relevant") else "out_of_domain"


def route_triage(state: RAGState) -> Literal["context_check", "out_of_domain", "crisis_response"]:
    """Tras el filtro de dominio y el detector de crisis: un mensaje de crisis nunca recibe "fuera de dominio"."""
    if state.get("is_relevant"):
        return "context_check"
    return "crisis_response" if state.get("in_crisis") else "out_of_domain"


def route_context(state: RAGState) -> Literal["embed_query", "generate"]:
    return "generate" if state.get("context_sufficient") else "embed_query"


def route_retrieval(state: RAGState) -> Literal["generate", "no_results"]:
    """Sin ningún chunk sobre el umbral de similitud no se genera: respuesta fija de "sin fuentes suficientes"."""
    return "generate" if state.get("reranked_docs") else "no_results"
