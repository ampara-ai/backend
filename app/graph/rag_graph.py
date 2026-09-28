from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph.edges import route_context, route_relevance
from app.graph.nodes import out_of_domain_node
from app.graph.nodes.context_check import context_check_node
from app.graph.nodes.embedding import embed_query_node
from app.graph.nodes.generation import generate_node
from app.graph.nodes.source_grounding import source_grounding_node
from app.graph.nodes.relevance_filter import relevance_filter_node
from app.graph.nodes.retrieval import semantic_search_node
from app.graph.state import RAGState


def build_rag_graph():
    # InMemorySaver is intentional: anonymous sessions with no user accounts.
    # Sessions live as long as the server process runs, matching frontend tab lifetime.
    checkpointer = InMemorySaver()

    # Recuperación: solo embedding finetuneado (sin BM25/RRF/reranker/reformulación
    # de consulta) — decisión tomada tras el experimento de ablación en
    # scripts/eval_ablation.py: el embedding solo rindió mejor que cualquier
    # combinación probada (RRF con o sin peso, reranker actual, reranker Qwen,
    # interpolación de scores, reformulación jurídica de la consulta).
    # No hay gating de "no_results" por score de retrieval: el score del
    # embedding no separa de forma confiable "chunk correcto" de "chunk
    # relacionado pero incorrecto" en este corpus angosto (ver análisis en la
    # sesión que motivó este cambio) — el filtro de dominio real es
    # relevance_filter_node, antes de llegar a recuperar.
    graph = (
        StateGraph(RAGState)
        .add_node("relevance_filter", relevance_filter_node)
        .add_node("context_check", context_check_node)
        .add_node("embed_query", embed_query_node)
        .add_node("semantic_search", semantic_search_node)
        .add_node("generate", generate_node)
        .add_node("source_grounding", source_grounding_node)
        .add_node("out_of_domain", out_of_domain_node)
        .add_edge(START, "relevance_filter")
        .add_conditional_edges(
            "relevance_filter", route_relevance, ["context_check", "out_of_domain"]
        )
        .add_conditional_edges(
            "context_check", route_context, ["embed_query", "generate"]
        )
        .add_edge("embed_query", "semantic_search")
        .add_edge("semantic_search", "generate")
        .add_edge("generate", "source_grounding")
        .add_edge("source_grounding", END)
        .add_edge("out_of_domain", END)
        .compile(checkpointer=checkpointer)
    )
    return graph


rag_graph = build_rag_graph()


def clear_session_history(session_id: str) -> None:
    cp = rag_graph.checkpointer
    if hasattr(cp, "storage"):
        cp.storage.pop(session_id, None)
    if hasattr(cp, "writes"):
        stale = [k for k in list(cp.writes) if k[0] == session_id]
        for k in stale:
            del cp.writes[k]
