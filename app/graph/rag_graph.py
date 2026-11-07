import time

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from app.core.config import settings
from app.graph.edges import route_context, route_retrieval, route_triage
from app.graph.nodes import no_results_node, out_of_domain_node
from app.graph.nodes.context_check import context_check_node
from app.graph.nodes.crisis import crisis_check_node, crisis_notice_node, crisis_response_node, triage_node
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
    # quantitative_experiments/ablation/eval_ablation.py: el embedding solo rindió mejor que cualquier
    # combinación probada (RRF con o sin peso, reranker actual, reranker Qwen,
    # interpolación de scores, reformulación jurídica de la consulta).
    # Gating por similitud: solo llegan al generador los chunks cuyo puntaje de Atlas supera
    # settings.retrieval_score_threshold ((1 + coseno) / 2; 0.5 equivale a coseno > 0). El umbral es una
    # red de seguridad, no un discriminador fino: el score del embedding no separa de forma confiable
    # "chunk correcto" de "chunk relacionado pero incorrecto" en este corpus angosto. Si ningún chunk
    # lo supera, el nodo no_results responde con el mensaje fijo y no se llama al generador. El filtro de
    # dominio real es relevance_filter_node, antes de llegar a recuperar.
    # Crisis: crisis_check corre en paralelo con relevance_filter (sin latencia extra) y no cambia sus decisiones. Un
    # mensaje de peligro actual que el filtro rechaza recibe indicaciones de seguridad (crisis_response) en lugar de
    # "fuera de dominio"; si sigue el camino normal, crisis_notice antepone esas indicaciones a la respuesta.
    graph = (
        StateGraph(RAGState)
        .add_node("crisis_check", crisis_check_node)
        .add_node("relevance_filter", relevance_filter_node)
        .add_node("triage", triage_node)
        .add_node("crisis_response", crisis_response_node)
        .add_node("crisis_notice", crisis_notice_node)
        .add_node("context_check", context_check_node)
        .add_node("embed_query", embed_query_node)
        .add_node("semantic_search", semantic_search_node)
        .add_node("generate", generate_node)
        .add_node("source_grounding", source_grounding_node)
        .add_node("out_of_domain", out_of_domain_node)
        .add_node("no_results", no_results_node)
        .add_edge(START, "crisis_check")
        .add_edge(START, "relevance_filter")
        .add_edge(["crisis_check", "relevance_filter"], "triage")
        .add_conditional_edges(
            "triage", route_triage, ["context_check", "out_of_domain", "crisis_response"]
        )
        .add_conditional_edges(
            "context_check", route_context, ["embed_query", "generate"]
        )
        .add_edge("embed_query", "semantic_search")
        .add_conditional_edges(
            "semantic_search", route_retrieval, ["generate", "no_results"]
        )
        .add_edge("generate", "source_grounding")
        .add_edge("source_grounding", "crisis_notice")
        .add_edge("no_results", "crisis_notice")
        .add_edge("crisis_notice", END)
        .add_edge("crisis_response", END)
        .add_edge("out_of_domain", END)
        .compile(checkpointer=checkpointer)
    )
    return graph


rag_graph = build_rag_graph()


def clear_session_history(session_id: str) -> None:
    _last_activity.pop(session_id, None)
    cp = rag_graph.checkpointer
    if hasattr(cp, "storage"):
        cp.storage.pop(session_id, None)
    if hasattr(cp, "writes"):
        stale = [k for k in list(cp.writes) if k[0] == session_id]
        for k in stale:
            del cp.writes[k]


# Última actividad por sesión (reloj monotónico). InMemorySaver no expira nada:
# sin esto, el texto de una conversación queda en memoria hasta que el proceso
# se reinicia, aunque el usuario recargue la página o la abandone.
_last_activity: dict[str, float] = {}


def touch_session(session_id: str, now: float | None = None) -> None:
    """Registra actividad de `session_id` y borra el historial de las sesiones
    inactivas por más de session_idle_ttl_minutes (incluida esta, si expiró).
    Se llama al inicio de cada request de chat."""
    now = time.monotonic() if now is None else now
    ttl_seconds = settings.session_idle_ttl_minutes * 60
    for sid, last in list(_last_activity.items()):
        if now - last > ttl_seconds:
            clear_session_history(sid)
    _last_activity[session_id] = now
