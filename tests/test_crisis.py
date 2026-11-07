"""Detector de crisis: un mensaje de peligro actual nunca recibe "fuera de dominio" y el resto del grafo no cambia."""

import uuid

import pytest

from app.api.v1.chat import _build_done_event
from app.graph import rag_graph as graph_module
from app.graph.nodes.crisis import CRISIS_ANSWER, SAFETY_NOTICE


def _graph(monkeypatch, in_crisis: bool, relevant: bool):
    async def crisis(state):
        return {"in_crisis": in_crisis}

    async def relevance(state):
        return {"is_relevant": relevant}

    async def context(state):
        return {"context_sufficient": True}

    async def generate(state):
        return {"answer": "Respuesta.", "sources": [], "from_context": True,
                "messages": [{"role": "assistant", "content": "Respuesta."}]}

    async def grounding(state):
        return {"sources": [], "answer": state["answer"], "from_context": True, "out_of_domain": False, "no_results": False}

    monkeypatch.setattr(graph_module, "crisis_check_node", crisis)
    monkeypatch.setattr(graph_module, "relevance_filter_node", relevance)
    monkeypatch.setattr(graph_module, "context_check_node", context)
    monkeypatch.setattr(graph_module, "generate_node", generate)
    monkeypatch.setattr(graph_module, "source_grounding_node", grounding)
    return graph_module.build_rag_graph()


async def _run(graph):
    return await graph.ainvoke(
        {"session_id": "t", "user_query": "x", "messages": [{"role": "user", "content": "x"}]},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )


@pytest.mark.parametrize("in_crisis, relevant, expected", [
    (True, False, "crisis_response"),
    (False, False, "out_of_domain"),
    (True, True, "notice"),
    (False, True, "normal"),
])
async def test_crisis_routing(monkeypatch, in_crisis, relevant, expected):
    out = await _run(_graph(monkeypatch, in_crisis, relevant))
    if expected == "crisis_response":
        assert out["answer"] == CRISIS_ANSWER and out["crisis"] and not out.get("out_of_domain")
    elif expected == "out_of_domain":
        assert out["out_of_domain"] and not out.get("crisis")
    elif expected == "notice":
        assert out["answer"] == f"{SAFETY_NOTICE}\n\nRespuesta." and out["crisis"]
    else:
        assert out["answer"] == "Respuesta." and not out.get("crisis")


def test_done_event_carries_crisis_flag():
    done = _build_done_event("s", {"answer": CRISIS_ANSWER, "crisis": True, "sources": []}, "")
    assert done["crisis"] is True and done["out_of_domain"] is False
