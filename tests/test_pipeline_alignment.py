from app.core import dependencies
from app.core.config import settings
from app.graph import rag_graph as graph_module
from app.graph.nodes import source_grounding
from app.graph.nodes.source_grounding import _GroundingOutput, _UsedDocument

# ---------------------------------------------------------------------------
# Proveedor del LLM fijado en el request
# ---------------------------------------------------------------------------


def test_provider_pinned_in_request_body(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_only", "groq")
    monkeypatch.setattr(settings, "llm_allow_fallbacks", False)
    monkeypatch.setattr(settings, "llm_data_collection", "deny")
    assert dependencies.get_llm().extra_body == {
        "provider": {"only": ["groq"], "allow_fallbacks": False, "data_collection": "deny"}
    }


def test_provider_unrestricted_when_empty(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_only", "")
    assert dependencies.get_llm().extra_body is None


# ---------------------------------------------------------------------------
# Expiración de sesiones inactivas
# ---------------------------------------------------------------------------


def test_idle_sessions_are_evicted(monkeypatch):
    monkeypatch.setattr(settings, "session_idle_ttl_minutes", 1)
    monkeypatch.setattr(graph_module, "_last_activity", {})
    storage = graph_module.rag_graph.checkpointer.storage
    storage["old"] = {"history": "texto de la víctima"}
    storage["recent"] = {"history": "otra conversación"}

    graph_module.touch_session("old", now=0.0)
    graph_module.touch_session("recent", now=30.0)
    graph_module.touch_session("someone-else", now=75.0)  # "old" lleva 75 s inactiva

    assert "old" not in storage
    assert "recent" in storage
    assert set(graph_module._last_activity) == {"recent", "someone-else"}
    storage.pop("recent", None)


def test_expired_session_is_cleared_before_reuse(monkeypatch):
    monkeypatch.setattr(settings, "session_idle_ttl_minutes", 1)
    monkeypatch.setattr(graph_module, "_last_activity", {})
    storage = graph_module.rag_graph.checkpointer.storage
    storage["s1"] = {"history": "antes de expirar"}

    graph_module.touch_session("s1", now=0.0)
    graph_module.touch_session("s1", now=120.0)

    assert "s1" not in storage
    assert graph_module._last_activity == {"s1": 120.0}


# ---------------------------------------------------------------------------
# Source grounding con salida estructurada
# ---------------------------------------------------------------------------

DOCS = [
    {"title": "Art. 22", "text": "Medidas de protección...", "source_type": "tipo_01_ley_30364", "metadata": {}, "score": 0.8},
    {"title": "Art. 122-B", "text": "Agresiones en contra de las mujeres...", "source_type": "tipo_02_codigo_penal", "metadata": {}, "score": 0.7},
]


class _FakeStructuredLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    async def ainvoke(self, messages):
        if self.error:
            raise self.error
        return self.result


async def test_grounding_maps_used_documents(monkeypatch):
    output = _GroundingOutput(used=[
        _UsedDocument(index=2, title="Delito de agresiones", summary="Tipifica la agresión."),
        _UsedDocument(index=9, title="Fuera de rango", summary="Se ignora."),
    ])
    monkeypatch.setattr(source_grounding, "get_llm_structured", lambda schema: _FakeStructuredLLM(output))

    result = await source_grounding.source_grounding_node({"reranked_docs": DOCS, "answer": "Respuesta"})

    assert [s["source_label"] for s in result["sources"]] == ["Delito de agresiones"]
    assert result["sources"][0]["source_type"] == "Código Penal"


async def test_grounding_falls_back_to_all_documents_on_error(monkeypatch):
    monkeypatch.setattr(
        source_grounding, "get_llm_structured", lambda schema: _FakeStructuredLLM(error=ValueError("bad output"))
    )

    result = await source_grounding.source_grounding_node({"reranked_docs": DOCS, "answer": "Respuesta"})

    assert [s["source_label"] for s in result["sources"]] == ["Art. 22", "Art. 122-B"]
