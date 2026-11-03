import asyncio

from app.core.config import settings
from app.graph.edges import route_retrieval
from app.graph.nodes import no_results_node
from app.graph.nodes import retrieval
from app.graph.nodes.retrieval import apply_score_threshold, atlas_score_to_cosine


def _doc(id_: str, score: float) -> dict:
    return {"_id": id_, "chunk_id": id_, "text": f"text {id_}", "metadata": {}, "score": score}


def test_atlas_score_is_normalized_cosine():
    assert atlas_score_to_cosine(0.5) == 0.0
    assert atlas_score_to_cosine(1.0) == 1.0
    assert atlas_score_to_cosine(0.75) == 0.5


def test_threshold_is_strict_and_keeps_order():
    docs = [_doc("A", 0.9), _doc("B", 0.5), _doc("C", 0.7), _doc("D", 0.49)]
    assert [d["_id"] for d in apply_score_threshold(docs, 0.5)] == ["A", "C"]


def test_route_retrieval():
    assert route_retrieval({"reranked_docs": [_doc("A", 0.9)]}) == "generate"
    assert route_retrieval({"reranked_docs": []}) == "no_results"
    assert route_retrieval({}) == "no_results"


def test_no_results_node_sets_flag_and_fixed_message():
    out = no_results_node({})
    assert out["no_results"] is True and out["sources"] == []
    assert "CEM" in out["answer"]


def test_semantic_search_node_filters_below_threshold(monkeypatch):
    async def fake_vector_search(embedding, k=20, exact=False):
        assert k == settings.retrieval_k
        return [_doc("A", 0.8), _doc("B", 0.4)]

    monkeypatch.setattr(retrieval.mongodb, "vector_search", fake_vector_search)
    out = asyncio.run(retrieval.semantic_search_node({"query_embedding": [0.0]}))
    assert [d["_id"] for d in out["semantic_results"]] == ["A", "B"]   # el ranking completo se conserva
    assert [d["_id"] for d in out["reranked_docs"]] == ["A"]            # al generador solo llegan los que superan el umbral
