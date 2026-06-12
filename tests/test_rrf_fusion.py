from app.graph.nodes.rrf_fusion import rrf_fusion_node


def _doc(id_: str) -> dict:
    return {"_id": id_, "text": f"text {id_}", "metadata": {}, "score": 1.0}


def test_shared_docs_rank_higher():
    bm25 = [_doc("A"), _doc("B"), _doc("C")]
    semantic = [_doc("A"), _doc("D"), _doc("E")]
    result = rrf_fusion_node({"bm25_results": bm25, "semantic_results": semantic})
    fused = result["fused_results"]
    ids = [d["_id"] for d in fused]
    # "A" appears in both lists, so it should be first
    assert ids[0] == "A"


def test_empty_bm25_uses_semantic():
    semantic = [_doc("X"), _doc("Y")]
    result = rrf_fusion_node({"bm25_results": [], "semantic_results": semantic})
    ids = [d["_id"] for d in result["fused_results"]]
    assert "X" in ids and "Y" in ids


def test_empty_semantic_uses_bm25():
    bm25 = [_doc("P"), _doc("Q")]
    result = rrf_fusion_node({"bm25_results": bm25, "semantic_results": []})
    ids = [d["_id"] for d in result["fused_results"]]
    assert "P" in ids and "Q" in ids


def test_both_empty():
    result = rrf_fusion_node({"bm25_results": [], "semantic_results": []})
    assert result["fused_results"] == []


def test_max_10_results():
    bm25 = [_doc(str(i)) for i in range(15)]
    semantic = [_doc(str(i + 5)) for i in range(15)]
    result = rrf_fusion_node({"bm25_results": bm25, "semantic_results": semantic})
    assert len(result["fused_results"]) <= 10
