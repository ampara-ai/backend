"""Pruebas de quantitative_experiments/revision sin modelos, sin red y sin claves."""

import asyncio
import sys

import numpy as np
import pytest

from quantitative_experiments.revision import citations, compare, llm, second_judge
from quantitative_experiments.revision.local_retrieval import topk


def test_released_tables_reproduce(monkeypatch, tmp_path):
    from quantitative_experiments.revision import reproduce_tables
    monkeypatch.setattr(sys, "argv", ["reproduce_tables", "--out", str(tmp_path / "t.md")])
    assert reproduce_tables.main() == 0


@pytest.mark.parametrize("text, expected", [
    ("Puedes pedir medidas (Ley 30364, art. 15) y el juez resuelve (art. 16-A).", [("ley30364", "15"), (None, "16-a")]),
    ("El artículo 122-B del Código Penal; ver los artículos 15 y 16 de la Ley N° 30364.",
     [("cp", "122-b"), ("ley30364", "15"), ("ley30364", "16")]),
    ("Según el art. 80 del CPP. El NCPP, art. 95, lista tus derechos.", [("cpp", "80"), ("cpp", "95")]),
    ("Acude al CEM o llama a la Línea 100.", []),
])
def test_extract_citations(text, expected):
    assert [(c.code, c.article) for c in citations.extract_citations(text)] == expected


def test_citation_support_levels():
    out = citations.score_answer("El artículo 122-B del Código Penal y el art. 15 de la Ley 30364.",
                                 ["chunk_tipo02_cp_art_122-b_01", "chunk_tipo04_jp_1_01"],
                                 ["Artículo 122-B. Agresiones...", "conforme al artículo 15 de la Ley"])
    assert (out["n_citations"], out["supported_strict"], out["supported_lenient"]) == (2, 1, 2)
    assert citations.chunk_article("chunk_tipo04_jp_00008_e0002_01") is None


def test_topk_breaks_ties_by_id():
    scores = np.array([[0.5, 0.9, 0.5, 0.1]])
    assert topk(scores, ["d", "a", "b", "c"], 3) == [["a", "b", "d"]]


def test_bgem3_sparse_and_colbert_scores():
    torch = pytest.importorskip("torch")
    from quantitative_experiments.revision.embed import colbert_score, sparse_score, sparse_weights
    w = sparse_weights([0, 7, 9, 7, 2], [0.9, 0.2, 0.0, 0.6, 0.5], special={0, 2})
    assert w == {7: pytest.approx(0.6)}  # máximo por token, sin especiales ni pesos nulos
    assert sparse_score({7: 0.6, 8: 1.0}, {7: 0.5}) == pytest.approx(0.3)
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    assert colbert_score(q, d) == pytest.approx((1.0 + 0.8) / 2)


def test_holm():
    adj = compare.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_cohen_kappa():
    assert second_judge.cohen_kappa(["si", "no", "si", "no"], ["si", "no", "si", "no"]) == pytest.approx(1.0)
    assert second_judge.cohen_kappa(["si", "si", "no", "no"], ["si", "no", "si", "no"]) == pytest.approx(0.0)


def test_llm_dry_run_and_cache(tmp_path):
    client = llm.LLM("m", tmp_path / "c.jsonl", dry_run=True)
    assert asyncio.run(client.json("sys", "user")) is None
    assert asyncio.run(client.text([{"role": "user", "content": "hola"}])) is None
    assert len(client.pending) == 2
    key = llm.request_key("m", [{"role": "user", "content": "x"}], {"temperature": 0})
    assert key == llm.request_key("m", [{"role": "user", "content": "x"}], {"temperature": 0})
    assert key != llm.request_key("m", [{"role": "user", "content": "x"}], {"temperature": 0.7})
