"""Regression checks for ranking on raw logits without loading a reranker."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantitative_experiments.revision import rerank_local


@pytest.fixture
def cross_encoder(monkeypatch):
    calls = []

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            calls.append({"name": name, **kwargs})

        def predict(self, pairs, **kwargs):
            # The default sigmoid merges these distinct FP16 logits into a tie.
            logits = torch.tensor([10.0, 12.0][:len(pairs)], dtype=torch.float16)
            activation = kwargs.get("activation_fn", torch.nn.Sigmoid())
            calls.append({"pairs": pairs, **kwargs})
            return activation(logits).numpy()

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    return calls


@pytest.mark.parametrize("device,fp16,dtype", [
    ("cuda", True, "fp16"), ("mps", True, "fp16"),
    ("cpu", True, "fp32"), ("cuda", False, "fp32"),
])
def test_cross_encoder_ranks_distinct_logits_without_sigmoid_saturation(cross_encoder, device, fp16, dtype):
    rr = rerank_local.Reranker("st:fake-local-model", device, 1024, fp16)
    # Reproduce why using CrossEncoder's default activation would be incorrect.
    assert torch.sigmoid(torch.tensor([10.0, 12.0], dtype=torch.float16)).tolist() == [1.0, 1.0]

    scores = rr.score("question", ["lower", "higher"])
    assert scores.tolist() == [10.0, 12.0]
    assert scores.dtype == np.float32
    assert isinstance(cross_encoder[-1]["activation_fn"], torch.nn.Identity)
    assert cross_encoder[-1]["apply_softmax"] is False
    assert rr.dtype == dtype
    assert rr.score_kind == "raw_logits"

    ranked, _ = rerank_local.rerank(
        rr, "question", ["a_lower", "z_higher", "tail"], 2,
        {"a_lower": "lower", "z_higher": "higher", "tail": "not scored"},
    )
    assert ranked == ["z_higher", "a_lower", "tail"]


def test_onnx_also_uses_raw_logits_without_saturating():
    rr = object.__new__(rerank_local.Reranker)
    rr.type, rr.max_length, rr.inputs = "onnx", 1024, {"input_ids"}
    calls = []

    class Tokenizer:
        def __call__(self, pairs, **kwargs):
            calls.append(pairs)
            return {"input_ids": np.array([[10], [12]], dtype=np.int64)}

    rr.tokenizer = Tokenizer()
    rr.session = SimpleNamespace(run=lambda _, enc: [enc["input_ids"].astype(np.float32)])
    scores = rr.score("question", ["lower", "higher"])
    assert scores.tolist() == [10.0, 12.0]
    assert scores.dtype == np.float32
    assert calls == [[["question", "lower"], ["question", "higher"]]]


def test_run_metadata_identifies_raw_logit_scores(cross_encoder, monkeypatch, tmp_path):
    dataset, qrels = tmp_path / "dataset.jsonl", tmp_path / "qrels.jsonl"
    dataset.write_text(json.dumps({"dataset_id": "q", "query": "question", "positive_chunk_ids": ["z"]}) + "\n")
    qrels.write_text(json.dumps({"qid": "q", "docid": "z", "relevance": 1}) + "\n")
    first = tmp_path / "first.json"
    first.write_text(json.dumps({"q": ["a", "z"]}))
    monkeypatch.setattr(rerank_local, "corpus", lambda: {"a": {"text": "lower"}, "z": {"text": "higher"}})
    out = tmp_path / "output"

    rerank_local.main([
        "--model", "st:fake-local-model", "--device", "cpu", "--first-stage", str(first),
        "--dataset", str(dataset), "--qrels", str(qrels), "--out-dir", str(out),
        "--depths", "2", "--name", "test",
    ])

    meta = json.loads((out / "test.meta.json").read_text())
    assert meta["score_kind"] == "raw_logits"
    assert json.loads((out / "test_k2.json").read_text()) == {"q": ["z", "a"]}
