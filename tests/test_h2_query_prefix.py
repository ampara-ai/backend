"""H2 train/dev/test formatting contracts, without loading models or training."""

import json
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantitative_experiments.revision import local_retrieval, select_checkpoint
from quantitative_experiments.revision.embed import DenseEncoder


PREFIX = "Represent this sentence for searching relevant passages: "


@pytest.mark.parametrize("prefix", ["", PREFIX])
def test_retrieval_formats_queries_once_and_leaves_documents_raw(monkeypatch, prefix):
    captured = []

    class CaptureModel:
        def encode(self, texts, **kwargs):
            captured.append((list(texts), kwargs["prompt"]))
            return np.array([[1.0, 0.0] for _ in texts])

    # Run the real DenseEncoder formatting path without constructing a model.
    encoder = DenseEncoder.__new__(DenseEncoder)
    encoder.type = "cls"
    encoder.prompts = {"query": "", "document": ""}
    encoder.kwargs = {"query": {}, "document": {}}
    encoder.model = CaptureModel()
    monkeypatch.setattr(local_retrieval, "corpus", lambda: {"d1": {"text": "Unmodified legal text"}})

    def documents(enc, ids, texts, batch_size, cache):
        return enc.encode_documents(texts, batch_size), 0.0

    monkeypatch.setattr(local_retrieval, "corpus_embeddings", documents)
    dataset = [{"dataset_id": "q1", "query": "Pregunta original", "rewritten": "Pregunta reformulada"}]
    run, _, _ = local_retrieval.retrieve("cls:checkpoint", dataset, query_field="rewritten", encoder=encoder,
                                        query_prefix=prefix)
    assert run == {"q1": ["d1"]}
    assert captured == [(["Unmodified legal text"], None), ([prefix + "Pregunta reformulada"], None)]
    assert dataset[0]["rewritten"] == "Pregunta reformulada"


def test_checkpoint_selection_propagates_and_records_prefix(tmp_path, monkeypatch):
    run_dir = tmp_path / "training"
    (run_dir / "checkpoint-85").mkdir(parents=True)
    (run_dir / "checkpoint-170").mkdir()
    dataset = [{"dataset_id": "q1", "query": "Pregunta", "source_chunk": "d1"}]
    monkeypatch.setattr(select_checkpoint, "load_dataset", lambda _: dataset)
    monkeypatch.setattr(select_checkpoint, "load_qrels", lambda _: {"q1": {"d1"}})
    calls = []

    def retrieve(spec, rows, **kwargs):
        calls.append((spec, rows, kwargs))
        return {"q1": ["d1"]}, None, {}

    monkeypatch.setattr(select_checkpoint, "retrieve", retrieve)
    select_checkpoint.main(["--run-dir", str(run_dir), "--dataset", "dev.jsonl", "--qrels", "qrels.jsonl",
                            "--query-prefix", PREFIX, "--device", "cpu", "--fp32"])
    assert len(calls) == 2
    assert all(c[2] == {"device": "cpu", "fp16": False, "query_prefix": PREFIX} for c in calls)
    selected = json.loads((run_dir / "checkpoint_selection.json").read_text())
    assert selected["query_prefix"] == PREFIX
    assert selected["selected"].endswith("checkpoint-85")


def test_retrieval_cli_records_prefix_and_uses_it_for_latency(tmp_path, monkeypatch):
    dataset = [{"dataset_id": "q1", "query": "Pregunta"}]
    monkeypatch.setattr(local_retrieval, "load_dataset", lambda _: dataset)
    corpus_file = tmp_path / "corpus.jsonl"
    corpus_file.write_text("source")
    monkeypatch.setattr(local_retrieval, "CORPUS_PATH", corpus_file)
    latency_queries = []
    encoder = SimpleNamespace(device="cpu", info=lambda: SimpleNamespace(spec="cls:model"),
                              encode_queries=lambda texts, batch: latency_queries.extend(texts))
    calls = []

    def retrieve(*args, **kwargs):
        calls.append(kwargs)
        return {"q1": ["d1"]}, encoder, {}

    monkeypatch.setattr(local_retrieval, "retrieve", retrieve)
    local_retrieval.main(["--model", "cls:model", "--name", "h2", "--dataset", str(tmp_path / "custom.jsonl"),
                          "--out-dir", str(tmp_path / "output"), "--query-prefix", PREFIX, "--latency-sample", "1"])
    assert calls == [{"query_prefix": PREFIX}]
    assert latency_queries == [PREFIX + "Pregunta"]
    metadata = json.loads((tmp_path / "output/h2.meta.json").read_text())
    assert metadata["query_prefix"] == PREFIX


def test_h2_preview_uses_identical_prefix_for_every_dev_and_test_call():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", "quantitative_experiments/revision/run_h2.sh", "--dry-run"],
                            cwd=root, capture_output=True, text=True, check=True)
    commands = [shlex.split(line[1:]) for line in result.stdout.splitlines()
                if line.startswith("+") and any(module in line for module in
                    ("revision.select_checkpoint", "revision.local_retrieval"))]
    assert len(commands) == 18
    assert all(command[command.index("--query-prefix") + 1] == PREFIX for command in commands)
