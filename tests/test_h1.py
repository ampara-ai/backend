"""Portable H1 checks using artificial data, without model downloads or inference."""

import copy
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantitative_experiments.revision import h1 as exp


@pytest.mark.parametrize(
    "cuda,mps,expected",
    [(True, True, "cuda"), (True, False, "cuda"),
     (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_prefers_cuda_then_mps_then_cpu(cuda, mps, expected):
    assert exp.select_device("auto", cuda=cuda, mps=mps) == expected


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_explicit_unavailable_accelerator_is_rejected(device):
    with pytest.raises(ValueError):
        exp.select_device(device, cuda=False, mps=False)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_explicit_device_is_respected(device):
    assert exp.select_device(device, cuda=True, mps=True) == device


@pytest.mark.parametrize(
    "requested,device,expected",
    [("auto", "cuda", "fp16"), ("auto", "mps", "fp16"),
     ("auto", "cpu", "fp32"), ("fp32", "cuda", "fp32"),
     ("fp32", "mps", "fp32"), ("fp32", "cpu", "fp32"),
     ("fp16", "cuda", "fp16"), ("fp16", "mps", "fp16")],
)
def test_precision_selection(requested, device, expected):
    assert exp.select_precision(requested, device) == expected


def test_cpu_half_precision_is_rejected():
    with pytest.raises(ValueError):
        exp.select_precision("fp16", "cpu")


@pytest.fixture
def inputs():
    return {
        "dataset": [
            {"dataset_id": "q1", "query": "First question?", "source_chunk": "d1",
             "positive_chunk_ids": ["d1"]},
            {"dataset_id": "q2", "query": "Second question?", "source_chunk": "d2",
             "positive_chunk_ids": ["d2"]},
        ],
        "qrels": {"q1": {"d1"}, "q2": {"d2"}},
        "corpus_rows": [{"chunk_id": "d1", "text": "First source."},
                        {"chunk_id": "d2", "text": "Second source."},
                        {"chunk_id": "irrelevant", "text": "Other source."}],
    }


def test_valid_inputs_are_accepted_without_modification(inputs):
    before = copy.deepcopy(inputs)
    exp.validate_inputs(**inputs)
    assert inputs == before


@pytest.mark.parametrize(
    "problem",
    ["duplicate_query", "missing_query_id", "empty_query_id", "empty_query",
     "missing_qrels", "empty_qrels", "absent_positive_doc", "source_not_positive"],
)
def test_invalid_inputs_are_rejected(inputs, problem):
    if problem == "duplicate_query":
        inputs["dataset"].append(copy.deepcopy(inputs["dataset"][0]))
    elif problem == "missing_query_id":
        del inputs["dataset"][0]["dataset_id"]
    elif problem == "empty_query_id":
        inputs["dataset"][0]["dataset_id"] = ""
    elif problem == "empty_query":
        inputs["dataset"][0]["query"] = "   "
    elif problem == "missing_qrels":
        del inputs["qrels"]["q1"]
    elif problem == "empty_qrels":
        inputs["qrels"]["q1"] = set()
    elif problem == "absent_positive_doc":
        inputs["qrels"]["q1"].add("not-in-corpus")
    else:
        inputs["qrels"]["q1"] = {"d2"}

    with pytest.raises(ValueError):
        exp.validate_inputs(**inputs)


def test_source_doc_must_exist_in_corpus(inputs):
    inputs["corpus_rows"] = [r for r in inputs["corpus_rows"] if r["chunk_id"] != "d1"]
    with pytest.raises(ValueError):
        exp.validate_inputs(**inputs)


def unit_vectors(count=2, dim=1024):
    vectors = np.zeros((count, dim), dtype=np.float32)
    vectors[:, 0] = 1
    return vectors


def test_normalized_embeddings_are_accepted_with_numeric_tolerance():
    vectors = unit_vectors()
    vectors[1, 0] = 1.01
    exp.validate_embeddings(vectors, 2)


@pytest.mark.parametrize("problem", ["wrong_count", "wrong_dim", "wrong_rank", "nan", "inf", "zero", "unnormalized"])
def test_invalid_embeddings_are_rejected(problem):
    vectors = unit_vectors()
    if problem == "wrong_count":
        vectors = unit_vectors(count=1)
    elif problem == "wrong_dim":
        vectors = unit_vectors(dim=768)
    elif problem == "wrong_rank":
        vectors = vectors[0]
    elif problem == "nan":
        vectors[0, 1] = np.nan
    elif problem == "inf":
        vectors[0, 1] = np.inf
    elif problem == "zero":
        vectors[0] = 0
    else:
        vectors[0, 0] = 2
    with pytest.raises(ValueError):
        exp.validate_embeddings(vectors, 2)


def test_embedding_dimension_can_be_explicitly_configured():
    exp.validate_embeddings(unit_vectors(dim=3), 2, expected_dim=3)


@pytest.fixture
def runs():
    return {
        "base": {"q1": ["irrelevant", "d1"], "q2": []},
        "finetuned": {"q1": ["d1", "irrelevant"], "q2": ["d2"]},
    }


def test_paired_scores_match_hand_computed_ndcg(inputs, runs):
    report, rows = exp.score_pair(inputs["dataset"], inputs["qrels"], runs, resamples=1000, seed=7)
    discount = 1 / math.log2(3)

    assert report["queries"] == 2
    assert report["source_clusters"] == 2
    assert report["base_ndcg10"] == pytest.approx(discount / 2)
    assert report["finetuned_ndcg10"] == pytest.approx(1)
    assert report["difference"] == pytest.approx({
        "estimate": 1 - discount / 2,
        "ci_low": 1 - discount,
        "ci_high": 1,
    })
    assert report["multiplicity"] == "H1-H3 Holm adjustment pending"
    assert len(rows) == 2


def test_paired_bootstrap_keeps_queries_from_one_source_together(inputs):
    inputs["dataset"][1]["source_chunk"] = "d1"
    inputs["dataset"][1]["positive_chunk_ids"] = ["d1"]
    inputs["qrels"]["q2"] = {"d1"}
    runs = {"base": {"q1": ["irrelevant", "d1"], "q2": ["d1"]},
            "finetuned": {"q1": ["d1"], "q2": ["irrelevant", "d1"]}}

    report, rows = exp.score_pair(inputs["dataset"], inputs["qrels"], runs, resamples=200, seed=19)
    repeated, repeated_rows = exp.score_pair(inputs["dataset"], inputs["qrels"], runs, resamples=200, seed=19)

    assert report["source_clusters"] == 1
    assert report["difference"] == pytest.approx({"estimate": 0, "ci_low": 0, "ci_high": 0})
    assert report["difference"] == repeated["difference"]
    assert rows == repeated_rows


def test_ndcg_uses_only_top_ten(inputs):
    ranked = [f"irrelevant-{i}" for i in range(10)] + ["d1"]
    runs = {"base": {"q1": ranked, "q2": []},
            "finetuned": {"q1": ranked, "q2": []}}
    report, _ = exp.score_pair(inputs["dataset"], inputs["qrels"], runs, resamples=20, seed=0)
    assert report["base_ndcg10"] == 0
    assert report["finetuned_ndcg10"] == 0
    assert report["difference"]["estimate"] == 0


@pytest.mark.parametrize("problem", ["missing_base", "missing_finetuned", "missing_query", "extra_query", "duplicate_doc"])
def test_pair_scoring_rejects_incomplete_or_duplicate_rankings(inputs, runs, problem):
    if problem == "missing_base":
        del runs["base"]
    elif problem == "missing_finetuned":
        del runs["finetuned"]
    elif problem == "missing_query":
        del runs["base"]["q2"]
    elif problem == "extra_query":
        runs["base"]["unexpected"] = []
    else:
        runs["finetuned"]["q1"] = ["d1", "d1"]
    with pytest.raises(ValueError):
        exp.score_pair(inputs["dataset"], inputs["qrels"], runs, resamples=20, seed=0)


@pytest.mark.parametrize("dry_run_flags", [[], ["--dry-run"]])
def test_cli_defaults_to_dry_run_without_torch_models_or_outputs(inputs, tmp_path, monkeypatch, capsys, dry_run_flags):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not resolve or construct a model")

    monkeypatch.setattr(exp, "resolve_model", forbidden)
    monkeypatch.setattr(exp, "make_encoder", forbidden)
    monkeypatch.setattr(exp, "detect_hardware", lambda: {"cuda": False, "mps": False, "torch_available": False})
    dataset_path = tmp_path / "dataset.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    out_dir = tmp_path / "must-not-be-created"
    dataset_path.write_text("".join(json.dumps(r) + "\n" for r in inputs["dataset"]), encoding="utf-8")
    qrels_path.write_text("".join(json.dumps({"qid": q, "docid": d, "relevance": 1}) + "\n"
                                  for q, docs in inputs["qrels"].items() for d in sorted(docs)), encoding="utf-8")
    corpus_path.write_text("".join(json.dumps(r) + "\n" for r in inputs["corpus_rows"]), encoding="utf-8")

    result = exp.main(["--dataset", str(dataset_path), "--qrels", str(qrels_path),
                       "--corpus", str(corpus_path), "--out-dir", str(out_dir), *dry_run_flags])

    assert result in (None, 0)
    assert isinstance(json.loads(capsys.readouterr().out), dict)
    assert not out_dir.exists()


@pytest.fixture
def fake_execution(inputs, tmp_path, monkeypatch):
    """Exercise file/protocol orchestration without importing a model or PyTorch."""
    paths = {}
    for name, rows in {
        "dataset": inputs["dataset"],
        "qrels": [{"qid": q, "docid": d, "relevance": 1}
                  for q, docs in inputs["qrels"].items() for d in sorted(docs)],
        "corpus": inputs["corpus_rows"],
    }.items():
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        paths[name] = {"path": str(path), "sha256": exp.file_hash(path)}

    models = {}
    for name in ("base", "finetuned"):
        folder = tmp_path / ".cache" / "models" / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text(json.dumps({"test_model": name}), encoding="utf-8")
        (folder / "fake.safetensors").write_bytes(f"not real weights: {name}".encode())
        models[name] = {"local_path": str(folder)}

    calls = {"resolved": [], "tokenizers": [], "encoders": [], "documents": [],
             "queries": [], "released": [], "fail_model": None}

    def resolve(spec):
        calls["resolved"].append(spec)
        return Path(spec["local_path"])

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only):
            assert local_files_only is True
            calls["tokenizers"].append(path)
            return FakeTokenizer()

        def __call__(self, texts, *, truncation, max_length):
            assert truncation is True
            return {"input_ids": [[ord(c) for c in text][:max_length] for text in texts]}

    class FakeEncoder:
        def __init__(self, name, device, precision):
            self.name = name
            self.device = device
            self.precision = precision

        def info(self):
            return SimpleNamespace(device=self.device, dtype=self.precision, model=self.name)

        def encode_documents(self, texts, batch_size):
            calls["documents"].append((self.name, list(texts), batch_size))
            assert texts == ["First source.", "Second source.", "Other source."]
            vectors = np.zeros((3, 1024), dtype=np.float32)
            vectors[np.arange(3), np.arange(3)] = 1
            return vectors

        def encode_queries(self, texts, batch_size):
            calls["queries"].append((self.name, list(texts), batch_size))
            if self.name == calls["fail_model"]:
                raise RuntimeError("Synthetic second-stage failure")
            assert texts == ["First question?", "Second question?"]
            vectors = np.zeros((2, 1024), dtype=np.float32)
            # Base puts each relevant document second; FT puts it first.
            columns = [1, 0] if self.name == "base" else [0, 1]
            vectors[np.arange(2), columns] = 1
            return vectors

        @staticmethod
        def scores(queries, documents):
            return queries @ documents.T

    def make_encoder(path, device, precision, max_length):
        assert (device, precision, max_length) == ("cpu", "fp32", 32)
        calls["encoders"].append(path.name)
        return FakeEncoder(path.name, device, precision)

    monkeypatch.setattr(exp, "resolve_model", resolve)
    monkeypatch.setattr(exp, "make_encoder", make_encoder)
    monkeypatch.setattr(exp, "release_memory", lambda device: calls["released"].append(device))
    monkeypatch.setattr(exp.importlib.metadata, "version", lambda package: "test")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeTokenizer))
    plan = {
        "mode": "run", "models": models, "inputs": paths,
        "output_dir": str(tmp_path / "output"),
        "settings": {"device": "cpu", "precision": "fp32", "max_length": 32,
                     "depth": 10, "doc_batch": 2, "query_batch": 2, "resamples": 100, "seed": 0},
    }
    return plan, calls


def test_mocked_execution_writes_complete_hashed_artifacts(inputs, fake_execution):
    plan, calls = fake_execution
    exp.execute(plan, **inputs)
    output = Path(plan["output_dir"])
    expected = {"status.json", "protocol.json", "base.json", "base.meta.json",
                "finetuned.json", "finetuned.meta.json", "comparison.json",
                "per_query.csv", "comparison.md"}
    assert {p.name for p in output.iterdir()} == expected
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "complete"
    assert status["artifacts"] == {name: exp.file_hash(output / name) for name in expected - {"status.json"}}
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["inputs"] == plan["inputs"]
    assert protocol["tokenizers_identical_on_evaluation_inputs"] is True
    assert set(protocol["model_file_sha256"]) == {"base", "finetuned"}
    for name, spec in plan["models"].items():
        # The normal Hugging Face parent directory is named .cache; its weights must be hashed.
        assert protocol["model_file_sha256"][name] == {
            filename: exp.file_hash(Path(spec["local_path"]) / filename)
            for filename in ("config.json", "fake.safetensors")
        }
    protocol_hash = exp.file_hash(output / "protocol.json")
    assert status["protocol_sha256"] == protocol_hash

    for name in ("base", "finetuned"):
        run = json.loads((output / f"{name}.json").read_text())
        meta = json.loads((output / f"{name}.meta.json").read_text())
        assert set(run) == {"q1", "q2"}
        assert all(set(ranking) == {"d1", "d2", "irrelevant"} for ranking in run.values())
        assert meta["protocol_sha256"] == protocol_hash
        assert meta["rankings_sha256"] == exp.file_hash(output / f"{name}.json")
    assert json.loads((output / "base.json").read_text())["q1"][:2] == ["d2", "d1"]
    assert json.loads((output / "finetuned.json").read_text())["q1"][0] == "d1"
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["base_ndcg10"] == pytest.approx(1 / math.log2(3))
    assert comparison["finetuned_ndcg10"] == 1
    assert comparison["difference"]["estimate"] == pytest.approx(1 - 1 / math.log2(3))
    assert comparison["protocol_sha256"] == protocol_hash
    with (output / "per_query.csv").open(newline="", encoding="utf-8") as stream:
        assert {row["qid"] for row in csv.DictReader(stream)} == {"q1", "q2"}
    assert calls["encoders"] == ["base", "finetuned"]
    assert [call[0] for call in calls["documents"]] == ["base", "finetuned"]
    assert [call[0] for call in calls["queries"]] == ["base", "finetuned"]
    assert calls["released"] == ["cpu", "cpu"]


def test_second_model_failure_records_failure_without_final_comparison(inputs, fake_execution):
    plan, calls = fake_execution
    calls["fail_model"] = "finetuned"
    with pytest.raises(RuntimeError, match="Synthetic second-stage failure"):
        exp.execute(plan, **inputs)

    output = Path(plan["output_dir"])
    status = json.loads((output / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["stage"] == "finetuned"
    assert status["error_type"] == "RuntimeError"
    assert (output / "protocol.json").is_file()
    assert (output / "base.json").is_file()
    for name in ("finetuned.json", "comparison.json", "comparison.md", "per_query.csv"):
        assert not (output / name).exists()
    assert calls["released"] == ["cpu", "cpu"]


def test_nonempty_output_is_rejected_before_models_or_rankings_are_reused(inputs, fake_execution):
    plan, calls = fake_execution
    output = Path(plan["output_dir"])
    output.mkdir()
    old_ranking = output / "base.json"
    old_content = '{"old-query": ["old-document"]}\n'
    old_ranking.write_text(old_content, encoding="utf-8")

    with pytest.raises(ValueError, match="nonempty"):
        exp.execute(plan, **inputs)

    assert calls["resolved"] == []
    assert calls["encoders"] == []
    assert calls["tokenizers"] == []
    assert old_ranking.read_text() == old_content
    assert {p.name for p in output.iterdir()} == {"base.json"}


@pytest.mark.parametrize("mode", ["--dry-run", "--run"])
def test_remote_cuda_preview_is_allowed_but_execution_requires_local_cuda(fake_execution, monkeypatch, capsys, mode):
    plan, calls = fake_execution

    def forbidden(*args, **kwargs):
        pytest.fail("Remote target preview or unavailable execution must not touch model/execution hooks")

    monkeypatch.setattr(exp, "detect_hardware", lambda: {"cuda": False, "mps": True, "torch_available": True})
    for name in ("resolve_model", "make_encoder", "execute"):
        monkeypatch.setattr(exp, name, forbidden)
    args = [mode, "--device", "cuda", "--out-dir", plan["output_dir"]]
    for name in ("dataset", "qrels", "corpus"):
        args.extend([f"--{name}", plan["inputs"][name]["path"]])

    if mode == "--dry-run":
        assert exp.main(args) == 0
        preview = json.loads(capsys.readouterr().out)
        assert preview["mode"] == "dry-run"
        assert preview["settings"]["device"] == "cuda"
        assert preview["settings"]["precision"] == "fp16"
        assert preview["settings"]["device_available_locally"] is False
        assert preview["hardware"] == {"cuda": False, "mps": True, "torch_available": True}
    else:
        with pytest.raises(ValueError, match="unavailable"):
            exp.main(args)

    assert not Path(plan["output_dir"]).exists()
    assert calls["resolved"] == []
    assert calls["encoders"] == []
