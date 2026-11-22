"""Preparation/analysis checks: no models, training, network, or paid API calls."""

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from quantitative_experiments.revision import confirmatory, h3_checks, rerank_local


@pytest.fixture
def primary_results():
    frozen = confirmatory.frozen_test_b()
    hashes = {k: frozen[k] for k in ("dataset_sha256", "qrels_sha256")}
    return {
        "h1": {"hypothesis": "H1", "metric": "strict NDCG@10", "raw_bootstrap_tail_p": 0.01,
               "difference": {"estimate": 0.1}, "queries": 200, **hashes},
        "h2": {"provenance": {"reference": "naive", "questions": 200, **hashes},
               "results": {"ndcg@10": {"full_matched": {"p_boot": 0.04, "diff": 0.07, "seeds": 3}, "naive": {"seeds": 3}}}},
        "h3": {"reference": "base", "n_questions": 200, "dataset_sha256": frozen["dataset_sha256"],
               "question_ids": frozen["question_ids"],
               "coverage": {c: {"faithfulness": {"scored": 200, "missing": 0, "not_applicable": 0}}
                            for c in ["base", "finetuned"]},
               "paired": {"finetuned|faithfulness": {"n": 200, "p_boot": 0.03, "estimate": 0.05}}},
    }


def test_exact_primary_family_is_adjusted_together(primary_results):
    out = confirmatory.aggregate(**primary_results)
    assert out["family"] == ["H1", "H2", "H3"]
    assert {h: v["p_holm"] for h, v in out["results"].items()} == pytest.approx({"H1": 0.03, "H2": 0.06, "H3": 0.06})
    assert out["results"]["H1"]["positive_effect_significant"]
    assert not out["results"]["H3"]["positive_effect_significant"]


@pytest.mark.parametrize("problem", ["missing_h2", "wrong_h2_reference", "wrong_h3_reference", "partial_h3", "paired_subset", "wrong_dataset", "one_seed", "nan", "out_of_range"])
def test_incomplete_or_invalid_family_is_rejected(primary_results, problem):
    x = copy.deepcopy(primary_results)
    if problem == "missing_h2":
        x["h2"]["results"] = {}
    elif problem == "wrong_h2_reference":
        x["h2"]["provenance"]["reference"] = "base"
    elif problem == "wrong_h3_reference":
        x["h3"]["reference"] = "source_context"
    elif problem == "partial_h3":
        x["h3"]["coverage"]["finetuned"]["faithfulness"] = {"scored": 199, "missing": 1, "not_applicable": 0}
    elif problem == "paired_subset":
        x["h3"]["paired"]["finetuned|faithfulness"]["n"] = 199
    elif problem == "wrong_dataset":
        x["h2"]["provenance"]["dataset_sha256"] = "test-a-supporting"
    elif problem == "one_seed":
        x["h2"]["results"]["ndcg@10"]["full_matched"]["seeds"] = 1
    elif problem == "nan":
        x["h1"]["raw_bootstrap_tail_p"] = float("nan")
    else:
        x["h1"]["raw_bootstrap_tail_p"] = 2
    with pytest.raises(ValueError):
        confirmatory.aggregate(**x)


def test_confirmatory_cli_records_inputs(primary_results, tmp_path):
    args = []
    for name, value in primary_results.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value))
        args += [f"--{name}", str(path)]
    output = tmp_path / "combined.json"
    assert confirmatory.main([*args, "--out", str(output)]) == 0
    result = json.loads(output.read_text())
    assert all(len(v["sha256"]) == 64 for v in result["inputs"].values())
    with pytest.raises(ValueError, match="overwrite"):
        confirmatory.main([*args, "--out", str(tmp_path / "h1.json")])


@pytest.mark.parametrize("change", [None, "prompt_file", "generator", "effective_prompt", "questions"])
def test_h3_generation_controls_are_checked_before_scoring(tmp_path, change):
    output = tmp_path / "h3"
    (output / "answers").mkdir(parents=True)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("fixed prompt")
    llm = {"llm_model": "generator", "llm_temperature": 0, "provider": {"only": ["fixed"]},
           "prompts_sha256": {"generation": "effective-hash"}}
    conditions = ["base", "finetuned", "base_rerank", "finetuned_rerank"]
    protocol = {"generation_llm": llm, "question_ids": ["q1", "q2"], "conditions": conditions,
                "frozen_inputs": {"prompt": {"path": str(prompt), "sha256": h3_checks.file_hash(prompt)}}}
    (output / "protocol.json").write_text(json.dumps(protocol))
    for condition in conditions:
        prov = {"llm": copy.deepcopy(llm), "question_ids": ["q1", "q2"], "k": 10, "system_prompt_sha256": "effective-hash"}
        if condition == "finetuned":
            if change == "generator": prov["llm"]["llm_model"] = "other"
            if change == "effective_prompt": prov["system_prompt_sha256"] = "changed"
            if change == "questions": prov["question_ids"] = ["q1"]
        (output / "answers" / f"{condition}.provenance.json").write_text(json.dumps(prov))
    if change == "prompt_file": prompt.write_text("edited during generation")
    if change:
        with pytest.raises(ValueError): h3_checks.verify_generation(output)
    else:
        h3_checks.verify_generation(output)


@pytest.mark.parametrize("scores", [[float("nan"), 0.1], [0.1], [float("inf"), 0.2]])
def test_reranker_invalid_scores_fail(scores):
    class Fake:
        def score(self, query, docs):
            return np.array(scores)
    with pytest.raises(ValueError, match="scores"):
        rerank_local.rerank(Fake(), "q", ["a", "b"], 2, {"a": "A", "b": "B"})


def test_reranker_dry_run_never_loads_model_or_creates_output(tmp_path, monkeypatch):
    monkeypatch.setattr(rerank_local, "load_dataset", lambda _: [{"dataset_id": "q", "query": "question"}])
    monkeypatch.setattr(rerank_local, "load_run", lambda _: {"q": ["a", "b"]})
    monkeypatch.setattr(rerank_local, "load_qrels", lambda _: {"q": {"a"}})
    monkeypatch.setattr(rerank_local, "corpus", lambda: {"a": {"text": "A"}, "b": {"text": "B"}})
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run must not construct a reranker")
    monkeypatch.setattr(rerank_local, "Reranker", forbidden)
    output = tmp_path / "must-not-exist"
    args = ["--first-stage", "fake.json", "--name", "test", "--depths", "2", "--out-dir", str(output)]
    rerank_local.main([*args, "--dry-run"])
    assert not output.exists()
    output.mkdir()
    (output / "test_k2.json").write_text('{}')
    with pytest.raises(ValueError, match="already exists"):
        rerank_local.main(args)


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _api_reranker(tmp_path, monkeypatch, results, calls):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")

    def fake_urlopen(request, timeout=0):
        calls.append(json.loads(request.data))
        return _FakeResponse({"results": results})
    monkeypatch.setattr(rerank_local.urllib.request, "urlopen", fake_urlopen)
    return rerank_local.Reranker("openrouter:mongodb/rerank-2.5", "auto", 1024, True, cache_path=tmp_path / "cache.jsonl")


def test_api_reranker_orders_by_score_breaks_ties_by_chunk_id_and_caches(tmp_path, monkeypatch):
    calls = []
    # API returns results sorted by score, with indices out of document order
    rr = _api_reranker(tmp_path, monkeypatch, [{"index": 2, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.5},
                                              {"index": 1, "relevance_score": 0.5}], calls)
    ranked, _ = rerank_local.rerank(rr, "q", ["b", "a", "c"], 3, {"a": "A", "b": "B", "c": "C"})
    assert ranked == ["c", "a", "b"]  # c highest; a/b tie broken alphabetically by chunk id
    assert calls[0]["model"] == "mongodb/rerank-2.5" and calls[0]["top_n"] == 3 and calls[0]["documents"] == ["B", "A", "C"]
    again = rerank_local.Reranker("openrouter:mongodb/rerank-2.5", "auto", 1024, True, cache_path=tmp_path / "cache.jsonl")
    rerank_local.rerank(again, "q", ["b", "a", "c"], 3, {"a": "A", "b": "B", "c": "C"})
    assert len(calls) == 1 and again.api_calls == 0  # second run is served from the disk cache


@pytest.mark.parametrize("results", [[{"index": 0, "relevance_score": 0.5}],
                                     [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}]])
def test_api_reranker_rejects_incomplete_or_duplicate_results(tmp_path, monkeypatch, results):
    rr = _api_reranker(tmp_path, monkeypatch, results, [])
    with pytest.raises(ValueError):
        rerank_local.rerank(rr, "q", ["a", "b"], 2, {"a": "A", "b": "B"})


def test_api_reranker_requires_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        rerank_local.Reranker("openrouter:mongodb/rerank-2.5", "auto", 1024, True)


@pytest.mark.parametrize("script", ["run_h2.sh", "run_h3.sh"])
@pytest.mark.parametrize("flags", [[], ["--dry-run"]])
def test_run_wrappers_prepare_only(script, flags, tmp_path):
    root = Path(__file__).resolve().parents[1]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    marker = tmp_path / "unexpected-execution"
    for name in ["python", "python3", "torchrun", "nvidia-smi"]:
        exe = fakebin / name
        exe.write_text('#!/bin/sh\ntouch "$EXECUTION_MARKER"\nexit 99\n')
        exe.chmod(0o755)
    env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"], "EXECUTION_MARKER": str(marker),
           "AMPARA_DATA_REPO": str(tmp_path / "absent data"), "AMPARA_BASE_SNAPSHOT": str(tmp_path / "absent model"),
           "H1_RUN_DIR": str(tmp_path / "absent h1"), "RERANKER_SPEC": "openrouter:mongodb/rerank-2.5",
           "H2_OUT_DIR": str(tmp_path / "h2 output"), "H3_OUT_DIR": str(tmp_path / "h3 output"), "PYTHON": "python3"}
    result = subprocess.run(["bash", str(root / "quantitative_experiments/revision" / script), *flags],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert not marker.exists()
    assert not (tmp_path / "h2 output").exists()
    assert not (tmp_path / "h3 output").exists()
    if script == "run_h2.sh":
        assert result.stdout.count("run_training.sh") == 6
        assert result.stdout.count("revision.select_checkpoint") == 6
        assert result.stdout.count("revision.local_retrieval") == 12
    else:
        assert result.stdout.count("revision.rerank_local") == 2
        assert result.stdout.count("revision.answer_quality generate") == 4


def test_protocol_differences_ignore_freeze_time_and_dirty_flag():
    saved = {"frozen_at_utc": "t1", "judge_model": "j", "git": {"commit": "abc", "dirty": False}, "generation_prompt_sha256": "p"}
    same = {**saved, "frozen_at_utc": "t2", "git": {"commit": "abc", "dirty": True}}
    assert h3_checks.protocol_differences(saved, same) == []
    changed = {**same, "judge_model": "other", "git": {"commit": "def", "dirty": False}, "generation_prompt_sha256": "q"}
    assert h3_checks.protocol_differences(saved, changed) == ["generation_prompt_sha256", "git", "judge_model"]
    assert h3_checks.protocol_differences(saved, {k: v for k, v in saved.items() if k != "judge_model"}) == ["judge_model"]


@pytest.mark.parametrize("flags", [["--resume"], ["--dry-run", "--resume"]])
def test_run_h3_resume_requires_run(flags, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "quantitative_experiments/revision/run_h3.sh"), *flags],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2 and "--resume only applies to --run" in result.stderr


def test_run_h3_resume_skips_only_completed_steps(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fakebin, log, out = tmp_path / "bin", tmp_path / "calls.log", tmp_path / "h3"
    fakebin.mkdir()
    for name in ["python", "python3"]:
        exe = fakebin / name
        exe.write_text('#!/bin/sh\necho "$@" >> "$CALL_LOG"\nexit 0\n')
        exe.chmod(0o755)
    for path in ["rerank/base/base_rerank_k100.json", "rerank/base/base_rerank.meta.json",
                 "answers/base.trace.jsonl", "answers/base.provenance.json"]:
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        (out / path).write_text("{}")
    env = {**os.environ, "PATH": str(fakebin) + os.pathsep + os.environ["PATH"], "CALL_LOG": str(log),
           "OPENROUTER_API_KEY": "not-a-real-key", "H3_OUT_DIR": str(out), "PYTHON": "python3"}
    result = subprocess.run(["bash", str(root / "quantitative_experiments/revision/run_h3.sh"), "--run", "--resume"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "resume: rerank for base already complete" in result.stdout
    assert "resume: answers for base already complete" in result.stdout
    assert calls.count("revision.rerank_local") == 1 and "--name finetuned_rerank" in calls  # base rerank skipped
    assert calls.count("answer_quality generate") == 3 and "--condition base --run" not in calls
    assert "--condition finetuned --run" in calls and "--condition base_rerank --run" in calls
    assert calls.count("answer_quality score") == 1 and calls.count("revision.h3_checks") == 2  # always re-run, cache-backed
