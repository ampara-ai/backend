"""Offline checks for relevance sets derived exclusively from the second judge."""

import json
import math

import pytest

from quantitative_experiments.revision import second_judge_independent as exp


MODEL = "google/gemma-4-31b-it"


def judgment(qid, chunk_id, relevance, **extra):
    return {"qid": qid, "chunk_id": chunk_id, "relevance": relevance,
            "judge_model": MODEL, **extra}


def test_consistent_duplicates_allow_different_rationales():
    rows = [judgment("q1", "d1", "si", rationale="First explanation"),
            judgment("q1", "d1", "si", rationale="Another explanation"),
            judgment("q1", "d2", "no")]
    expected = {("q1", "d1"), ("q1", "d2")}

    labels, audit = exp.load_judgments(rows, expected, MODEL)

    assert labels == {("q1", "d1"): "si", ("q1", "d2"): "no"}
    assert audit["raw_rows"] == 3
    assert audit["unique_pairs"] == 2
    assert audit["duplicate_rows"] == 1
    assert audit["expected_pairs"] == 2


def test_conflicting_duplicate_roles_are_rejected():
    rows = [judgment("q1", "d1", "si", role="origin"), judgment("q1", "d1", "si", role="pool")]
    with pytest.raises(ValueError, match="Conflicting duplicate role"):
        exp.load_judgments(rows, {("q1", "d1")}, MODEL)


def test_equal_counts_do_not_hide_missing_and_extra_pairs():
    rows = [judgment("q1", "d1", "si"), judgment("q1", "unexpected", "no")]
    with pytest.raises(ValueError, match="1 missing, 1 extra"):
        exp.load_judgments(rows, {("q1", "d1"), ("q1", "d2")}, MODEL)


@pytest.mark.parametrize("problem", ["conflicting_duplicate", "invalid_label", "wrong_model", "missing", "extra"])
def test_invalid_judgment_inputs_are_rejected(problem):
    rows = [judgment("q1", "d1", "si"), judgment("q1", "d2", "no")]
    expected = {("q1", "d1"), ("q1", "d2")}
    if problem == "conflicting_duplicate":
        rows.append(judgment("q1", "d1", "no"))
    elif problem == "invalid_label":
        rows[0]["relevance"] = "yes"
    elif problem == "wrong_model":
        rows[0]["judge_model"] = "a-different-judge"
    elif problem == "missing":
        rows.pop()
    else:
        rows.append(judgment("q1", "unrequested", "si"))

    with pytest.raises(ValueError):
        exp.load_judgments(rows, expected, MODEL)


def test_source_labels_are_not_forced_into_independent_qrels():
    labels = {("q1", "source1"): "no", ("q1", "pool1"): "si",
              ("q2", "source2"): "parcial", ("q2", "pool2"): "no"}
    source = {"q1": {"source1"}, "q2": {"source2"}}

    variants = exp.build_qrels(labels, source, ["q1", "q2"])

    assert variants["independent"] == {"q1": {"pool1"}, "q2": set()}
    assert variants["source_anchored"] == {"q1": {"source1", "pool1"}, "q2": {"source2"}}
    assert source == {"q1": {"source1"}, "q2": {"source2"}}


@pytest.mark.parametrize("problem", ["missing_query", "extra_query", "non_list", "non_string_doc",
                                     "empty_doc", "duplicate_doc"])
def test_invalid_run_inputs_are_rejected(problem):
    run = {"q1": ["d1"], "q2": ["d2"]}
    if problem == "missing_query":
        del run["q2"]
    elif problem == "extra_query":
        run["q3"] = ["d3"]
    elif problem == "non_list":
        run["q1"] = "d1"
    elif problem == "non_string_doc":
        run["q1"] = [1]
    elif problem == "empty_doc":
        run["q1"] = [""]
    else:
        run["q1"] = ["d1", "d1"]

    with pytest.raises(ValueError):
        exp.validate_runs({"dense_base": run}, ["q1", "q2"])


def test_empty_rankings_are_valid_observed_failures():
    exp.validate_runs({"dense_base": {"q1": [], "q2": ["d2"]}}, ["q1", "q2"])


@pytest.fixture
def small_experiment():
    return {
        "dataset": [{"dataset_id": "q1", "source_chunk": "source1"},
                    {"dataset_id": "q2", "source_chunk": "source2"}],
        "source_qrels": {"q1": {"source1"}, "q2": {"source2"}},
        "pool_rows": [{"qid": "q1", "chunk_id": "pool1"},
                      {"qid": "q2", "chunk_id": "pool2"}],
        "judgment_rows": [judgment("q1", "source1", "no"), judgment("q1", "pool1", "si"),
                          judgment("q2", "source2", "parcial"), judgment("q2", "pool2", "no")],
        "runs": {
            "dense_base": {"q1": ["source1", "pool1"], "q2": ["source2", "pool2"]},
            "dense_finetuned": {"q1": ["pool1", "source1"], "q2": ["pool2", "source2"]},
            "empty_system": {"q1": [], "q2": []},
        },
        "model": MODEL,
    }


def test_hand_computed_metrics_keep_zero_positive_queries(small_experiment):
    report, per_query, variants = exp.analyze(**small_experiment, n_resamples=1000, seed=0)
    discount = 1 / math.log2(3)

    assert report["qrels"]["independent"]["relevant_pairs"] == 1
    assert report["qrels"]["independent"]["zero_relevant_qids"] == ["q2"]
    assert report["qrels"]["source_anchored"]["relevant_pairs"] == 3
    assert report["qrels"]["source_anchored"]["zero_relevant_qids"] == []
    assert variants["independent"]["q2"] == set()
    assert per_query

    independent = report["by_config"]["independent"]
    anchored = report["by_config"]["source_anchored"]
    assert independent["dense_base"]["ndcg@10"] == pytest.approx(discount / 2)
    assert independent["dense_finetuned"]["ndcg@10"] == pytest.approx(0.5)
    assert independent["empty_system"]["ndcg@10"] == 0
    assert anchored["dense_base"]["ndcg@10"] == pytest.approx(1)
    assert anchored["dense_finetuned"]["ndcg@10"] == pytest.approx((1 + discount) / 2)

    primary = report["primary_ft_minus_base"]
    assert primary["independent"]["ndcg@10"]["estimate"] == pytest.approx((1 - discount) / 2)
    assert primary["source_anchored"]["ndcg@10"]["estimate"] == pytest.approx((discount - 1) / 2)
    for metric in ("recall@10", "recall_capped@10"):
        assert independent["dense_base"][metric] == pytest.approx(0.5)
        assert independent["dense_finetuned"][metric] == pytest.approx(0.5)
        assert anchored["dense_base"][metric] == pytest.approx(1)
        assert primary["independent"][metric]["estimate"] == pytest.approx(0)


def test_bootstrap_respects_shared_source_clusters(small_experiment):
    small_experiment["dataset"][1]["source_chunk"] = "source1"
    small_experiment["source_qrels"]["q2"] = {"source1"}
    small_experiment["judgment_rows"][2]["chunk_id"] = "source1"
    small_experiment["runs"]["dense_base"]["q2"] = ["source1", "pool2"]
    small_experiment["runs"]["dense_finetuned"]["q2"] = ["pool2", "source1"]
    report, _, _ = exp.analyze(**small_experiment, n_resamples=100, seed=7)
    repeated, _, _ = exp.analyze(**small_experiment, n_resamples=100, seed=7)
    primary = report["primary_ft_minus_base"]

    assert primary == repeated["primary_ft_minus_base"]
    # With only one source cluster, resampling cannot alter its query-weighted mean.
    interval = primary["independent"]["ndcg@10"]
    assert interval["ci_low"] == pytest.approx(interval["estimate"])
    assert interval["ci_high"] == pytest.approx(interval["estimate"])


def test_ordinary_recall_is_not_capped_when_more_than_ten_chunks_are_relevant():
    chunks = [f"d{i}" for i in range(12)]
    report, _, _ = exp.analyze(
        [{"dataset_id": "q", "source_chunk": "d0"}], {"q": {"d0"}},
        [{"qid": "q", "chunk_id": c} for c in chunks[1:]],
        [judgment("q", c, "si") for c in chunks],
        {name: {"q": chunks} for name in ("dense_base", "dense_finetuned")}, MODEL,
        n_resamples=10,
    )
    scores = report["by_config"]["independent"]["dense_base"]
    assert scores["recall@10"] == pytest.approx(10 / 12)
    assert scores["recall_capped@10"] == pytest.approx(1)


@pytest.mark.parametrize("required", ["dense_base", "dense_finetuned"])
def test_analysis_requires_both_primary_runs(small_experiment, required):
    del small_experiment["runs"][required]
    with pytest.raises(ValueError):
        exp.analyze(**small_experiment, n_resamples=100, seed=0)


def test_released_experiment_cli_reproduces_independent_result(tmp_path):
    assert exp.main(["--model", MODEL, "--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "analysis.json").read_text())
    assert report["queries"] == 198
    assert report["source_clusters"] == 114
    assert report["coverage"]["unique_pairs"] == 11963
    assert report["coverage"]["duplicate_rows"] == 138
    assert report["source_label_counts"] == {"si": 197, "parcial": 2, "no": 1}
    assert len(report["source_disagreements"]) == 3
    assert report["qrels"]["independent"] == {"relevant_pairs": 811, "zero_relevant_qids": []}
    assert report["qrels"]["source_anchored"]["relevant_pairs"] == 814
    result = report["primary_ft_minus_base"]["independent"]["ndcg@10"]
    assert result == pytest.approx({"estimate": 0.11718947360668314,
                                    "ci_low": 0.07334810558247873, "ci_high": 0.16262455227734807})
    assert report["by_config"]["independent"]["dense_base"]["ndcg@10"] == pytest.approx(0.428253722227635)
    assert report["by_config"]["independent"]["dense_finetuned"]["ndcg@10"] == pytest.approx(0.5454431958343182)
    assert all(v == 0 for v in report["unjudged_top10_by_config"].values())
    independent = exp.load_qrels(tmp_path / "qrels_independent.jsonl")
    anchored = exp.load_qrels(tmp_path / "qrels_source_anchored.jsonl")
    overrides = {(q, c) for q in anchored for c in anchored[q] - independent.get(q, set())}
    assert overrides == {(r["qid"], r["chunk_id"]) for r in report["source_disagreements"]}
    assert len(exp.read_jsonl(tmp_path / "per_query.jsonl")) == 396
    assert all(len(v["sha256"]) == 64 for v in report["provenance"].values())
    assert (tmp_path / "analysis.md").is_file()
