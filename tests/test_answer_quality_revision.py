"""H3 orchestration and cache checks with fake metrics; no API or model execution."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantitative_experiments.revision import answer_quality as exp


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    dataset = [
        {"dataset_id": "testb::1", "query": "Question one?", "positive_chunk_ids": ["d1"]},
        {"dataset_id": "testb::2", "query": "Question two?", "positive_chunk_ids": ["d2"]},
    ]
    dataset_path = tmp_path / "test_b.jsonl"
    write_rows(dataset_path, dataset)
    directory = tmp_path / "isolated"
    for condition in ("base", "finetuned"):
        write_rows(directory / f"{condition}.trace.jsonl", [
            {"dataset_id": row["dataset_id"], "answer": f"Answer {condition}", "ranked": ["d1", "d2"],
             "is_relevant": True, "no_results": False} for row in dataset])
    documents = {"d1": {"chunk_id": "d1", "text": "First source."},
                 "d2": {"chunk_id": "d2", "text": "Second source."}}
    monkeypatch.setattr(exp, "corpus", lambda: documents)
    settings = SimpleNamespace(openrouter_base_url="https://example.invalid/v1", openrouter_api_key="fake",
                               ragas_realtime_embeddings_model="fake-embeddings")
    monkeypatch.setattr(exp, "_app_settings", lambda dry_run: settings)
    args = SimpleNamespace(dataset=dataset_path, out_dir=directory, conditions=["base", "finetuned"],
                           reference="base", judge_model="fake-judge", judge_api="openrouter",
                           max_tokens=100, concurrency=2)
    return args, documents, settings


@pytest.fixture
def fake_ragas(monkeypatch):
    state = {"calls": [], "fail_once": False}

    class Faithfulness:
        def __init__(self, **kwargs):
            pass

        async def ascore(self, **kwargs):
            state["calls"].append(("faithfulness", kwargs))
            if state["fail_once"]:
                state["fail_once"] = False
                raise RuntimeError("Artificial metric failure")
            return SimpleNamespace(value=0.8)

    class AnswerRelevancy:
        def __init__(self, **kwargs):
            pass

        async def ascore(self, **kwargs):
            state["calls"].append(("answer_relevancy", kwargs))
            return SimpleNamespace(value=0.7)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=lambda **kwargs: object()))
    monkeypatch.setitem(sys.modules, "ragas", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ragas.embeddings", SimpleNamespace(OpenAIEmbeddings=lambda **kwargs: object()))
    monkeypatch.setitem(sys.modules, "ragas.llms", SimpleNamespace(llm_factory=lambda *a, **kw: SimpleNamespace(model_args={})))
    monkeypatch.setitem(sys.modules, "ragas.metrics", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", SimpleNamespace(
        Faithfulness=Faithfulness, AnswerRelevancy=AnswerRelevancy))
    return state


def test_helpers_keep_legacy_defaults_and_do_not_mutate_global_paths(tmp_path):
    assert exp.output_dir(SimpleNamespace()) == exp.DIR
    assert exp.dataset_path(SimpleNamespace()) == exp.TEST_DATASET
    assert exp.output_dir(SimpleNamespace(out_dir=tmp_path)) == tmp_path
    assert exp.DIR != tmp_path


def test_custom_dataset_requires_all_before_generation_dependencies(experiment, monkeypatch):
    args, _, _ = experiment
    args.all = False
    monkeypatch.setattr(exp, "_app_settings", lambda *_: pytest.fail("Must reject before importing app/settings"))
    with pytest.raises(ValueError, match="requires --all"):
        asyncio.run(exp.generate(args))
    assert len(exp.questions(True, args)) == 2


def test_source_context_has_legacy_alias_and_base_rerank_default(experiment, monkeypatch):
    args, _, _ = experiment
    dataset = exp.selected_dataset(args)
    expected = {"testb::1": ["d1"], "testb::2": ["d2"]}
    assert exp.contexts_for("source_context", None, dataset, 10) == expected
    assert exp.contexts_for("oracle", None, dataset, 10) == expected
    seen = []
    monkeypatch.setattr(exp, "released_run", lambda name: seen.append(name) or expected)
    assert exp.contexts_for("base_rerank", None, dataset, 10) == expected
    assert seen == ["dense_base_rerank100"]


@pytest.mark.parametrize("problem", ["duplicate", "different_population", "unknown", "duplicate_document"])
def test_invalid_traces_rejected_before_settings_or_scoring(experiment, monkeypatch, problem):
    args, _, _ = experiment
    path = args.out_dir / "finetuned.trace.jsonl"
    rows = exp.read_jsonl(path)
    if problem == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif problem == "different_population":
        rows.pop()
    elif problem == "unknown":
        rows[0]["dataset_id"] = "unknown"
    else:
        rows[0]["ranked"] = ["d1", "d1"]
    write_rows(path, rows)
    monkeypatch.setattr(exp, "_app_settings", lambda *_: pytest.fail("Invalid traces must fail before API setup"))
    with pytest.raises(ValueError):
        asyncio.run(exp.score(args))


def test_legacy_subset_is_allowed_but_custom_dataset_requires_all(experiment):
    args, _, _ = experiment
    dataset = exp.selected_dataset(args)
    row = exp.read_jsonl(args.out_dir / "base.trace.jsonl")[0]
    assert exp.validate_trace_population({"base": [row]}, dataset) == ["testb::1"]
    with pytest.raises(ValueError, match="cover all"):
        exp.validate_trace_population({"base": [row]}, dataset, require_all=True)


@pytest.mark.parametrize("change", ["answer", "query", "context_text", "context_order", "source_text", "source_identity", "judge", "max_tokens", "embeddings"])
def test_cache_key_covers_all_metric_inputs_and_configuration(experiment, change):
    args, documents, settings = experiment
    question = exp.selected_dataset(args)[0]
    trace = exp.read_jsonl(args.out_dir / "base.trace.jsonl")[0]
    trace["ranked"] = ["d1", "d2"] if change == "context_order" else ["d2"]

    def key():
        return exp.fingerprint({"input": exp.score_input_fingerprint(question, trace, documents),
                                "config": exp.score_configuration(args, settings)})

    before = key()
    if change == "answer":
        trace["answer"] += " changed"
    elif change == "query":
        question["query"] += " changed"
    elif change == "context_text":
        documents["d2"]["text"] += " changed"
    elif change == "context_order":
        trace["ranked"].reverse()
    elif change == "source_text":
        documents["d1"]["text"] += " changed"
    elif change == "source_identity":
        question["source_chunk"] = "d2"
    elif change == "judge":
        args.judge_model = "different-judge"
    elif change == "max_tokens":
        args.max_tokens += 1
    else:
        settings.ragas_realtime_embeddings_model = "other-embeddings"
    assert key() != before


def test_cached_scoring_reuses_only_matching_complete_rows(experiment, fake_ragas):
    args, _, _ = experiment
    asyncio.run(exp.score(args))
    assert len(fake_ragas["calls"]) == 12
    asyncio.run(exp.score(args))
    assert len(fake_ragas["calls"]) == 12
    rows = exp.read_jsonl(args.out_dir / "finetuned.trace.jsonl")
    rows[0]["answer"] += " revised"
    write_rows(args.out_dir / "finetuned.trace.jsonl", rows)
    asyncio.run(exp.score(args))
    assert len(fake_ragas["calls"]) == 15
    args.judge_model = "new-judge"
    asyncio.run(exp.score(args))
    assert len(fake_ragas["calls"]) == 27
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["n_questions"] == 2
    assert summary["coverage"]["base"]["faithfulness"] == {"scored": 2, "missing": 0, "not_applicable": 0}


def test_incomplete_scores_are_retried_and_unfingerprinted_legacy_cache_is_ignored(experiment, fake_ragas):
    args, _, _ = experiment
    write_rows(args.out_dir / "base.ragas_partial.jsonl", [{"dataset_id": "testb::1", "faithfulness": 1,
                                                          "gold_support": 1, "answer_relevancy": 1}])
    fake_ragas["fail_once"] = True
    asyncio.run(exp.score(args))
    assert len(fake_ragas["calls"]) == 12
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["coverage"]["base"]["faithfulness"]["missing"] == 1
    asyncio.run(exp.score(args))
    assert len(fake_ragas["calls"]) == 15
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["coverage"]["base"]["faithfulness"]["missing"] == 0


def test_no_context_faithfulness_is_not_retried_as_failure(experiment, fake_ragas):
    args, _, _ = experiment
    args.conditions = ["none"]
    args.reference = "none"
    rows = exp.read_jsonl(args.out_dir / "base.trace.jsonl")
    for row in rows:
        row["ranked"] = []
    write_rows(args.out_dir / "none.trace.jsonl", rows)
    asyncio.run(exp.score(args))
    asyncio.run(exp.score(args))
    assert len(fake_ragas["calls"]) == 4
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["coverage"]["none"]["faithfulness"] == {"scored": 0, "missing": 0, "not_applicable": 2}


def test_summary_does_not_drop_whole_queries_missing_from_one_condition(experiment):
    args, _, _ = experiment
    complete = {"faithfulness": 0.8, "gold_support": 0.7, "answer_relevancy": 0.6}
    per = {"base": {"testb::1": complete, "testb::2": complete}, "finetuned": {"testb::1": complete}}
    before = copy.deepcopy(per)
    exp.summarize(args, per)
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["n_questions"] == 2
    assert summary["question_ids"] == ["testb::1", "testb::2"]
    assert summary["coverage"]["finetuned"]["faithfulness"] == {"scored": 1, "missing": 1, "not_applicable": 0}
    assert summary["paired"]["finetuned|faithfulness"]["n"] == 1
    assert summary["paired"]["finetuned|faithfulness"]["missing"] == 1
    assert summary["paired"]["finetuned|faithfulness"]["status"] == "insufficient_paired_scores"
    assert per == before


def test_all_200_custom_queries_are_scored_and_paired(experiment, fake_ragas):
    args, _, _ = experiment
    rows = [{"dataset_id": f"testb::{i}", "query": f"Question {i}?", "positive_chunk_ids": ["d1" if i % 2 else "d2"]}
            for i in range(200)]
    write_rows(args.dataset, rows)
    for condition in args.conditions:
        write_rows(args.out_dir / f"{condition}.trace.jsonl", [
            {"dataset_id": r["dataset_id"], "answer": f"Answer {condition}", "ranked": ["d1", "d2"]} for r in rows])
    asyncio.run(exp.score(args))
    summary = json.loads((args.out_dir / "summary.json").read_text())
    assert summary["n_questions"] == 200
    assert summary["paired"]["finetuned|faithfulness"]["n"] == 200
    assert summary["coverage"]["finetuned"]["faithfulness"]["scored"] == 200
    assert len(fake_ragas["calls"]) == 1200


@pytest.mark.parametrize("command,extra,handler", [
    ("generate", ["--condition", "base", "--all", "--dry-run"], "generate"),
    ("score", ["--conditions", "base", "finetuned"], "score"),
    ("citations", ["--conditions", "base"], "citations_cmd"),
    ("code-errors", ["--condition", "base", "--model", "fake", "--dry-run"], "code_errors"),
])
def test_every_subcommand_accepts_dataset_and_isolated_output(experiment, monkeypatch, command, extra, handler):
    args, _, _ = experiment
    seen = []

    async def async_handler(parsed):
        seen.append(parsed)

    def sync_handler(parsed):
        seen.append(parsed)

    monkeypatch.setattr(exp, handler, sync_handler if handler == "citations_cmd" else async_handler)
    exp.main([command, *extra, "--dataset", str(args.dataset), "--out-dir", str(args.out_dir)])
    assert len(seen) == 1
    assert seen[0].dataset == args.dataset
    assert seen[0].out_dir == args.out_dir


def test_citations_uses_selected_population_and_output(experiment):
    args, _, _ = experiment
    exp.citations_cmd(args)
    output = json.loads((args.out_dir / "citations.json").read_text())
    assert set(output) == {"base", "finetuned"}
    assert {r["dataset_id"] for r in output["base"]["per_answer"]} == {"testb::1", "testb::2"}


def test_custom_dataset_cannot_use_released_test_a_final_trace(experiment):
    args, _, _ = experiment
    args.conditions = ["released_final"]
    with pytest.raises(ValueError, match="Test A"):
        exp.citations_cmd(args)


def test_code_errors_rejects_scores_from_changed_answers(experiment, fake_ragas):
    args, _, _ = experiment
    asyncio.run(exp.score(args))
    args.condition = "base"
    rows = exp.read_jsonl(args.out_dir / "base.trace.jsonl")
    for row in rows:
        row["answer"] += " changed"
    write_rows(args.out_dir / "base.trace.jsonl", rows)
    with pytest.raises(ValueError, match="No scores match"):
        asyncio.run(exp.code_errors(args))


def test_generation_routes_selected_questions_and_cache_to_isolated_directory(experiment, monkeypatch):
    args, _, settings = experiment
    settings.llm_model = "fake-generator"
    settings.llm_temperature = 0
    settings.llm_max_output_tokens = 100
    args.all, args.condition, args.k, args.dry_run = True, "base", 10, False
    run_path = args.out_dir / "input_run.json"
    run_path.write_text(json.dumps({"testb::1": ["d1"], "testb::2": ["d2"]}), encoding="utf-8")
    args.run = str(run_path)
    calls = []

    class FakeLLM:
        def __init__(self, model, cache_path, **kwargs):
            assert model == "fake-generator"
            assert cache_path == args.out_dir / "cache_generate.jsonl"

        async def text(self, messages):
            calls.append(messages)
            return "A synthetic answer."

        def report(self):
            return "fake generation"

    monkeypatch.setattr(exp, "LLM", FakeLLM)
    monkeypatch.setitem(sys.modules, "app.core.dependencies", SimpleNamespace(_provider_preferences=lambda: None))
    monkeypatch.setitem(sys.modules, "app.core.prompt_provider", SimpleNamespace(prompts=SimpleNamespace(generation="Fixed prompt")))
    monkeypatch.setitem(sys.modules, "app.graph.nodes.generation", SimpleNamespace(_build_context=lambda docs: "\n".join(d["text"] for d in docs)))
    monkeypatch.setitem(sys.modules, "quantitative_experiments.provenance", SimpleNamespace(
        run_provenance=lambda **kwargs: kwargs, sha256_text=exp.fingerprint))

    asyncio.run(exp.generate(args))

    assert len(calls) == 2
    assert all(c[0]["content"] == "Fixed prompt" for c in calls)
    assert "Question one?" in calls[0][1]["content"]
    assert "First source." in calls[0][1]["content"]
    rows = exp.read_jsonl(args.out_dir / "base.trace.jsonl")
    assert [r["dataset_id"] for r in rows] == ["testb::1", "testb::2"]
    provenance = json.loads((args.out_dir / "base.provenance.json").read_text())
    assert provenance["dataset"] == str(args.dataset.resolve())
    assert provenance["question_ids"] == ["testb::1", "testb::2"]


def test_code_errors_uses_selected_dataset_and_isolated_cache(experiment, fake_ragas, monkeypatch):
    args, _, _ = experiment
    asyncio.run(exp.score(args))
    args.condition, args.threshold, args.model, args.dry_run = "base", 0.9, "fake-coder", False
    calls = []

    class FakeLLM:
        def __init__(self, model, cache_path, **kwargs):
            assert model == "fake-coder"
            assert cache_path == args.out_dir / "cache_code_errors.jsonl"

        async def json(self, system, user, validate):
            calls.append(user)
            result = {"claims": [], "main_cause": "consejo_general"}
            validate(result)
            return result

        def report(self):
            return "fake coding"

    monkeypatch.setattr(exp, "LLM", FakeLLM)
    asyncio.run(exp.code_errors(args))
    assert len(calls) == 2
    assert any("Question one?" in query for query in calls)
    output = json.loads((args.out_dir / "base.error_codes.json").read_text())
    assert output["answers"] == 2
    assert set(output["per_answer"]) == {"testb::1", "testb::2"}
