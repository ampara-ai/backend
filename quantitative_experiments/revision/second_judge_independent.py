"""Offline sensitivity analysis: let the second judge reassess source positives too.

Unlike second_judge.analyze, the independent policy never forces source chunks to
be relevant. The source-anchored policy is retained as a controlled comparison.
This is a retrospective retrieval robustness analysis, not legal validation.

Usage:
    python -m quantitative_experiments.revision.second_judge_independent \
        --model google/gemma-4-31b-it
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import numpy as np

from evaluation.metrics import ndcg_at_k, recall_at_k
from quantitative_experiments.revision.data import (
    CONFIGS, EXP, OUT, RESULTS, STRICT_QRELS, TEST_DATASET, bootstrap_for,
    fmt_ci, load_dataset, load_qrels, load_run, paired, read_jsonl, write_jsonl,
)

LABELS = ("si", "parcial", "no")
METRICS = {
    "ndcg@10": lambda ranking, relevant: ndcg_at_k(ranking, relevant, 10),
    "recall@10": lambda ranking, relevant: recall_at_k(ranking, relevant, 10),
    "recall_capped@10": lambda ranking, relevant: recall_at_k(ranking, relevant, 10, capped=True),
}


def load_judgments(rows, expected_pairs, model):
    """Collapse consistent retry duplicates; never silently choose a conflicting label."""
    labels, roles = {}, {}
    for index, row in enumerate(rows, 1):
        qid, chunk = row.get("qid"), row.get("chunk_id")
        if not all(isinstance(v, str) and v for v in (qid, chunk)):
            raise ValueError(f"Judgment row {index}: invalid qid/chunk_id")
        key = (qid, chunk)
        label = row.get("relevance")
        if label not in LABELS:
            raise ValueError(f"Judgment {key}: invalid relevance {label!r}")
        if row.get("judge_model") != model:
            raise ValueError(f"Judgment {key}: judge_model must be {model!r}")
        if key in labels and labels[key] != label:
            raise ValueError(f"Conflicting duplicate judgment for {key}")
        if key in roles and roles[key] != row.get("role"):
            raise ValueError(f"Conflicting duplicate role for {key}")
        labels[key] = label
        roles[key] = row.get("role")
    missing, extra = expected_pairs - labels.keys(), labels.keys() - expected_pairs
    if missing or extra:
        raise ValueError(f"Judgment coverage mismatch: {len(missing)} missing, {len(extra)} extra pairs; "
                         f"examples missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")
    counts = Counter(labels.values())
    return labels, {
        "expected_pairs": len(expected_pairs), "raw_rows": len(rows),
        "unique_pairs": len(labels), "duplicate_rows": len(rows) - len(labels),
        "labels": {label: counts[label] for label in LABELS},
    }


def build_qrels(labels, source_qrels, qids):
    independent = {q: set() for q in qids}
    for (q, chunk), label in labels.items():
        if label == "si":
            independent[q].add(chunk)
    return {
        "independent": independent,
        "source_anchored": {q: independent[q] | source_qrels[q] for q in qids},
    }


def validate_runs(runs, qids):
    for name, run in runs.items():
        if not isinstance(run, dict) or set(run) != set(qids):
            raise ValueError(f"Run {name}: query IDs must match the dataset exactly")
        for q, ranking in run.items():
            if not isinstance(ranking, list) or any(not isinstance(c, str) or not c for c in ranking):
                raise ValueError(f"Run {name}, {q}: ranking must be a list of nonempty chunk IDs")
            if len(ranking) != len(set(ranking)):
                raise ValueError(f"Run {name}, {q}: duplicate documents in ranking")


def analyze(dataset, source_qrels, pool_rows, judgment_rows, runs, model, *, n_resamples=10_000, seed=0):
    qids = [d["dataset_id"] for d in dataset]
    if not qids or len(qids) != len(set(qids)):
        raise ValueError("Dataset must contain unique query IDs and at least one query")
    if set(source_qrels) != set(qids) or any(not source_qrels[q] for q in qids):
        raise ValueError("Source qrels must cover every dataset query exactly")
    if any(d["source_chunk"] not in source_qrels[d["dataset_id"]] for d in dataset):
        raise ValueError("Dataset clustering source must belong to source qrels")
    if n_resamples < 1 or seed < 0:
        raise ValueError("Bootstrap resamples must be positive and seed nonnegative")
    source_pairs = {(q, c) for q, chunks in source_qrels.items() for c in chunks}
    pool_pairs = set()
    for row in pool_rows:
        q, c = row.get("qid"), row.get("chunk_id")
        if not isinstance(q, str) or q not in source_qrels or not isinstance(c, str) or not c:
            raise ValueError("Pool contains an unknown query or invalid chunk ID")
        pool_pairs.add((q, c))
    universe = pool_pairs | source_pairs
    labels, coverage = load_judgments(judgment_rows, universe, model)
    if not {"dense_base", "dense_finetuned"} <= runs.keys():
        raise ValueError("Runs must include dense_base and dense_finetuned")
    validate_runs(runs, qids)
    variants = build_qrels(labels, source_qrels, qids)
    boot = bootstrap_for(dataset, n_resamples=n_resamples, seed=seed)
    primary, table, per_query_rows = {}, {}, []
    for policy, qrels in variants.items():
        scores = {name: {metric: np.array([fn(run[q], qrels[q]) for q in qids])
                         for metric, fn in METRICS.items()} for name, run in runs.items()}
        table[policy] = {name: {metric: float(v.mean()) for metric, v in values.items()}
                         for name, values in scores.items()}
        primary[policy] = {metric: paired(boot, scores["dense_finetuned"][metric], scores["dense_base"][metric])
                           for metric in METRICS}
        for i, d in enumerate(dataset):
            per_query_rows.append({
                "qid": d["dataset_id"], "source_chunk": d["source_chunk"], "policy": policy,
                "relevant_pairs": len(qrels[d["dataset_id"]]),
                "scores": {name: {metric: float(v[i]) for metric, v in values.items()}
                           for name, values in scores.items() if name in ("dense_base", "dense_finetuned")},
            })
    disagreements = [{"qid": q, "chunk_id": c, "relevance": labels[q, c]}
                     for q, c in sorted(source_pairs) if labels[q, c] != "si"]
    report = {
        "schema_version": 1, "judge_model": model,
        "design": "retrospective sensitivity analysis of automated relevance labels; no legal assessors",
        "primary_endpoint": "independent dense_finetuned minus dense_base NDCG@10",
        "queries": len(qids), "source_clusters": boot.n_clusters,
        "coverage": {**coverage, "pool_pairs": len(pool_pairs), "source_pairs": len(source_pairs),
                     "pool_source_overlap": len(pool_pairs & source_pairs)},
        "source_label_counts": {label: sum(labels[k] == label for k in source_pairs) for label in LABELS},
        "source_disagreements": disagreements,
        "policies": {
            "independent": "Relevant iff second judge says si, including every source pair",
            "source_anchored": "Independent positives plus all source qrels, regardless of second judge label",
            "partial_label": "parcial and no both count as nonrelevant",
            "zero_relevant_query": "Retain in all means and bootstrap samples; all metrics equal zero",
            "unjudged_retrieved_pair": "Treat as nonrelevant; report top-10 counts per configuration",
        },
        "metric_definitions": {
            "ndcg@10": "Binary gain, log2 discount, IDCG uses min(relevant, 10)",
            "recall@10": "Top-10 hits / number of relevant chunks",
            "recall_capped@10": "Top-10 hits / min(number of relevant chunks, 10); historical paper convention",
        },
        "bootstrap": {"resamples": n_resamples, "seed": seed, "confidence": 0.95,
                      "unit": "source_chunk", "method": "paired cluster percentile; query-weighted means"},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "qrels": {name: {"relevant_pairs": sum(map(len, qr.values())),
                         "zero_relevant_qids": [q for q in qids if not qr[q]]} for name, qr in variants.items()},
        "unjudged_top10_by_config": {name: sum((q, c) not in universe for q in qids for c in run[q][:10])
                                     for name, run in runs.items()},
        "primary_ft_minus_base": primary, "by_config": table,
    }
    return report, per_query_rows, variants


def markdown(report):
    coverage = report["coverage"]
    lines = ["# Independent second-judge sensitivity analysis", "",
             f"Judge: `{report['judge_model']}`. {report['queries']} queries; {report['source_clusters']} source clusters.", "",
             f"Coverage: {coverage['unique_pairs']}/{coverage['expected_pairs']} unique pairs; "
             f"{coverage['duplicate_rows']} consistent retry duplicates collapsed. "
             f"Source labels: {report['source_label_counts']}.", "",
             "Retrospective automated retrieval evaluation; this does not establish legal correctness, "
             "answer quality, or safety. The independent policy uses only `si` labels, including source chunks. "
             "The anchored policy additionally forces all source pairs to be relevant.", "",
             "| Policy | Relevant pairs | Queries with no positives | FT − base NDCG@10 (95% CI) |",
             "| --- | ---: | ---: | --- |"]
    for policy, qr in report["qrels"].items():
        lines.append(f"| {policy} | {qr['relevant_pairs']} | {len(qr['zero_relevant_qids'])} | "
                     f"{fmt_ci(report['primary_ft_minus_base'][policy]['ndcg@10'])} |")
    lines += ["", "| Configuration | Independent NDCG@10 | Anchored NDCG@10 | Unjudged top-10 pairs |",
              "| --- | ---: | ---: | ---: |"]
    for name, scores in report["by_config"]["independent"].items():
        lines.append(f"| {name} | {scores['ndcg@10']:.6f} | "
                     f"{report['by_config']['source_anchored'][name]['ndcg@10']:.6f} | "
                     f"{report['unjudged_top10_by_config'][name]} |")
    b = report["bootstrap"]
    lines += ["", f"Paired percentile bootstrap by source chunk: {b['resamples']} resamples, seed {b['seed']}. "
              "All queries are retained; queries with zero relevant chunks score zero. "
              "Unjudged retrieved pairs are treated as nonrelevant; pooling may still omit relevant documents.", "",
              "`analysis.json` includes ordinary and capped Recall@10, zero-positive query IDs, source disagreements, "
              "and input/code hashes. Capped recall uses the manuscript's denominator min(relevant, 10). "
              "The two policies are paired sensitivity analyses on the same queries, not independent replications. "
              "These results are not a new preregistered H1–H3 test."]
    return "\n".join(lines) + "\n"


def fingerprint(path):
    resolved = path.resolve()
    display = resolved.relative_to(EXP.parent) if resolved.is_relative_to(EXP.parent) else resolved
    return {"path": str(display), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--judgments", type=Path, help="Default: stored judgments for --model")
    ap.add_argument("--dataset", type=Path, default=TEST_DATASET)
    ap.add_argument("--source-qrels", type=Path, default=STRICT_QRELS)
    ap.add_argument("--pool-pairs", type=Path, default=EXP / "pooling/output/pool_pairs_test.jsonl")
    ap.add_argument("--runs-dir", type=Path, default=RESULTS / "runs")
    ap.add_argument("--output-dir", type=Path, help="Default: second_judge/<model>/independent (preserves original analysis)")
    ap.add_argument("--bootstrap-resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    directory = OUT / "second_judge" / args.model.replace("/", "__").replace(":", "_")
    judgments = args.judgments or directory / "judgments.jsonl"
    output = args.output_dir or directory / "independent"
    paths = {"dataset": args.dataset, "source_qrels": args.source_qrels,
             "pool_pairs": args.pool_pairs, "judgments": judgments,
             **{f"run:{name}": args.runs_dir / f"{name}.json" for name in CONFIGS},
             "analysis_code": Path(__file__), "data_code": Path(__file__).with_name("data.py"),
             "bootstrap_code": EXP.parent / "evaluation/bootstrap.py",
             "metrics_code": EXP.parent / "evaluation/metrics.py"}
    report, rows, variants = analyze(
        load_dataset(args.dataset), load_qrels(args.source_qrels), read_jsonl(args.pool_pairs), read_jsonl(judgments),
        {name: load_run(paths[f"run:{name}"]) for name in CONFIGS}, args.model,
        n_resamples=args.bootstrap_resamples, seed=args.seed,
    )
    report["provenance"] = {name: fingerprint(path) for name, path in paths.items()}
    output.mkdir(parents=True, exist_ok=True)
    for policy, qr in variants.items():
        write_jsonl(output / f"qrels_{policy}.jsonl",
                    ({"qid": q, "docid": c, "relevance": 1} for q in sorted(qr) for c in sorted(qr[q])))
    write_jsonl(output / "per_query.jsonl", rows)
    (output / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "analysis.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"Saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
