"""Collect exactly the three prespecified contrasts and adjust their raw p-values.

Offline only. Requires completed H1, H2 Test-B and H3 outputs; no partial family.
python -m quantitative_experiments.revision.confirmatory --h1 .../comparison.json \
    --h2 .../test_b_primary.json --h3 .../answers/summary.json
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from quantitative_experiments.revision.compare import holm
from quantitative_experiments.revision.data import OUT, read_jsonl


def frozen_test_b():
    dataset = OUT / "test_b/eval_test_b_dataset.jsonl"
    qrels = OUT / "test_b/eval_test_b_qrels.jsonl"
    return {"dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "qrels_sha256": hashlib.sha256(qrels.read_bytes()).hexdigest(),
            "question_ids": [d["dataset_id"] for d in read_jsonl(dataset)]}


def aggregate(h1, h2, h3):
    frozen = frozen_test_b()
    expected_n = len(frozen["question_ids"])
    if h1.get("queries") != expected_n or h2.get("provenance", {}).get("questions") != expected_n or h3.get("n_questions") != expected_n:
        raise ValueError("Every primary result must include the full frozen Test B population")
    for name, result in (("H1", h1), ("H2", h2.get("provenance", {}))):
        if any(result.get(key) != frozen[key] for key in ("dataset_sha256", "qrels_sha256")):
            raise ValueError(f"{name} must use the frozen Test B dataset and strict qrels")
    if h3.get("dataset_sha256") != frozen["dataset_sha256"] or sorted(h3.get("question_ids", [])) != sorted(frozen["question_ids"]):
        raise ValueError("H3 must use all questions in the frozen Test B dataset")
    if h1.get("hypothesis") != "H1" or h1.get("metric") != "strict NDCG@10":
        raise ValueError("H1 must be the portable runner's strict NDCG@10 comparison")
    if h2.get("provenance", {}).get("reference") != "naive":
        raise ValueError("H2 must compare full_matched against reference naive")
    if h3.get("reference") != "base":
        raise ValueError("H3 must use base as its reference")
    n = h3.get("n_questions", 0)
    if not isinstance(n, int) or n < 10:
        raise ValueError("H3 must include at least 10 matched questions")
    for condition in ("base", "finetuned"):
        coverage = h3.get("coverage", {}).get(condition, {}).get("faithfulness", {})
        if coverage.get("scored") != n or coverage.get("missing") != 0 or coverage.get("not_applicable") != 0:
            raise ValueError("H3 primary Faithfulness scores must be complete for base and finetuned")
    try:
        second = h2["results"]["ndcg@10"]["full_matched"]
        if second.get("seeds") != 3 or h2["results"]["ndcg@10"]["naive"].get("seeds") != 3:
            raise ValueError("H2 requires three seeds for each training condition")
        third = h3["paired"]["finetuned|faithfulness"]
        if third["n"] != n:
            raise ValueError("H3 paired population is incomplete")
        values = {
            "H1": (h1["raw_bootstrap_tail_p"], h1["difference"]["estimate"]),
            "H2": (second["p_boot"], second["diff"]),
            "H3": (third["p_boot"], third["estimate"]),
        }
    except KeyError as exc:
        raise ValueError(f"Missing primary contrast: {exc}") from exc
    for name, (p, effect) in values.items():
        if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f"Invalid raw p-value for {name}")
        if not isinstance(effect, (int, float)) or not math.isfinite(effect):
            raise ValueError(f"Invalid effect for {name}")
    adjusted = holm({name: p for name, (p, _) in values.items()})
    return {"family": ["H1", "H2", "H3"], "alpha": 0.05,
            "results": {name: {"effect": effect, "p_raw": p, "p_holm": adjusted[name],
                               "positive_effect_significant": effect > 0 and adjusted[name] < 0.05}
                        for name, (p, effect) in values.items()},
            "note": "Raw values are bootstrap-tail estimates. A reported zero is Monte Carlo resolution, not an exact p=0."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("h1", "h2", "h3"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT / "confirmatory.json")
    args = parser.parse_args(argv)
    paths = {name: getattr(args, name) for name in ("h1", "h2", "h3")}
    if args.out.resolve() in {p.resolve() for p in paths.values()}:
        raise ValueError("Output must not overwrite an input result")
    result = aggregate(**{name: json.loads(p.read_text(encoding="utf-8")) for name, p in paths.items()})
    result["inputs"] = {name: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for name, p in paths.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
