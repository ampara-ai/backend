"""Calcula las métricas de recuperación (Recall/MRR/NDCG/MAP @10 y @100) a partir de los rankings guardados por
build_runs.py, sin tocar MongoDB ni el model service. Permite re-puntuar las mismas corridas con distintos qrels
(estrictos u ampliados por pooling) sin reindexar.

Escribe, en --out-dir, el mismo formato que eval_ablation.py: resultados_ablation_<config>.csv por configuración y
comparacion_ablation.csv (consumido por build_final_table.py y paired_bootstrap.py).

Uso:
    uv run python -m quantitative_experiments.ablation.score_runs \
        --dataset-path quantitative_experiments/test_set/eval_test_dataset.jsonl \
        --runs-dir quantitative_experiments/results/runs \
        --qrels quantitative_experiments/pooling/output/eval_test_qrels_pooled.jsonl \
        --out-dir quantitative_experiments/results/scores/retriever_pooled
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantitative_experiments.ablation.eval_ablation import (  # noqa: E402
    AT_K_METRIC_COLS,
    CONFIG_NAMES,
    CSV_COLS,
    _score_docs_multi_k,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--qrels", type=Path, default=None,
                    help="JSONL {qid, docid, relevance}; si se omite, se usan los positive_chunk_ids del dataset")
    ap.add_argument("--configs", nargs="+", default=CONFIG_NAMES)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    dataset = read_jsonl(args.dataset_path)
    if args.qrels:
        qrels: dict[str, set[str]] = {}
        for r in read_jsonl(args.qrels):
            if r.get("relevance", 1) > 0:
                qrels.setdefault(r["qid"], set()).add(r["docid"])
    else:
        qrels = {d["dataset_id"]: set(d["positive_chunk_ids"]) for d in dataset}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config in args.configs:
        run_path = args.runs_dir / f"{config}.json"
        if not run_path.exists():
            print(f"[omitida] {config}: no existe {run_path}")
            continue
        run = json.loads(run_path.read_text(encoding="utf-8"))
        rows = []
        for item in dataset:
            qid = item["dataset_id"]
            ref = qrels[qid]
            docs = [{"chunk_id": c} for c in run[qid]]
            rows.append({"dataset_id": qid, "query": item["query"], "n_positives": len(ref),
                         **_score_docs_multi_k(docs, ref), "topic": item.get("topic"),
                         "barrier": item.get("barrier"), "model_variant": config})
        df = pd.DataFrame(rows).reindex(columns=CSV_COLS)
        df.to_csv(args.out_dir / f"resultados_ablation_{config}.csv", index=False)
        summaries.append({"config": config, "n_queries": len(df),
                          **{f"mean_{c}": df[c].mean() for c in AT_K_METRIC_COLS}})
        print(f"{config}: " + "  ".join(f"{c}={df[c].mean():.3f}" for c in AT_K_METRIC_COLS if "10" in c and "100" not in c))
    pd.DataFrame(summaries).to_csv(args.out_dir / "comparacion_ablation.csv", index=False)
    print("->", args.out_dir)


if __name__ == "__main__":
    main()
