"""Rejilla de pesos de la fusión híbrida (BM25 + denso) y del k de RRF, sobre los rankings de DEV de una variante.

Se ejecutó UNA vez con el embedding base para elegir los valores fijos HYBRID_FUSION de eval_ablation.py (peso del denso 2,0, k=5);
esos valores se aplican sin reajustar a las demás variantes. La rejilla completa está en hybrid_tuning_dev_base.csv.

Trabaja sobre rankings guardados por build_runs.py (bm25_only.json y dense_<variante>.json, profundidad 100, preguntas de dev),
sin MongoDB ni model service; la fusión usa rrf_fuse_ids, la misma función que la etapa híbrida de eval_ablation.py.
Criterio: NDCG@10 con los qrels ESTRICTOS de dev (chunk de origen), desempate por Recall@10 y Recall@100. No escribe parámetros:
solo imprime la mejor combinación y guarda la rejilla en CSV.

Uso:
    uv run python -m quantitative_experiments.ablation.tune_hybrid --dataset-path <eval_dev_dataset.jsonl> \
        --runs-dir <runs dev> --variant base --out-csv hybrid_tuning_dev_base.csv
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quantitative_experiments.ablation.eval_ablation import _score_docs_multi_k, rrf_fuse_ids  # noqa: E402

W_DENSE_GRID = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)  # peso de BM25 fijo en 1
RRF_K_GRID = (5, 10, 20, 30, 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=Path, required=True, help="eval_dev_dataset.jsonl")
    ap.add_argument("--runs-dir", type=Path, required=True, help="carpeta con bm25_only.json y dense_<variante>.json de dev")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    dataset = [json.loads(l) for l in args.dataset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    refs = {d["dataset_id"]: set(d["positive_chunk_ids"]) for d in dataset}
    bm25 = json.loads((args.runs_dir / "bm25_only.json").read_text(encoding="utf-8"))
    dense = json.loads((args.runs_dir / f"dense_{args.variant}.json").read_text(encoding="utf-8"))

    rows = []
    for w_dense, rrf_k in itertools.product(W_DENSE_GRID, RRF_K_GRID):
        per_q = [_score_docs_multi_k([{"chunk_id": c} for c in rrf_fuse_ids(bm25[q], dense[q], 1.0, w_dense, rrf_k)], ref)
                 for q, ref in refs.items()]
        df = pd.DataFrame(per_q)
        rows.append({"variant": args.variant, "w_bm25": 1.0, "w_dense": w_dense, "rrf_k": rrf_k,
                     "ndcg_at10": df["ndcg_at10"].mean(), "recall_at10": df["recall_at10"].mean(),
                     "recall_at100": df["recall_at100"].mean(), "mrr_at10": df["mrr_at10"].mean()})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(["ndcg_at10", "recall_at10", "recall_at100"], ascending=False).iloc[0]
    base = grid[(grid.w_dense == 1.0) & (grid.rrf_k == 60)].iloc[0]
    print(f"mejor: w_dense={best.w_dense} rrf_k={int(best.rrf_k)}  NDCG@10={best.ndcg_at10:.4f}  (1:1, k=60: {base.ndcg_at10:.4f})")
    grid.to_csv(args.out_csv, index=False)
    print("->", args.out_csv)


if __name__ == "__main__":
    main()
