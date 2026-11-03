"""Bootstrap pareado: la configuración de referencia contra cada una de las demás, por métrica.

Lee <results-dir>/resultados_ablation_<config>.csv (uno por configuración, mismas
dataset_id) y remuestrea preguntas con reemplazo (--resamples, --seed fija) para obtener el IC 95 % de la
diferencia de medias y un p-valor bilateral aproximado.

--cluster-by agrupa las preguntas antes de remuestrear (p. ej. por chunk positivo): varias preguntas sobre el
mismo chunk no son independientes, y remuestrear por pregunta subestimaría la incertidumbre.

Uso:
    uv run python -m quantitative_experiments.ablation.paired_bootstrap \
        --results-dir quantitative_experiments/results/scores/retriever_pooled \
        --dataset-path quantitative_experiments/test_set/eval_test_dataset.jsonl \
        --cluster-by positive_chunk
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from quantitative_experiments.ablation.eval_ablation import config_arg, parse_config  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
class _Labels(dict):
    """Etiqueta legible por configuración; cualquier nombre válido de eval_ablation.py es aceptado (se etiqueta con su nombre)."""
    def __missing__(self, key):
        parse_config(key)
        return key


LABELS = _Labels({"bm25_only": "BM25", "dense_base": "Dense base", "dense_finetuned": "Dense fine-tuned", "hybrid_finetuned": "Hybrid fine-tuned"})
METRICS = [f"{m}_at{k}" for k in (10, 100) for m in ("recall", "mrr", "ndcg", "map")]


def load(config: str, results_dir: Path) -> pd.DataFrame:
    return pd.read_csv(results_dir / f"resultados_ablation_{config}.csv").set_index("dataset_id").sort_index()


def cluster_labels(dataset_path: Path, ids: list[str], mode: str) -> np.ndarray:
    """Etiqueta de cluster por pregunta (en el orden de ids)."""
    if mode == "none":
        return np.arange(len(ids))
    rows = {}
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["dataset_id"]] = r["positive_chunk_ids"][0] if mode == "positive_chunk" else r["case_id"]
    return pd.factorize(pd.Series([rows[i] for i in ids]))[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--reference", default="dense_finetuned", type=config_arg)
    ap.add_argument("--others", nargs="+", type=config_arg,
                    default=["dense_base", "bm25_only", "hybrid_finetuned"])
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster-by", choices=["none", "positive_chunk", "case_id"], default="none")
    ap.add_argument("--dataset-path", type=Path, help="Necesario si --cluster-by != none")
    args = ap.parse_args()

    REFERENCE = (LABELS[args.reference], args.reference)
    CONFIGS = {LABELS[c]: c for c in args.others}
    ref = load(REFERENCE[1], args.results_dir)
    ids = list(ref.index)
    labels = cluster_labels(args.dataset_path, ids, args.cluster_by)
    members = [np.where(labels == c)[0] for c in range(labels.max() + 1)]
    rng = np.random.default_rng(args.seed)
    draws = rng.integers(0, len(members), size=(args.resamples, len(members)))
    idx_per_draw = [np.concatenate([members[c] for c in d]) for d in draws]

    print(f"n queries = {len(ids)}; clusters = {len(members)} ({args.cluster_by}); "
          f"resamples = {args.resamples}; seed = {args.seed}; reference = {REFERENCE[0]}")
    for metric in METRICS:
        print(f"\n== {metric}: {REFERENCE[0]} mean = {ref[metric].mean():.3f}")
        for label, cfg in CONFIGS.items():
            other = load(cfg, args.results_dir)
            assert list(other.index) == ids, f"{cfg}: dataset_id distintos a la referencia"
            diff = (ref[metric] - other[metric]).to_numpy()
            boots = np.array([diff[i].mean() for i in idx_per_draw])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()) + 1 / args.resamples)
            print(f"   vs {label:<16} mean={other[metric].mean():.3f}  diff={diff.mean():+.3f}  "
                  f"95% CI [{lo:+.3f}, {hi:+.3f}]  p~{p:.4f}")
    print("\nNota: 8 métricas x 4 comparaciones; fijar una métrica primaria (recall@10) o corregir por multiplicidad.")


if __name__ == "__main__":
    main()
