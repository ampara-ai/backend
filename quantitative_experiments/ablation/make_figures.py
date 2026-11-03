"""Figura de la ablación: Recall@10, NDCG@10 y MRR@10 de las nueve configuraciones (mismo estilo que la figura del paper anterior).

Lee los puntajes que escribe score_runs.py (comparacion_ablation.csv) y guarda cada figura en PNG y en PDF vectorial.

Uso:
    uv run python -m quantitative_experiments.ablation.make_figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
CONFIGS = [  # (config, etiqueta en la figura)
    ("bm25_only", "BM25"),
    ("dense_base", "Base\ndense"),
    ("dense_finetuned", "Fine-tuned\ndense"),
    ("hybrid_base", "Base\nhybrid"),
    ("hybrid_finetuned", "Fine-tuned\nhybrid"),
    ("dense_base_rerank100", "Base dense\n+ reranker"),
    ("dense_finetuned_rerank100", "Fine-tuned dense\n+ reranker"),
    ("hybrid_base_rerank100", "Base hybrid\n+ reranker"),
    ("hybrid_finetuned_rerank100", "Fine-tuned hybrid\n+ reranker"),
]
METRICS = [("mean_recall_at10", "Recall@10", "#4C72B0"), ("mean_ndcg_at10", "NDCG@10", "#DD8452"), ("mean_mrr_at10", "MRR@10", "#55A868")]
HIGHLIGHT = "dense_finetuned"   # la configuración desplegada
N_FIRST_STAGE = 5               # las cinco primeras no usan reranker


def draw(scores: pd.DataFrame, title: str, out_stem: Path) -> None:
    n = len(CONFIGS)
    width = 0.26
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    for j, (col, label, color) in enumerate(METRICS):
        vals = [scores.loc[c, col] for c, _ in CONFIGS]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=label, color=color, edgecolor="white", linewidth=0.6)
        for b, (c, _), v in zip(bars, CONFIGS, vals):
            if c == HIGHLIGHT:
                b.set_edgecolor("black")
                b.set_linewidth(1.8)
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.axvline(N_FIRST_STAGE - 0.5, color="gray", linestyle="--", linewidth=0.9)
    ax.text((N_FIRST_STAGE - 1) / 2, 1.07, "First stage only", ha="center", fontsize=9, color="dimgray")
    ax.text(N_FIRST_STAGE + (n - N_FIRST_STAGE - 1) / 2, 1.07, "With second-stage reranker (top 100 candidates)", ha="center", fontsize=9, color="dimgray")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in CONFIGS], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylabel("Metric value")
    ax.set_title(title, fontsize=11, pad=22)
    ax.yaxis.grid(True, linestyle=":", color="lightgray")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0.0, 0.98))
    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("->", out_stem.with_suffix(".png"), "y .pdf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", type=Path, default=EXP / "results" / "scores")
    ap.add_argument("--out-dir", type=Path, default=EXP / "results" / "figures")
    args = ap.parse_args()
    for name, title, stem in (
        ("retriever_pooled", "Comparison of retrieval metrics by configuration (pooled relevance judgments)", "retrieval_metrics_by_configuration"),
        ("retriever_strict", "Comparison of retrieval metrics by configuration (strict relevance judgments)", "retrieval_metrics_by_configuration_strict"),
    ):
        scores = pd.read_csv(args.scores_dir / name / "comparacion_ablation.csv").set_index("config")
        draw(scores, title, args.out_dir / stem)


if __name__ == "__main__":
    main()
