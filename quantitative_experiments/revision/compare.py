"""Compara configuraciones sobre un conjunto de preguntas: métricas por sistema y diferencias pareadas contra una referencia,
con bootstrap por chunk de origen. Un sistema con varias semillas (`nombre=run1.json,run2.json,...`) se promedia por pregunta y
su intervalo remuestrea también las semillas (bootstrap jerárquico: semillas con reemplazo y, dentro, clústeres).

Para la familia confirmatoria del registro (H1–H3) pasa --confirmatory: los valores p bootstrap bilaterales de esas
comparaciones se ajustan con Holm.

Uso:
    uv run python -m quantitative_experiments.revision.compare \
        --dataset results/revision/test_b/eval_test_b_dataset.jsonl --qrels results/revision/test_b/eval_test_b_qrels.jsonl \
        --system base=results/revision/runs/eval_test_b_dataset/dense_base_fp16.json \
        --system finetuned=results/revision/runs/eval_test_b_dataset/ft_seed42.json,results/revision/runs/eval_test_b_dataset/ft_seed1.json \
        --reference base --metric ndcg@10 --out results/revision/compare_test_b.md
    # Corridas publicadas: --system dense_base=released:dense_base
"""

import argparse
import json
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    METRICS, OUT, STRICT_QRELS, TEST_DATASET, load_dataset, load_qrels, load_run, per_query, released_run,
)


def load_system(spec: str) -> list[dict[str, list[str]]]:
    return [released_run(p.split(":", 1)[1]) if p.startswith("released:") else load_run(Path(p)) for p in spec.split(",")]


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    order = sorted(pvalues, key=pvalues.get)
    m, adjusted, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        adjusted[k] = running
    return adjusted


class HierarchicalBootstrap:
    """Remuestrea clústeres (chunks de origen) y, para sistemas con varias semillas, también las semillas."""

    def __init__(self, clusters: list[str], n: int = 10_000, seed: int = 0) -> None:
        labels, self.inv = np.unique(np.asarray(clusters), return_inverse=True)
        self.k = len(labels)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.k, size=(n, self.k))
        self.mult = np.zeros((n, self.k))
        np.add.at(self.mult, (np.arange(n)[:, None], draws), 1)
        self.rng, self.n = np.random.default_rng(seed + 1), n
        self.sizes = np.bincount(self.inv, minlength=self.k)

    def replicate_means(self, per_seed: np.ndarray) -> np.ndarray:
        """per_seed: [semillas, preguntas] -> media por réplica (semillas remuestreadas, luego clústeres)."""
        s = per_seed.shape[0]
        seed_idx = self.rng.integers(0, s, size=(self.n, s)) if s > 1 else np.zeros((self.n, 1), dtype=int)
        out = np.empty(self.n)
        for b in range(self.n):  # media por pregunta sobre las semillas elegidas, luego agregado por clúster
            v = per_seed[seed_idx[b]].mean(axis=0)
            out[b] = (self.mult[b] @ np.bincount(self.inv, weights=v, minlength=self.k)) / (self.mult[b] @ self.sizes)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=TEST_DATASET)
    ap.add_argument("--qrels", type=Path, default=STRICT_QRELS)
    ap.add_argument("--system", action="append", required=True, help="nombre=run.json[,run2.json...] o nombre=released:<config>")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--metric", action="append", default=None, help="p. ej. ndcg@10 (repetible; por defecto ndcg@10 y recall@10)")
    ap.add_argument("--confirmatory", action="append", default=[], help="nombre del sistema cuya diferencia con la referencia es confirmatoria")
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=OUT / "compare.md")
    args = ap.parse_args()

    metrics = args.metric or ["ndcg@10", "recall@10"]
    ds = load_dataset(args.dataset)
    qids = [d["dataset_id"] for d in ds]
    qrels = load_qrels(args.qrels)
    systems = {s.split("=", 1)[0]: load_system(s.split("=", 1)[1]) for s in args.system}
    missing = {n: sum(q not in r for r in runs for q in qids) for n, runs in systems.items()}
    if any(missing.values()):
        raise SystemExit(f"preguntas sin ranking: {missing}")
    boot = HierarchicalBootstrap([d["source_chunk"] for d in ds], args.resamples)

    rows, pvals, result = [], {}, {}
    for m in metrics:
        name, k = m.split("@")
        assert name in METRICS, f"métrica desconocida: {m}"
        vals = {n: np.stack([per_query(r, qrels, qids, name, int(k)) for r in runs]) for n, runs in systems.items()}
        ref = vals[args.reference]
        for n, v in vals.items():
            entry = {"seeds": v.shape[0], "mean": float(v.mean()), "seed_sd": float(v.mean(axis=1).std(ddof=1)) if v.shape[0] > 1 else None}
            if n != args.reference:
                # diferencia pareada por pregunta; semillas de cada sistema remuestreadas por separado
                reps = boot.replicate_means(v) - boot.replicate_means(ref)
                lo, hi = np.quantile(reps, [0.025, 0.975])
                p = float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))
                entry.update({"diff": float(v.mean() - ref.mean()), "ci": [float(lo), float(hi)], "p_boot": p})
                if n in args.confirmatory:
                    pvals[f"{n}|{m}"] = p
            result.setdefault(m, {})[n] = entry
    adjusted = holm(pvals) if pvals else {}

    lines = [f"# Comparison on {args.dataset.name} ({len(qids)} questions, qrels {args.qrels.name})", "",
             f"Reference: {args.reference}. Intervals: 95% bootstrap over source chunks"
             + (", seeds resampled within each replicate." if any(len(r) > 1 for r in systems.values()) else "."), ""]
    for m, entries in result.items():
        lines += [f"## {m}", "", "| System | Seeds | Mean | Seed SD | Diff vs reference | 95% CI | p (boot) | p (Holm) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for n, e in entries.items():
            sd = f"{e['seed_sd']:.3f}" if e["seed_sd"] is not None else "–"
            if "diff" in e:
                holm_p = adjusted.get(f"{n}|{m}")
                lines.append(f"| {n} | {e['seeds']} | {e['mean']:.3f} | {sd} | {e['diff']:+.3f} | [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}] "
                             f"| {e['p_boot']:.4f} | {holm_p:.4f} |" if holm_p is not None else
                             f"| {n} | {e['seeds']} | {e['mean']:.3f} | {sd} | {e['diff']:+.3f} | [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}] "
                             f"| {e['p_boot']:.4f} | – |")
            else:
                lines.append(f"| {n} (reference) | {e['seeds']} | {e['mean']:.3f} | {sd} | – | – | – | – |")
        lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps({"results": result, "holm": adjusted}, indent=1), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
