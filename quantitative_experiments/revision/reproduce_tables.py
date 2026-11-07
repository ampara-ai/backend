"""Regenera las Tablas 9, 10 y 12 del artículo desde los rankings publicados y las compara con los valores impresos.

Sin servicios: lee results/runs/<config>.json, los qrels estrictos y agrupados y el conjunto de prueba. Sale con código 1 si
alguna celda difiere del artículo en más de --tol (por defecto 0.0015, el redondeo a tres decimales), de modo que sirve como
prueba de regresión antes de reportar cualquier número nuevo.

Uso:
    uv run python -m quantitative_experiments.revision.reproduce_tables [--tol 0.0015]
"""

import argparse
import json
import sys
from pathlib import Path

from quantitative_experiments.revision.data import (
    CONFIGS, OUT, POOLED_QRELS, STRICT_QRELS, bootstrap_for, load_dataset, load_qrels, paired, per_query, released_run,
)

PAPER = json.loads((Path(__file__).parent / "paper_values.json").read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0015)
    ap.add_argument("--out", type=Path, default=OUT / "reproduce_tables.md")
    args = ap.parse_args()

    dataset = load_dataset()
    qids = [d["dataset_id"] for d in dataset]
    qrels = {"pooled": load_qrels(POOLED_QRELS), "strict": load_qrels(STRICT_QRELS)}
    runs = {c: released_run(c) for c in CONFIGS}
    boot = bootstrap_for(dataset)
    lines, worst, failures = [], 0.0, []

    for table, label in (("table9_pooled", "pooled"), ("table10_strict", "strict")):
        lines += [f"## {table} ({label} qrels)", "", "| Config | " + " | ".join(PAPER["columns"]) + " |",
                  "| --- |" + " --- |" * len(PAPER["columns"])]
        for c in CONFIGS:
            cells = []
            for col, expected in zip(PAPER["columns"], PAPER[table][c]):
                metric, k = col.split("@")
                got = float(per_query(runs[c], qrels[label], qids, metric, int(k)).mean())
                worst = max(worst, abs(got - expected))
                if abs(got - expected) > args.tol:
                    failures.append(f"{table} {c} {col}: {got:.4f} vs {expected:.3f}")
                cells.append(f"{got:.3f}")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## table12 (paired differences, bootstrap over source chunks)", "",
              "| Comparison | Metric | Strict (recomputed) | Pooled (recomputed) |", "| --- | --- | --- | --- |"]
    for row in PAPER["table12"]:
        metric, k = row["metric"].split("@")
        cells = []
        for label in ("strict", "pooled"):
            d = paired(boot, per_query(runs[row["a"]], qrels[label], qids, metric, int(k)),
                       per_query(runs[row["b"]], qrels[label], qids, metric, int(k)))
            cells.append(f"{d['estimate']:+.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]")
            if row[label] is not None:
                for got, expected in zip((d["estimate"], d["ci_low"], d["ci_high"]), row[label]):
                    worst = max(worst, abs(got - expected))
                    if abs(got - expected) > args.tol:
                        failures.append(f"table12 {row['a']} - {row['b']} {row['metric']} {label}: {got:.4f} vs {expected:.3f}")
        lines.append(f"| {row['a']} − {row['b']} | {row['metric']} | {cells[0]} | {cells[1]} |")

    verdict = "OK" if not failures else f"{len(failures)} cells differ by more than {args.tol}"
    header = [f"# Reproduction of Tables 9, 10 and 12", "", f"Largest absolute difference from the manuscript: {worst:.4f}. "
              f"Verdict: {verdict}.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"max |recomputed - paper| = {worst:.4f} -> {verdict} ({args.out})")
    for f in failures:
        print("  ", f)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
