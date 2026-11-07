"""Reanálisis del panel de expertos con los datos existentes (MAJOR #8): sin nuevas rondas.

Entradas (en --dir, por defecto results/revision/expert_panel/):
  ratings.csv   expert_id, item_id, criterion, rating (1-5), comment (opcional)
                criterion: relevancia_legal | claridad | fundamentacion_normativa
  items.csv     item_id, category (una de las 5 categorías temáticas), y opcionalmente query, answer, system_version, sampling
  experts.csv   expert_id, specialty, years, sector  (sin nombres)
  comments_coded.csv (opcional) la plantilla que escribe este script, con la columna error_type completada

Salidas: summary.md / summary.json con, por criterio, media, DE, rango de medias por experto, acuerdo (AC1 y AC2 ordinal de
Gwet, alfa de Krippendorff ordinal; con calificaciones concentradas en 4-5 el alfa se desploma y AC1/AC2 son los que
corresponde reportar), medias por categoría temática y para los especialistas en familia por separado; y
comments_to_code.csv, la plantilla para codificar los comentarios en la taxonomía de errores.

Uso:
    uv run python -m quantitative_experiments.revision.expert_panel [--dir <carpeta>]
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluation.agreement import gwet_ac, krippendorff_alpha
from quantitative_experiments.revision.data import OUT

SCALE = [1, 2, 3, 4, 5]
ERROR_TYPES = ["matiz_faltante", "articulo_equivocado", "norma_desactualizada", "omision_riesgosa", "derivacion_faltante",
               "sin_error", "otro"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matrix(rows: list[dict], experts: list[str]) -> list[list]:
    """Ítems × expertos para un criterio (None si falta la calificación)."""
    by_item = defaultdict(dict)
    for r in rows:
        by_item[r["item_id"]][r["expert_id"]] = int(r["rating"])
    return [[by_item[i].get(e) for e in experts] for i in sorted(by_item)]


def agreement(rows: list[dict], experts: list[str]) -> dict:
    m = matrix(rows, experts)
    return {"gwet_ac1": gwet_ac(m, SCALE, "identity"), "gwet_ac2_ordinal": gwet_ac(m, SCALE, "ordinal"),
            "krippendorff_alpha_ordinal": krippendorff_alpha(m, "ordinal", SCALE)}


def describe(rows: list[dict]) -> dict:
    x = np.array([int(r["rating"]) for r in rows], float)
    per_expert = defaultdict(list)
    for r in rows:
        per_expert[r["expert_id"]].append(int(r["rating"]))
    means = [float(np.mean(v)) for v in per_expert.values()]
    return {"n": int(x.size), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else None,
            "expert_mean_range": [min(means), max(means)], "distribution": dict(sorted(Counter(map(int, x)).items()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=OUT / "expert_panel")
    args = ap.parse_args()
    d = args.dir
    if not (d / "ratings.csv").exists():
        raise SystemExit(f"falta {d / 'ratings.csv'} (expert_id, item_id, criterion, rating, comment)")
    ratings = read_csv(d / "ratings.csv")
    items = {r["item_id"]: r for r in read_csv(d / "items.csv")} if (d / "items.csv").exists() else {}
    experts_meta = {r["expert_id"]: r for r in read_csv(d / "experts.csv")} if (d / "experts.csv").exists() else {}
    experts = sorted({r["expert_id"] for r in ratings})
    criteria = sorted({r["criterion"] for r in ratings})
    family = [e for e in experts if "famil" in experts_meta.get(e, {}).get("specialty", "").lower()]

    out = {"experts": len(experts), "items": len({r["item_id"] for r in ratings}), "ratings": len(ratings), "criteria": {}}
    for c in criteria + ["overall"]:
        rows = ratings if c == "overall" else [r for r in ratings if r["criterion"] == c]
        entry = {**describe(rows)}
        if c != "overall":
            entry["agreement"] = agreement(rows, experts)
        if family:
            entry["family_law_experts"] = describe([r for r in rows if r["expert_id"] in family])
        if items:
            by_cat = defaultdict(list)
            for r in rows:
                by_cat[items.get(r["item_id"], {}).get("category", "sin_categoria")].append(r)
            entry["by_category"] = {k: describe(v) for k, v in sorted(by_cat.items())}
        out["criteria"][c] = entry

    comments = [r for r in ratings if (r.get("comment") or "").strip()]
    with open(d / "comments_to_code.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["expert_id", "item_id", "criterion", "rating", "comment", "error_type"])
        for r in comments:
            w.writerow([r["expert_id"], r["item_id"], r["criterion"], r["rating"], r["comment"].strip(), ""])
    if (d / "comments_coded.csv").exists():
        coded = [r["error_type"].strip() for r in read_csv(d / "comments_coded.csv") if r.get("error_type", "").strip()]
        unknown = sorted(set(coded) - set(ERROR_TYPES))
        out["error_types"] = {"counts": dict(Counter(coded)), "unknown_labels": unknown}

    lines = [f"# Expert panel ({out['experts']} experts × {out['items']} answers, {out['ratings']} ratings)", "",
             "| Criterion | Mean | SD | Range of expert means | AC1 | AC2 (ordinal) | Krippendorff α (ordinal) |"
             + (" Family-law experts mean |" if family else ""),
             "| --- | --- | --- | --- | --- | --- | --- |" + (" --- |" if family else "")]
    for c, e in out["criteria"].items():
        ag = e.get("agreement", {})
        f = lambda k: f"{ag[k]:.3f}" if k in ag and ag[k] == ag[k] else "–"  # noqa: E731
        lines.append(f"| {c} | {e['mean']:.2f} | {e['sd']:.2f} | {e['expert_mean_range'][0]:.2f}–{e['expert_mean_range'][1]:.2f} "
                     f"| {f('gwet_ac1')} | {f('gwet_ac2_ordinal')} | {f('krippendorff_alpha_ordinal')} |"
                     + (f" {e['family_law_experts']['mean']:.2f} |" if family else ""))
    if items:
        cats = sorted({k for e in out["criteria"].values() for k in e.get("by_category", {})})
        lines += ["", "| Category | " + " | ".join(out["criteria"]) + " |", "| --- |" + " --- |" * len(out["criteria"])]
        for k in cats:
            lines.append(f"| {k} | " + " | ".join(f"{e['by_category'][k]['mean']:.2f}" if k in e.get("by_category", {}) else "–"
                                                  for e in out["criteria"].values()) + " |")
    if "error_types" in out:
        lines += ["", "Comment error types: " + ", ".join(f"{k} {v}" for k, v in sorted(out["error_types"]["counts"].items()))]
    lines += ["", f"{len(comments)} comments written to comments_to_code.csv; code error_type with: {', '.join(ERROR_TYPES)}."]
    (d / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
