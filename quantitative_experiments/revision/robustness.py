"""Análisis de robustez sobre los resultados publicados (sin servicios ni LLM): responde con datos a los comentarios
#27, #30, #32, #40, #42, #47 y #6 del revisor y fija la potencia del conjunto de prueba A.

Cada análisis escribe una fila en results/revision/robustness.md (y el detalle en robustness.json):
  memorization   corpus sin los chunks de entrenamiento (exacto para rankings densos y reordenados: el puntaje de cada chunk
                 no depende del resto del corpus)
  pooled_train   qrels agrupados que son chunks de entrenamiento (#42) y la comparación primaria sin ellos
  strict_pairs   preguntas con más de un chunk relevante estricto (#40)
  table2_vs_3    miembros de grupos de casi duplicados por partición (#30)
  negatives      falsos negativos con intervalo agrupado por pregunta (#32); lee judgments.jsonl del repo de datos
  lexical        ganancia por mitad de solapamiento léxico pregunta-chunk (#27)
  generator      ganancia por modelo que escribió la pregunta de prueba
  faithfulness   Faithfulness según si el chunk de origen se recuperó (#6); lee results/pipeline_final
  domain_filter  preguntas rechazadas por el filtro y su recall a nivel recuperador (#47)
  power          error estándar y diferencia mínima detectable (80 % de potencia, alfa 0.05 bilateral)
  source_type    ganancia por tipo de fuente del chunk de origen
  win_loss       preguntas cuyo chunk de origen solo uno de los dos modelos pone en el top-10

Uso:
    uv run python -m quantitative_experiments.revision.robustness
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    CONFIGS, GENERATOR_ALIASES, OUT, POOLED_QRELS, RESULTS, STRICT_QRELS, bootstrap_for, chunk_split, dataset_v2,
    filter_run, fmt_ci, load_dataset, load_qrels, paired, partition, per_query, read_jsonl, released_run,
)
from evaluation.bootstrap import ClusterBootstrap

FT, BASE = "dense_finetuned", "dense_base"


def mde(boot: ClusterBootstrap, diff: np.ndarray) -> tuple[float, float]:
    """Error estándar bootstrap de la diferencia media y diferencia mínima detectable ≈ (1.96 + 0.84) · SE."""
    se = float(boot.resampled_means(diff).std())
    return se, 2.8 * se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()

    ds = load_dataset()
    qids = [d["dataset_id"] for d in ds]
    by_id = {d["dataset_id"]: d for d in ds}
    S, P = load_qrels(STRICT_QRELS), load_qrels(POOLED_QRELS)
    runs = {c: released_run(c) for c in CONFIGS}
    boot = bootstrap_for(ds)
    nd = lambda run, qr=S, sub=qids: per_query(run, qr, sub, "ndcg", 10)  # noqa: E731
    out: dict[str, dict] = {}
    rows: list[tuple[str, str, str]] = []

    # memorization: el corpus sin chunks de entrenamiento
    mem = {}
    for metric in ("ndcg", "recall"):
        for corpus_kind, keep in (("full", lambda c: True), ("heldout", lambda c: partition(c) != "train")):
            r = {c: filter_run(runs[c], keep) for c in (BASE, FT, "dense_base_rerank100", "dense_finetuned_rerank100")}
            d = paired(boot, per_query(r[FT], S, qids, metric, 10), per_query(r[BASE], S, qids, metric, 10))
            mem[f"{metric}@10_{corpus_kind}"] = d
            if corpus_kind == "heldout":
                mem["min_depth_after_filter"] = min(len(v) for v in r[BASE].values())
    mem["top10_share_train_chunks"] = {c: float(np.mean([np.mean([partition(x) == "train" for x in runs[c][q][:10]]) for q in qids]))
                                       for c in (BASE, FT)}
    mem["corpus_share_train_chunks"] = sum(v == "train" for v in chunk_split()["assignments"].values()) / 893
    out["memorization"] = mem
    rows.append(("Memorization: corpus without training chunks", "strict NDCG@10 FT − base",
                 f"{fmt_ci(mem['ndcg@10_heldout'])} vs {fmt_ci(mem['ndcg@10_full'])} on the full corpus"))

    # #42 qrels agrupados de la partición de entrenamiento
    added = [(q, c) for q in qids for c in P[q] - S[q]]
    train_pos = {c for r in read_jsonl(dataset_v2() / "train_questions_final.jsonl")
                 for c in (r.get("positive_chunk_ids_final") or r["positive_chunk_ids"])}
    p_nt = {q: {c for c in P[q] if partition(c) != "train"} for q in qids}
    d_nt = paired(boot, nd(runs[FT], p_nt), nd(runs[BASE], p_nt))
    out["pooled_train"] = {"added": len(added), "by_partition": dict(Counter(partition(c) for _, c in added)),
                           "training_positives": sum(c in train_pos for _, c in added), "ft_minus_base_without_train": d_nt}
    rows.append(("#42 pooled positives from the training partition", "pooled NDCG@10 FT − base without them",
                 f"{out['pooled_train']['by_partition'].get('train', 0)} of {len(added)}; {fmt_ci(d_nt)}"))

    # #40 pares estrictos
    multi = {q: sorted(S[q]) for q in qids if len(S[q]) > 1}
    out["strict_pairs"] = {"pairs": sum(len(S[q]) for q in qids), "queries": len(qids), "multi": multi}
    rows.append(("#40 strict pairs", "questions with >1 strict relevant",
                 f"{out['strict_pairs']['pairs']} pairs / {len(qids)} questions; " + "; ".join(f"{q}: {v}" for q, v in multi.items())))

    # #30 Tabla 2 vs Tabla 3
    groups = chunk_split()["duplicate_groups"]
    extra = Counter(partition(c) for g in groups for c in g[1:])
    out["table2_vs_3"] = {"split_counts": chunk_split()["counts"], "non_representative_duplicates": dict(extra)}
    rows.append(("#30 Table 2 vs Table 3", "near-duplicate members beyond the first, by partition",
                 f"train {extra['train']}, dev {extra['dev']}, test {extra['test']}"))

    # #32 negativos juzgados, intervalo por pregunta
    neg = [r for r in read_jsonl(dataset_v2() / "judgments.jsonl") if r["role"] == "negative"]
    nb = ClusterBootstrap([r["dataset_id"] for r in neg])
    rates = {}
    for label, sel in (("si", ("si",)), ("parcial", ("parcial",)), ("si+parcial", ("si", "parcial"))):
        rates[label] = nb.mean([float(r["relevance"] in sel) for r in neg])
    out["negatives"] = {"judged": len(neg), "questions": nb.n_clusters, "rates": rates}
    pct = lambda d: f"{100 * d['estimate']:.1f}% [{100 * d['ci_low']:.1f}, {100 * d['ci_high']:.1f}]"  # noqa: E731
    rows.append(("#32 judged negatives, clustered by question", f"{len(neg)} pairs, {nb.n_clusters} questions",
                 f"yes {pct(rates['si'])}; partial {pct(rates['parcial'])}"))

    # #27 solapamiento léxico
    lo = {q: float(by_id[q]["generation"]["lexical_overlap"]) for q in qids}
    med = float(np.median(list(lo.values())))
    lex = {"quantiles": np.round(np.quantile(list(lo.values()), [0, .25, .5, .75, 1]), 3).tolist()}
    for half, sub in (("low", [q for q in qids if lo[q] <= med]), ("high", [q for q in qids if lo[q] > med])):
        lex[half] = {"n": len(sub), **{c: float(nd(runs[c], S, sub).mean()) for c in ("bm25_only", BASE, FT)}}
    out["lexical"] = lex
    rows.append(("#27 lexical overlap halves", "strict NDCG@10 BM25 / base / FT",
                 f"low (n={lex['low']['n']}): {lex['low']['bm25_only']:.3f} / {lex['low'][BASE]:.3f} / {lex['low'][FT]:.3f}; "
                 f"high (n={lex['high']['n']}): {lex['high']['bm25_only']:.3f} / {lex['high'][BASE]:.3f} / {lex['high'][FT]:.3f}"))

    # ganancia por generador de la pregunta
    gen = {q: GENERATOR_ALIASES.get(by_id[q]["generation"]["generator"], by_id[q]["generation"]["generator"]) for q in qids}
    out["generator"] = {}
    for g in sorted(set(gen.values())):
        sub = [q for q in qids if gen[q] == g]
        b = ClusterBootstrap([by_id[q]["source_chunk"] for q in sub])
        out["generator"][g] = {"n": len(sub), "ndcg@10": b.paired_difference(nd(runs[FT], S, sub), nd(runs[BASE], S, sub))}
    rows.append(("Test questions by generator", "strict NDCG@10 FT − base",
                 "; ".join(f"{g} (n={v['n']}): {fmt_ci(v['ndcg@10'])}" for g, v in out["generator"].items())))

    # #6 Faithfulness según si se recuperó el chunk de origen
    final = RESULTS / "pipeline_final"
    trace = {r["dataset_id"]: r for r in read_jsonl(final / "dense_finetuned.trace.jsonl")}
    with open(final / "ragas_luna" / "ragas_y_latencia_por_pregunta.csv", encoding="utf-8") as f:
        faith = {r["dataset_id"]: float(r["faithfulness"]) for r in csv.DictReader(f) if r["faithfulness"] not in ("", "nan")}
    fa = {}
    for label, qr in (("strict", S), ("pooled", P)):
        for hit in (True, False):
            xs = [v for q, v in faith.items() if bool(set(trace[q]["ranked"][:10]) & qr[q]) == hit]
            fa[f"{label}_{'hit' if hit else 'miss'}"] = {"n": len(xs), "mean": float(np.mean(xs)) if xs else None}
    out["faithfulness"] = fa
    rows.append(("#6 Faithfulness by retrieval outcome", "mean Faithfulness (n)",
                 f"strict hit {fa['strict_hit']['mean']:.2f} ({fa['strict_hit']['n']}), strict miss {fa['strict_miss']['mean']:.2f} "
                 f"({fa['strict_miss']['n']}), pooled miss {fa['pooled_miss']['mean']:.2f} ({fa['pooled_miss']['n']})"))

    # #47 filtro de dominio
    pipe = {c: json.loads((RESULTS / "pipeline" / f"{c}.json").read_text(encoding="utf-8")) for c in CONFIGS}
    rejected_sets = {frozenset(q for q in qids if not pipe[c].get(q)) for c in CONFIGS}
    rej = sorted(next(iter(rejected_sets)))
    acc = [q for q in qids if q not in set(rej)]
    out["domain_filter"] = {
        "rejected": len(rej), "same_set_all_configs": len(rejected_sets) == 1,
        "by_source_type": dict(Counter(by_id[q]["source_chunk"].split("_")[1] for q in rej)),
        "retriever_recall@10_rejected": float(per_query(runs[FT], S, rej, "recall", 10).mean()),
        "retriever_recall@10_accepted": float(per_query(runs[FT], S, acc, "recall", 10).mean()),
        "rejected_questions": [{"dataset_id": q, "query": by_id[q]["query"]} for q in rej],
    }
    df = out["domain_filter"]
    rows.append(("#47 domain filter rejections", "FT retriever strict Recall@10",
                 f"{df['rejected']} rejected (same set in all configs: {df['same_set_all_configs']}); "
                 f"{df['retriever_recall@10_rejected']:.2f} on rejected vs {df['retriever_recall@10_accepted']:.2f} on accepted"))

    # potencia
    pw = {}
    for a, b in ((FT, BASE), (FT, "hybrid_finetuned"), ("dense_finetuned_rerank100", "dense_base_rerank100")):
        se, m = mde(boot, nd(runs[a]) - nd(runs[b]))
        pw[f"{a} vs {b}"] = {"se": se, "mde": m}
    out["power"] = pw
    rows.append(("Power of Test A", "minimum detectable NDCG@10 difference (80% power)",
                 "; ".join(f"{k}: {v['mde']:.3f}" for k, v in pw.items())))

    # tipo de fuente y victorias/derrotas
    st = {}
    for t in sorted({by_id[q]["source_chunk"].split("_")[1] for q in qids}):
        sub = [q for q in qids if by_id[q]["source_chunk"].split("_")[1] == t]
        st[t] = {"n": len(sub), "base": float(nd(runs[BASE], S, sub).mean()), "ft": float(nd(runs[FT], S, sub).mean())}
    out["source_type"] = st
    rows.append(("Gain by source type", "strict NDCG@10 base → FT",
                 "; ".join(f"{t} (n={v['n']}): {v['base']:.3f} → {v['ft']:.3f}" for t, v in st.items())))
    hit_ft = per_query(runs[FT], S, qids, "recall", 10) > 0
    hit_b = per_query(runs[BASE], S, qids, "recall", 10) > 0
    out["win_loss"] = {"ft_only": int((hit_ft & ~hit_b).sum()), "base_only": int((hit_b & ~hit_ft).sum()),
                       "both": int((hit_ft & hit_b).sum()), "neither": int((~hit_ft & ~hit_b).sum())}
    wl = out["win_loss"]
    rows.append(("Paired wins and losses", "source chunk in top-10 for only one model",
                 f"FT only {wl['ft_only']}, base only {wl['base_only']}, both {wl['both']}, neither {wl['neither']}"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "robustness.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    md = ["# Robustness analyses on the released results", "",
          "95% bootstrap intervals over source chunks unless noted. Regenerate with "
          "`uv run python -m quantitative_experiments.revision.robustness`.", "",
          "| Analysis | Measure | Result |", "| --- | --- | --- |"] + [f"| {a} | {m} | {r} |" for a, m, r in rows]
    (args.out_dir / "robustness.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
