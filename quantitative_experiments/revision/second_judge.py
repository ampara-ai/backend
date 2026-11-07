"""Segundo juez de relevancia (MAJOR #4): vuelve a juzgar los pares del pool de test con un modelo de OTRA familia y mide
cuánto dependen los resultados del juez.

Pasos:
  judge    Juzga con --model los 11,763 pares del pool (pool_pairs_test.jsonl) y, además, los 200 pares estrictos (pregunta,
           chunk de origen), con el MISMO prompt y la misma presentación que el pooling original (una pregunta por petición,
           chunks en el mismo orden determinista). --chunks-per-task parte las preguntas con muchos chunks en varias peticiones.
  analyze  Acuerdo entre jueces (kappa de Cohen con 3 clases y binario "si" vs resto, matriz de confusión) y la comparación
           primaria (FT − base, NDCG@10 y Recall@10) con qrels agrupados según el juez 1 (los publicados), el juez 2, la
           intersección y la unión de sus "si". También la tabla de NDCG@10 de las 9 configuraciones con cada variante.

Uso:
    uv run python -m quantitative_experiments.revision.second_judge judge --model <id OpenRouter> [--dry-run]
    uv run python -m quantitative_experiments.revision.second_judge analyze --model <id OpenRouter>
"""

import argparse
import asyncio
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    CONFIGS, EXP, OUT, STRICT_QRELS, bootstrap_for, corpus, dataset_v2, fmt_ci, load_dataset, load_qrels, paired,
    per_query, read_jsonl, released_run, write_jsonl,
)
from quantitative_experiments.revision.llm import LLM

POOL = EXP / "pooling" / "output"
JUDGE_PROMPT = EXP / "pooling" / "prompts" / "judge_system.txt"
TEXT_CHARS, SEED = 8000, 520
LABELS = ("si", "parcial", "no")


def stable_unit(*parts: object) -> float:
    """El mismo orden de chunks que pooling/pool_eval.py."""
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


def slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def cohen_kappa(a: list[str], b: list[str], labels=LABELS) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[label] * cb[label] for label in labels) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


async def judge(args) -> None:
    C = corpus()
    question = {d["dataset_id"]: d["query"] for d in load_dataset()}
    pairs: dict[str, set[str]] = {}
    for r in read_jsonl(POOL / "pool_pairs_test.jsonl"):
        pairs.setdefault(r["qid"], set()).add(r["chunk_id"])
    role = {}
    if args.include_origin:
        for q, rel in load_qrels(STRICT_QRELS).items():
            for c in rel:
                pairs.setdefault(q, set()).add(c)
                role[(q, c)] = "origin"
    system = JUDGE_PROMPT.read_text(encoding="utf-8")
    llm = LLM(args.model, OUT / "second_judge" / slug(args.model) / "cache.jsonl", temperature=0,
              concurrency=args.concurrency, dry_run=args.dry_run, reasoning_effort=args.effort, max_tokens=args.max_tokens)

    tasks = []
    for q in sorted(pairs):
        ordered = sorted(pairs[q], key=lambda c: stable_unit(SEED, "pool", q, c))
        step = args.chunks_per_task or len(ordered)
        tasks += [(q, ordered[i:i + step]) for i in range(0, len(ordered), step)]

    async def one(q, chunks):
        payload = [{"dataset_id": q, "question": question[q],
                    "chunks": [{"chunk_id": c, "title": C[c].get("title"), "text": C[c]["text"][:TEXT_CHARS]} for c in chunks]}]

        def check(resp):
            got = {j["chunk_id"] for j in resp.get("judgments") or [] if j.get("relevance") in LABELS}
            if not set(chunks) <= got:
                raise ValueError(f"{len(set(chunks) - got)} pares sin veredicto válido")

        resp = await llm.json(system, "Juzga estos pares:\n\n" + json.dumps(payload, ensure_ascii=False, indent=1), validate=check)
        return q, resp

    results = await asyncio.gather(*(one(q, cs) for q, cs in tasks))
    print(llm.report())
    if args.dry_run:
        print(f"pares: {sum(len(v) for v in pairs.values())} en {len(tasks)} peticiones")
        return
    rows = []
    for q, resp in results:
        for j in resp["judgments"]:
            if j["chunk_id"] in pairs[q]:
                rows.append({"qid": q, "chunk_id": j["chunk_id"], "relevance": j["relevance"], "rationale": j.get("rationale"),
                             "judge_model": args.model, "role": role.get((q, j["chunk_id"]), "pool")})
    write_jsonl(OUT / "second_judge" / slug(args.model) / "judgments.jsonl", rows)
    print(f"{len(rows)} veredictos -> {OUT / 'second_judge' / slug(args.model)}")


def analyze(args) -> None:
    d2 = OUT / "second_judge" / slug(args.model)
    j2 = {(r["qid"], r["chunk_id"]): r["relevance"] for r in read_jsonl(d2 / "judgments.jsonl")}
    j1 = {(r["qid"], r["chunk_id"]): r["relevance"] for r in read_jsonl(POOL / "pool_judgments_test.jsonl")}
    # veredictos del juez 1 sobre los chunks de origen: los del armado del dataset (rol "origin")
    j1.update({(r["dataset_id"], r["chunk_id"]): r["relevance"] for r in read_jsonl(dataset_v2() / "judgments.jsonl")
               if r["role"] == "origin" and r["dataset_id"].startswith("test::")})
    common = sorted(set(j1) & set(j2))
    a, b = [j1[k] for k in common], [j2[k] for k in common]
    agreement = {
        "pairs": len(common), "percent_agreement": float(np.mean([x == y for x, y in zip(a, b)])),
        "kappa_3class": cohen_kappa(a, b),
        "kappa_binary_si": cohen_kappa(["si" if x == "si" else "no" for x in a], ["si" if x == "si" else "no" for x in b], ("si", "no")),
        "confusion_j1_rows_j2_cols": {x: {y: sum(1 for p, q in zip(a, b) if p == x and q == y) for y in LABELS} for x in LABELS},
    }

    ds = load_dataset()
    qids = [d["dataset_id"] for d in ds]
    S = load_qrels(STRICT_QRELS)
    pool_keys = [k for k in common if k[1] not in S.get(k[0], set())]
    variants = {}
    for name, rule in (("judge1", lambda k: j1[k] == "si"), ("judge2", lambda k: j2[k] == "si"),
                       ("intersection", lambda k: j1[k] == "si" and j2[k] == "si"),
                       ("union", lambda k: j1[k] == "si" or j2[k] == "si")):
        qr = {q: set(S[q]) for q in qids}
        for k in pool_keys:
            if rule(k):
                qr[k[0]].add(k[1])
        variants[name] = qr
    runs = {c: released_run(c) for c in CONFIGS}
    boot = bootstrap_for(ds)
    primary, table = {}, {}
    for name, qr in variants.items():
        primary[name] = {m: paired(boot, per_query(runs["dense_finetuned"], qr, qids, m, 10),
                                   per_query(runs["dense_base"], qr, qids, m, 10)) for m in ("ndcg", "recall")}
        table[name] = {c: float(per_query(runs[c], qr, qids, "ndcg", 10).mean()) for c in CONFIGS}
    out = {"judge2": args.model, "agreement": agreement, "relevant_pairs": {n: sum(len(v) for v in qr.values()) for n, qr in variants.items()},
           "primary_ft_minus_base": primary, "ndcg@10_by_config": table}
    (d2 / "analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Second judge: {args.model}", "",
          f"Agreement on {agreement['pairs']} pairs: {100 * agreement['percent_agreement']:.1f}%; "
          f"Cohen's kappa {agreement['kappa_3class']:.3f} (3 classes), {agreement['kappa_binary_si']:.3f} (si vs rest).", "",
          "| Qrels | Relevant pairs | FT − base NDCG@10 | FT − base Recall@10 |", "| --- | --- | --- | --- |"]
    md += [f"| {n} | {out['relevant_pairs'][n]} | {fmt_ci(p['ndcg'])} | {fmt_ci(p['recall'])} |" for n, p in primary.items()]
    md += ["", "| Config | " + " | ".join(variants) + " |", "| --- |" + " --- |" * len(variants)]
    md += [f"| {c} | " + " | ".join(f"{table[n][c]:.3f}" for n in variants) + " |" for c in CONFIGS]
    (d2 / "analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("judge")
    j.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    j.add_argument("--concurrency", type=int, default=4)
    j.add_argument("--chunks-per-task", type=int, default=0, help="0 = todos los chunks de la pregunta en una petición (como el pooling)")
    j.add_argument("--max-tokens", type=int, default=16000)
    j.add_argument("--no-origin", dest="include_origin", action="store_false", help="no juzgar los 200 pares estrictos")
    j.add_argument("--dry-run", action="store_true")
    an = sub.add_parser("analyze")
    for p in (j, an):
        p.add_argument("--model", required=True, help="juez 2: open-weight y de otra familia que GPT, Claude y gpt-oss")
    args = ap.parse_args()
    if args.cmd == "judge":
        asyncio.run(judge(args))
    else:
        analyze(args)


if __name__ == "__main__":
    main()
