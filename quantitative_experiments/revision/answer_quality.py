"""Calidad de respuesta con el generador fijo (MAJOR #6, H3 del registro): ¿mejor recuperación da mejores respuestas?

Condiciones (mismo generador, temperatura, prompt y preguntas; solo cambia el contexto):
  none                  sin recuperación (prompt prompts/no_retrieval_system.txt: responde con su conocimiento)
  base                  top-k de released:dense_base
  finetuned             top-k de released:dense_finetuned
  finetuned_rerank      top-k de released:dense_finetuned_rerank100 (o de una corrida de rerank_local.py)
  oracle                solo el chunk de origen y sus casi duplicados (techo: separa error de recuperación y de generación)
  cualquier otra        --run <ruta a corrida JSON> o released:<config>

Pasos:
  generate     Escribe <condición>.trace.jsonl (mismo formato que ablation/full_pipeline.py, legible por ragas_final.py) y
               <condición>.provenance.json (modelo, temperatura, proveedor, archivo y SHA-256 del prompt, commit).
  score        RAGAS Faithfulness con los contextos recibidos, Faithfulness contra el chunk de origen ("gold_support",
               medible también sin recuperación) y Answer Relevancy; diferencias pareadas contra --reference con bootstrap
               por chunk de origen. Reanudable.
  citations    Exactitud de las citas normativas (citations.py) por condición; `released_final` puntúa la corrida publicada.
  code-errors  Un LLM clasifica las respuestas con Faithfulness < --threshold en causas (afirmación sin respaldo, artículo
               inventado o equivocado, texto de derivación o seguridad, consejo general, paráfrasis que el juez no reconoce).

Preguntas: por defecto las 174 que el filtro de dominio dejó pasar en la corrida publicada (comparables con la Tabla 13);
--all usa las 198. El umbral de similitud del pipeline no se aplica: no se activó en ninguna pregunta de prueba.

Uso:
    GENERATION_PROMPT_FILE=generation_prose_v1.md uv run python -m quantitative_experiments.revision.answer_quality generate \
        --condition finetuned [--dry-run]
    uv run python -m quantitative_experiments.revision.answer_quality score --conditions none base finetuned oracle --reference base
    uv run python -m quantitative_experiments.revision.answer_quality citations --conditions released_final base finetuned
    uv run python -m quantitative_experiments.revision.answer_quality code-errors --condition finetuned --model <juez>
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    OUT, RESULTS, bootstrap_for, corpus, fmt_ci, load_dataset, load_run, read_jsonl, released_run, write_jsonl,
)
from quantitative_experiments.revision.llm import LLM

DIR = OUT / "answer_quality"
HERE = Path(__file__).resolve().parent
FINAL_TRACE = RESULTS / "pipeline_final" / "dense_finetuned.trace.jsonl"
DEFAULT_RUNS = {"base": "released:dense_base", "finetuned": "released:dense_finetuned",
                "finetuned_rerank": "released:dense_finetuned_rerank100"}


def _app_settings(dry_run: bool):
    """app.core.config exige claves al importarse; en seco basta con valores de relleno (no se llama a nada)."""
    if dry_run:
        os.environ.setdefault("OPENROUTER_API_KEY", "dry-run")
        os.environ.setdefault("MONGODB_URI", "mongodb://unused")
    from app.core.config import settings
    return settings


def questions(all_questions: bool) -> list[dict]:
    ds = load_dataset()
    if all_questions:
        return ds
    answered = {r["dataset_id"] for r in read_jsonl(FINAL_TRACE) if r["is_relevant"]}
    return [d for d in ds if d["dataset_id"] in answered]


def contexts_for(condition: str, run_spec: str | None, ds: list[dict], k: int) -> dict[str, list[str]]:
    if condition == "none":
        return {d["dataset_id"]: [] for d in ds}
    if condition == "oracle":
        return {d["dataset_id"]: list(d["positive_chunk_ids"]) for d in ds}
    spec = run_spec or DEFAULT_RUNS.get(condition)
    if not spec:
        raise SystemExit(f"la condición {condition!r} necesita --run")
    run = released_run(spec.split(":", 1)[1]) if spec.startswith("released:") else load_run(Path(spec))
    return {d["dataset_id"]: run[d["dataset_id"]][:k] for d in ds}


async def generate(args) -> None:
    settings = _app_settings(args.dry_run)
    from app.core.dependencies import _provider_preferences
    from app.core.prompt_provider import prompts
    from app.graph.nodes.generation import _build_context
    from quantitative_experiments.provenance import run_provenance, sha256_text

    ds = questions(args.all)
    ctx = contexts_for(args.condition, args.run, ds, args.k)
    C = corpus()
    system = (HERE / "prompts" / "no_retrieval_system.txt").read_text(encoding="utf-8") if args.condition == "none" else prompts.generation
    prov = _provider_preferences()
    llm = LLM(settings.llm_model, DIR / "cache_generate.jsonl", temperature=settings.llm_temperature,
              max_tokens=settings.llm_max_output_tokens, concurrency=args.concurrency, dry_run=args.dry_run,
              provider=prov["provider"] if prov else None)

    async def one(d):
        docs = [C[c] for c in ctx[d["dataset_id"]]]
        user = (f"Contexto legal recuperado:\n\n{_build_context(docs)}\n\nPregunta: {d['query']}" if docs
                else f"Pregunta: {d['query']}")  # mismo formato de mensaje que app/graph/nodes/generation.py
        return d, await llm.text([{"role": "system", "content": system}, {"role": "user", "content": user}])

    results = await asyncio.gather(*(one(d) for d in ds))
    print(llm.report())
    if args.dry_run:
        return
    write_jsonl(DIR / f"{args.condition}.trace.jsonl", [
        {"dataset_id": d["dataset_id"], "condition": args.condition, "is_relevant": True, "no_results": False,
         "ranked": ctx[d["dataset_id"]], "n_above_threshold": len(ctx[d["dataset_id"]]), "answer": a} for d, a in results])
    (DIR / f"{args.condition}.provenance.json").write_text(json.dumps(run_provenance(
        condition=args.condition, run=args.run or DEFAULT_RUNS.get(args.condition), k=args.k, questions=len(ds),
        system_prompt_sha256=sha256_text(system)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {DIR / args.condition}.trace.jsonl")


def openai_reasoning_args(llm) -> None:
    """RAGAS 0.4.3 reconoce los modelos de razonamiento de OpenAI por un número de versión entero (gpt-5, gpt-6) y no
    reconoce gpt-5.6-luna: le enviaría max_tokens y temperatura, que la API de OpenAI rechaza. Se aplica el mismo mapeo
    que RAGAS usa para gpt-5: max_completion_tokens y sin temperatura ni top_p (solo aceptan el valor por defecto)."""
    args = llm.model_args
    if "max_tokens" in args:
        args["max_completion_tokens"] = args.pop("max_tokens")
    args.pop("temperature", None)
    args.pop("top_p", None)


async def score(args) -> None:
    settings = _app_settings(False)
    from openai import AsyncOpenAI
    from ragas.embeddings import OpenAIEmbeddings
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)
    # juez de RAGAS por OpenRouter o por la API de OpenAI directa (mismo modelo, facturación de OpenAI);
    # los embeddings de Answer Relevancy siguen por OpenRouter
    if args.judge_api == "openai":
        judge_client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        judge_model = args.judge_model.removeprefix("openai/")
    else:
        judge_client, judge_model = client, args.judge_model
    llm = llm_factory(judge_model, provider="openai", client=judge_client, max_tokens=args.max_tokens)
    if args.judge_api == "openai":
        openai_reasoning_args(llm)
    faith = Faithfulness(llm=llm)
    rel = AnswerRelevancy(llm=llm, embeddings=OpenAIEmbeddings(client=client, model=settings.ragas_realtime_embeddings_model))
    C = corpus()
    ds = {d["dataset_id"]: d for d in load_dataset()}
    sem = asyncio.Semaphore(args.concurrency)

    async def one(cond: str, r: dict) -> dict:
        q, out = ds[r["dataset_id"]], {"dataset_id": r["dataset_id"], "condition": cond}
        gold = [C[c]["text"] for c in q["positive_chunk_ids"]]
        got = [C[c]["text"] for c in r["ranked"]]
        async with sem:
            for name, coro in (("faithfulness", faith.ascore(user_input=q["query"], response=r["answer"], retrieved_contexts=got) if got else None),
                               ("gold_support", faith.ascore(user_input=q["query"], response=r["answer"], retrieved_contexts=gold)),
                               ("answer_relevancy", rel.ascore(user_input=q["query"], response=r["answer"]))):
                if coro is None:
                    out[name] = None
                    continue
                try:
                    out[name] = (await coro).value
                except Exception as exc:  # noqa: BLE001
                    print(f"  {cond} {name} {r['dataset_id']}: {exc}")
                    out[name] = None
        return out

    per = {}
    for cond in args.conditions:
        partial = DIR / f"{cond}.ragas_partial.jsonl"
        done = {r["dataset_id"]: r for r in read_jsonl(partial)} if partial.exists() else {}
        todo = [r for r in read_jsonl(DIR / f"{cond}.trace.jsonl") if r["dataset_id"] not in done]
        for res in asyncio.as_completed([one(cond, r) for r in todo]):
            row = await res
            done[row["dataset_id"]] = row
            with open(partial, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        per[cond] = done
        print(f"{cond}: {len(done)} respuestas puntuadas")
    summarize(args, per)


def summarize(args, per: dict[str, dict[str, dict]]) -> None:
    qids = sorted(set.intersection(*(set(v) for v in per.values())))
    sub = [d for d in load_dataset() if d["dataset_id"] in set(qids)]
    sub_ids = [d["dataset_id"] for d in sub]
    boot = bootstrap_for(sub)
    lines = [f"# Answer quality with the generator fixed ({len(sub_ids)} questions; judge {args.judge_model})", "",
             "| Condition | Faithfulness | Gold support | Answer relevancy |", "| --- | --- | --- | --- |"]
    summary = {}
    for cond, rows in per.items():
        vals = {m: np.array([rows[q][m] if rows[q][m] is not None else np.nan for q in sub_ids], float)
                for m in ("faithfulness", "gold_support", "answer_relevancy")}
        summary[cond] = {m: float(np.nanmean(v)) if not np.all(np.isnan(v)) else None for m, v in vals.items()}
        lines.append(f"| {cond} | " + " | ".join(f"{summary[cond][m]:.3f}" if summary[cond][m] is not None else "–"
                                                 for m in ("faithfulness", "gold_support", "answer_relevancy")) + " |")
        per[cond] = vals
    diffs = {}
    if args.reference in per:
        lines += ["", f"Paired differences vs {args.reference} (questions with both scores):", "",
                  "| Condition | Metric | Diff [95% CI] | p (boot) |", "| --- | --- | --- | --- |"]
        for cond, vals in per.items():
            if cond == args.reference:
                continue
            for m in ("faithfulness", "gold_support", "answer_relevancy"):
                a, b = vals[m], per[args.reference][m]
                ok = ~np.isnan(a) & ~np.isnan(b)
                if ok.sum() < 10:
                    continue
                keep = [d for d, k in zip(sub, ok) if k]
                bb = bootstrap_for(keep)
                diff = a[ok] - b[ok]
                reps = bb.resampled_means(diff)
                p = float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))
                d = bb.mean(diff)
                diffs[f"{cond}|{m}"] = {**d, "p_boot": p, "n": int(ok.sum())}
                lines.append(f"| {cond} | {m} | {fmt_ci({'estimate': d['estimate'], 'ci_low': d['ci_low'], 'ci_high': d['ci_high']})} | {p:.4f} |")
    (DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DIR / "summary.json").write_text(json.dumps({"means": summary, "paired": diffs}, indent=1), encoding="utf-8")
    print("\n".join(lines))


def citations_cmd(args) -> None:
    from quantitative_experiments.revision.citations import aggregate, score_answer
    C = corpus()
    lines = ["| Condition | Answers with citations | Citations | Precision (strict) | Precision (lenient) | Answers with an unsupported citation |",
             "| --- | --- | --- | --- | --- | --- |"]
    out = {}
    for cond in args.conditions:
        path = FINAL_TRACE if cond == "released_final" else DIR / f"{cond}.trace.jsonl"
        rows = []
        for r in read_jsonl(path):
            if not r.get("answer") or not r.get("is_relevant") or r.get("no_results"):
                continue  # solo respuestas generadas (no el mensaje fijo de fuera de dominio)
            ids = r["ranked"][: (r.get("n_above_threshold") or 10)]
            rows.append({"dataset_id": r["dataset_id"], **score_answer(r["answer"], ids, [C[c]["text"] for c in ids])})
        agg = aggregate(rows)
        out[cond] = {"aggregate": agg, "per_answer": rows}
        f = lambda x: f"{x:.3f}" if x is not None else "–"  # noqa: E731
        lines.append(f"| {cond} | {agg['answers_with_citations']}/{agg['answers']} | {agg['citations']} | {f(agg['precision_strict'])} "
                     f"| {f(agg['precision_lenient'])} | {f(agg['answers_with_unsupported_citation'])} |")
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "citations.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (DIR / "citations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


CODE_ERRORS_SYSTEM = """Eres un revisor de respuestas de un asistente legal sobre violencia familiar en el Perú. Recibirás una
pregunta, la respuesta del asistente y los documentos que el asistente recibió como contexto. Un evaluador automático marcó que
parte de la respuesta no está respaldada por esos documentos. Identifica las afirmaciones de la respuesta que NO están
respaldadas por los documentos y clasifica cada una en una de estas causas:
- "afirmacion_legal_sin_respaldo": una regla, derecho, plazo o consecuencia legal que los documentos no dicen.
- "articulo_inventado_o_equivocado": cita un artículo o norma que no está en los documentos o que dice otra cosa.
- "derivacion_o_seguridad": orientación sobre a dónde acudir, teléfonos o seguridad que no viene de los documentos.
- "consejo_general": consejo práctico o empático sin contenido legal.
- "parafrasis_respaldada": en realidad sí está en los documentos con otras palabras (el evaluador fue estricto).
Responde SOLO con JSON: {"claims": [{"text": "...", "cause": "..."}], "main_cause": "..."}"""
CAUSES = {"afirmacion_legal_sin_respaldo", "articulo_inventado_o_equivocado", "derivacion_o_seguridad", "consejo_general",
          "parafrasis_respaldada"}


async def code_errors(args) -> None:
    C = corpus()
    ds = {d["dataset_id"]: d for d in load_dataset()}
    if args.condition == "released_final":
        trace = {r["dataset_id"]: r for r in read_jsonl(FINAL_TRACE)}
        with open(RESULTS / "pipeline_final" / "ragas_luna" / "ragas_y_latencia_por_pregunta.csv", encoding="utf-8") as f:
            faith = {r["dataset_id"]: float(r["faithfulness"]) for r in csv.DictReader(f) if r["faithfulness"] not in ("", "nan")}
    else:
        trace = {r["dataset_id"]: r for r in read_jsonl(DIR / f"{args.condition}.trace.jsonl")}
        faith = {r["dataset_id"]: r["faithfulness"] for r in read_jsonl(DIR / f"{args.condition}.ragas_partial.jsonl")
                 if r.get("faithfulness") is not None}
    low = sorted(q for q, v in faith.items() if v < args.threshold)
    llm = LLM(args.model, DIR / "cache_code_errors.jsonl", temperature=0, concurrency=args.concurrency, dry_run=args.dry_run)

    def check(resp):
        if resp.get("main_cause") not in CAUSES or any(c.get("cause") not in CAUSES for c in resp.get("claims") or []):
            raise ValueError("causa fuera de la lista")

    async def one(q):
        r = trace[q]
        ids = r["ranked"][: (r.get("n_above_threshold") or 10)]
        docs = "\n\n".join(f"[Documento {i}] {C[c].get('title', '')}\n{C[c]['text']}" for i, c in enumerate(ids, 1))
        user = f"PREGUNTA:\n{ds[q]['query']}\n\nRESPUESTA:\n{r['answer']}\n\nDOCUMENTOS:\n{docs}"
        return q, await llm.json(CODE_ERRORS_SYSTEM, user, validate=check)

    res = await asyncio.gather(*(one(q) for q in low))
    print(llm.report())
    if args.dry_run:
        return
    from collections import Counter
    main = Counter(r["main_cause"] for _, r in res)
    claims = Counter(c["cause"] for _, r in res for c in r.get("claims") or [])
    out = {"condition": args.condition, "threshold": args.threshold, "answers": len(low), "coder": args.model,
           "main_cause": dict(main), "claims_by_cause": dict(claims), "per_answer": {q: r for q, r in res}}
    (DIR / f"{args.condition}.error_codes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("answers", "main_cause", "claims_by_cause")}, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--condition", required=True)
    g.add_argument("--run", default=None, help="corrida de recuperación (ruta o released:<config>) para condiciones nuevas")
    g.add_argument("--k", type=int, default=10, help="chunks que recibe el generador (10, como el pipeline)")
    g.add_argument("--all", action="store_true", help="las 198 preguntas, no solo las 174 que pasaron el filtro")
    g.add_argument("--concurrency", type=int, default=4)
    g.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--conditions", nargs="+", required=True)
    s.add_argument("--reference", default="base")
    s.add_argument("--judge-model", default="openai/gpt-5.6-luna", help="el juez de RAGAS del artículo")
    s.add_argument("--judge-api", choices=["openrouter", "openai"], default="openrouter",
                   help="openai = API de OpenAI directa (requiere OPENAI_API_KEY en .env)")
    s.add_argument("--max-tokens", type=int, default=20_000)
    s.add_argument("--concurrency", type=int, default=4)
    c = sub.add_parser("citations")
    c.add_argument("--conditions", nargs="+", required=True, help="condiciones con traza, o released_final")
    e = sub.add_parser("code-errors")
    e.add_argument("--condition", required=True, help="condición puntuada, o released_final (la corrida de la Tabla 13)")
    e.add_argument("--model", required=True, help="codificador: un modelo distinto del generador y del juez de RAGAS")
    e.add_argument("--threshold", type=float, default=0.5)
    e.add_argument("--concurrency", type=int, default=4)
    e.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "generate":
        asyncio.run(generate(args))
    elif args.cmd == "score":
        asyncio.run(score(args))
    elif args.cmd == "citations":
        citations_cmd(args)
    else:
        asyncio.run(code_errors(args))


if __name__ == "__main__":
    sys.exit(main())
