"""Calidad de respuesta con el generador fijo (MAJOR #6, H3 del registro): ¿mejor recuperación da mejores respuestas?

Condiciones (mismo generador, temperatura, prompt y preguntas; solo cambia el contexto):
  none                  sin recuperación (prompt prompts/no_retrieval_system.txt: responde con su conocimiento)
  base                  top-k de released:dense_base
  base_rerank           top-k de released:dense_base_rerank100
  finetuned             top-k de released:dense_finetuned
  finetuned_rerank      top-k de released:dense_finetuned_rerank100 (o de una corrida de rerank_local.py)
  source_context        solo el chunk de origen y sus casi duplicados; `oracle` es un alias histórico.
                        No es un oráculo jurídico ni un techo de calidad: las fuentes pueden ser incompletas.
  cualquier otra        --run <ruta a corrida JSON> o released:<config>

Pasos:
  generate     Escribe <condición>.trace.jsonl (mismo formato que ablation/full_pipeline.py, legible por ragas_final.py) y
               <condición>.provenance.json (modelo, temperatura, proveedor, archivo y SHA-256 del prompt, commit).
  score        RAGAS Faithfulness con los contextos recibidos, soporte por el chunk de origen (clave histórica "gold_support",
               medible también sin recuperación) y Answer Relevancy; diferencias pareadas contra --reference con bootstrap
               por chunk de origen. Reanudable.
  citations    Exactitud de las citas normativas (citations.py) por condición; `released_final` puntúa la corrida publicada.
  code-errors  Un LLM clasifica las respuestas con Faithfulness < --threshold en causas (afirmación sin respaldo, artículo
               inventado o equivocado, texto de derivación o seguridad, consejo general, paráfrasis que el juez no reconoce).

Preguntas: por defecto las 174 que el filtro de dominio dejó pasar en la corrida publicada (comparables con la Tabla 13);
--all usa todo --dataset (198 por defecto). Un dataset diferente requiere --all; --out-dir aísla las salidas y cachés.
Estas métricas automáticas no validan corrección jurídica. El umbral de similitud del pipeline no se aplica.

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
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

from quantitative_experiments.revision.data import (
    OUT, RESULTS, TEST_DATASET, bootstrap_for, corpus, fmt_ci, load_dataset, load_run, read_jsonl, released_run, write_jsonl,
)
from quantitative_experiments.revision.llm import LLM

DIR = OUT / "answer_quality"
HERE = Path(__file__).resolve().parent
FINAL_TRACE = RESULTS / "pipeline_final" / "dense_finetuned.trace.jsonl"
DEFAULT_RUNS = {"base": "released:dense_base", "finetuned": "released:dense_finetuned",
                "base_rerank": "released:dense_base_rerank100",
                "finetuned_rerank": "released:dense_finetuned_rerank100"}
METRICS = ("faithfulness", "gold_support", "answer_relevancy")


def output_dir(args) -> Path:
    return Path(getattr(args, "out_dir", None) or DIR)


def dataset_path(args) -> Path:
    return Path(getattr(args, "dataset", None) or TEST_DATASET)


def custom_dataset(args) -> bool:
    return dataset_path(args).resolve() != TEST_DATASET.resolve()


def selected_dataset(args) -> list[dict]:
    rows = load_dataset(dataset_path(args))
    ids = [r.get("dataset_id") for r in rows]
    if not ids or any(not isinstance(q, str) or not q for q in ids) or len(ids) != len(set(ids)):
        raise ValueError("Dataset needs unique, nonempty query IDs")
    return rows


def _app_settings(dry_run: bool):
    """app.core.config exige claves al importarse; en seco basta con valores de relleno (no se llama a nada)."""
    if dry_run:
        os.environ.setdefault("OPENROUTER_API_KEY", "dry-run")
        os.environ.setdefault("MONGODB_URI", "mongodb://unused")
    from app.core.config import settings
    return settings


def questions(all_questions: bool, args=None) -> list[dict]:
    if custom_dataset(args) and not all_questions:
        raise ValueError("A custom --dataset requires --all; the historical 174-query filter is only for Test A")
    ds = selected_dataset(args)
    if all_questions:
        return ds
    answered = {r["dataset_id"] for r in read_jsonl(FINAL_TRACE) if r["is_relevant"]}
    return [d for d in ds if d["dataset_id"] in answered]


def contexts_for(condition: str, run_spec: str | None, ds: list[dict], k: int) -> dict[str, list[str]]:
    if condition == "none":
        return {d["dataset_id"]: [] for d in ds}
    if condition in ("source_context", "oracle"):
        return {d["dataset_id"]: list(d["positive_chunk_ids"]) for d in ds}
    spec = run_spec or DEFAULT_RUNS.get(condition)
    if not spec:
        raise SystemExit(f"la condición {condition!r} necesita --run")
    run = released_run(spec.split(":", 1)[1]) if spec.startswith("released:") else load_run(Path(spec))
    return {d["dataset_id"]: run[d["dataset_id"]][:k] for d in ds}


async def generate(args) -> None:
    ds = questions(args.all, args)
    ctx = contexts_for(args.condition, args.run, ds, args.k)
    directory = output_dir(args)
    settings = _app_settings(args.dry_run)
    from app.core.dependencies import _provider_preferences
    from app.core.prompt_provider import prompts
    from app.graph.nodes.generation import _build_context
    from quantitative_experiments.provenance import run_provenance, sha256_text

    C = corpus()
    system = (HERE / "prompts" / "no_retrieval_system.txt").read_text(encoding="utf-8") if args.condition == "none" else prompts.generation
    prov = _provider_preferences()
    llm = LLM(settings.llm_model, directory / "cache_generate.jsonl", temperature=settings.llm_temperature,
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
    write_jsonl(directory / f"{args.condition}.trace.jsonl", [
        {"dataset_id": d["dataset_id"], "condition": args.condition, "is_relevant": True, "no_results": False,
         "ranked": ctx[d["dataset_id"]], "n_above_threshold": len(ctx[d["dataset_id"]]), "answer": a} for d, a in results])
    (directory / f"{args.condition}.provenance.json").write_text(json.dumps(run_provenance(
        condition=args.condition, run=args.run or DEFAULT_RUNS.get(args.condition), k=args.k, questions=len(ds),
        dataset=str(dataset_path(args).resolve()), question_ids=[d["dataset_id"] for d in ds],
        system_prompt_sha256=sha256_text(system)), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {directory / args.condition}.trace.jsonl")


def openai_reasoning_args(llm) -> None:
    """RAGAS 0.4.3 reconoce los modelos de razonamiento de OpenAI por un número de versión entero (gpt-5, gpt-6) y no
    reconoce gpt-5.6-luna: le enviaría max_tokens y temperatura, que la API de OpenAI rechaza. Se aplica el mismo mapeo
    que RAGAS usa para gpt-5: max_completion_tokens y sin temperatura ni top_p (solo aceptan el valor por defecto)."""
    args = llm.model_args
    if "max_tokens" in args:
        args["max_completion_tokens"] = args.pop("max_tokens")
    args.pop("temperature", None)
    args.pop("top_p", None)


def validate_trace_population(traces: dict[str, list[dict]], dataset: list[dict], *, require_all=False) -> list[str]:
    """Reject duplicate, unknown or unmatched query populations before any scoring API call."""
    known = {d["dataset_id"] for d in dataset}
    population = None
    for condition, rows in traces.items():
        ids = [r.get("dataset_id") for r in rows]
        if not ids or any(not isinstance(q, str) or not q for q in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{condition}: trace needs unique, nonempty query IDs")
        found = set(ids)
        if not found <= known:
            raise ValueError(f"{condition}: trace contains queries outside the selected dataset")
        if population is not None and found != population:
            raise ValueError("All conditions must contain exactly the same query population")
        if require_all and found != known:
            raise ValueError(f"{condition}: custom dataset traces must cover all selected queries")
        for row in rows:
            ranked = row.get("ranked")
            if not isinstance(row.get("answer"), str) or not isinstance(ranked, list):
                raise ValueError(f"{condition}: trace requires an answer and ranked document list")
            if any(not isinstance(c, str) or not c for c in ranked) or len(ranked) != len(set(ranked)):
                raise ValueError(f"{condition}: invalid or duplicate context document IDs")
        population = found
    if population is None:
        raise ValueError("At least one condition is required")
    return [d["dataset_id"] for d in dataset if d["dataset_id"] in population]


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def score_input_fingerprint(question: dict, trace: dict, documents: dict) -> str:
    """Hash the full response and every text actually supplied to the metrics, in order."""
    def contexts(ids):
        try:
            return [{"chunk_id": c, "text": documents[c]["text"]} for c in ids]
        except KeyError as exc:
            raise ValueError(f"Trace/source document is absent from corpus: {exc}") from exc
    return fingerprint({"dataset_id": question["dataset_id"], "query": question["query"],
                        "answer": trace["answer"], "contexts": contexts(trace["ranked"]),
                        "source_chunk": question["source_chunk"],
                        "source_contexts": contexts(question["positive_chunk_ids"])})


def score_configuration(args, settings) -> dict:
    try:
        version = importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        version = "unavailable"
    return {"schema": 2, "judge_model": args.judge_model, "judge_api": args.judge_api,
            "judge_base_url": (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                               if args.judge_api == "openai" else settings.openrouter_base_url),
            "max_tokens": args.max_tokens, "embeddings_model": settings.ragas_realtime_embeddings_model,
            "embeddings_base_url": settings.openrouter_base_url, "ragas_version": version,
            "reference_condition": args.reference, "metrics": list(METRICS)}


def finite_score(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value) and 0 <= value <= 1


def cached_scores_complete(row: dict, trace: dict) -> bool:
    required = ("gold_support", "answer_relevancy") + (("faithfulness",) if trace["ranked"] else ())
    return all(finite_score(row.get(metric)) for metric in required)


async def score(args) -> None:
    directory = output_dir(args)
    dataset = selected_dataset(args)
    if not args.conditions or len(args.conditions) != len(set(args.conditions)):
        raise ValueError("Conditions must be unique and nonempty")
    if args.reference not in args.conditions:
        raise ValueError("Reference must be one of the scored conditions")
    traces = {cond: read_jsonl(directory / f"{cond}.trace.jsonl") for cond in args.conditions}
    qids = validate_trace_population(traces, dataset, require_all=custom_dataset(args))
    C = corpus()
    ds = {d["dataset_id"]: d for d in dataset}
    input_hashes = {cond: {r["dataset_id"]: score_input_fingerprint(ds[r["dataset_id"]], r, C) for r in rows}
                    for cond, rows in traces.items()}
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
    config = score_configuration(args, settings)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(cond: str, r: dict) -> dict:
        input_hash = input_hashes[cond][r["dataset_id"]]
        q, out = ds[r["dataset_id"]], {"dataset_id": r["dataset_id"], "condition": cond,
            "input_fingerprint": input_hash, "cache_key": fingerprint({"input": input_hash, "config": config}),
            "score_config": config, "not_applicable": [] if r["ranked"] else ["faithfulness"]}
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
                    value = float((await coro).value)
                    if not finite_score(value):
                        raise ValueError("Metric returned a nonfinite or out-of-range score")
                    out[name] = value
                except Exception as exc:  # noqa: BLE001
                    print(f"  {cond} {name} {r['dataset_id']}: {exc}")
                    out[name] = None
        return out

    per = {}
    for cond in args.conditions:
        partial = directory / f"{cond}.ragas_partial.jsonl"
        cached = {r.get("cache_key"): r for r in read_jsonl(partial)} if partial.exists() else {}
        done, todo = {}, []
        for row in traces[cond]:
            qid = row["dataset_id"]
            key = fingerprint({"input": input_hashes[cond][qid], "config": config})
            previous = cached.get(key)
            if previous is not None and cached_scores_complete(previous, row):
                done[qid] = previous
            else:
                todo.append(row)
        for res in asyncio.as_completed([one(cond, r) for r in todo]):
            row = await res
            done[row["dataset_id"]] = row
            with open(partial, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        per[cond] = done
        print(f"{cond}: {len(done)} respuestas puntuadas")
    summarize(args, per, expected_qids=qids)


def summarize(args, per: dict[str, dict[str, dict]], expected_qids=None) -> None:
    dataset = selected_dataset(args)
    known = {d["dataset_id"] for d in dataset}
    qids = set(expected_qids) if expected_qids is not None else set().union(*(set(v) for v in per.values()))
    if not qids or not qids <= known or any(not set(rows) <= qids for rows in per.values()):
        raise ValueError("Invalid score population for selected dataset")
    sub = [d for d in dataset if d["dataset_id"] in qids]
    sub_ids = [d["dataset_id"] for d in sub]
    lines = [f"# Answer quality with the generator fixed ({len(sub_ids)} questions; judge {args.judge_model})", "",
             "Source-context support retains the legacy JSON key `gold_support`; it is not legal validation.", "",
             "| Condition | Faithfulness | Source-context support | Answer relevancy |", "| --- | --- | --- | --- |"]
    summary, coverage, values = {}, {}, {}
    for cond, rows in per.items():
        vals = {m: np.array([rows.get(q, {}).get(m) if finite_score(rows.get(q, {}).get(m)) else np.nan
                            for q in sub_ids], float) for m in METRICS}
        summary[cond] = {m: float(np.nanmean(v)) if not np.all(np.isnan(v)) else None for m, v in vals.items()}
        coverage[cond] = {}
        for metric, scores in vals.items():
            scored = int(np.isfinite(scores).sum())
            not_applicable = sum(metric in rows.get(q, {}).get("not_applicable", []) and not np.isfinite(scores[i])
                                 for i, q in enumerate(sub_ids))
            coverage[cond][metric] = {"scored": scored, "missing": len(sub_ids) - scored - int(not_applicable),
                                      "not_applicable": int(not_applicable)}
        lines.append(f"| {cond} | " + " | ".join(f"{summary[cond][m]:.3f}" if summary[cond][m] is not None else "–"
                                                 for m in METRICS) + " |")
        values[cond] = vals
    lines += ["", "| Condition | Metric | Scored | Missing | Not applicable |", "| --- | --- | --- | --- | --- |"]
    for condition, metrics in coverage.items():
        lines += [f"| {condition} | {metric} | {c['scored']} | {c['missing']} | {c['not_applicable']} |"
                  for metric, c in metrics.items()]
    diffs = {}
    if args.reference in values:
        lines += ["", f"Paired differences vs {args.reference} (questions with both scores):", "",
                  "| Condition | Metric | Paired n | Missing pairs | Diff [95% CI] | p (boot) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for cond, vals in values.items():
            if cond == args.reference:
                continue
            for m in METRICS:
                a, b = vals[m], values[args.reference][m]
                ok = ~np.isnan(a) & ~np.isnan(b)
                n = int(ok.sum())
                entry = {"n": n, "missing": len(sub_ids) - n}
                if ok.sum() < 10:
                    diffs[f"{cond}|{m}"] = {**entry, "estimate": None, "ci_low": None, "ci_high": None,
                                            "p_boot": None, "status": "insufficient_paired_scores"}
                    lines.append(f"| {cond} | {m} | {n} | {entry['missing']} | Not estimated (<10 pairs) | – |")
                    continue
                keep = [d for d, k in zip(sub, ok) if k]
                bb = bootstrap_for(keep)
                diff = a[ok] - b[ok]
                reps = bb.resampled_means(diff)
                p = float(min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean())))
                d = bb.mean(diff)
                diffs[f"{cond}|{m}"] = {**d, **entry, "p_boot": p, "status": "estimated"}
                lines.append(f"| {cond} | {m} | {n} | {entry['missing']} | {fmt_ci(d)} | {p:.4f} |")
    directory = output_dir(args)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps({"n_questions": len(sub_ids), "question_ids": sub_ids,
        "dataset_sha256": hashlib.sha256(dataset_path(args).read_bytes()).hexdigest(),
        "means": summary, "coverage": coverage, "paired": diffs, "reference": args.reference,
        "judge_model": args.judge_model, "multiplicity": "H1-H3 Holm adjustment pending"}, indent=1, allow_nan=False), encoding="utf-8")
    print("\n".join(lines))


def citations_cmd(args) -> None:
    from quantitative_experiments.revision.citations import aggregate, score_answer
    directory = output_dir(args)
    dataset = selected_dataset(args)
    C = corpus()
    lines = ["| Condition | Answers with citations | Citations | Precision (strict) | Precision (lenient) | Answers with an unsupported citation |",
             "| --- | --- | --- | --- | --- | --- |"]
    out = {}
    for cond in args.conditions:
        if cond == "released_final" and custom_dataset(args):
            raise ValueError("released_final belongs to Test A, not a custom dataset")
        path = FINAL_TRACE if cond == "released_final" else directory / f"{cond}.trace.jsonl"
        traces = read_jsonl(path)
        validate_trace_population({cond: traces}, dataset, require_all=custom_dataset(args))
        rows = []
        for r in traces:
            if not r.get("answer") or not r.get("is_relevant") or r.get("no_results"):
                continue  # solo respuestas generadas (no el mensaje fijo de fuera de dominio)
            ids = r["ranked"][: (r.get("n_above_threshold") or 10)]
            rows.append({"dataset_id": r["dataset_id"], **score_answer(r["answer"], ids, [C[c]["text"] for c in ids])})
        agg = aggregate(rows)
        out[cond] = {"aggregate": agg, "per_answer": rows}
        f = lambda x: f"{x:.3f}" if x is not None else "–"  # noqa: E731
        lines.append(f"| {cond} | {agg['answers_with_citations']}/{agg['answers']} | {agg['citations']} | {f(agg['precision_strict'])} "
                     f"| {f(agg['precision_lenient'])} | {f(agg['answers_with_unsupported_citation'])} |")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "citations.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    (directory / "citations.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    directory = output_dir(args)
    dataset = selected_dataset(args)
    C = corpus()
    ds = {d["dataset_id"]: d for d in dataset}
    if args.condition == "released_final":
        if custom_dataset(args):
            raise ValueError("released_final belongs to Test A, not a custom dataset")
        trace = {r["dataset_id"]: r for r in read_jsonl(FINAL_TRACE)}
        with open(RESULTS / "pipeline_final" / "ragas_luna" / "ragas_y_latencia_por_pregunta.csv", encoding="utf-8") as f:
            faith = {r["dataset_id"]: float(r["faithfulness"]) for r in csv.DictReader(f) if r["faithfulness"] not in ("", "nan")}
    else:
        rows = read_jsonl(directory / f"{args.condition}.trace.jsonl")
        validate_trace_population({args.condition: rows}, dataset, require_all=custom_dataset(args))
        trace = {r["dataset_id"]: r for r in rows}
        hashes = {qid: score_input_fingerprint(ds[qid], row, C) for qid, row in trace.items()}
        latest = {r["dataset_id"]: r for r in read_jsonl(directory / f"{args.condition}.ragas_partial.jsonl")
                  if r["dataset_id"] in hashes and r.get("input_fingerprint") == hashes[r["dataset_id"]]}
        if set(latest) != set(trace):
            raise ValueError("No scores match every current answer/context; run score before code-errors")
        faith = {qid: r["faithfulness"] for qid, r in latest.items() if finite_score(r.get("faithfulness"))}
    low = sorted(q for q, v in faith.items() if v < args.threshold)
    llm = LLM(args.model, directory / "cache_code_errors.jsonl", temperature=0, concurrency=args.concurrency, dry_run=args.dry_run)

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
    (directory / f"{args.condition}.error_codes.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("answers", "main_cause", "claims_by_cause")}, ensure_ascii=False, indent=1))


def main(argv=None) -> None:
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
    for parser in (g, s, c, e):
        parser.add_argument("--dataset", type=Path, default=TEST_DATASET)
        parser.add_argument("--out-dir", type=Path, default=DIR)
    args = ap.parse_args(argv)
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
