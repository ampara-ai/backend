"""Test B (H1 del registro): preguntas nuevas para los mismos chunks de prueba, escritas por OTRA familia de modelos y con
OTRO prompt, sin el filtro de solapamiento léxico. Responde al comentario MAJOR #1 (fuga de estilo del generador) y al #27.

Pasos:
  generate  El modelo generador (--model, de otra familia que Claude) escribe hasta --per-chunk preguntas por chunk de la
            partición de prueba (165 chunks), con prompts/test_b_system.txt. Filtros: largo, sin números de artículo o ley, sin
            casi duplicados (Jaccard >= 0.8) entre sí ni con las preguntas del Test A. El solapamiento léxico se registra pero NO
            se filtra.
  verify    El mismo juez del artículo (gpt-6.1-sol, mismo prompt que el pooling) confirma que el chunk de origen responde la
            pregunta; solo quedan las "si". Escribe el conjunto en el formato del Test A: eval_test_b_dataset.jsonl,
            eval_test_b_qrels.jsonl (origen + sus casi duplicados, la definición estricta del Test A) y eval_test_b_queries.jsonl.

Ambos pasos son reanudables (caché de llamadas) y aceptan --dry-run (sin clave ni costo: cuenta peticiones y muestra la primera).

Uso:
    uv run python -m quantitative_experiments.revision.test_b generate --model <id OpenRouter> [--dry-run]
    uv run python -m quantitative_experiments.revision.test_b verify [--judge-model openai/gpt-6.1-sol] [--dry-run]
"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path

from quantitative_experiments.revision.data import (
    EXP, OUT, TEST_DATASET, chunk_split, corpus, data_repo, load_dataset, read_jsonl, write_jsonl,
)
from quantitative_experiments.revision.llm import LLM

HERE = Path(__file__).resolve().parent
PROMPT = HERE / "prompts" / "test_b_system.txt"
JUDGE_PROMPT = EXP / "pooling" / "prompts" / "judge_system.txt"
DIR = OUT / "test_b"
TEXT_CHARS = 8000  # como el pooling: el juez y el generador leen el chunk completo
CITATION = re.compile(r"\b(art[íi]culo|art\.|ley|decreto|expediente|casaci[óo]n|exp\.)\s*(n[°º.]?\s*)?\d", re.IGNORECASE)


def _pipeline_module(name: str):
    """Carga un módulo de pipeline_v2 del repo de datos (misma definición de solapamiento léxico y de etiquetas)."""
    path = data_repo() / "generacion_dataset" / "pipeline_v2" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"ampara_pipeline_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def duplicates_of(chunk_id: str) -> list[str]:
    for g in chunk_split()["duplicate_groups"]:
        if chunk_id in g:
            return [chunk_id] + [c for c in g if c != chunk_id]
    return [chunk_id]


def test_chunks() -> list[str]:
    C = corpus()
    return sorted(c for c, s in chunk_split()["assignments"].items() if s == "test" and c in C)


async def generate(args) -> None:
    common, dcfg = _pipeline_module("common"), _pipeline_module("config")
    system = (PROMPT.read_text(encoding="utf-8").replace("<<K>>", str(args.per_chunk))
              .replace("<<TOPICS>>", ", ".join(dcfg.TOPICS)).replace("<<BARRIERS>>", ", ".join(dcfg.BARRIERS)))
    C = corpus()
    chunks = test_chunks()[: args.limit_chunks or None]
    llm = LLM(args.model, DIR / f"cache_generate.jsonl", temperature=args.temperature, concurrency=args.concurrency,
              dry_run=args.dry_run, max_tokens=8000)
    batches = [chunks[i:i + args.batch] for i in range(0, len(chunks), args.batch)]

    def validator(batch):
        def check(resp):
            got = {r.get("chunk_id") for r in resp.get("results") or []}
            if not set(batch) <= got:
                raise ValueError(f"faltan chunks: {set(batch) - got}")
        return check

    async def one(batch):
        payload = [{"chunk_id": c, "title": C[c].get("title"), "text": C[c]["text"][:TEXT_CHARS]} for c in batch]
        user = "Fragmentos:\n\n" + json.dumps(payload, ensure_ascii=False, indent=1)
        return batch, await llm.json(system, user, validate=validator(batch))

    results = await asyncio.gather(*(one(b) for b in batches))
    print(llm.report())
    if args.dry_run:
        return

    test_a = {}
    for d in load_dataset(TEST_DATASET):
        test_a.setdefault(d["source_chunk"], []).append(d["query"])
    rows, report = [], Counter()
    for batch, resp in results:
        by_chunk = {r["chunk_id"]: r for r in resp["results"]}
        for c in batch:
            r = by_chunk[c]
            if r.get("skip"):
                report["chunk_omitido"] += 1
                continue
            kept: list[str] = []
            for q in r.get("questions") or []:
                text = (q.get("query") or "").strip()
                if not 30 <= len(text) <= 300:
                    report["rechazada_largo"] += 1
                elif CITATION.search(text):
                    report["rechazada_cita_norma"] += 1
                elif any(common.jaccard(text, k) >= 0.8 for k in kept + test_a.get(c, [])):
                    report["rechazada_casi_duplicada"] += 1
                else:
                    kept.append(text)
                    rows.append({"dataset_id": f"testb::{c}::q{len(kept):02d}", "origin_chunk_id": c, "query": text,
                                 "topic": q.get("topic"), "barrier": q.get("barrier"),
                                 "lexical_overlap": round(common.lexical_overlap(text, C[c]["text"]), 3),
                                 "generator_model": args.model})
    report["preguntas_aceptadas"] = len(rows)
    report["chunks_con_preguntas"] = len({r["origin_chunk_id"] for r in rows})
    write_jsonl(DIR / "generated.jsonl", rows)
    (DIR / "generate_report.json").write_text(json.dumps(
        {"generator_model": args.model, "temperature": args.temperature, "prompt_sha256": sha256(system),
         "chunks": len(chunks), "counts": dict(report)}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(report), ensure_ascii=False))


async def verify(args) -> None:
    C = corpus()
    gen = read_jsonl(DIR / "generated.jsonl")
    system = JUDGE_PROMPT.read_text(encoding="utf-8")
    llm = LLM(args.judge_model, DIR / "cache_verify.jsonl", temperature=0, concurrency=args.concurrency,
              dry_run=args.dry_run, reasoning_effort=args.effort, max_tokens=8000, api=args.judge_api)
    tasks = [gen[i:i + args.per_task] for i in range(0, len(gen), args.per_task)]

    def validator(task):
        want = {(q["dataset_id"], q["origin_chunk_id"]) for q in task}

        def check(resp):
            got = {(j["dataset_id"], j["chunk_id"]) for j in resp.get("judgments") or []
                   if j.get("relevance") in ("si", "parcial", "no")}
            if not want <= got:
                raise ValueError(f"{len(want - got)} pares sin veredicto válido")
        return check

    async def one(task):
        payload = [{"dataset_id": q["dataset_id"], "question": q["query"],
                    "chunks": [{"chunk_id": q["origin_chunk_id"], "title": C[q["origin_chunk_id"]].get("title"),
                                "text": C[q["origin_chunk_id"]]["text"][:TEXT_CHARS]}]} for q in task]
        user = "Juzga estos pares:\n\n" + json.dumps(payload, ensure_ascii=False, indent=1)
        return await llm.json(system, user, validate=validator(task))

    responses = await asyncio.gather(*(one(t) for t in tasks))
    print(llm.report())
    if args.dry_run:
        return
    verdict = {(j["dataset_id"], j["chunk_id"]): j["relevance"] for resp in responses for j in resp["judgments"]}
    counts = Counter(verdict.get((q["dataset_id"], q["origin_chunk_id"])) for q in gen)
    kept = []
    for q in gen:
        if verdict.get((q["dataset_id"], q["origin_chunk_id"])) != "si":
            continue
        pos = duplicates_of(q["origin_chunk_id"])
        kept.append({
            "dataset_id": q["dataset_id"], "case_id": f"heldout_{q['origin_chunk_id']}", "query": q["query"],
            "topic": q["topic"], "barrier": q["barrier"], "positive_chunk_ids": pos,
            "positive_source_types": sorted({C[c]["source_type"] for c in pos}),
            "generation": {"method": "llm_synthetic_from_chunk_test_b", "generator": q["generator_model"],
                           "judge": args.judge_model, "split": "test", "seen_in_training": False, "qrels_complete": False,
                           "lexical_overlap": q["lexical_overlap"], "prompt_version": "test_b_v1"},
        })
    write_jsonl(DIR / "eval_test_b_dataset.jsonl", kept)
    write_jsonl(DIR / "eval_test_b_qrels.jsonl",
                [{"qid": r["dataset_id"], "docid": c, "relevance": 1} for r in kept for c in r["positive_chunk_ids"]])
    write_jsonl(DIR / "eval_test_b_queries.jsonl", [{"id": r["dataset_id"], "text": r["query"]} for r in kept])
    report = {"judge_model": args.judge_model, "judge_prompt_sha256": sha256(system), "verdicts": dict(counts),
              "kept": len(kept), "source_chunks": len({r["positive_chunk_ids"][0] for r in kept})}
    (DIR / "verify_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True, help="generador de otra familia que Claude (id de OpenRouter)")
    g.add_argument("--per-chunk", type=int, default=2)
    g.add_argument("--batch", type=int, default=4, help="chunks por petición")
    g.add_argument("--temperature", type=float, default=0.7)
    g.add_argument("--limit-chunks", type=int, default=0)
    v = sub.add_parser("verify")
    v.add_argument("--judge-model", default="openai/gpt-6.1-sol", help="el juez del artículo, para que A y B sean comparables")
    v.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    v.add_argument("--judge-api", choices=["openrouter", "openai"], default="openrouter",
                   help="openai = API de OpenAI directa (requiere OPENAI_API_KEY en .env)")
    v.add_argument("--per-task", type=int, default=3)
    for p in (g, v):
        p.add_argument("--concurrency", type=int, default=6)
        p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(generate(args) if args.cmd == "generate" else verify(args))


if __name__ == "__main__":
    main()
