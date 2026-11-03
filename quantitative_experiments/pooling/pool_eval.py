#!/usr/bin/env python
"""Pooling de relevancia para el set de TEST de la ablación (autosuficiente: no depende del repo de datos).

Los qrels estrictos de test solo marcan el chunk de origen de cada pregunta; otro chunk que también la responde cuenta como
error y el recall queda como cota inferior, de forma desigual entre sistemas. El pooling junta los top-k de TODOS los sistemas
que se comparan y un juez independiente (ciego al sistema y al rol) decide la relevancia de cada par (pregunta, chunk).

Pasos (ver README.md de esta carpeta):
  export   Arma el pool a partir de los rankings y deja una petición por tarea en output/work/pool_test/requests/.
  pending  Lista las peticiones sin respuesta (para repartirlas entre agentes: --shard K/N).
  ingest   Valida las respuestas del juez, escribe la caché de juicios y los qrels pooled.

  python pool_eval.py export --runs ../results/runs/*.json --depth 20
  python pool_eval.py pending --shard 1/6
  python pool_eval.py ingest

Los rankings son JSON {dataset_id: [chunk_id ordenados]} por sistema (build_runs.py); el nombre del archivo es el del sistema.
Salidas en output/: pool_pairs_test.jsonl (pares y sistemas que los recuperaron; el juez NO lo ve), pool_judgments_test.jsonl,
eval_test_qrels_pooled.jsonl (origen + "si"), eval_test_qrels_pooled_parcial.jsonl (+ "parcial") y pool_test_report.json.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
DATASET = EXP / "test_set" / "eval_test_dataset.jsonl"
CORPUS = EXP / "data" / "inputs" / "document_chunks_all.jsonl"
PROMPT = HERE / "prompts" / "judge_system.txt"
OUT = Path(os.environ.get("POOL_OUT", HERE / "output"))  # POOL_OUT solo para pruebas

SEED = 520
JUDGE_TEXT_CHARS = 8000   # el juez lee el chunk completo (los del corpus miden <= 7.600 caracteres)
STEP = "pool_test"


# ---------------------------------------------------------------- utilidades
def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_unit(*parts: object) -> float:
    """Número determinista en [0, 1): orden mezclado de los chunks de una pregunta, independiente de rol y sistema."""
    return int(hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16) / float(16 ** 12)


def load_corpus() -> dict[str, dict]:
    chunks = read_jsonl(CORPUS)
    corpus = {c["chunk_id"]: c for c in chunks}
    if len(corpus) != len(chunks):
        sys.exit("chunk_id duplicados en el corpus")
    return corpus


def fingerprint() -> dict:
    return {"corpus_sha256": sha256_file(CORPUS), "dataset_sha256": sha256_file(DATASET), "judge_prompt_sha256": sha256_file(PROMPT)}


def check_fingerprint() -> None:
    path = OUT / "pool_manifest.json"
    if not path.exists():
        sys.exit("falta output/pool_manifest.json: ejecuta primero `export`")
    saved = read_json(path)["fingerprint"]
    if saved != fingerprint():
        sys.exit("el corpus, el dataset o el prompt del juez cambiaron desde `export`: repite el pooling desde cero "
                 "(borra output/) para no mezclar juicios de entradas distintas")


# ---------------------------------------------------------------- main
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mode", choices=["export", "pending", "ingest"])
ap.add_argument("--runs", nargs="*", default=[], help="export: rankings por sistema (JSON {dataset_id: [chunk_ids]})")
ap.add_argument("--depth", type=int, default=20, help="export: top-k de cada sistema que entra al pool")
ap.add_argument("--per-task", type=int, default=3, help="export: preguntas por tarea")
ap.add_argument("--shard", default=None, help="pending: K/N (reparte lo pendiente entre N agentes)")
ap.add_argument("--part", default=None, help="pending: K/N (parte estable: cada petición pertenece siempre a la misma de las N partes)")
ap.add_argument("--judge-model", default="gpt-6.1-sol", help="se registra como procedencia si la respuesta no trae _model")
args = ap.parse_args()

req_dir, res_dir = OUT / "work" / STEP / "requests", OUT / "work" / STEP / "responses"
tasks_path = OUT / "work" / STEP / "tasks.json"

if args.mode == "pending":
    done = {p.name for p in res_dir.glob("*.json")} if res_dir.exists() else set()
    all_reqs = sorted(req_dir.glob("*.json"))
    if args.part:   # parte = posición en la lista COMPLETA ordenada: balanceada y estable al avanzar el juez
        k, n = (int(x) for x in args.part.split("/"))
        all_reqs = [p for i, p in enumerate(all_reqs) if i % n == k - 1]
    reqs = [p for p in all_reqs if p.name not in done]
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        reqs = [p for i, p in enumerate(reqs) if i % n == k - 1]
    if reqs:   # sin pendientes no se imprime nada (una línea en blanco se contaría como una petición)
        print("\n".join(str(p) for p in reqs))
    sys.exit(0)

corpus = load_corpus()
eval_rows = read_jsonl(DATASET)
origin = {r["dataset_id"]: set(r["positive_chunk_ids"]) for r in eval_rows}
question = {r["dataset_id"]: r["query"] for r in eval_rows}
cache_path = OUT / "pool_judgments_test.jsonl"
cache = {(r["qid"], r["chunk_id"]): r for r in read_jsonl(cache_path)} if cache_path.exists() else {}

# ---------------------------------------------------------------- export
if args.mode == "export":
    args.runs = [r for r in args.runs if not r.endswith(".meta.json")]   # build_runs deja un .meta.json junto a cada ranking
    if not args.runs:
        sys.exit("export necesita --runs")
    mpath = OUT / "pool_manifest.json"
    if mpath.exists() and read_json(mpath)["fingerprint"] != fingerprint():
        sys.exit("el corpus, el dataset o el prompt del juez cambiaron desde el export anterior: borra output/ y repite")
    pool: dict[str, dict[str, list[str]]] = {}
    for rp in args.runs:
        name = Path(rp).stem
        for qid, ranked in read_json(Path(rp)).items():
            if qid not in origin:
                continue
            for cid in ranked[: args.depth]:
                if cid in corpus and cid not in origin[qid]:
                    pool.setdefault(qid, {}).setdefault(cid, []).append(name)
    pairs = [(q, c) for q in sorted(pool) for c in sorted(pool[q])]
    write_jsonl(OUT / "pool_pairs_test.jsonl", [{"qid": q, "chunk_id": c, "systems": sorted(pool[q][c])} for q, c in pairs])
    todo = {q: [c for c in sorted(pool[q]) if (q, c) not in cache] for q in sorted(pool)}
    todo = {q: cs for q, cs in todo.items() if cs}
    system = PROMPT.read_text(encoding="utf-8")
    qids = sorted(todo)
    manifest = {}
    for i in range(0, len(qids), args.per_task):
        batch = qids[i: i + args.per_task]
        key = "|".join(f"{q}:{','.join(todo[q])}" for q in batch)
        task_id = f"{STEP}__{hashlib.sha1(key.encode()).hexdigest()[:12]}"
        payload = [{"dataset_id": q, "question": question[q],
                    "chunks": [{"chunk_id": c, "title": corpus[c].get("title"), "text": corpus[c]["text"][:JUDGE_TEXT_CHARS]}
                               for c in sorted(todo[q], key=lambda c: stable_unit(SEED, "pool", q, c))]}
                   for q in batch]
        manifest[task_id] = [[q, c] for q in batch for c in todo[q]]
        req = req_dir / f"{task_id}.json"
        if not (res_dir / f"{task_id}.json").exists() and not req.exists():
            write_json(req, {"task_id": task_id, "system": system,
                             "user": "Juzga estos pares:\n\n" + json.dumps(payload, ensure_ascii=False, indent=1),
                             "write_response_to": str((res_dir / f"{task_id}.json").resolve())})
    res_dir.mkdir(parents=True, exist_ok=True)
    write_json(tasks_path, manifest)
    write_json(mpath, {"fingerprint": fingerprint(), "runs": sorted(Path(r).stem for r in args.runs), "depth": args.depth})
    n_new = sum(len(v) for v in todo.values())
    print(f"[{STEP}] pool: {len(pairs)} pares ({len(pool)} preguntas, prof. {args.depth}, {len(args.runs)} sistemas) | "
          f"ya juzgados: {len(pairs) - n_new} | por juzgar: {n_new} en {len(manifest)} tareas")
    print(f"   peticiones: {req_dir}\n   siguiente: uv run python quantitative_experiments/pooling/run_judge.py")
    sys.exit(0)

# ---------------------------------------------------------------- ingest
check_fingerprint()
manifest = read_json(tasks_path)
new, pending, bad = 0, 0, []
for task_id, pairs in manifest.items():
    res_path = res_dir / f"{task_id}.json"
    if not res_path.exists():
        pending += 1
        continue
    resp = read_json(res_path)
    got = {(j["dataset_id"], j["chunk_id"]): j for j in resp.get("judgments") or []}
    missing = [(q, c) for q, c in pairs if (q, c) not in got]
    invalid = [k for k, j in got.items() if j.get("relevance") not in ("si", "parcial", "no")]
    if missing or invalid:
        bad.append((task_id, len(missing), len(invalid)))
        continue
    for q, c in pairs:
        j = got[(q, c)]
        cache[(q, c)] = {"qid": q, "chunk_id": c, "relevance": j["relevance"], "rationale": str(j.get("rationale") or ""),
                         "judge_model": str(resp.get("_model") or args.judge_model), "task_id": task_id}
        new += 1
write_jsonl(cache_path, [cache[k] for k in sorted(cache)])


def qrels(include_parcial: bool) -> list[dict]:
    out = []
    for q in sorted(origin):
        rel = set(origin[q])
        rel |= {c for (qq, c), r in cache.items() if qq == q and (r["relevance"] == "si" or (include_parcial and r["relevance"] == "parcial"))}
        out += [{"qid": q, "docid": c, "relevance": 1} for c in sorted(rel)]
    return out


main_q, parcial_q = qrels(False), qrels(True)
write_jsonl(OUT / "eval_test_qrels_pooled.jsonl", main_q)
write_jsonl(OUT / "eval_test_qrels_pooled_parcial.jsonl", parcial_q)

nq = len(origin)
rep = {"preguntas": nq, "pares_juzgados": len(cache), "nuevos_en_esta_corrida": new, "tareas_pendientes": pending,
       "tareas_invalidas": bad, "veredictos": dict(Counter(r["relevance"] for r in cache.values())),
       "jueces": dict(Counter(r["judge_model"] for r in cache.values())),
       "positivos_origen": sum(len(v) for v in origin.values()), "positivos_ampliados_si": len(main_q),
       "positivos_ampliados_si_parcial": len(parcial_q),
       "positivos_extra_por_pregunta_si": round((len(main_q) - sum(len(v) for v in origin.values())) / max(1, nq), 2)}
write_json(OUT / "pool_test_report.json", rep)
print(json.dumps(rep, ensure_ascii=False, indent=1))
if pending or bad:
    print(f"\nFaltan {pending} tareas por responder y {len(bad)} están mal formadas: repítelas y vuelve a ejecutar ingest.")
    sys.exit(3)
