"""Evalúa las configuraciones de la ablación DENTRO del pipeline completo de AMPARA.

Reutiliza los nodos de producción (app/graph/nodes): filtro de dominio, decisión de recuperación (contexto suficiente),
generación con las fuentes recuperadas y source grounding, y el mismo ruteo (app/graph/edges.py), incluido el umbral de similitud (mensaje fijo si ningún chunk lo supera) en las configuraciones densas. La única diferencia con el grafo
desplegado (app/graph/rag_graph.py, que no se modifica) es el paso de recuperación: en lugar de `embed_query -> semantic_search`
con k = 10, un nodo `retrieve` ejecuta la configuración de la ablación (BM25, denso, híbrido, con o sin reranker; las mismas funciones
de eval_ablation.get_config_docs) y guarda el ranking completo de 100 candidatos para calcular las métricas @10 y @100. Al generador
se le pasan los mismos chunks que recibiría en producción para esa configuración (top-10; con reranker, su top-5).

Qué mide:
  * Nivel recuperador: ranking de 100 candidatos de cada pregunta que el filtro de dominio dejó pasar (`<config>.retriever.json`).
  * Nivel pipeline: igual, pero las preguntas rechazadas por el filtro cuentan con ranking vacío, de modo que Recall/NDCG incluyen los
    errores de enrutamiento (`<config>.json`; es el archivo que consume score_runs.py).
  * Traza por pregunta (`<config>.trace.jsonl`): decisión del filtro, decisión de contexto, ranking, y con `--with-generation` la respuesta
    y las fuentes citadas.

El filtro de dominio es independiente de la configuración de recuperación; su decisión se guarda en caché por texto de pregunta para
no repetir la llamada al LLM en cada configuración (`filter_cache.json` en --out-dir). Las preguntas son de un solo turno: el nodo de
decisión de recuperación responde siempre "contexto insuficiente" (no hay historial), y el historial conversacional no se ejercita.

Requiere Mongo indexado con la variante de embedding de la configuración (igual que build_runs.py) y las claves de OpenRouter.

Uso:
    uv run python -m quantitative_experiments.ablation.full_pipeline --dataset-path <dataset.jsonl> \
        --out-dir <carpeta> --configs dense_finetuned hybrid_finetuned [--with-generation] [--limit N]
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from typing import Annotated
from pathlib import Path

from langgraph.graph import END, START, StateGraph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import settings  # noqa: E402
from app.graph.edges import route_context, route_relevance, route_retrieval  # noqa: E402
from app.graph.nodes import no_results_node, out_of_domain_node  # noqa: E402
from app.graph.nodes.context_check import context_check_node  # noqa: E402
from app.graph.nodes.generation import generate_node  # noqa: E402
from app.graph.nodes.relevance_filter import relevance_filter_node  # noqa: E402
from app.graph.nodes.retrieval import apply_score_threshold, atlas_score_to_cosine  # noqa: E402
from app.graph.nodes.source_grounding import source_grounding_node  # noqa: E402
from app.graph.state import RAGState  # noqa: E402
from quantitative_experiments.ablation.build_runs import model_meta, verify_index  # noqa: E402
from quantitative_experiments.ablation.eval_ablation import HYBRID_FUSION, config_arg, get_config_docs, parse_config  # noqa: E402
from scripts.eval_dataset_ragas import load_dataset  # noqa: E402


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class AblationState(RAGState):
    node_ms: Annotated[dict, _merge]   # duración de cada nodo en ms (latencia por etapa)
    config: str
    reranker_backend: str
    ranked_chunk_ids: list[str]
    top1_cosine: float | None
    n_above_threshold: int | None


def timed(name, fn):
    """Envuelve un nodo para registrar cuánto tarda (ms) en el campo node_ms del estado."""
    async def run(state):
        t0 = time.perf_counter()
        out = fn(state)
        if asyncio.iscoroutine(out):
            out = await out
        return {**out, "node_ms": {name: round((time.perf_counter() - t0) * 1000, 1)}}
    return run


def build_graph(filter_cache: dict, depth: int, with_generation: bool, use_filter_cache: bool = True):
    async def cached_relevance_filter(state: AblationState) -> dict:
        key = hashlib.sha1(state["user_query"].encode("utf-8")).hexdigest()
        t0 = time.perf_counter()
        if not use_filter_cache or key not in filter_cache:
            filter_cache[key] = (await relevance_filter_node(state))["is_relevant"]
        return {"is_relevant": filter_cache[key], "node_ms": {"relevance_filter": round((time.perf_counter() - t0) * 1000, 1)}}

    async def retrieve(state: AblationState) -> dict:
        spec = parse_config(state["config"])
        docs = await get_config_docs(state["config"], state["user_query"], state.get("reranker_backend", "local"))
        generation_docs, top1, n_above = docs["generation"], None, None
        if spec["kind"] == "dense" and spec["rerank_k"] is None:
            # configuración desplegada: al generador solo llegan los chunks sobre el umbral de similitud, igual que en producción
            top1 = atlas_score_to_cosine(docs["comparable"][0]["score"]) if docs["comparable"] else None
            generation_docs = apply_score_threshold(docs["generation"], settings.retrieval_score_threshold)
            n_above = len(generation_docs)
        return {"ranked_chunk_ids": [d["chunk_id"] for d in docs["comparable"][:depth]], "reranked_docs": generation_docs,
                "top1_cosine": top1, "n_above_threshold": n_above}

    g = (
        StateGraph(AblationState)
        .add_node("relevance_filter", cached_relevance_filter)
        .add_node("context_check", timed("context_check", context_check_node))
        .add_node("retrieve", timed("retrieve", retrieve))
        .add_node("out_of_domain", out_of_domain_node)
        .add_node("no_results", no_results_node)
        .add_edge(START, "relevance_filter")
        .add_conditional_edges("relevance_filter", route_relevance, ["context_check", "out_of_domain"])
        .add_edge("out_of_domain", END)
        .add_edge("no_results", END)
    )
    if with_generation:
        g = (
            g.add_node("generate", timed("generate", generate_node))
            .add_node("source_grounding", timed("source_grounding", source_grounding_node))
            .add_conditional_edges("context_check", lambda s: "generate" if route_context(s) == "generate" else "retrieve", ["retrieve", "generate"])
            .add_conditional_edges("retrieve", route_retrieval, ["generate", "no_results"])
            .add_edge("generate", "source_grounding")
            .add_edge("source_grounding", END)
        )
    else:  # solo hasta la recuperación (las configuraciones de la ablación sin generación)
        g = (
            g.add_conditional_edges("context_check", lambda s: "retrieve" if not s.get("context_sufficient") else END, ["retrieve", END])
            .add_edge("retrieve", END)
        )
    return g.compile()


async def run_config(config: str, dataset: list[dict], graph, concurrency: int, backend: str, with_generation: bool) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict) -> dict:
        async with sem:
            q = item.get("normalized_query") or item["query"]
            t0 = time.perf_counter()
            out = await graph.ainvoke({
                "session_id": item["dataset_id"], "user_query": q, "messages": [{"role": "user", "content": q}],
                "config": config, "reranker_backend": backend,
            })
            total_ms = round((time.perf_counter() - t0) * 1000, 1)
            row = {"dataset_id": item["dataset_id"], "is_relevant": bool(out.get("is_relevant")),
                   "context_sufficient": bool(out.get("context_sufficient")), "ranked": out.get("ranked_chunk_ids", []),
                   "top1_cosine": out.get("top1_cosine"), "n_above_threshold": out.get("n_above_threshold"),
                   "no_results": bool(out.get("no_results")), "latency_ms": total_ms, "node_ms": out.get("node_ms", {})}
            if with_generation:
                row.update({"answer": out.get("answer", ""), "sources": out.get("sources", []), "from_context": bool(out.get("from_context"))})
            return row

    return list(await asyncio.gather(*(one(it) for it in dataset)))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", required=True, type=config_arg)
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--with-generation", action="store_true", help="además de recuperar, genera la respuesta y fundamenta las fuentes")
    ap.add_argument("--reranker-backend", choices=["local", "openrouter"], default="local")
    ap.add_argument("--concurrency", type=int, default=5, help="usar 1 para medir latencias por respuesta sin interferencia entre consultas")
    ap.add_argument("--no-filter-cache", action="store_true", help="no leer ni escribir la caché del filtro de dominio (necesario para medir su latencia)")
    ap.add_argument("--model-service-dir", type=Path, default=Path(os.environ.get("MODEL_SERVICE_DIR", "../model-service")))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_path, args.limit)
    cache_path = args.out_dir / "filter_cache.json"
    filter_cache: dict = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    graph = build_graph(filter_cache, args.depth, args.with_generation, use_filter_cache=not args.no_filter_cache)

    for config in args.configs:
        variant = parse_config(config)["variant"]
        if variant is not None:
            await verify_index(variant)
        rows = await run_config(config, dataset, graph, args.concurrency, args.reranker_backend, args.with_generation)
        if not args.no_filter_cache:
            cache_path.write_text(json.dumps(filter_cache), encoding="utf-8")
        rejected = [r["dataset_id"] for r in rows if not r["is_relevant"]]
        (args.out_dir / f"{config}.retriever.json").write_text(
            json.dumps({r["dataset_id"]: r["ranked"] for r in rows if r["is_relevant"]}, ensure_ascii=False), encoding="utf-8")
        (args.out_dir / f"{config}.json").write_text(
            json.dumps({r["dataset_id"]: (r["ranked"] if r["is_relevant"] else []) for r in rows}, ensure_ascii=False), encoding="utf-8")
        with open(args.out_dir / f"{config}.trace.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if variant is not None:
            await verify_index(variant)  # el índice no cambió durante la corrida
            spec = parse_config(config)
            meta = {"config": config, **await model_meta(variant, args.model_service_dir),
                    "rrf": HYBRID_FUSION if spec["kind"] == "hybrid" else None, "rerank_k": spec["rerank_k"],
                    "reranker_backend": args.reranker_backend if spec["rerank_k"] else None, "with_generation": args.with_generation}
            (args.out_dir / f"{config}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        no_res = sum(r["no_results"] for r in rows)
        print(f"{config}: {len(rows)} preguntas | rechazadas por el filtro de dominio: {len(rejected)} | sin chunks sobre el umbral: {no_res} -> {args.out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
