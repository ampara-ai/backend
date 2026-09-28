"""
Experimento de ablación — 5 configuraciones del pipeline RAG sobre el mismo
dataset y el mismo LLM generador que scripts/eval_dataset_ragas.py, para que
los resultados sean comparables entre sí (Objetivo Específico 4, dimensión
"Recuperación": Recall/MRR/NDCG/MAP @10 y @100 por configuración).

Configuraciones:
  1. bm25_only          : solo BM25.
  2. dense_base         : solo denso con embedding base BGE-M3 (/embed_base).
  3. dense_finetuned    : solo denso con embedding ajustado BGE-M3 (/embed).
     Esta es la arquitectura final de recuperación (decisión tomada tras este
     experimento): le ganó a todas las combinaciones probadas (RRF sin peso,
     RRF pesado hacia el embedding, reranker actual sobre el fusionado, reranker
     actual sobre el denso solo, reranker Qwen, interpolación de scores con α
     elegido en validación, y reformulación jurídica de la consulta) — ver
     resultados en results/.
  4. hybrid_no_rerank   : bm25 + denso (finetuned) → fusión RRF sin peso (1:1), sin reranker.
  5. hybrid_rerank      : bm25 + denso → RRF sin peso → rerank (la arquitectura
     que se consideró y se descartó como final; queda como punto de
     comparación del experimento de ablación, no como candidata a producción).

Comparabilidad:
  - Para las 5 configuraciones se reportan dos bloques comparables: "at10" y
    "at100" (recall/mrr/ndcg/map @10 y @100), calculados sobre el ranking
    completo de esa etapa. Para hybrid_rerank ese ranking es el fusionado
    PRE-reranker (no el top-5 real), para que sea comparable en igualdad de
    condiciones con las demás.
  - hybrid_rerank reporta además un bloque "post_rerank_top5", informativo,
    calculado sobre su top-5 real (queda vacío para las demás). Este bloque NO
    participa en la comparación entre configuraciones.
  - faithfulness/answer_relevancy se calculan UNA sola vez por configuración,
    sobre el contexto realmente usado para generar la respuesta: el top-10 de
    la etapa en configs 1-4, el top-5 post-reranker en 5 y 6.

Nota sobre @100 y los nodos de producción: bm25_search_node y
semantic_search_node (app/graph/nodes/retrieval.py) llaman a
mongodb.bm25_search/vector_search con k=20 fijo como literal — no aceptan k
desde el state. rrf_fusion_node (app/graph/nodes/rrf_fusion.py) además
recorta el fusionado a top-10 fijo (_TOP_N=10). Ninguno de los dos permite
pedir un top-100. Para no modificar esos nodos (decisión confirmada), este
script llama directo a mongodb.bm25_search/vector_search con k=100 (esas
funciones de servicio sí aceptan k) y reimplementa la fórmula de RRF
localmente (_rrf_fuse, misma constante _RRF_K=60 que
app/graph/nodes/rrf_fusion.py) sin el corte fijo a 10 — así se obtiene un
único ranking del que se derivan tanto el top-10 como el top-100 de una sola
pasada. Sí se sigue llamando a rerank_node (nodo real) para hybrid_rerank,
pero alimentado solo con el top-10 fusionado — igual que hubiera sido en
producción, no con los 100 candidatos.

No modifica scripts/eval_dataset_ragas.py ni los nodos de app/graph/nodes/.

Dataset: questions_with_positive_context.jsonl (mismo que eval_dataset_ragas.py).
positive_chunk_ids puede tener entre 1 y ~4 elementos por consulta: todas las
métricas de este script tratan reference_ids como conjunto de tamaño variable.

Uso:
    uv run python scripts/eval_ablation.py
    uv run python scripts/eval_ablation.py --configs bm25_only dense_finetuned
    uv run python scripts/eval_ablation.py --samples 20 --skip-ragas
    uv run python scripts/eval_ablation.py --filter-single --min-confidence 0.9
"""

import argparse
import asyncio
import math
import sys
from pathlib import Path

import httpx
import pandas as pd
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from app.core.config import settings
from app.graph.nodes.generation import generate_node
from app.graph.nodes.reranker import rerank_node
from app.services import model_service, mongodb
from scripts.eval_dataset_ragas import (
    DATASET_PATH,
    context_recall,
    filter_dataset,
    load_dataset,
)
from scripts.eval_dataset_ragas import mrr as _mrr_equivalent

OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAMES = [
    "bm25_only",
    "dense_base",
    "dense_finetuned",
    "hybrid_no_rerank",
    "hybrid_rerank",
    "dense_finetuned_rerank_openrouter",
]

# Reranker alternativo vía API de OpenRouter (voyageai/rerank-2.5) — EXPLORATORIO,
# cuesta llamadas reales pagas a OpenRouter (a diferencia del reranker local
# ONNX del model service, que es gratis). Solo se usa en
# dense_finetuned_rerank_openrouter, no toca producción.
OPENROUTER_RERANK_URL_SUFFIX = "/rerank"
OPENROUTER_RERANK_MODEL = "voyageai/rerank-2.5"

CANDIDATE_POOL_K = 100  # candidatos reales pedidos a Mongo (bypass del cap de 20 en los nodos)
AT_K_VALUES = [10, 100]  # bloques comparables entre las configuraciones
GENERATION_K = 10        # contexto usado para generar + RAGAS en configs 1-4
RERANK_K = 5              # corte real de producción, solo informativo en hybrid_rerank

# AnswerRelevancy usa settings.ragas_realtime_embeddings_model (qwen3-embedding-8b
# vía API de OpenRouter, el mismo que app/services/ragas_eval.py usa para la
# evaluación RAGAS en vivo) — NO el embedding local (nomic/sentence-transformers),
# que corriendo en MPS (Apple Silicon) revienta con un assertion de Metal
# ("MTLCommandBufferStatusCommitted") al llamarlo desde varias tareas de asyncio
# a la vez. Al ser puro HTTP, soporta concurrencia real sin necesitar lock.

AT_K_METRIC_COLS = [f"{m}_at{k}" for k in AT_K_VALUES for m in ("recall", "mrr", "ndcg", "map")]
POST_RERANK_METRIC_COLS = [
    "recall_post_rerank_top5",
    "mrr_post_rerank_top5",
    "ndcg_post_rerank_top5",
    "map_post_rerank_top5",
]
CSV_COLS = (
    ["dataset_id", "query", "n_positives"]
    + AT_K_METRIC_COLS
    + ["faithfulness", "answer_relevancy"]
    + POST_RERANK_METRIC_COLS
    + ["topic", "barrier", "model_variant", "error"]
)


# ---------------------------------------------------------------------------
# Métricas de retrieval — reference_ids como conjunto de tamaño variable (1..N)
# ---------------------------------------------------------------------------
# recall@k y mrr@k reutilizan context_recall/mrr de eval_dataset_ragas.py: ya
# operan sobre reference_ids de tamaño variable y cumplen exactamente la
# definición pedida (basta con pasarles la lista ya truncada a top-k).
# map@k y ndcg@k NO se reutilizan de eval_dataset_ragas.py: su context_precision
# normaliza por el total de relevantes encontrados, no por min(|reference|, k)
# como se pide aquí — se implementan nuevas.

def recall_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    return context_recall(retrieved_ids[:k], reference_ids)


def mrr_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    return _mrr_equivalent(retrieved_ids[:k], reference_ids)


def map_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    """Average Precision@k, normalizada por min(|reference_ids|, k) — acotada en [0,1]
    incluso si no se recuperan todos los positivos."""
    normalizer = min(len(reference_ids), k)
    if normalizer == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits, precision_sum = 0, 0.0
    for i, cid in enumerate(top_k, 1):
        if cid in reference_ids:
            hits += 1
            precision_sum += hits / i
    return precision_sum / normalizer


def ndcg_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    """NDCG@k con ganancia binaria y descuento log2 estándar. IDCG@k usa el número
    real de relevantes de la consulta (min(|reference_ids|, k)), no asume un solo
    positivo."""
    if not reference_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(i + 1) for i, cid in enumerate(top_k, 1) if cid in reference_ids
    )
    ideal_hits = min(len(reference_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _score_docs(docs: list[dict], reference_ids: set[str], k: int) -> dict[str, float]:
    ids = [d["chunk_id"] for d in docs]
    return {
        "recall": recall_at_k(ids, reference_ids, k),
        "mrr": mrr_at_k(ids, reference_ids, k),
        "ndcg": ndcg_at_k(ids, reference_ids, k),
        "map": map_at_k(ids, reference_ids, k),
    }


def _score_docs_multi_k(docs: list[dict], reference_ids: set[str]) -> dict[str, float]:
    """_score_docs para cada k en AT_K_VALUES, aplanado a columnas tipo recall_at10."""
    out: dict[str, float] = {}
    for k in AT_K_VALUES:
        scores = _score_docs(docs, reference_ids, k)
        for metric, value in scores.items():
            out[f"{metric}_at{k}"] = value
    return out


# ---------------------------------------------------------------------------
# Etapas del pipeline por configuración
# ---------------------------------------------------------------------------
# bm25/denso/RRF bypasean bm25_search_node/semantic_search_node/rrf_fusion_node
# (piden k=100 vía mongodb.py directo + RRF local sin corte a 10 — ver nota de
# módulo). rerank_node SÍ se llama tal cual, con el top-10 fusionado.

_RRF_K = 60  # misma constante que app/graph/nodes/rrf_fusion.py


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def _rrf_fuse(bm25_docs: list[dict], semantic_docs: list[dict]) -> list[dict]:
    """Reimplementación local de la fórmula de rrf_fusion_node, sin el corte
    fijo a top-10, para poder derivar top-10 y top-100 fusionados de una sola
    pasada sobre el mismo ranking. Idéntica matemáticamente a rrf_fusion_node."""
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for rank, doc in enumerate(bm25_docs):
        doc_id = doc["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        docs[doc_id] = doc
    for rank, doc in enumerate(semantic_docs):
        doc_id = doc["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
        docs[doc_id] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**docs[doc_id], "rrf_score": score} for doc_id, score in ranked]


async def _stage_bm25(query: str) -> list[dict]:
    return await mongodb.bm25_search(query, k=CANDIDATE_POOL_K)


async def _stage_dense(query: str, use_base: bool) -> list[dict]:
    embedding = await model_service.get_embedding(query, use_base=use_base)
    return await mongodb.vector_search(embedding, k=CANDIDATE_POOL_K)


async def _stage_hybrid_fused(query: str) -> list[dict]:
    embedding = await model_service.get_embedding(query, use_base=False)
    bm25_docs, semantic_docs = await asyncio.gather(
        mongodb.bm25_search(query, k=CANDIDATE_POOL_K),
        mongodb.vector_search(embedding, k=CANDIDATE_POOL_K),
    )
    return _rrf_fuse(bm25_docs, semantic_docs)  # ranking completo, ordenado desc


async def _stage_hybrid_rerank(query: str, reranker_backend: str = "local") -> tuple[list[dict], list[dict]]:
    fused_docs = await _stage_hybrid_fused(query)          # para bloques at10/at100
    fused_top10 = fused_docs[:GENERATION_K]                 # lo que rrf_fusion_node daría
    if reranker_backend == "openrouter":
        # Solo para pruebas: el reranker local ONNX es lento. Bypasea
        # rerank_node (que no acepta backend alternativo) y llama directo a
        # la API de OpenRouter, misma f_stage que dense_finetuned_rerank_openrouter.
        reranked_docs = await _rerank_openrouter(query, fused_top10, RERANK_K)
    else:
        reranked_state = await rerank_node({
            "fused_results": fused_top10,
            "synthesized_query": query,
        })
        reranked_docs = reranked_state.get("reranked_docs") or []
    return fused_docs, reranked_docs


async def _rerank_openrouter(query: str, docs: list[dict], top_n: int) -> list[dict]:
    """Reranker alternativo vía API de OpenRouter (llamada real, paga —
    distinta de model_service.rerank(), que es local y gratis)."""
    if not docs:
        return []
    texts = [d["text"] for d in docs]
    url = settings.openrouter_base_url.rstrip("/") + OPENROUTER_RERANK_URL_SUFFIX
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_RERANK_MODEL,
                "query": query,
                "documents": texts,
                "top_n": top_n,
            },
        )
        response.raise_for_status()
        results = response.json()["results"]
    return [{**docs[item["index"]], "rerank_score": item["relevance_score"]} for item in results]


async def _stage_dense_rerank_openrouter(query: str) -> tuple[list[dict], list[dict]]:
    """Denso finetuneado solo (sin BM25/RRF) → reranker de OpenRouter
    (voyageai/rerank-2.5), en vez del reranker local ONNX."""
    dense_docs = await _stage_dense(query, use_base=False)  # para bloques at10/at100
    dense_top10 = dense_docs[:GENERATION_K]
    reranked_docs = await _rerank_openrouter(query, dense_top10, RERANK_K)
    return dense_docs, reranked_docs


async def get_config_docs(config: str, query: str, reranker_backend: str = "local") -> dict:
    """Retorna {"comparable": ranking completo (≥100 cuando hay suficientes
    candidatos) para los bloques at10/at100, "generation": docs usados para
    generar + RAGAS (top-10, o top-5 post-rerank en hybrid_rerank),
    "post_rerank": docs top-5 informativos (solo hybrid_rerank, si no None)}.
    reranker_backend solo afecta a hybrid_rerank ("local" = rerank_node ONNX,
    "openrouter" = Voyage/Cohere vía API, más rápido para pruebas)."""
    if config == "bm25_only":
        docs = await _stage_bm25(query)
        return {"comparable": docs, "generation": docs[:GENERATION_K], "post_rerank": None}
    if config == "dense_base":
        docs = await _stage_dense(query, use_base=True)
        return {"comparable": docs, "generation": docs[:GENERATION_K], "post_rerank": None}
    if config == "dense_finetuned":
        docs = await _stage_dense(query, use_base=False)
        return {"comparable": docs, "generation": docs[:GENERATION_K], "post_rerank": None}
    if config == "hybrid_no_rerank":
        docs = await _stage_hybrid_fused(query)
        return {"comparable": docs, "generation": docs[:GENERATION_K], "post_rerank": None}
    if config == "hybrid_rerank":
        fused_docs, reranked_docs = await _stage_hybrid_rerank(query, reranker_backend=reranker_backend)
        return {"comparable": fused_docs, "generation": reranked_docs, "post_rerank": reranked_docs}
    if config == "dense_finetuned_rerank_openrouter":
        dense_docs, reranked_docs = await _stage_dense_rerank_openrouter(query)
        return {"comparable": dense_docs, "generation": reranked_docs, "post_rerank": reranked_docs}
    raise ValueError(f"Configuración desconocida: {config}")


# ---------------------------------------------------------------------------
# Evaluación de una configuración sobre todo el dataset
# ---------------------------------------------------------------------------

async def _evaluate_item(
    item: dict,
    idx: int,
    total: int,
    config: str,
    faithfulness_metric,
    relevancy_metric,
    skip_ragas: bool,
    reranker_backend: str,
) -> dict:
    """Evalúa una sola consulta — extraído de evaluate_config para poder
    correr muchas en paralelo (asyncio.gather + semáforo). Sin estado
    compartido mutable: get_llm() crea un cliente nuevo por llamada, y las
    métricas RAGAS ya se usan así en su propio runner interno."""
    query: str = item["normalized_query"]
    reference_ids: set[str] = set(item.get("positive_chunk_ids", []))
    n_positives = len(reference_ids)

    try:
        docs = await get_config_docs(config, query, reranker_backend=reranker_backend)
    except Exception as exc:
        print(f"  [{config}] [{idx + 1}/{total}] ERROR en pipeline: {exc}")
        return {
            "dataset_id": item.get("dataset_id"),
            "query": query,
            "n_positives": n_positives,
            **{col: None for col in AT_K_METRIC_COLS},
            "faithfulness": None,
            "answer_relevancy": None,
            **{col: None for col in POST_RERANK_METRIC_COLS},
            "topic": item.get("topic"),
            "barrier": item.get("barrier"),
            "model_variant": config,
            "error": str(exc),
        }

    at_k_row = _score_docs_multi_k(docs["comparable"], reference_ids)

    post_row = {col: None for col in POST_RERANK_METRIC_COLS}
    if docs["post_rerank"] is not None:
        post = _score_docs(docs["post_rerank"], reference_ids, RERANK_K)
        post_row = {
            "recall_post_rerank_top5": post["recall"],
            "mrr_post_rerank_top5": post["mrr"],
            "ndcg_post_rerank_top5": post["ndcg"],
            "map_post_rerank_top5": post["map"],
        }

    generation_docs = docs["generation"]
    answer = ""
    f_val, r_val = None, None

    # skip_ragas evita también la generación: sin juez que la evalúe, no
    # hay razón para pagar/esperar la llamada real al LLM generador.
    if not skip_ragas and generation_docs:
        gen_state = await generate_node({
            "reranked_docs": generation_docs,
            "user_query": query,
            "synthesized_query": query,
            "messages": [{"role": "user", "content": query}],
            "context_sufficient": False,
        })
        answer = gen_state.get("answer", "")

    if not skip_ragas and generation_docs and answer:
        retrieved_texts = [d["text"] for d in generation_docs]
        try:
            f_result = await faithfulness_metric.ascore(
                user_input=query, response=answer, retrieved_contexts=retrieved_texts,
            )
            f_val = f_result.value
        except Exception as exc:
            print(f"    faithfulness error: {exc}")
        try:
            r_result = await relevancy_metric.ascore(user_input=query, response=answer)
            r_val = r_result.value
        except Exception as exc:
            print(f"    answer_relevancy error: {exc}")

    row = {
        "dataset_id": item.get("dataset_id"),
        "query": query,
        "n_positives": n_positives,
        **at_k_row,
        "faithfulness": f_val,
        "answer_relevancy": r_val,
        **post_row,
        "topic": item.get("topic"),
        "barrier": item.get("barrier"),
        "model_variant": config,
        "error": None,
    }

    faith_str = f"{f_val:.3f}" if f_val is not None else "n/a"
    rel_str = f"{r_val:.3f}" if r_val is not None else "n/a"
    print(
        f"  [{config}] [{idx + 1}/{total}] at10: recall={at_k_row['recall_at10']:.3f} "
        f"mrr={at_k_row['mrr_at10']:.3f} ndcg={at_k_row['ndcg_at10']:.3f} map={at_k_row['map_at10']:.3f} "
        f"| faith={faith_str} rel={rel_str} | {query[:60]}"
    )
    return row


async def evaluate_config(
    dataset: list[dict],
    config: str,
    evaluator_llm,
    evaluator_embeddings,
    skip_ragas: bool,
    reranker_backend: str = "local",
    concurrency: int = 1,
) -> pd.DataFrame:
    faithfulness_metric = None if skip_ragas else Faithfulness(llm=evaluator_llm)
    relevancy_metric = (
        None if skip_ragas else AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(item: dict, idx: int) -> dict:
        async with semaphore:
            return await _evaluate_item(
                item, idx, len(dataset), config,
                faithfulness_metric, relevancy_metric, skip_ragas, reranker_backend,
            )

    tasks = [bounded(item, i) for i, item in enumerate(dataset)]
    rows = await asyncio.gather(*tasks)

    return pd.DataFrame(rows, columns=CSV_COLS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento de ablación del pipeline RAG")
    parser.add_argument("--configs", nargs="+", choices=CONFIG_NAMES, default=CONFIG_NAMES,
        help="Configuraciones a correr (default: las 5)")
    parser.add_argument("--samples", type=int, default=100,
        help="Número de ejemplos a cargar del dataset (default: 100)")
    parser.add_argument("--filter-single", action="store_true",
        help="Evaluar solo ejemplos con 1 positive_chunk_id")
    parser.add_argument("--min-confidence", type=float, default=0.0,
        help="Filtrar ejemplos por confidence mínima (ej: 0.9)")
    parser.add_argument("--skip-ragas", action="store_true",
        help="Solo métricas de recuperación, sin llamadas al LLM generador ni al juez RAGAS")
    parser.add_argument("--dataset-path", type=Path, default=DATASET_PATH,
        help=f"Archivo JSONL a evaluar (default: {DATASET_PATH.name}). Usar "
             f"datasets/test90_clean.jsonl para el set de test genuino, sin fuga "
             f"de las preguntas de entrenamiento del embedding.")
    parser.add_argument("--reranker-backend", choices=["local", "openrouter"], default="local",
        help="Reranker que usa hybrid_rerank: 'local' (ONNX, lento) u 'openrouter' "
             f"(API, más rápido — modelo: {OPENROUTER_RERANK_MODEL})")
    parser.add_argument("--concurrency", type=int, default=1,
        help="Consultas en paralelo por configuración (default: 1 — secuencial). "
             "Con RAGAS activado esto es solo llamadas a OpenRouter (soporta concurrencia real, "
             "a diferencia del model service local); sin reranker de por medio no hay motivo para "
             "correr secuencial.")
    parser.add_argument("--resume", action="store_true",
        help="Si ya existe results/resultados_ablation_<config>.csv, no lo sobrescribe: "
             "salta las consultas (por dataset_id) que ya tienen resultado sin error, "
             "corre solo las que faltan, y junta todo en el mismo CSV.")
    args = parser.parse_args()

    print(f"Cargando {args.samples} ejemplos de {args.dataset_path.name}...")
    dataset = load_dataset(args.dataset_path, args.samples)
    dataset = filter_dataset(
        dataset, filter_single=args.filter_single, min_confidence=args.min_confidence,
    )

    filters_desc = []
    if args.filter_single:
        filters_desc.append("single-positive")
    if args.min_confidence > 0.0:
        filters_desc.append(f"confidence≥{args.min_confidence}")
    filter_label = f" [{', '.join(filters_desc)}]" if filters_desc else ""
    print(f"Ejemplos a evaluar: {len(dataset)}{filter_label}")
    print(f"Configuraciones: {', '.join(args.configs)}")
    if "hybrid_rerank" in args.configs:
        print(f"hybrid_rerank usa reranker_backend={args.reranker_backend}")
    print()

    evaluator_llm, evaluator_embeddings = None, None
    if not args.skip_ragas:
        async_client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        evaluator_llm = llm_factory(
            settings.ragas_llm_model, provider="openai", client=async_client, max_tokens=10000,
        )
        evaluator_embeddings = OpenAIEmbeddings(
            client=async_client, model=settings.ragas_realtime_embeddings_model,
        )

    summaries = []
    for config in args.configs:
        print(f"=== Configuración: {config} ===")

        out_csv = OUT_DIR / f"resultados_ablation_{config}.csv"
        existing_df = None
        pending_dataset = dataset
        if args.resume and out_csv.exists():
            existing_df = pd.read_csv(out_csv)
            # "hecha" = sin error Y con recall_at10 real — algunas excepciones
            # stringifican vacío (str(exc) == ""), que pandas lee como NaN
            # igual que "sin error": basta con "error is NaN" para que una
            # fila rota (todas las métricas None) quede marcada como "ya
            # hecha" para siempre y nunca se reintente.
            done_ids = set(existing_df.loc[
                existing_df["error"].isna() & existing_df["recall_at10"].notna(), "dataset_id"
            ])
            pending_dataset = [item for item in dataset if item.get("dataset_id") not in done_ids]
            print(f"--resume: {len(done_ids)} ya hechas en {out_csv.name}, "
                  f"faltan {len(pending_dataset)}/{len(dataset)}")

        if pending_dataset:
            new_df = await evaluate_config(
                pending_dataset, config, evaluator_llm, evaluator_embeddings, args.skip_ragas,
                reranker_backend=args.reranker_backend, concurrency=args.concurrency,
            )
        else:
            new_df = pd.DataFrame(columns=CSV_COLS)

        if existing_df is not None:
            # las nuevas reemplazan cualquier fila vieja con el mismo dataset_id
            # (ej. reintentos de filas con error) en vez de duplicarla
            new_ids = set(new_df["dataset_id"]) if len(new_df) else set()
            existing_df = existing_df[~existing_df["dataset_id"].isin(new_ids)]
            df = pd.concat([existing_df, new_df], ignore_index=True)[CSV_COLS]
        else:
            df = new_df

        df.to_csv(out_csv, index=False)
        print(f"CSV guardado en {out_csv} ({len(df)} filas totales)\n")

        summary = {"config": config, "n_queries": len(df)}
        for col in AT_K_METRIC_COLS + ["faithfulness", "answer_relevancy"] + POST_RERANK_METRIC_COLS:
            summary[f"mean_{col}"] = df[col].mean()
        summaries.append(summary)

    comparison = pd.DataFrame(summaries)
    comp_path = OUT_DIR / "comparacion_ablation.csv"
    comparison.to_csv(comp_path, index=False)

    print("\n=== Comparación de ablación (bloques at10/at100 = comparación válida entre las 5) ===")
    cols_main = ["config", "n_queries"] + [f"mean_{c}" for c in AT_K_METRIC_COLS + ["faithfulness", "answer_relevancy"]]
    print(comparison[cols_main].to_string(index=False))
    print("\n--- Bloque post_rerank_top5 (informativo, solo hybrid_rerank) ---")
    cols_post = ["config"] + [f"mean_{c}" for c in POST_RERANK_METRIC_COLS]
    print(comparison[cols_post].to_string(index=False))
    print(f"\nTabla comparativa guardada en {comp_path}")


if __name__ == "__main__":
    asyncio.run(main())
