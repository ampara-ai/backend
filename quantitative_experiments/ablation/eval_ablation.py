"""
Experimento de ablación — 5 configuraciones del pipeline RAG sobre el mismo
dataset y el mismo LLM generador que scripts/eval_dataset_ragas.py, para que
los resultados sean comparables entre sí (Objetivo Específico 4, dimensión
"Recuperación": Recall/MRR/NDCG/MAP @10 y @100 por configuración).

Configuraciones (nombre = <etapa>_<embedding>[_rerank<K>]):
  bm25_only                 : solo BM25.
  dense_<variante>          : solo denso con el embedding de la variante.
  hybrid_<variante>         : BM25 + denso, fusión RRF ponderada con parámetros FIJOS: peso del denso 2,0 (BM25 1,0) y
                              k=5. Se eligieron UNA vez en dev con el embedding base (tune_hybrid.py, rejilla en
                              hybrid_tuning_dev_base.csv) y se aplican sin reajustar a las dos variantes; el
                              RRF estándar (1:1, k=60) daba 0,271 de NDCG@10 estricto en dev frente a 0,378.
  <denso|hybrid>_<v>_rerank100 : las dos anteriores seguidas de un reranker que puntúa los 100 candidatos y los reordena
                              (se evalúa el ranking completo, no solo los 10 que se entregan: el reranker puede traer al
                              top-10 candidatos de las posiciones 11..100).
  Variantes de embedding: base (BGE-M3 sin entrenar, /embed_base) y finetuned (BGE-M3 entrenado con el dataset v2, /embed).

Comparabilidad:
  - Todas las configuraciones se puntúan con los mismos bloques "at10" y "at100" (recall/mrr/ndcg/map @10 y @100) sobre el
    ranking completo de la etapa. Con reranker, Recall@10 SÍ puede cambiar respecto a la etapa previa (entran candidatos de
    las posiciones 11..100); Recall@100 coincide con la etapa previa por construcción porque el reranker solo reordena
    dentro de esos 100 candidatos (MRR/NDCG@100 sí reflejan el reordenamiento).
  - Recall@k = aciertos / min(|relevantes|, k), igual que NDCG y MAP (misma normalización en las tres).
  - Con reranker se reporta además "post_rerank_top5" (informativo, no participa en la comparación).
  - faithfulness/answer_relevancy se calculan UNA vez por configuración sobre el contexto realmente usado para generar:
    el top-10 de la etapa sin reranker, el top-5 post-reranker en las configuraciones con reranker.

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
pasada. Las configuraciones con reranker le pasan los K primeros candidatos de la etapa previa y llaman directo a
model_service.rerank (local) o a la API de OpenRouter, sin pasar por rerank_node (que devuelve solo 5). Los puntajes del
reranker se guardan en una caché en disco (rerank_cache.jsonl) para no repetir pares (consulta, chunk) entre
configuraciones que comparten candidatos.

No modifica scripts/eval_dataset_ragas.py ni los nodos de app/graph/nodes/.

Dataset: questions_with_positive_context.jsonl (mismo que eval_dataset_ragas.py).
positive_chunk_ids puede tener entre 1 y ~4 elementos por consulta: todas las
métricas de este script tratan reference_ids como conjunto de tamaño variable.

Uso:
    uv run python -m quantitative_experiments.ablation.eval_ablation
    uv run python -m quantitative_experiments.ablation.eval_ablation --configs bm25_only dense_finetuned
    uv run python -m quantitative_experiments.ablation.eval_ablation --samples 20 --skip-ragas
    uv run python -m quantitative_experiments.ablation.eval_ablation --qrels <qrels.jsonl> --skip-ragas
"""

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import httpx
import pandas as pd
from openai import AsyncOpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from app.core.config import settings
from app.graph.nodes.generation import generate_node
from app.services import model_service, mongodb
from scripts.eval_dataset_ragas import (
    DATASET_PATH,
    filter_dataset,
    load_dataset,
)
from scripts.eval_dataset_ragas import mrr as _mrr_equivalent

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = EXPERIMENT_DIR / "results"  # sobrescribible con --out-dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_VARIANTS = ["base", "finetuned"]  # finetuned = el modelo entrenado con el dataset v2, servido en /embed
RERANK_CANDIDATE_GRID = (100,)  # el reranker puntúa los 100 candidatos que se evalúan; sin ajuste de K


def _build_config_names() -> list[str]:
    names = ["bm25_only"]
    for kind in ("dense", "hybrid"):
        for variant in EMBEDDING_VARIANTS:
            names.append(f"{kind}_{variant}")
            names += [f"{kind}_{variant}_rerank{k}" for k in RERANK_CANDIDATE_GRID]
    return names


CONFIG_NAMES = _build_config_names()
_CONFIG_RE = re.compile(r"^(dense|hybrid)_(" + "|".join(EMBEDDING_VARIANTS) + r")(?:_rerank(\d+))?$")


def parse_config(name: str) -> dict:
    """{"kind": bm25|dense|hybrid, "variant": str|None, "rerank_k": int|None}. Acepta cualquier K entre 1 y 100."""
    if name == "bm25_only":
        return {"kind": "bm25", "variant": None, "rerank_k": None}
    m = _CONFIG_RE.match(name)
    if not m:
        raise ValueError(f"Configuración desconocida: {name}")
    k = int(m.group(3)) if m.group(3) else None
    if k is not None and not 1 <= k <= CANDIDATE_POOL_K:
        raise ValueError(f"K del reranker fuera de rango (1..{CANDIDATE_POOL_K}): {name}")
    return {"kind": m.group(1), "variant": m.group(2), "rerank_k": k}


def config_arg(name: str) -> str:
    parse_config(name)
    return name


# Reranker alternativo vía API de OpenRouter (voyageai/rerank-2.5) — EXPLORATORIO,
# cuesta llamadas reales pagas a OpenRouter (a diferencia del reranker local
# ONNX del model service, que es gratis). Solo se usa en
# dense_finetuned_rerank_openrouter, no toca producción.
OPENROUTER_RERANK_URL_SUFFIX = "/rerank"
OPENROUTER_RERANK_MODEL = "voyageai/rerank-2.5"

CANDIDATE_POOL_K = 100  # candidatos reales pedidos a Mongo (bypass del cap de 20 en los nodos)
AT_K_VALUES = [10, 100]  # bloques comparables entre las configuraciones
GENERATION_K = 10        # contexto usado para generar + RAGAS en configs 1-4
RERANK_K = 5              # top-5 post-reranker: contexto de generación en las configuraciones con reranker (informativo)

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
# Recall@k, MAP@k y NDCG@k normalizan por min(|reference_ids|, k) (acotadas en [0,1] y comparables entre sí); MRR@k es el
# recíproco del primer acierto. reference_ids es un conjunto de tamaño variable (1..N).
# No se reutilizan context_recall/context_precision de eval_dataset_ragas.py: normalizan distinto (|G| y relevantes
# encontrados).

def recall_at_k(retrieved_ids: list[str], reference_ids: set[str], k: int) -> float:
    """Aciertos en el top-k / min(|reference_ids|, k): misma normalización que NDCG@k y MAP@k."""
    if not reference_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & reference_ids) / min(len(reference_ids), k)


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
# módulo). El reranker recibe solo el top-10 fusionado.

_RRF_K = 60  # misma constante que app/graph/nodes/rrf_fusion.py (valor por defecto)
# Parámetros de la fusión híbrida: elegidos una sola vez en dev con el embedding base (NDCG@10 estricto 0,378; el RRF 1:1 con k=60
# daba 0,271) y aplicados sin reajustar a todas las variantes. Ver tune_hybrid.py y hybrid_tuning_dev_base.csv.
HYBRID_FUSION = {"w_bm25": 1.0, "w_dense": 2.0, "rrf_k": 5}


def rrf_fuse_ids(bm25_ids: list[str], dense_ids: list[str], w_bm25: float, w_dense: float, rrf_k: int) -> list[str]:
    """Fusión RRF ponderada de dos rankings de ids: score = sum_w w / (rrf_k + rank + 1). Con w=1, k=60 coincide con
    rrf_fusion_node de producción (sin su corte a top-10)."""
    scores: dict[str, float] = {}
    for weight, ids in ((w_bm25, bm25_ids), (w_dense, dense_ids)):
        for rank, doc_id in enumerate(ids):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rrf_k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def _rrf_fuse(bm25_docs: list[dict], semantic_docs: list[dict], params: dict | None = None) -> list[dict]:
    """Versión sobre documentos de rrf_fuse_ids."""
    params = params or HYBRID_FUSION
    docs = {d["_id"]: d for d in [*bm25_docs, *semantic_docs]}
    order = rrf_fuse_ids([d["_id"] for d in bm25_docs], [d["_id"] for d in semantic_docs],
                         params["w_bm25"], params["w_dense"], params["rrf_k"])
    return [docs[doc_id] for doc_id in order]


# La colección de Mongo puede tener chunks ajenos al corpus congelado del experimento (p. ej. documentos de
# prueba subidos por la API de admin, cuyos embeddings no se regeneran con la ingesta del corpus). Se descartan
# de cada lista de candidatos ANTES de fusionar o reordenar; se piden FILTER_MARGIN candidatos extra para que,
# tras descartarlos, cada lista conserve CANDIDATE_POOL_K elementos.
CORPUS_PATH = EXPERIMENT_DIR / "data" / "inputs" / "document_chunks_all.jsonl"
FILTER_MARGIN = 30
_corpus_ids: set[str] | None = None


def _restrict_to_corpus(docs: list[dict]) -> list[dict]:
    global _corpus_ids
    if _corpus_ids is None:
        with open(CORPUS_PATH, encoding="utf-8") as f:
            _corpus_ids = {json.loads(line)["chunk_id"] for line in f if line.strip()}
    return [d for d in docs if d["chunk_id"] in _corpus_ids][:CANDIDATE_POOL_K]


async def _stage_bm25(query: str) -> list[dict]:
    return _restrict_to_corpus(await mongodb.bm25_search(query, k=CANDIDATE_POOL_K + FILTER_MARGIN))


# Variantes de embedding del experimento y el endpoint del model service que las sirve. Mongo guarda UN solo
# campo `embedding`: antes de correr una configuración hay que reindexar con la variante correspondiente
# (scripts/ingest_chunks.py --with-base-embeddings | --with-embeddings).
EMBED_ENDPOINTS = {
    "base": settings.embedding_base_endpoint,
    "finetuned": settings.embedding_endpoint,
}


async def _stage_dense(query: str, variant: str) -> list[dict]:
    embedding = await model_service.get_embedding(query, endpoint=EMBED_ENDPOINTS[variant])
    return _restrict_to_corpus(await mongodb.vector_search(embedding, k=CANDIDATE_POOL_K + FILTER_MARGIN, exact=True))


async def _stage_hybrid_fused(query: str, variant: str) -> list[dict]:
    embedding = await model_service.get_embedding(query, endpoint=EMBED_ENDPOINTS[variant])
    bm25_docs, semantic_docs = await asyncio.gather(
        mongodb.bm25_search(query, k=CANDIDATE_POOL_K + FILTER_MARGIN),
        mongodb.vector_search(embedding, k=CANDIDATE_POOL_K + FILTER_MARGIN, exact=True),
    )
    return _rrf_fuse(_restrict_to_corpus(bm25_docs), _restrict_to_corpus(semantic_docs), HYBRID_FUSION)


RERANK_CACHE_PATH = EXPERIMENT_DIR / "data" / "rerank_cache.jsonl"
RERANK_LOCAL_BATCH = 20  # documentos por petición al reranker local (evita el timeout de 30 s de model_service.rerank)
_rerank_cache: dict[str, float] | None = None
_rerank_cache_lock: asyncio.Lock | None = None


def _rerank_key(backend: str, query: str, doc: dict) -> str:
    text_hash = hashlib.md5(doc["text"].encode("utf-8")).hexdigest()[:12]
    return f"{backend}|{hashlib.md5(query.encode('utf-8')).hexdigest()[:12]}|{doc['chunk_id']}|{text_hash}"


def _load_rerank_cache() -> dict[str, float]:
    global _rerank_cache
    if _rerank_cache is None:
        _rerank_cache = {}
        if RERANK_CACHE_PATH.exists():
            for line in RERANK_CACHE_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    _rerank_cache[row["key"]] = row["score"]
    return _rerank_cache


async def _score_with_reranker(query: str, docs: list[dict], backend: str) -> dict[int, float]:
    """Puntaje del reranker para cada documento (índice -> score); pide a la API solo los pares sin caché."""
    global _rerank_cache_lock
    cache = _load_rerank_cache()
    if _rerank_cache_lock is None:
        _rerank_cache_lock = asyncio.Lock()
    keys = [_rerank_key(backend, query, d) for d in docs]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    fresh: dict[int, float] = {}
    if backend == "openrouter":
        if missing:
            ranked = await _rerank_openrouter(query, [docs[i] for i in missing], len(missing))
            by_id = {d["chunk_id"]: d["rerank_score"] for d in ranked}
            fresh = {i: by_id[docs[i]["chunk_id"]] for i in missing}
    else:
        for start in range(0, len(missing), RERANK_LOCAL_BATCH):
            part = missing[start:start + RERANK_LOCAL_BATCH]
            for item in await model_service.rerank(query, [docs[i]["text"] for i in part]):
                fresh[part[item["index"]]] = item["score"]
    if fresh:
        async with _rerank_cache_lock:
            RERANK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RERANK_CACHE_PATH, "a", encoding="utf-8") as f:
                for i, score in fresh.items():
                    cache[keys[i]] = score
                    f.write(json.dumps({"key": keys[i], "score": score}) + "\n")
    return {i: cache[keys[i]] for i in range(len(docs))}


async def _stage_rerank(query: str, base_docs: list[dict], k: int, backend: str = "local") -> tuple[list[dict], list[dict]]:
    """Reordena los K primeros candidatos de `base_docs` con el reranker y deja el resto en su orden original.

    Retorna (comparable, reranked_top5): `comparable` = los K reordenados + base_docs[K:] (para los bloques at10/at100);
    `reranked_top5` = los 5 primeros reordenados (contexto de generación). Con K > 10 el reranker puede subir al top-10
    candidatos de las posiciones 11..K, de modo que Recall@10 deja de quedar fijo por construcción; Recall@100 sigue
    coincidiendo con el de la etapa previa (el reranker solo reordena dentro de los primeros K <= 100)."""
    candidates = base_docs[:k]
    scores = await _score_with_reranker(query, candidates, backend)
    order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)  # estable: empates conservan el orden previo
    reranked = [{**candidates[i], "rerank_score": scores[i]} for i in order]
    return reranked + base_docs[k:], reranked[:RERANK_K]


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


async def get_config_docs(config: str, query: str, reranker_backend: str = "local") -> dict:
    """Retorna {"comparable": ranking completo (>=100 cuando hay suficientes candidatos) para los bloques at10/at100,
    "generation": docs usados para generar + RAGAS (top-10 de la etapa, o top-5 post-reranker), "post_rerank": top-5
    informativo (solo configuraciones con reranker, si no None)}.
    reranker_backend: "local" = reranker ONNX del model service, "openrouter" = Voyage vía API (de pago)."""
    spec = parse_config(config)
    if spec["kind"] == "bm25":
        docs = await _stage_bm25(query)
    elif spec["kind"] == "dense":
        docs = await _stage_dense(query, spec["variant"])
    else:
        docs = await _stage_hybrid_fused(query, spec["variant"])
    if spec["rerank_k"] is None:
        return {"comparable": docs, "generation": docs[:GENERATION_K], "post_rerank": None}
    comparable, top5 = await _stage_rerank(query, docs, spec["rerank_k"], reranker_backend)
    return {"comparable": comparable, "generation": top5, "post_rerank": top5}


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
    query: str = item.get("normalized_query") or item["query"]
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
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Experimento de ablación del pipeline RAG")
    parser.add_argument("--configs", nargs="+", type=config_arg, default=["bm25_only", "dense_finetuned", "hybrid_finetuned"],
        help="Configuraciones a correr, p. ej. dense_finetuned hybrid_finetuned_rerank50 (ver docstring del módulo)")
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
             f"quantitative_experiments/unseen_dataset/outputs/unseen_eval_dataset.jsonl "
             f"(chunks nunca vistos en el entrenamiento del embedding).")
    parser.add_argument("--qrels", type=Path, default=None,
        help="JSONL {qid, docid, relevance}: reemplaza los positive_chunk_ids del dataset por "
             "los docid con relevance>0 de cada dataset_id (p. ej. qrels ampliados por pooling).")
    parser.add_argument("--out-dir", type=Path, default=None,
        help=f"Carpeta de salida (default: {OUT_DIR}). Usar una por variante de qrels.")
    parser.add_argument("--reranker-backend", choices=["local", "openrouter"], default="local",
        help="Reranker de las configuraciones *_rerank<K>: 'local' (ONNX, gratis) u 'openrouter' "
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
    if args.out_dir is not None:
        OUT_DIR = args.out_dir
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.qrels is not None:
        qrels: dict[str, set[str]] = {}
        with open(args.qrels, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("relevance", 1) > 0:
                        qrels.setdefault(r["qid"], set()).add(r["docid"])
        for it in dataset:
            it["positive_chunk_ids"] = sorted(qrels[it["dataset_id"]])
        print(f"Positivos reemplazados desde {args.qrels.name}: "
              f"{sum(len(it['positive_chunk_ids']) for it in dataset)} pares para {len(dataset)} preguntas")
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
    if any("_rerank" in c for c in args.configs):
        print(f"configuraciones con reranker: reranker_backend={args.reranker_backend}")
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

    print("\n=== Comparación de ablación (bloques at10/at100 = comparación válida entre configuraciones) ===")
    cols_main = ["config", "n_queries"] + [f"mean_{c}" for c in AT_K_METRIC_COLS + ["faithfulness", "answer_relevancy"]]
    print(comparison[cols_main].to_string(index=False))
    print("\n--- Bloque post_rerank_top5 (informativo, solo configuraciones con reranker) ---")
    cols_post = ["config"] + [f"mean_{c}" for c in POST_RERANK_METRIC_COLS]
    print(comparison[cols_post].to_string(index=False))
    print(f"\nTabla comparativa guardada en {comp_path}")


if __name__ == "__main__":
    asyncio.run(main())
