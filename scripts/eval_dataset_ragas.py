"""
Evaluación RAGAS del sistema RAG completo — un modelo a la vez.

Pipeline evaluado (arquitectura real del RAG, sin routing):
  normalized_query
    → embed             (/embed finetuned  |  /embed_base base)
    → bm25 + semantic   (en paralelo)
    → RRF fusion        → métricas PRE-reranker (retrieval stage)
    → rerank            → métricas POST-reranker (context window stage)
    → generate

Métricas retrieval (chunk_id exacto):
  - context_recall_pre      : recall sobre RRF top-10
  - hit_rate_pre            : ≥1 positive en top-10 (binario)
  - mrr_pre                 : Mean Reciprocal Rank en top-10
  - context_precision_pre   : AP@k sobre RRF top-10
  - context_precision_cond  : AP@k solo sobre queries donde recall_pre > 0
  - context_recall_post     : recall sobre reranker top-5
  - context_precision_post  : AP@k sobre reranker top-5

Métricas LLM-based (RAGAS):
  - faithfulness       : RAGAS (Qwen3.7-plus via OpenRouter)
  - answer_relevancy   : RAGAS (Qwen3.7-plus + nomic-embed via OpenRouter)

Nota: el threshold del reranker (0.5) es un guardia de UX en producción.
Para medir retrieval quality se evalúa sobre todos los docs rerankeados
independientemente del score.

Dataset: questions_with_positive_context.jsonl

Uso:
    uv run python scripts/eval_dataset_ragas.py --model finetuned
    uv run python scripts/eval_dataset_ragas.py --model finetuned --filter-single
    uv run python scripts/eval_dataset_ragas.py --model finetuned --filter-single --min-confidence 0.9
    uv run python scripts/eval_dataset_ragas.py --model finetuned --samples 20
    uv run python scripts/eval_dataset_ragas.py --benchmark
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from app.core.config import settings
from app.graph.nodes.generation import generate_node
from app.graph.nodes.reranker import rerank_node
from app.graph.nodes.retrieval import bm25_search_node, semantic_search_node
from app.graph.nodes.rrf_fusion import rrf_fusion_node
from app.services import model_service

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parents[1] / "questions_with_positive_context.jsonl"
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRIC_COLS = [
    "recall_candidates",
    "hit_rate_candidates",
    "context_recall_pre",
    "hit_rate_pre",
    "mrr_pre",
    "context_precision_pre",
    "context_precision_cond",
    "context_recall_post",
    "context_precision_post",
    "faithfulness",
    "answer_relevancy",
]


# ---------------------------------------------------------------------------
# Carga y filtrado del dataset
# ---------------------------------------------------------------------------

def load_dataset(path: Path, n: int) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


def filter_dataset(
    items: list[dict],
    filter_single: bool,
    min_confidence: float,
) -> list[dict]:
    result = items
    if filter_single:
        result = [i for i in result if len(i.get("positive_chunk_ids", [])) == 1]
    if min_confidence > 0.0:
        result = [i for i in result if i.get("confidence", 0.0) >= min_confidence]
    return result


# ---------------------------------------------------------------------------
# Métricas de retrieval basadas en chunk_id exacto
# ---------------------------------------------------------------------------

def context_recall(retrieved_ids: list[str], reference_ids: set[str]) -> float:
    if not reference_ids:
        return 0.0
    return len(set(retrieved_ids) & reference_ids) / len(reference_ids)


def hit_rate(retrieved_ids: list[str], reference_ids: set[str]) -> float:
    """1.0 si al menos un positive fue recuperado, 0.0 si no."""
    return 1.0 if set(retrieved_ids) & reference_ids else 0.0


def mrr(retrieved_ids: list[str], reference_ids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/posición del primer hit (1-indexed)."""
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in reference_ids:
            return 1.0 / i
    return 0.0


def context_precision(retrieved_ids: list[str], reference_ids: set[str]) -> float:
    """Average Precision@k ponderado por posición en el ranking."""
    if not retrieved_ids:
        return 0.0
    relevances = [1 if cid in reference_ids else 0 for cid in retrieved_ids]
    total_relevant = sum(relevances)
    if total_relevant == 0:
        return 0.0
    ap_sum, relevant_seen = 0.0, 0
    for k, rel in enumerate(relevances, 1):
        if rel:
            relevant_seen += 1
            ap_sum += relevant_seen / k
    return ap_sum / total_relevant


# ---------------------------------------------------------------------------
# Pipeline RAG (nodos reales, sin routing)
# ---------------------------------------------------------------------------

async def run_rag_pipeline(
    query: str, use_base: bool, retrieval_only: bool = False,
) -> tuple[list[dict], list[dict], list[dict], str]:
    """
    Corre el pipeline: embed → bm25+semantic → RRF → rerank → generate.
    Retorna (candidate_pool, fused_docs, reranked_docs, answer).

    El threshold del reranker se ignora para evaluación: si el reranker
    encontró docs los usamos para medir retrieval quality, aunque en
    producción el sistema hubiera respondido "no encontré información".

    Con retrieval_only=True se salta el paso 5 (generate_node) por completo:
    no llama al LLM generador — el único paso de este pipeline que cuesta
    dinero/tiempo real (bm25/semantic/RRF/rerank son Mongo + el modelo ONNX
    local del model service, no un LLM). answer queda "" en ese caso.
    """
    # 1. Embedding
    query_embedding = await model_service.get_embedding(query, use_base=use_base)

    # 2. Búsqueda BM25 + semántica en paralelo
    bm25_state, semantic_state = await asyncio.gather(
        bm25_search_node({"synthesized_query": query}),
        semantic_search_node({
            "synthesized_query": query,
            "query_embedding": query_embedding,
        }),
    )

    # Pool completo BM25+Semantic antes del RRF (para recall@candidates)
    bm25_docs = bm25_state["bm25_results"]
    semantic_docs = semantic_state["semantic_results"]
    seen: set[str] = set()
    candidate_pool: list[dict] = []
    for doc in bm25_docs + semantic_docs:
        if doc["_id"] not in seen:
            seen.add(doc["_id"])
            candidate_pool.append(doc)

    # 3. RRF fusion — punto de medición PRE-reranker (top-10)
    fused_state = rrf_fusion_node({
        "bm25_results": bm25_docs,
        "semantic_results": semantic_docs,
    })
    fused_docs: list[dict] = fused_state["fused_results"]

    # 4. Rerank — punto de medición POST-reranker (sin corte por threshold)
    reranked_state = await rerank_node({
        "fused_results": fused_docs,
        "synthesized_query": query,
    })
    reranked_docs: list[dict] = reranked_state.get("reranked_docs") or []

    if not reranked_docs:
        return candidate_pool, fused_docs, [], "No se encontraron documentos relevantes para responder la pregunta."

    if retrieval_only:
        return candidate_pool, fused_docs, reranked_docs, ""

    # 5. Generación (usa los docs rerankeados sin importar el threshold)
    gen_state = await generate_node({
        "reranked_docs": reranked_docs,
        "user_query": query,
        "synthesized_query": query,
        "messages": [{"role": "user", "content": query}],
        "context_sufficient": False,
    })
    answer: str = gen_state.get("answer", "")

    return candidate_pool, fused_docs, reranked_docs, answer


# ---------------------------------------------------------------------------
# Evaluación de un modelo (finetuned o base)
# ---------------------------------------------------------------------------

async def evaluate_model(
    dataset: list[dict],
    use_base: bool,
    evaluator_llm,
    evaluator_embeddings,
    label: str,
    retrieval_only: bool = False,
) -> tuple[dict, pd.DataFrame]:
    faithfulness_metric = None if retrieval_only else Faithfulness(llm=evaluator_llm)
    relevancy_metric = None if retrieval_only else AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

    rows = []
    for i, item in enumerate(dataset):
        query: str = item["normalized_query"]
        reference_ids: set[str] = set(item["positive_chunk_ids"])

        print(f"  [{label}] [{i+1}/{len(dataset)}] {query[:70]}...")

        try:
            candidate_pool, fused_docs, reranked_docs, answer = await run_rag_pipeline(
                query, use_base=use_base, retrieval_only=retrieval_only,
            )
        except Exception as exc:
            print(f"    ERROR en pipeline: {exc}")
            rows.append({
                "dataset_id": item.get("dataset_id"),
                "query": query,
                **{col: None for col in METRIC_COLS},
                "topic": item.get("topic"),
                "barrier": item.get("barrier"),
                "model_variant": label,
                "error": str(exc),
            })
            continue

        # Métricas sobre pool completo BM25+Semantic (pre-RRF)
        candidate_ids = [doc["chunk_id"] for doc in candidate_pool]
        recall_cands = context_recall(candidate_ids, reference_ids)
        hit_cands = hit_rate(candidate_ids, reference_ids)

        # Métricas retrieval — PRE reranker (RRF top-10)
        fused_ids = [doc["chunk_id"] for doc in fused_docs]
        pre_recall = context_recall(fused_ids, reference_ids)
        pre_hit = hit_rate(fused_ids, reference_ids)
        pre_mrr = mrr(fused_ids, reference_ids)
        pre_precision = context_precision(fused_ids, reference_ids)

        # Métricas retrieval — POST reranker (top-5, sin corte por threshold)
        reranked_ids = [doc["chunk_id"] for doc in reranked_docs]
        post_recall = context_recall(reranked_ids, reference_ids)
        post_precision = context_precision(reranked_ids, reference_ids)

        retrieved_texts = [doc["text"] for doc in reranked_docs]

        # RAGAS LLM-based metrics
        f_val, r_val = None, None
        if not retrieval_only and retrieved_texts and answer:
            try:
                f_result = await faithfulness_metric.ascore(
                    user_input=query,
                    response=answer,
                    retrieved_contexts=retrieved_texts,
                )
                f_val = f_result.value
            except Exception as exc:
                print(f"    faithfulness error: {exc}")

            try:
                r_result = await relevancy_metric.ascore(
                    user_input=query,
                    response=answer,
                )
                r_val = r_result.value
            except Exception as exc:
                print(f"    answer_relevancy error: {exc}")

        row = {
            "dataset_id": item.get("dataset_id"),
            "query": query,
            "recall_candidates": recall_cands,
            "hit_rate_candidates": hit_cands,
            "context_recall_pre": pre_recall,
            "hit_rate_pre": pre_hit,
            "mrr_pre": pre_mrr,
            "context_precision_pre": pre_precision,
            "context_precision_cond": pre_precision if pre_recall > 0 else None,
            "context_recall_post": post_recall,
            "context_precision_post": post_precision,
            "faithfulness": f_val,
            "answer_relevancy": r_val,
            "topic": item.get("topic"),
            "barrier": item.get("barrier"),
            "model_variant": label,
            "error": None,
        }
        rows.append(row)

        faith_str = f"{f_val:.3f}" if f_val is not None else "n/a"
        rel_str = f"{r_val:.3f}" if r_val is not None else "n/a"
        print(
            f"    cands={recall_cands:.3f} | rrf={pre_recall:.3f} hit={pre_hit:.0f} "
            f"mrr={pre_mrr:.3f} prec={pre_precision:.3f} | post={post_recall:.3f} "
            f"| faith={faith_str} rel={rel_str}"
        )

    df = pd.DataFrame(rows)
    summary = {"modelo": label, "n_queries": len(df)}
    for col in METRIC_COLS:
        summary[f"mean_{col}"] = df[col].mean()
        summary[f"std_{col}"] = df[col].std()

    return summary, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark() -> None:
    """Lee los CSVs existentes y genera la tabla comparativa sin correr el pipeline."""
    ft_path = OUT_DIR / "resultados_finetuned.csv"
    base_path = OUT_DIR / "resultados_base.csv"

    missing = [p for p in (ft_path, base_path) if not p.exists()]
    if missing:
        for p in missing:
            model = "finetuned" if "finetuned" in p.name else "base"
            print(f"  ✗ Falta {p.name} — córrelo primero con: "
                  f"uv run python scripts/eval_dataset_ragas.py --model {model}")
        sys.exit(1)

    df_ft = pd.read_csv(ft_path)
    df_base = pd.read_csv(base_path)

    rows = []
    for col in METRIC_COLS:
        ft_mean = df_ft[col].mean()
        base_mean = df_base[col].mean()
        delta = ft_mean - base_mean
        pct = (delta / base_mean * 100) if base_mean else float("nan")
        rows.append({
            "metric": col,
            "finetuned": ft_mean,
            "base": base_mean,
            "delta": delta,
            "% mejora": pct,
        })

    comparison = pd.DataFrame(rows)

    print("\n=== Comparación FINETUNED vs BASE ===")
    print(f"{'metric':<26} {'finetuned':>10} {'base':>10} {'delta':>10} {'% mejora':>10}")
    print("-" * 70)
    for _, r in comparison.iterrows():
        sign = "+" if r["delta"] >= 0 else ""
        print(
            f"{r['metric']:<26} {r['finetuned']:>10.4f} {r['base']:>10.4f} "
            f"{sign}{r['delta']:>9.4f} {sign}{r['% mejora']:>9.2f}%"
        )

    out_path = OUT_DIR / "comparacion_ragas.csv"
    comparison.to_csv(out_path, index=False)
    print(f"\nGuardado en {out_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación RAGAS del sistema RAG")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=["finetuned", "base"],
        help="Variante de embedding a evaluar")
    group.add_argument("--benchmark", action="store_true",
        help="Comparar resultados de ejecuciones previas")
    parser.add_argument("--samples", type=int, default=100,
        help="Número de ejemplos a cargar del dataset (default: 100)")
    parser.add_argument("--filter-single", action="store_true",
        help="Evaluar solo ejemplos con 1 positive_chunk_id (elimina ambigüedad de anotación)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
        help="Filtrar ejemplos por confidence mínima (ej: 0.9)")
    parser.add_argument("--retrieval-only", action="store_true",
        help="Solo métricas de retrieval (sin llamadas LLM — faithfulness y answer_relevancy se omiten)")
    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
        return

    use_base = args.model == "base"
    label = args.model
    n_samples = args.samples

    print(f"Cargando {n_samples} ejemplos de {DATASET_PATH.name}...")
    dataset = load_dataset(DATASET_PATH, n_samples)

    dataset = filter_dataset(
        dataset,
        filter_single=args.filter_single,
        min_confidence=args.min_confidence,
    )

    filters_desc = []
    if args.filter_single:
        filters_desc.append("single-positive")
    if args.min_confidence > 0.0:
        filters_desc.append(f"confidence≥{args.min_confidence}")
    filter_label = f" [{', '.join(filters_desc)}]" if filters_desc else ""
    print(f"Ejemplos a evaluar: {len(dataset)}{filter_label}\n")

    evaluator_llm, evaluator_embeddings = None, None
    if not args.retrieval_only:
        async_client = AsyncOpenAI(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
        )
        evaluator_llm = llm_factory(
            settings.ragas_llm_model,
            provider="openai",
            client=async_client,
            max_tokens=10000
        )
        evaluator_embeddings = HuggingFaceEmbeddings(
            model=settings.ragas_embeddings_model,
        )

    endpoint = "/embed_base" if use_base else "/embed"
    mode_label = " [retrieval-only]" if args.retrieval_only else ""
    print(f"=== Evaluando RAG con embedding {label.upper()} ({endpoint}){filter_label}{mode_label} ===")
    summary, df = await evaluate_model(
        dataset,
        use_base=use_base,
        evaluator_llm=evaluator_llm,
        evaluator_embeddings=evaluator_embeddings,
        label=label,
        retrieval_only=args.retrieval_only,
    )

    out_csv = OUT_DIR / f"resultados_{label}.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n=== Resumen {label}{filter_label} ===")
    print(f"  {'Métrica':<26} {'Media':>8} {'Std':>8}")
    print(f"  {'-'*44}")
    for col in METRIC_COLS:
        val = summary.get(f"mean_{col}", float("nan"))
        std = summary.get(f"std_{col}", float("nan"))
        val_str = f"{val:>8.4f}" if val == val else "     n/a"
        std_str = f"±{std:>7.4f}" if std == std else ""
        print(f"  {col:<26} {val_str} {std_str}")
    print(f"\nCSV guardado en {out_csv}")

    # ── Validación OE4-I1 ────────────────────────────────────────────────────
    OE4_TARGETS = {
        "Context Recall (recall_candidates)": ("recall_candidates", 0.65),
        "Context Precision (precision_cond)": ("context_precision_cond", 0.45),
        "Faithfulness": ("faithfulness", 0.80),
        "Answer Relevancy": ("answer_relevancy", 0.70),
    }
    print(f"\n{'═' * 60}")
    print(f"  === Validación OE4-I1 ===")
    print(f"{'─' * 60}")
    print(f"  {'Indicador':<38} {'Valor':>7} {'Target':>7} {'':>4}")
    print(f"  {'-'*56}")
    all_pass = True
    for label_oe4, (col, target) in OE4_TARGETS.items():
        val = summary.get(f"mean_{col}", float("nan"))
        if val != val:  # nan
            status = "⚠ n/a"
            all_pass = False
        else:
            passed = val >= target
            status = "✓" if passed else "✗"
            if not passed:
                all_pass = False
        val_str = f"{val:.4f}" if val == val else "   n/a"
        print(f"  {label_oe4:<38} {val_str:>7} ≥{target:.2f}  {status}")
    print(f"{'─' * 60}")
    verdict = "APROBADO ✓" if all_pass else "NO APROBADO ✗ (verificar métricas pendientes)"
    print(f"  Resultado OE4-I1: {verdict}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    asyncio.run(main())
