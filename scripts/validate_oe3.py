"""
Validación OE3-I2: 100 consultas end-to-end a través del pipeline RAG completo.

Métricas evaluadas:
  - Tasa de éxito: % de consultas que completan el flujo sin error (target ≥ 95%)
  - Latencia p95: percentil 95 de tiempo de respuesta en segundos (target ≤ 10s)

Una consulta se considera exitosa cuando:
  - Completa sin excepción
  - Produce una respuesta no vacía
  - Retorna fuentes (sources), o el sistema indicó out_of_domain / from_context

Uso:
    uv run python scripts/validate_oe3.py
    uv run python scripts/validate_oe3.py --samples 100
    uv run python scripts/validate_oe3.py --samples 100 --concurrency 5
"""

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.rag_graph import rag_graph

DATASET_PATH = Path(__file__).resolve().parents[1] / "questions_with_positive_context.jsonl"

TARGET_SUCCESS_RATE = 0.95
TARGET_P95_LATENCY = 10.0


def load_queries(path: Path, n: int) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
            if len(items) >= n:
                break
    return items


async def run_query(item: dict) -> dict:
    query: str = item["normalized_query"]
    session_id = f"oe3_val_{uuid.uuid4().hex[:12]}"

    t0 = time.perf_counter()
    try:
        result = await rag_graph.ainvoke(
            {
                "session_id": session_id,
                "user_query": query,
                "messages": [{"role": "user", "content": query}],
            },
            config={"configurable": {"thread_id": session_id}},
        )
        elapsed = time.perf_counter() - t0

        has_answer = bool(result.get("answer", "").strip())
        has_sources = bool(result.get("sources"))
        out_of_domain = bool(result.get("out_of_domain"))
        from_context = bool(result.get("from_context"))
        no_results = bool(result.get("no_results"))

        # Éxito: produjo respuesta Y (tiene fuentes OR es caso especial válido)
        success = has_answer and (has_sources or out_of_domain or from_context)

        return {
            "dataset_id": item.get("dataset_id", "?"),
            "query": query[:80],
            "success": success,
            "latency": elapsed,
            "has_answer": has_answer,
            "has_sources": has_sources,
            "out_of_domain": out_of_domain,
            "from_context": from_context,
            "no_results": no_results,
            "error": None,
        }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "dataset_id": item.get("dataset_id", "?"),
            "query": query[:80],
            "success": False,
            "latency": elapsed,
            "has_answer": False,
            "has_sources": False,
            "out_of_domain": False,
            "from_context": False,
            "no_results": False,
            "error": str(exc),
        }


async def run_all(items: list[dict], concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded(item: dict, idx: int) -> dict:
        async with semaphore:
            r = await run_query(item)
            status = "✓" if r["success"] else "✗"
            print(f"  [{idx+1:>3}/{len(items)}] {status} {r['latency']:>5.1f}s  {r['query']}")
            return r

    tasks = [bounded(item, i) for i, item in enumerate(items)]
    results = await asyncio.gather(*tasks)
    return list(results)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p / 100
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def print_report(results: list[dict]) -> None:
    total = len(results)
    successes = sum(1 for r in results if r["success"])
    failures = [r for r in results if not r["success"]]
    latencies = [r["latency"] for r in results]

    success_rate = successes / total if total > 0 else 0.0
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    ok_rate = "✓" if success_rate >= TARGET_SUCCESS_RATE else "✗"
    ok_p95 = "✓" if p95 <= TARGET_P95_LATENCY else "✗"

    print(f"\n{'═' * 60}")
    print(f"  === Validación OE3-I2 ===")
    print(f"{'─' * 60}")
    print(f"  Consultas ejecutadas : {total}")
    print(f"  Éxito                : {successes}/{total} ({success_rate*100:.1f}%)  "
          f"→ TARGET ≥{TARGET_SUCCESS_RATE*100:.0f}% {ok_rate}")
    print(f"{'─' * 60}")
    print(f"  Latencia p50         : {p50:.2f}s")
    print(f"  Latencia p95         : {p95:.2f}s  "
          f"→ TARGET ≤{TARGET_P95_LATENCY:.0f}s {ok_p95}")
    print(f"  Latencia p99         : {p99:.2f}s")
    print(f"  Latencia mín/máx     : {min(latencies):.2f}s / {max(latencies):.2f}s")

    if failures:
        print(f"\n{'─' * 60}")
        print(f"  Failures ({len(failures)}):")
        for r in failures:
            reason = r["error"] or (
                "sin respuesta" if not r["has_answer"] else "sin fuentes ni caso especial"
            )
            print(f"    ✗ [{r['dataset_id']}] {r['query']}")
            print(f"      → {reason}")

    print(f"\n{'─' * 60}")
    overall_pass = (success_rate >= TARGET_SUCCESS_RATE) and (p95 <= TARGET_P95_LATENCY)
    verdict = "APROBADO ✓" if overall_pass else "NO APROBADO ✗"
    print(f"  Resultado OE3-I2: {verdict}")
    print(f"{'═' * 60}\n")

    return overall_pass


async def main() -> None:
    parser = argparse.ArgumentParser(description="Validación OE3-I2: 100 consultas end-to-end")
    parser.add_argument("--samples", type=int, default=100,
        help="Número de consultas a ejecutar (default: 100)")
    parser.add_argument("--concurrency", type=int, default=1,
        help="Consultas en paralelo (default: 1 — secuencial para medir latencia real)")
    args = parser.parse_args()

    print(f"Cargando {args.samples} consultas de {DATASET_PATH.name}...")
    items = load_queries(DATASET_PATH, args.samples)
    print(f"Consultas cargadas: {len(items)}\n")
    print(f"Ejecutando pipeline RAG completo (concurrency={args.concurrency})...\n")

    results = await run_all(items, concurrency=args.concurrency)
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
