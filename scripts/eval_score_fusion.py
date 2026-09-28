"""
Interpolación de scores (fusión lineal embedding + reranker) — propuesta del
asesor de tesis como alternativa al módulo de fusión actual (RRF + reranker
reemplazando el orden). Referencia: RRF solo usa el rank de cada candidato,
tira a la basura la magnitud del score; el reranker reemplaza el orden del
embedding sin nada que lo corrija cuando se equivoca. La interpolación en
cambio conserva la confianza del embedding y deja que el reranker desempate:

    score_final = alpha * emb_norm + (1 - alpha) * rerank_norm

- emb_norm / rerank_norm: normalización min-max POR CONSULTA, sobre el mismo
  pool de POOL_K candidatos (--pool-k, default 30, rango sugerido 20-30). El
  reranker se corre sobre TODOS los POOL_K candidatos (no solo el top-5/10 de
  producción), porque se necesita su score para cada uno, no solo para
  reordenar los primeros.
- alpha se barre (grid 0.00-1.00 paso 0.05) en un split de VALIDACIÓN (30% del
  dataset, semilla fija --seed) para elegir el mejor por --alpha-metric
  (default ndcg@10). El resultado final se reporta en el split de TEST (70%)
  con ese alpha ya fijo — nunca se elige alpha mirando test, para que sea una
  comparación honesta.
- alpha=1.0 es exactamente el embedding solo (dense_finetuned de
  eval_ablation.py) y alpha=0.0 es el reranker solo — ambos se reportan en
  test junto al alpha elegido, como baseline de referencia. Si el mejor alpha
  en validación resulta ser 1.0, el reranker no está aportando nada y hay que
  reportarlo tal cual, no forzar un intermedio.
- BM25 queda FUERA de esta primera versión (el asesor lo marcó como opcional
  para empezar: "agrégale sus top-M... pero no es obligatorio para empezar").
  Se puede sumar después como un tercer término w_bm25*bm25_norm.
- Recall@N con N=POOL_K no cambia entre valores de alpha (es la misma lista de
  candidatos, solo se reordena) — lo que cambia con alpha es qué entra al
  top-10 (FUSION_K). Por eso se guarda el pool completo, para poder auditar
  ese invariante si hace falta.

Dos fases, separadas porque la parte cara (Mongo + reranker) se corre UNA sola
vez y el barrido de alpha es pura recombinación numérica después (como dijo
el asesor: "esto es solo recombinar números, sin volver a correr modelos"):

  --mode collect  arma el pool (embedding top-N) + rerankea los N completos +
                  guarda scores crudos por candidato en
                  results/score_fusion_candidates.csv. No elige alpha, no
                  calcula métricas.
  --mode sweep    lee ese CSV, normaliza min-max por consulta, barre alpha en
                  el split de validación, guarda la curva
                  (results/score_fusion_alpha_sweep.csv) y el resultado final
                  en test con el mejor alpha vs alpha=0 vs alpha=1
                  (results/score_fusion_test_results.csv). No vuelve a llamar
                  a Mongo/reranker.
  --mode all      (default) hace las dos fases seguidas.

Uso:
    uv run python scripts/eval_score_fusion.py --samples 100
    uv run python scripts/eval_score_fusion.py --mode sweep --alpha-metric mrr
    uv run python scripts/eval_score_fusion.py --samples 100 --pool-k 20
    uv run python scripts/eval_score_fusion.py --samples 100 --reranker-endpoint /rerank_qwen
"""

import argparse
import asyncio
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import model_service, mongodb
from scripts.eval_dataset_ragas import DATASET_PATH, filter_dataset, load_dataset

OUT_DIR = Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES_PATH = OUT_DIR / "score_fusion_candidates.csv"
SWEEP_PATH = OUT_DIR / "score_fusion_alpha_sweep.csv"
TEST_RESULTS_PATH = OUT_DIR / "score_fusion_test_results.csv"

DEFAULT_POOL_K = 30          # candidatos del embedding, rerankeados completos (asesor: 20-30)
DEFAULT_VAL_FRACTION = 0.30  # 30% validación / 70% test
DEFAULT_SEED = 42
FUSION_K = 10                # corte final tras fusionar (recall@10, mrr@10, ndcg@10, map@10)
ALPHA_GRID = [round(a, 2) for a in np.arange(0.0, 1.0001, 0.05)]


# ---------------------------------------------------------------------------
# Fase 1: collect — arma el pool y guarda scores crudos (Mongo + reranker)
# ---------------------------------------------------------------------------

def _split_dataset_ids(dataset_ids: list[str], val_fraction: float, seed: int) -> dict[str, str]:
    """{dataset_id: "val"|"test"}, split aleatorio reproducible (semilla fija)."""
    shuffled = dataset_ids[:]
    random.Random(seed).shuffle(shuffled)
    n_val = round(len(shuffled) * val_fraction)
    val_ids = set(shuffled[:n_val])
    return {did: ("val" if did in val_ids else "test") for did in dataset_ids}


async def _collect_query(query: str, pool_k: int, reranker_endpoint: str | None) -> list[dict]:
    """Embedding top-pool_k (mismo score que usaría dense_finetuned) + rerank
    COMPLETO de esos pool_k (no solo el top-5/10 real de producción — acá se
    necesita el score de cada candidato, no solo reordenar los primeros)."""
    embedding = await model_service.get_embedding(query, use_base=False)
    docs = await mongodb.vector_search(embedding, k=pool_k)
    if not docs:
        return []
    texts = [d["text"] for d in docs]
    ranked = await model_service.rerank(query, texts, endpoint=reranker_endpoint)
    rerank_score_by_index = {item["index"]: item["score"] for item in ranked}
    for i, doc in enumerate(docs):
        doc["_emb_score"] = doc.get("score", 0.0)
        doc["_rerank_score"] = rerank_score_by_index.get(i, 0.0)
    return docs


async def collect(
    dataset: list[dict], pool_k: int, reranker_endpoint: str | None, splits: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(dataset):
        query: str = item["normalized_query"]
        dataset_id = item.get("dataset_id")
        reference_ids: set[str] = set(item.get("positive_chunk_ids", []))
        n_positives = len(reference_ids)

        print(f"  [collect] [{i + 1}/{len(dataset)}] {query[:70]}...")
        try:
            docs = await _collect_query(query, pool_k, reranker_endpoint)
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        for doc in docs:
            rows.append({
                "dataset_id": dataset_id,
                "query": query,
                "chunk_id": doc["chunk_id"],
                "is_positive": doc["chunk_id"] in reference_ids,
                "n_positives": n_positives,
                "emb_score": doc["_emb_score"],
                "rerank_score": doc["_rerank_score"],
                "split": splits.get(dataset_id, "test"),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fase 2: sweep — normaliza, fusiona, barre alpha, reporta (pura recombinación)
# ---------------------------------------------------------------------------

def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo <= 0:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def _fuse_and_rank_hits(group: pd.DataFrame, alpha: float) -> list[bool]:
    """Normaliza min-max los scores de este pool (una consulta), fusiona con
    alpha, ordena desc y retorna la lista de is_positive en ese orden."""
    emb_norm = _minmax(group["emb_score"].to_numpy(dtype=float))
    rerank_norm = _minmax(group["rerank_score"].to_numpy(dtype=float))
    fused = alpha * emb_norm + (1 - alpha) * rerank_norm
    order = np.argsort(-fused)
    return group["is_positive"].to_numpy()[order].tolist()


def _metrics_from_hits(hits: list[bool], n_positives: int, k: int) -> dict[str, float]:
    """recall/mrr/ndcg/map@k a partir de una lista de hits ya ordenada por
    score final — mismas fórmulas exactas que recall_at_k/mrr_at_k/ndcg_at_k/
    map_at_k en eval_ablation.py, adaptadas a booleanos porque acá no hace
    falta reconstruir chunk_ids: ya sabemos cuáles son positivos y n_positives
    trae el total real (incluye positivos fuera del pool, si los hay)."""
    top_k = hits[:k]
    recall = (sum(top_k) / n_positives) if n_positives > 0 else 0.0

    mrr = 0.0
    for i, hit in enumerate(top_k, 1):
        if hit:
            mrr = 1.0 / i
            break

    normalizer = min(n_positives, k)
    map_ = 0.0
    if normalizer > 0:
        hits_seen, precision_sum = 0, 0.0
        for i, hit in enumerate(top_k, 1):
            if hit:
                hits_seen += 1
                precision_sum += hits_seen / i
        map_ = precision_sum / normalizer

    ideal_hits = min(n_positives, k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    dcg = sum(1.0 / math.log2(i + 1) for i, hit in enumerate(top_k, 1) if hit)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg, "map": map_}


def _evaluate_alpha(subset: pd.DataFrame, alpha: float, k: int) -> dict[str, float]:
    per_query = []
    for _, group in subset.groupby("dataset_id"):
        n_positives = int(group["n_positives"].iloc[0])
        hits = _fuse_and_rank_hits(group, alpha)
        per_query.append(_metrics_from_hits(hits, n_positives, k))
    dfm = pd.DataFrame(per_query)
    return {
        "n_queries": len(dfm),
        "mean_recall": dfm["recall"].mean(),
        "mean_mrr": dfm["mrr"].mean(),
        "mean_ndcg": dfm["ndcg"].mean(),
        "mean_map": dfm["map"].mean(),
    }


def sweep(df: pd.DataFrame, alpha_metric: str) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    print(f"\nBarriendo alpha en validación (n_queries={val['dataset_id'].nunique()})...")
    sweep_rows = []
    for alpha in ALPHA_GRID:
        result = _evaluate_alpha(val, alpha, FUSION_K)
        sweep_rows.append({"alpha": alpha, **result})
    sweep_df = pd.DataFrame(sweep_rows)

    metric_col = f"mean_{alpha_metric}"
    best_row = sweep_df.loc[sweep_df[metric_col].idxmax()]
    best_alpha = float(best_row["alpha"])
    print(f"Mejor alpha en validación por {metric_col}: {best_alpha} ({metric_col}={best_row[metric_col]:.4f})")

    print(f"\nEvaluando en test (n_queries={test['dataset_id'].nunique()}) con alpha={best_alpha}, "
          f"y los baselines alpha=0.0 (reranker solo) / alpha=1.0 (embedding solo)...")
    test_rows = []
    for label, alpha in [
        ("mejor_alpha", best_alpha),
        ("alpha_0.0_reranker_solo", 0.0),
        ("alpha_1.0_embedding_solo", 1.0),
    ]:
        result = _evaluate_alpha(test, alpha, FUSION_K)
        test_rows.append({"config": label, "alpha": alpha, **result})
    test_df = pd.DataFrame(test_rows)

    return sweep_df, test_df, best_alpha


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Interpolación de scores embedding + reranker")
    parser.add_argument("--mode", choices=["collect", "sweep", "all"], default="all",
        help="collect: solo arma el pool y guarda scores. sweep: solo barre alpha (usa el CSV "
             "ya guardado). all (default): las dos fases seguidas.")
    parser.add_argument("--samples", type=int, default=100,
        help="Número de ejemplos a cargar del dataset (default: 100), solo aplica a --mode collect/all")
    parser.add_argument("--filter-single", action="store_true",
        help="Evaluar solo ejemplos con 1 positive_chunk_id")
    parser.add_argument("--min-confidence", type=float, default=0.0,
        help="Filtrar ejemplos por confidence mínima (ej: 0.9)")
    parser.add_argument("--pool-k", type=int, default=DEFAULT_POOL_K,
        help=f"Candidatos del embedding a rerankear completos (default: {DEFAULT_POOL_K})")
    parser.add_argument("--reranker-endpoint", default=None,
        help="Endpoint del reranker en el model service (default: el de producción, "
             "settings.reranker_endpoint). Ej: /rerank_qwen para probar con el reranker Qwen.")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION,
        help=f"Fracción del dataset para validación, el resto es test (default: {DEFAULT_VAL_FRACTION})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
        help=f"Semilla del split validación/test, reproducible (default: {DEFAULT_SEED})")
    parser.add_argument("--alpha-metric", choices=["recall", "mrr", "ndcg", "map"], default="ndcg",
        help="Métrica usada para elegir el mejor alpha en validación (default: ndcg, o sea NDCG@10)")
    args = parser.parse_args()

    if args.mode in ("collect", "all"):
        print(f"Cargando {args.samples} ejemplos de {DATASET_PATH.name}...")
        dataset = load_dataset(DATASET_PATH, args.samples)
        dataset = filter_dataset(
            dataset, filter_single=args.filter_single, min_confidence=args.min_confidence,
        )
        dataset_ids = [item.get("dataset_id") for item in dataset]
        splits = _split_dataset_ids(dataset_ids, args.val_fraction, args.seed)
        n_val = sum(1 for s in splits.values() if s == "val")
        print(f"Ejemplos: {len(dataset)} (split: {n_val} validación / {len(dataset) - n_val} test, "
              f"seed={args.seed})")
        print(f"pool_k={args.pool_k}  reranker_endpoint={args.reranker_endpoint or '(producción)'}\n")

        df = await collect(dataset, args.pool_k, args.reranker_endpoint, splits)
        df.to_csv(CANDIDATES_PATH, index=False)
        print(f"\nScores crudos guardados en {CANDIDATES_PATH} ({len(df)} filas candidato)")

    if args.mode in ("sweep", "all"):
        if args.mode == "sweep":
            if not CANDIDATES_PATH.exists():
                print(f"✗ No encontrado: {CANDIDATES_PATH.name} — corre --mode collect primero.")
                sys.exit(1)
            df = pd.read_csv(CANDIDATES_PATH)

        sweep_df, test_df, best_alpha = sweep(df, args.alpha_metric)

        sweep_df.to_csv(SWEEP_PATH, index=False)
        test_df.to_csv(TEST_RESULTS_PATH, index=False)

        print(f"\n=== Curva de alpha en validación (por {args.alpha_metric}@{FUSION_K}) ===")
        print(sweep_df.to_string(index=False))
        print(f"\nGuardado en {SWEEP_PATH}")

        print(f"\n=== Resultado final en TEST (alpha elegido en validación = {best_alpha}) ===")
        print(test_df.to_string(index=False))
        print(f"\nGuardado en {TEST_RESULTS_PATH}")

        if best_alpha >= 0.999:
            print("\n⚠ El mejor alpha fue 1.0 — el reranker no está aportando nada en validación "
                  "con esta configuración. Repórtalo tal cual, no fuerces un intermedio.")


if __name__ == "__main__":
    asyncio.run(main())
