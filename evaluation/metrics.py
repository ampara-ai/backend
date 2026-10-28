"""Métricas de recuperación con relevancia binaria.

Mismas definiciones que scripts/eval_ablation.py (las del artículo):
  - recall@k : aciertos en el top-k / |relevantes|. Con capped=True divide entre
               min(|relevantes|, k), la variante de la Tabla 7 del artículo.
  - MRR@k    : 1 / posición del primer acierto dentro del top-k.
  - AP@k     : suma de precision@i en cada acierto, dividida entre
               min(|relevantes|, k) (acotada en [0, 1]).
  - NDCG@k   : ganancia binaria y descuento log2; el IDCG usa min(|relevantes|, k)
               aciertos.

Cada función devuelve el valor de UNA consulta; el valor reportado es la media
sobre consultas. Una consulta sin relevantes en el top-k aporta 0.
"""

import math
from collections.abc import Sequence, Set


def recall_at_k(ranking: Sequence[str], relevant: Set[str], k: int, capped: bool = False) -> float:
    if not relevant:
        return 0.0
    hits = len(set(ranking[:k]) & relevant)
    return hits / (min(len(relevant), k) if capped else len(relevant))


def reciprocal_rank_at_k(ranking: Sequence[str], relevant: Set[str], k: int) -> float:
    for i, doc_id in enumerate(ranking[:k], 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def average_precision_at_k(ranking: Sequence[str], relevant: Set[str], k: int) -> float:
    normalizer = min(len(relevant), k)
    if normalizer == 0:
        return 0.0
    hits, precision_sum = 0, 0.0
    for i, doc_id in enumerate(ranking[:k], 1):
        if doc_id in relevant:
            hits += 1
            precision_sum += hits / i
    return precision_sum / normalizer


def ndcg_at_k(ranking: Sequence[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, doc_id in enumerate(ranking[:k], 1) if doc_id in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


METRICS = {
    "recall": recall_at_k,
    "mrr": reciprocal_rank_at_k,
    "map": average_precision_at_k,
    "ndcg": ndcg_at_k,
}


def score_query(ranking: Sequence[str], relevant: Set[str], ks: Sequence[int] = (10, 100)) -> dict[str, float]:
    """Todas las métricas para una consulta, con claves tipo "ndcg@10"."""
    return {f"{name}@{k}": fn(ranking, relevant, k) for k in ks for name, fn in METRICS.items()}
