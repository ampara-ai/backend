"""Acuerdo entre evaluadores para la validación de expertos.

`ratings` es una matriz ítems × evaluadores; None o NaN marcan un dato faltante.

  - gwet_ac            : AC1 de Gwet (weights="identity") o AC2 con pesos
                         "ordinal" / "quadratic" (Gwet, 2014). Robusto cuando
                         casi todas las calificaciones caen en 4-5, donde kappa
                         y alfa se desploman (paradoja de kappa).
  - krippendorff_alpha : alfa de Krippendorff con métrica "nominal", "ordinal"
                         o "interval" (Krippendorff, 2011).
"""

import math
from collections.abc import Sequence

import numpy as np


def _is_missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _category_counts(ratings: Sequence[Sequence], categories: Sequence) -> np.ndarray:
    """counts[i, k] = número de evaluadores que asignaron el ítem i a la categoría k."""
    index = {c: k for k, c in enumerate(categories)}
    counts = np.zeros((len(ratings), len(categories)))
    for i, row in enumerate(ratings):
        for value in row:
            if _is_missing(value):
                continue
            if value not in index:
                raise ValueError(f"valor {value!r} fuera de las categorías {list(categories)}")
            counts[i, index[value]] += 1
    return counts


def _gwet_weights(categories: Sequence, weights: str) -> np.ndarray:
    q = len(categories)
    if weights == "identity":
        return np.eye(q)
    if weights == "ordinal":
        # Pesos ordinales de Gwet: según cuántas categorías separan k y l.
        span = np.abs(np.subtract.outer(np.arange(q), np.arange(q))) + 1
        pairs = span * (span - 1) / 2
        return 1 - pairs / pairs.max()
    if weights == "quadratic":
        values = np.asarray(categories, dtype=float)
        return 1 - np.subtract.outer(values, values) ** 2 / (values.max() - values.min()) ** 2
    raise ValueError(f"pesos desconocidos: {weights!r}")


def gwet_ac(ratings: Sequence[Sequence], categories: Sequence, weights: str = "identity") -> float:
    """`categories` son TODAS las categorías posibles (p. ej. 1-5 en una Likert), no solo las observadas."""
    q = len(categories)
    w = _gwet_weights(categories, weights)
    counts = _category_counts(ratings, categories)
    raters = counts.sum(axis=1)
    rated, pairable = raters >= 1, raters >= 2
    if not pairable.any():
        raise ValueError("ningún ítem tiene dos o más calificaciones")

    weighted = counts @ w.T  # r*_ik = Σ_l w_kl r_il
    agreement = (counts[pairable] * (weighted[pairable] - 1)).sum(axis=1)
    p_a = (agreement / (raters[pairable] * (raters[pairable] - 1))).mean()

    pi = (counts[rated] / raters[rated, None]).mean(axis=0)
    p_e = w.sum() / (q * (q - 1)) * (pi * (1 - pi)).sum()
    return float((p_a - p_e) / (1 - p_e))


def krippendorff_alpha(
    ratings: Sequence[Sequence], level: str = "ordinal", categories: Sequence | None = None
) -> float:
    if categories is None:
        categories = sorted({v for row in ratings for v in row if not _is_missing(v)})
    counts = _category_counts(ratings, categories)
    per_unit = counts.sum(axis=1)
    counts, per_unit = counts[per_unit >= 2], per_unit[per_unit >= 2]

    # Matriz de coincidencias: cada par ordenado de valores de una unidad pesa 1 / (m_u − 1).
    coincidences = sum(
        (np.outer(row, row) - np.diag(row)) / (m - 1) for row, m in zip(counts, per_unit)
    )
    marginals = coincidences.sum(axis=1)
    n = marginals.sum()

    q = len(categories)
    if level == "nominal":
        delta2 = 1 - np.eye(q)
    elif level == "interval":
        values = np.asarray(categories, dtype=float)
        delta2 = np.subtract.outer(values, values) ** 2
    elif level == "ordinal":
        delta2 = np.zeros((q, q))
        for c in range(q):
            for k in range(c, q):
                distance = marginals[c : k + 1].sum() - (marginals[c] + marginals[k]) / 2
                delta2[c, k] = delta2[k, c] = distance**2
    else:
        raise ValueError(f"nivel desconocido: {level!r}")

    observed = (coincidences * delta2).sum() / n
    expected = (np.outer(marginals, marginals) * delta2).sum() / (n * (n - 1))
    return float(1 - observed / expected)
