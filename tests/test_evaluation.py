import math

import numpy as np
import pytest

from evaluation.agreement import gwet_ac, krippendorff_alpha
from evaluation.bootstrap import ClusterBootstrap
from evaluation.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    reciprocal_rank_at_k,
    recall_at_k,
    score_query,
)

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

RANKING = ["a", "x", "b", "y", "c"]
RELEVANT = {"a", "b", "c", "d"}


def test_recall_uncapped_and_capped():
    assert recall_at_k(RANKING, RELEVANT, 3) == pytest.approx(2 / 4)
    assert recall_at_k(RANKING, RELEVANT, 3, capped=True) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank_at_k(["x", "a"], {"a"}, 10) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(["x", "a"], {"a"}, 1) == 0.0


def test_average_precision_normalized_by_min_relevant_k():
    # aciertos en 1, 3, 5 → (1/1 + 2/3 + 3/5) / min(4, 5)
    assert average_precision_at_k(RANKING, RELEVANT, 5) == pytest.approx((1 + 2 / 3 + 3 / 5) / 4)


def test_ndcg_binary():
    dcg = 1 + 1 / math.log2(4) + 1 / math.log2(6)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, 5))
    assert ndcg_at_k(RANKING, RELEVANT, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 10) == pytest.approx(1.0)


def test_no_relevant_scores_zero():
    assert score_query(RANKING, set()) == {key: 0.0 for key in score_query(RANKING, set())}


def test_score_query_keys():
    assert set(score_query(RANKING, RELEVANT, ks=(10,))) == {"recall@10", "mrr@10", "map@10", "ndcg@10"}


# ---------------------------------------------------------------------------
# Bootstrap por clústeres
# ---------------------------------------------------------------------------

CLUSTERS = ["c1", "c1", "c2", "c3", "c3", "c3", "c4"]


def test_constant_difference_has_degenerate_interval():
    boot = ClusterBootstrap(CLUSTERS, n_resamples=500, seed=1)
    a = np.linspace(0, 1, len(CLUSTERS))
    result = boot.paired_difference(a + 0.1, a)
    assert result["estimate"] == pytest.approx(0.1)
    assert result["ci_low"] == pytest.approx(0.1)
    assert result["ci_high"] == pytest.approx(0.1)


def test_vectorized_matches_naive_resampling():
    boot = ClusterBootstrap(CLUSTERS, n_resamples=200, seed=7)
    values = np.array([0.2, 0.4, 1.0, 0.0, 0.5, 0.1, 0.9])
    labels = np.unique(CLUSTERS)
    expected = []
    for counts in boot._multiplicity:
        picked = [v for c, n in zip(labels, counts) for v in values[np.array(CLUSTERS) == c] for _ in range(int(n))]
        expected.append(np.mean(picked))
    assert np.allclose(boot.resampled_means(values), expected)


def test_same_seed_same_interval():
    values = np.random.default_rng(0).random(len(CLUSTERS))
    first = ClusterBootstrap(CLUSTERS, n_resamples=300, seed=3).mean(values)
    second = ClusterBootstrap(CLUSTERS, n_resamples=300, seed=3).mean(values)
    assert first == second


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        ClusterBootstrap(CLUSTERS, n_resamples=10).mean([0.1, 0.2])


# ---------------------------------------------------------------------------
# Acuerdo entre evaluadores
# ---------------------------------------------------------------------------

N = None
# Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability": 12 unidades × 4 observadores.
KRIPPENDORFF_EXAMPLE = [
    list(unit)
    for unit in zip(
        [1, 2, 3, 3, 2, 1, 4, 1, 2, N, N, N],
        [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, N, 3],
        [N, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, N],
        [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, N],
    )
]


@pytest.mark.parametrize("level, expected", [("nominal", 0.743), ("ordinal", 0.815), ("interval", 0.849)])
def test_krippendorff_published_example(level, expected):
    assert krippendorff_alpha(KRIPPENDORFF_EXAMPLE, level) == pytest.approx(expected, abs=5e-4)


def test_gwet_ac1_matches_irrcac_on_published_example():
    # Valor de referencia calculado con irrCAC (CAC(...).gwet()), mismos datos y categorías 1-5.
    assert gwet_ac(KRIPPENDORFF_EXAMPLE, [1, 2, 3, 4, 5]) == pytest.approx(0.77544, abs=5e-5)


def test_perfect_agreement():
    ratings = [[5, 5, 5], [4, 4, 4], [5, 5, 5]]
    assert gwet_ac(ratings, [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert krippendorff_alpha(ratings, "ordinal") == pytest.approx(1.0)


def test_gwet_stays_high_where_alpha_collapses():
    # Calificaciones concentradas en 4-5 con desacuerdos aislados: la paradoja de kappa.
    ratings = [[5, 5, 5, 5, 4]] * 4 + [[5, 5, 5, 5, 5]] * 14 + [[4, 4, 4, 4, 4]] * 2
    assert gwet_ac(ratings, [1, 2, 3, 4, 5], weights="ordinal") > 0.9
    assert krippendorff_alpha(ratings, "ordinal", [1, 2, 3, 4, 5]) < 0.8


def test_value_outside_categories_raises():
    with pytest.raises(ValueError):
        gwet_ac([[6, 5]], [1, 2, 3, 4, 5])
