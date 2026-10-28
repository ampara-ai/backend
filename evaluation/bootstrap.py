"""Bootstrap pareado por clústeres para intervalos de confianza.

Varias consultas se generaron a partir del mismo chunk fuente, así que no son
independientes: se remuestrean los clústeres (chunks) con reemplazo, no las
consultas (Field & Welsh, 2007). Una instancia fija los remuestreos, de modo
que todas las comparaciones que la usan comparten los mismos (como en el
artículo).
"""

from collections.abc import Sequence

import numpy as np


class ClusterBootstrap:
    def __init__(self, clusters: Sequence[str], n_resamples: int = 10_000, seed: int = 0) -> None:
        labels, self._cluster_of = np.unique(np.asarray(clusters), return_inverse=True)
        self.n_clusters = len(labels)
        self.n_resamples = n_resamples
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, self.n_clusters, size=(n_resamples, self.n_clusters))
        # multiplicity[b, c] = veces que el clúster c aparece en el remuestreo b
        self._multiplicity = np.zeros((n_resamples, self.n_clusters))
        np.add.at(self._multiplicity, (np.arange(n_resamples)[:, None], draws), 1)

    def resampled_means(self, values: Sequence[float]) -> np.ndarray:
        """Media por consulta de `values` en cada remuestreo (clústeres de tamaño desigual)."""
        values = np.asarray(values, dtype=float)
        if values.shape != self._cluster_of.shape:
            raise ValueError(f"se esperaban {len(self._cluster_of)} valores, llegaron {len(values)}")
        sums = np.bincount(self._cluster_of, weights=values, minlength=self.n_clusters)
        sizes = np.bincount(self._cluster_of, minlength=self.n_clusters)
        return (self._multiplicity @ sums) / (self._multiplicity @ sizes)

    def mean(self, values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
        tail = (1 - confidence) / 2
        low, high = np.quantile(self.resampled_means(values), [tail, 1 - tail])
        return {"estimate": float(np.mean(values)), "ci_low": float(low), "ci_high": float(high)}

    def paired_difference(
        self, a: Sequence[float], b: Sequence[float], confidence: float = 0.95
    ) -> dict[str, float]:
        """Diferencia media a − b por consulta, con su intervalo percentil."""
        return self.mean(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), confidence)
