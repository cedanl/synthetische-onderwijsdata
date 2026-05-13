"""
Generative Correlation Manifolds (GCM) engine.

Preserves inter-column correlations via the Gaussian copula:
  1. Transform each marginal to N(0,1) through rank-based normalisation.
  2. Estimate the Spearman correlation matrix; compute its Cholesky factor L.
  3. Generate z ~ N(0, I), produce x̃ = L z, then invert each marginal.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


class GCMEngine:
    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)
        self._columns: List[str] = []
        self._marginals: Dict[str, _EmpiricalCDF] = {}
        self._cholesky: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "GCMEngine":
        self._columns = list(data.columns)
        n_cols = len(self._columns)
        normalised = np.empty((len(data), n_cols))

        for i, col in enumerate(self._columns):
            ecdf = _EmpiricalCDF(data[col].to_numpy())
            self._marginals[col] = ecdf
            normalised[:, i] = ecdf.to_normal(data[col].to_numpy())

        if n_cols == 1:
            corr = np.array([[1.0]])
        elif n_cols == 2:
            # spearmanr returns a scalar for 2-column input
            r = float(stats.spearmanr(normalised).statistic)
            r = np.clip(r, -1 + 1e-6, 1 - 1e-6)
            corr = np.array([[1.0, r], [r, 1.0]])
        else:
            corr = np.array(stats.spearmanr(normalised).statistic)

        corr = _nearest_psd(corr)
        self._cholesky = np.linalg.cholesky(corr)
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, n: int) -> pd.DataFrame:
        if self._cholesky is None:
            raise RuntimeError("Call fit() before generate().")

        d = len(self._columns)
        z = self._rng.standard_normal((n, d))
        correlated = z @ self._cholesky.T  # shape (n, d)

        result: Dict[str, np.ndarray] = {}
        for i, col in enumerate(self._columns):
            u = stats.norm.cdf(correlated[:, i])
            u = np.clip(u, 1e-10, 1 - 1e-10)
            result[col] = self._marginals[col].from_uniform(u)

        return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _EmpiricalCDF:
    """Rank-based empirical CDF with linear interpolation for inversion."""

    def __init__(self, values: np.ndarray) -> None:
        self._sorted = np.sort(values)
        n = len(values)
        self._quantiles = (np.arange(1, n + 1) - 0.5) / n
        self._dtype = values.dtype

    def to_uniform(self, x: np.ndarray) -> np.ndarray:
        u = np.interp(x, self._sorted, self._quantiles, left=0.0, right=1.0)
        return np.clip(u, 1e-10, 1 - 1e-10)

    def to_normal(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.ppf(self.to_uniform(x))

    def from_uniform(self, u: np.ndarray) -> np.ndarray:
        result = np.interp(u, self._quantiles, self._sorted)
        if np.issubdtype(self._dtype, np.integer):
            return np.round(result).astype(self._dtype)
        return result.astype(self._dtype)


def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
    """Project to nearest positive semi-definite matrix (Higham 2002)."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-8)
    psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(psd))
    return psd / np.outer(d, d)
