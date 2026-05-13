"""
Degree model: predicts how many fact-table records belong to each entity.

Uses a Negative Binomial distribution (mu, dispersion) fitted on empirical
per-entity record counts.  This naturally handles the over-dispersion common
in student trajectory data (most students have 3–4 years; a few have 8+).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class DegreeModel:
    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)
        self._mu: float = 3.0
        self._var: float = 4.0
        self._min: int = 1
        self._max: Optional[int] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, entity_key: str) -> "DegreeModel":
        counts = data.groupby(entity_key).size().to_numpy(dtype=float)
        self._mu = float(counts.mean())
        self._var = float(counts.var()) if len(counts) > 1 else self._mu + 1.0
        self._var = max(self._var, self._mu + 1e-3)  # var > mu for NB
        self._min = int(counts.min())
        self._max = int(counts.max())
        return self

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, n_entities: int) -> np.ndarray:
        """
        Return an array of *n_entities* integer degree values drawn from a
        Negative Binomial parameterised by (mu, var).
        """
        # NB parameterisation: p = mu/var,  r = mu²/(var - mu)
        p = min(self._mu / self._var, 1 - 1e-6)
        r = max(self._mu**2 / (self._var - self._mu), 0.1)
        counts = self._rng.negative_binomial(r, p, size=n_entities)
        counts = np.clip(counts, self._min, self._max)
        return counts.astype(int)
