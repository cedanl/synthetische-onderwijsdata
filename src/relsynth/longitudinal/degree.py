"""
Graadmodel: voorspelt hoeveel feiten-rijen bij elke entiteit horen.

Gebruikt een Negatief-Binomiale verdeling (mu, variantie) gefit op empirische
tellingen per entiteit.  Aanvullend: `from_config` voor schema-only generatie.
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
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        mean: float,
        dispersion: float,
        random_state: Optional[int] = None,
    ) -> "DegreeModel":
        """Maak een DegreeModel aan zonder echte data, op basis van YAML-config."""
        model = cls(random_state)
        model._mu = mean
        model._var = mean + dispersion   # var = mu + dispersion (NB-parameterisatie)
        model._min = 1
        model._max = None
        return model

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame, entity_key: str) -> "DegreeModel":
        counts = data.groupby(entity_key).size().to_numpy(dtype=float)
        self._mu = float(counts.mean())
        self._var = float(counts.var()) if len(counts) > 1 else self._mu + 1.0
        self._var = max(self._var, self._mu + 1e-3)
        self._min = int(counts.min())
        self._max = int(counts.max())
        return self

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, n_entities: int) -> np.ndarray:
        """Geef een array van *n_entities* graadwaarden terug."""
        p = min(self._mu / self._var, 1 - 1e-6)
        r = max(self._mu**2 / (self._var - self._mu), 0.1)
        counts = self._rng.negative_binomial(r, p, size=n_entities)
        counts = np.clip(counts, self._min, self._max)
        return counts.astype(int)
