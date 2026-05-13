"""
Kolomniveau-generatoren.

Elke functie neemt een ColumnSchema en een rng, en geeft een numpy-array terug.
De engine gebruikt deze functies; ze bevatten geen orchestratielogica.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from relsynth.schema import ColumnSchema


def sample_column(
    col: ColumnSchema,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Genereer *n* waarden voor *col* op basis van dtype en parameters."""
    if col.dtype == "categorical":
        return _sample_categorical(col, n, rng)
    if col.dtype in ("integer", "float", "numeric"):
        return _sample_numeric(col, n, rng)
    return np.full(n, None, dtype=object)


def make_pk(col: ColumnSchema, n: int) -> np.ndarray:
    """Genereer een oplopende reeks van *n* primary-key waarden."""
    start = int(col.min_val) if col.min_val is not None else 1
    return np.arange(start, start + n, dtype=np.int64)


def _sample_categorical(
    col: ColumnSchema,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cats = col.categories or ["unknown"]
    probs: Optional[np.ndarray] = None
    if col.probabilities is not None:
        probs = np.array(col.probabilities, dtype=float)
        probs /= probs.sum()
    return rng.choice(cats, size=n, p=probs)


def _sample_numeric(
    col: ColumnSchema,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lo = col.min_val if col.min_val is not None else 0.0
    hi = col.max_val if col.max_val is not None else lo + 100.0
    if col.dtype == "integer":
        return rng.integers(int(lo), int(hi) + 1, size=n)
    return rng.uniform(lo, hi, size=n)
