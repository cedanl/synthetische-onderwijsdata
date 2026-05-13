"""
Dynamisch Bayesiaans Netwerk (DBN) transitiemodel.

Modelleert P(X_t | X_{t-1}) per kolom:
  - Categorisch: empirische Markov-transitiematrix (Laplace-glad)
  - Numeriek:    AR(1)-proces gefit via OLS
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TransitionModel:
    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)
        self._cat: Dict[str, Tuple[List, np.ndarray]] = {}
        self._ar1: Dict[str, Tuple[float, float, float]] = {}

    def fit(
        self,
        sequences: List[pd.DataFrame],
        categorical_cols: List[str],
        numeric_cols: List[str],
    ) -> "TransitionModel":
        for col in categorical_cols:
            self._fit_categorical(sequences, col)
        for col in numeric_cols:
            self._fit_ar1(sequences, col)
        return self

    def _fit_categorical(self, sequences: List[pd.DataFrame], col: str) -> None:
        frames = [seq[[col]].assign(_g=i) for i, seq in enumerate(sequences)]
        if not frames:
            return
        combined = pd.concat(frames, ignore_index=True)
        combined["_prev"] = combined.groupby("_g", sort=False)[col].shift(1)
        mask = combined[col].notna() & combined["_prev"].notna()
        pairs = combined.loc[mask]
        if pairs.empty:
            return
        all_vals = sorted(set(pairs[col].unique()) | set(pairs["_prev"].unique()))
        n = len(all_vals)
        idx = {v: i for i, v in enumerate(all_vals)}
        counts = np.ones((n, n))  # Laplace smoothing
        np.add.at(counts, (pairs["_prev"].map(idx).to_numpy(dtype=int),
                           pairs[col].map(idx).to_numpy(dtype=int)), 1)
        probs = counts / counts.sum(axis=1, keepdims=True)
        self._cat[col] = (all_vals, probs)

    def _fit_ar1(self, sequences: List[pd.DataFrame], col: str) -> None:
        frames = [seq[[col]].assign(_g=i) for i, seq in enumerate(sequences)]
        if not frames:
            self._ar1[col] = (0.0, 1.0, 1.0)
            return
        combined = pd.concat(frames, ignore_index=True)
        combined["_prev"] = combined.groupby("_g", sort=False)[col].shift(1)
        mask = combined[col].notna() & combined["_prev"].notna()
        pairs = combined.loc[mask]
        if len(pairs) < 2:
            self._ar1[col] = (0.0, 1.0, 1.0)
            return
        xp = pairs["_prev"].to_numpy(dtype=float)
        xc = pairs[col].to_numpy(dtype=float)
        beta = np.cov(xp, xc)[0, 1] / (np.var(xp) + 1e-10)
        alpha = float(xc.mean()) - beta * float(xp.mean())
        sigma = float(np.std(xc - (alpha + beta * xp))) + 1e-6
        self._ar1[col] = (alpha, beta, sigma)

    def step(self, state: pd.Series) -> pd.Series:
        next_state = state.copy()
        for col, (vals, probs) in self._cat.items():
            if col not in state or pd.isna(state[col]):
                continue
            idx_map = {v: i for i, v in enumerate(vals)}
            curr_i = idx_map.get(state[col], 0)
            next_state[col] = vals[int(self._rng.choice(len(vals), p=probs[curr_i]))]
        for col, (alpha, beta, sigma) in self._ar1.items():
            if col not in state or pd.isna(state[col]):
                continue
            next_state[col] = alpha + beta * float(state[col]) + self._rng.normal(0, sigma)
        return next_state
