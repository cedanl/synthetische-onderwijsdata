"""
Dynamic Bayesian Network (DBN) transition model.

Models P(X_t | X_{t-1}) for each variable independently:
  - Categorical columns: empirical Markov transition matrix (Laplace-smoothed)
  - Numeric columns: AR(1) process fitted by OLS
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TransitionModel:
    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)
        # col -> (categories, transition_matrix)
        self._cat: Dict[str, Tuple[List, np.ndarray]] = {}
        # col -> (alpha, beta, sigma)
        self._ar1: Dict[str, Tuple[float, float, float]] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

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

    def _fit_categorical(
        self, sequences: List[pd.DataFrame], col: str
    ) -> None:
        all_vals = sorted(
            {v for seq in sequences for v in seq[col].dropna().tolist()}
        )
        if not all_vals:
            return
        idx = {v: i for i, v in enumerate(all_vals)}
        n = len(all_vals)
        counts = np.ones((n, n))  # Laplace smoothing

        for seq in sequences:
            vals = seq[col].to_numpy()
            for t in range(1, len(vals)):
                if pd.notna(vals[t - 1]) and pd.notna(vals[t]):
                    counts[idx[vals[t - 1]], idx[vals[t]]] += 1

        probs = counts / counts.sum(axis=1, keepdims=True)
        self._cat[col] = (all_vals, probs)

    def _fit_ar1(self, sequences: List[pd.DataFrame], col: str) -> None:
        x_prev: List[float] = []
        x_curr: List[float] = []
        for seq in sequences:
            vals = seq[col].dropna().to_numpy(dtype=float)
            if len(vals) >= 2:
                x_prev.extend(vals[:-1].tolist())
                x_curr.extend(vals[1:].tolist())

        if not x_prev:
            self._ar1[col] = (0.0, 1.0, 1.0)
            return

        xp = np.array(x_prev)
        xc = np.array(x_curr)
        beta = np.cov(xp, xc)[0, 1] / (np.var(xp) + 1e-10)
        alpha = float(xc.mean()) - beta * float(xp.mean())
        sigma = float(np.std(xc - (alpha + beta * xp))) + 1e-6
        self._ar1[col] = (alpha, beta, sigma)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: pd.Series) -> pd.Series:
        """Produce the next-period state given the current *state*."""
        next_state = state.copy()

        for col, (vals, probs) in self._cat.items():
            if col not in state or pd.isna(state[col]):
                continue
            idx_map = {v: i for i, v in enumerate(vals)}
            curr_i = idx_map.get(state[col], 0)
            next_i = int(self._rng.choice(len(vals), p=probs[curr_i]))
            next_state[col] = vals[next_i]

        for col, (alpha, beta, sigma) in self._ar1.items():
            if col not in state or pd.isna(state[col]):
                continue
            next_val = alpha + beta * float(state[col]) + self._rng.normal(0, sigma)
            next_state[col] = next_val

        return next_state
