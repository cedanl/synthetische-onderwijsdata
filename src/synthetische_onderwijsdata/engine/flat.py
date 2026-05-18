"""
FlatSynthesizer — entiteitsgerichte longitudinale synthesizer.

Werkt op een plat DataFrame (één rij per entiteit-tijdstap) zonder FK/PK-schema.
Cross-tabel correlaties blijven bewaard omdat alle kolommen samen in één rij zitten
tijdens fit() en generate().
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from synthetische_onderwijsdata.engine.longitudinal.dbn import TransitionModel
from synthetische_onderwijsdata.engine.longitudinal.degree import DegreeModel


class FlatSynthesizer:
    """
    Synthetiseert longitudinale platte tabellen.

    Parameters
    ----------
    entity_key : Kolomnaam die entiteiten identificeert (bijv. 'pgn').
    time_key   : Kolomnaam voor tijdstappen (bijv. 'jaar'). Optioneel.
    stable_cols: Kolommen die niet veranderen over tijdstappen binnen een entiteit.
    random_state: Seed voor reproduceerbaarheid.
    """

    def __init__(
        self,
        entity_key: str,
        time_key: Optional[str] = None,
        stable_cols: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.entity_key = entity_key
        self.time_key = time_key
        self.stable_cols = list(stable_cols or [])
        self._rng = np.random.default_rng(random_state)
        self._degree: Optional[DegreeModel] = None
        self._transition: Optional[TransitionModel] = None
        self._initial_states: Optional[pd.DataFrame] = None
        self._feature_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._num_cols: List[str] = []
        self._gap_values: np.ndarray = np.array([1], dtype=int)
        self._gap_probs: np.ndarray = np.array([1.0])

    def fit(self, data: pd.DataFrame) -> "FlatSynthesizer":
        """Pas alle deelmodellen aan op *data*."""
        if self.time_key and self.time_key in data.columns:
            data = data.sort_values([self.entity_key, self.time_key])

        exclude = {self.entity_key}
        if self.time_key:
            exclude.add(self.time_key)

        self._cat_cols = [
            c for c in data.columns
            if c not in exclude and not pd.api.types.is_numeric_dtype(data[c])
        ]
        self._num_cols = [
            c for c in data.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(data[c])
        ]
        self._feature_cols = self._cat_cols + self._num_cols

        self._initial_states = (
            data.drop_duplicates(subset=[self.entity_key], keep="first")
            .reset_index(drop=True)
        )

        sequences = [
            grp[self._feature_cols].reset_index(drop=True)
            for _, grp in data.groupby(self.entity_key, sort=False)
        ]
        self._transition = TransitionModel(int(self._rng.integers(0, 2**31)))
        self._transition.fit(sequences, self._cat_cols, self._num_cols)

        self._degree = DegreeModel(int(self._rng.integers(0, 2**31)))
        self._degree.fit(data, self.entity_key)

        if self.time_key and self.time_key in data.columns:
            time_num = pd.to_numeric(data[self.time_key], errors="coerce")
            gaps = time_num.groupby(data[self.entity_key], sort=False).diff().dropna()
            gaps = gaps[gaps > 0].astype(int)
            if len(gaps):
                vals, counts = np.unique(gaps.to_numpy(), return_counts=True)
                self._gap_values = vals.astype(int)
                self._gap_probs = counts / counts.sum()

        return self

    def generate(self, n_entities: int) -> pd.DataFrame:
        """Genereer *n_entities* synthetische entiteiten met hun volledige trajecten."""
        if self._transition is None:
            raise RuntimeError("Roep fit() aan voor generate().")

        degrees = self._degree.sample(n_entities)  # type: ignore[union-attr]
        all_rows: List[Dict] = []

        for i in range(n_entities):
            entity_id = f"E{i:08d}"
            n_steps = int(degrees[i])

            idx = int(self._rng.integers(0, len(self._initial_states)))  # type: ignore[arg-type]
            init_row = self._initial_states.iloc[idx]  # type: ignore[index]
            state = init_row[self._feature_cols].copy()
            stable_vals = {c: state[c] for c in self.stable_cols if c in state.index}

            current_time: Optional[int] = None
            if self.time_key:
                raw = init_row.get(self.time_key)
                try:
                    current_time = int(pd.to_numeric(raw, errors="coerce"))
                except (TypeError, ValueError):
                    current_time = 2020

            for t in range(n_steps):
                row: Dict = state.to_dict()
                row[self.entity_key] = entity_id
                if self.time_key and current_time is not None:
                    row[self.time_key] = current_time
                row.update(stable_vals)
                all_rows.append(row)

                if t < n_steps - 1:
                    if self.time_key and current_time is not None:
                        gap = int(self._rng.choice(self._gap_values, p=self._gap_probs))
                        current_time += gap
                    state = self._transition.step(state)
                    for col, val in stable_vals.items():
                        state[col] = val

        return pd.DataFrame(all_rows)
