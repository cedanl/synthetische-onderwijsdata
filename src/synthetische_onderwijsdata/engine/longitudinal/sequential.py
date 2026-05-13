"""
State-aware sequentiële generator voor longitudinale records.

Genereert elke tijdstap via DBN-transities vanuit de vorige toestand.
Het aantal stappen per entiteit wordt bepaald door het DegreeModel.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import pandas as pd

from synthetische_onderwijsdata.engine.longitudinal.dbn import TransitionModel

if TYPE_CHECKING:
    from synthetische_onderwijsdata.engine.longitudinal.degree import DegreeModel
    from synthetische_onderwijsdata.schema import TableSchema


class SequentialGenerator:
    def __init__(
        self, table_schema: "TableSchema", random_state: Optional[int] = None
    ) -> None:
        self._schema = table_schema
        self._rng = np.random.default_rng(random_state)
        self._transition: Optional[TransitionModel] = None
        self._initial_states: Optional[pd.DataFrame] = None
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "SequentialGenerator":
        seq_cfg = self._schema.sequential  # type: ignore[union-attr]
        entity_key: str = seq_cfg["entity_key"]
        time_key: Optional[str] = seq_cfg.get("time_key")

        cat_cols = [
            c for c, col in self._schema.columns.items()
            if col.dtype == "categorical"
            and c not in {entity_key, time_key}
            and col.role not in ("primary_key", "foreign_key")
        ]
        num_cols = [
            c for c, col in self._schema.columns.items()
            if col.dtype in ("integer", "float", "numeric")
            and c not in {entity_key, time_key}
            and col.role not in ("primary_key", "foreign_key")
        ]
        self._feature_cols = cat_cols + num_cols

        sequences: List[pd.DataFrame] = []
        initial_rows: List[Dict[str, Any]] = []

        for _, group in data.groupby(entity_key):
            if time_key and time_key in group.columns:
                group = group.sort_values(time_key)
            sequences.append(group[self._feature_cols].reset_index(drop=True))
            initial_rows.append(group.iloc[0].to_dict())

        self._initial_states = pd.DataFrame(initial_rows).reset_index(drop=True)
        self._transition = TransitionModel(int(self._rng.integers(0, 2**31)))
        self._transition.fit(sequences, cat_cols, num_cols)
        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        parent_df: pd.DataFrame,
        entity_key: str,
        degree_model: Optional["DegreeModel"],
    ) -> pd.DataFrame:
        seq_cfg = self._schema.sequential  # type: ignore[union-attr]
        time_key: Optional[str] = seq_cfg.get("time_key")
        pk_col = next(
            (c for c, col in self._schema.columns.items() if col.role == "primary_key"),
            None,
        )

        degrees = (
            degree_model.sample(len(parent_df))
            if degree_model is not None
            else np.full(len(parent_df), 3, dtype=int)
        )

        all_rows: List[Dict[str, Any]] = []
        pk_counter = 1

        for (_, parent_row), n_steps in zip(parent_df.iterrows(), degrees):
            entity_id = parent_row[entity_key]
            state = self._sample_initial_state()
            state[entity_key] = entity_id
            base_time = int(state.get(time_key, 2020)) if time_key else 2020

            for t in range(int(n_steps)):
                row: Dict[str, Any] = state.to_dict()
                if pk_col:
                    row[pk_col] = pk_counter
                    pk_counter += 1
                if time_key:
                    row[time_key] = base_time + t
                for col_name in self._schema.columns:
                    row.setdefault(col_name, None)
                all_rows.append(row)

                if t < int(n_steps) - 1 and self._transition is not None:
                    state = self._transition.step(state)

        return pd.DataFrame(all_rows)

    def _sample_initial_state(self) -> pd.Series:
        if self._initial_states is not None and len(self._initial_states) > 0:
            idx = int(self._rng.integers(0, len(self._initial_states)))
            return self._initial_states.iloc[idx].copy()
        return pd.Series(dtype=object)
