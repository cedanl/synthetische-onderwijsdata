"""
State-aware sequentiële generator voor longitudinale records.

Genereert elke tijdstap via DBN-transities vanuit de vorige toestand.
Het aantal stappen per entiteit wordt bepaald door het DegreeModel.

Twee patronen die expliciet gemodelleerd worden:
- Tussenjaren: de gap tussen opeenvolgende inschrijvingen wordt geleerd als
  empirische verdeling (meeste gaps zijn 1, maar 2 of 3 zijn mogelijk).
- Switchers: per FK-kolom (opleiding, instelling) wordt een switchkans gefit.
  Bij elke tijdstap kan de FK-waarde met kans p_switch veranderen.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

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
        # Empirische gapverdeling: gaps van 1 jaar zijn normaal, 2+ zijn tussenjaren
        self._gap_values: np.ndarray = np.array([1], dtype=int)
        self._gap_probs: np.ndarray = np.array([1.0])
        # Per FK-kolom (excl. entity_key): P(switch naar andere waarde op tijdstap t)
        self._fk_switch_probs: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: pd.DataFrame) -> "SequentialGenerator":
        seq_cfg = self._schema.sequential  # type: ignore[union-attr]
        entity_key: str = seq_cfg["entity_key"]
        time_key: Optional[str] = seq_cfg.get("time_key")

        present = set(data.columns)
        cat_cols = [
            c for c, col in self._schema.columns.items()
            if col.dtype == "categorical"
            and c not in {entity_key, time_key}
            and col.role not in ("primary_key", "foreign_key")
            and c in present
        ]
        num_cols = [
            c for c, col in self._schema.columns.items()
            if col.dtype in ("integer", "float", "numeric")
            and c not in {entity_key, time_key}
            and col.role not in ("primary_key", "foreign_key")
            and c in present
            and is_numeric_dtype(data[c])
        ]
        self._feature_cols = cat_cols + num_cols

        sequences: List[pd.DataFrame] = []
        initial_rows: List[Dict[str, Any]] = []
        all_gaps: List[int] = []

        # FK-kolommen die kunnen switchen (alles behalve de entity_key zelf)
        switchable_fk_cols = [
            c for c, col in self._schema.columns.items()
            if col.role == "foreign_key" and c != entity_key and c in data.columns
        ]
        fk_switch_counts: Dict[str, int] = {c: 0 for c in switchable_fk_cols}
        fk_transition_counts: Dict[str, int] = {c: 0 for c in switchable_fk_cols}

        for _, group in data.groupby(entity_key):
            if time_key and time_key in group.columns:
                group = group.sort_values(time_key)

            sequences.append(group[self._feature_cols].reset_index(drop=True))
            initial_rows.append(group.iloc[0].to_dict())

            # Gapverdeling leren
            if time_key and time_key in group.columns and len(group) > 1:
                years = group[time_key].to_numpy(dtype=int)
                gaps = np.diff(years)
                all_gaps.extend(gaps[gaps > 0].tolist())

            # Switchkans per FK-kolom leren
            for col_name in switchable_fk_cols:
                vals = group[col_name].to_numpy()
                if len(vals) > 1:
                    fk_switch_counts[col_name] += int(np.sum(vals[1:] != vals[:-1]))
                    fk_transition_counts[col_name] += len(vals) - 1

        self._initial_states = pd.DataFrame(initial_rows).reset_index(drop=True)
        self._transition = TransitionModel(int(self._rng.integers(0, 2**31)))
        self._transition.fit(sequences, cat_cols, num_cols)

        if all_gaps:
            vals, counts = np.unique(all_gaps, return_counts=True)
            self._gap_values = vals.astype(int)
            self._gap_probs = counts / counts.sum()

        for col_name in switchable_fk_cols:
            n_trans = fk_transition_counts[col_name]
            if n_trans > 0:
                self._fk_switch_probs[col_name] = fk_switch_counts[col_name] / n_trans

        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        parent_df: pd.DataFrame,
        entity_key: str,
        degree_model: Optional["DegreeModel"],
        fk_pools: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        fk_pools:
            Dict van FK-kolomnaam → array van geldige PK-waarden uit de al
            gegenereerde dimensietabel. Zonder dit dict worden FK-waarden
            niet geswitcht.
        """
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
            # entity_key in het row-dict zetten, niet in de state Series
            # (voorkomt dtype-conflict tussen trainingsdata en gegenereerde PKs)

            # Begintijd uit initiële toestand; valt terug op 2020
            current_time = int(state.get(time_key, 2020)) if time_key else 2020

            # Initiële FK-waarden: sample uit pool zodat we geldige synthetische
            # waarden hebben (niet de echte codes uit de trainingsdata)
            fk_state: Dict[str, Any] = {}
            if fk_pools:
                for col_name, pool in fk_pools.items():
                    fk_state[col_name] = self._rng.choice(pool)

            for t in range(int(n_steps)):
                row: Dict[str, Any] = state.to_dict()
                row[entity_key] = entity_id

                if pk_col:
                    row[pk_col] = pk_counter
                    pk_counter += 1
                if time_key:
                    row[time_key] = current_time

                # FK-waarden uit fk_state overschrijven wat er in state zit
                for col_name, val in fk_state.items():
                    row[col_name] = val

                for col_name in self._schema.columns:
                    row.setdefault(col_name, None)
                all_rows.append(row)

                if t < int(n_steps) - 1:
                    # Tijdstap vooruit met empirische gap (tussenjaar mogelijk)
                    if time_key:
                        gap = int(self._rng.choice(self._gap_values, p=self._gap_probs))
                        current_time += gap

                    # Feature-toestand via DBN-transitie
                    if self._transition is not None:
                        state = self._transition.step(state)

                    # FK-switching: met kans p_switch een andere waarde samplen
                    if fk_pools:
                        for col_name, pool in fk_pools.items():
                            p = self._fk_switch_probs.get(col_name, 0.0)
                            if p > 0.0 and self._rng.random() < p:
                                fk_state[col_name] = self._rng.choice(pool)

        return pd.DataFrame(all_rows)

    def _sample_initial_state(self) -> pd.Series:
        if self._initial_states is not None and len(self._initial_states) > 0:
            idx = int(self._rng.integers(0, len(self._initial_states)))
            return self._initial_states.iloc[idx].copy()
        return pd.Series(dtype=object)
