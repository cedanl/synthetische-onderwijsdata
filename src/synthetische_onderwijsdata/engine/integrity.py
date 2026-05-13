"""Referential integrity registry — tracks PKs and enforces FK constraints."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class IntegrityRegistry:
    """
    Maintains a mapping of table.column -> unique primary-key values so that
    foreign-key sampling always stays within the valid PK pool.
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)
        self._pks: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # PK management
    # ------------------------------------------------------------------

    def register_pk(self, table: str, column: str, values: np.ndarray) -> None:
        self._pks[f"{table}.{column}"] = np.unique(values)

    def pk_values(self, table: str, column: str) -> np.ndarray:
        key = f"{table}.{column}"
        if key not in self._pks:
            raise KeyError(
                f"No primary keys registered for {key}. "
                "Generate the parent table first."
            )
        return self._pks[key]

    # ------------------------------------------------------------------
    # FK sampling
    # ------------------------------------------------------------------

    def sample_fk(
        self,
        ref_table: str,
        ref_column: str,
        n: int,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Draw *n* FK values uniformly (or with *weights*) from the registered
        PK pool for *ref_table.ref_column*.
        """
        pool = self.pk_values(ref_table, ref_column)
        if weights is not None:
            weights = weights / weights.sum()
        return self._rng.choice(pool, size=n, replace=True, p=weights)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_fk(
        self, fk_values: np.ndarray, ref_table: str, ref_column: str
    ) -> bool:
        valid = set(self.pk_values(ref_table, ref_column).tolist())
        return bool(np.all(np.isin(fk_values, list(valid))))
