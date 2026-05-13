"""
Main synthesis engine — implements the Incremental Relational Generator (IRG)
principle: tables are generated in topological order so every FK reference is
guaranteed to resolve to an already-generated PK.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from relsynth.core.gcm import GCMEngine
from relsynth.core.integrity import IntegrityRegistry
from relsynth.core.topology import SchemaGraph
from relsynth.logic.degree import DegreeModel
from relsynth.logic.sequential import SequentialGenerator
from relsynth.presets.hooks import HookEngine
from relsynth.presets.loader import ColumnSchema, Schema, TableSchema


class RelationalSynthesizer:
    """
    Fits statistical sub-models on real data and generates synthetic
    relational tables that preserve:
      - referential integrity (FK → PK),
      - column correlations (GCM / Gaussian copula),
      - temporal structure of longitudinal records (DBN transitions).
    """

    def __init__(self, schema: Schema, random_state: Optional[int] = None) -> None:
        self.schema = schema
        self._seed = random_state
        self._rng = np.random.default_rng(random_state)
        self._registry = IntegrityRegistry(random_state)
        self._hooks = HookEngine()
        self._gcm: Dict[str, GCMEngine] = {}
        self._degree: Dict[str, DegreeModel] = {}
        self._sequential: Dict[str, SequentialGenerator] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: Dict[str, pd.DataFrame]) -> "RelationalSynthesizer":
        """Fit all sub-models to a dictionary of real DataFrames."""
        for table_name in self._topological_order():
            table = self.schema.tables[table_name]
            df = data[table_name]
            self._fit_table(table_name, table, df)
        self._fitted = True
        return self

    def _fit_table(
        self, name: str, table: TableSchema, df: pd.DataFrame
    ) -> None:
        numeric_cols = self._numeric_feature_cols(name, table)
        if numeric_cols:
            gcm = GCMEngine(self._child_seed())
            gcm.fit(df[numeric_cols])
            self._gcm[name] = gcm

        if table.sequential:
            entity_key: str = table.sequential["entity_key"]
            dm = DegreeModel(self._child_seed())
            dm.fit(df, entity_key)
            self._degree[name] = dm

            sg = SequentialGenerator(table, self._child_seed())
            sg.fit(df)
            self._sequential[name] = sg

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self, n_entities: Dict[str, int]
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all tables in topological order.

        Parameters
        ----------
        n_entities:
            Mapping of *dimension* table name → number of rows to generate.
            Fact-table sizes are derived from their parent dimensions.
        """
        generated: Dict[str, pd.DataFrame] = {}

        for table_name in self._topological_order():
            table = self.schema.tables[table_name]

            if table.table_type == "dimension":
                n = n_entities.get(table_name, 100)
                df = self._generate_dimension(table_name, table, n)
            else:
                df = self._generate_fact(table_name, table, generated)

            df = self._hooks.apply(df, table_name, self.schema)
            generated[table_name] = df

            # Register generated PKs for downstream FK sampling
            for col_name, col in table.columns.items():
                if col.role == "primary_key":
                    self._registry.register_pk(table_name, col_name, df[col_name].to_numpy())

        return generated

    # ------------------------------------------------------------------
    # Dimension generation
    # ------------------------------------------------------------------

    def _generate_dimension(
        self, name: str, table: TableSchema, n: int
    ) -> pd.DataFrame:
        result: Dict[str, Any] = {}
        gcm_cols = set(self._gcm[name]._columns) if name in self._gcm else set()

        for col_name, col in table.columns.items():
            if col.role == "primary_key":
                result[col_name] = self._make_pk(col, n)
            elif col.role == "foreign_key":
                result[col_name] = self._registry.sample_fk(
                    col.references_table, col.references_column, n  # type: ignore[arg-type]
                )
            elif col_name not in gcm_cols:
                result[col_name] = self._generate_column(col, n)

        if name in self._gcm:
            gcm_df = self._gcm[name].generate(n)
            for c in gcm_df.columns:
                result[c] = gcm_df[c].to_numpy()

        return pd.DataFrame(result)

    # ------------------------------------------------------------------
    # Fact generation
    # ------------------------------------------------------------------

    def _generate_fact(
        self,
        name: str,
        table: TableSchema,
        generated: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        if table.sequential and name in self._sequential:
            return self._generate_sequential_fact(name, table, generated)
        return self._generate_flat_fact(name, table, generated)

    def _generate_sequential_fact(
        self,
        name: str,
        table: TableSchema,
        generated: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        entity_key: str = table.sequential["entity_key"]  # type: ignore[index]
        parent_name = self._find_parent_table(table, entity_key)
        parent_df = generated[parent_name]
        degree_model = self._degree.get(name)
        return self._sequential[name].generate(
            parent_df, entity_key, degree_model, self._registry
        )

    def _generate_flat_fact(
        self,
        name: str,
        table: TableSchema,
        generated: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        n = self._estimate_fact_size(table, generated)
        result: Dict[str, Any] = {}
        gcm_cols = set(self._gcm[name]._columns) if name in self._gcm else set()

        for col_name, col in table.columns.items():
            if col.role == "primary_key":
                result[col_name] = self._make_pk(col, n)
            elif col.role == "foreign_key":
                result[col_name] = self._registry.sample_fk(
                    col.references_table, col.references_column, n  # type: ignore[arg-type]
                )
            elif col_name not in gcm_cols:
                result[col_name] = self._generate_column(col, n)

        if name in self._gcm:
            gcm_df = self._gcm[name].generate(n)
            for c in gcm_df.columns:
                result[c] = gcm_df[c].to_numpy()

        return pd.DataFrame(result)

    # ------------------------------------------------------------------
    # Column-level helpers
    # ------------------------------------------------------------------

    def _generate_column(self, col: ColumnSchema, n: int) -> np.ndarray:
        if col.dtype == "categorical":
            return self._sample_categorical(col, n)
        if col.dtype in ("integer", "float", "numeric"):
            lo = col.min_val if col.min_val is not None else 0
            hi = col.max_val if col.max_val is not None else lo + 100
            if col.dtype == "integer":
                return self._rng.integers(int(lo), int(hi) + 1, size=n)
            return self._rng.uniform(lo, hi, size=n)
        # date / string — return placeholder
        return np.full(n, None, dtype=object)

    def _sample_categorical(self, col: ColumnSchema, n: int) -> np.ndarray:
        cats = col.categories or ["unknown"]
        probs = col.probabilities
        if probs is not None:
            probs = np.array(probs, dtype=float)
            probs /= probs.sum()
        return self._rng.choice(cats, size=n, p=probs)

    def _make_pk(self, col: ColumnSchema, n: int) -> np.ndarray:
        start = int(col.min_val) if col.min_val is not None else 1
        return np.arange(start, start + n, dtype=np.int64)

    # ------------------------------------------------------------------
    # Schema utilities
    # ------------------------------------------------------------------

    def _topological_order(self) -> List[str]:
        g = SchemaGraph()
        for t in self.schema.tables:
            g.add_table(t)
        for t_name, table in self.schema.tables.items():
            for col in table.columns.values():
                if col.role == "foreign_key" and col.references_table:
                    g.add_dependency(col.references_table, t_name)
        return g.topological_order()

    def _find_parent_table(self, table: TableSchema, entity_key: str) -> str:
        col = table.columns.get(entity_key)
        if col and col.role == "foreign_key" and col.references_table:
            return col.references_table
        raise ValueError(
            f"Entity key '{entity_key}' is not a foreign key in table "
            f"'{table.name}'. Cannot determine parent table."
        )

    def _estimate_fact_size(
        self, table: TableSchema, generated: Dict[str, pd.DataFrame]
    ) -> int:
        for col in table.columns.values():
            if col.role == "foreign_key" and col.references_table in generated:
                return len(generated[col.references_table]) * 3
        return 300

    def _numeric_feature_cols(self, name: str, table: TableSchema) -> List[str]:
        return [
            c
            for c, col in table.columns.items()
            if col.dtype in ("integer", "float", "numeric")
            and col.role not in ("primary_key", "foreign_key")
        ]

    def _child_seed(self) -> int:
        return int(self._rng.integers(0, 2**31))
