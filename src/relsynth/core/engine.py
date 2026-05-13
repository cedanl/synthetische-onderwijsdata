"""
IRG-synthese-engine.

Verantwoordelijkheid: tabellen in topologische volgorde genereren zodat elke
FK verwijst naar een al bestaande PK.  Kolomgeneratie is gedelegeerd aan
`core.samplers`; longitudinale logica aan `longitudinal.*`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from relsynth.core.gcm import GCMEngine
from relsynth.core.integrity import IntegrityRegistry
from relsynth.core.samplers import make_pk, sample_column
from relsynth.core.topology import SchemaGraph
from relsynth.longitudinal.degree import DegreeModel
from relsynth.longitudinal.sequential import SequentialGenerator
from relsynth.presets.hooks import HookEngine
from relsynth.schema import Schema, TableSchema


class RelationalSynthesizer:
    """
    Past statistische deelmodellen aan op echte data en genereert synthetische
    relationele tabellen met behoud van:
      - referentiële integriteit (FK → PK),
      - kolomcorrelaties (GCM / Gaussian copula),
      - temporele structuur van longitudinale records (DBN-transities).
    """

    def __init__(self, schema: Schema, random_state: Optional[int] = None) -> None:
        self.schema = schema
        self._rng = np.random.default_rng(random_state)
        self._registry = IntegrityRegistry(random_state)
        self._hooks = HookEngine()
        self._gcm: Dict[str, GCMEngine] = {}
        self._degree: Dict[str, DegreeModel] = {}
        self._sequential: Dict[str, SequentialGenerator] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, data: Dict[str, pd.DataFrame]) -> "RelationalSynthesizer":
        """Pas alle deelmodellen aan op een dict van echte DataFrames."""
        for table_name in self._topological_order():
            table = self.schema.tables[table_name]
            df = data[table_name]

            numeric_cols = _numeric_feature_cols(table)
            if numeric_cols:
                gcm = GCMEngine(self._child_seed())
                gcm.fit(df[numeric_cols])
                self._gcm[table_name] = gcm

            if table.sequential:
                entity_key: str = table.sequential["entity_key"]
                dm = DegreeModel(self._child_seed())
                dm.fit(df, entity_key)
                self._degree[table_name] = dm

                sg = SequentialGenerator(table, self._child_seed())
                sg.fit(df)
                self._sequential[table_name] = sg

        return self

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, n_entities: Dict[str, int]) -> Dict[str, pd.DataFrame]:
        """
        Genereer alle tabellen in topologische volgorde.

        Parameters
        ----------
        n_entities:
            Dimensietabelnaam → aantal rijen.  Feitentabelgroottes worden
            afgeleid van de bijbehorende ouderdimensies.
        """
        generated: Dict[str, pd.DataFrame] = {}

        for table_name in self._topological_order():
            table = self.schema.tables[table_name]

            if table.table_type == "dimension":
                n = n_entities.get(table_name, 100)
                df = self._generate_table(table_name, table, n)
            else:
                df = self._generate_fact(table_name, table, generated)

            df = self._hooks.apply(df, table_name, self.schema)
            generated[table_name] = df

            for col_name, col in table.columns.items():
                if col.role == "primary_key":
                    self._registry.register_pk(
                        table_name, col_name, df[col_name].to_numpy()
                    )

        return generated

    # ------------------------------------------------------------------
    # Tabelgeneratie
    # ------------------------------------------------------------------

    def _generate_table(
        self, name: str, table: TableSchema, n: int
    ) -> pd.DataFrame:
        """Genereer *n* rijen voor een dimensie- of platte feitentabel."""
        gcm_cols = set(self._gcm[name]._columns) if name in self._gcm else set()
        result: Dict[str, Any] = {}

        for col_name, col in table.columns.items():
            if col.role == "primary_key":
                result[col_name] = make_pk(col, n)
            elif col.role == "foreign_key":
                result[col_name] = self._registry.sample_fk(
                    col.references_table, col.references_column, n  # type: ignore[arg-type]
                )
            elif col_name not in gcm_cols:
                result[col_name] = sample_column(col, n, self._rng)

        if name in self._gcm:
            for col_name, values in self._gcm[name].generate(n).items():
                result[col_name] = values.to_numpy()

        return pd.DataFrame(result)

    def _generate_fact(
        self,
        name: str,
        table: TableSchema,
        generated: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        if table.sequential and name in self._sequential:
            entity_key: str = table.sequential["entity_key"]  # type: ignore[index]
            parent_name = _find_parent_table(table, entity_key)
            parent_df = generated[parent_name]
            degree_model = self._degree.get(name) or _degree_from_config(
                name, self.schema, self._child_seed()
            )
            return self._sequential[name].generate(parent_df, entity_key, degree_model)

        n = _estimate_fact_size(table, generated)
        return self._generate_table(name, table, n)

    # ------------------------------------------------------------------
    # Hulpfuncties
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

    def _child_seed(self) -> int:
        return int(self._rng.integers(0, 2**31))


# ---------------------------------------------------------------------------
# Module-niveau hulpfuncties (geen enginetoestand nodig)
# ---------------------------------------------------------------------------

def _numeric_feature_cols(table: TableSchema) -> List[str]:
    return [
        c
        for c, col in table.columns.items()
        if col.dtype in ("integer", "float", "numeric")
        and col.role not in ("primary_key", "foreign_key")
    ]


def _find_parent_table(table: TableSchema, entity_key: str) -> str:
    col = table.columns.get(entity_key)
    if col and col.role == "foreign_key" and col.references_table:
        return col.references_table
    raise ValueError(
        f"Entity key '{entity_key}' is geen foreign key in tabel '{table.name}'."
    )


def _estimate_fact_size(
    table: TableSchema, generated: Dict[str, pd.DataFrame]
) -> int:
    for col in table.columns.values():
        if col.role == "foreign_key" and col.references_table in generated:
            return len(generated[col.references_table]) * 3
    return 300


def _degree_from_config(
    table_name: str, schema: Schema, seed: int
) -> Optional[DegreeModel]:
    """Maak een DegreeModel aan uit de YAML degree_config als er geen fit is."""
    cfg = schema.degree_config.get(table_name)
    if not cfg:
        return None
    return DegreeModel.from_config(
        mean=cfg.get("mean", 3.0),
        dispersion=cfg.get("dispersion", 1.5),
        random_state=seed,
    )
