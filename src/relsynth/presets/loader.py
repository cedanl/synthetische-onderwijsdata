"""
YAML preset loader — converts schema definitions into typed dataclasses
consumed by the engine and logic layers.

Preset YAML structure
---------------------
name: <str>
tables:
  <table_name>:
    type: dimension | fact
    columns:
      <col_name>:
        dtype: integer | float | numeric | categorical | date | string
        role: primary_key | foreign_key   # optional
        references: <table>.<column>      # required when role=foreign_key
        categories: [...]                 # required when dtype=categorical
        probabilities: [...]              # optional, must sum to 1
        min: <number>                     # optional, for numeric types
        max: <number>                     # optional
        nullable: bool                    # optional, default false
    sequential:                           # optional, for longitudinal facts
      entity_key: <col_name>
      time_key: <col_name>
    hooks:                                # optional business rules
      - rule: "col_a > col_b"
        condition: "col_a is not null"
degree:
  <fact_table>:
    distribution: negative_binomial
    mean: 4.2
    dispersion: 1.8
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ColumnSchema:
    name: str
    dtype: str
    role: Optional[str] = None
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    categories: Optional[List[Any]] = None
    probabilities: Optional[List[float]] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    nullable: bool = False


@dataclass
class TableSchema:
    name: str
    table_type: str
    columns: Dict[str, ColumnSchema]
    sequential: Optional[Dict[str, Any]] = None
    hooks: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Schema:
    name: str
    tables: Dict[str, TableSchema]
    degree_config: Dict[str, Any] = field(default_factory=dict)


class PresetLoader:
    """Load and parse a relsynth schema preset from a YAML file."""

    BUILTIN_DIR: Path = Path(__file__).parent / "schemas"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Schema:
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        return cls._parse(raw)

    @classmethod
    def from_builtin(cls, name: str) -> Schema:
        path = cls.BUILTIN_DIR / f"{name}.yaml"
        if not path.exists():
            available = sorted(p.stem for p in cls.BUILTIN_DIR.glob("*.yaml"))
            raise FileNotFoundError(
                f"Built-in preset '{name}' not found.  "
                f"Available presets: {available}"
            )
        return cls.from_yaml(path)

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    @classmethod
    def _parse(cls, raw: Dict[str, Any]) -> Schema:
        tables: Dict[str, TableSchema] = {}
        for t_name, t_def in raw.get("tables", {}).items():
            columns: Dict[str, ColumnSchema] = {}
            for c_name, c_def in t_def.get("columns", {}).items():
                ref_table: Optional[str] = None
                ref_col: Optional[str] = None
                if c_def.get("role") == "foreign_key" and "references" in c_def:
                    ref_table, ref_col = c_def["references"].split(".", 1)
                columns[c_name] = ColumnSchema(
                    name=c_name,
                    dtype=c_def.get("dtype", "string"),
                    role=c_def.get("role"),
                    references_table=ref_table,
                    references_column=ref_col,
                    categories=c_def.get("categories"),
                    probabilities=c_def.get("probabilities"),
                    min_val=c_def.get("min"),
                    max_val=c_def.get("max"),
                    nullable=c_def.get("nullable", False),
                )
            tables[t_name] = TableSchema(
                name=t_name,
                table_type=t_def.get("type", "dimension"),
                columns=columns,
                sequential=t_def.get("sequential"),
                hooks=t_def.get("hooks", []),
            )
        return Schema(
            name=raw.get("name", "unnamed"),
            tables=tables,
            degree_config=raw.get("degree", {}),
        )
