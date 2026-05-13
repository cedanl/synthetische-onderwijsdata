"""
Laad een relsynth-preset vanuit een YAML-bestand.

Gebruik `schema.py` voor de dataklassen zelf.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from relsynth.schema import ColumnSchema, Schema, TableSchema


class PresetLoader:
    """Laad en parseer een relsynth-schema vanuit een YAML-bestand."""

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
            dim_keys=raw.get("dim_keys", {}),
        )
