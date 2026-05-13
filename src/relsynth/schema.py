"""
Datamodellen voor een relsynth-schema.

Deze module bevat alleen pure dataklassen — geen I/O, geen YAML, geen logica.
Importeer hier vanuit als je schema-objecten nodig hebt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ColumnSchema:
    name: str
    dtype: str                               # integer | float | categorical | date | string
    role: Optional[str] = None               # primary_key | foreign_key | None
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
    table_type: str                          # dimension | fact
    columns: Dict[str, ColumnSchema]
    sequential: Optional[Dict[str, Any]] = None
    hooks: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class Schema:
    name: str
    tables: Dict[str, TableSchema]
    degree_config: Dict[str, Any] = field(default_factory=dict)
    dim_keys: Dict[str, str] = field(default_factory=dict)
