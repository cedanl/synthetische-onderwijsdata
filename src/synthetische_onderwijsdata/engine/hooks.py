"""
Hook-engine voor het afdwingen van declaratieve bedrijfsregels.

Ondersteunde regelvormen:
  "col_a > col_b"   — col_a moet strikt groter zijn dan col_b
  "col_a < col_b"   — col_a moet strikt kleiner zijn dan col_b

Conditiesyntax:
  "col_name is not null"  — pas de regel alleen toe waar col_name niet null is
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from synthetische_onderwijsdata.schema import Schema


class HookEngine:
    def apply(
        self, df: pd.DataFrame, table_name: str, schema: "Schema"
    ) -> pd.DataFrame:
        table = schema.tables.get(table_name)
        if not table or not table.hooks:
            return df
        for hook in table.hooks:
            df = _apply_rule(df, hook.get("rule", ""), hook.get("condition"))
        return df


def _apply_rule(
    df: pd.DataFrame, rule: str, condition: Optional[str]
) -> pd.DataFrame:
    mask = _build_mask(df, condition)
    if ">" in rule:
        bigger, smaller = [s.strip() for s in rule.split(">", 1)]
        _enforce_order(df, bigger, smaller, mask)
    elif "<" in rule:
        smaller, bigger = [s.strip() for s in rule.split("<", 1)]
        _enforce_order(df, bigger, smaller, mask)
    return df


def _enforce_order(
    df: pd.DataFrame, bigger: str, smaller: str, mask: pd.Series
) -> None:
    """Zorg dat df[bigger] > df[smaller] voor alle rijen waar mask True is.

    Correctiestrategie: verhoog df[bigger] zodat het groter wordt dan df[smaller].
    We wisselen *geen* waarden om — dat zou de semantiek omdraaien
    (bijv. datum_inschrijving wordt diplomadatum).
    """
    if bigger not in df.columns or smaller not in df.columns:
        return
    violations = mask & (df[bigger] <= df[smaller])
    if not violations.any():
        return
    # Zet bigger op smaller + 1 zodat de orde gegarandeerd is
    df.loc[violations, bigger] = df.loc[violations, smaller] + 1


def _build_mask(df: pd.DataFrame, condition: Optional[str]) -> pd.Series:
    if not condition:
        return pd.Series(True, index=df.index)
    cond_lower = condition.strip().lower()
    if "is not null" in cond_lower:
        col = cond_lower.replace("is not null", "").strip()
        if col in df.columns:
            return df[col].notna()
    return pd.Series(True, index=df.index)
