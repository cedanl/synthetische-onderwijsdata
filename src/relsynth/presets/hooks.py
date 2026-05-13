"""
Logic hook engine — enforces declarative business rules on generated DataFrames.

Supported rule syntax (column expressions):
  "col_a > col_b"   — ensure col_a is strictly greater than col_b
  "col_a < col_b"   — ensure col_a is strictly less than col_b

Condition syntax:
  "col_name is not null"  — only apply rule where col_name is non-null
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from relsynth.presets.loader import Schema


class HookEngine:
    def apply(
        self, df: pd.DataFrame, table_name: str, schema: "Schema"
    ) -> pd.DataFrame:
        table = schema.tables.get(table_name)
        if not table or not table.hooks:
            return df
        for hook in table.hooks:
            rule = hook.get("rule", "")
            condition = hook.get("condition")
            df = _apply_rule(df, rule, condition)
        return df


# ---------------------------------------------------------------------------
# Rule application helpers
# ---------------------------------------------------------------------------

def _apply_rule(
    df: pd.DataFrame, rule: str, condition: Optional[str]
) -> pd.DataFrame:
    mask = _build_mask(df, condition)
    if ">" in rule:
        left, right = [s.strip() for s in rule.split(">", 1)]
        _enforce_order(df, left, right, mask, strict_gt=True)
    elif "<" in rule:
        left, right = [s.strip() for s in rule.split("<", 1)]
        _enforce_order(df, right, left, mask, strict_gt=True)
    return df


def _enforce_order(
    df: pd.DataFrame,
    bigger: str,
    smaller: str,
    mask: pd.Series,
    strict_gt: bool,
) -> None:
    if bigger not in df.columns or smaller not in df.columns:
        return
    if strict_gt:
        violations = mask & (df[bigger] <= df[smaller])
    else:
        violations = mask & (df[bigger] < df[smaller])

    if not violations.any():
        return

    # Swap values for violating rows so the ordering holds
    b_vals = df.loc[violations, bigger].to_numpy().copy()
    s_vals = df.loc[violations, smaller].to_numpy().copy()
    df.loc[violations, bigger] = s_vals
    df.loc[violations, smaller] = b_vals

    # If after swap they are still equal (both same value), nudge bigger up
    still_equal = mask & (df[bigger] == df[smaller])
    if still_equal.any():
        df.loc[still_equal, bigger] = df.loc[still_equal, bigger] + 1


def _build_mask(df: pd.DataFrame, condition: Optional[str]) -> pd.Series:
    if not condition:
        return pd.Series(True, index=df.index)
    cond_lower = condition.strip().lower()
    if "is not null" in cond_lower:
        col = cond_lower.replace("is not null", "").strip()
        if col in df.columns:
            return df[col].notna()
    return pd.Series(True, index=df.index)
