"""
Validatie-utilities voor synthetische data.

Vergelijkt verdelingen van echte en synthetische tabellen via:
- Total variation (TV) afstand voor categorische kolommen  [0 = identiek, 1 = maximaal afwijkend]
- Wasserstein-1 afstand voor numerieke kolommen            [schaal-afhankelijk, kleiner is beter]

Gebruik::

    from synthetische_onderwijsdata.validate import report
    df = report(real_tables, synthetic_tables, schema)
    print(df.to_string())
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from synthetische_onderwijsdata.schema import Schema


def tv_distance(real: pd.Series, synth: pd.Series) -> float:
    """Total variation afstand tussen twee categorische verdelingen."""
    real_p = real.value_counts(normalize=True, dropna=True)
    synth_p = synth.value_counts(normalize=True, dropna=True)
    all_cats = real_p.index.union(synth_p.index)
    return float(0.5 * (real_p.reindex(all_cats, fill_value=0.0)
                        - synth_p.reindex(all_cats, fill_value=0.0)).abs().sum())


def compare_marginals(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    TV-afstand per categorische kolom.

    Parameters
    ----------
    real, synth:
        DataFrames met dezelfde kolomnamen.
    columns:
        Subset van kolommen om te vergelijken. Standaard: alle gedeelde kolommen.

    Returns
    -------
    DataFrame met kolommen ``column``, ``tv_distance``, ``n_real_cats``, ``n_synth_cats``.
    """
    cols = columns or list(set(real.columns) & set(synth.columns))
    rows = []
    for col in cols:
        r, s = real[col].dropna(), synth[col].dropna()
        if r.empty or s.empty:
            continue
        rows.append({
            "column": col,
            "tv_distance": round(tv_distance(r, s), 4),
            "n_real_cats": r.nunique(),
            "n_synth_cats": s.nunique(),
        })
    return pd.DataFrame(rows).sort_values("tv_distance", ascending=False).reset_index(drop=True)


def compare_numeric(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Wasserstein-1 afstand en beschrijvende statistieken per numerieke kolom.

    Returns
    -------
    DataFrame met ``column``, ``wasserstein``, en statistieken voor echte en
    synthetische data (mean, std, p25, p50, p75).
    """
    cols = columns or [
        c for c in set(real.columns) & set(synth.columns)
        if pd.api.types.is_numeric_dtype(real[c])
    ]
    rows = []
    for col in cols:
        r = real[col].dropna().to_numpy(dtype=float)
        s = synth[col].dropna().to_numpy(dtype=float)
        if len(r) == 0 or len(s) == 0:
            continue
        rows.append({
            "column": col,
            "wasserstein": round(float(wasserstein_distance(r, s)), 4),
            "real_mean": round(float(r.mean()), 3),
            "synth_mean": round(float(s.mean()), 3),
            "real_std": round(float(r.std()), 3),
            "synth_std": round(float(s.std()), 3),
            "real_p50": round(float(np.median(r)), 3),
            "synth_p50": round(float(np.median(s)), 3),
        })
    return pd.DataFrame(rows).sort_values("wasserstein", ascending=False).reset_index(drop=True)


def report(
    real_tables: Dict[str, pd.DataFrame],
    synth_tables: Dict[str, pd.DataFrame],
    schema: "Schema",
) -> pd.DataFrame:
    """
    Overzichtsrapport voor alle tabellen en kolommen.

    Berekent per kolom de relevante afstandsmaat (TV voor categorisch,
    Wasserstein voor numeriek) en geeft één DataFrame terug gesorteerd op
    slechtste overeenkomst eerst.

    Parameters
    ----------
    real_tables, synth_tables:
        Output van ``split_flat()`` resp. ``RelationalSynthesizer.generate()``.
    schema:
        Het schema dat bij de tabel hoort (voor dtype-annotaties).

    Returns
    -------
    DataFrame met kolommen ``table``, ``column``, ``dtype``, ``distance``,
    ``metric`` (``tv`` of ``wasserstein``).
    """
    rows = []
    for table_name, table in schema.tables.items():
        real = real_tables.get(table_name)
        synth = synth_tables.get(table_name)
        if real is None or synth is None:
            continue
        for col_name, col in table.columns.items():
            if col.role in ("primary_key", "foreign_key"):
                continue
            if col_name not in real.columns or col_name not in synth.columns:
                continue
            r, s = real[col_name].dropna(), synth[col_name].dropna()
            if r.empty or s.empty:
                continue
            if col.dtype == "categorical":
                dist = tv_distance(r, s)
                metric = "tv"
            elif col.dtype in ("integer", "float", "numeric") and pd.api.types.is_numeric_dtype(r):
                dist = float(wasserstein_distance(
                    r.to_numpy(dtype=float), s.to_numpy(dtype=float)
                ))
                metric = "wasserstein"
            else:
                continue
            rows.append({
                "table": table_name,
                "column": col_name,
                "dtype": col.dtype,
                "distance": round(dist, 4),
                "metric": metric,
            })
    return (
        pd.DataFrame(rows)
        .sort_values("distance", ascending=False)
        .reset_index(drop=True)
    )
