"""
Splits een plat 1cijferHO CSV/Parquet-bestand (output van cedanl/1cijferho-tool)
in de dim/feit-tabellen zoals gedefinieerd in het synthetische_onderwijsdata-preset.

Gebruik
-------
    import pandas as pd
    from synthetische_onderwijsdata.engine.loader import PresetLoader
    from synthetische_onderwijsdata.sources._1cijferho.splitter import split_flat

    schema = PresetLoader.from_builtin("1cijferho")
    df = pd.read_parquet("data/ev_inschrijving.parquet")

    tables = split_flat(df, schema)
    # → {"dim_persoon": ..., "dim_opleiding": ..., "fac_inschrijving": ..., ...}

Werking
-------
Voor elke dimensietabel:
  - Selecteer de kolommen die in de preset zijn gedefinieerd EN aanwezig zijn in df.
  - Dedupliceer op de primary-key: neem de eerste rij per PK-waarde.

Voor elke feitentabel:
  - Selecteer de kolommen die in de preset zijn gedefinieerd EN aanwezig zijn in df.
  - Alle rijen blijven staan (geen deduplicatie).

Ontbrekende kolommen worden stilzwijgend overgeslagen; ze worden als None gegenereerd
door de synthesizer.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from synthetische_onderwijsdata.engine.loader import Schema


def split_flat(df: pd.DataFrame, schema: Schema) -> Dict[str, pd.DataFrame]:
    """
    Splits het platte DataFrame *df* in dim/feit-tabellen op basis van *schema*.

    Parameters
    ----------
    df:
        Plat DataFrame, direct geladen uit de cedanl/1cijferho-tooloutput.
    schema:
        Ingeladen synthetische_onderwijsdata-schema (PresetLoader.from_builtin("1cijferho")).

    Returns
    -------
    Dict van tabelnaam → DataFrame, klaar voor ``RelationalSynthesizer.fit()``.
    """
    result: Dict[str, pd.DataFrame] = {}

    for table_name, table in schema.tables.items():
        preset_cols = list(table.columns.keys())
        available = [c for c in preset_cols if c in df.columns]

        if not available:
            continue

        pk_col = next(
            (c for c, col in table.columns.items() if col.role == "primary_key"),
            None,
        )

        subset = df[available].copy()

        if table.table_type == "dimension" and pk_col and pk_col in subset.columns:
            subset = (
                subset
                .drop_duplicates(subset=[pk_col])
                .reset_index(drop=True)
            )
        else:
            subset = subset.reset_index(drop=True)

        result[table_name] = subset

    return result
