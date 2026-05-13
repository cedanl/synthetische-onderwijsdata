"""
Leest CBS vaste-breedte ASCII-bestanden (*.asc) in op basis van
de fwf-posities uit een relsynth YAML-preset.

Gebruik
-------
    from relsynth.io import read_fwf_asc
    from relsynth.presets.loader import PresetLoader

    schema = PresetLoader.from_builtin("1cijferho")
    df = read_fwf_asc(
        path="data/EV299XX24.asc",
        layout_name="ev_inschrijving",
        schema=schema,
        encoding="latin-1",
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from relsynth.presets.loader import Schema


def read_fwf_asc(
    path: str | Path,
    layout_name: str,
    schema: Schema,
    encoding: str = "latin-1",
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Lees een vast-breedte CBS .asc-bestand in.

    Parameters
    ----------
    path:
        Pad naar het .asc-bestand.
    layout_name:
        Sleutel onder ``fwf:`` in de preset (bijv. ``"ev_inschrijving"``).
    schema:
        Ingeladen Schema-object (via PresetLoader).
    encoding:
        Tekencodering van het bestand (CBS-bestanden zijn doorgaans latin-1).
    nrows:
        Maximaal aantal rijen om in te lezen (None = alles).

    Returns
    -------
    pd.DataFrame met één kolom per veld.
    """
    fwf_layouts: Dict[str, Dict[str, List[int]]] = getattr(schema, "fwf", {})
    if layout_name not in fwf_layouts:
        available = list(fwf_layouts.keys())
        raise KeyError(
            f"Layout '{layout_name}' niet gevonden in preset '{schema.name}'. "
            f"Beschikbaar: {available}"
        )

    layout = fwf_layouts[layout_name]
    colspecs: List[Tuple[int, int]] = []
    colnames: List[str] = []

    for col, (start, end) in layout.items():
        # Bestandsbeschrijving is 1-gebaseerd inclusief → omzetten naar 0-gebaseerd exclusief
        colspecs.append((start - 1, end))
        colnames.append(col)

    df = pd.read_fwf(
        path,
        colspecs=colspecs,
        names=colnames,
        encoding=encoding,
        dtype=str,          # alles als string inlezen; type-conversie later
        nrows=nrows,
        header=None,
    )

    # Verwijder trailing whitespace (CBS-bestanden zijn spatie-opgevuld)
    for col in df.columns:
        df[col] = df[col].str.rstrip()

    return df


def fwf_layouts(schema: Schema) -> Dict[str, Dict[str, List[int]]]:
    """Geef alle fwf-layouts terug die in het schema zijn gedefinieerd."""
    return getattr(schema, "fwf", {})
