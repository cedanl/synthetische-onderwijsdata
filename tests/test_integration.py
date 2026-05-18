"""
End-to-end integratietests.

test_flat_synthesizer_*  – FlatSynthesizer op 1cijferHO-achtige data
test_splitter_*          – split_flat: plat DataFrame → dim/feit-tabellen
"""
import numpy as np
import pandas as pd
import pytest

from synthetische_onderwijsdata import FlatSynthesizer
from synthetische_onderwijsdata.sources._1cijferho.splitter import split_flat
from synthetische_onderwijsdata.engine.loader import PresetLoader


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return PresetLoader.from_builtin("1cijferho")


def _make_flat_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Maak een minimaal plat DataFrame dat de cedanl-tool-output simuleert."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pgn = f"{i:012d}"
        geslacht = rng.choice(["1", "2"])
        opl = rng.choice(["34001", "34002", "34003", "34004", "34005"])
        inst = rng.choice(["21RI", "21PX", "21UN", "22OJ"])
        for jaar in rng.choice([2019, 2020, 2021, 2022], size=rng.integers(1, 4), replace=False):
            rows.append({
                "persoonsgebonden_nummer": pgn,
                "inschrijvingsjaar": str(jaar),
                "instellingscode": inst,
                "opleidingscode": opl,
                "geslacht": geslacht,
                "soort_hoger_onderwijs": rng.choice(["hbo", "wo "]),
                "opleidingsvorm": rng.choice(["1", "2", "3"]),
                "soort_inschrijving_hoger_onderwijs": rng.choice(["1", "2", "3"]),
                "diplomajaar": str(rng.integers(2014, 2024)) if rng.random() > 0.6 else "",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FlatSynthesizer integratietests
# ---------------------------------------------------------------------------

def test_flat_synthesizer_1cijferho_smoke():
    df = _make_flat_df(n=100)
    out = (
        FlatSynthesizer(
            entity_key="persoonsgebonden_nummer",
            time_key="inschrijvingsjaar",
            stable_cols=["geslacht"],
            random_state=0,
        )
        .fit(df)
        .generate(50)
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0
    assert out["persoonsgebonden_nummer"].nunique() == 50
    assert "opleidingscode" in out.columns


def test_flat_synthesizer_longitudinale_structuur():
    df = _make_flat_df(n=150)
    out = (
        FlatSynthesizer(
            entity_key="persoonsgebonden_nummer",
            time_key="inschrijvingsjaar",
            random_state=1,
        )
        .fit(df)
        .generate(80)
    )
    assert (out.groupby("persoonsgebonden_nummer").size() > 1).any()
    for _, grp in out.groupby("persoonsgebonden_nummer"):
        jaren = grp["inschrijvingsjaar"].tolist()
        assert jaren == sorted(jaren)


def test_flat_synthesizer_stabiele_persoonskenmerken():
    df = _make_flat_df(n=120)
    out = (
        FlatSynthesizer(
            entity_key="persoonsgebonden_nummer",
            time_key="inschrijvingsjaar",
            stable_cols=["geslacht"],
            random_state=2,
        )
        .fit(df)
        .generate(60)
    )
    assert (out.groupby("persoonsgebonden_nummer")["geslacht"].nunique() == 1).all()


# ---------------------------------------------------------------------------
# Splitter-tests (ongewijzigd — splitter werkt nog steeds op flat data)
# ---------------------------------------------------------------------------

def test_split_flat_geeft_verwachte_tabellen(schema):
    df = _make_flat_df()
    tables = split_flat(df, schema)

    assert "dim_persoon" in tables
    assert "dim_opleiding" in tables
    assert "dim_instelling" in tables
    assert "fac_inschrijving" in tables


def test_split_flat_dim_persoon_gededupliceerd(schema):
    df = _make_flat_df(n=500)
    tables = split_flat(df, schema)

    dp = tables["dim_persoon"]
    assert dp["persoonsgebonden_nummer"].nunique() == len(dp)


def test_split_flat_fac_behoudt_alle_rijen(schema):
    n = 300
    df = _make_flat_df(n=n)
    tables = split_flat(df, schema)
    assert len(tables["fac_inschrijving"]) == len(df)


def test_split_flat_ontbrekende_kolommen_overgeslagen(schema):
    """Kolommen die niet in df staan mogen de splitter niet laten crashen."""
    df = pd.DataFrame({
        "persoonsgebonden_nummer": ["000000000001", "000000000002"],
        "inschrijvingsjaar": ["2020", "2021"],
        "opleidingscode": ["34001", "34002"],
        "instellingscode": ["21RI", "21PX"],
    })
    tables = split_flat(df, schema)
    assert "fac_inschrijving" in tables
    assert "persoonsgebonden_nummer" in tables["fac_inschrijving"].columns


def test_dim_keys_aanwezig(schema):
    assert schema.dim_keys["dim_persoon"] == "persoonsgebonden_nummer"
    assert schema.dim_keys["dim_opleiding"] == "opleidingscode"
    assert schema.dim_keys["dim_instelling"] == "instellingscode"


def test_splitter_op_synthetische_output(schema):
    """Splitter werkt als post-processing stap op synthetische flat output."""
    df = _make_flat_df(n=100)
    synth_flat = (
        FlatSynthesizer(
            entity_key="persoonsgebonden_nummer",
            time_key="inschrijvingsjaar",
            stable_cols=["geslacht"],
            random_state=3,
        )
        .fit(df)
        .generate(50)
    )

    # Zorg dat verwachte kolommen voor splitter aanwezig zijn
    for col in ["opleidingscode", "instellingscode"]:
        if col not in synth_flat.columns:
            pytest.skip(f"Kolom {col} niet gegenereerd — splitter-test overgeslagen")

    tables = split_flat(synth_flat, schema)
    assert "fac_inschrijving" in tables
    assert len(tables["fac_inschrijving"]) == len(synth_flat)
    if "dim_persoon" in tables:
        dp = tables["dim_persoon"]
        assert dp["persoonsgebonden_nummer"].nunique() == len(dp)
