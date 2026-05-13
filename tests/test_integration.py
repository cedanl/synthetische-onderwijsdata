"""
End-to-end integratietests.

test_generate_*      – schema-gedreven generatie (geen echte data)
test_splitter_*      – split_flat: plat DataFrame → dim/feit-tabellen
"""
import numpy as np
import pandas as pd
import pytest

from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.sources._1cijferho.splitter import split_flat
from synthetische_onderwijsdata.engine.loader import PresetLoader


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return PresetLoader.from_builtin("1cijferho")


def _make_flat_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Maak een minimaal plat DataFrame dat de cedanl-tool-output simuleert."""
    rng = np.random.default_rng(seed)
    pgns   = [f"{i:012d}" for i in rng.integers(1_000_000, 9_999_999, n)]
    isats  = [f"{c:05d}" for c in rng.choice([34001, 34002, 34003, 34004, 34005], n)]
    brins  = [rng.choice(["21RI", "21PX", "21UN", "22OJ"]) for _ in range(n)]
    years  = rng.integers(2010, 2024, n).astype(str)

    return pd.DataFrame({
        "persoonsgebonden_nummer": pgns,
        "inschrijvingsjaar": years,
        "instellingscode": brins,
        "opleidingscode": isats,
        "geslacht": rng.choice(["1","2"], n),
        "soort_hoger_onderwijs": rng.choice(["hbo","wo "], n),
        "opleidingsvorm": rng.choice(["1","2","3"], n),
        "soort_inschrijving_hoger_onderwijs": rng.choice(["1","2","3"], n),
        "diplomajaar": [str(y) if rng.random() > 0.6 else "" for y in rng.integers(2014, 2024, n)],
    })


# ---------------------------------------------------------------------------
# Generatietests
# ---------------------------------------------------------------------------

def test_generate_1cijferho_schema_only(schema):
    synth = RelationalSynthesizer(schema, random_state=0)
    tables = synth.generate(
        n_entities={
            "dim_persoon":    200,
            "dim_opleiding":   30,
            "dim_instelling":  15,
        }
    )

    expected = {"dim_persoon", "dim_opleiding", "dim_instelling", "fac_inschrijving", "fac_vak"}
    assert set(tables) == expected

    dp = tables["dim_persoon"]
    assert len(dp) == 200
    assert set(dp["geslacht"].dropna().unique()).issubset({"1", "2", "0"})

    fac = tables["fac_inschrijving"]
    assert len(fac) > 0

    # FK-integriteit
    pgn_pks = set(dp["persoonsgebonden_nummer"].tolist())
    assert set(fac["persoonsgebonden_nummer"].tolist()).issubset(pgn_pks)

    opl_pks = set(tables["dim_opleiding"]["opleidingscode"].tolist())
    assert set(fac["opleidingscode"].tolist()).issubset(opl_pks)


# ---------------------------------------------------------------------------
# Splitter-tests
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
    # Geen dubbele PGNs in de dimensie
    assert dp["persoonsgebonden_nummer"].nunique() == len(dp)


def test_split_flat_fac_behoudt_alle_rijen(schema):
    df = _make_flat_df(n=300)
    tables = split_flat(df, schema)
    assert len(tables["fac_inschrijving"]) == 300


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


# ---------------------------------------------------------------------------
# Tussenjaren en switchers
# ---------------------------------------------------------------------------

def _make_flat_df_met_tussenjaren_en_switchers(seed: int = 42) -> pd.DataFrame:
    """
    Gesimuleerd plat bestand met:
    - Studenten met een gat in hun inschrijvingsjaren (tussenjaar)
    - Studenten die van opleiding wisselen (switcher)
    """
    rng = np.random.default_rng(seed)
    rows = []
    opleidingen = ["34001", "34002", "34003"]
    instellingen = ["21RI", "21PX"]

    for i in range(80):
        pgn = f"{i:012d}"
        opl = rng.choice(opleidingen)
        inst = rng.choice(instellingen)

        # Normaal traject: 3 aaneengesloten jaren
        for jaar in [2019, 2020, 2021]:
            rows.append({"persoonsgebonden_nummer": pgn, "inschrijvingsjaar": str(jaar),
                         "opleidingscode": opl, "instellingscode": inst,
                         "geslacht": "1", "soort_hoger_onderwijs": "hbo",
                         "opleidingsvorm": "1", "soort_inschrijving_hoger_onderwijs": "1",
                         "diplomajaar": ""})

    for i in range(80, 100):
        pgn = f"{i:012d}"
        inst = rng.choice(instellingen)
        # Tussenjaar: 2019, dan gat, dan 2021
        for jaar in [2019, 2021]:
            rows.append({"persoonsgebonden_nummer": pgn, "inschrijvingsjaar": str(jaar),
                         "opleidingscode": "34001", "instellingscode": inst,
                         "geslacht": "2", "soort_hoger_onderwijs": "wo ",
                         "opleidingsvorm": "2", "soort_inschrijving_hoger_onderwijs": "1",
                         "diplomajaar": ""})

    for i in range(100, 120):
        pgn = f"{i:012d}"
        inst = rng.choice(instellingen)
        # Switcher: begint opleiding 34001, gaat naar 34002
        rows.append({"persoonsgebonden_nummer": pgn, "inschrijvingsjaar": "2019",
                     "opleidingscode": "34001", "instellingscode": inst,
                     "geslacht": "1", "soort_hoger_onderwijs": "hbo",
                     "opleidingsvorm": "1", "soort_inschrijving_hoger_onderwijs": "1",
                     "diplomajaar": ""})
        rows.append({"persoonsgebonden_nummer": pgn, "inschrijvingsjaar": "2020",
                     "opleidingscode": "34002", "instellingscode": inst,
                     "geslacht": "1", "soort_hoger_onderwijs": "hbo",
                     "opleidingsvorm": "1", "soort_inschrijving_hoger_onderwijs": "1",
                     "diplomajaar": ""})
        rows.append({"persoonsgebonden_nummer": pgn, "inschrijvingsjaar": "2021",
                     "opleidingscode": "34002", "instellingscode": inst,
                     "geslacht": "1", "soort_hoger_onderwijs": "hbo",
                     "opleidingsvorm": "1", "soort_inschrijving_hoger_onderwijs": "1",
                     "diplomajaar": ""})

    return pd.DataFrame(rows)


def test_tussenjaren_optreden_na_fit(schema):
    """Na fit op data met tussenjaren moet de synthetische output ook jaar-gaps bevatten."""
    df = _make_flat_df_met_tussenjaren_en_switchers()
    tables = split_flat(df, schema)

    synth = RelationalSynthesizer(schema, random_state=0)
    synth.fit(tables)
    synthetic = synth.generate(
        n_entities={"dim_persoon": 300, "dim_opleiding": 10, "dim_instelling": 5}
    )

    fac = synthetic["fac_inschrijving"]
    fac["inschrijvingsjaar"] = fac["inschrijvingsjaar"].astype(int)

    gaps = (
        fac.sort_values("inschrijvingsjaar")
        .groupby("persoonsgebonden_nummer")["inschrijvingsjaar"]
        .apply(lambda s: (np.diff(s.to_numpy()) > 1).any())
    )
    assert gaps.any(), "Verwacht ten minste één student met een tussenjaar"


def test_switchers_optreden_na_fit(schema):
    """Na fit op data met switchers moet de synthetische output ook opleidingswissels bevatten."""
    df = _make_flat_df_met_tussenjaren_en_switchers()
    tables = split_flat(df, schema)

    synth = RelationalSynthesizer(schema, random_state=0)
    synth.fit(tables)
    synthetic = synth.generate(
        n_entities={"dim_persoon": 300, "dim_opleiding": 10, "dim_instelling": 5}
    )

    fac = synthetic["fac_inschrijving"]
    fac["inschrijvingsjaar"] = fac["inschrijvingsjaar"].astype(int)

    switchers = (
        fac.sort_values("inschrijvingsjaar")
        .groupby("persoonsgebonden_nummer")["opleidingscode"]
        .apply(lambda s: s.nunique() > 1)
    )
    assert switchers.any(), "Verwacht ten minste één switcher in de synthetische data"


def test_fk_opleidingscode_valide_na_fit(schema):
    """FK opleidingscode in fac_inschrijving moet altijd verwijzen naar een bestaande dim_opleiding."""
    df = _make_flat_df_met_tussenjaren_en_switchers()
    tables = split_flat(df, schema)

    synth = RelationalSynthesizer(schema, random_state=0)
    synth.fit(tables)
    synthetic = synth.generate(
        n_entities={"dim_persoon": 200, "dim_opleiding": 10, "dim_instelling": 5}
    )

    opl_pks = set(synthetic["dim_opleiding"]["opleidingscode"].tolist())
    fac_opls = set(synthetic["fac_inschrijving"]["opleidingscode"].dropna().tolist())
    assert fac_opls.issubset(opl_pks), "Orphan opleidingscodes gevonden in fac_inschrijving"
