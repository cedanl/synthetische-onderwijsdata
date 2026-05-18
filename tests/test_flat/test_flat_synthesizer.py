"""Tests voor FlatSynthesizer — TDD: schrijf eerst, implementeer daarna."""
import numpy as np
import pandas as pd
import pytest

from synthetische_onderwijsdata.engine.flat import FlatSynthesizer


def _make_flat_df(n_students: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_students):
        pgn = f"S{i:06d}"
        geslacht = rng.choice(["M", "V"])
        opl = rng.choice(["34001", "34002", "34003"])
        for jaar in [2020, 2021, 2022]:
            rows.append({
                "pgn": pgn, "jaar": jaar, "geslacht": geslacht,
                "opleiding": opl, "verblijfsjaar": jaar - 2020 + 1,
            })
    return pd.DataFrame(rows)


def _make_gecorreleerde_df() -> pd.DataFrame:
    """V-studenten altijd opleiding X, M-studenten altijd opleiding Y."""
    rows = []
    for i in range(100):
        for jaar in [2020, 2021, 2022]:
            rows.append({"pgn": f"V{i:04d}", "jaar": jaar, "geslacht": "V", "opleiding": "X"})
    for i in range(100):
        for jaar in [2020, 2021, 2022]:
            rows.append({"pgn": f"M{i:04d}", "jaar": jaar, "geslacht": "M", "opleiding": "Y"})
    return pd.DataFrame(rows)


def test_requires_fit():
    synth = FlatSynthesizer(entity_key="pgn")
    with pytest.raises(RuntimeError, match="fit"):
        synth.generate(10)


def test_output_is_dataframe():
    out = FlatSynthesizer("pgn", "jaar").fit(_make_flat_df()).generate(20)
    assert isinstance(out, pd.DataFrame)
    assert len(out) > 0


def test_n_entities_correct():
    out = FlatSynthesizer("pgn", "jaar").fit(_make_flat_df()).generate(50)
    assert out["pgn"].nunique() == 50


def test_output_kolommen_aanwezig():
    df = _make_flat_df()
    out = FlatSynthesizer("pgn", "jaar").fit(df).generate(20)
    for col in df.columns:
        assert col in out.columns


def test_meerdere_rijen_per_entiteit():
    out = FlatSynthesizer("pgn", "jaar").fit(_make_flat_df()).generate(50)
    assert (out.groupby("pgn").size() > 1).any()


def test_tijdstappen_oplopend_per_entiteit():
    out = FlatSynthesizer("pgn", "jaar").fit(_make_flat_df()).generate(30)
    for _, grp in out.groupby("pgn"):
        jaren = grp["jaar"].tolist()
        assert jaren == sorted(jaren)


def test_stabiele_kolom_constant_per_entiteit():
    out = (FlatSynthesizer("pgn", "jaar", stable_cols=["geslacht"])
           .fit(_make_flat_df()).generate(60))
    assert (out.groupby("pgn")["geslacht"].nunique() == 1).all()


def test_cross_correlatie_bewaard():
    """Kerntest: geslacht → opleiding correlatie moet bewaard blijven na synthese."""
    df = _make_gecorreleerde_df()
    out = (FlatSynthesizer("pgn", "jaar", stable_cols=["geslacht"], random_state=0)
           .fit(df).generate(100))
    eerste = out.sort_values("jaar").groupby("pgn").first().reset_index()
    v_opl = eerste[eerste["geslacht"] == "V"]["opleiding"]
    m_opl = eerste[eerste["geslacht"] == "M"]["opleiding"]
    assert (v_opl == "X").mean() > 0.7, f"V→X correlatie verloren: {v_opl.value_counts().to_dict()}"
    assert (m_opl == "Y").mean() > 0.7, f"M→Y correlatie verloren: {m_opl.value_counts().to_dict()}"
