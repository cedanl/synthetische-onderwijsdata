"""Tests voor validate.py."""
import numpy as np
import pandas as pd
import pytest

from synthetische_onderwijsdata.validate import (
    compare_marginals,
    compare_numeric,
    report,
    tv_distance,
)
from synthetische_onderwijsdata.engine.loader import PresetLoader


@pytest.fixture
def schema():
    return PresetLoader.from_builtin("1cijferho")


class TestTvDistance:
    def test_identieke_verdeling_geeft_nul(self):
        s = pd.Series(["a", "a", "b", "b"])
        assert tv_distance(s, s) == pytest.approx(0.0)

    def test_volledig_disjunct_geeft_een(self):
        assert tv_distance(pd.Series(["a", "a"]), pd.Series(["b", "b"])) == pytest.approx(1.0)

    def test_gedeeltelijk_overlap(self):
        r = pd.Series(["a", "b"])
        s = pd.Series(["a", "c"])
        # p_a=0.5, p_b=0.5 vs p_a=0.5, p_c=0.5 → TV = 0.5*( |0| + |0.5| + |0.5| ) = 0.5
        assert tv_distance(r, s) == pytest.approx(0.5)


class TestCompareMarginals:
    def test_geeft_dataframe_terug(self):
        real = pd.DataFrame({"geslacht": ["1", "2", "1", "2"]})
        synth = pd.DataFrame({"geslacht": ["1", "1", "1", "2"]})
        result = compare_marginals(real, synth)
        assert isinstance(result, pd.DataFrame)
        assert "tv_distance" in result.columns
        assert "column" in result.columns

    def test_gesorteerd_op_afstand(self):
        real = pd.DataFrame({"a": ["x", "y"], "b": ["p", "p"]})
        synth = pd.DataFrame({"a": ["x", "y"], "b": ["q", "q"]})
        result = compare_marginals(real, synth)
        # b heeft hogere TV dan a (die identiek is)
        assert result.iloc[0]["column"] == "b"

    def test_subset_kolommen(self):
        real = pd.DataFrame({"a": ["x"], "b": ["y"]})
        synth = pd.DataFrame({"a": ["x"], "b": ["z"]})
        result = compare_marginals(real, synth, columns=["a"])
        assert list(result["column"]) == ["a"]


class TestCompareNumeric:
    def test_identieke_data_geeft_nul_wasserstein(self):
        data = pd.DataFrame({"val": [1.0, 2.0, 3.0, 4.0]})
        result = compare_numeric(data, data)
        assert result.iloc[0]["wasserstein"] == pytest.approx(0.0)

    def test_verschoven_verdeling_heeft_hogere_afstand(self):
        real = pd.DataFrame({"val": np.arange(10, dtype=float)})
        synth = pd.DataFrame({"val": np.arange(10, dtype=float) + 5.0})
        result = compare_numeric(real, synth)
        assert result.iloc[0]["wasserstein"] > 0

    def test_statistieken_aanwezig(self):
        data = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        result = compare_numeric(data, data)
        for col in ("real_mean", "synth_mean", "real_std", "synth_std", "real_p50", "synth_p50"):
            assert col in result.columns


class TestReport:
    def test_geeft_dataframe_terug(self, schema):
        from synthetische_onderwijsdata import RelationalSynthesizer

        synth = RelationalSynthesizer(schema, random_state=0)
        tables = synth.generate(
            n_entities={"dim_persoon": 100, "dim_opleiding": 10, "dim_instelling": 5}
        )
        result = report(tables, tables, schema)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) >= {"table", "column", "dtype", "distance", "metric"}

    def test_identieke_input_geeft_nul_distances(self, schema):
        from synthetische_onderwijsdata import RelationalSynthesizer

        synth = RelationalSynthesizer(schema, random_state=0)
        tables = synth.generate(
            n_entities={"dim_persoon": 100, "dim_opleiding": 10, "dim_instelling": 5}
        )
        result = report(tables, tables, schema)
        assert result["distance"].max() == pytest.approx(0.0, abs=1e-9)

    def test_pk_fk_uitgesloten(self, schema):
        from synthetische_onderwijsdata import RelationalSynthesizer

        synth = RelationalSynthesizer(schema, random_state=0)
        tables = synth.generate(
            n_entities={"dim_persoon": 50, "dim_opleiding": 5, "dim_instelling": 3}
        )
        result = report(tables, tables, schema)
        pk_fk_cols = {
            c for t in schema.tables.values()
            for c, col in t.columns.items()
            if col.role in ("primary_key", "foreign_key")
        }
        assert set(result["column"]).isdisjoint(pk_fk_cols)
