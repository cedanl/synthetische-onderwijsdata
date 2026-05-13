"""End-to-end integration test: schema-only generation (no real data fit)."""
import pytest
from relsynth import RelationalSynthesizer
from relsynth.presets.loader import PresetLoader


def test_generate_1cijferho_schema_only():
    schema = PresetLoader.from_builtin("1cijferho")
    synth = RelationalSynthesizer(schema, random_state=0)

    tables = synth.generate(
        n_entities={"dim_student": 100, "dim_opleiding": 10}
    )

    assert set(tables) == {"dim_student", "dim_opleiding", "fac_inschrijving"}

    dim_s = tables["dim_student"]
    assert len(dim_s) == 100
    assert "studentnummer" in dim_s.columns

    dim_o = tables["dim_opleiding"]
    assert len(dim_o) == 10

    fac = tables["fac_inschrijving"]
    assert len(fac) > 0

    # All FK values must reference existing PKs
    student_pks = set(dim_s["studentnummer"].tolist())
    assert set(fac["studentnummer"].tolist()).issubset(student_pks)

    opleiding_pks = set(dim_o["isat_code"].tolist())
    assert set(fac["isat_code"].tolist()).issubset(opleiding_pks)
