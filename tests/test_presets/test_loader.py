import pytest
from relsynth.presets.loader import PresetLoader


def test_load_builtin_1cijferho():
    schema = PresetLoader.from_builtin("1cijferho")
    assert schema.name == "1cijferho"
    assert "dim_student" in schema.tables
    assert "dim_opleiding" in schema.tables
    assert "fac_inschrijving" in schema.tables


def test_sequential_config_parsed():
    schema = PresetLoader.from_builtin("1cijferho")
    fac = schema.tables["fac_inschrijving"]
    assert fac.sequential is not None
    assert fac.sequential["entity_key"] == "studentnummer"
    assert fac.sequential["time_key"] == "collegejaar"


def test_fk_references_resolved():
    schema = PresetLoader.from_builtin("1cijferho")
    col = schema.tables["fac_inschrijving"].columns["studentnummer"]
    assert col.role == "foreign_key"
    assert col.references_table == "dim_student"
    assert col.references_column == "studentnummer"


def test_hooks_loaded():
    schema = PresetLoader.from_builtin("1cijferho")
    hooks = schema.tables["fac_inschrijving"].hooks
    assert len(hooks) >= 1
    assert any("diplomadatum" in h["rule"] for h in hooks)


def test_missing_builtin_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        PresetLoader.from_builtin("this_does_not_exist")
