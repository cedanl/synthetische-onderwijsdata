import pytest
from relsynth.presets.loader import PresetLoader


def test_load_builtin_1cijferho():
    schema = PresetLoader.from_builtin("1cijferho")
    assert schema.name == "1cijferho"
    assert "dim_persoon" in schema.tables
    assert "dim_opleiding" in schema.tables
    assert "dim_instelling" in schema.tables
    assert "fac_inschrijving" in schema.tables
    assert "fac_vak" in schema.tables


def test_sequential_config_parsed():
    schema = PresetLoader.from_builtin("1cijferho")
    fac = schema.tables["fac_inschrijving"]
    assert fac.sequential is not None
    assert fac.sequential["entity_key"] == "persoonsgebonden_nummer"
    assert fac.sequential["time_key"] == "inschrijvingsjaar"


def test_fk_references_resolved():
    schema = PresetLoader.from_builtin("1cijferho")
    fac = schema.tables["fac_inschrijving"]

    pgn_col = fac.columns["persoonsgebonden_nummer"]
    assert pgn_col.role == "foreign_key"
    assert pgn_col.references_table == "dim_persoon"
    assert pgn_col.references_column == "persoonsgebonden_nummer"

    opl_col = fac.columns["opleidingscode"]
    assert opl_col.role == "foreign_key"
    assert opl_col.references_table == "dim_opleiding"

    inst_col = fac.columns["instellingscode"]
    assert inst_col.role == "foreign_key"
    assert inst_col.references_table == "dim_instelling"


def test_hooks_loaded():
    schema = PresetLoader.from_builtin("1cijferho")
    hooks = schema.tables["fac_inschrijving"].hooks
    assert len(hooks) >= 2
    rules = [h["rule"] for h in hooks]
    assert any("datum_tekening_diploma" in r for r in rules)
    assert any("diplomajaar" in r for r in rules)


def test_categorische_codes_aanwezig():
    schema = PresetLoader.from_builtin("1cijferho")
    fac = schema.tables["fac_inschrijving"]

    geslacht = schema.tables["dim_persoon"].columns["geslacht"]
    assert "1" in geslacht.categories
    assert "2" in geslacht.categories

    oplvorm = fac.columns["opleidingsvorm"]
    assert "1" in oplvorm.categories   # voltijd
    assert "2" in oplvorm.categories   # deeltijd

    si_ho = fac.columns["soort_inschrijving_ho"]
    assert "1" in si_ho.categories     # hoofdinschrijving


def test_fwf_posities_correct():
    schema = PresetLoader.from_builtin("1cijferho")
    layout = schema.fwf["ev_inschrijving"]
    assert layout["persoonsgebonden_nummer"] == [1, 12]
    assert layout["inschrijvingsjaar"] == [13, 16]
    assert layout["geslacht"] == [155, 155]
    assert layout["onderwijsnummer"] == [315, 323]
    assert len(layout) == 115


def test_missing_builtin_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        PresetLoader.from_builtin("bestaat_niet")
