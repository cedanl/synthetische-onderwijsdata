import pytest
from synthetische_onderwijsdata.engine.loader import PresetLoader


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

    pgn = fac.columns["persoonsgebonden_nummer"]
    assert pgn.role == "foreign_key"
    assert pgn.references_table == "dim_persoon"

    opl = fac.columns["opleidingscode"]
    assert opl.role == "foreign_key"
    assert opl.references_table == "dim_opleiding"

    inst = fac.columns["instellingscode"]
    assert inst.role == "foreign_key"
    assert inst.references_table == "dim_instelling"


def test_hooks_laden():
    schema = PresetLoader.from_builtin("1cijferho")
    hooks = schema.tables["fac_inschrijving"].hooks
    rules = [h["rule"] for h in hooks]
    assert any("datum_tekening_diploma" in r for r in rules)
    assert any("diplomajaar" in r for r in rules)


def test_categorische_codes_correct():
    schema = PresetLoader.from_builtin("1cijferho")

    geslacht = schema.tables["dim_persoon"].columns["geslacht"]
    assert "1" in geslacht.categories   # man
    assert "2" in geslacht.categories   # vrouw

    oplvorm = schema.tables["fac_inschrijving"].columns["opleidingsvorm"]
    assert "1" in oplvorm.categories    # voltijd
    assert "2" in oplvorm.categories    # deeltijd
    assert "3" in oplvorm.categories    # duaal

    si_ho = schema.tables["fac_inschrijving"].columns["soort_inschrijving_hoger_onderwijs"]
    assert "1" in si_ho.categories      # hoofdinschrijving


def test_kolomnamen_snake_case_zonder_accenten():
    schema = PresetLoader.from_builtin("1cijferho")
    dp = schema.tables["dim_persoon"]
    # 'vóór' → 'voor'  en  'hoger onderwijs' → 'hoger_onderwijs' (via normalize_name)
    assert "hoogste_vooropleiding_voor_het_ho" in dp.columns
    assert "gem_eindcijfer_vo_van_de_hoogste_vooropl_voor_het_ho" in dp.columns

    fac = schema.tables["fac_inschrijving"]
    assert "soort_hoger_onderwijs" not in fac.columns  # zit in dim_opleiding
    assert "soort_inschrijving_hoger_onderwijs" in fac.columns
    assert "verblijfsjaar_hoger_onderwijs" in fac.columns


def test_dim_keys_aanwezig():
    schema = PresetLoader.from_builtin("1cijferho")
    assert schema.dim_keys["dim_persoon"] == "persoonsgebonden_nummer"
    assert schema.dim_keys["dim_opleiding"] == "opleidingscode"
    assert schema.dim_keys["dim_instelling"] == "instellingscode"


def test_missing_builtin_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        PresetLoader.from_builtin("bestaat_niet")
