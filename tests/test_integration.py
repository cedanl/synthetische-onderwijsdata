"""End-to-end integratietest: schema-gedreven generatie (geen echte data)."""
from relsynth import RelationalSynthesizer
from relsynth.presets.loader import PresetLoader
from relsynth.io import read_fwf_asc


def test_generate_1cijferho_schema_only():
    schema = PresetLoader.from_builtin("1cijferho")
    synth = RelationalSynthesizer(schema, random_state=0)

    tables = synth.generate(
        n_entities={
            "dim_persoon": 200,
            "dim_opleiding": 30,
            "dim_instelling": 15,
        }
    )

    expected = {"dim_persoon", "dim_opleiding", "dim_instelling", "fac_inschrijving", "fac_vak"}
    assert set(tables) == expected

    # dim_persoon
    dp = tables["dim_persoon"]
    assert len(dp) == 200
    assert "persoonsgebonden_nummer" in dp.columns
    assert set(dp["geslacht"].dropna().unique()).issubset({"1", "2", "0"})

    # dim_opleiding
    do = tables["dim_opleiding"]
    assert len(do) == 30
    assert set(do["soort_ho"].dropna().unique()).issubset({"hbo", "wo "})

    # fac_inschrijving FK integriteit
    fac = tables["fac_inschrijving"]
    assert len(fac) > 0
    pgn_pks = set(dp["persoonsgebonden_nummer"].tolist())
    assert set(fac["persoonsgebonden_nummer"].tolist()).issubset(pgn_pks)

    opl_pks = set(do["opleidingscode"].tolist())
    assert set(fac["opleidingscode"].tolist()).issubset(opl_pks)

    # fac_vak FK integriteit
    fac_vak = tables["fac_vak"]
    assert len(fac_vak) > 0
    assert set(fac_vak["persoonsgebonden_nummer"].tolist()).issubset(pgn_pks)


def test_fwf_layout_aanwezig():
    schema = PresetLoader.from_builtin("1cijferho")
    assert "ev_inschrijving" in schema.fwf
    assert "vak_havovwo" in schema.fwf

    layout = schema.fwf["ev_inschrijving"]
    # Positie-check: persoonsgebonden_nummer = [1, 12]
    assert layout["persoonsgebonden_nummer"] == [1, 12]
    # Geslacht = [155, 155]
    assert layout["geslacht"] == [155, 155]
    # Laatste veld onderwijsnummer = [315, 323]
    assert layout["onderwijsnummer"] == [315, 323]


def test_fwf_reader_importeerbaar():
    # Importeer en controleer de publieke API (geen echt .asc bestand vereist)
    from relsynth.io.fwf_reader import read_fwf_asc, fwf_layouts
    schema = PresetLoader.from_builtin("1cijferho")
    layouts = fwf_layouts(schema)
    assert "ev_inschrijving" in layouts
    assert len(layouts["ev_inschrijving"]) == 115  # 115 velden conform bestandsbeschrijving
