"""
Example: generate synthetic 1cijferHO data from the built-in preset.

Run:
    uv run python examples/1cijferho_example.py
"""
from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.presets.loader import PresetLoader

schema = PresetLoader.from_builtin("1cijferho")

synth = RelationalSynthesizer(schema, random_state=42)

# Schema-driven generation (no real data required for the first run)
tables = synth.generate(
    n_entities={
        "dim_student": 500,
        "dim_opleiding": 20,
    }
)

for name, df in tables.items():
    print(f"\n{'─' * 60}")
    print(f"  {name}  ({len(df):,} rows × {len(df.columns)} cols)")
    print("─" * 60)
    print(df.head(5).to_string(index=False))

fac = tables["fac_inschrijving"]
print("\n\nReferential integrity check")
print("─" * 40)
student_pks = set(tables["dim_student"]["studentnummer"])
orphan_fks = set(fac["studentnummer"]) - student_pks
print(f"Orphan FK values (should be 0): {len(orphan_fks)}")

print("\nRows per student (sample):")
print(
    fac.groupby("studentnummer")
    .size()
    .describe()
    .to_string()
)
