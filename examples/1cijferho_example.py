"""
Example: generate synthetic 1cijferHO data from the built-in preset.

Run:
    uv run python examples/1cijferho_example.py
"""
import pandas as pd

from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.engine.loader import PresetLoader
from synthetische_onderwijsdata.sources._1cijferho.splitter import split_flat

schema = PresetLoader.from_builtin("1cijferho")
synth = RelationalSynthesizer(schema, random_state=42)

# ── Optie A: schema-gedreven generatie (geen echte data nodig) ────────────
tables = synth.generate(
    n_entities={
        "dim_persoon": 500,
        "dim_opleiding": 20,
        "dim_instelling": 10,
    }
)

# ── Optie B: fitten op echte data (OBT → split → fit) ────────────────────
# df = pd.read_parquet("data/ev_inschrijving.parquet")
# tables = split_flat(df, schema)
# synth.fit(tables)
# tables = synth.generate(n_entities={"dim_persoon": 5000, "dim_opleiding": 50, "dim_instelling": 20})

for name, df in tables.items():
    print(f"\n{'─' * 60}")
    print(f"  {name}  ({len(df):,} rows × {len(df.columns)} cols)")
    print("─" * 60)
    print(df.head(5).to_string(index=False))

fac = tables["fac_inschrijving"]
print("\n\nReferentiële integriteitscheck")
print("─" * 40)
persoon_pks = set(tables["dim_persoon"]["persoonsgebonden_nummer"])
orphan_fks = set(fac["persoonsgebonden_nummer"]) - persoon_pks
print(f"Orphan FK-waarden (moet 0 zijn): {len(orphan_fks)}")

print("\nRijen per persoon (steekproef):")
print(fac.groupby("persoonsgebonden_nummer").size().describe().to_string())
