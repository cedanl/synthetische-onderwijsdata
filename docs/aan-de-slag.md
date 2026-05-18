# Aan de slag

## Installatie

```bash
pip install synthetische-onderwijsdata
```

Of lokaal vanuit de broncode (aanbevolen voor ontwikkeling):

```bash
git clone https://github.com/cedanl/synthetische-onderwijsdata
cd synthetische-onderwijsdata
pip install -e ".[dev]"
```

Vereisten: Python ≥ 3.10, numpy, scipy, pandas, pyyaml.

---

## Workflow: inlezen → fitten → genereren

`FlatSynthesizer` werkt direct op het platte 1cijferHO-bestand — één rij per inschrijving, alle kolommen samen. Je hebt altijd echte data nodig: het model leert de statistische structuur van jouw dataset en reproduceert die.

```python
import pandas as pd
from synthetische_onderwijsdata.engine.flat import FlatSynthesizer

# 1. Plat bestand inlezen (output van cedanl/1cijferho-tool)
df = pd.read_parquet("data/ev_inschrijving.parquet")

# 2. Model initialiseren
synth = FlatSynthesizer(
    entity_key="persoonsgebonden_nummer",   # kolom die studenten identificeert
    time_key="inschrijvingsjaar",            # kolom die tijdstappen aangeeft
    stable_cols=[                            # kolommen die niet veranderen per student
        "geslacht",
        "geboorteland",
        "geboorteland_ouder_1",
        "geboorteland_ouder_2",
        "nationaliteit_1",
        "hoogste_vooropleiding_voor_het_ho",
        "diplomajaar_van_de_hoogste_vooropl_voor_het_ho",
        "gem_eindcijfer_vo_van_de_hoogste_vooropl_voor_het_ho",
    ],
    random_state=42,
)

# 3. Fitten op echte data
synth.fit(df)

# 4. Synthetische data genereren — zelfde structuur als invoer
synthetic = synth.generate(n_entities=5000)
```

**Wat `fit()` leert:**

| Wat | Hoe |
|---|---|
| Begintoestanden per student | Bootstrap uit eerste inschrijvingsrijen — behoudt alle cross-correlaties (bijv. geslacht × opleidingsrichting) |
| Verloop van trajecten over tijd | Markov-transitiematrix per categorische kolom; AR(1) per numerieke kolom |
| Tussenjaren | Empirische verdeling van jaarsprongen (gap van 1, 2, 3 jaar) |
| Aantal inschrijvingsjaren per student | Negatief-binomiale verdeling gefit op echte tellingen |

---

## Stabiele kolommen (`stable_cols`)

Kolommen die per student nooit veranderen — zoals geslacht of geboorteland — geef je op als `stable_cols`. De synthesizer kopieert de beginwaarde naar alle volgende tijdstappen zodat een student nooit halverwege van geslacht wisselt.

Kolommen die je weglaat kunnen over tijd veranderen: opleidingscode, instellingscode, inschrijvingsvorm, bekostiging, enzovoort.

---

## Valideren

Vergelijk de marginale verdelingen van echte en synthetische data direct op de flat bestanden:

```python
from synthetische_onderwijsdata import validate

# Categorische kolommen
cat_df = validate.compare_marginals(df, synthetic)
print(cat_df)

# Numerieke kolommen
num_df = validate.compare_numeric(df, synthetic)
print(num_df)
```

| column | dtype | distance | metric |
|---|---|---|---|
| geslacht | categorical | 0.032 | tv |
| verblijfsjaar_hoger_onderwijs | numeric | 0.41 | wasserstein |
| … | … | … | … |

- **`tv`** (Total Variation): 0 = identiek, 1 = volledig anders. Vuistregel: < 0.05 is goed.
- **`wasserstein`**: schaalafhankelijk (zelfde eenheid als de kolom).

Zie [Hoe het werkt](hoe-het-werkt.md) voor een uitleg van de methodologie en de beperkingen.
