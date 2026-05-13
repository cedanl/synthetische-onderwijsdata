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

## Twee manieren van werken

### Modus 1 — Schema-gedreven (geen echte data)

Genereer data puur op basis van de kolomdefinities in het preset-bestand. Verdelingen en categorieën komen uit het YAML-schema.

Gebruik dit wanneer je geen toegang hebt tot echte data, of voor een eerste snelle test.

```python
from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.engine.loader import PresetLoader

schema = PresetLoader.from_builtin("1cijferho")
synth = RelationalSynthesizer(schema, random_state=42)

tables = synth.generate(
    n_entities={
        "dim_persoon":    1000,
        "dim_opleiding":    30,
        "dim_instelling":   10,
    }
)
```

**Wat je krijgt:** tabellen met de juiste kolommen, correcte FK→PK-relaties, en willekeurige waarden binnen de schema-grenzen. De onderlinge verhoudingen (bijv. hoe lang studenten gemiddeld ingeschreven zijn) komen uit de preset-defaults, niet uit echte data.

---

### Modus 2 — Fitten op echte data (aanbevolen)

Lever een plat 1cijferHO-bestand aan. Het package splitst het automatisch in dimensie- en feitentabellen en leert de statistische eigenschappen van je data.

```python
import pandas as pd
from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.engine.loader import PresetLoader
from synthetische_onderwijsdata.sources._1cijferho.splitter import split_flat

# 1. Schema laden
schema = PresetLoader.from_builtin("1cijferho")

# 2. Plat bestand inlezen (output van cedanl/1cijferho-tool)
df = pd.read_parquet("data/ev_inschrijving.parquet")

# 3. Splitsen in dim/feit-tabellen
tables = split_flat(df, schema)
# → {"dim_persoon": ..., "dim_opleiding": ..., "fac_inschrijving": ..., ...}

# 4. Model fitten op echte data
synth = RelationalSynthesizer(schema, random_state=42)
synth.fit(tables)

# 5. Genereren
synthetic = synth.generate(
    n_entities={
        "dim_persoon":    5000,
        "dim_opleiding":    50,
        "dim_instelling":   20,
    }
)
```

**Wat het model leert bij `fit()`:**

| Wat | Hoe |
|---|---|
| Categorische kolomverdelingen | Empirische frequenties (bijv. echte geslachtsverhouding) |
| Correlaties tussen numerieke kolommen | Gaussian copula via Cholesky-decompositie |
| Aantal inschrijvingen per student | Negatief-binomiale verdeling gefit op echte tellingen |
| Verloop van trajecten over tijd | Markov-transitiematrix (categorisch) + AR(1) (numeriek) |

Zie [Hoe het werkt](hoe-het-werkt.md) voor een uitleg van de methodologie en de beperkingen.

---

## Het platte bestand splitsen

1cijferHO-data komt als één groot plat bestand (één rij per inschrijving). `split_flat()` vertaalt dit naar de dim/feit-structuur die het model verwacht.

```python
tables = split_flat(df, schema)
```

**Wat split_flat doet:**

- **Dimensietabellen** (`dim_persoon`, `dim_opleiding`, `dim_instelling`): neemt de meest recente rij per entiteit (gesorteerd op `inschrijvingsjaar`). Zo bepaalt de laatste bekende jaarsstand de stabiele attributen van een student of opleiding.
- **Feitentabellen** (`fac_inschrijving`, `fac_vak`): alle rijen blijven staan, geen deduplicatie.
- Kolommen die in het preset zijn gedefinieerd maar niet in `df` zitten, worden overgeslagen en als `None` gegenereerd.

---

## Referentiële integriteit controleren

Na het genereren kun je snel de integriteit checken:

```python
fac = synthetic["fac_inschrijving"]
persoon_pks = set(synthetic["dim_persoon"]["persoonsgebonden_nummer"])

orphans = set(fac["persoonsgebonden_nummer"]) - persoon_pks
print(f"Orphan FK's (moet 0 zijn): {len(orphans)}")
```

---

## Eigen schema definiëren

Je bent niet beperkt tot 1cijferHO. Maak een YAML-bestand op basis van jouw dataset:

```yaml
name: mijn_schema
tables:
  dim_student:
    type: dimension
    columns:
      student_id:
        dtype: integer
        role: primary_key
        min: 1
      opleiding:
        dtype: categorical
        categories: [Bachelor, Master, Associate]
        probabilities: [0.6, 0.3, 0.1]

  fac_resultaat:
    type: fact
    columns:
      resultaat_id:
        dtype: integer
        role: primary_key
        min: 1
      student_id:
        dtype: integer
        role: foreign_key
        references: dim_student.student_id
      cijfer:
        dtype: float
        min: 1.0
        max: 10.0
```

```python
from synthetische_onderwijsdata.engine.loader import PresetLoader

schema = PresetLoader.from_yaml("mijn_schema.yaml")
```
