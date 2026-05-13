# synthetische_onderwijsdata

Modulair Python-package voor het genereren van synthetische **relationele** data met behoud van:

- **Referentiële integriteit** — FK-waarden verwijzen altijd naar bestaande PK-waarden
- **Statistische correlaties** — Generative Correlation Manifolds (GCM) via Cholesky-decompositie
- **Temporele semantiek** — Dynamic Bayesian Networks (DBN) voor longitudinale trajecten

Ontworpen voor complexe schema's zoals **1cijferHO** (CBS/DUO).

---

## Installatie

```bash
pip install synthetische_onderwijsdata          # via PyPI (toekomstig)
# of lokaal:
pip install -e ".[dev]"
```

Vereisten: Python ≥ 3.10, numpy, scipy, pandas, pyyaml.

---

## Snel starten

```python
from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.presets.loader import PresetLoader

# Laad het ingebouwde 1cijferHO-schema
schema = PresetLoader.from_builtin("1cijferho")

# Genereer — geen echte data nodig voor schema-gedreven generatie
synth = RelationalSynthesizer(schema, random_state=42)
tables = synth.generate(
    n_entities={"dim_student": 1000, "dim_opleiding": 30}
)

print(tables["fac_inschrijving"].head())
```

### Fitten op echte data

```python
import pandas as pd

real_data = {
    "dim_student": pd.read_parquet("data/dim_student.parquet"),
    "dim_opleiding": pd.read_parquet("data/dim_opleiding.parquet"),
    "fac_inschrijving": pd.read_parquet("data/fac_inschrijving.parquet"),
}

synth.fit(real_data)
synthetic = synth.generate(n_entities={"dim_student": 5000, "dim_opleiding": 50})
```

---

## Architectuur

```
src/synthetische_onderwijsdata/
├── core/
│   ├── engine.py       # IRG-engine — topologische generatielus
│   ├── topology.py     # DAG + Kahn's topologisch sorteren
│   ├── gcm.py          # Gaussian Copula / Cholesky-correlaties
│   └── integrity.py    # PK-registry + FK-sampling
├── logic/
│   ├── sequential.py   # State-aware longitudinale generator
│   ├── dbn.py          # DBN-transitiemodel (Markov + AR(1))
│   └── degree.py       # Negatief-binomiale graad-steekproef
└── presets/
    ├── loader.py        # YAML-schema parser → dataklassen
    ├── hooks.py         # Business-rule handhaving (post-generatie)
    └── schemas/
        └── 1cijferho.yaml
```

### Generatievolgorde (IRG-principe)

```
dim_student ──┐
               ▼
dim_opleiding ─► fac_inschrijving
```

Tabellen worden in topologische volgorde gegenereerd zodat elke FK naar een al bestaande PK verwijst.

---

## Eigen preset aanmaken

```yaml
# myproject/schemas/mijn_schema.yaml
name: mijn_schema
tables:
  dim_klant:
    type: dimension
    columns:
      klant_id:
        dtype: integer
        role: primary_key
        min: 1
      segment:
        dtype: categorical
        categories: [A, B, C]
        probabilities: [0.5, 0.3, 0.2]

  fac_order:
    type: fact
    columns:
      order_id:
        dtype: integer
        role: primary_key
        min: 1
      klant_id:
        dtype: integer
        role: foreign_key
        references: dim_klant.klant_id
      bedrag:
        dtype: float
        min: 5.0
        max: 500.0
    hooks:
      - rule: "bedrag > 0"
```

```python
from synthetische_onderwijsdata.presets.loader import PresetLoader

schema = PresetLoader.from_yaml("myproject/schemas/mijn_schema.yaml")
```

---

## Tests uitvoeren

```bash
pytest
```

---

## Referenties

| Techniek | Bron |
|---|---|
| IRG (Incremental Relational Generator) | [li-jiayu-ljy/irg](https://github.com/li-jiayu-ljy/irg) |
| GCM (Generative Correlation Manifolds) | [JdHondt/gcm](https://github.com/JdHondt/gcm) |
| Referentiële integriteit schema-extractie | [rasinmuhammed/misata](https://github.com/rasinmuhammed/misata) |
| RC-TGAN conditionele generatie | [croesuslab/RCTGAN](https://github.com/croesuslab/RCTGAN) |
| DBN studenttrajecten | [doi:10.1145/3785022.3785085](https://doi.org/10.1145/3785022.3785085) |

---

## Licentie

EUPL-1.2
