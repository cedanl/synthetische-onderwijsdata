# Aan de slag

## Installatie

```bash
pip install synthetische-onderwijsdata
```

Of lokaal vanuit de broncode:

```bash
pip install -e ".[dev]"
```

Vereisten: Python ≥ 3.10, numpy, scipy, pandas, pyyaml.

## Eerste gebruik

```python
from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.engine.loader import PresetLoader

schema = PresetLoader.from_builtin("1cijferho")
synth = RelationalSynthesizer(schema, random_state=42)
tables = synth.generate(
    n_entities={"dim_persoon": 1000, "dim_opleiding": 30}
)

print(tables["fac_inschrijving"].head())
```

## Fitten op echte data

```python
import pandas as pd
from synthetische_onderwijsdata.sources._1cijferho.splitter import split_flat

schema = PresetLoader.from_builtin("1cijferho")
df = pd.read_parquet("data/ev_inschrijving.parquet")

tables = split_flat(df, schema)
synth.fit(tables)
synthetic = synth.generate(n_entities={"dim_persoon": 5000, "dim_opleiding": 50})
```
