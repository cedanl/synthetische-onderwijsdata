# Synthetische Onderwijsdata

Een Python-package voor het genereren van synthetische onderwijsdata op basis van echte datasets zoals **1cijferHO**. Het genereert nep-studenten met nep-inschrijvingshistorie die statistisch lijken op echte data — maar geen echte personen bevatten.

## Waarvoor gebruik je dit?

- **Privacy-veilig delen** van analysebestanden met derden
- **Testen van dashboards en pipelines** zonder echte persoonsgegevens
- **Demonstraties en trainingen** waarbij een realistisch-ogende dataset nodig is
- **Methodeontwikkeling** waarbij een grote longitudinale dataset als testbed dient

## Wat is het NIET?

Dit package vervangt geen privacywetgeving. Synthetische data is niet automatisch AVG-proof. Het is een hulpmiddel, geen garantie.

## Snel starten

```bash
pip install synthetische-onderwijsdata
```

```python
import pandas as pd
from synthetische_onderwijsdata.engine.flat import FlatSynthesizer

# Echte data inlezen (output van cedanl/1cijferho-tool)
df = pd.read_parquet("data/ev_inschrijving.parquet")

# Model fitten en synthetische data genereren
synth = FlatSynthesizer(
    entity_key="persoonsgebonden_nummer",
    time_key="inschrijvingsjaar",
    stable_cols=["geslacht", "geboorteland", "hoogste_vooropleiding_voor_het_ho"],
)
synth.fit(df)
synthetic = synth.generate(n_entities=1000)

print(synthetic.head())
```

Zie [Aan de slag](aan-de-slag.md) voor de volledige workflow inclusief normalisatie naar dim/feit-tabellen.

## Navigatie

| | |
|---|---|
| [Aan de slag](aan-de-slag.md) | Installatie, eerste gebruik, fitten op echte data |
| [Hoe het werkt](hoe-het-werkt.md) | Statistische aanpak, wat het model leert, beperkingen |
| [API-referentie](api/index.md) | Volledige klasse- en functiebeschrijving |
