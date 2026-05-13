# Synthetische Onderwijsdata

Een Python-package voor het genereren van synthetische onderwijsdata op basis van echte datasets zoals **1cijferHO**. Het genereert nep-studenten, nep-inschrijvingen en nep-vakresultaten die statistisch lijken op echte data — maar geen echte personen bevatten.

## Waarvoor gebruik je dit?

- **Privacy-veilig delen** van analysebestanden met derden
- **Testen van dashboards en pipelines** zonder echte persoonsgegevens
- **Demonstraties en trainingen** waarbij een realistisch-ogende dataset nodig is
- **Methodeontwikkeling** waarbij een grote relationele dataset als testbed dient

## Wat is het NIET?

Dit package vervangt geen privacywetgeving. Synthetische data is niet automatisch AVG-proof. Het is een hulpmiddel, geen garantie.

## Snel starten

```bash
pip install synthetische-onderwijsdata
```

```python
from synthetische_onderwijsdata import RelationalSynthesizer
from synthetische_onderwijsdata.engine.loader import PresetLoader

schema = PresetLoader.from_builtin("1cijferho")
synth = RelationalSynthesizer(schema, random_state=42)
tables = synth.generate(n_entities={"dim_persoon": 1000, "dim_opleiding": 30, "dim_instelling": 10})

print(tables["fac_inschrijving"].head())
```

Zie [Aan de slag](aan-de-slag.md) voor de volledige workflow inclusief fitten op echte data.

## Navigatie

| | |
|---|---|
| [Aan de slag](aan-de-slag.md) | Installatie, eerste gebruik, fitten op echte data |
| [Hoe het werkt](hoe-het-werkt.md) | Statistische aanpak, wat het model leert, beperkingen |
| [API-referentie](api/index.md) | Volledige klasse- en functiebeschrijving |
