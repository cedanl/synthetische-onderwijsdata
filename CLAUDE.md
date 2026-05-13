# CLAUDE.md — synthetische-onderwijsdata

## Wat is dit project

Modulair Python-package voor het genereren van synthetische relationele onderwijsdata. Primaire doelgroep: CBS/DUO-datasets zoals **1cijferHO**. Het package genereert data die referentiële integriteit en longitudinale structuur behoudt — zonder echte persoonsgegevens.

Licentie: EUPL-1.2.

---

## Pakketstructuur

```
src/synthetische_onderwijsdata/
├── schema.py                        # pure dataklassen (ColumnSchema, TableSchema, Schema)
├── engine/
│   ├── core.py                      # RelationalSynthesizer — IRG-generatielus
│   ├── topology.py                  # DAG + Kahn topologisch sorteren
│   ├── gcm.py                       # Gaussian Copula / Cholesky-correlaties
│   ├── integrity.py                 # PK-registry + FK-sampling
│   ├── samplers.py                  # kolomgeneratie per dtype
│   ├── hooks.py                     # declaratieve bedrijfsregels (generiek)
│   ├── loader.py                    # YAML-schemaparser → dataklassen (generiek)
│   └── longitudinal/
│       ├── dbn.py                   # DBN transitiemodel (Markov + AR(1))
│       ├── degree.py                # Negatief-binomiaal graadmodel
│       └── sequential.py           # state-aware longitudinale generator
└── sources/
    └── _1cijferho/
        ├── schema.yaml              # 1cijferHO kolomdefinities
        └── splitter.py             # plat bestand → dim/feit-tabellen
```

**Snijlijnen:**
- `engine/` weet niets van specifieke databronnen — puur generiek.
- `sources/<naam>/` bevat alles wat brongebonden is: schema YAML, splitter, eventuele eigen hooks.
- Nieuwe databron = nieuwe directory in `sources/`, nul aanraking van `engine/`.

---

## Documentatie bijhouden

Bij elke wijziging: overweeg of `docs/` mee moet. Vuistregels:

- Nieuwe publieke klasse of functie → voeg toe aan `docs/api/index.md` via `:::` directive.
- Nieuw concept of gebruik-flow → voeg een sectie toe aan `docs/aan-de-slag.md`.
- Nieuwe databron onder `sources/` → voeg een eigen pagina toe en update de `nav:` in `mkdocs.yml`.
- Gewijzigde install-stap of dependency → update `docs/aan-de-slag.md`.

Lokaal testen: `uv run mkdocs build --strict`. De `--strict` vlag faalt op gebroken `:::` directives en ontbrekende nav-items.

---

## Ontwerpprincipes

### IRG — topologische generatievolgorde
Tabellen worden gegenereerd in de volgorde die de DAG oplegt: eerst dimensies, dan feiten. FK-waarden worden na PK-registratie gesampled. Wijzig deze volgorde nooit impliciet — het is de garantie voor referentiële integriteit.

### Longitudinale data — niet platslaan
Meerdere rijen per entiteit (bijv. studiejaren per student) worden gegenereerd via `engine/longitudinal/sequential.py` met DBN-transities. Nooit `pivot` of `unstack` toepassen om temporele data te herstructureren — dat vernietigt de sequentiële semantiek.

### Graadmodel
Het aantal feitenrijen per dimensie-entiteit wordt bepaald door `DegreeModel` (negatief-binomiaal). Wanneer er geen echte data is: gebruik `DegreeModel.from_config(mean=..., dispersion=...)` vanuit de YAML `degree:` sectie.

### Geen evaluatielogica in de engine
XGBoost, LightGBM en andere utility-evaluatie (TSTR — Train on Synthetic, Test on Real) horen **niet** in dit package. Evaluatie is een externe stap. Voeg geen ML-modellen voor validatie toe aan `engine/` of `sources/`.

### Schone input
Het package gaat uit van schone input (cedanl/1cijferho-tool output). Geen data-cleaning logica toevoegen — dat is verantwoordelijkheid van de caller.

---

## Databronnen en academische basis

| Component | Bron |
|---|---|
| IRG-engine (topologische generatielus) | [li-jiayu-ljy/irg](https://github.com/li-jiayu-ljy/irg) |
| GCM (Cholesky-correlaties) | [JdHondt/gcm](https://github.com/JdHondt/gcm) |
| Referentiële integriteit / schema-extractie | [rasinmuhammed/misata](https://github.com/rasinmuhammed/misata) |
| Conditionele generatie child-records | [croesuslab/RCTGAN](https://github.com/croesuslab/RCTGAN) |
| DBN voor longitudinale studenttrajecten | [doi:10.1145/3785022.3785085](https://doi.org/10.1145/3785022.3785085) |

De DBN-methodiek (dbn.py) modelleert P(X_t | X_{t-1}) per kolom: Markov-transitiematrix voor categorische kolommen, AR(1) voor numerieke. Dit volgt direct de aanpak uit het ACM-paper over studenttrajecten.

---

## Testen

```bash
uv run pytest           # volledige suite
uv run pytest -k gcm    # specifieke module
```

Tests staan in `tests/` en spiegelen de pakketstructuur (`test_core/` → `engine/`, `test_logic/` → `engine/longitudinal/`, `test_presets/` → `engine/loader` + `engine/hooks`).

---

## Nieuwe databron toevoegen

1. Maak `src/synthetische_onderwijsdata/sources/<naam>/`.
2. Voeg `schema.yaml` toe (zie `_1cijferho/schema.yaml` als voorbeeld).
3. Voeg optioneel een `splitter.py` toe als de bron een plat bestand levert.
4. Registreer de bron in `docs/` en update de `nav:` in `mkdocs.yml`.
5. Voeg integratietests toe in `tests/test_integration.py`.
