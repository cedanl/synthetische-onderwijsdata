# Hoe het werkt

Deze pagina legt uit wat het model statistisch doet, wat het wél en niet kan reproduceren, en wanneer je de output kunt vertrouwen. Bedoeld voor onderzoekers en BI-specialisten die de output willen beoordelen of in een rapport willen verantwoorden.

---

## De generatiepijplijn in drie stappen

```
Echte data (OBT)
      │
      ▼
split_flat()          ← OBT → dim/feit-tabellen
      │
      ▼
synth.fit()           ← statistische modellen leren
      │
      ▼
synth.generate()      ← synthetische tabellen genereren
```

Het package genereert tabellen in **topologische volgorde**: eerst dimensietabellen (persoon, opleiding, instelling), dan feitentabellen (inschrijvingen, vakresultaten). Zo verwijst elke foreign key altijd naar een al bestaande primary key. Dit garandeert structurele consistentie, maar zegt niets over de statistische kwaliteit van de waarden zelf.

---

## Wat het model leert — per kolom-type

### Categorische kolommen
Na `fit()` worden de empirische frequenties uit de data gebruikt. Als in je data 53% van de inschrijvingen `man` is, produceert de synthesizer ook ~53% `man`. Zonder `fit()` komen de verhoudingen uit de preset-defaults.

### Numerieke kolommen — correlaties
Numerieke niet-sleutelkolommen worden gemodelleerd via een **Gaussian copula**:

1. Elke kolom wordt via de empirische cumulatieve verdeling (ECDF) genormaliseerd naar N(0,1).
2. De **Spearman-correlatiegekritiek** tussen de genormaliseerde kolommen wordt berekend.
3. Een **Cholesky-decompositie** van die correlatiematrix produceert gecorreleerde trekkingen.
4. De getrokken waarden worden teruggetransformeerd via de inverse ECDF naar de originele marginale verdeling.

Dit behoudt zowel de vorm van elke marginale verdeling (niet zomaar een Gaussiaan) als de rangcorrelaties tussen kolommen. Het behoudt **geen** niet-lineaire afhankelijkheden die niet in de Spearman-correlatie zitten.

### Longitudinale kolommen — trajecten over tijd
Voor feitentabellen met een `sequential`-configuratie (zoals `fac_inschrijving`) wordt een **Markov-transitiemodel** gefit:

- **Categorische kolommen**: een transitiematrix P(waarde_t | waarde_{t-1}) per kolom, Laplace-glad om ijle combinaties te ondervangen.
- **Numerieke kolommen**: een AR(1)-proces: waarde_t = α + β · waarde_{t-1} + ε, gefit via OLS.

Het startpunt van elk traject wordt willekeurig getrokken uit de begintoestanden in de trainingsdata. De **duur** van het traject (aantal inschrijvingsjaren) wordt getrokken uit een negatief-binomiale verdeling gefit op de tellingen per entiteit in de trainingsdata.

### Referentiële integriteit
Foreign keys worden na aanmaak van de parent-tabel geregistreerd in een interne registry. Child-records samplen hun FK-waarden uniform uit die registry. Dit garandeert dat er geen orphan records zijn.

---

## Wat het model NIET leert

Dit zijn de bekende beperkingen. Ze zijn relevant als je de kwaliteit van de output wilt beoordelen.

| Wat | Waarom niet |
|---|---|
| **Duur van trajecten afhankelijk van persoonskenmerken** | Het graadmodel is onconditional: elke student trekt uit dezelfde NB-verdeling, ongeacht opleiding of achtergrond |
| **Trajectverloop afhankelijk van persoonskenmerken** | De transitiematrix is populatie-breed; er is geen aparte matrix per cohort, opleiding of instelling |
| **Niet-lineaire afhankelijkheden tussen kolommen** | De Gaussian copula vangt alleen rangcorrelaties; bijv. een interactie-effect tussen opleiding en jaar wordt niet geleerd |
| **Temporele afhankelijkheden over meer dan één stap** | Het Markov-model heeft geheugen van één stap; langere patronen (bijv. terugkeerders na twee jaar uitschrijving) worden niet vastgelegd |
| **Schaarste en zeldzame combinaties** | Zeer zeldzame categorische combinaties in de data worden via de Laplace-smoothing afgevlakt of kunnen in synthetische data ontbreken |

---

## Wanneer vertrouw je de output?

**Geschikt voor:**

- Testen of een dashboard correct omgaat met ontbrekende waarden, uitzonderlijke cohorten of grote aantallen inschrijvingen
- Demonstraties waarbij de data er realistisch uit moet zien maar exacte verhoudingen niet kritisch zijn
- Methodeontwikkeling waarbij een gevuld relationeel schema als testbed dient
- Privacy-veilig delen van een bestand met externe partijen voor exploratieve analyse

**Kritisch beoordelen bij:**

- Analyses waarbij de verhouding van een specifieke categorische variabele (bijv. % uitval per opleiding) moet kloppen → fit op echte data, valideer de marginale verdeling van die kolom
- Analyses waarbij correlaties tussen twee specifieke kolommen cruciaal zijn → controleer of beide kolommen numeriek zijn en dus via de copula zijn gemodelleerd; categorische kolommen worden onafhankelijk gegenereerd
- Trajectanalyses waarbij het verloop sterk afhangt van instroomkenmerken → het model gebruikt één populatie-brede transitiematrix; subgroepverschillen worden niet gereproduceerd

**Niet geschikt voor:**

- Formele privacy-evaluaties (re-identificatierisico is niet gekwantificeerd)
- Causaliteitsonderzoek (synthetische data heeft geen causale structuur van echte data)
- Publicatie als "representatief voor de populatie" zonder validatie

---

## Validatie van output

Het package bevat ingebouwde validatie-utilities in `synthetische_onderwijsdata.validate`.

### Overzichtsrapport (alle tabellen tegelijk)

```python
from synthetische_onderwijsdata import validate

df = validate.report(tables, synthetic, schema)
print(df.to_string())
```

Geeft één DataFrame terug, gesorteerd op grootste afwijking eerst:

| table | column | dtype | distance | metric |
|---|---|---|---|---|
| dim_persoon | geslacht | categorical | 0.032 | tv |
| fac_inschrijving | soort_ho | categorical | 0.018 | tv |
| … | … | … | … | … |

- **`tv`** (Total Variation): 0 = identiek, 1 = volledig anders. Vuistregel: < 0.05 is goed.
- **`wasserstein`**: schaalafhankelijk (zelfde eenheid als de kolom). Vuistregel: < 5% van het bereik.

### Per tabel

```python
# Categorische kolommen
cat_df = validate.compare_marginals(tables["dim_persoon"], synthetic["dim_persoon"])
print(cat_df)

# Numerieke kolommen
num_df = validate.compare_numeric(tables["fac_inschrijving"], synthetic["fac_inschrijving"])
print(num_df)
```

---

## Referenties

| Component | Methodologische basis |
|---|---|
| Topologische generatievolgorde | IRG — [li-jiayu-ljy/irg](https://github.com/li-jiayu-ljy/irg) |
| Gaussian copula met Cholesky | GCM — [JdHondt/gcm](https://github.com/JdHondt/gcm) |
| Referentiële integriteit en schema-extractie | Misata — [rasinmuhammed/misata](https://github.com/rasinmuhammed/misata) |
| Conditionele generatie child-records | RC-TGAN — [croesuslab/RCTGAN](https://github.com/croesuslab/RCTGAN) |
| Longitudinale studenttrajecten | [Huang et al., LAK 2023](https://doi.org/10.1145/3785022.3785085) |

De implementatie gebruikt een vereenvoudigde versie van de aanpakken uit deze bronnen. Zo gebruikt het transitiemodel Markov-ketens en AR(1) in plaats van de volledige Bayesiaanse netwerkstructuurlering uit het LAK-paper, en is het graadmodel onconditional waar IRG een regressiemodel op parent-features gebruikt.
