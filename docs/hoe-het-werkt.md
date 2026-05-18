# Hoe het werkt

Deze pagina legt uit wat het model statistisch doet, wat het wél en niet kan reproduceren, en wanneer je de output kunt vertrouwen. Bedoeld voor onderzoekers en BI-specialisten die de output willen beoordelen of in een rapport willen verantwoorden.

---

## De generatiepijplijn

```
Echte data (flat — één rij per student-jaar)
      │
      ▼
synth.fit()       ← statistische modellen leren op alle kolommen tegelijk
      │
      ▼
synth.generate()  ← n synthetische studenten genereren, elk met volledig traject
```

Het model werkt op de **denormalized flat tabel**: alle kolommen — persoonskenmerken, opleidingskenmerken, inschrijvingsattributen — zitten samen in één rij per student per jaar. Hierdoor leert het model de verbanden tussen kolommen die in een genormaliseerde opzet in aparte tabellen zouden zitten (bijv. geslacht en opleidingsrichting).

---

## Wat het model leert

### Begintoestanden — bootstrap

Per synthetische student wordt een begintoestand gesampled door één willekeurige eerste inschrijvingsrij te trekken uit de trainingsdata. Omdat dit een complete rij is, worden alle cross-kolom verbanden uit de echte data direct meegenomen: een student die begint met `geslacht=V` heeft ook de bijbehorende opleidingskeuze, instelling en overige kenmerken van een echte vrouwelijke student.

### Categorische kolommen — Markov-transitie

Voor elke categorische kolom wordt een empirische transitiematrix P(waarde_t | waarde_{t-1}) gefit over alle studenttrajecten. Met Laplace-smoothing om zeldzame overgangen te ondervangen.

Kolommen die als `stable_cols` zijn opgegeven worden na elke stap teruggezet naar de beginwaarde — ze kunnen nooit wisselen binnen een traject.

### Numerieke kolommen — AR(1)

Voor elke numerieke kolom wordt een AR(1)-proces gefit: waarde_t = α + β · waarde_{t-1} + ε, via OLS op aaneengesloten tijdstapparen. Dit reproduceert zowel het niveau als de autocorrelatie (bijv. stijgende verblijfsjaren).

### Trajectduur — negatief-binomiale verdeling

Het aantal inschrijvingsjaren per student wordt getrokken uit een negatief-binomiale verdeling die gefit is op de empirische tellingen per student in de trainingsdata.

### Tussenjaren

De verdeling van jaarsprongen (gap = 1 jaar normaal, gap = 2 of 3 = tussenjaar) wordt empirisch geleerd en gebruikt om de tijdas van elk traject op te bouwen.

---

## Wat het model NIET leert

| Wat | Waarom niet |
|---|---|
| **Trajectduur afhankelijk van persoonskenmerken** | Het graadmodel is onconditional: elke student trekt uit dezelfde NB-verdeling, ongeacht opleiding of achtergrond |
| **Transitiematrix verschilt per subgroep** | Er is één populatie-brede matrix per kolom; subgroepverschillen (bijv. uitvalpatronen per opleiding) worden niet gereproduceerd |
| **Niet-lineaire afhankelijkheden** | De bootstrap vangt het startpunt correct, maar de DBN-transities modelleren kolommen onafhankelijk van elkaar; interactie-effecten over tijd worden niet geleerd |
| **Temporele patronen over meer dan één stap** | Het Markov-model heeft geheugen van één stap; langere patronen zoals terugkeerders na twee jaar onderbreking worden afgevlakt |
| **Nieuwe combinaties buiten de trainingsdata** | Begintoestanden zijn bootstraps uit echte rijen: combinaties die niet in de trainingsdata voorkomen kunnen niet als startpunt opduiken |

---

## Wanneer vertrouw je de output?

**Geschikt voor:**

- Testen of een dashboard correct omgaat met ontbrekende waarden, uitzonderlijke cohorten of grote aantallen inschrijvingen
- Demonstraties waarbij de data er realistisch uit moet zien maar exacte verhoudingen niet kritisch zijn
- Methodeontwikkeling waarbij een gevuld longitudinaal bestand als testbed dient
- Privacy-veilig delen van een bestand met externe partijen voor exploratieve analyse

**Kritisch beoordelen bij:**

- Analyses waarbij de verhouding van een specifieke categorische variabele (bijv. % uitval per opleiding) moet kloppen → valideer de marginale verdeling van die kolom na synthese
- Analyses waarbij subgroepverschillen (bijv. uitval hbo vs. wo) cruciaal zijn → het model gebruikt één populatie-brede transitiematrix; subgroepen worden niet apart gemodelleerd
- Trajectanalyses waarbij de duur sterk afhangt van instroomkenmerken → het graadmodel is onconditional

**Niet geschikt voor:**

- Formele privacy-evaluaties (re-identificatierisico is niet gekwantificeerd)
- Causaliteitsonderzoek (synthetische data heeft geen causale structuur van echte data)
- Publicatie als "representatief voor de populatie" zonder validatie

---

## Validatie van output

```python
from synthetische_onderwijsdata import validate

# Categorische kolommen — Total Variation per kolom
cat_df = validate.compare_marginals(df, synthetic)

# Numerieke kolommen — Wasserstein-afstand + statistieken
num_df = validate.compare_numeric(df, synthetic)
```

Vuistregels: TV < 0.05 is goed; Wasserstein < 5% van het kolombereik is goed.

---

## Referenties

| Component | Methodologische basis |
|---|---|
| Longitudinale studenttrajecten (DBN) | [Huang et al., LAK 2023](https://doi.org/10.1145/3785022.3785085) |
| Gaussian copula met Cholesky | GCM — [JdHondt/gcm](https://github.com/JdHondt/gcm) |
