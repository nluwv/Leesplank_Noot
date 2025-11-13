---
language:
- nl
tags:
- text-simplification
- dutch
- b1
- government
- public-service
- ai-act
- responsible-ai
license: eupl-1.2
pipeline_tag: text2text-generation
---

# Leesplank: Noot

**Leesplank: Noot** is een set van drie Nederlandstalige taalmodellen, elk gefinetuned op **een zorgvuldig samengestelde Nederlandstalige dataset** voor tekstvereenvoudiging naar **B1-niveau**. De modellen zijn ontwikkeld voor **overheids- en publieke communicatie** en sluiten nauw aan bij de **geest én letter van de EU AI Act**: transparant, herleidbaar, controleerbaar en veilig inzetbaar in productie. Door deze modellen open source te publiceren, voorkomen we onnodige herontwikkeling, besparen we publieke middelen en verhogen we de consistentie van communicatie.

## 🌍 Impact & Value
- **Situatie:** Er zijn al tientallen chatbots en meer dan tien briefvereenvoudigers ontwikkeld in de publieke sector, vaak met overlappende functionaliteit.  
- **Preventief effect:** Door deze modellen open source te publiceren, kunnen naar schatting **100 toekomstige duplicaatprojecten** worden voorkomen.  
- **Tweede-orde voordelen:**  
  - Lagere ontwikkelkosten.  
  - Minder externe consultancy-trajecten met beperkte meerwaarde.  
  - Hogere consistentie in communicatie tussen organisaties.  
- **Geschatte ROI:** 14–18× binnen het eerste jaar.

## Hoe gebruik je Leesplank: Noot AI-compliant volgens de AI Act?

Om de Leesplank: Noot modellen AI Act-compliant te gebruiken, volg je deze stappen:

1. **Doelbinding documenteren**: Beschrijf duidelijk waarvoor je het model gebruikt, zoals tekstvereenvoudiging voor publieke communicatie.
1. **Menselijk toezicht waarborgen**: Laat alle kritieke of impactvolle output door een mens controleren voordat het gebruikt wordt.
1. **Transparantie bieden**: Informeer gebruikers altijd dat de tekst (deels) door AI is gegenereerd, bijvoorbeeld via een label zoals "AI-vereenvoudigde tekst".
1. **Technische documentatie raadplegen**: Gebruik de gedetailleerde documentatie om de capaciteiten en beperkingen van het model te begrijpen. Wij willen samenwerken om deze documentatie compleet te krijgen.
1. **Bias en risico monitoren**: Voer regelmatig tests uit met representatieve data om bias en risico's te minimaliseren.
1. **Logboek bijhouden**: Registreer input en output van het model veilig en versleuteld.
1. **Licentie naleven**: Zorg dat je wijzigingen of verdere distributie van het model onder de EUPL-1.2 licentie houdt.
1. **registreren (aanbevolen)**: registreer het gebruik van deze modellen in een productieomgeving in het algoritmeregister, omwille van transparantie en vertrouwen.
1. **FRIA (fundamentele rechten impact assessment, aanbevolen): voer een lichte versie van deze assessment uit, vooral bij gebruik in de publieke sector.
1. **beoogd gebruik: Limited Risk**: wanneer de toepassing van dit model maakt dat de risicoklasse hoger uitkomt, gelden er strengere regels. Raadpleeg artikel 6 en annex III van de AI act.

## 📊 Gebruik
### Aanbevolen voor:
- Vereenvoudigen van overheidsteksten.
- Begrijpelijk maken van klantcommunicatie en formulieren.
- Genereren van B1-content voor publieke dienstverlening.

### Niet aanbevolen voor:
- Besluitvorming zonder menselijk toezicht.
- Domeinen waar nuanceverlies direct schadelijk kan zijn.

---

## 📦 Modellen
Deze release bevat drie afzonderlijke modellen, identiek qua trainingsdata maar met verschillende basismodellen:

1. **Leesplank: Noot – A**  
   Basismodel: *Granite-3.3-2b-instruct* – [HuggingFace](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
2. **Leesplank: Noot – B**  
   Basismodel: *Llama-3.2-3b-instruct* – [HuggingFace](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
3. **Leesplank: Noot – C**  
   Basismodel: *[Naam]* – [onderscheidende kenmerken?]

## 🛠 Training Data
`UWV/Leesplank_NL_wikipedia_simplifications_preprocessed` is een dataset bestaande uit vereenvoudigingen van Wikipedia teksten. Unieke aanpak: in plaats van op woord- of zinsniveau te vereenvoudigen, vereenvoudigen wij op paragraaf niveau. Dit leidt tot samenhangender en leesbaarder teksten, waarin out of vocabulary termen op natuurlijke wijze in tekst worden uitgelegd. Deze trainingsdata is legaal verkregen en er is geen inbreuk gedaan op auteursrechten. Een uitspraak over de onderliggende modellen doen wij niet.
- **Filtering:**    
  - Duplicaten verwijderd.
  - Persoonsgegevens werden niet ingevoerd, maar ook niet gewist voor zover die op Wikipedia stonden t.t.v. de snapshot.
  - gesorteerd op verschil tussen in- en uitvoer.
  - Schadelijke of discriminerende taal uitgesloten doordat de vereenvoudigen door de meest strikte filtering van MicroSoft/OpenAI Gpt zijn gevoerd voor zowel in- als uitvoer. Dit is beschreven op: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/content-filters
  - Zie de [HuggingFace repository](https://huggingface.co/datasets/UWV/Leesplank_NL_wikipedia_simplifications_preprocessed) voor meer info.

## 🔍 Evaluatie

### SARI Scores (hogere score = betere vereenvoudiging)

**Base Modellen:**
- **Granite-3.3-2b-instruct (base):** `44.16`
- **Llama-3.2-3b-instruct (base):** `##.#`
- **[Model C base]:** `##.#`

**Finetuned Modellen:**
- **Leesplank: Noot – A (finetuned):** `68.89`
- **Leesplank: Noot – B (finetuned):** `##.#`
- **Leesplank: Noot – C (finetuned):** `##.#`

**Gemiddelde score over alle finetuned modellen:** `##.#`

> **Waarom geen BLEU?** BLEU meet woord-overlap en is niet geschikt voor de kwaliteit van vereenvoudiging.  
> SARI meet behouden, toevoegen en verwijderen van relevante woorden en is de standaard voor tekstvereenvoudiging.

---

## ⚠️ Beperkingen & Risico's
- Nuanceverlies kan optreden.
- Bias kan aanwezig blijven door bronteksten.
- Minder geschikt buiten domeinen in de trainingsdata.

**Mitigatie:**
- Menselijke review.
- Domeinspecifieke finetuning.
- Continue kwaliteitsmonitoring.

---

## 📬 Contact & Info
- **Maintainer:** UWV Innovatie Hub - innovatie@uwv.nl
- **Feedback:** via Hugging Face issues.  
- **Bijdragen:** [GitHub]

---

## 📄 Licentie
- **Licentie:** [EUPL-1.2] – open, herbruikbaar, en compatibel met AIA-vereisten.
- Vrij gebruik, wijziging en distributie met bronvermelding.

- Aangepaste versies moeten onder dezelfde licentie blijven.
