# Verb Category <-> Noun Trends Analysis — Results

How does the noun profile associated with each verb *category* (not individual verb) shift year by year? Method: `README.md` §6, extended along a per-year axis — see this script's module docstring for the full method.

## Corpus summary

| Metric | Value |
|--------|-------|
| Years | 2020–2026 |
| Non-misc categories | 8 |
| Candidate nouns (top 500, stopwords excluded) | 500 |
| Total sentences scanned | 30,573 |

## Top nouns per category, by year

*(top 5 nouns by within-year G² log-likelihood; pairs below 3 co-occurring sentences in a given year are excluded, which is why some year cells show fewer than 5 or read "—")*

### built_environment

| Year | Top nouns (by G²) |
|---|---|
| 2020 | code, territory, economy, city, fuel |
| 2021 | identity, earth, city, house, space |
| 2022 | soil, river, surface, ground, street |
| 2023 | science, material, future, space, energy |
| 2024 | technique, air, energy, supply, area |
| 2025 | quality, place, site, addition, area |
| 2026 | water, ground, assembly, century, condition |

### collaborating

| Year | Top nouns (by G²) |
|---|---|
| 2020 | design, community, knowledge, partner, format |
| 2021 | opportunity, issue, video, community, order |
| 2022 | opportunity, project, resource, experience, approach |
| 2023 | community, experience, generation, collaborator, perspective |
| 2024 | design, lab, initiative, architecture, designer |
| 2025 | community, design, experience, colleague, success |
| 2026 | community, design, framework, project, architecture |

### doing

| Year | Top nouns (by G²) |
|---|---|
| 2020 | environment, design, method, climate, infrastructure |
| 2021 | environment, design, community, climate, structure |
| 2022 | environment, drawing, condition, model, datum |
| 2023 | environment, design, material, space, mean |
| 2024 | environment, order, future, climate, technique |
| 2025 | environment, material, system, scale, resource |
| 2026 | environment, science, modeling, performance, system |

### environment_negative

| Year | Top nouns (by G²) |
|---|---|
| 2020 | water, woman, supply, system, place |
| 2021 | people, disaster, weather, oil, resource |
| 2022 | behavior, resource, environment, finding, landscape |
| 2023 | people, area, climate, future, island |
| 2024 | sea, emission, land, situation, climate |
| 2025 | event, view, weather, inequality, ecosystem |
| 2026 | sea, community, infrastructure, resource, population |

### environment_positive

| Year | Top nouns (by G²) |
|---|---|
| 2020 | change, climate, capacity, emission, system |
| 2021 | design, emission, effect, aim, energy |
| 2022 | transition, community, leader, nation, generation |
| 2023 | change, requirement, effect, climate, awareness |
| 2024 | site, change, sustainability, climate, design |
| 2025 | community, construction, effort, building, housing |
| 2026 | community, connection, place, ecosystem, adaptation |

### knowledge

| Year | Top nouns (by G²) |
|---|---|
| 2020 | architecture, environment, climate, approach, change |
| 2021 | design, architecture, environment, knowledge, issue |
| 2022 | project, design, environment, architecture, humanity |
| 2023 | design, architecture, impact, climate, dynamic |
| 2024 | climate, technique, design, change, energy |
| 2025 | environment, climate, design, system, change |
| 2026 | architecture, design, condition, datum, ability |

### positioning

| Year | Top nouns (by G²) |
|---|---|
| 2020 | climate, change, landscape, design, issue |
| 2021 | climate, design, community, practice, change |
| 2022 | climate, issue, change, project, reflection |
| 2023 | climate, boundary, power, change, justice |
| 2024 | design, science, climate, mission, oil |
| 2025 | climate, community, policy, landscape, site |
| 2026 | climate, design, architecture, community, change |

### questioning

| Year | Top nouns (by G²) |
|---|---|
| 2020 | question, order, change, climate, environment |
| 2021 | change, climate, environment, nature, challenge |
| 2022 | nature, environment, today, architecture, opportunity |
| 2023 | future, world, people, narrative, environment |
| 2024 | site, fuel, form, system, transition |
| 2025 | neighborhood, fiction, narrative, challenge, future |
| 2026 | material, crisis, landscape, geography, climate |

## Noun-profile turnover, first year vs. last year

*(Jaccard overlap of each category's top-10 co-occurring nouns, 2020 vs. 2026 — 1.0 = identical top-10 sets, 0.0 = no overlap at all. Lower means the category's associated vocabulary shifted more.)*

| Category | Jaccard overlap (top-10, first vs. last year) |
|---|---|
| environment_negative | 0.05 |
| built_environment | 0.11 |
| doing | 0.11 |
| environment_positive | 0.11 |
| questioning | 0.11 |
| collaborating | 0.18 |
| knowledge | 0.18 |
| positioning | 0.18 |

## Output files

| File | Description |
|------|-------------|
| `outputs/verb_category_noun_trends.csv` | Full per-year (category, noun) G²/PMI table |

## Notes

- "misc" is excluded — it's the leftover bucket for generic/administrative verbs, not an analytically meaningful category to trend.
- Each year's G²/PMI is computed from that year's OWN sentence totals, not pooled across years — this is what makes the year-to-year comparison meaningful rather than an artifact of corpus size differences between years (see `0_preprocessing/`'s corpus summary for per-year syllabus counts).
- A category is "present" in a sentence if ANY of its member verbs appears — individual member verbs are not weighted or distinguished within a category here; see `verb_noun_occurrence_analysis.py` for the individual-verb-level version.
- Same fragment/bullet-point sentence-segmentation caveat as the rest of this folder applies.