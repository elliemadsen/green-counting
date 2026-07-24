# Verb <-> Noun Occurrence Analysis — Results

Two independent methods per `README.md` §6: sentence-level co-occurrence scored with log-likelihood (G²) and PMI (primary), cross-checked against a freshly-trained Word2Vec embedding (secondary).

## Corpus summary

| Metric | Value |
|--------|-------|
| Sentences scanned | 30,573 |
| Candidate verbs (corpus_count >= 10) | 710 |
| Candidate nouns (top 500, stopwords excluded) | 500 |
| Scored verb-noun pairs (co-occur >= 3 sentences) | 26,432 |

## Top 30 verb-noun pairs, corpus-wide

*(ranked by G² log-likelihood — a higher score means the pairing is more statistically surprising, not just more frequent; PMI shown alongside as a magnitude-of-association measure)*

| Verb | Noun | Category | Co-occur (sentences) | G² | p | PMI |
|---|---|---|---|---|---|---|
| build | environment | doing | 528 | 1995.5 | p < 0.0001 | 3.40 |
| address | challenge | misc | 128 | 399.9 | p < 0.0001 | 3.29 |
| address | climate | misc | 250 | 398.4 | p < 0.0001 | 2.01 |
| embody | carbon | misc | 70 | 376.8 | p < 0.0001 | 4.83 |
| review | peer | misc | 39 | 344.1 | p < 0.0001 | 7.09 |
| solve | problem | misc | 41 | 307.7 | p < 0.0001 | 6.03 |
| tell | story | misc | 27 | 298.2 | p < 0.0001 | 8.21 |
| publish | journal | misc | 39 | 292.4 | p < 0.0001 | 6.25 |
| register | architect | misc | 44 | 271.4 | p < 0.0001 | 5.07 |
| integrate | design | misc | 199 | 266.9 | p < 0.0001 | 1.76 |
| address | change | misc | 160 | 257.9 | p < 0.0001 | 2.12 |
| imagine | future | misc | 53 | 239.9 | p < 0.0001 | 4.31 |
| relate | climate | positioning | 130 | 237.9 | p < 0.0001 | 2.20 |
| engage | community | collaborating | 140 | 228.0 | p < 0.0001 | 2.14 |
| address | issue | misc | 96 | 223.7 | p < 0.0001 | 2.74 |
| face | challenge | misc | 54 | 219.0 | p < 0.0001 | 4.00 |
| rise | sea | misc | 29 | 206.7 | p < 0.0001 | 6.13 |
| contact | question | misc | 36 | 201.3 | p < 0.0001 | 5.00 |
| answer | question | misc | 25 | 199.8 | p < 0.0001 | 6.02 |
| reduce | carbon | misc | 46 | 184.1 | p < 0.0001 | 4.01 |
| learn | lesson | knowledge | 31 | 174.8 | p < 0.0001 | 5.03 |
| rise | level | misc | 32 | 166.7 | p < 0.0001 | 4.78 |
| engage | design | collaborating | 206 | 166.3 | p < 0.0001 | 1.33 |
| ask | question | misc | 45 | 161.1 | p < 0.0001 | 3.72 |
| provide | opportunity | misc | 59 | 155.7 | p < 0.0001 | 3.01 |
| drive | data | misc | 22 | 153.0 | p < 0.0001 | 6.03 |
| inform | design | misc | 108 | 151.1 | p < 0.0001 | 1.81 |
| focus | design | misc | 226 | 150.7 | p < 0.0001 | 1.20 |
| present | conference | misc | 29 | 143.2 | p < 0.0001 | 4.69 |
| build | climate | doing | 257 | 142.2 | p < 0.0001 | 1.10 |

## Embedding cross-check (sample verbs per category)

*(PMI/G² top nouns vs. Word2Vec nearest-neighbour nouns for the 3 most frequent verbs in each non-misc category — agreement between the two independent methods is a stronger signal than either alone; disagreement flags a pairing worth reading in context before citing.)*

| Category | Verb | Top nouns (PMI/G²) | Top nouns (embedding) |
|---|---|---|---|
| built_environment | occupy | people, space, land, innovation, construction | people, area, question, space, issue |
| built_environment | inhabit | territory, world, planet, crisis, population | territory, plant, planet, nation, people |
| built_environment | site | city, neighborhood, technique, country, project | visit, precedent, condition, body, exercise |
| collaborating | engage | community, design, architecture, climate, opportunity | approach, engagement, intersection, community, expert |
| collaborating | contribute | greenhouse, project, gas, emission, construction | peer, influence, addition, outcome, chapter |
| collaborating | share | knowledge, framework, finding, expertise, experience | finding, community, colleague, peer, exchange |
| doing | build | environment, climate, design, performance, science | environment, building, design, envelope, performance |
| doing | design | building, prototype, project, system, space | architecture, thinking, performance, project, planning |
| doing | create | future, community, space, narrative, infrastructure | set, vision, order, neighborhood, access |
| environment_negative | marginalize | community, people, population, environment, inequity | injustice, people, community, population, impact |
| environment_negative | exacerbate | change, climate, weather, impact, event | weather, inequity, change, flooding, temperature |
| environment_negative | extract | component, resource, oil, system, land | resource, transition, supply, insight, ecosystem |
| environment_positive | mitigate | effect, change, climate, impact, risk | risk, intervention, solution, impact, effect |
| environment_positive | empower | community, leader, generation, design, knowledge | knowledge, tool, benefit, method, generation |
| environment_positive | sustain | self, community, engagement, life, platform | care, factor, influence, network, value |
| knowledge | teach | design, theory, architecture, level, history | design, architecture, theory, fellow, coursework |
| knowledge | understand | climate, environment, change, impact, system | impact, approach, influence, factor, decision |
| knowledge | learn | lesson, outcome, objective, opportunity, tool | outcome, lesson, knowledge, opportunity, objective |
| positioning | relate | climate, issue, change, topic, heat | issue, topic, climate, theme, build |
| positioning | connect | opportunity, space, community, region, design | mission, connection, aim, politic, dialogue |
| positioning | align | mission, goal, commitment, justice, priority | mission, goal, priority, commitment, aim |
| questioning | challenge | environment, notion, material, world, society | issue, face, complexity, problem, approach |
| questioning | reimagine | material, system, geography, architecture, neighborhood | approach, agency, aim, critique, agent |
| questioning | rethink | approach, agency, architecture, culture, architect | option, challenge, build, culture, model |

## Output files

| File | Description |
|------|-------------|
| `outputs/verb_noun_pairs.csv` | Full scored verb-noun co-occurrence table (G², PMI, category) |

## Notes

- Co-occurrence is measured at the sentence level (spaCy sentence segmentation on `full_text`) — the same fragment/bullet-point caveat as the active/passive analysis in `verb_frequency_voice_category_analysis.py` applies: syllabus prose that isn't a complete sentence may segment inconsistently, which adds noise but shouldn't systematically bias which nouns a verb associates with.
- Noun candidates exclude both project stopword lists (`0_preprocessing/stopwords/`) so the table isn't dominated by institutional boilerplate ("student", "course", "semester").
- Pairs below 3 co-occurring sentences are dropped before scoring — too sparse for G²/PMI to be meaningful.
- The embedding cross-check retrains its own Word2Vec model (same recipe as `2_semantic_analysis.py`); it's a secondary signal, not a replacement for the G²/PMI numbers above.
- For the per-year, per-category version of this analysis, see `verb_category_noun_trends_analysis.py`.