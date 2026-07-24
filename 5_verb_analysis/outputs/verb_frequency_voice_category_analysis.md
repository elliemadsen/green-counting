# Verb Frequency, Voice & Category Analysis — Results

Tagged with spaCy `en_core_web_sm` over `data/syllabi_text.csv`'s `full_text` column (raw, unfiltered — see README.md §0 for why).

## Corpus summary

| Metric | Value |
|--------|-------|
| Syllabi tagged | 347 |
| Years covered | 2020–2026 |
| Distinct content-verb lemmas (excl. stopverbs) | 3,020 |
| Total VERB-tagged instances (incl. stopverbs, for active/passive) | 63,892 |

## Active vs. passive voice

Of the 28,427 verb instances with an explicit subject (active *or* passive — see caveat below), **19.35%** are passive.

| | Count | % of all verb instances |
|---|---|---|
| Active | 22,926 | 35.9% |
| Passive | 5,501 | 8.6% |
| No explicit subject (imperative/fragment) | 35,465 | 55.5% |

*Syllabus prose is heavy on bullet-point learning objectives and imperatives ("Analyze the role of…") with no explicit subject — these can't be classified active/passive and are reported separately rather than forced into either bucket. See `outputs/active_passive_by_year.csv` for the year-by-year breakdown.*

## Top 20 content verbs (corpus-wide)

| Rank | Lemma | Count |
|------|-------|-------|
| 1 | build | 1,430 |
| 2 | include | 1,261 |
| 3 | develop | 1,246 |
| 4 | work | 856 |
| 5 | base | 836 |
| 6 | focus | 785 |
| 7 | teach | 735 |
| 8 | address | 716 |
| 9 | explore | 678 |
| 10 | engage | 654 |
| 11 | design | 632 |
| 12 | provide | 622 |
| 13 | propose | 596 |
| 14 | create | 586 |
| 15 | support | 565 |
| 16 | integrate | 466 |
| 17 | offer | 464 |
| 18 | understand | 461 |
| 19 | serve | 447 |
| 20 | learn | 443 |

## Verb categories

*(frequency per 1,000 words; slope = OLS trend per year across 2020–2026. Categories are curated by hand in `verb_categories.txt`, not clustered — see notes below.)*

| Category | Corpus count | Distinct lemmas | Freq 2020 | Freq 2026 | Slope/yr | Top members |
|---|---|---|---|---|---|---|
| misc | 43,742 | 2,660 | 58.87 | 66.32 | +1.420 | include, develop, work, base, focus, address, explore, provide, propose, support |
| doing | 5,094 | 59 | 6.78 | 8.40 | +0.153 | build, design, create, write, produce, draw, represent, generate, construct, implement |
| knowledge | 4,877 | 74 | 6.83 | 8.10 | +0.113 | teach, understand, learn, examine, study, investigate, think, recognize, demonstrate, analyze |
| positioning | 2,683 | 43 | 3.36 | 5.99 | +0.348 | relate, connect, align, center, ground, situate, embed, link, position, frame |
| collaborating | 1,639 | 8 | 1.73 | 3.00 | +0.223 | engage, contribute, share, participate, collaborate, join, coordinate, partner |
| questioning | 1,092 | 45 | 1.70 | 1.50 | -0.007 | challenge, reimagine, rethink, question, speculate, pose, redefine, interrogate, reframe, argue |
| environment_negative | 627 | 52 | 0.84 | 0.80 | -0.000 | marginalize, exacerbate, extract, threaten, displace, ignore, overlook, dominate, destroy, damage |
| environment_positive | 623 | 36 | 0.74 | 0.88 | +0.026 | mitigate, empower, sustain, advocate, protect, preserve, thrive, restore, mobilize, decarbonize |
| built_environment | 464 | 43 | 0.55 | 0.67 | +0.005 | occupy, inhabit, site, rebuild, house, zone, isolate, enclose, reclaim, renew |

## Output files

| File | Description |
|------|-------------|
| `outputs/verbs_corpus.csv` | Per-lemma verb counts, corpus + per year, plus observed surface forms |
| `outputs/active_passive.csv` | Per-syllabus active/passive/no-subject counts |
| `outputs/active_passive_by_year.csv` | Year-by-year active/passive/no-subject summary |
| `outputs/verb_categories_corpus.csv` | Per-category verb counts, corpus + per year, plus overall slope (§5) |
| `outputs/verb_categories_trend_by_year.csv` | Long-format per-year frequency table (one row per category x year) — the data behind the Freq/Slope columns below; use this for charting |

## Notes

- Tagged from `full_text` (raw), not `filtered_text` — POS tagging and passive-voice detection need function words that step 0's stopword filter removes.
- Lemmatization collapses inflected forms to one dictionary base form *within a token's tagged part of speech* — it does not merge a word's noun and verb senses. "building" lemmatizes to `build` when spaCy tags that instance VERB ("building a shelter"), but stays `building` when tagged NOUN ("the building faces south") — same surface form, disambiguated per-instance from sentence context, not a static word list. Only VERB-tagged instances are counted here; the noun sense of a word like "building" never enters this report at all (see README.md's lemmatization note for the full explanation, and `verb_noun_occurrence_analysis.py` / `verb_category_noun_trends_analysis.py` for where NOUN-tagged tokens are counted in their own right).
- `stopverbs.txt` excludes 10 generic/administrative verb lemmas from the content-verb counts (still counted in active/passive totals).
- Non-alphabetic tokens and 1-2 letter lemmas are dropped from VERB-tagged tokens before any counting — a cheap filter against PDF-extraction artifacts (stray bullets, hyphen-wrap fragments) that otherwise get mistagged as verbs.
- Verb categories are curated by hand in `verb_categories.txt` (label TAB comma-separated lemmas, same shape as `data/go-words.txt`) rather than produced by algorithmic clustering — any verb lemma not listed there is counted under `misc`. Edit that file and re-run to revise categories; no retraining or reclustering involved.
- A handful of lemmas are first-letter-clipped words (e.g. "nclude" for "include", "esign" for "design") from a PDF font/extraction quirk in a few source files, not real verbs — length filtering catches most extraction noise but not this specific pattern; these fall into `misc` and can be ignored.
- Verb-noun co-occurrence is a separate script (`verb_noun_occurrence_analysis.py`) — see `README.md`.