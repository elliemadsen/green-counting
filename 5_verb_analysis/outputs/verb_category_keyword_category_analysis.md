# Verb Category <-> Keyword Category Analysis — Results

Does a syllabus's *topic* vocabulary (per `data/Categories.csv`) predict which *verb* vocabulary (per `verb_categories.txt`) it leans on? Method: sentence-level co-occurrence scored with log-likelihood (G²) and PMI — same approach as `verb_noun_occurrence_analysis.py` (README.md §6), with the keyword side pooled into topic categories instead of scored per individual word. See this script's module docstring for the full method.

## Corpus summary

| Metric | Value |
|--------|-------|
| Sentences scanned | 30,573 |
| Verb categories | 8 (built_environment, collaborating, doing, environment_negative, environment_positive, knowledge, positioning, questioning) |
| Keyword categories | 11 (architecture, economic, land, local, material, problem, quantification, representation, solution, theory, universal) |
| Contingency cells | 88 (88 with >= 3 co-occurring sentences, scored below) |

## Full contingency table

*(ranked by G² log-likelihood — how statistically surprising the pairing is, not just how frequent; `% of verb-cat` reads as "of sentences using this verb category, what share also touch this keyword category" and vice versa for `% of keyword-cat`)*

| Verb category | Keyword category | Co-occur (sentences) | % of verb-cat | % of keyword-cat | G² | p | PMI |
|---|---|---|---|---|---|---|---|
| positioning | theory | 1187 | 48.0% | 13.9% | 490.5 | p < 0.0001 | 0.78 |
| positioning | universal | 991 | 40.1% | 15.2% | 490.1 | p < 0.0001 | 0.91 |
| doing | architecture | 2521 | 57.8% | 18.5% | 356.8 | p < 0.0001 | 0.37 |
| doing | universal | 1414 | 32.4% | 21.7% | 341.2 | p < 0.0001 | 0.60 |
| doing | material | 1122 | 25.7% | 23.2% | 335.2 | p < 0.0001 | 0.70 |
| knowledge | theory | 1714 | 39.5% | 20.1% | 318.7 | p < 0.0001 | 0.50 |
| doing | quantification | 1119 | 25.6% | 22.8% | 312.1 | p < 0.0001 | 0.68 |
| knowledge | architecture | 2469 | 56.9% | 18.1% | 311.7 | p < 0.0001 | 0.35 |
| doing | theory | 1714 | 39.3% | 20.1% | 307.1 | p < 0.0001 | 0.49 |
| collaborating | theory | 726 | 46.7% | 8.5% | 261.3 | p < 0.0001 | 0.74 |
| doing | local | 1118 | 25.6% | 21.8% | 258.4 | p < 0.0001 | 0.61 |
| positioning | architecture | 1480 | 59.9% | 10.9% | 253.7 | p < 0.0001 | 0.43 |
| positioning | problem | 550 | 22.2% | 15.5% | 247.7 | p < 0.0001 | 0.94 |
| knowledge | universal | 1322 | 30.5% | 20.3% | 234.8 | p < 0.0001 | 0.52 |
| positioning | local | 705 | 28.5% | 13.7% | 232.2 | p < 0.0001 | 0.77 |
| doing | land | 1224 | 28.0% | 20.6% | 224.7 | p < 0.0001 | 0.53 |
| positioning | land | 776 | 31.4% | 13.1% | 220.7 | p < 0.0001 | 0.69 |
| doing | representation | 635 | 14.5% | 24.6% | 211.6 | p < 0.0001 | 0.78 |
| doing | problem | 786 | 18.0% | 22.2% | 182.8 | p < 0.0001 | 0.64 |
| positioning | solution | 572 | 23.1% | 13.6% | 171.5 | p < 0.0001 | 0.74 |
| questioning | universal | 400 | 38.9% | 6.1% | 170.3 | p < 0.0001 | 0.87 |
| positioning | material | 631 | 25.5% | 13.1% | 168.9 | p < 0.0001 | 0.69 |
| positioning | quantification | 638 | 25.8% | 13.0% | 168.6 | p < 0.0001 | 0.69 |
| knowledge | quantification | 992 | 22.9% | 20.2% | 161.3 | p < 0.0001 | 0.51 |
| doing | economic | 691 | 15.8% | 22.2% | 158.5 | p < 0.0001 | 0.64 |
| collaborating | architecture | 922 | 59.3% | 6.8% | 143.7 | p < 0.0001 | 0.41 |
| questioning | problem | 250 | 24.3% | 7.1% | 134.5 | p < 0.0001 | 1.07 |
| knowledge | local | 998 | 23.0% | 19.5% | 130.9 | p < 0.0001 | 0.46 |
| environment_negative | material | 202 | 34.8% | 4.2% | 128.7 | p < 0.0001 | 1.14 |
| questioning | material | 301 | 29.3% | 6.2% | 122.9 | p < 0.0001 | 0.89 |
| positioning | economic | 423 | 17.1% | 13.6% | 121.6 | p < 0.0001 | 0.75 |
| questioning | theory | 449 | 43.7% | 5.3% | 120.8 | p < 0.0001 | 0.65 |
| environment_positive | problem | 164 | 27.6% | 4.6% | 115.2 | p < 0.0001 | 1.25 |
| environment_negative | land | 222 | 38.3% | 3.7% | 113.2 | p < 0.0001 | 0.98 |
| knowledge | problem | 718 | 16.6% | 20.2% | 111.0 | p < 0.0001 | 0.51 |
| environment_positive | universal | 236 | 39.7% | 3.6% | 104.6 | p < 0.0001 | 0.90 |
| environment_negative | problem | 156 | 26.9% | 4.4% | 103.9 | p < 0.0001 | 1.21 |
| collaborating | universal | 491 | 31.6% | 7.5% | 93.8 | p < 0.0001 | 0.57 |
| environment_negative | economic | 137 | 23.6% | 4.4% | 89.2 | p < 0.0001 | 1.21 |
| environment_positive | material | 185 | 31.1% | 3.8% | 88.1 | p < 0.0001 | 0.98 |
| environment_negative | universal | 222 | 38.3% | 3.4% | 87.8 | p < 0.0001 | 0.84 |
| environment_positive | local | 192 | 32.3% | 3.7% | 87.1 | p < 0.0001 | 0.94 |
| doing | solution | 804 | 18.4% | 19.0% | 85.0 | p < 0.0001 | 0.42 |
| knowledge | land | 1066 | 24.6% | 18.0% | 82.5 | p < 0.0001 | 0.34 |
| questioning | land | 319 | 31.1% | 5.4% | 82.1 | p < 0.0001 | 0.68 |
| environment_positive | economic | 135 | 22.7% | 4.3% | 80.4 | p < 0.0001 | 1.16 |
| collaborating | solution | 340 | 21.9% | 8.1% | 78.9 | p < 0.0001 | 0.66 |
| knowledge | solution | 787 | 18.2% | 18.6% | 75.1 | p < 0.0001 | 0.40 |
| knowledge | material | 882 | 20.3% | 18.3% | 73.7 | p < 0.0001 | 0.36 |
| knowledge | representation | 518 | 11.9% | 20.0% | 72.7 | p < 0.0001 | 0.50 |
| knowledge | economic | 601 | 13.9% | 19.3% | 69.0 | p < 0.0001 | 0.44 |
| environment_positive | land | 198 | 33.3% | 3.3% | 65.1 | p < 0.0001 | 0.78 |
| environment_positive | architecture | 362 | 60.8% | 2.7% | 64.7 | p < 0.0001 | 0.45 |
| built_environment | local | 144 | 32.2% | 2.8% | 64.6 | p < 0.0001 | 0.94 |
| collaborating | land | 428 | 27.5% | 7.2% | 63.7 | p < 0.0001 | 0.51 |
| collaborating | local | 380 | 24.5% | 7.4% | 62.5 | p < 0.0001 | 0.54 |
| collaborating | quantification | 366 | 23.6% | 7.5% | 61.9 | p < 0.0001 | 0.55 |
| questioning | economic | 184 | 17.9% | 5.9% | 58.4 | p < 0.0001 | 0.81 |
| built_environment | land | 155 | 34.7% | 2.6% | 58.3 | p < 0.0001 | 0.84 |
| questioning | local | 262 | 25.5% | 5.1% | 51.9 | p < 0.0001 | 0.60 |
| questioning | architecture | 569 | 55.4% | 4.2% | 50.2 | p < 0.0001 | 0.31 |
| positioning | representation | 303 | 12.3% | 11.7% | 45.1 | p < 0.0001 | 0.54 |
| environment_negative | local | 160 | 27.6% | 3.1% | 43.2 | p < 0.0001 | 0.72 |
| built_environment | material | 121 | 27.1% | 2.5% | 37.2 | p < 0.0001 | 0.78 |
| environment_positive | theory | 232 | 39.0% | 2.7% | 34.5 | p < 0.0001 | 0.48 |
| built_environment | universal | 148 | 33.1% | 2.3% | 33.7 | p < 0.0001 | 0.63 |
| environment_positive | quantification | 149 | 25.0% | 3.0% | 32.2 | p < 0.0001 | 0.64 |
| environment_positive | solution | 132 | 22.2% | 3.1% | 31.2 | p < 0.0001 | 0.68 |
| environment_negative | theory | 221 | 38.1% | 2.6% | 28.6 | p < 0.0001 | 0.45 |
| questioning | quantification | 228 | 22.2% | 4.6% | 27.4 | p < 0.0001 | 0.47 |
| collaborating | representation | 188 | 12.1% | 7.3% | 25.2 | p < 0.0001 | 0.52 |
| collaborating | economic | 212 | 13.6% | 6.8% | 19.6 | p < 0.0001 | 0.42 |
| collaborating | problem | 234 | 15.1% | 6.6% | 17.8 | p < 0.0001 | 0.38 |
| built_environment | economic | 74 | 16.6% | 2.4% | 17.3 | p < 0.0001 | 0.70 |
| collaborating | material | 300 | 19.3% | 6.2% | 14.3 | p < 0.001 | 0.29 |
| built_environment | architecture | 236 | 52.8% | 1.7% | 12.4 | p < 0.001 | 0.24 |
| environment_negative | architecture | 294 | 50.7% | 2.2% | 8.9 | p < 0.01 | 0.19 |
| environment_negative | quantification | 119 | 20.5% | 2.4% | 8.2 | p < 0.01 | 0.35 |
| questioning | representation | 108 | 10.5% | 4.2% | 5.5 | p < 0.05 | 0.32 |
| built_environment | problem | 66 | 14.8% | 1.9% | 4.1 | p < 0.05 | 0.35 |
| environment_negative | representation | 63 | 10.9% | 2.4% | 4.1 | p < 0.05 | 0.36 |
| built_environment | theory | 144 | 32.2% | 1.7% | 4.0 | p < 0.05 | 0.21 |
| environment_positive | representation | 60 | 10.1% | 2.3% | 2.0 | not significant | 0.25 |
| built_environment | quantification | 79 | 17.7% | 1.6% | 0.9 | not significant | 0.14 |
| built_environment | representation | 33 | 7.4% | 1.3% | 0.7 | not significant | -0.20 |
| questioning | solution | 150 | 14.6% | 3.6% | 0.6 | not significant | 0.08 |
| environment_negative | solution | 86 | 14.8% | 2.0% | 0.5 | not significant | 0.10 |
| built_environment | solution | 62 | 13.9% | 1.5% | 0.0 | not significant | 0.01 |

## Which verb category is *distinctive* to each keyword category

*(top 2 by PMI, not raw co-occurrence % — raw % mostly just reflects which verb category is common everywhere: `doing` and `knowledge` are corpus-wide the two most frequent categories by a wide margin (see README.md's category-trends table), so ranking by raw co-occurrence share surfaces them almost everywhere and says little about any given topic specifically. PMI instead measures how far above chance the pairing is, after accounting for how common each category already is on its own — that's the reading that actually answers "what verb vocabulary is this topic's own," as opposed to "what verb vocabulary is common in general.")*

| Keyword category | 1st by PMI | 2nd by PMI |
|---|---|---|
| architecture | environment_positive (PMI 0.45) | positioning (PMI 0.43) |
| economic | environment_negative (PMI 1.21) | environment_positive (PMI 1.16) |
| land | environment_negative (PMI 0.98) | built_environment (PMI 0.84) |
| local | environment_positive (PMI 0.94) | built_environment (PMI 0.94) |
| material | environment_negative (PMI 1.14) | environment_positive (PMI 0.98) |
| problem | environment_positive (PMI 1.25) | environment_negative (PMI 1.21) |
| quantification | positioning (PMI 0.69) | doing (PMI 0.68) |
| representation | doing (PMI 0.78) | positioning (PMI 0.54) |
| solution | positioning (PMI 0.74) | environment_positive (PMI 0.68) |
| theory | positioning (PMI 0.78) | collaborating (PMI 0.74) |
| universal | positioning (PMI 0.91) | environment_positive (PMI 0.90) |

## Which keyword category is *distinctive* to each verb category

*(same PMI-based reading, the other direction)*

| Verb category | 1st by PMI | 2nd by PMI |
|---|---|---|
| built_environment | local (PMI 0.94) | land (PMI 0.84) |
| collaborating | theory (PMI 0.74) | solution (PMI 0.66) |
| doing | representation (PMI 0.78) | material (PMI 0.70) |
| environment_negative | economic (PMI 1.21) | problem (PMI 1.21) |
| environment_positive | problem (PMI 1.25) | economic (PMI 1.16) |
| knowledge | universal (PMI 0.52) | problem (PMI 0.51) |
| positioning | problem (PMI 0.94) | universal (PMI 0.91) |
| questioning | problem (PMI 1.07) | material (PMI 0.89) |

## Output files

| File | Description |
|------|-------------|
| `outputs/verb_category_keyword_category.csv` | Full scored contingency table (all cells, including those below the co-occurrence floor) |
| `outputs/verb_keyword_pmi_heatmap.png` | Heatmap: distinctiveness (PMI, diverging color scale centered on independence) |
| `outputs/verb_keyword_count_heatmap.png` | Heatmap: evidence weight (raw co-occurring sentence counts) |

## Notes

- `data/Categories.csv` has 186 surface forms across 11 topic categories — much sparser than the top-500-word noun vocabulary used elsewhere in this folder, since that file was hand-picked for a different, smaller purpose and had not previously been used in any analysis script. Treat coverage, not just the associations themselves, as a finding: a low keyword-category sentence count means most of the corpus's topic vocabulary isn't captured by this file yet, not that the topic is rare. Extending `data/Categories.csv` the way `verb_categories.txt` was extended (README.md §5 — a handful of seed words per category, grown by LLM semantic judgment) is a natural next step if these results look promising but thin.
- Categories.csv's `verb` tag marks a word's part of speech, not a topic — it's dropped here, not treated as a ninth keyword category.
- A sentence counts toward a keyword category if any of its non-VERB-tagged tokens matches a Categories.csv word or alias, by lemma or by raw surface form (both are checked because the source file itself mixes conventions — see module docstring).
- Keyword categories are multi-label (a word like "economy" carries `economic, universal, quantification` simultaneously, per the source file); verb categories remain single-label per verb, per `verb_categories.txt`'s existing design.
- Same sentence-fragment/bullet-point caveat as the rest of this folder: syllabus prose that isn't a complete sentence may segment inconsistently.
- Cells below 3 co-occurring sentences are excluded from the report tables above (still present, flagged `scored=False`, in the CSV) — too sparse for G²/PMI to be meaningful.