# Step 4: Verb Analysis

## What this step does

Steps 0–2 look at *which topics* show up in the syllabi (climate, material, justice, …)
and how their meaning drifts over time. This step looks at the **verbs** instead: not just
what a syllabus is about, but what it asks students to *do* to that subject matter — build
it, question it, position it, teach it — and whether that framing is changing between 2020
and 2026.

Three independent, self-contained scripts:

| Script | Answers |
|---|---|
| `verb_frequency_voice_category_analysis.py` | Which verbs are most common, corpus-wide and per year? What fraction of verb usage is active vs. passive voice? Which thematic category (from `verb_categories.txt`) does each verb belong to, and how is each category trending? |
| `verb_noun_occurrence_analysis.py` | Which nouns does a given verb tend to appear alongside — e.g. does "analyze" really co-occur with "climate"? |
| `verb_category_noun_trends_analysis.py` | For a whole category of verbs (not one verb at a time), how does its noun profile shift year by year — e.g. what does the "doing" category get done *to* in 2020 vs. 2026? |

All three tag the **raw, unfiltered** `full_text` column of `data/syllabi_text.csv` with
spaCy (`en_core_web_sm`) — see §0 below for why that matters, and why no changes to step
0's preprocessing were needed to make this possible.

## How to run

```bash
python3 verb_frequency_voice_category_analysis.py            # ~2–3 min, full corpus
python3 verb_noun_occurrence_analysis.py                      # ~3–4 min, full corpus
python3 verb_category_noun_trends_analysis.py                 # ~2–3 min, full corpus
```

Each accepts `--limit N` to tag only the first N syllabi, for a quick smoke-test before a
full run. Scripts are independent — none of them import from another — so they can be
re-run individually after editing `verb_categories.txt` or `stopverbs.txt` without
re-running the others.

## Files in this folder

| File | Purpose |
|------|---------|
| `verb_categories.txt` | Curated thematic verb categories. One category per line: `label` TAB `comma, separated, verb, lemmas` (same shape as `data/go-words.txt`). Any verb lemma not listed defaults to `misc`. Hand-edit and re-run — no retraining or reclustering step. |
| `stopverbs.txt` | Generic/administrative verb lemmas (one per line) excluded from the *content*-verb counts — still counted toward active/passive totals, since e.g. passive "attendance is required" is a genre feature worth keeping, not noise. Currently a small placeholder starter list; expand as needed. |
| `outputs/` | All generated CSVs and markdown reports (see table below) |

## Output files

| File | Description |
|------|-------------|
| `outputs/verbs_corpus.csv` | Per-lemma verb counts, corpus + per year, plus the observed surface forms that lemmatized into each row |
| `outputs/active_passive.csv` | Per-syllabus active / passive / no-explicit-subject counts |
| `outputs/active_passive_by_year.csv` | Same, aggregated per year |
| `outputs/verb_categories_corpus.csv` | Per-category verb counts, corpus + per year |
| `outputs/verb_frequency_voice_category_analysis.md` | Generated report for the above three |
| `outputs/verb_noun_pairs.csv` | Full verb↔noun co-occurrence table, scored with G² and PMI |
| `outputs/verb_noun_occurrence_analysis.md` | Generated report for the above, plus the embedding cross-check |
| `outputs/verb_category_noun_trends.csv` | Per-year (category, noun) co-occurrence table |
| `outputs/verb_category_noun_trends_analysis.md` | Generated report for the above, plus a turnover summary |

## Methodology notes

**§0 — Why `full_text`, not `filtered_text`.** `data/syllabi_text.csv` already stores both
a raw, unfiltered `full_text` column and a stopword-stripped, lowercased `filtered_text`
column (used by steps 1–2). POS tagging, lemmatization, and passive-voice detection all
depend on function words and casing (*is, was, to, will, by*) that the stopword filter
deliberately removes. So no changes to step 0's preprocessing were needed — this step just
reads the column that was already there.

**§0a — Lemmatization, and how ambiguous noun/verb forms like "building" are handled.**
Lemmatization reduces an inflected token to its dictionary base form — *built, builds,
building* (as a verb) all collapse to `build` — using spaCy's rule-based lemmatizer for
regular patterns plus a lookup table for irregulars (*taught → teach*, *wrote → write*).
`verbs_corpus.csv`'s `surface_forms` column records exactly which inflected forms rolled
into each lemma row, for auditing.

The important part is that **lemmatization happens per-token, using that token's own POS
tag** — it does not merge a word's noun sense and verb sense. "building" is the textbook
case: the same surface form is tagged (and lemmatized) differently depending on its role in
the sentence, not looked up in a static word list. Verified directly against the corpus's
own kind of sentences:

| Sentence | Token | POS tag | Lemma |
|---|---|---|---|
| "Students will spend the semester **building** a full-scale shelter prototype." | building | VERB (VBG) | `build` |
| "This studio focuses on **building** community resilience through design." | building | VERB (VBG) | `build` |
| "The **building** faces south to maximize passive solar heating." | building | NOUN (NN) | `building` |
| "The studio surveyed several existing **buildings** on the site." | buildings | NOUN (NNS) | `building` |

Since `verb_frequency_voice_category_analysis.py` only counts tokens tagged `POS == VERB`,
the noun sense of "building" (or "model", "design", "plan", "frame", and every other
noun/verb-ambiguous word in English) never enters that report's counts at all — it's
neither dropped nor miscounted, it simply belongs to a different analysis. Those NOUN-tagged
instances *are* picked up, in their own right, as noun candidates in
`verb_noun_occurrence_analysis.py` and `verb_category_noun_trends_analysis.py` — which is
exactly how "represent" ended up co-occurring with the nouns "model" and "modeling" in the
worked examples above: two independently-tagged words, not a tagging error.

**§3 — Active vs. passive voice.** A clause is scored passive if its verb has an
`nsubjpass`/`auxpass`/`agent` dependency child; active if it has an `nsubj` child; anything
else — mainly imperative/infinitive learning-objective bullets ("Analyze the role of…")
with no explicit subject — falls into a third **"no explicit subject"** bucket rather than
being forced into active or passive. That third bucket is currently the *majority* of all
verb instances (~55%), which is itself a genre feature of syllabus prose worth noting:
these documents are written more as instructions and fragments than as complete sentences.
Sentence-fragment-heavy text is also harder for spaCy's parser (trained on continuous
prose) than full sentences, so treat the active/passive split as directionally reliable
rather than exact — a spot-check of ~20 flagged instances is worth doing before citing the
headline percentage.

**§5 — Category curation.** Categories were **not** produced by algorithmic clustering. An
earlier version of this step trained a Word2Vec model on the corpus and k-means-clustered
verb-lemma embeddings, but that only produced unlabeled groups that still needed
interpreting from scratch, with a cluster count (`k=18`) that had no principled connection
to the categories that actually matter for this research question. Instead, categories were
defined directly: 7 seeded by the researcher with example verbs, extended across the
observed ~3,000-lemma vocabulary by LLM semantic judgment, plus one added category
(`collaborating`) for a large, coherent group of verbs — *engage, collaborate, participate,
contribute, partner, coordinate, join, share* — that had no home in the other 7. Any verb
not assigned to a category defaults to `misc`, which is a legitimate bucket, not an error:
most verb *instances* in syllabus prose are generic/administrative filler ("include",
"provide", "focus") that don't belong to any thematic category. On the full corpus, about
28% of verb instances land in a named category.

**§6 — Verb↔noun co-occurrence.** Two independent methods, both computed:
1. **Sentence-level co-occurrence**, scored with the Dunning (1993) log-likelihood
   statistic (G²) for a 2×2 contingency table (co-occur / verb-only / noun-only / neither),
   plus pointwise mutual information (PMI) as a magnitude-of-association complement. Verb
   candidates: content verbs with corpus count ≥ 10 (for statistical stability). Noun
   candidates: the top-500 most frequent NOUN-tagged lemmas, with both of the project's
   stopword lists applied so the table isn't dominated by "student", "course", "semester".
2. **A Word2Vec embedding cross-check** — a fresh model trained on the lemmatized corpus
   (same recipe as `2_semantic_analysis.py`), queried for each verb's nearest-neighbour
   nouns. This is a secondary signal, not a replacement: agreement between the two methods
   is a stronger basis for a claim than either alone; disagreement flags a pairing worth
   reading in context before citing.

The per-year, per-category version (`verb_category_noun_trends_analysis.py`) pools all
member verbs of a category together — a category is "present" in a sentence if *any* of its
member verbs appears — and scores each year against its **own** sentence totals rather than
a pooled 2020–2026 baseline. That's what makes "did this category's noun profile shift"
a meaningful year-over-year comparison rather than an artifact of some years having more
syllabi (and therefore more sentences) than others.

## Key findings (full corpus, 347 syllabi, 2020–2026)

**Voice.** Of ~28,400 verb instances with an identifiable subject, **19.4% are passive**.
Roughly 55% of all verb instances have no explicit subject at all (imperative
learning-objective bullets), which swamps the active/passive split in absolute terms even
though it doesn't affect the ratio between them.

**Category trends** (frequency per 1,000 words; slope = OLS trend per year):

| Category | Freq 2020 | Freq 2026 | Slope/yr |
|---|---|---|---|
| positioning | 3.36 | 5.99 | **+0.348** (rising fastest) |
| collaborating | 1.73 | 3.00 | +0.223 |
| doing | 6.78 | 8.40 | +0.153 |
| knowledge | 6.83 | 8.10 | +0.113 |
| environment_positive | 0.74 | 0.88 | +0.026 |
| built_environment | 0.55 | 0.67 | +0.005 |
| environment_negative | 0.84 | 0.80 | ~flat |
| questioning | 1.70 | 1.50 | essentially flat |

`questioning` (reconsider, deconstruct, destabilize, reframe, …) is not rising in raw
frequency — if the working hypothesis was that critical/deconstructive language is
increasing, this doesn't support that on its own. But its *noun profile* is the one that
drifts the most in substance (see below) — frequency and framing are different questions.

**Noun-profile turnover, 2020 vs. 2026** (Jaccard overlap of each category's top-10
co-occurring nouns — lower means more change):

| Category | Overlap |
|---|---|
| environment_negative | 0.05 (almost no overlap) |
| built_environment | 0.11 |
| doing | 0.11 |
| environment_positive | 0.11 |
| questioning | 0.11 |
| collaborating | 0.18 |
| knowledge | 0.18 |
| positioning | 0.18 (most stable) |

Two specific shifts worth reading in context: `doing`'s top nouns move from
*environment, design, method, climate, infrastructure* (2020) toward *environment, science,
modeling, performance, system* (2026) — a tilt toward technical/computational framing.
`questioning`'s move from *question, order, change, climate, environment* (2020) toward
*material, crisis, landscape, geography, climate* (2026) — a tilt toward explicit crisis
framing, even though the category's raw frequency didn't rise. Full year-by-year tables for
every category are in `outputs/verb_category_noun_trends_analysis.md`.

## Worked examples: what does a verb co-occur with?

Pulled directly from `outputs/verb_noun_pairs.csv` (ranked by G²):

| Verb | Category | Top co-occurring nouns |
|---|---|---|
| speculate | questioning | future, possibility, reality, material, world |
| position | positioning | climate, design, agency, leader, adaptation |
| frame | positioning | housing, problem, design, climate, question |
| model | doing | block, system, change, policy, thinking |
| represent | doing | model, contribution, modeling, drawing, benefit |

`represent` pairing most strongly with the nouns *model* and *modeling* is a nice organic
illustration of the noun/verb ambiguity spaCy is resolving per-token throughout this
step — "represent" (verb) and "model"/"modeling" (noun) show up together because students
represent something *via* a model, not because the tagger is confusing the two.

Two earlier illustrative examples, checked against the real data:
- *"analyze co-occurs with climate"* — holds up: G² = 32.3, PMI = 1.38.
- *"complicate corresponds to landscape"* — **could not be tested**: *complicate* occurs
  only 5 times corpus-wide, below the count-≥10 candidacy threshold for the co-occurrence
  table. This is a coverage gap, not a negative finding — rare critical-theory verbs (the
  exact vocabulary `questioning` is built around) are disproportionately likely to fall
  below that bar. For pairings involving specific rare verbs, direct concordance /
  keyword-in-context reading will serve better than this statistical table.

## Known limitations

- **Sentence fragments.** Syllabi are heavy on bullet points, schedules, and headers rather
  than continuous prose. spaCy's parser (trained on news/web text) is noisier here than on
  full sentences — this affects the active/passive split and sentence-level co-occurrence
  counts alike.
- **Rare verbs are invisible to the co-occurrence analyses.** Any verb below the count-≥10
  threshold (roughly a third of the 3,020 distinct lemmas) never enters
  `verb_noun_pairs.csv` or the per-year trends table.
- **PDF extraction artifacts.** A few source PDFs have a font/extraction quirk that clips
  the first letter of some words (e.g. "nclude" for "include"). These fall into `misc` and
  can be ignored — length filtering on VERB tokens catches most extraction noise but not
  this specific clipping pattern.
- **Category assignment is single-label per verb**, not per-instance. A verb like *model*
  or *frame* keeps one category regardless of which sense is active in a given sentence.
  Individual-instance disambiguation would require a different (and much more expensive)
  method than the corpus-wide lemma categorization used here.

## References / precedents

- **Dunning, T.** "Accurate Methods for the Statistics of Surprise and Coincidence" (1993)
  — the log-likelihood (G²) collocation statistic used directly in the co-occurrence
  scripts.
- **Halliday, M.A.K. & Matthiessen, C.** *An Introduction to Functional Grammar* — the
  transitivity system (who is Actor vs. Goal in a clause) behind why active/passive voice
  and verb↔noun pairing are meaningful things to measure in the first place.
- **Fairclough, N.** *Language and Power* — passivization as a device that can background
  agency ("emissions were reduced" vs. "we reduced emissions"); relevant background for the
  active/passive results.
- **Stibbe, A.** *Ecolinguistics: Language, Ecology and the Stories We Live By* (2015) —
  applies this same actor/agency lens specifically to environmental and climate texts; the
  closest direct precedent for the `environment_negative` / `environment_positive`
  categories.
- **Biber, D. & Conrad, S.** *Register, Genre, and Style* — precedent for treating
  "syllabus" as its own genre with its own administrative-verb conventions (motivating
  `stopverbs.txt`).
- Existing repo precedent: `2_semantic_analysis.py`'s per-year Word2Vec training is the
  direct model for the embedding cross-check in `verb_noun_occurrence_analysis.py`.
