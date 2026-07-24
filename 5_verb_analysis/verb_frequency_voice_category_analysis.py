"""
Verb Frequency, Voice & Category Analysis
==========================================
Reads data/syllabi_text.csv (produced by 0_preprocessing.py) and tags the
RAW `full_text` column with spaCy (en_core_web_sm) — not `filtered_text`,
since POS tagging, lemmatization, and passive-voice detection all depend on
function words and casing that step 0's stopword filter deliberately strips.
See README.md for the full rationale and design decisions.

This script:
  1. Verb lemma frequency, corpus + per-year (outputs/verbs_corpus.csv)
  2. Active vs. passive voice (outputs/active_passive.csv, _by_year.csv)
  3. Verb category counts — categories are curated by hand (or LLM judgment)
     in verb_categories.txt, not clustered algorithmically. Any verb lemma
     not listed there defaults to "misc" (outputs/verb_categories_corpus.csv)

Verb-noun co-occurrence is a separate script — see verb_noun_occurrence_analysis.py.

Usage:
  python3 verb_frequency_voice_category_analysis.py            # process all syllabi
  python3 verb_frequency_voice_category_analysis.py --limit 20 # first 20 syllabi only
"""

import argparse
import pathlib
import collections

import numpy as np
import pandas as pd
import spacy

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = pathlib.Path(__file__).parent
INPUT_CSV      = BASE_DIR.parent / "data" / "syllabi_text.csv"
STOPVERBS_FILE = BASE_DIR / "stopverbs.txt"
CATEGORIES_FILE = BASE_DIR / "verb_categories.txt"
OUT_DIR        = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_VERBS_CORPUS   = OUT_DIR / "verbs_corpus.csv"
OUT_ACTIVE_PASSIVE = OUT_DIR / "active_passive.csv"
OUT_AP_BY_YEAR     = OUT_DIR / "active_passive_by_year.csv"
OUT_CATEGORIES     = OUT_DIR / "verb_categories_corpus.csv"
OUT_CATEGORIES_TREND = OUT_DIR / "verb_categories_trend_by_year.csv"
OUT_MD             = OUT_DIR / "verb_frequency_voice_category_analysis.md"

# ── Load stopverbs ─────────────────────────────────────────────────────────────
def load_stopverbs(path: pathlib.Path) -> set:
    words = set()
    if not path.exists():
        print(f"  [WARN] stopverbs file not found: {path}")
        return words
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words

# ── Load curated verb categories ────────────────────────────────────────────────
def load_verb_categories(path: pathlib.Path) -> dict:
    """Parses verb_categories.txt: one category per line, `label TAB comma,
    separated, verb, lemmas`. Returns {lemma: category}. Any verb lemma not
    present in the returned mapping should be treated as "misc" by the caller."""
    mapping = {}
    if not path.exists():
        print(f"  [WARN] verb categories file not found: {path}")
        return mapping
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        category, _, verbs_str = line.partition("\t")
        category = category.strip()
        if not category:
            continue
        for v in verbs_str.split(","):
            v = v.strip().lower()
            if v:
                mapping[v] = category
    return mapping

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verb frequency, voice, and category analysis over syllabi full_text.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N syllabi (default: all)")
    args = parser.parse_args()

    stopverbs = load_stopverbs(STOPVERBS_FILE)
    print(f"Loaded {len(stopverbs)} stopverbs from {STOPVERBS_FILE.name}")

    verb_categories = load_verb_categories(CATEGORIES_FILE)
    category_names = sorted(set(verb_categories.values())) + ["misc"]
    print(f"Loaded {len(verb_categories)} categorized verb lemmas from {CATEGORIES_FILE.name} "
          f"across {len(category_names) - 1} categories (+ misc default)")

    print("Loading data …")
    df = pd.read_csv(INPUT_CSV, dtype={"year": "Int64"})
    df = df.dropna(subset=["full_text", "year"])
    if args.limit:
        df = df.head(args.limit)
    print(f"  {len(df)} syllabi loaded, years {df['year'].min()}–{df['year'].max()}")

    print("Loading spaCy model (en_core_web_sm) …")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # ── Tag each syllabus, collect per-token records ───────────────────────────
    print("Tagging full_text with spaCy (this can take a few minutes for the full corpus) …")
    verb_year_counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    verb_corpus_counts: collections.Counter = collections.Counter()
    surface_forms: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)

    year_token_totals: collections.Counter = collections.Counter()   # all alpha tokens, for normalisation

    ap_rows = []          # per-syllabus active/passive counts

    texts = df["full_text"].astype(str).tolist()
    meta  = list(zip(df["pdf_title"], df["year"]))

    for i, (doc, (pdf_title, year)) in enumerate(
            zip(nlp.pipe(texts, batch_size=20), meta)):
        year = int(year)

        n_active = n_passive = n_no_subject = 0

        for tok in doc:
            if tok.is_alpha:
                year_token_totals[year] += 1

            if tok.pos_ != "VERB":
                continue

            # PDF-extraction artifacts (stray bullets, hyphen-wrap fragments like
            # "re" / "frame" split from "re-frame") sometimes get mistagged as
            # VERB. A real English verb lemma is never 1-2 alphabetic characters,
            # so this threshold is a cheap, safe noise filter, not a content cut.
            if not tok.is_alpha or len(tok.lemma_) < 3:
                continue

            lemma = tok.lemma_.lower()

            # Content-verb frequency counts (§4/§5) — excludes stopverbs
            if lemma not in stopverbs:
                verb_year_counts[year][lemma] += 1
                verb_corpus_counts[lemma] += 1
                surface_forms[lemma][tok.text.lower()] += 1

            # Active/passive classification (§3) — every VERB instance, including
            # stopverbs, since e.g. "is required" is a genre feature worth keeping.
            children_deps = {c.dep_ for c in tok.children}
            if children_deps & {"nsubjpass", "auxpass", "agent"}:
                n_passive += 1
            elif "nsubj" in children_deps:
                n_active += 1
            else:
                n_no_subject += 1   # imperatives / fragments — see README.md §3 caveat

        ap_rows.append({
            "pdf_title": pdf_title, "year": year,
            "active": n_active, "passive": n_passive, "no_subject": n_no_subject,
            "total_verb_instances": n_active + n_passive + n_no_subject,
        })

        if (i + 1) % 50 == 0:
            print(f"  tagged {i + 1}/{len(texts)} syllabi …")

    print(f"Tagged {len(texts)} syllabi. {len(verb_corpus_counts)} distinct content-verb lemmas found.")

    years_sorted = sorted(verb_year_counts.keys())
    first_year, last_year = years_sorted[0], years_sorted[-1]
    year_token_series = pd.Series(dict(year_token_totals))

    # ── 1. verbs_corpus.csv — mirrors keywords_corpus.csv shape ───────────────
    print("Writing verb frequency table …")
    kw_rows = []
    for lemma, corpus_count in verb_corpus_counts.most_common():
        row = {
            "lemma": lemma,
            "surface_forms": ", ".join(w for w, _ in surface_forms[lemma].most_common()),
            "corpus_count": corpus_count,
        }
        for yr in years_sorted:
            row[str(yr)] = verb_year_counts[yr].get(lemma, 0)
        kw_rows.append(row)
    verbs_df = pd.DataFrame(kw_rows)
    verbs_df.to_csv(OUT_VERBS_CORPUS, index=False)
    print(f"  Written: {OUT_VERBS_CORPUS.name} ({len(verbs_df)} lemmas)")

    # ── 2. active_passive.csv + by-year summary ────────────────────────────────
    print("Writing active/passive tables …")
    ap_df = pd.DataFrame(ap_rows)
    ap_df.to_csv(OUT_ACTIVE_PASSIVE, index=False)

    ap_year = ap_df.groupby("year")[["active", "passive", "no_subject", "total_verb_instances"]].sum()
    ap_year["passive_pct_of_classified"] = (
        ap_year["passive"] / (ap_year["active"] + ap_year["passive"]).replace(0, pd.NA) * 100
    ).round(2)
    ap_year["no_subject_pct_of_total"] = (
        ap_year["no_subject"] / ap_year["total_verb_instances"].replace(0, pd.NA) * 100
    ).round(2)
    ap_year.to_csv(OUT_AP_BY_YEAR)
    print(f"  Written: {OUT_ACTIVE_PASSIVE.name}, {OUT_AP_BY_YEAR.name}")

    # ── 3. Verb category counts (curated categories, not clustered) ────────────
    print("Aggregating verb category counts …")

    def category_of(lemma: str) -> str:
        return verb_categories.get(lemma, "misc")

    cat_corpus_counts: collections.Counter = collections.Counter()
    cat_year_counts: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for lemma, cnt in verb_corpus_counts.items():
        cat_corpus_counts[category_of(lemma)] += cnt
    for yr, counter in verb_year_counts.items():
        for lemma, cnt in counter.items():
            cat_year_counts[yr][category_of(lemma)] += cnt

    def cat_freq(cat: str, yr: int) -> float:
        total = year_token_totals.get(yr, 0)
        return (cat_year_counts[yr].get(cat, 0) / total * 1000) if total else 0.0

    def cat_slope(cat: str) -> float:
        ys = np.array([cat_freq(cat, yr) for yr in years_sorted], dtype=float)
        xs = np.array(years_sorted, dtype=float)
        return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0

    cat_rows = []
    for cat in category_names:
        members = sorted(l for l in verb_corpus_counts if category_of(l) == cat)
        row = {
            "category": cat,
            "distinct_lemmas_observed": len(members),
            "corpus_count": cat_corpus_counts.get(cat, 0),
            "slope_freq_per_1000_per_yr": round(cat_slope(cat), 4),
            "top_members": ", ".join(
                sorted(members, key=lambda l: -verb_corpus_counts[l])[:10]
            ),
        }
        for yr in years_sorted:
            row[str(yr)] = cat_year_counts[yr].get(cat, 0)
        cat_rows.append(row)
    cat_df = pd.DataFrame(cat_rows).sort_values("corpus_count", ascending=False)
    cat_df.to_csv(OUT_CATEGORIES, index=False)
    print(f"  Written: {OUT_CATEGORIES.name} ({len(cat_df)} categories incl. misc)")

    # ── Long-format per-year trend table — the data behind the Freq/Slope
    # columns in the markdown report below, for charting (one row per
    # category x year, not just the two-endpoint summary shown in the .md) ──
    trend_rows = []
    for cat in category_names:
        for yr in years_sorted:
            trend_rows.append({
                "category": cat,
                "year": yr,
                "count": cat_year_counts[yr].get(cat, 0),
                "tokens_that_year": int(year_token_totals.get(yr, 0)),
                "freq_per_1000": round(cat_freq(cat, yr), 4),
            })
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(OUT_CATEGORIES_TREND, index=False)
    print(f"  Written: {OUT_CATEGORIES_TREND.name} ({len(trend_df)} rows, long format for charting)")

    # ── Markdown summary ────────────────────────────────────────────────────────
    print("Writing markdown report …")
    n_docs = len(df)
    total_verb_instances = int(ap_df["total_verb_instances"].sum())
    total_active  = int(ap_df["active"].sum())
    total_passive = int(ap_df["passive"].sum())
    total_nosubj  = int(ap_df["no_subject"].sum())
    classified    = total_active + total_passive
    passive_pct   = round(total_passive / classified * 100, 2) if classified else 0.0

    top20 = verb_corpus_counts.most_common(20)

    md_lines = [
        "# Verb Frequency, Voice & Category Analysis — Results",
        "",
        "Tagged with spaCy `en_core_web_sm` over `data/syllabi_text.csv`'s `full_text` "
        "column (raw, unfiltered — see README.md §0 for why).",
        "",
        "## Corpus summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Syllabi tagged | {n_docs:,} |",
        f"| Years covered | {min(years_sorted)}–{max(years_sorted)} |",
        f"| Distinct content-verb lemmas (excl. stopverbs) | {len(verb_corpus_counts):,} |",
        f"| Total VERB-tagged instances (incl. stopverbs, for active/passive) | {total_verb_instances:,} |",
        "",
        "## Active vs. passive voice",
        "",
        f"Of the {classified:,} verb instances with an explicit subject "
        f"(active *or* passive — see caveat below), **{passive_pct}%** are passive.",
        "",
        "| | Count | % of all verb instances |",
        "|---|---|---|",
        f"| Active | {total_active:,} | {total_active/total_verb_instances*100:.1f}% |",
        f"| Passive | {total_passive:,} | {total_passive/total_verb_instances*100:.1f}% |",
        f"| No explicit subject (imperative/fragment) | {total_nosubj:,} | {total_nosubj/total_verb_instances*100:.1f}% |",
        "",
        "*Syllabus prose is heavy on bullet-point learning objectives and imperatives "
        "(\"Analyze the role of…\") with no explicit subject — these can't be classified "
        "active/passive and are reported separately rather than forced into either bucket. "
        "See `outputs/active_passive_by_year.csv` for the year-by-year breakdown.*",
        "",
        "## Top 20 content verbs (corpus-wide)",
        "",
        "| Rank | Lemma | Count |",
        "|------|-------|-------|",
    ]
    for rank, (lemma, count) in enumerate(top20, 1):
        md_lines.append(f"| {rank} | {lemma} | {count:,} |")

    md_lines += [
        "",
        "## Verb categories",
        "",
        f"*(frequency per 1,000 words; slope = OLS trend per year across {first_year}–{last_year}. "
        "Categories are curated by hand in `verb_categories.txt`, not clustered — see notes below.)*",
        "",
        "| Category | Corpus count | Distinct lemmas | Freq " + str(first_year) + " | Freq " + str(last_year)
        + " | Slope/yr | Top members |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in cat_df.iterrows():
        cat = r["category"]
        f0, f1 = cat_freq(cat, first_year), cat_freq(cat, last_year)
        slope = cat_slope(cat)
        md_lines.append(
            f"| {cat} | {int(r['corpus_count']):,} | {int(r['distinct_lemmas_observed']):,} "
            f"| {f0:.2f} | {f1:.2f} | {'%+.3f' % slope} | {r['top_members']} |"
        )

    md_lines += [
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `outputs/verbs_corpus.csv` | Per-lemma verb counts, corpus + per year, plus observed surface forms |",
        "| `outputs/active_passive.csv` | Per-syllabus active/passive/no-subject counts |",
        "| `outputs/active_passive_by_year.csv` | Year-by-year active/passive/no-subject summary |",
        "| `outputs/verb_categories_corpus.csv` | Per-category verb counts, corpus + per year, plus overall slope (§5) |",
        "| `outputs/verb_categories_trend_by_year.csv` | Long-format per-year frequency table (one row per category x year) — the data behind the Freq/Slope columns below; use this for charting |",
        "",
        "## Notes",
        "",
        "- Tagged from `full_text` (raw), not `filtered_text` — POS tagging and passive-voice "
        "detection need function words that step 0's stopword filter removes.",
        "- Lemmatization collapses inflected forms to one dictionary base form *within a "
        "token's tagged part of speech* — it does not merge a word's noun and verb senses. "
        "\"building\" lemmatizes to `build` when spaCy tags that instance VERB (\"building a "
        "shelter\"), but stays `building` when tagged NOUN (\"the building faces south\") — "
        "same surface form, disambiguated per-instance from sentence context, not a static "
        "word list. Only VERB-tagged instances are counted here; the noun sense of a word "
        "like \"building\" never enters this report at all (see README.md's lemmatization "
        "note for the full explanation, and `verb_noun_occurrence_analysis.py` / "
        "`verb_category_noun_trends_analysis.py` for where NOUN-tagged tokens are counted "
        "in their own right).",
        f"- `stopverbs.txt` excludes {len(stopverbs)} generic/administrative verb lemmas from the "
        "content-verb counts (still counted in active/passive totals).",
        "- Non-alphabetic tokens and 1-2 letter lemmas are dropped from VERB-tagged tokens "
        "before any counting — a cheap filter against PDF-extraction artifacts (stray bullets, "
        "hyphen-wrap fragments) that otherwise get mistagged as verbs.",
        "- Verb categories are curated by hand in `verb_categories.txt` (label TAB comma-separated "
        "lemmas, same shape as `data/go-words.txt`) rather than produced by algorithmic clustering — "
        "any verb lemma not listed there is counted under `misc`. Edit that file and re-run to "
        "revise categories; no retraining or reclustering involved.",
        "- A handful of lemmas are first-letter-clipped words (e.g. \"nclude\" for \"include\", "
        "\"esign\" for \"design\") from a PDF font/extraction quirk in a few source files, not real "
        "verbs — length filtering catches most extraction noise but not this specific pattern; "
        "these fall into `misc` and can be ignored.",
        "- Verb-noun co-occurrence is a separate script (`verb_noun_occurrence_analysis.py`) — "
        "see `README.md`.",
    ]

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Written: {OUT_MD.name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
