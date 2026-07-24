"""
Verb Category <-> Noun Trends Analysis (per year)
===================================================
Extends verb_noun_occurrence_analysis.py along a second axis: instead of one
pooled 2020-2026 co-occurrence table per individual verb, this script pools
all verbs *within a category* (from verb_categories.txt) together and scores
co-occurrence with candidate nouns separately for each year. Answers: how do
the nouns associated with the "doing" category (build, design, write, ...)
change from 2020 to 2026 -- not just whether they do.

Method:
  - A category is "present" in a sentence if ANY of its member verb lemmas
    (per verb_categories.txt, unfiltered by frequency -- pooling absorbs
    individually-rare members) appears as a VERB-tagged token in that
    sentence. The "misc" category is excluded (not analytically interesting
    -- see README.md §5).
  - Noun candidates are the same top-500 NOUN-tagged lemmas (by corpus-wide
    frequency, project stopword lists applied) used throughout this folder,
    fixed across all years so the candidate vocabulary doesn't shift under
    the comparison.
  - For each (year, category, noun) triple, log-likelihood (G²) and PMI are
    computed from that year's OWN sentence totals (not pooled across years)
    so each year is its own independent baseline -- this is what makes "did
    the association change" a meaningful question rather than an artifact
    of one year having more sentences than another.

Outputs:
  outputs/verb_category_noun_trends.csv        Full per-year scored table
  outputs/verb_category_noun_trends_analysis.md Results report (top nouns
                                                  per category, per year)

Usage:
  python3 verb_category_noun_trends_analysis.py            # all syllabi
  python3 verb_category_noun_trends_analysis.py --limit 20 # first 20 only
"""

import argparse
import itertools
import pathlib
import collections

import numpy as np
import pandas as pd
import spacy

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = pathlib.Path(__file__).parent
INPUT_CSV       = BASE_DIR.parent / "data" / "syllabi_text.csv"
CATEGORIES_FILE = BASE_DIR / "verb_categories.txt"
STOPWORD_FILES  = [
    BASE_DIR.parent / "0_preprocessing" / "stopwords" / "stopwords.txt",
    BASE_DIR.parent / "0_preprocessing" / "stopwords" / "stopwords_2.txt",
]
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_TRENDS = OUT_DIR / "verb_category_noun_trends.csv"
OUT_MD     = OUT_DIR / "verb_category_noun_trends_analysis.md"

# ── Config ─────────────────────────────────────────────────────────────────────
TOP_NOUNS_N    = 500  # candidate noun vocabulary size, fixed across all years
MIN_COOCCUR    = 3    # minimum co-occurring sentences within a year before scoring
TOP_NOUNS_SHOWN = 5   # nouns shown per category per year in the report

# ── Loaders (shared shape with verb_noun_occurrence_analysis.py) ──────────────
def load_wordlist(path: pathlib.Path) -> set:
    words = set()
    if not path.exists():
        print(f"  [WARN] wordlist not found: {path}")
        return words
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words


def load_verb_categories(path: pathlib.Path) -> dict:
    """Returns {category: set(verb_lemmas)}, excluding "misc" (there is no
    explicit misc list -- it's the default for unlisted lemmas elsewhere)."""
    cats: dict = collections.defaultdict(set)
    if not path.exists():
        return cats
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
                cats[category].add(v)
    return cats


def collocation_stats(a: int, b: int, c: int, d: int) -> tuple:
    """a = sentences with both, b = category only, c = noun only, d = neither."""
    n = a + b + c + d
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    if n == 0 or row1 == 0 or col1 == 0:
        return 0.0, 0.0
    e_a = row1 * col1 / n
    e_b = row1 * col2 / n
    e_c = row2 * col1 / n
    e_d = row2 * col2 / n
    g2 = 0.0
    for obs, exp in ((a, e_a), (b, e_b), (c, e_c), (d, e_d)):
        if obs > 0 and exp > 0:
            g2 += 2 * obs * np.log(obs / exp)
    p_a, p_v, p_n = a / n, row1 / n, col1 / n
    pmi = float(np.log2(p_a / (p_v * p_n))) if p_a > 0 and p_v > 0 and p_n > 0 else float("-inf")
    return float(g2), pmi


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-year verb-category <-> noun co-occurrence trends.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N syllabi (default: all)")
    args = parser.parse_args()

    stopwords = set()
    for f in STOPWORD_FILES:
        stopwords |= load_wordlist(f)
    cat_verbs = load_verb_categories(CATEGORIES_FILE)
    category_names = sorted(cat_verbs.keys())
    lemma_to_cats: dict = collections.defaultdict(set)
    for cat, verbs in cat_verbs.items():
        for v in verbs:
            lemma_to_cats[v].add(cat)
    print(f"Loaded {len(category_names)} non-misc categories covering "
          f"{len(lemma_to_cats)} verb lemmas, {len(stopwords)} stopwords")

    print("Loading data …")
    df = pd.read_csv(INPUT_CSV, dtype={"year": "Int64"})
    df = df.dropna(subset=["full_text", "year"])
    if args.limit:
        df = df.head(args.limit)
    print(f"  {len(df)} syllabi loaded, years {df['year'].min()}–{df['year'].max()}")

    print("Loading spaCy model (en_core_web_sm) …")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    texts = df["full_text"].astype(str).tolist()
    years_col = df["year"].astype(int).tolist()

    # ── Stage 1: corpus-wide noun lemma frequency (fixed candidate vocabulary) ─
    print("Tagging full_text (stage 1/2 — building noun candidate vocabulary) …")
    noun_counts: collections.Counter = collections.Counter()
    docs_cache = list(nlp.pipe(texts, batch_size=20))
    for i, doc in enumerate(docs_cache):
        for tok in doc:
            if not tok.is_alpha or len(tok.lemma_) < 3:
                continue
            if tok.pos_ == "NOUN":
                lemma = tok.lemma_.lower()
                if lemma not in stopwords:
                    noun_counts[lemma] += 1
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(texts)} …")

    noun_vocab = {lemma for lemma, _ in noun_counts.most_common(TOP_NOUNS_N)}
    print(f"Candidate nouns (top {TOP_NOUNS_N} by corpus frequency, stopwords excluded): "
          f"{len(noun_vocab)}")

    # ── Stage 2: per-year sentence-level co-occurrence, category-pooled ────────
    print("Scanning sentences for category<->noun co-occurrence, per year …")
    pair_cooccur: dict = collections.defaultdict(collections.Counter)   # year -> Counter[(cat,noun)]
    cat_sent_count: dict = collections.defaultdict(collections.Counter) # year -> Counter[cat]
    noun_sent_count: dict = collections.defaultdict(collections.Counter) # year -> Counter[noun]
    total_sentences: collections.Counter = collections.Counter()         # year -> int

    for doc, year in zip(docs_cache, years_col):
        for sent in doc.sents:
            total_sentences[year] += 1
            cats_here, nouns_here = set(), set()
            for tok in sent:
                if not tok.is_alpha or len(tok.lemma_) < 3:
                    continue
                lemma = tok.lemma_.lower()
                if tok.pos_ == "VERB" and lemma in lemma_to_cats:
                    cats_here |= lemma_to_cats[lemma]
                elif tok.pos_ == "NOUN" and lemma in noun_vocab:
                    nouns_here.add(lemma)
            for c in cats_here:
                cat_sent_count[year][c] += 1
            for n in nouns_here:
                noun_sent_count[year][n] += 1
            for c, n in itertools.product(cats_here, nouns_here):
                pair_cooccur[year][(c, n)] += 1

    years_sorted = sorted(total_sentences.keys())
    print(f"  Years: {years_sorted}, total sentences: {sum(total_sentences.values()):,}")

    # ── Score pairs, per year ───────────────────────────────────────────────────
    print("Scoring (year, category, noun) triples …")
    rows = []
    for year in years_sorted:
        n_total = total_sentences[year]
        for (cat, noun), a in pair_cooccur[year].items():
            if a < MIN_COOCCUR:
                continue
            b = cat_sent_count[year][cat] - a
            c = noun_sent_count[year][noun] - a
            d = n_total - cat_sent_count[year][cat] - noun_sent_count[year][noun] + a
            g2, pmi = collocation_stats(a, b, c, d)
            rows.append({
                "year": year, "category": cat, "noun": noun,
                "co_occur_sentences": a, "category_sentences": cat_sent_count[year][cat],
                "noun_sentences": noun_sent_count[year][noun], "total_sentences_year": n_total,
                "pmi": round(pmi, 3), "G2": round(g2, 2),
            })
    trends_df = pd.DataFrame(rows).sort_values(["category", "year", "G2"], ascending=[True, True, False])
    trends_df.to_csv(OUT_TRENDS, index=False)
    print(f"  Written: {OUT_TRENDS.name} ({len(trends_df):,} scored (year, category, noun) rows)")

    # ── Markdown report ──────────────────────────────────────────────────────────
    print("Writing markdown report …")
    first_year, last_year = years_sorted[0], years_sorted[-1]

    md_lines = [
        "# Verb Category <-> Noun Trends Analysis — Results",
        "",
        "How does the noun profile associated with each verb *category* (not "
        "individual verb) shift year by year? Method: `README.md` §6, extended along "
        "a per-year axis — see this script's module docstring for the full method.",
        "",
        "## Corpus summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Years | {first_year}–{last_year} |",
        f"| Non-misc categories | {len(category_names)} |",
        f"| Candidate nouns (top {TOP_NOUNS_N}, stopwords excluded) | {len(noun_vocab):,} |",
        f"| Total sentences scanned | {sum(total_sentences.values()):,} |",
        "",
        "## Top nouns per category, by year",
        "",
        f"*(top {TOP_NOUNS_SHOWN} nouns by within-year G² log-likelihood; "
        f"pairs below {MIN_COOCCUR} co-occurring sentences in a given year are excluded, "
        "which is why some year cells show fewer than "
        f"{TOP_NOUNS_SHOWN} or read \"—\")*",
        "",
    ]

    turnover_rows = []
    for cat in category_names:
        cat_df = trends_df[trends_df["category"] == cat]
        md_lines += [
            f"### {cat}",
            "",
            "| Year | Top nouns (by G²) |",
            "|---|---|",
        ]
        top_sets = {}
        for year in years_sorted:
            yr_df = cat_df[cat_df["year"] == year].sort_values("G2", ascending=False)
            top_n = yr_df["noun"].head(TOP_NOUNS_SHOWN).tolist()
            top_sets[year] = set(yr_df["noun"].head(10).tolist())
            md_lines.append(f"| {year} | {', '.join(top_n) if top_n else '—'} |")
        overlap = jaccard(top_sets.get(first_year, set()), top_sets.get(last_year, set()))
        turnover_rows.append({"category": cat, "jaccard_top10_first_vs_last": round(overlap, 2)})
        md_lines.append("")

    md_lines += [
        "## Noun-profile turnover, first year vs. last year",
        "",
        f"*(Jaccard overlap of each category's top-10 co-occurring nouns, {first_year} vs. "
        f"{last_year} — 1.0 = identical top-10 sets, 0.0 = no overlap at all. Lower means "
        "the category's associated vocabulary shifted more.)*",
        "",
        "| Category | Jaccard overlap (top-10, first vs. last year) |",
        "|---|---|",
    ]
    for r in sorted(turnover_rows, key=lambda r: r["jaccard_top10_first_vs_last"]):
        md_lines.append(f"| {r['category']} | {r['jaccard_top10_first_vs_last']:.2f} |")

    md_lines += [
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `outputs/verb_category_noun_trends.csv` | Full per-year (category, noun) G²/PMI table |",
        "",
        "## Notes",
        "",
        "- \"misc\" is excluded — it's the leftover bucket for generic/administrative verbs, "
        "not an analytically meaningful category to trend.",
        "- Each year's G²/PMI is computed from that year's OWN sentence totals, not pooled "
        "across years — this is what makes the year-to-year comparison meaningful rather "
        "than an artifact of corpus size differences between years (see `0_preprocessing/`'s "
        "corpus summary for per-year syllabus counts).",
        "- A category is \"present\" in a sentence if ANY of its member verbs appears — "
        "individual member verbs are not weighted or distinguished within a category here; "
        "see `verb_noun_occurrence_analysis.py` for the individual-verb-level version.",
        "- Same fragment/bullet-point sentence-segmentation caveat as the rest of this "
        "folder applies.",
    ]

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Written: {OUT_MD.name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
