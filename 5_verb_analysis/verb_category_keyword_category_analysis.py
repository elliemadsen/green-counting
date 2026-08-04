"""
Verb Category <-> Keyword Category Analysis
=============================================
Tests a specific hypothesis: do syllabi that lean on a particular *topic*
vocabulary (quantitative/technical, theoretical, economic, ...) also lean on
a particular *verb* vocabulary (questioning, positioning, doing, ...)? E.g.
"quantitative-methods syllabi skew toward questioning verbs; theoretical
syllabi skew toward positioning verbs."

This crosses two taxonomies that already exist in the repo but had never
been joined:
  - verb_categories.txt        (this folder)    -- 8 hand-curated verb-lemma categories
  - data/Categories.csv        (project root)   -- keyword-lemma topic categories
                                                    (theory, quantification, material,
                                                    architecture, economic, land, solution,
                                                    problem, representation, universal,
                                                    local); a "verb" tag in that file is a
                                                    part-of-speech marker, not a topic, and
                                                    is excluded here -- see README.md.

Method: same sentence-level co-occurrence + log-likelihood (G²) / PMI approach
as verb_noun_occurrence_analysis.py (README.md §6), except the noun side is
pooled into Categories.csv's topic categories instead of scored per individual
word, so the output is a compact verb-category x keyword-category contingency
table rather than thousands of individual pairs -- small enough to report and
read in full, unlike the noun-pair tables in this folder's other scripts.

A sentence counts toward a keyword-category if any of its non-VERB-tagged
tokens (NOUN, ADJ, PROPN, ...) matches a Categories.csv word or alias, by
lemma or by raw surface form (Categories.csv mixes both conventions, e.g.
"economic" is listed as an alias of the noun "economy" despite spaCy
lemmatizing the adjective "economic" to itself -- matching on both sides
catches this). It counts toward a verb-category exactly as elsewhere in this
folder: any of the category's member verb lemmas appears as a VERB-tagged
token in that sentence.

Outputs:
  outputs/verb_category_keyword_category.csv          Full scored contingency table
  outputs/verb_category_keyword_category_analysis.md  Results report
  outputs/verb_keyword_pmi_heatmap.png                 Heatmap: distinctiveness (PMI)
  outputs/verb_keyword_count_heatmap.png               Heatmap: evidence weight (raw counts)

Usage:
  python3 verb_category_keyword_category_analysis.py            # all syllabi
  python3 verb_category_keyword_category_analysis.py --limit 20 # first 20 only
"""

import argparse
import csv
import itertools
import pathlib
import collections

import numpy as np
import pandas as pd
import spacy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = pathlib.Path(__file__).parent
INPUT_CSV        = BASE_DIR.parent / "data" / "syllabi_text.csv"
VERB_CATS_FILE   = BASE_DIR / "verb_categories.txt"
KEYWORD_CATS_CSV = BASE_DIR.parent / "data" / "Categories.csv"
OUT_DIR          = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_TABLE      = OUT_DIR / "verb_category_keyword_category.csv"
OUT_MD         = OUT_DIR / "verb_category_keyword_category_analysis.md"
CHART_PMI      = OUT_DIR / "verb_keyword_pmi_heatmap.png"
CHART_COUNT    = OUT_DIR / "verb_keyword_count_heatmap.png"

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_COOCCUR   = 3   # minimum co-occurring sentences before a cell is scored
POS_TAG_LABEL = "verb"  # Categories.csv tag meaning "this word is a verb", not a topic

# Verb-category row order: matches the frequency-trends table in README.md
# (rising fastest -> flat), so the chart reads consistently with the rest of the report.
VERB_CAT_DISPLAY_ORDER = [
    "positioning", "collaborating", "doing", "knowledge",
    "environment_positive", "built_environment", "environment_negative", "questioning",
]

# ── Loaders ────────────────────────────────────────────────────────────────────
def load_verb_categories(path: pathlib.Path) -> dict:
    """{verb_lemma: category}, mirroring verb_frequency_voice_category_analysis.py."""
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        category, _, verbs_str = line.partition("\t")
        for v in verbs_str.split(","):
            v = v.strip().lower()
            if v:
                mapping[v] = category.strip()
    return mapping


def load_keyword_categories(path: pathlib.Path) -> dict:
    """Returns {surface_form: {category, ...}} from data/Categories.csv, keyed by
    both the canonical Word and every Aliases entry, lowercased. The "verb"
    tag is dropped (part-of-speech marker, not a topic); rows left with no
    topic category after that (blank-category rows, and words tagged only
    "verb") contribute nothing and are skipped."""
    lookup: dict = collections.defaultdict(set)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cats = {c.strip() for c in row[""].split(",") if c.strip()}
            cats.discard(POS_TAG_LABEL)
            if not cats:
                continue
            forms = [row["Word"].strip().lower()]
            forms += [a.strip().lower() for a in row["Aliases"].split(",") if a.strip()]
            for form in forms:
                if form:
                    lookup[form] |= cats
    return lookup


def keyword_category_word_counts(path: pathlib.Path) -> collections.Counter:
    """Distinct Categories.csv *words* per topic category (not surface forms/aliases) --
    used only to order the heatmap columns by vocabulary size, largest first, mirroring
    how VERB_CAT_DISPLAY_ORDER orders rows by an established, meaningful sequence rather
    than alphabetically."""
    counts: collections.Counter = collections.Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cats = {c.strip() for c in row[""].split(",") if c.strip()}
            cats.discard(POS_TAG_LABEL)
            for c in cats:
                counts[c] += 1
    return counts


def collocation_stats(a: int, b: int, c: int, d: int) -> tuple:
    """a = sentences with both, b = verb-cat only, c = keyword-cat only, d = neither."""
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


def p_label(g2: float) -> str:
    if g2 >= 15.13: return "p < 0.0001"
    if g2 >= 10.83: return "p < 0.001"
    if g2 >= 6.63:  return "p < 0.01"
    if g2 >= 3.84:  return "p < 0.05"
    return "not significant"


# ── Charts ─────────────────────────────────────────────────────────────────────
def render_pmi_heatmap(table_df: pd.DataFrame, verb_order: list, kw_order: list, outpath: pathlib.Path):
    """Diverging heatmap (not the project's usual sequential YlOrRd): PMI has a real
    zero-point -- independence, i.e. no association either way -- so red/blue centered
    on white at 0 is the correct encoding, not a magnitude ramp. This is what answers
    "what's distinctive to what," the same reading as the report's PMI-ranked tables."""
    pivot = table_df.pivot_table(index="verb_category", columns="keyword_category", values="pmi")
    pivot = pivot.reindex(index=verb_order, columns=kw_order)
    vmax = float(np.nanmax(np.abs(pivot.values))) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(max(8, len(kw_order) * 0.9), max(5, len(verb_order) * 0.6)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(kw_order)))
    ax.set_xticklabels(kw_order, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(verb_order)))
    ax.set_yticklabels(verb_order, fontsize=10)
    ax.set_xlabel("Nouns + Adjectives", fontsize=11)
    ax.set_ylabel("Verbs", fontsize=11)
    for yi in range(len(verb_order)):
        for xi in range(len(kw_order)):
            val = pivot.values[yi, xi]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(xi, yi, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label="PMI  (0 = independence · + = co-occur more than chance)")
    ax.set_title("Verb category × keyword-topic category — distinctiveness (PMI)", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def render_count_heatmap(table_df: pd.DataFrame, verb_order: list, kw_order: list, outpath: pathlib.Path):
    """Sequential YlOrRd, matching this project's existing heatmap convention (see
    1_keyword_analysis.py) -- raw counts are a true unsigned magnitude, unlike PMI.
    Reads alongside the PMI heatmap as the evidence-weight check: a high-PMI cell
    backed by only a handful of sentences is a thinner claim than one backed by
    hundreds, even though PMI alone doesn't show that."""
    pivot = table_df.pivot_table(index="verb_category", columns="keyword_category",
                                  values="co_occur_sentences")
    pivot = pivot.reindex(index=verb_order, columns=kw_order)
    vmax = float(np.nanmax(pivot.values)) or 1.0

    fig, ax = plt.subplots(figsize=(max(8, len(kw_order) * 0.9), max(5, len(verb_order) * 0.6)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(kw_order)))
    ax.set_xticklabels(kw_order, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(len(verb_order)))
    ax.set_yticklabels(verb_order, fontsize=10)
    ax.set_xlabel("Nouns", fontsize=11)
    ax.set_ylabel("Verbs", fontsize=11)
    for yi in range(len(verb_order)):
        for xi in range(len(kw_order)):
            val = pivot.values[yi, xi]
            if np.isnan(val):
                continue
            color = "white" if val > vmax * 0.6 else "black"
            ax.text(xi, yi, f"{int(val)}", ha="center", va="center", fontsize=8, color=color)
    plt.colorbar(im, ax=ax, label="Co-occurring sentences")
    ax.set_title("Verb category × keyword-topic category — co-occurring sentences", fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verb-category <-> keyword-category (Categories.csv) co-occurrence.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N syllabi (default: all)")
    args = parser.parse_args()

    verb_cat_of = load_verb_categories(VERB_CATS_FILE)
    lemma_to_verbcats: dict = collections.defaultdict(set)
    for lemma, cat in verb_cat_of.items():
        lemma_to_verbcats[lemma].add(cat)
    verb_cat_names = sorted(set(verb_cat_of.values()))

    kw_lookup = load_keyword_categories(KEYWORD_CATS_CSV)
    kw_cat_names = sorted({c for cats in kw_lookup.values() for c in cats})
    print(f"Loaded {len(verb_cat_names)} verb categories ({len(lemma_to_verbcats)} lemmas), "
          f"{len(kw_cat_names)} keyword categories ({len(kw_lookup)} surface forms) "
          f"from {KEYWORD_CATS_CSV.relative_to(BASE_DIR.parent)}")

    print("Loading data …")
    df = pd.read_csv(INPUT_CSV, dtype={"year": "Int64"})
    df = df.dropna(subset=["full_text", "year"])
    if args.limit:
        df = df.head(args.limit)
    print(f"  {len(df)} syllabi loaded")

    print("Loading spaCy model (en_core_web_sm) …")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    texts = df["full_text"].astype(str).tolist()
    docs_cache = list(nlp.pipe(texts, batch_size=20))

    # ── Sentence-level co-occurrence ────────────────────────────────────────────
    print("Scanning sentences for verb-category <-> keyword-category co-occurrence …")
    pair_cooccur: collections.Counter = collections.Counter()
    verbcat_sent_count: collections.Counter = collections.Counter()
    kwcat_sent_count: collections.Counter = collections.Counter()
    total_sentences = 0

    for i, doc in enumerate(docs_cache):
        for sent in doc.sents:
            total_sentences += 1
            vcats_here, kcats_here = set(), set()
            for tok in sent:
                if not tok.is_alpha or len(tok.lemma_) < 3:
                    continue
                lemma = tok.lemma_.lower()
                if tok.pos_ == "VERB":
                    vcats_here |= lemma_to_verbcats.get(lemma, set())
                else:
                    kcats_here_hit = kw_lookup.get(lemma) or kw_lookup.get(tok.text.lower())
                    if kcats_here_hit:
                        kcats_here |= kcats_here_hit
            for v in vcats_here:
                verbcat_sent_count[v] += 1
            for k in kcats_here:
                kwcat_sent_count[k] += 1
            for v, k in itertools.product(vcats_here, kcats_here):
                pair_cooccur[(v, k)] += 1
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(docs_cache)} syllabi …")

    print(f"  {total_sentences:,} sentences scanned")
    print(f"  Sentences with a categorized verb: {sum(verbcat_sent_count.values()):,} (category instances, sentences may count toward >1)")
    print(f"  Sentences with a categorized keyword: {sum(kwcat_sent_count.values()):,} (category instances, sentences may count toward >1)")

    # ── Score every (verb_category, keyword_category) cell ─────────────────────
    print("Scoring contingency table …")
    rows = []
    for v, k in itertools.product(verb_cat_names, kw_cat_names):
        a = pair_cooccur.get((v, k), 0)
        b = verbcat_sent_count[v] - a
        c = kwcat_sent_count[k] - a
        d = total_sentences - verbcat_sent_count[v] - kwcat_sent_count[k] + a
        g2, pmi = collocation_stats(a, b, c, d)
        rows.append({
            "verb_category": v, "keyword_category": k,
            "co_occur_sentences": a, "verb_category_sentences": verbcat_sent_count[v],
            "keyword_category_sentences": kwcat_sent_count[k], "total_sentences": total_sentences,
            "pct_of_verb_category_sentences": round(100 * a / verbcat_sent_count[v], 1) if verbcat_sent_count[v] else 0.0,
            "pct_of_keyword_category_sentences": round(100 * a / kwcat_sent_count[k], 1) if kwcat_sent_count[k] else 0.0,
            "pmi": round(pmi, 3), "G2": round(g2, 2), "p_label": p_label(g2),
            "scored": a >= MIN_COOCCUR,
        })
    table_df = pd.DataFrame(rows).sort_values("G2", ascending=False)
    table_df.to_csv(OUT_TABLE, index=False)
    print(f"  Written: {OUT_TABLE.name} ({len(table_df)} cells, "
          f"{int(table_df['scored'].sum())} with >= {MIN_COOCCUR} co-occurring sentences)")

    # ── Charts ───────────────────────────────────────────────────────────────────
    print("Rendering heatmaps …")
    verb_order = [v for v in VERB_CAT_DISPLAY_ORDER if v in verb_cat_names]
    verb_order += [v for v in verb_cat_names if v not in verb_order]  # any not in the fixed list
    kw_word_counts = keyword_category_word_counts(KEYWORD_CATS_CSV)
    kw_order = [k for k, _ in kw_word_counts.most_common() if k in kw_cat_names]
    kw_order += [k for k in kw_cat_names if k not in kw_order]

    render_pmi_heatmap(table_df, verb_order, kw_order, CHART_PMI)
    print(f"  Written: {CHART_PMI.name}")
    render_count_heatmap(table_df, verb_order, kw_order, CHART_COUNT)
    print(f"  Written: {CHART_COUNT.name}")

    # ── Markdown report ──────────────────────────────────────────────────────────
    print("Writing markdown report …")
    scored_df = table_df[table_df["scored"]].copy()

    md_lines = [
        "# Verb Category <-> Keyword Category Analysis — Results",
        "",
        "Does a syllabus's *topic* vocabulary (per `data/Categories.csv`) predict which "
        "*verb* vocabulary (per `verb_categories.txt`) it leans on? Method: sentence-level "
        "co-occurrence scored with log-likelihood (G²) and PMI — same approach as "
        "`verb_noun_occurrence_analysis.py` (README.md §6), with the keyword side pooled "
        "into topic categories instead of scored per individual word. See this script's "
        "module docstring for the full method.",
        "",
        "## Corpus summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Sentences scanned | {total_sentences:,} |",
        f"| Verb categories | {len(verb_cat_names)} ({', '.join(verb_cat_names)}) |",
        f"| Keyword categories | {len(kw_cat_names)} ({', '.join(kw_cat_names)}) |",
        f"| Contingency cells | {len(table_df)} ({int(table_df['scored'].sum())} with "
        f">= {MIN_COOCCUR} co-occurring sentences, scored below) |",
        "",
        "## Full contingency table",
        "",
        "*(ranked by G² log-likelihood — how statistically surprising the pairing is, not "
        "just how frequent; `% of verb-cat` reads as \"of sentences using this verb "
        "category, what share also touch this keyword category\" and vice versa for "
        "`% of keyword-cat`)*",
        "",
        "| Verb category | Keyword category | Co-occur (sentences) | % of verb-cat | "
        "% of keyword-cat | G² | p | PMI |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in scored_df.iterrows():
        md_lines.append(
            f"| {r['verb_category']} | {r['keyword_category']} | {int(r['co_occur_sentences'])} "
            f"| {r['pct_of_verb_category_sentences']:.1f}% | {r['pct_of_keyword_category_sentences']:.1f}% "
            f"| {r['G2']:.1f} | {r['p_label']} | {r['pmi']:.2f} |"
        )

    md_lines += [
        "",
        "## Which verb category is *distinctive* to each keyword category",
        "",
        "*(top 2 by PMI, not raw co-occurrence % — raw % mostly just reflects which verb "
        "category is common everywhere: `doing` and `knowledge` are corpus-wide the two "
        "most frequent categories by a wide margin (see README.md's category-trends "
        "table), so ranking by raw co-occurrence share surfaces them almost everywhere and "
        "says little about any given topic specifically. PMI instead measures how far above "
        "chance the pairing is, after accounting for how common each category already is on "
        "its own — that's the reading that actually answers \"what verb vocabulary is this "
        "topic's own,\" as opposed to \"what verb vocabulary is common in general.\")*",
        "",
        "| Keyword category | 1st by PMI | 2nd by PMI |",
        "|---|---|---|",
    ]
    for k in kw_cat_names:
        sub = scored_df[scored_df["keyword_category"] == k].sort_values("pmi", ascending=False)
        top2 = sub.head(2)
        cells = [f"{r['verb_category']} (PMI {r['pmi']:.2f})" for _, r in top2.iterrows()]
        while len(cells) < 2:
            cells.append("—")
        md_lines.append(f"| {k} | {cells[0]} | {cells[1]} |")

    md_lines += [
        "",
        "## Which keyword category is *distinctive* to each verb category",
        "",
        "*(same PMI-based reading, the other direction)*",
        "",
        "| Verb category | 1st by PMI | 2nd by PMI |",
        "|---|---|---|",
    ]
    for v in verb_cat_names:
        sub = scored_df[scored_df["verb_category"] == v].sort_values("pmi", ascending=False)
        top2 = sub.head(2)
        cells = [f"{r['keyword_category']} (PMI {r['pmi']:.2f})" for _, r in top2.iterrows()]
        while len(cells) < 2:
            cells.append("—")
        md_lines.append(f"| {v} | {cells[0]} | {cells[1]} |")

    md_lines += [
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `outputs/verb_category_keyword_category.csv` | Full scored contingency table "
        "(all cells, including those below the co-occurrence floor) |",
        f"| `outputs/{CHART_PMI.name}` | Heatmap: distinctiveness (PMI, diverging color scale "
        "centered on independence) |",
        f"| `outputs/{CHART_COUNT.name}` | Heatmap: evidence weight (raw co-occurring "
        "sentence counts) |",
        "",
        "## Notes",
        "",
        f"- `data/Categories.csv` has {len(kw_lookup):,} surface forms across "
        f"{len(kw_cat_names)} topic categories — much sparser than the top-500-word noun "
        "vocabulary used elsewhere in this folder, since that file was hand-picked for a "
        "different, smaller purpose and had not previously been used in any analysis "
        "script. Treat coverage, not just the associations themselves, as a finding: a low "
        "keyword-category sentence count means most of the corpus's topic vocabulary isn't "
        "captured by this file yet, not that the topic is rare. Extending "
        "`data/Categories.csv` the way `verb_categories.txt` was extended (README.md §5 — "
        "a handful of seed words per category, grown by LLM semantic judgment) is a natural "
        "next step if these results look promising but thin.",
        "- Categories.csv's `verb` tag marks a word's part of speech, not a topic — it's "
        "dropped here, not treated as a ninth keyword category.",
        "- A sentence counts toward a keyword category if any of its non-VERB-tagged tokens "
        "matches a Categories.csv word or alias, by lemma or by raw surface form (both are "
        "checked because the source file itself mixes conventions — see module docstring).",
        "- Keyword categories are multi-label (a word like \"economy\" carries `economic, "
        "universal, quantification` simultaneously, per the source file); verb categories "
        "remain single-label per verb, per `verb_categories.txt`'s existing design.",
        "- Same sentence-fragment/bullet-point caveat as the rest of this folder: syllabus "
        "prose that isn't a complete sentence may segment inconsistently.",
        f"- Cells below {MIN_COOCCUR} co-occurring sentences are excluded from the report "
        "tables above (still present, flagged `scored=False`, in the CSV) — too sparse for "
        "G²/PMI to be meaningful.",
    ]

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Written: {OUT_MD.name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
