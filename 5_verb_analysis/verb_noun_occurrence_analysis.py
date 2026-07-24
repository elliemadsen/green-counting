"""
Verb <-> Noun Occurrence Analysis
==================================
Follow-up to verb_frequency_voice_category_analysis.py, run separately once
that script's verb list had been reviewed. Answers: which nouns does a given
verb tend to appear alongside ("analyze co-occurs with climate", "complicate
corresponds to landscape")? See README.md §6 for the design rationale.

Two independent methods, per §6's decision to do both:
  1. Sentence-level co-occurrence, scored with log-likelihood (G²) and PMI —
     the primary, citable number. Verb candidates: content verbs with
     corpus_count >= MIN_VERB_COUNT (see verbs_corpus.csv). Noun candidates:
     the top-500 most frequent NOUN-tagged lemmas, with the project's
     existing academic/institutional stopword lists applied so the table
     isn't dominated by "student", "course", "semester", etc.
  2. A fresh Word2Vec embedding trained on the lemmatized corpus (same
     recipe used for the clustering approach that verb category assignment
     ultimately superseded — see README.md §5), queried for each verb's
     nearest-neighbour nouns — a secondary cross-check, not a replacement
     for the co-occurrence numbers.

For the per-year, per-category version of this analysis (how does the
"doing" category's noun profile shift 2020 -> 2026?), see
verb_category_noun_trends_analysis.py.

Outputs:
  outputs/verb_noun_pairs.csv                    Full co-occurrence + G²/PMI table
  outputs/verb_noun_occurrence_analysis.md       Results report

Usage:
  python3 verb_noun_occurrence_analysis.py            # process all syllabi
  python3 verb_noun_occurrence_analysis.py --limit 20 # first 20 syllabi only
"""

import argparse
import itertools
import pathlib
import collections

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = pathlib.Path(__file__).parent
INPUT_CSV       = BASE_DIR.parent / "data" / "syllabi_text.csv"
STOPVERBS_FILE  = BASE_DIR / "stopverbs.txt"
CATEGORIES_FILE = BASE_DIR / "verb_categories.txt"
STOPWORD_FILES  = [
    BASE_DIR.parent / "0_preprocessing" / "stopwords" / "stopwords.txt",
    BASE_DIR.parent / "0_preprocessing" / "stopwords" / "stopwords_2.txt",
]
OUT_DIR         = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_PAIRS = OUT_DIR / "verb_noun_pairs.csv"
OUT_MD    = OUT_DIR / "verb_noun_occurrence_analysis.md"

# ── Config ─────────────────────────────────────────────────────────────────────
MIN_VERB_COUNT   = 10   # candidate verbs must appear at least this often corpus-wide
TOP_NOUNS_N      = 500  # candidate noun vocabulary size (mirrors 2_semantic_analysis.py's top-500 restriction)
MIN_COOCCUR      = 3    # minimum co-occurring sentences before a pair is scored/reported
TOP_PAIRS_N      = 30   # pairs shown in the corpus-wide top table
TOP_PARTNERS_N   = 8    # noun partners shown per verb in per-category tables
EMBED_CHECK_VERBS_N = 3 # verbs per category sampled for the embedding cross-check

W2V_PARAMS = dict(
    vector_size=100, window=8, min_count=3,
    workers=4, epochs=40, sg=1, seed=42,
)  # same recipe as 2_semantic_analysis.py / the superseded clustering approach (README.md §5)

# ── Loaders ────────────────────────────────────────────────────────────────────
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
    mapping = {}
    if not path.exists():
        return mapping
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


# ── Log-likelihood (Dunning G²) + PMI for a 2x2 sentence co-occurrence table ───
def collocation_stats(a: int, b: int, c: int, d: int) -> tuple:
    """a = sentences with both, b = verb only, c = noun only, d = neither."""
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


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Verb<->noun co-occurrence analysis over syllabi full_text.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N syllabi (default: all)")
    args = parser.parse_args()

    stopverbs = load_wordlist(STOPVERBS_FILE)
    stopwords = set()
    for f in STOPWORD_FILES:
        stopwords |= load_wordlist(f)
    verb_categories = load_verb_categories(CATEGORIES_FILE)
    print(f"Loaded {len(stopverbs)} stopverbs, {len(stopwords)} stopwords, "
          f"{len(verb_categories)} categorized verb lemmas")

    print("Loading data …")
    df = pd.read_csv(INPUT_CSV, dtype={"year": "Int64"})
    df = df.dropna(subset=["full_text", "year"])
    if args.limit:
        df = df.head(args.limit)
    print(f"  {len(df)} syllabi loaded")

    print("Loading spaCy model (en_core_web_sm) …")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # ── Stage 1: corpus-wide verb & noun lemma frequency (to pick candidates) ──
    print("Tagging full_text (stage 1/2 — building candidate vocabularies) …")
    verb_counts: collections.Counter = collections.Counter()
    noun_counts: collections.Counter = collections.Counter()
    texts = df["full_text"].astype(str).tolist()

    docs_cache = list(nlp.pipe(texts, batch_size=20))
    for i, doc in enumerate(docs_cache):
        for tok in doc:
            if not tok.is_alpha or len(tok.lemma_) < 3:
                continue
            lemma = tok.lemma_.lower()
            if tok.pos_ == "VERB" and lemma not in stopverbs:
                verb_counts[lemma] += 1
            elif tok.pos_ in ("NOUN",) and lemma not in stopwords:
                noun_counts[lemma] += 1
        if (i + 1) % 100 == 0:
            print(f"  scanned {i + 1}/{len(texts)} …")

    verb_vocab = {lemma for lemma, cnt in verb_counts.items() if cnt >= MIN_VERB_COUNT}
    noun_vocab = {lemma for lemma, _ in noun_counts.most_common(TOP_NOUNS_N)}
    print(f"Candidate verbs (count >= {MIN_VERB_COUNT}): {len(verb_vocab)}")
    print(f"Candidate nouns (top {TOP_NOUNS_N} by corpus frequency, stopwords excluded): {len(noun_vocab)}")

    # ── Stage 2: sentence-level co-occurrence over candidate vocab ────────────
    print("Scanning sentences for verb<->noun co-occurrence …")
    pair_cooccur: collections.Counter = collections.Counter()
    verb_sent_count: collections.Counter = collections.Counter()
    noun_sent_count: collections.Counter = collections.Counter()
    total_sentences = 0
    lemma_sentences = []  # for the Word2Vec cross-check (all content POS, not just candidates)

    for doc in docs_cache:
        for sent in doc.sents:
            total_sentences += 1
            sent_lemmas = []
            verbs_here, nouns_here = set(), set()
            for tok in sent:
                if not tok.is_alpha or len(tok.lemma_) < 3:
                    continue
                lemma = tok.lemma_.lower()
                sent_lemmas.append(lemma)
                if tok.pos_ == "VERB" and lemma in verb_vocab:
                    verbs_here.add(lemma)
                elif tok.pos_ == "NOUN" and lemma in noun_vocab:
                    nouns_here.add(lemma)
            lemma_sentences.append(sent_lemmas)
            for v in verbs_here:
                verb_sent_count[v] += 1
            for n in nouns_here:
                noun_sent_count[n] += 1
            for v, n in itertools.product(verbs_here, nouns_here):
                pair_cooccur[(v, n)] += 1

    print(f"  {total_sentences:,} sentences scanned, {len(pair_cooccur):,} distinct verb-noun pairs observed")

    # ── Score pairs ─────────────────────────────────────────────────────────────
    print("Scoring pairs (log-likelihood G² + PMI) …")
    rows = []
    for (v, n), a in pair_cooccur.items():
        if a < MIN_COOCCUR:
            continue
        b = verb_sent_count[v] - a
        c = noun_sent_count[n] - a
        d = total_sentences - verb_sent_count[v] - noun_sent_count[n] + a
        g2, pmi = collocation_stats(a, b, c, d)
        rows.append({
            "verb": v, "noun": n, "verb_category": verb_categories.get(v, "misc"),
            "co_occur_sentences": a, "verb_sentences": verb_sent_count[v],
            "noun_sentences": noun_sent_count[n], "total_sentences": total_sentences,
            "pmi": round(pmi, 3), "G2": round(g2, 2), "p_label": p_label(g2),
        })
    pairs_df = pd.DataFrame(rows).sort_values("G2", ascending=False)
    pairs_df.to_csv(OUT_PAIRS, index=False)
    print(f"  Written: {OUT_PAIRS.name} ({len(pairs_df):,} scored pairs, min co-occurrence {MIN_COOCCUR})")

    # ── Embedding cross-check ───────────────────────────────────────────────────
    print("Training Word2Vec for embedding cross-check …")
    w2v = Word2Vec(sentences=lemma_sentences, **W2V_PARAMS)

    def top_embedding_nouns(verb: str, n: int = 5) -> list:
        if verb not in w2v.wv:
            return []
        sims = [(w, w2v.wv.similarity(verb, w)) for w in noun_vocab if w in w2v.wv and w != verb]
        sims.sort(key=lambda x: -x[1])
        return sims[:n]

    # Sample verbs for the cross-check: the top verbs by co-occurrence-table
    # presence within each non-misc category, so the report ties back to the
    # categories in verb_categories.txt rather than an arbitrary global top-N.
    category_names = sorted(c for c in set(verb_categories.values()) if c != "misc")
    cross_check_rows = []
    for cat in category_names:
        cat_verbs = [v for v in verb_vocab if verb_categories.get(v) == cat]
        cat_verbs.sort(key=lambda v: -verb_counts[v])
        for v in cat_verbs[:EMBED_CHECK_VERBS_N]:
            pmi_top = (pairs_df[pairs_df["verb"] == v]
                       .sort_values("G2", ascending=False)["noun"].head(5).tolist())
            embed_top = [w for w, _ in top_embedding_nouns(v)]
            cross_check_rows.append({
                "category": cat, "verb": v,
                "pmi_g2_top_nouns": ", ".join(pmi_top),
                "embedding_top_nouns": ", ".join(embed_top),
            })
    cross_df = pd.DataFrame(cross_check_rows)

    # ── Markdown report ──────────────────────────────────────────────────────────
    print("Writing markdown report …")
    top_pairs = pairs_df.head(TOP_PAIRS_N)

    md_lines = [
        "# Verb <-> Noun Occurrence Analysis — Results",
        "",
        "Two independent methods per `README.md` §6: sentence-level co-occurrence scored "
        "with log-likelihood (G²) and PMI (primary), cross-checked against a "
        "freshly-trained Word2Vec embedding (secondary).",
        "",
        "## Corpus summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Sentences scanned | {total_sentences:,} |",
        f"| Candidate verbs (corpus_count >= {MIN_VERB_COUNT}) | {len(verb_vocab):,} |",
        f"| Candidate nouns (top {TOP_NOUNS_N}, stopwords excluded) | {len(noun_vocab):,} |",
        f"| Scored verb-noun pairs (co-occur >= {MIN_COOCCUR} sentences) | {len(pairs_df):,} |",
        "",
        f"## Top {TOP_PAIRS_N} verb-noun pairs, corpus-wide",
        "",
        "*(ranked by G² log-likelihood — a higher score means the pairing is more "
        "statistically surprising, not just more frequent; PMI shown alongside as a "
        "magnitude-of-association measure)*",
        "",
        "| Verb | Noun | Category | Co-occur (sentences) | G² | p | PMI |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in top_pairs.iterrows():
        md_lines.append(
            f"| {r['verb']} | {r['noun']} | {r['verb_category']} | {int(r['co_occur_sentences'])} "
            f"| {r['G2']:.1f} | {r['p_label']} | {r['pmi']:.2f} |"
        )

    md_lines += [
        "",
        "## Embedding cross-check (sample verbs per category)",
        "",
        "*(PMI/G² top nouns vs. Word2Vec nearest-neighbour nouns for the "
        f"{EMBED_CHECK_VERBS_N} most frequent verbs in each non-misc category — agreement "
        "between the two independent methods is a stronger signal than either alone; "
        "disagreement flags a pairing worth reading in context before citing.)*",
        "",
        "| Category | Verb | Top nouns (PMI/G²) | Top nouns (embedding) |",
        "|---|---|---|---|",
    ]
    for _, r in cross_df.iterrows():
        md_lines.append(f"| {r['category']} | {r['verb']} | {r['pmi_g2_top_nouns']} | {r['embedding_top_nouns']} |")

    md_lines += [
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `outputs/verb_noun_pairs.csv` | Full scored verb-noun co-occurrence table (G², PMI, category) |",
        "",
        "## Notes",
        "",
        "- Co-occurrence is measured at the sentence level (spaCy sentence segmentation on "
        "`full_text`) — the same fragment/bullet-point caveat as the active/passive analysis "
        "in `verb_frequency_voice_category_analysis.py` applies: syllabus prose that isn't a "
        "complete sentence may segment inconsistently, which adds noise but shouldn't "
        "systematically bias which nouns a verb associates with.",
        f"- Noun candidates exclude both project stopword lists (`0_preprocessing/stopwords/`) "
        "so the table isn't dominated by institutional boilerplate (\"student\", \"course\", "
        "\"semester\").",
        f"- Pairs below {MIN_COOCCUR} co-occurring sentences are dropped before scoring — too "
        "sparse for G²/PMI to be meaningful.",
        "- The embedding cross-check retrains its own Word2Vec model (same recipe as "
        "`2_semantic_analysis.py`); it's a secondary signal, not a replacement for the "
        "G²/PMI numbers above.",
        "- For the per-year, per-category version of this analysis, see "
        "`verb_category_noun_trends_analysis.py`.",
    ]

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"  Written: {OUT_MD.name}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
