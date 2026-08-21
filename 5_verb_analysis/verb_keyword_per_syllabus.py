"""
Per-Syllabus Verb Category <-> Keyword Category Data
=====================================================
Companion to verb_category_keyword_category_analysis.py: identical
sentence-level scan (same loaders, same token rules — see that script's
module docstring for the method), except co-occurrence tallies are kept
per syllabus instead of pooled corpus-wide.

Output is not a report but a data file for the interactive heatmap
explorer in ../web/word_associations/ (same build_data.py -> data.js pattern
as the bibliography/ visualization):

  ../web/word_associations/data.js  window.VK_DATA = { verb_cats, kw_cats,
                              corpus totals, per-syllabus tallies }

All derived statistics (PMI, per-100-sentence rates, enrichment vs the
corpus baseline) are computed in the browser from these raw counts, so
the explorer can offer multiple readings without re-running spaCy.

Usage:
  python3 verb_keyword_per_syllabus.py            # all syllabi
  python3 verb_keyword_per_syllabus.py --limit 20 # first 20 only
"""

import argparse
import csv
import itertools
import json
import pathlib
import re
import collections

import pandas as pd
import spacy

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR         = pathlib.Path(__file__).parent
INPUT_CSV        = BASE_DIR.parent / "data" / "syllabi_text.csv"
VERB_CATS_FILE   = BASE_DIR / "verb_categories.txt"
KEYWORD_CATS_CSV = BASE_DIR.parent / "data" / "Categories.csv"
OUT_DIR          = BASE_DIR.parent / "web" / "word_associations"
OUT_DIR.mkdir(exist_ok=True)
OUT_DATA         = OUT_DIR / "data.js"

POS_TAG_LABEL = "verb"  # Categories.csv tag meaning "this word is a verb", not a topic

# Same display orders as verb_category_keyword_category_analysis.py, so the
# explorer's grid reads consistently with the static PNG heatmaps.
VERB_CAT_DISPLAY_ORDER = [
    "positioning", "collaborating", "doing", "knowledge",
    "environment_positive", "built_environment", "environment_negative", "questioning",
]


# ── Loaders (mirroring verb_category_keyword_category_analysis.py) ─────────────
def load_verb_categories(path: pathlib.Path) -> dict:
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


def load_keyword_categories(path: pathlib.Path) -> tuple:
    """Returns (form -> {categories}, form -> canonical word) so per-category
    word frequencies can be tallied against the canonical Word, not each alias."""
    lookup: dict = collections.defaultdict(set)
    canonical: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cats = {c.strip() for c in row[""].split(",") if c.strip()}
            cats.discard(POS_TAG_LABEL)
            if not cats:
                continue
            word = row["Word"].strip().lower()
            forms = [word]
            forms += [a.strip().lower() for a in row["Aliases"].split(",") if a.strip()]
            for form in forms:
                if form:
                    lookup[form] |= cats
                    canonical.setdefault(form, word)
    return lookup, canonical


def load_syllabus_references(path: pathlib.Path) -> dict:
    """{syllabus key like '2023cprize-06': [[title, first author], ...]} from the
    bibliography analysis output, kept in that file's citation-count order."""
    refs: dict = collections.defaultdict(list)
    if not path.exists():
        print(f"  [WARN] bibliography not found, skipping references: {path}")
        return refs
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            title = row["title"].strip()
            authors = [a.strip() for a in row["authors"].split(";") if a.strip()]
            author = authors[0] + (" et al." if len(authors) > 1 else "") if authors else ""
            for src in row["source_syllabi"].split(";"):
                src = src.strip()
                if src:
                    refs[src].append([title, author])
    return refs


def keyword_category_word_counts(path: pathlib.Path) -> collections.Counter:
    counts: collections.Counter = collections.Counter()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cats = {c.strip() for c in row[""].split(",") if c.strip()}
            cats.discard(POS_TAG_LABEL)
            for c in cats:
                counts[c] += 1
    return counts


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-syllabus verb-category <-> keyword-category tallies for the heatmap explorer.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N syllabi (default: all)")
    args = parser.parse_args()

    verb_cat_of = load_verb_categories(VERB_CATS_FILE)
    lemma_to_verbcats: dict = collections.defaultdict(set)
    for lemma, cat in verb_cat_of.items():
        lemma_to_verbcats[lemma].add(cat)
    verb_cat_names = sorted(set(verb_cat_of.values()))

    kw_lookup, kw_canonical = load_keyword_categories(KEYWORD_CATS_CSV)
    kw_cat_names = sorted({c for cats in kw_lookup.values() for c in cats})
    print(f"Loaded {len(verb_cat_names)} verb categories ({len(lemma_to_verbcats)} lemmas), "
          f"{len(kw_cat_names)} keyword categories ({len(kw_lookup)} surface forms)")

    refs_by_syllabus = load_syllabus_references(
        BASE_DIR.parent / "3_bibliography_analysis" / "outputs" / "bibliography.csv")
    print(f"Loaded references for {len(refs_by_syllabus)} syllabi")

    print("Loading data …")
    df = pd.read_csv(INPUT_CSV, dtype={"year": "Int64"})
    df = df.dropna(subset=["full_text", "year"]).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"  {len(df)} syllabi loaded")

    print("Loading spaCy model (en_core_web_sm) …")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    print("Scanning sentences per syllabus …")
    syllabi_records = []
    corpus_vcat: collections.Counter = collections.Counter()
    corpus_kcat: collections.Counter = collections.Counter()
    corpus_cell: collections.Counter = collections.Counter()
    verb_word_counts: dict = collections.defaultdict(collections.Counter)
    kw_word_counts: dict = collections.defaultdict(collections.Counter)
    corpus_sentences = 0
    # Sentences containing at least one categorized word (verb or keyword).
    # Used as the "chance" baseline so administrative boilerplate (dates,
    # grading policies) doesn't inflate every pairing above independence.
    corpus_active_sentences = 0

    texts = df["full_text"].astype(str).tolist()
    for i, doc in enumerate(nlp.pipe(texts, batch_size=20)):
        vcat_n: collections.Counter = collections.Counter()
        kcat_n: collections.Counter = collections.Counter()
        cell_n: collections.Counter = collections.Counter()
        n_sents = 0
        for sent in doc.sents:
            n_sents += 1
            vcats_here, kcats_here = set(), set()
            for tok in sent:
                if not tok.is_alpha or len(tok.lemma_) < 3:
                    continue
                lemma = tok.lemma_.lower()
                if tok.pos_ == "VERB":
                    cats = lemma_to_verbcats.get(lemma, set())
                    vcats_here |= cats
                    for c in cats:
                        verb_word_counts[c][lemma] += 1
                else:
                    form = lemma if lemma in kw_lookup else tok.text.lower()
                    hit = kw_lookup.get(form)
                    if hit:
                        kcats_here |= hit
                        word = kw_canonical.get(form, form)
                        for c in hit:
                            kw_word_counts[c][word] += 1
            if vcats_here or kcats_here:
                corpus_active_sentences += 1
            for v in vcats_here:
                vcat_n[v] += 1
            for k in kcats_here:
                kcat_n[k] += 1
            for v, k in itertools.product(vcats_here, kcats_here):
                cell_n[(v, k)] += 1

        row = df.iloc[i]
        title = str(row["pdf_title"])
        m = re.match(r"(\d{4})cprize-(\d+)", title)
        short = f"{m.group(1)}-{int(m.group(2)):02d}" if m else title
        ref_key = re.sub(r"nb$", "", title)
        syllabi_records.append({
            "t": title,
            "s": short,
            "y": int(row["year"]),
            "n": n_sents,
            "v": dict(vcat_n),
            "k": dict(kcat_n),
            "c": {f"{v}|{k}": n for (v, k), n in cell_n.items()},
            "refs": refs_by_syllabus.get(ref_key, []),
        })
        corpus_vcat.update(vcat_n)
        corpus_kcat.update(kcat_n)
        corpus_cell.update(cell_n)
        corpus_sentences += n_sents
        if (i + 1) % 50 == 0:
            print(f"  scanned {i + 1}/{len(texts)} syllabi …")

    print(f"  {corpus_sentences:,} sentences across {len(syllabi_records)} syllabi")

    verb_order = [v for v in VERB_CAT_DISPLAY_ORDER if v in verb_cat_names]
    verb_order += [v for v in verb_cat_names if v not in verb_order]
    kw_vocab_sizes = keyword_category_word_counts(KEYWORD_CATS_CSV)
    kw_order = [k for k, _ in kw_vocab_sizes.most_common() if k in kw_cat_names]
    kw_order += [k for k in kw_cat_names if k not in kw_order]

    payload = {
        "verb_cats": verb_order,
        "kw_cats": kw_order,
        "verb_words": {c: [[w, n] for w, n in verb_word_counts[c].most_common()]
                       for c in verb_order},
        "kw_words": {c: [[w, n] for w, n in kw_word_counts[c].most_common()]
                     for c in kw_order},
        "corpus": {
            "total_sentences": corpus_sentences,
            "active_sentences": corpus_active_sentences,
            "v": dict(corpus_vcat),
            "k": dict(corpus_kcat),
            "c": {f"{v}|{k}": n for (v, k), n in corpus_cell.items()},
        },
        "syllabi": syllabi_records,
    }
    OUT_DATA.write_text(
        "window.VK_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Written: {OUT_DATA} ({OUT_DATA.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
