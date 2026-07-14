"""
Step 0: Preprocessing
=====================
Reads all PDFs from data/ALL2020-2026-nb/, extracts text via pdfplumber,
applies both stopword lists, and writes:

  data/syllabi_text.csv           — one row per PDF (pdf_title, year, full_text, filtered_text)
  outputs/top500_corpus.csv       — top 500 words across all years (rank, word, count)
  outputs/top100_per_year.csv     — top 100 words per year (long format: year, rank, word, count)
  outputs/keywords_corpus.csv     — per-keyword counts from data/keywords.txt (corpus + per year)
  0_preprocessing.md              — output summary (generated)

Stopword lists in stopwords/:
  stopwords.txt   — general English stopwords
  stopwords_2.txt — academic/institutional boilerplate

Usage:
  python3 0_preprocessing.py            # process all PDFs
  python3 0_preprocessing.py --limit 5  # process first 5 PDFs only
"""

import re
import csv
import pathlib
import argparse
from collections import Counter

import pdfplumber
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = pathlib.Path(__file__).parent
DATA_DIR     = BASE_DIR.parent / "data"
PDF_DIR      = DATA_DIR / "ALL2020-2026-nb"
SW_DIR       = BASE_DIR / "stopwords"
OUT_DIR      = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

STOPWORD_FILES = [
    SW_DIR / "stopwords.txt",
    SW_DIR / "stopwords_2.txt",
]

KEYWORDS_FILE   = DATA_DIR / "keywords.txt"
OUTPUT_CSV      = DATA_DIR / "syllabi_text.csv"
OUT_TOP500_ALL  = OUT_DIR / "top500_corpus.csv"
OUT_TOP100_YEAR = OUT_DIR / "top100_per_year.csv"
OUT_KEYWORDS    = OUT_DIR / "keywords_corpus.csv"
OUT_MD             = BASE_DIR / "0_preprocessing.md"
KEYWORDS_FILE_2 = DATA_DIR / "keywords-2.txt"
OUT_KEYWORDS_2  = OUT_DIR / "keywords-2_corpus.csv"

TOP_CORPUS_N = 500
TOP_YEAR_N   = 100

# ── Load stopwords ─────────────────────────────────────────────────────────────
def load_stopwords(paths: list) -> set:
    words = set()
    for path in paths:
        path = pathlib.Path(path)
        if not path.exists():
            print(f"  [WARN] stopword file not found: {path}")
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip().lower()
            if line and not line.startswith("#"):
                words.add(line)
    return words

# ── Parse keywords.txt ────────────────────────────────────────────────────────
def parse_keywords(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """
    Returns [(label, [alias, alias, ...]), ...].
    Label = first alias (lowercased).  Aliases are all comma-separated tokens on the line.
    """
    groups = []
    if not path.exists():
        print(f"  [WARN] keywords file not found: {path}")
        return groups
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            aliases = [a.strip().lower() for a in line.split(",") if a.strip()]
            if aliases:
                groups.append((aliases[0], aliases))
    return groups

# ── Extract text from a single PDF ────────────────────────────────────────────
def extract_text(pdf_path: pathlib.Path) -> str:
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    except Exception as exc:
        tqdm.write(f"  [WARN] could not read {pdf_path.name}: {exc}")
    raw = "\n".join(pages)
    raw = raw.replace("\x00", "")
    raw = re.sub(r"\(cid:\d+\)", "", raw)
    return raw

# ── Filter text ────────────────────────────────────────────────────────────────
def filter_text(raw: str, stopwords: set) -> str:
    tokens = re.findall(r"[a-zA-Z]+", raw.lower())
    kept = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(kept)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract text from syllabi PDFs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N PDFs (default: all)")
    args = parser.parse_args()

    stopwords = load_stopwords(STOPWORD_FILES)
    print(f"Loaded {len(stopwords)} stopwords from {len(STOPWORD_FILES)} file(s)")

    keyword_groups = parse_keywords(KEYWORDS_FILE)
    print(f"Loaded {len(keyword_groups)} keyword groups from {KEYWORDS_FILE.name}")

    keyword_groups_2 = parse_keywords(KEYWORDS_FILE_2)
    print(f"Loaded {len(keyword_groups_2)} keyword groups from {KEYWORDS_FILE_2.name}")

    if OUTPUT_CSV.exists() and not args.limit:
        print(f"Found existing {OUTPUT_CSV} — skipping PDF extraction …")
        rows = []
        with OUTPUT_CSV.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({
                    "pdf_title":     row["pdf_title"],
                    "year":          int(row["year"]) if row["year"] else None,
                    "full_text":     row.get("full_text", ""),
                    "filtered_text": row.get("filtered_text", ""),
                })
        print(f"Loaded {len(rows)} rows from {OUTPUT_CSV}")
    else:
        pdf_files = sorted(PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")
        if args.limit:
            pdf_files = pdf_files[: args.limit]

        rows = []
        for pdf_path in tqdm(pdf_files, desc="Extracting PDFs", unit="pdf"):
            year_match = re.match(r"^(\d{4})", pdf_path.name)
            year = int(year_match.group(1)) if year_match else None
            full_text     = extract_text(pdf_path)
            filtered_text = filter_text(full_text, stopwords)
            rows.append({
                "pdf_title":     pdf_path.stem,
                "year":          year,
                "full_text":     full_text,
                "filtered_text": filtered_text,
            })

        # ── Write syllabi CSV ──────────────────────────────────────────────────
        fieldnames = ["pdf_title", "year", "full_text", "filtered_text"]
        with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Done. {len(rows)} rows written to {OUTPUT_CSV}")

    # ── Compute word counts ────────────────────────────────────────────────────
    print("Computing word counts …")

    year_counters: dict[int, Counter] = {}
    corpus_counter: Counter = Counter()

    for row in rows:
        yr = row["year"]
        if yr is None:
            continue
        tokens = row["filtered_text"].split()
        if yr not in year_counters:
            year_counters[yr] = Counter()
        year_counters[yr].update(tokens)
        corpus_counter.update(tokens)

    years_sorted = sorted(year_counters.keys())

    # ── Write corpus-wide top 500 ──────────────────────────────────────────────
    top_corpus = corpus_counter.most_common(TOP_CORPUS_N)
    with OUT_TOP500_ALL.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rank", "word", "count"])
        for rank, (word, count) in enumerate(top_corpus, 1):
            writer.writerow([rank, word, count])
    print(f"Corpus top-{TOP_CORPUS_N} written to {OUT_TOP500_ALL}")

    # ── Write per-year top 100 (long format) ──────────────────────────────────
    with OUT_TOP100_YEAR.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year", "rank", "word", "count"])
        for yr in years_sorted:
            for rank, (word, count) in enumerate(year_counters[yr].most_common(TOP_YEAR_N), 1):
                writer.writerow([yr, rank, word, count])
    print(f"Per-year top-{TOP_YEAR_N} written to {OUT_TOP100_YEAR}")

    # ── Write keywords_corpus.csv ──────────────────────────────────────────────
    if keyword_groups:
        kw_rows = []
        for label, aliases in keyword_groups:
            corpus_count = sum(corpus_counter.get(a, 0) for a in aliases)
            yr_counts = {str(yr): sum(year_counters[yr].get(a, 0) for a in aliases)
                         for yr in years_sorted}
            row_dict = {
                "label":         label,
                "aliases":       ", ".join(aliases),
                "corpus_count":  corpus_count,
            }
            row_dict.update(yr_counts)
            kw_rows.append(row_dict)

        fieldnames_kw = ["label", "aliases", "corpus_count"] + [str(y) for y in years_sorted]
        with OUT_KEYWORDS.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_kw)
            writer.writeheader()
            writer.writerows(kw_rows)
        print(f"Keyword counts written to {OUT_KEYWORDS}")

    # ── Write keywords-2/keywords_corpus.csv ──────────────────────────────────
    if keyword_groups_2:
        kw2_rows = []
        for label, aliases in keyword_groups_2:
            corpus_count = sum(corpus_counter.get(a, 0) for a in aliases)
            yr_counts = {str(yr): sum(year_counters[yr].get(a, 0) for a in aliases)
                         for yr in years_sorted}
            row_dict = {
                "label":        label,
                "aliases":      ", ".join(aliases),
                "corpus_count": corpus_count,
            }
            row_dict.update(yr_counts)
            kw2_rows.append(row_dict)

        fieldnames_kw2 = ["label", "aliases", "corpus_count"] + [str(y) for y in years_sorted]
        with OUT_KEYWORDS_2.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_kw2)
            writer.writeheader()
            writer.writerows(kw2_rows)
        print(f"Keyword-2 counts written to {OUT_KEYWORDS_2}")

    # ── Write markdown summary ─────────────────────────────────────────────────
    n_syllabi     = len(rows)
    total_tokens  = sum(corpus_counter.values())
    unique_tokens = len(corpus_counter)

    md_lines = [
        "# Step 0: Preprocessing",
        "",
        "## What this script does",
        "",
        "- Extracts text from all PDFs in `data/ALL2020-2026-nb/`",
        "- Applies two stopword lists (`stopwords/stopwords.txt` and `stopwords/stopwords_2.txt`)",
        "- Writes a syllabi CSV, keyword counts, and top-word frequency tables",
        "",
        "## Output files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `data/syllabi_text.csv` | One row per PDF: `pdf_title`, `year`, `full_text`, `filtered_text` |",
        f"| `outputs/top{TOP_CORPUS_N}_corpus.csv` | Top {TOP_CORPUS_N} words across all years: `rank`, `word`, `count` |",
        f"| `outputs/top{TOP_YEAR_N}_per_year.csv` | Top {TOP_YEAR_N} words per year (long format): `year`, `rank`, `word`, `count` |",
        "| `outputs/keywords_corpus.csv` | Per-keyword counts from `data/keywords.txt`: `label`, `aliases`, `corpus_count`, per-year counts |",
        "| `outputs/keywords-2_corpus.csv` | Per-keyword counts from `data/keywords-2.txt`: `label`, `aliases`, `corpus_count`, per-year counts |",
        "",
        "## Corpus summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total syllabi | {n_syllabi:,} |",
        f"| Years covered | {min(years_sorted)} – {max(years_sorted)} |",
        f"| Total tokens (after stopword filter) | {total_tokens:,} |",
        f"| Unique tokens | {unique_tokens:,} |",
        "",
        f"## Top {TOP_CORPUS_N} words (corpus-wide)",
        "",
        "| Rank | Word | Count |",
        "|------|------|-------|",
    ]
    for rank, (word, count) in enumerate(top_corpus, 1):
        md_lines.append(f"| {rank} | {word} | {count:,} |")

    if keyword_groups:
        md_lines += [
            "",
            "## Keyword counts",
            "",
            "| Keyword | Aliases | Corpus count |",
            "|---------|---------|--------------|",
        ]
        for label, aliases in keyword_groups:
            corpus_count = sum(corpus_counter.get(a, 0) for a in aliases)
            alias_str = ", ".join(aliases[1:]) if len(aliases) > 1 else "—"
            md_lines.append(f"| {label} | {alias_str} | {corpus_count:,} |")

    if keyword_groups_2:
        md_lines += [
            "",
            "## Keyword-2 counts",
            "",
            "| Keyword | Aliases | Corpus count |",
            "|---------|---------|--------------|",
        ]
        for label, aliases in keyword_groups_2:
            corpus_count = sum(corpus_counter.get(a, 0) for a in aliases)
            alias_str = ", ".join(aliases[1:]) if len(aliases) > 1 else "—"
            md_lines.append(f"| {label} | {alias_str} | {corpus_count:,} |")

    md_lines += [
        "",
        "## Stopword files",
        "",
        "| File | Purpose |",
        "|------|---------|",
        "| `stopwords/stopwords.txt` | General English stopwords |",
        "| `stopwords/stopwords_2.txt` | Academic/institutional boilerplate |",
    ]

    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Summary written to {OUT_MD}")


if __name__ == "__main__":
    main()
