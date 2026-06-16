"""
Step 0: PDF Preprocessing (scrape text, filter stopwords, write CSV)
Reads all PDFs from ALL2020-2026-nb/, extracts text via pdfplumber,
and writes a CSV with one row per PDF containing:
  - pdf_title : filename stem
  - year      : extracted from filename prefix (e.g. 2020, 2021, …)
  - full_text : all scraped text joined across pages
  - filtered_text : full_text with stopwords (stopwords.txt) and short tokens removed

Usage:
  python3 0_pdf_preprocessing.py            # process all PDFs
  python3 0_pdf_preprocessing.py --limit 5  # process first 5 PDFs only
"""

import re
import csv
import pathlib
import argparse
import pdfplumber
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent
PDF_DIR     = BASE_DIR / "ALL2020-2026-nb"
STOPWORDS_F = BASE_DIR / "stopwords.txt"
OUTPUT_CSV  = BASE_DIR / "syllabi_text.csv"

# ── Load stopwords ─────────────────────────────────────────────────────────────
def load_stopwords(path: pathlib.Path) -> set:
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words

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
    # Strip PDF font-encoding artifacts like (cid:80) that appear when a PDF
    # uses a custom font without a proper Unicode mapping.
    raw = re.sub(r"\(cid:\d+\)", "", raw)
    return raw

# ── Filter text for stopwords ──────────────────────────────────────────────────
def filter_text(raw: str, stopwords: set) -> str:
    # Lowercase, keep only alphabetic tokens, remove stopwords and length-1 tokens
    tokens = re.findall(r"[a-zA-Z]+", raw.lower())
    kept = [t for t in tokens if t not in stopwords and len(t) > 1]
    return " ".join(kept)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract text from syllabi PDFs.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N PDFs (default: all)")
    args = parser.parse_args()

    stopwords = load_stopwords(STOPWORDS_F)
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")

    if args.limit:
        pdf_files = pdf_files[: args.limit]

    rows = []
    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs", unit="pdf"):
        # Extract year from filename prefix (first 4 chars are the year)
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

    # Write CSV
    fieldnames = ["pdf_title", "year", "full_text", "filtered_text"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} rows written to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
