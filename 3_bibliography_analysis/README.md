# Bibliography Analysis

Extracts every work cited in the syllabi PDFs, deduplicates across syllabi, tallies citations per year, and (optionally) enriches each work with Open Library metadata.

## Pipeline

1. **PDF → bibliography text.** Pulls the bibliography/reading-list section out of each syllabus in `data/ALL2020-2026-full`, using the corresponding `data/ALL2020-2026-nb` PDF as a diff target where no explicit header is found.
2. **Text → structured citations.** Parses each bibliography's text into `{title, authors, year, type, publisher, journal, place}` records. Two interchangeable parsers:
   - **Claude API** (default, requires `ANTHROPIC_API_KEY`) — handles messy/garbled PDF text and non-English titles far better than regex.
   - **Regex fallback** (`--no-llm`) — no API key needed, faster, but noticeably rougher on garbled text.
3. **Citations → deduplicated table.** Merges duplicate works across syllabi, tallies per-year citation counts, enriches with Open Library (ISBN/genre/description) where reachable, and writes the final CSV and keyword charts.

All three stages run from one script.

## Usage

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 3_bibliography_analysis.py
```

```
--no-llm      Use the regex parser instead of the Claude API
--limit N     Process only the first N PDFs
--skip-ol     Skip Open Library enrichment
--reparse     Re-extract and re-parse from the PDFs even if outputs/raw_citations.json already exists
```

If `outputs/raw_citations.json` exists, the script loads it and skips straight to deduplication + enrichment + output — it does not re-extract or re-parse. Pass `--reparse` to force a full rebuild from the PDFs.

## Outputs

| File                                                | Contents                                                                                                  |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `outputs/raw_citations.json`                        | Intermediate: one entry per syllabus, citations before dedup. Determines everything downstream.           |
| `outputs/bibliography.csv`                          | Final table: one row per unique work, with a citation count per year (2020–2026) and a total.             |
| `outputs/extraction_log.csv`                        | Per-syllabus summary: which bibliography-text-extraction method was used and how many citations came out. |
| `outputs/summary_stats.csv`                         | Dataset-wide counts: raw vs. unique citations, per-syllabus avg/median/min/max, dedup rate, most-cited work, Open Library match rate, and other headline stats. |
| `outputs/keyword_analysis/bibliography_matched.csv` | Same works annotated against `data/keywords.txt` and `data/go-words.txt`.                                 |
| `outputs/keyword_analysis/*.png`                    | Term-frequency bar chart and per-year heatmaps for keywords and go-words.                                 |
| `outputs/overview/top_20_titles.png`                | Most-cited works, ranked by citation count across syllabi.                                                |
| `outputs/overview/top_20_authors.png`               | Most-cited authors, ranked by citation count across syllabi.                                              |
| `outputs/overview/citation_types.png`               | Citations by work type (book, article, website, chapter, report, other).                                  |
| `outputs/overview/publication_years.png`            | Citations by the cited work's original publication year, in 5-year bins.                                  |
| `outputs/overview/syllabi_per_genre.png`            | Distinct syllabi citing at least one work tagged with each Open Library genre/subject.                    |
| `outputs/overview/top_20_publishers.png`            | Most-cited publishers, ranked by citation count across syllabi.                                           |
| `outputs/overview/top_20_journals.png`              | Most-cited journals, ranked by citation count across syllabi (article-typed works only — see Notes).      |
| `outputs/open_library_cache.json`                   | Cache of Open Library lookups, keyed by normalized title.                                                 |
| `3_bibliography_analysis/comparing_citations/`      | Standalone script + outputs comparing citation overlap between specific texts — see its own directory.    |

## Notes

- Open Library enrichment needs outbound internet access. Genuine network/connection failures are never cached, so they're retried on the next run; a cached `null` genre means Open Library was actually reached and either found no matching work, or found a match with no subject data — not that the request failed.
- The regex fallback is a last resort, not a substitute for the Claude API path — expect it to occasionally swap an author into the title field or vice versa on irregular citation formats.
- Genre coverage depends entirely on Open Library match rate, which is typically low (single digits of a percent of unique works) — `syllabi_per_genre.png` and the `pct_unique_works_with_open_library_genre` stat make that coverage explicit. Treat both as a partial view, not a full breakdown.
- All single-series bar/histogram charts in `outputs/overview/` use one fixed hue (`#0a5c6b`) for magnitude, not an identity color per category.
- Author names are normalized and fuzzy-matched (`normalise_author`/`author_match_key` in the main script) so `"Olgyay, Victor"`, `"Victor Olgyay"`, and `"V. Olgyay"` consolidate into one author — except for organizational authors (AIA, U.S. Department of Energy, etc.), which are left as-is.
- A book chapter's containing volume was originally extracted into the same `journal` field as a real periodical name, which made `bibliography.csv`'s `journal` column misleading for chapters (e.g. showing a book title where a journal name was expected). `split_journal_and_container` now moves that value to its own `container_title` column, so `journal` — and `top_20_journals.png` — only ever reflect genuine periodicals. Publisher and journal name variants (e.g. `"MIT Press"` vs `"The MIT Press"`) are not yet consolidated the way author names are.
- `description` values (from Open Library) have embedded newlines stripped at CSV-write time.
