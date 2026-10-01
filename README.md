# Green Counting

Text analysis of architecture syllabi (2020–2026), tracking how climate-related
language, topics, citations, verbs, images, narratives, and discourse emerge and change
across the corpus. Conducted for the Buell Center at Columbia GSAPP.

## Pipeline

Each numbered folder is a pipeline step (run roughly in order; some are independent).
Every folder has its own `README.md` with full details — this is just the map.

| Step | Folder                                                            | What it does                                                                                                                   |
| ---- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 0    | [`0_preprocessing/`](0_preprocessing/README.md)                   | PDF → text extraction, stopword filtering, top-word and keyword counts.                                                        |
| 1    | [`1_keyword_analysis/`](1_keyword_analysis/1_keyword_analysis.py) | Keyword frequency trends over time (charts + report), log-likelihood "top movers."                                             |
| 2    | [`2_semantic_analysis/`](2_semantic_analysis/README.md)           | Word2Vec embeddings per year to map how keyword meaning shifts 2020→2026 (landscape maps, trajectories, concentric diagrams).  |
| 3    | [`3_bibliography_analysis/`](3_bibliography_analysis/README.md)   | Extracts and dedupes every citation from the syllabi bibliographies, tallies per year, enriches with Open Library metadata.    |
| 5    | [`5_verb_analysis/`](5_verb_analysis/README.md)                   | spaCy verb/voice analysis — which verbs are used, active vs. passive, verb↔noun and verb-category↔topic-category associations. |
| 6    | [`6_images/`](6_images/extract_images.py)                         | Extracts embedded images from syllabus PDFs for the image gallery, filtering out logos/blank space/duplicates.                 |

## Other folders

- **`data/`** — source PDFs and shared derived files are ignored from the public repo.
- **`web/`** — the plaintext source for the public site (landing page, image gallery, bibliography network, word-association explorer) at https://elliemadsen.net/green-counting/.
- **`docs/`** — generated, password-gated build of `web/` (GitHub Pages serves from here). Don't hand-edit; rebuild with `node web/build.js`.
