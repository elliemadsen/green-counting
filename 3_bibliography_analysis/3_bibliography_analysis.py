"""
Step 3: Bibliography Analysis
==============================
Extracts all cited texts from syllabi bibliographies / reading-lists
(ALL2020-2026-full vs ALL2020-2026-nb), deduplicates across syllabi,
and enriches each entry with Open Library metadata.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 3_bibliography_analysis.py
    python3 3_bibliography_analysis.py --no-llm   # regex-only fallback (faster, less accurate)
    python3 3_bibliography_analysis.py --limit 5  # test with first 5 PDFs only

Outputs (in outputs/):
    bibliography.csv        — deduplicated citation table with per-year tallies
    raw_citations.json      — intermediate: all parsed citations before dedup
    extraction_log.csv      — per-PDF extraction summary
"""

from __future__ import annotations

import re
import csv
import json
import time
import pathlib
import argparse
import difflib
import unicodedata
import os
import statistics
from collections import defaultdict, Counter
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pdfplumber
import requests
from tqdm import tqdm

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
FULL_DIR = DATA_DIR / "ALL2020-2026-full"
NB_DIR   = DATA_DIR / "ALL2020-2026-nb"
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

RAW_JSON    = OUT_DIR / "raw_citations.json"
BIB_CSV     = OUT_DIR / "bibliography.csv"
LOG_CSV     = OUT_DIR / "extraction_log.csv"
STATS_CSV   = OUT_DIR / "summary_stats.csv"
OL_CACHE    = OUT_DIR / "open_library_cache.json"

KEYWORDS_FILE = DATA_DIR / "keywords.txt"
GO_WORDS_FILE = DATA_DIR / "go-words.txt"
KW_OUT_DIR    = OUT_DIR / "keyword_analysis"

YEARS = list(range(2020, 2027))

# Patterns that signal the start of a bibliography section
BIB_HEADER_RE = re.compile(
    r'^\s*(bibliography|references?|works\s+cited|reading\s+list|'
    r'course\s+readings?|required\s+readings?|supplementary\s+readings?|'
    r'bibliography\s+and\s+complementary\s+readings?|'
    r'relevant\s+articles?\s*\+?\s*publications?)\s*[:\n]?',
    re.IGNORECASE | re.MULTILINE,
)

YEAR_RE = re.compile(r'\b(19[5-9]\d|20[0-3]\d)\b')


# ── PDF helpers ────────────────────────────────────────────────────────────────

def extract_pages(pdf_path: pathlib.Path) -> list[str]:
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                raw = page.extract_text() or ''
                raw = raw.replace('\x00', '')
                raw = re.sub(r'\(cid:\d+\)', '', raw)
                pages.append(raw)
    except Exception as exc:
        tqdm.write(f"  [WARN] {pdf_path.name}: {exc}")
    return pages


def get_bibliography_text(full_path: pathlib.Path,
                          nb_path: pathlib.Path | None) -> tuple[str, str]:
    """
    Returns (bib_text, method) where method describes how it was extracted.
    Three strategies tried in order:
      A) Explicit bibliography header found in full text
      B) Line-level diff between full and nb pages
      C) Heuristic: lines that look like citations
    """
    full_pages = extract_pages(full_path)
    if not full_pages:
        return '', 'empty'

    full_text = '\n'.join(full_pages)

    # Strategy A: explicit header
    m = BIB_HEADER_RE.search(full_text)
    if m:
        bib = full_text[m.start():].strip()
        return bib, 'header'

    # Strategy B: diff against nb version
    if nb_path and nb_path.exists():
        nb_pages = extract_pages(nb_path)
        extra = _diff_extra_lines(full_pages, nb_pages)
        if extra:
            return extra, 'diff'

    # Strategy C: heuristic citation lines
    heuristic = _citation_heuristic(full_text)
    if heuristic:
        return heuristic, 'heuristic'

    return '', 'none'


def _diff_extra_lines(full_pages: list[str], nb_pages: list[str]) -> str:
    """Lines present in full pages but absent from the corresponding nb pages."""
    extra = []
    for i, pg_full in enumerate(full_pages):
        if i < len(nb_pages):
            nb_line_set = {ln.strip() for ln in nb_pages[i].splitlines()}
        else:
            nb_line_set = set()

        for line in pg_full.splitlines():
            stripped = line.strip()
            if (stripped
                    and stripped not in nb_line_set
                    and len(stripped) > 25):
                extra.append(stripped)

    return '\n'.join(extra)


def _citation_heuristic(text: str) -> str:
    """Extract lines that look like citations (have a year + enough text)."""
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) > 40 and YEAR_RE.search(s):
            lines.append(s)
    return '\n'.join(lines)


# ── LLM-based citation parsing ────────────────────────────────────────────────

LLM_SYSTEM = """\
You are a bibliographic data extractor for architecture syllabi.
Given raw text from a bibliography or reading list, extract every cited work.

Return a JSON object with a single key "citations" whose value is an array.
Each array element is an object with these keys (use null if unknown):
  title     : string — full title of the work
  authors   : array of strings — author names as they appear
  year      : integer or null — publication year
  type      : one of "book", "article", "chapter", "report", "film", "website", "other"
  publisher : string or null — publisher name (for books/reports)
  journal   : string or null — journal or series name (for articles/chapters)
  place     : string or null — place of publication

Rules:
- Include every distinct cited work, including journal articles and book chapters.
- Do NOT include week/session headers, instructor notes, or course admin text.
- Do NOT include bare URLs unless they represent a standalone online work.
- If the same work appears multiple times, include it only once.
- Return ONLY valid JSON — no markdown fences, no prose before or after.
"""

def parse_with_llm(texts: list[tuple[str, str]]) -> list[list[dict]]:
    """
    texts: list of (pdf_stem, bib_text)
    Returns: list of citation lists, one per entry.
    One API call per PDF for reliability.
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if not api_key or not HAS_ANTHROPIC:
        raise RuntimeError("ANTHROPIC_API_KEY not set or anthropic not installed")

    client = anthropic.Anthropic(api_key=api_key)
    all_results: list[list[dict]] = []

    for stem, bib in tqdm(texts, desc="LLM parsing", unit="pdf"):
        user_msg = (
            "Extract all cited works from the bibliography / reading-list text below. "
            "The text comes from an architecture course syllabus.\n\n"
            f"{bib}"
        )
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=4096,
                system=LLM_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = msg.content[0].text.strip()
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)
            parsed = json.loads(raw)
            citations = parsed.get('citations', [])
            if not isinstance(citations, list):
                citations = []
            all_results.append(citations)
        except Exception as exc:
            tqdm.write(f"  [WARN] LLM failed for {stem}: {exc}")
            all_results.append([])

    return all_results


# ── Regex-based citation parsing (fallback) ───────────────────────────────────
#
# No API key needed, but purely pattern-based — expect noticeably rougher
# titles/authors than the LLM path, especially on garbled PDF-extracted text.
# Ported from the former standalone parse_bib_texts.py, which duplicated this
# job with a stronger implementation than the one that used to live here.

# Curly/smart quote char class (open+close, double+single)
_QOPEN  = '“‘„'
_QCLOSE = '”’‛'
_QUOTE  = _QOPEN + _QCLOSE + '"'

_QUOTED_TITLE_RE = re.compile('[' + _QUOTE + '][^' + _QUOTE + ']{10,150}[' + _QUOTE + ']')

_NOISE_PATTERNS = [
    re.compile(r'\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}'),     # phone
    re.compile(r'\b[\w.+-]+@[\w-]+\.\w{2,}\b'),            # email
    re.compile(
        r'\b(she has|he has|his work|her work|has been honored|'
        r'has received|has taught|has served|has published|'
        r'is a professor|is associate professor|is assistant professor|'
        r'faculty member|studio coordinator|visiting critic|'
        r'her research|his research|her practice|his practice)\b',
        re.I,
    ),
    re.compile(
        r'\b(shortlisted|longlisted|first place|second place|third place|'
        r'runner.up|finalist|laureate|honorable mention)\b',
        re.I,
    ),
    re.compile(r'^(week|module|session|lecture|unit|lab)\s*\d', re.I),
    re.compile(r'^(office hours|prerequisites|credit hours|course number|grading)', re.I),
    re.compile(r'\b(curriculum vitae|cv:)\b', re.I),
    re.compile(r'\b(acsa|buell center)\b.*\b(prize|award)\b', re.I),
    re.compile(r'^\s*fax\s*:', re.I),
    re.compile(r'^\s*tel(?:ephone)?:', re.I),
]

_PUBLISHER_RE = re.compile(
    r'\b(MIT Press|Princeton University Press|Columbia University Press|'
    r'University of Chicago Press|Oxford University Press|'
    r'Cambridge University Press|Yale University Press|'
    r'Harvard University Press|Stanford University Press|'
    r'University of Minnesota Press|Duke University Press|'
    r'University of California Press|Cornell University Press|'
    r'University of Pennsylvania Press|Johns Hopkins University Press|'
    r'University of Toronto Press|University of Washington Press|'
    r'Routledge|Verso|Phaidon|Penguin|Knopf|Norton|'
    r'Zone Books|Actar|Lars M.ller|Steidl|Birkh.user|'
    r'Palgrave|Bloomsbury|Thames.Hudson|Hatje Cantz|Prestel|'
    r'Architectural Press|Wiley|Springer|Elsevier|Earthscan|'
    r'Chelsea Green|New Society|Island Press|Metropolis Books|'
    r'Walter de Gruyter|Taschen|Rizzoli|Monacelli|'
    r'[A-Z][a-z]+ University Press)',
    re.I,
)

_JOURNAL_SIGNAL_RE = re.compile(
    r'\b(vol\.?\s*\d|volume\s*\d|no\.?\s*\d|issue\s*\d|'
    r'pp\.?\s*\d+[-–]\d+|\d+\s*\(\d+\)\s*:\s*\d+)',
    re.I,
)

_AUTHOR_FORMAT_RE = re.compile(r'\b[A-Z][a-z\xc0-\xff]{1,20},\s+[A-Z]')

_PLACE_RE = re.compile(
    r'^(new york|london|chicago|cambridge|oxford|boston|'
    r'washington|toronto|amsterdam|berlin|paris|princeton|'
    r'cambridge|ithaca|durham|minneapolis)',
    re.I,
)

_FALSE_POSITIVE_TITLES = re.compile(
    r'^(introduction|conclusion|preface|foreword|acknowledgements?|'
    r'appendix|chapter|section|part)\s*\.?\s*$',
    re.I,
)

_KNOWN_JOURNALS = [
    'Architectural Record', 'Architecture + Urbanism',
    "Architect's Journal", 'Journal of Architecture',
    'Journal of Architectural Education',
    'Log', 'AA Files', 'e-flux', 'Perspecta',
    'Critical Inquiry', 'Environmental History', 'Assemblage',
    'Grey Room', 'New Left Review', 'October', 'Representations',
    'Harvard Design Magazine', 'Metropolis', 'Architectural Review',
    'Environment and Planning', 'Urban Studies',
    'Landscape and Urban Planning', 'Landscape Journal',
    'Places Journal', 'Journal of Urban Design',
    'International Journal of Architectural Research',
    'Science', 'Nature', 'PNAS',
]


def parse_with_regex(bib_text: str) -> list[dict]:
    """Heuristic entry splitter + field extractor — works without an API key."""
    bib = BIB_HEADER_RE.sub('', bib_text, count=1).strip()
    results: list[dict] = []
    seen: set[str] = set()
    for block in _split_into_blocks(bib):
        rec = _parse_block(block)
        if not rec:
            continue
        key = rec['title'].lower()[:60]
        if key not in seen:
            seen.add(key)
            results.append(rec)
    return results


def _is_noise(text: str) -> bool:
    return any(pat.search(text) for pat in _NOISE_PATTERNS)


def _citation_score(text: str) -> int:
    """Higher score = more confident this block is an actual citation."""
    score = 0
    if YEAR_RE.search(text):
        score += 1
    if _PUBLISHER_RE.search(text):
        score += 2
    if _JOURNAL_SIGNAL_RE.search(text):
        score += 2
    if _AUTHOR_FORMAT_RE.search(text):
        score += 1
    if _QUOTED_TITLE_RE.search(text):
        score += 1
    if re.match(r'^(\d{1,3}[.)]\s|[•–—-]\s)', text.strip()):
        score += 1
    return score


def _split_into_blocks(text: str) -> list[str]:
    """Split bibliography text into candidate citation blocks."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    blank_blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]

    if len(blank_blocks) >= 3:
        result: list[str] = []
        for block in blank_blocks:
            if len(block) > 600:
                result.extend(_split_on_entry_starts(block))
            else:
                result.append(block)
        return result

    return _split_on_entry_starts(text)


def _split_on_entry_starts(text: str) -> list[str]:
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if not lines:
        return []

    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        is_new = (
            bool(re.match(r'^\d{1,3}[.)]\s', line)) or
            bool(re.match(r'^[•–—-]\s', line)) or
            (
                bool(re.match(r'^[A-Z][a-z\xc0-\xff]+[,.]', line)) and
                bool(current) and
                len(' '.join(current)) > 60
            )
        )
        if is_new and current:
            blocks.append(' '.join(current))
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append(' '.join(current))
    return [b for b in blocks if b.strip()]


def _extract_year(text: str) -> int | None:
    m = YEAR_RE.search(text)
    return int(m.group(1)) if m else None


def _strip_year_prefix(s: str) -> str:
    """Remove a leading year like '2010. ' or '2010, '."""
    return re.sub(r'^(?:19|20)\d{2}[.,]\s*', '', s)


def _extract_title(text: str) -> str:
    # Strip leading number / bullet
    t = re.sub(r'^\d{1,3}[.)]\s+', '', text.strip())
    t = re.sub(r'^[•–—-]\s+', '', t)

    # 1. Quoted title (articles / chapters)
    q = _QUOTED_TITLE_RE.search(t)
    if q:
        return q.group(0).strip(_QUOTE + '"').strip()

    # 2. APA style: "Author(s). (YEAR). Title. Publisher."
    m = re.search(r'\(\s*(?:19|20)\d{2}\s*\)[.,]?\s+([A-Z][^.]{8,150})\.', t)
    if m:
        candidate = m.group(1).strip()
        if not _PLACE_RE.match(candidate) and not _PUBLISHER_RE.search(candidate):
            return candidate

    # 3. Chicago style: "Lastname, First[ M.]. YEAR. Title." or
    #    "Lastname, First. Title. Place: Publisher, YEAR."
    # Strategy: if the first period is within 60 chars (likely an author segment),
    # take text after it, strip a leading year, then grab text up to next period.
    first_dot = t.find('. ')
    if 4 < first_dot < 65:
        after_first = _strip_year_prefix(t[first_dot + 2:].strip())
        m3 = re.match(r'([A-ZÀ-ÖØ-Ý][^.]{8,150})\.', after_first)
        if m3:
            candidate = m3.group(1).strip()
            if (len(candidate) >= 8
                    and not _PLACE_RE.match(candidate)
                    and not _PUBLISHER_RE.search(candidate)
                    and not re.match(r'^[A-Z][a-z]+,\s+[A-Z]', candidate)):
                return candidate

    # 4. Year-then-title: "Author YEAR Title (Publisher, YEAR)."
    m4 = re.search(r'\b(?:19|20)\d{2}\b[.,]?\s+([A-Z][^.]{8,150})\.', t)
    if m4:
        candidate = m4.group(1).strip()
        if not _PLACE_RE.match(candidate) and not _PUBLISHER_RE.search(candidate):
            return candidate

    # 5. Strip short author prefix, look for title before opening paren
    no_author = re.sub(
        r'^[A-Z][a-z\xc0-\xff]+(?:,\s*[A-Z][a-z\xc0-\xff]*\.?)+[.,]\s*', '', t
    )
    if no_author != t:
        no_author = _strip_year_prefix(no_author)
        m5 = re.match(r'([A-Z][^(]{8,150})\s*\(', no_author)
        if m5:
            candidate = m5.group(1).strip().rstrip('.,')
            if len(candidate) >= 8 and not _PLACE_RE.match(candidate):
                return candidate

    # 6. Last resort on author-stripped text: up to first period / colon
    after = no_author if no_author != t else t
    m6 = re.match(r'([A-Z][^.:]{10,150})[.:]', after)
    if m6:
        candidate = m6.group(1).strip().rstrip('.,')
        if (len(candidate) >= 8
                and not _PLACE_RE.match(candidate)
                and not re.match(r'^[A-Z][a-z]+,\s+[A-Z]', candidate)):
            return candidate

    return ''


def _extract_authors(text: str, title: str) -> list[str]:
    t = re.sub(r'^\d{1,3}[.)]\s+', '', text.strip())
    t = re.sub(r'^[•–—-]\s+', '', t)

    before = t
    if title and len(title) >= 8:
        idx = t.find(title)
        if idx > 0:
            before = t[:idx].strip().rstrip('.,')
        else:
            m = YEAR_RE.search(t)
            before = t[:m.start()].strip().rstrip('.,') if m else ''
    else:
        m = YEAR_RE.search(t)
        before = t[:m.start()].strip().rstrip('.,') if m else ''

    if not before:
        return []

    parts = re.split(r'\s*[;&]\s*|\s+and\s+|\s*,\s+(?=[A-Z])', before)
    authors: list[str] = []
    for p in parts:
        p = p.strip().rstrip('.,')
        if 3 < len(p) < 60 and re.search(r'[A-Z][a-z]', p):
            authors.append(p)
    return authors[:6]


def _classify_type(text: str) -> str:
    if _JOURNAL_SIGNAL_RE.search(text):
        return 'article'
    if _QUOTED_TITLE_RE.search(text):
        return 'article'
    if re.search(r'\bin\b.{0,80}\b(ed\.|eds\.|editors?\b|edited by)\b', text, re.I):
        return 'chapter'
    if _PUBLISHER_RE.search(text):
        return 'book'
    if re.search(r'\b(report|working paper|ipcc|united nations|government|ministry)\b',
                 text, re.I):
        return 'report'
    if re.search(r'https?://', text):
        return 'website'
    return 'book'


def _extract_publisher(text: str) -> str | None:
    m = _PUBLISHER_RE.search(text)
    return m.group(0) if m else None


def _extract_journal(text: str) -> str | None:
    tl = text.lower()
    for j in _KNOWN_JOURNALS:
        if j.lower() in tl:
            return j
    m = re.search(r',\s+([A-Z][^,]{3,60}),?\s+(?:vol\.?|volume|no\.?)\s*\d', text)
    return m.group(1).strip() if m else None


def _parse_block(raw: str) -> dict | None:
    """Extract structured fields from a single citation block."""
    text = ' '.join(raw.split())

    if len(text) < 35:
        return None
    if _is_noise(text):
        return None
    if _citation_score(text) < 2:
        return None

    title = _extract_title(text)
    if not title or len(title) < 8:
        return None
    if _FALSE_POSITIVE_TITLES.match(title):
        return None

    authors = _extract_authors(text, title)
    citation_type = _classify_type(text)
    publisher = _extract_publisher(text)
    journal = _extract_journal(text) if citation_type in ('article', 'chapter') else None

    return {
        'title':     title,
        'authors':   authors,
        'year':      _extract_year(text),
        'type':      citation_type,
        'publisher': publisher,
        'journal':   journal,
        'place':     None,
    }


# ── Normalisation & deduplication ─────────────────────────────────────────────

def normalise_title(title: str) -> str:
    if not title:
        return ''
    t = unicodedata.normalize('NFKD', title)
    t = t.encode('ascii', 'ignore').decode()
    t = t.lower()
    t = re.sub(r'^(the|a|an)\s+', '', t)   # remove leading articles
    # Strip generic "Chapter 3:" / "Part II:" style prefixes so the meaningful
    # part of the title survives. Without this, the subtitle rule below reduces
    # "Chapter 1: Introduction: Emergy and Real Wealth" to just "chapter 1",
    # which fuzzy-matches every other "chapter N" citation in the corpus
    # (SequenceMatcher("chapter 1", "chapter 5") = 0.89, above the 0.87
    # threshold) and merges genuinely different chapters into one work.
    t = re.sub(r'^(chapter|part|section|volume|vol|book|appendix)\s+[0-9ivxlc]+\s*[:.\-]\s*', '', t)
    # Only remove subtitle if the part before the colon is substantial (>=2 words)
    m = re.match(r'^(.+?)\s*:(.*)$', t)
    if m and len(m.group(1).split()) >= 2:
        t = m.group(1)
    t = re.sub(r'[^a-z0-9\s]', '', t)      # keep alphanum + space
    t = re.sub(r'\s+', ' ', t).strip()
    return t


# Organizations get cited as "authors" too (AIA, U.S. Department of Energy, …).
# Never reorder or initial-match these — only real personal names.
_ORG_KEYWORDS = {
    'university', 'department', 'institute', 'society', 'association',
    'committee', 'council', 'group', 'press', 'foundation', 'agency',
    'administration', 'organization', 'organisation', 'office', 'bureau',
    'commission', 'corporation', 'company', 'inc', 'llc', 'ltd', 'school',
    'college', 'center', 'centre', 'laboratory', 'union', 'coalition',
    'partnership', 'network', 'alliance', 'authority', 'board', 'trust',
    'fund', 'programme', 'program', 'habitat',
}

# "Last, First" or "Last, F." — only matches a single-word surname followed
# by a short, capitalized first-name/initials block, so it won't misfire on
# organization names that happen to contain a comma (e.g. "American Society
# of Heating, Refrigerating and Air-Conditioning Engineers").
_LASTFIRST_RE = re.compile(
    r"^([A-Za-z][A-Za-z'\-]+)\s*,\s*([A-Z][A-Za-z.\-]*(?:\s+[A-Z][A-Za-z.\-]*){0,2})$"
)


def _looks_organizational(name: str) -> bool:
    lower = name.lower()
    if any(kw in lower for kw in _ORG_KEYWORDS):
        return True
    letters = [c for c in name if c.isalpha()]
    # a short all-caps token (AIA, IPCC, ASHRAE, UN-Habitat) reads as an org/acronym
    if letters and all(c.isupper() for c in letters) and len(name.split()) <= 3:
        return True
    return False


def normalise_author(name: str) -> str:
    """Reorder 'Last, First' -> 'First Last'; leave everything else as-is."""
    name = (name or '').strip()
    if not name or _looks_organizational(name):
        return name
    m = _LASTFIRST_RE.match(name)
    if m:
        last, first = m.group(1), m.group(2)
        return f'{first} {last}'.strip()
    return name


def author_match_key(name: str) -> str:
    """Grouping key that treats 'Olgyay, Victor', 'Victor Olgyay', and
    'V. Olgyay' as the same author (surname + first-initial), while leaving
    organizational / single-token authors keyed on their full string."""
    norm = normalise_author(name)
    if not norm:
        return ''
    if _looks_organizational(norm):
        return norm.lower()
    tokens = norm.split()
    if len(tokens) < 2:
        return norm.lower()
    surname = tokens[-1].lower().strip('.,')
    first_initial = tokens[0][0].lower() if tokens[0] else ''
    return f'{surname}|{first_initial}'


# Journal/publisher names that show up under multiple guises depending on how
# a syllabus cites them (abbreviation, "The " prefix, sub-imprint name, …).
# Keyed on the lowercased/stripped raw value, mapped to the canonical display
# name so charts and vote-based dedup treat every variant as one entity.
_JOURNAL_ALIASES = {
    'jae': 'Journal of Architectural Education',
    'places': 'Places Journal',
    'e-flux architecture': 'e-flux',
}

_PUBLISHER_ALIASES = {
    'the mit press': 'MIT Press',
}


def canonicalize_journal(name: str | None) -> str | None:
    if not name:
        return name
    return _JOURNAL_ALIASES.get(name.strip().lower(), name.strip())


def canonicalize_publisher(name: str | None) -> str | None:
    if not name:
        return name
    return _PUBLISHER_ALIASES.get(name.strip().lower(), name.strip())


# Titles known to fuzzy-match a genuinely different work above the 0.87
# ratio threshold purely because one title is a short-title-plus-one-word
# variant of the other (e.g. "Design with Nature Now", a 2019 tribute
# anthology with four unrelated authors, scores 0.90 against "Design with
# Nature", McHarg's 1969 original — well past the threshold, and too close
# to the ratio for a legitimate same-work variant like "Design with Nature
# (1969)" (0.878) to fix with a general rule). Titles listed here require an
# exact normalised-title match instead of fuzzy matching. A broader review
# of the matching threshold/approach itself is a separate, deferred task.
_EXACT_MATCH_ONLY_TITLES = {'design with nature now'}

# Chapter titles so generic that identical wording says nothing about identity:
# every book has an "Introduction". These get the first author's match key
# appended to their dedup key so intros of different books never merge.
_GENERIC_TITLES = {
    'introduction', 'preface', 'foreword', 'conclusion', 'epilogue',
    'prologue', 'afterword', 'overview', 'summary', 'interview',
}


def deduplicate(all_citations: list[dict]) -> list[dict]:
    """
    Cluster citations that refer to the same work.
    Uses a prefix-bucket approach for O(n) average-case performance.
    Returns one canonical record per work, aggregating per-year counts.
    """
    canonical: list[dict] = []
    # Map from 5-char normalised-title prefix → list of (canonical_idx, norm_title)
    buckets: dict[str, list[tuple[int, str]]] = {}

    def _find_match(norm: str) -> int:
        """Return index in canonical[] of a matching entry, or -1."""
        prefix5 = norm[:5]
        # Check exact-prefix bucket first, then neighbouring prefixes
        candidates: list[tuple[int, str]] = []
        for p in [prefix5, norm[:4], norm[:3]]:
            candidates.extend(buckets.get(p, []))

        best_ratio = 0.0
        best_idx   = -1
        seen = set()
        for idx, existing_norm in candidates:
            if idx in seen:
                continue
            seen.add(idx)
            if (norm in _EXACT_MATCH_ONLY_TITLES or existing_norm in _EXACT_MATCH_ONLY_TITLES) \
                    and norm != existing_norm:
                continue
            ratio = difflib.SequenceMatcher(None, norm, existing_norm).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_idx   = idx

        return best_idx if best_ratio >= 0.87 else -1

    for cit in tqdm(all_citations, desc="Deduplicating", unit="cit"):
        title = re.sub(r'^[•\-\*]\s*', '', cit.get('title') or '').strip()
        if len(title) < 8:
            continue
        norm = normalise_title(title)
        if not norm or len(norm) < 5:
            continue
        if norm in _GENERIC_TITLES:
            first_author = (cit.get('authors') or [''])[0] or ''
            norm = f'{norm} {author_match_key(first_author)}'.strip()

        match_idx = _find_match(norm)
        pdf_key   = cit.get('source_pdf', '')
        year_key  = str(cit.get('source_year', ''))

        if match_idx >= 0:
            existing = canonical[match_idx]
            # Count each unique syllabus at most once per canonical work
            yr_seen = existing.setdefault('_seen', set())
            if pdf_key not in yr_seen:
                yr_seen.add(pdf_key)
                if year_key:
                    existing['year_counts'][year_key] = \
                        existing['year_counts'].get(year_key, 0) + 1
                if pdf_key:
                    existing['source_syllabi'].append(pdf_key)
            if len(title) > len(existing.get('title', '')):
                existing['title'] = title
            incoming_authors = [normalise_author(a) for a in (cit.get('authors') or [])]
            existing_authors = existing.get('authors') or []
            if incoming_authors and (
                len(incoming_authors) > len(existing_authors)
                or (len(incoming_authors) == len(existing_authors)
                    and sum(len(a) for a in incoming_authors) > sum(len(a) for a in existing_authors))
            ):
                existing['authors'] = incoming_authors
            if not existing.get('year') and cit.get('year'):
                existing['year'] = cit['year']
            if not existing.get('publisher') and cit.get('publisher'):
                existing['publisher'] = cit['publisher']
            # A single mis-extracted duplicate (e.g. one syllabus's chapter/
            # container mixup) shouldn't override what most citations of the
            # same work agree on — so type/journal are decided by plurality
            # vote across all duplicates, not "whichever came first".
            # 'journal' None is a valid vote too, so a majority of citations
            # with no journal correctly outvotes a minority with a wrong one.
            existing['_type_votes'][cit.get('type', 'other')] += 1
            existing['type'] = existing['_type_votes'].most_common(1)[0][0]
            existing['_journal_votes'][cit.get('journal')] += 1
            existing['journal'] = existing['_journal_votes'].most_common(1)[0][0]
        else:
            new_idx = len(canonical)
            new_rec = {
                'title':          title,
                'authors':        [normalise_author(a) for a in (cit.get('authors') or [])],
                'year':           cit.get('year'),
                'type':           cit.get('type', 'other'),
                'publisher':      cit.get('publisher'),
                'journal':        cit.get('journal'),
                'year_counts':    {year_key: 1} if year_key else {},
                'source_syllabi': [pdf_key] if pdf_key else [],
                '_seen':          {pdf_key} if pdf_key else set(),
                '_type_votes':    Counter({cit.get('type', 'other'): 1}),
                '_journal_votes': Counter({cit.get('journal'): 1}),
                'isbn':           None,
                'genre':          None,
                'description':    None,
            }
            canonical.append(new_rec)
            for p in [norm[:5], norm[:4], norm[:3]]:
                if p:
                    buckets.setdefault(p, []).append((new_idx, norm))

    # Remove internal tracking fields
    for rec in canonical:
        rec.pop('_seen', None)
        rec.pop('_type_votes', None)
        rec.pop('_journal_votes', None)

    return canonical


# ── Open Library enrichment ───────────────────────────────────────────────────

OL_SEARCH = "https://openlibrary.org/search.json"
OL_WORKS  = "https://openlibrary.org/works/{key}.json"
OL_DELAY  = 0.25  # seconds between requests


def load_ol_cache() -> dict:
    if OL_CACHE.exists():
        return json.loads(OL_CACHE.read_text())
    return {}


def save_ol_cache(cache: dict) -> None:
    OL_CACHE.write_text(json.dumps(cache, indent=2, ensure_ascii=False))


def ol_lookup(title: str, authors: list[str], cache: dict) -> dict:
    """Query Open Library for a work and return {isbn, genre, description}."""
    key = normalise_title(title)
    if key in cache:
        return cache[key]

    result = {'isbn': None, 'genre': None, 'description': None}

    try:
        author_q = authors[0] if authors else ''
        params = {
            'title':  title[:100],
            'author': author_q[:60],
            'limit':  1,
            'fields': 'key,title,author_name,first_publish_year,isbn,subject,ia_loaded_id',
        }
        r = requests.get(OL_SEARCH, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        docs = data.get('docs', [])
        if docs:
            doc = docs[0]
            # ISBN: take first ISBN-13 if available, else ISBN-10
            isbns = doc.get('isbn', [])
            isbn13 = [i for i in isbns if len(i) == 13]
            isbn10 = [i for i in isbns if len(i) == 10]
            result['isbn'] = (isbn13 or isbn10 or [None])[0]

            # Subjects as genre
            subjects = doc.get('subject', [])[:5]
            result['genre'] = '; '.join(subjects) if subjects else None

            # Fetch description from works API if we have a key
            work_key = doc.get('key', '')
            if work_key:
                try:
                    wr = requests.get(
                        f"https://openlibrary.org{work_key}.json", timeout=10
                    )
                    if wr.ok:
                        wdata = wr.json()
                        desc = wdata.get('description', '')
                        if isinstance(desc, dict):
                            desc = desc.get('value', '')
                        if desc:
                            desc = re.sub(r'\s+', ' ', desc).strip()
                        result['description'] = desc[:400] if desc else None
                except Exception:
                    pass
                time.sleep(OL_DELAY)

    except Exception as exc:
        tqdm.write(f"  [WARN] Open Library lookup failed for '{title[:40]}': {exc}")
        return result  # network/connection failure — don't cache, retry on next run

    cache[key] = result
    return result


def enrich_with_open_library(canonical: list[dict]) -> None:
    """Mutates each record in-place, adding isbn / genre / description."""
    cache = load_ol_cache()

    # Only query books and chapters (articles are usually journals)
    to_enrich = [r for r in canonical
                 if r.get('type') in ('book', 'chapter', 'other') and not r.get('isbn')]

    n_new_lookups = 0
    for rec in tqdm(to_enrich, desc="Open Library lookup", unit="work"):
        was_cached = normalise_title(rec['title']) in cache
        result = ol_lookup(rec['title'], rec.get('authors', []), cache)
        rec['isbn']        = result['isbn']
        rec['genre']       = result['genre']
        rec['description'] = result['description']
        # ol_lookup() already rate-limits and caches its own network calls;
        # a cache hit needs neither a delay nor a redundant full-cache write.
        if not was_cached:
            n_new_lookups += 1
            save_ol_cache(cache)

    save_ol_cache(cache)
    print(f"Open Library: {n_new_lookups} new lookup(s), "
          f"{len(to_enrich) - n_new_lookups} served from cache")


# ── Output ─────────────────────────────────────────────────────────────────────

def split_journal_and_container(canonical: list[dict]) -> None:
    """Citation extraction had no separate field for a book chapter's
    containing volume, so it reused 'journal' for that too (e.g. a chapter
    from "Climates: Architecture and the Planetary Imaginary" shows that
    book's title as its 'journal'). Split it into 'container_title' so
    'journal' only ever holds a genuine periodical name."""
    for rec in canonical:
        if rec.get('type') == 'chapter' and rec.get('journal'):
            rec['container_title'] = rec['journal']
            rec['journal'] = None
        else:
            rec.setdefault('container_title', None)


def write_bibliography_csv(canonical: list[dict]) -> None:
    year_cols = [str(y) for y in YEARS]
    fieldnames = [
        'title', 'authors', 'year_published', 'type',
        'genre', 'description', 'publisher', 'journal', 'container_title', 'isbn',
        'total_citations', *year_cols, 'source_syllabi',
    ]

    rows = []
    for rec in canonical:
        authors_str = '; '.join(rec.get('authors') or [])
        total = sum(rec.get('year_counts', {}).values())
        yr_vals = {y: rec.get('year_counts', {}).get(y, 0) for y in year_cols}
        source = '; '.join(filter(None, rec.get('source_syllabi', [])))
        description = re.sub(r'\s+', ' ', rec.get('description') or '').strip()

        row = {
            'title':           rec.get('title', ''),
            'authors':         authors_str,
            'year_published':  rec.get('year', ''),
            'type':            rec.get('type', ''),
            'genre':           rec.get('genre', ''),
            'description':     description,
            'publisher':       rec.get('publisher', '') or '',
            'journal':         rec.get('journal', '') or '',
            'container_title': rec.get('container_title', '') or '',
            'isbn':            rec.get('isbn', '') or '',
            'total_citations': total,
            **yr_vals,
            'source_syllabi':  source,
        }
        rows.append(row)

    # Sort by total citations desc, then title
    rows.sort(key=lambda r: (-r['total_citations'], r['title'].lower()))

    with BIB_CSV.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Bibliography written to {BIB_CSV}  ({len(rows)} unique works)")


def write_log_csv(log_rows: list[dict]) -> None:
    if not log_rows:
        return
    with LOG_CSV.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=log_rows[0].keys(), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Extraction log written to {LOG_CSV}")


def write_summary_stats(all_raw: list[dict], canonical: list[dict]) -> None:
    """Dataset-wide counts and distribution stats, written as metric/value rows."""
    per_syllabus = [len(e.get('citations', [])) for e in all_raw]
    n_syllabi = len(all_raw)
    with_citations = [n for n in per_syllabus if n > 0]
    total_raw = sum(per_syllabus)
    total_unique = len(canonical)

    syllabi_counts = [len(rec.get('source_syllabi', [])) for rec in canonical]
    shared_works = sum(1 for c in syllabi_counts if c > 1)
    single_syllabus_works = sum(1 for c in syllabi_counts if c == 1)

    most_cited = max(
        canonical,
        key=lambda r: sum(r.get('year_counts', {}).values()),
        default=None,
    )
    max_syllabus_idx = per_syllabus.index(max(per_syllabus)) if per_syllabus else None

    pub_years = [rec['year'] for rec in canonical if rec.get('year')]
    syllabus_years = [e['year'] for e in all_raw if e.get('year')]

    has_isbn = sum(1 for rec in canonical if rec.get('isbn'))
    has_genre = sum(1 for rec in canonical if rec.get('genre'))

    publisher_counts: dict[str, int] = defaultdict(int)
    for rec in canonical:
        publisher = (rec.get('publisher') or '').strip()
        if publisher:
            publisher_counts[publisher] += sum(rec.get('year_counts', {}).values())
    top_publisher = max(publisher_counts.items(), key=lambda kv: kv[1], default=None)

    journal_counts: dict[str, int] = defaultdict(int)
    articles = [rec for rec in canonical if rec.get('type') == 'article']
    for rec in articles:
        journal = (rec.get('journal') or '').strip()
        if journal:
            journal_counts[journal] += sum(rec.get('year_counts', {}).values())
    top_journal = max(journal_counts.items(), key=lambda kv: kv[1], default=None)

    rows = [
        ('total_syllabi_processed', n_syllabi),
        ('syllabi_with_citations_extracted', len(with_citations)),
        ('syllabi_with_zero_citations', n_syllabi - len(with_citations)),
        ('total_raw_citations', total_raw),
        ('total_unique_works_after_dedup', total_unique),
        ('dedup_reduction_pct', round(100 * (1 - total_unique / total_raw), 1) if total_raw else 0),
        ('avg_citations_per_syllabus_all', round(total_raw / n_syllabi, 1) if n_syllabi else 0),
        ('avg_citations_per_syllabus_with_bib', round(total_raw / len(with_citations), 1) if with_citations else 0),
        ('median_citations_per_syllabus', statistics.median(per_syllabus) if per_syllabus else 0),
        ('min_citations_per_syllabus', min(per_syllabus) if per_syllabus else 0),
        ('max_citations_per_syllabus', max(per_syllabus) if per_syllabus else 0),
        ('syllabus_with_max_citations', all_raw[max_syllabus_idx]['pdf'] if max_syllabus_idx is not None else ''),
        ('avg_syllabi_citing_each_unique_work', round(sum(syllabi_counts) / total_unique, 2) if total_unique else 0),
        ('works_cited_by_multiple_syllabi', shared_works),
        ('works_cited_by_single_syllabus', single_syllabus_works),
        ('most_cited_work_title', most_cited['title'] if most_cited else ''),
        ('most_cited_work_citation_count', sum(most_cited.get('year_counts', {}).values()) if most_cited else 0),
        ('earliest_cited_work_year', min(pub_years) if pub_years else ''),
        ('latest_cited_work_year', max(pub_years) if pub_years else ''),
        ('earliest_syllabus_year_processed', min(syllabus_years) if syllabus_years else ''),
        ('latest_syllabus_year_processed', max(syllabus_years) if syllabus_years else ''),
        ('pct_unique_works_with_open_library_isbn', round(100 * has_isbn / total_unique, 1) if total_unique else 0),
        ('pct_unique_works_with_open_library_genre', round(100 * has_genre / total_unique, 1) if total_unique else 0),
        ('unique_publishers_count', len(publisher_counts)),
        ('pct_unique_works_with_publisher', round(100 * sum(1 for r in canonical if (r.get('publisher') or '').strip()) / total_unique, 1) if total_unique else 0),
        ('most_cited_publisher', top_publisher[0] if top_publisher else ''),
        ('most_cited_publisher_citation_count', top_publisher[1] if top_publisher else 0),
        ('unique_journals_count', len(journal_counts)),
        ('pct_articles_with_journal', round(100 * sum(1 for r in articles if (r.get('journal') or '').strip()) / len(articles), 1) if articles else 0),
        ('most_cited_journal', top_journal[0] if top_journal else ''),
        ('most_cited_journal_citation_count', top_journal[1] if top_journal else 0),
    ]

    with STATS_CSV.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(['metric', 'value'])
        writer.writerows(rows)
    print(f"Summary stats written to {STATS_CSV}")


# ── Keyword & go-word matching ────────────────────────────────────────────────

def parse_keywords_file(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """Parse keywords.txt — comma-separated aliases per line."""
    groups = []
    if not path.exists():
        print(f"  [WARN] {path} not found")
        return groups
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            aliases = [a.strip().lower() for a in line.split(',') if a.strip()]
            if aliases:
                groups.append((aliases[0], aliases))
    return groups


def parse_go_words_file(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """Parse go-words.txt — tab-separated: term TAB comma-separated aliases."""
    groups = []
    if not path.exists():
        print(f"  [WARN] {path} not found")
        return groups
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        term_part, _, aliases_str = line.partition('\t')
        term = term_part.strip().lower()
        if not term:
            continue
        aliases = [a.strip().lower() for a in aliases_str.split(',') if a.strip()]
        groups.append((term, [term] + aliases))
    return groups


def match_terms_in_title(title: str,
                          groups: list[tuple[str, list[str]]]) -> list[str]:
    """Return canonical labels whose aliases appear in the title."""
    title_lower = title.lower()
    matched = []
    for label, aliases in groups:
        for alias in aliases:
            # phrase / hyphenated: substring match; single word: word-boundary match
            if ' ' in alias or '-' in alias:
                if alias in title_lower:
                    matched.append(label)
                    break
            else:
                if re.search(r'(?<![a-z])' + re.escape(alias) + r'(?![a-z])',
                             title_lower):
                    matched.append(label)
                    break
    return matched


def _write_term_heatmap(year_data: dict[str, dict[int, int]],
                         term_list: list[str],
                         title: str,
                         outpath: pathlib.Path,
                         label_color: str = '#333333') -> None:
    present = [t for t in term_list if any(year_data[t].values())]
    if not present:
        return
    data = np.array([[year_data[t].get(yr, 0) for yr in YEARS]
                     for t in present], dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(8, len(YEARS) * 0.8), max(4, len(present) * 0.4)))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels([str(y) for y in YEARS], fontsize=10)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=9, color=label_color)
    vmax = data.max() or 1
    for yi in range(len(present)):
        for xi in range(len(YEARS)):
            val = data[yi, xi]
            if val > 0:
                ax.text(xi, yi, str(int(val)), ha='center', va='center',
                        fontsize=7,
                        color='black' if val < vmax * 0.6 else 'white')
    plt.colorbar(im, ax=ax, label='Citations to matching entries')
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  → {outpath.name}")


def write_keyword_outputs(canonical: list[dict],
                           kw_groups: list[tuple[str, list[str]]],
                           gw_groups: list[tuple[str, list[str]]]) -> None:
    KW_OUT_DIR.mkdir(exist_ok=True)

    # Annotate each entry
    for rec in canonical:
        title = rec.get('title', '')
        rec['matched_keywords'] = match_terms_in_title(title, kw_groups)
        rec['matched_go_words'] = match_terms_in_title(title, gw_groups)

    # Write annotated bibliography CSV
    year_cols = [str(y) for y in YEARS]
    bm_fields = [
        'title', 'authors', 'year_published', 'type',
        'total_citations', *year_cols,
        'matched_keywords', 'matched_go_words', 'source_syllabi',
    ]
    bm_rows = []
    for rec in canonical:
        total = sum(rec.get('year_counts', {}).values())
        yr_vals = {y: rec.get('year_counts', {}).get(y, 0) for y in year_cols}
        bm_rows.append({
            'title':            rec.get('title', ''),
            'authors':          '; '.join(rec.get('authors') or []),
            'year_published':   rec.get('year', ''),
            'type':             rec.get('type', ''),
            'total_citations':  total,
            **yr_vals,
            'matched_keywords': ', '.join(rec['matched_keywords']),
            'matched_go_words': ', '.join(rec['matched_go_words']),
            'source_syllabi':   '; '.join(filter(None, rec.get('source_syllabi', []))),
        })
    bm_rows.sort(key=lambda r: (-r['total_citations'], r['title'].lower()))
    bm_csv = KW_OUT_DIR / 'bibliography_matched.csv'
    with bm_csv.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=bm_fields, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(bm_rows)
    print(f"Keyword-matched bibliography → {bm_csv}  ({len(bm_rows)} rows)")

    # Accumulate per-year counts weighted by citation count
    kw_labels = [label for label, _ in kw_groups]
    gw_labels = [label for label, _ in gw_groups]
    kw_year: dict[str, dict[int, int]] = {lb: {yr: 0 for yr in YEARS}
                                           for lb in kw_labels}
    gw_year: dict[str, dict[int, int]] = {lb: {yr: 0 for yr in YEARS}
                                           for lb in gw_labels}
    for rec in canonical:
        for yr in YEARS:
            cnt = rec.get('year_counts', {}).get(str(yr), 0)
            if cnt:
                for lb in rec['matched_keywords']:
                    kw_year[lb][yr] += cnt
                for lb in rec['matched_go_words']:
                    gw_year[lb][yr] += cnt

    kw_total = {lb: sum(kw_year[lb].values()) for lb in kw_labels}
    gw_total = {lb: sum(gw_year[lb].values()) for lb in gw_labels}

    # Combined horizontal bar chart — keywords in red, go-words in grey
    print("Writing keyword/go-word charts …")
    kw_hits = [(lb, kw_total[lb]) for lb in kw_labels if kw_total[lb] > 0]
    gw_hits = [(lb, gw_total[lb]) for lb in gw_labels if gw_total[lb] > 0]
    combined = ([(lb, cnt, 'keyword') for lb, cnt in kw_hits] +
                [(lb, cnt, 'go_word') for lb, cnt in gw_hits])
    combined.sort(key=lambda x: -x[1])

    if combined:
        c_labels, c_counts, c_types = zip(*combined)
        c_colors = ['#c0392b' if t == 'keyword' else '#555555' for t in c_types]
        fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.28)))
        ax.barh(range(len(c_labels)), c_counts, color=c_colors, height=0.7)
        ax.set_yticks(range(len(c_labels)))
        ax.set_yticklabels(c_labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Weighted citation count')
        ax.set_title('Keywords and go-words in bibliography titles', fontsize=12)
        ax.legend(handles=[
            Patch(facecolor='#c0392b', label='Keywords (keywords.txt)'),
            Patch(facecolor='#555555', label='Go-words (go-words.txt)'),
        ], loc='lower right', fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis='both', length=0)
        plt.tight_layout()
        plt.savefig(KW_OUT_DIR / 'term_frequency.png', dpi=150)
        plt.close()
        print(f"  → term_frequency.png")

    # Per-year heatmaps
    _write_term_heatmap(
        kw_year, kw_labels,
        'Keywords in bibliography titles – citations per year',
        KW_OUT_DIR / 'keywords_by_year.png', label_color='#c0392b')
    gw_with_hits = [lb for lb in gw_labels if gw_total.get(lb, 0) > 0]
    if gw_with_hits:
        _write_term_heatmap(
            {lb: gw_year[lb] for lb in gw_with_hits},
            gw_with_hits,
            'Go-words in bibliography titles – citations per year',
            KW_OUT_DIR / 'go_words_by_year.png', label_color='#333333')


# ── Overview visualizations ───────────────────────────────────────────────────
#
# Dataset-wide charts (independent of keywords.txt/go-words.txt): most-cited
# titles/authors, a type breakdown, and a publication-year histogram.

OVERVIEW_OUT_DIR = OUT_DIR / "overview"

_BAR_COLOR = '#0a5c6b'  # single-series magnitude charts: one hue, not a rainbow


def _bare_bar_chart(labels: list[str], counts: list[int], title: str,
                     xlabel: str, outpath: pathlib.Path,
                     label_fontsize: int = 9) -> None:
    """Shared styling for the single-hue horizontal bar charts below."""
    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.35)))
    ax.barh(range(len(labels)), counts, color=_BAR_COLOR, height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=label_fontsize)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  → {outpath.name}")


def write_top_titles_chart(canonical: list[dict], outpath: pathlib.Path,
                            top_n: int = 20) -> None:
    ranked = sorted(canonical, key=lambda r: -sum(r.get('year_counts', {}).values()))[:top_n]
    ranked = [r for r in ranked if sum(r.get('year_counts', {}).values()) > 0]
    if not ranked:
        return

    labels, counts = [], []
    for rec in ranked:
        author = (rec.get('authors') or [''])[0].split(',')[0]
        year = rec.get('year') or ''
        label = rec.get('title', '')[:55]
        if author:
            label += f" — {author}"
        if year:
            label += f" ({year})"
        labels.append(label)
        counts.append(sum(rec.get('year_counts', {}).values()))

    _bare_bar_chart(labels, counts, f'Most cited titles (top {len(labels)})',
                     'Citations across syllabi', outpath, label_fontsize=8)


# Shorthand tokens that show up in the `authors` field of a citation but aren't
# an author name — normalized (lowercase, periods stripped) before comparison.
_NOT_AN_AUTHOR = {'et al', 'eds', 'ed', 'editors', 'anonymous'}


def write_top_authors_chart(canonical: list[dict], outpath: pathlib.Path,
                             top_n: int = 20) -> None:
    author_counts: dict[str, int] = defaultdict(int)
    author_display: dict[str, str] = {}
    for rec in canonical:
        total = sum(rec.get('year_counts', {}).values())
        if not total:
            continue
        for a in rec.get('authors') or []:
            a = normalise_author(a.strip())
            if not a or a.lower().replace('.', '').strip() in _NOT_AN_AUTHOR:
                continue
            key = author_match_key(a)
            if not key:
                continue
            author_counts[key] += total
            if key not in author_display or len(a) > len(author_display[key]):
                author_display[key] = a

    ranked = sorted(author_counts.items(), key=lambda kv: -kv[1])[:top_n]
    if not ranked:
        return
    labels = [author_display[k] for k, _ in ranked]
    counts = [v for _, v in ranked]

    _bare_bar_chart(labels, counts, f'Most cited authors (top {len(labels)})',
                     'Citations across syllabi (co-authored works count for each author)',
                     outpath)


def write_citation_types_chart(canonical: list[dict], outpath: pathlib.Path) -> None:
    type_counts: dict[str, int] = defaultdict(int)
    for rec in canonical:
        type_counts[rec.get('type') or 'other'] += sum(rec.get('year_counts', {}).values())
    ranked = sorted(type_counts.items(), key=lambda kv: -kv[1])
    ranked = [(t, c) for t, c in ranked if c > 0]
    if not ranked:
        return
    labels = [t.capitalize() for t, _ in ranked]
    counts = [c for _, c in ranked]

    _bare_bar_chart(labels, counts, 'Cited works by type', 'Citations across syllabi',
                     outpath, label_fontsize=10)


def write_top_publishers_chart(canonical: list[dict], outpath: pathlib.Path,
                                top_n: int = 20) -> None:
    counts: dict[str, int] = defaultdict(int)
    for rec in canonical:
        publisher = (rec.get('publisher') or '').strip()
        if not publisher:
            continue
        counts[publisher] += sum(rec.get('year_counts', {}).values())

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    ranked = [(p, c) for p, c in ranked if c > 0]
    if not ranked:
        return
    labels = [p for p, _ in ranked]
    values = [c for _, c in ranked]

    _bare_bar_chart(labels, values, f'Most cited publishers (top {len(labels)})',
                     'Citations across syllabi', outpath, label_fontsize=9)


def write_top_journals_chart(canonical: list[dict], outpath: pathlib.Path,
                              top_n: int = 20) -> None:
    """Only counts records typed 'article' — 'journal' on a 'chapter' record
    holds its containing book title (see split_journal_and_container), not a
    periodical, so it's excluded here to avoid mixing the two."""
    counts: dict[str, int] = defaultdict(int)
    for rec in canonical:
        if rec.get('type') != 'article':
            continue
        journal = (rec.get('journal') or '').strip()
        if not journal:
            continue
        counts[journal] += sum(rec.get('year_counts', {}).values())

    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    ranked = [(j, c) for j, c in ranked if c > 0]
    if not ranked:
        return
    labels = [j for j, _ in ranked]
    values = [c for _, c in ranked]

    _bare_bar_chart(labels, values, f'Most cited journals (top {len(labels)})',
                     'Citations across syllabi', outpath, label_fontsize=9)


def write_publication_year_chart(canonical: list[dict], outpath: pathlib.Path) -> None:
    years: list[int] = []
    pre1950 = 0
    for rec in canonical:
        y = rec.get('year')
        total = sum(rec.get('year_counts', {}).values())
        if not y or not total:
            continue
        if y < 1950:
            pre1950 += total
        elif y <= YEARS[-1]:
            years.extend([y] * total)
    if not years and not pre1950:
        return

    bins = list(range(1950, 2031, 5))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.hist(years, bins=bins, color=_BAR_COLOR, edgecolor='white', linewidth=0.5)
    if pre1950:
        ax.text(0.01, 0.95, f'+ {pre1950} citation(s) to works published before 1950',
                transform=ax.transAxes, fontsize=8, color='#52514e', va='top')
    ax.set_xticks(bins)
    ax.set_xticklabels([str(b) for b in bins], rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Original publication year of cited work (5-year bins)')
    ax.set_ylabel('Citations across syllabi')
    ax.set_title('When were the cited works originally published?', fontsize=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  → {outpath.name}")


def write_syllabi_per_genre_chart(canonical: list[dict], outpath: pathlib.Path,
                                   top_n: int = 15) -> None:
    """Genre/subject tags come from Open Library ('genre' is a '; '-joined
    list of subjects on the matched work). Counts distinct syllabi that cite
    at least one work carrying that tag — not citation counts, since a
    heavily-cited work would otherwise dominate every genre it touches."""
    genre_syllabi: dict[str, set[str]] = defaultdict(set)
    for rec in canonical:
        genre = rec.get('genre')
        if not genre:
            continue
        syllabi = rec.get('source_syllabi', [])
        if not syllabi:
            continue
        for tag in genre.split(';'):
            tag = tag.strip()
            if tag:
                genre_syllabi[tag].update(syllabi)

    if not genre_syllabi:
        return

    ranked = sorted(genre_syllabi.items(), key=lambda kv: -len(kv[1]))[:top_n]
    labels = [tag for tag, _ in ranked]
    counts = [len(syllabi) for _, syllabi in ranked]

    n_matched = sum(1 for rec in canonical if rec.get('genre'))
    coverage_pct = round(100 * n_matched / len(canonical), 1) if canonical else 0

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.35)))
    ax.barh(range(len(labels)), counts, color=_BAR_COLOR, height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Distinct syllabi citing a work tagged with this genre')
    ax.set_title(
        f'Syllabi by cited-work genre (top {len(labels)})\n'
        f'Open Library genre coverage: {coverage_pct}% of unique works — treat as a partial view',
        fontsize=11,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  → {outpath.name}")


def write_syllabi_distribution_chart(canonical: list[dict], outpath: pathlib.Path) -> None:
    """Histogram: x = number of syllabi citing a single work, y = how many
    works have that many citing syllabi."""
    if not canonical:
        return
    syllabi_counts = [len(rec.get('source_syllabi', [])) for rec in canonical]
    dist = Counter(syllabi_counts)
    xs = sorted(dist)
    ys = [dist[x] for x in xs]
    avg = sum(syllabi_counts) / len(syllabi_counts)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar([str(x) for x in xs], ys, color=_BAR_COLOR, width=0.7)
    for x, y in zip(xs, ys):
        ax.text(str(x), y + max(ys) * 0.01, str(y), ha='center', va='bottom', fontsize=8, color='#52514e')
    ax.set_xlabel('Number of syllabi citing a single work')
    ax.set_ylabel('Number of works')
    ax.set_title(
        f'How many syllabi cite each work?\n'
        f'Average: {avg:.2f} syllabi per work, across {len(canonical)} unique works',
        fontsize=12,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.grid(axis='y', color='#e1e0d9', linewidth=0.6, which='major')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  → {outpath.name}")


def write_overview_charts(canonical: list[dict]) -> None:
    OVERVIEW_OUT_DIR.mkdir(exist_ok=True)
    print("Writing overview charts …")
    write_top_titles_chart(canonical, OVERVIEW_OUT_DIR / 'top_20_titles.png')
    write_top_authors_chart(canonical, OVERVIEW_OUT_DIR / 'top_20_authors.png')
    write_citation_types_chart(canonical, OVERVIEW_OUT_DIR / 'citation_types.png')
    write_publication_year_chart(canonical, OVERVIEW_OUT_DIR / 'publication_years.png')
    write_syllabi_per_genre_chart(canonical, OVERVIEW_OUT_DIR / 'syllabi_per_genre.png')
    write_top_publishers_chart(canonical, OVERVIEW_OUT_DIR / 'top_20_publishers.png')
    write_top_journals_chart(canonical, OVERVIEW_OUT_DIR / 'top_20_journals.png')
    write_syllabi_distribution_chart(canonical, OVERVIEW_OUT_DIR / 'syllabi_per_text_distribution.png')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract bibliography citations from syllabi.")
    parser.add_argument('--no-llm', action='store_true',
                        help='Use regex parser instead of Claude API')
    parser.add_argument('--limit', type=int, default=None,
                        help='Process only first N PDFs')
    parser.add_argument('--skip-ol', action='store_true',
                        help='Skip Open Library enrichment')
    parser.add_argument('--reparse', action='store_true',
                        help='Re-parse PDFs even if raw_citations.json exists')
    args = parser.parse_args()

    use_llm = not args.no_llm
    if use_llm and not os.environ.get('ANTHROPIC_API_KEY'):
        print("[WARN] ANTHROPIC_API_KEY not set — falling back to regex parser.")
        use_llm = False

    # ── Load or build raw citations ────────────────────────────────────────────
    if RAW_JSON.exists() and not args.reparse:
        print(f"Loading existing raw citations from {RAW_JSON} (use --reparse to redo)")
        all_raw = json.loads(RAW_JSON.read_text())
    else:
        all_raw = _run_extraction(use_llm, args.limit)
        RAW_JSON.write_text(json.dumps(all_raw, indent=2, ensure_ascii=False))
        print(f"Raw citations saved to {RAW_JSON}")

    # ── Flatten and annotate with source info ─────────────────────────────────
    flat: list[dict] = []
    for entry in all_raw:
        pdf_stem  = entry['pdf']
        year      = entry['year']
        method    = entry['method']
        for cit in entry.get('citations', []):
            cit['source_pdf']  = pdf_stem
            cit['source_year'] = year
            cit['journal']     = canonicalize_journal(cit.get('journal'))
            cit['publisher']   = canonicalize_publisher(cit.get('publisher'))
            flat.append(cit)

    print(f"Total raw citations across all syllabi: {len(flat)}")

    # ── Deduplicate ────────────────────────────────────────────────────────────
    canonical = deduplicate(flat)
    print(f"Unique works after deduplication: {len(canonical)}")

    # ── Enrich with Open Library ───────────────────────────────────────────────
    if not args.skip_ol:
        enrich_with_open_library(canonical)

    split_journal_and_container(canonical)

    # ── Write outputs ──────────────────────────────────────────────────────────
    write_bibliography_csv(canonical)

    kw_groups = parse_keywords_file(KEYWORDS_FILE)
    gw_groups = parse_go_words_file(GO_WORDS_FILE)
    if kw_groups or gw_groups:
        write_keyword_outputs(canonical, kw_groups, gw_groups)

    write_overview_charts(canonical)
    write_summary_stats(all_raw, canonical)

    # Log summary
    log_rows = [{
        'pdf':    e['pdf'],
        'year':   e['year'],
        'method': e['method'],
        'n_citations': len(e.get('citations', [])),
        'bib_chars':   e.get('bib_chars', 0),
    } for e in all_raw]
    write_log_csv(log_rows)

    total = sum(r['n_citations'] for r in log_rows)
    with_bib = sum(1 for r in log_rows if r['n_citations'] > 0)
    print(f"\nSummary: {len(all_raw)} syllabi processed, "
          f"{with_bib} had bibliography text, {total} raw citations extracted.")


def _run_extraction(use_llm: bool, limit: int | None) -> list[dict]:
    """Extract bibliography text and parse citations for all full PDFs."""
    full_pdfs = sorted(FULL_DIR.glob('*.pdf'))
    if limit:
        full_pdfs = full_pdfs[:limit]

    # Collect bibliography texts
    bib_texts: list[tuple[str, str]] = []  # (pdf_stem, bib_text)
    log: list[dict] = []
    for pdf_path in tqdm(full_pdfs, desc="Extracting bib text", unit="pdf"):
        year_m = re.match(r'^(\d{4})', pdf_path.name)
        year   = int(year_m.group(1)) if year_m else None

        nb_name = pdf_path.stem + 'nb.pdf'
        nb_path = NB_DIR / nb_name

        bib_text, method = get_bibliography_text(pdf_path, nb_path)
        bib_texts.append((pdf_path.stem, bib_text))
        log.append({
            'pdf':       pdf_path.stem,
            'year':      year,
            'method':    method,
            'bib_chars': len(bib_text),
        })

    # Parse citations
    if use_llm:
        print("Parsing citations with Claude API …")
        # Only pass non-empty texts to LLM
        nonempty = [(stem, txt) for stem, txt in bib_texts if txt.strip()]
        empty    = {stem for stem, txt in bib_texts if not txt.strip()}

        try:
            parsed_lists = parse_with_llm(nonempty)
        except RuntimeError as e:
            print(f"[ERROR] LLM parsing failed: {e}")
            print("Falling back to regex parser.")
            parsed_lists = [parse_with_regex(txt) for _, txt in nonempty]

        citation_map = dict(zip([s for s, _ in nonempty], parsed_lists))
        for s in empty:
            citation_map[s] = []
    else:
        print("Parsing citations with regex …")
        citation_map = {}
        for stem, bib_text in tqdm(bib_texts, desc="Regex parsing", unit="pdf"):
            citation_map[stem] = parse_with_regex(bib_text) if bib_text else []

    # Assemble result
    all_raw = []
    for entry in log:
        all_raw.append({
            **entry,
            'citations': citation_map.get(entry['pdf'], []),
        })
    return all_raw


if __name__ == '__main__':
    main()
