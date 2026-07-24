"""
Compares groups (3 or 4) of landmark environmental-design texts to see
whether the syllabi citing them overlap (cited together) or compete (cited
as alternatives), and what else each text tends to be co-cited with.

Each group's outputs (overlap_stats.csv, overlap_regions.png, and the
per-text co-citation charts) are written to their own subdirectory, named
after the group.

Usage:
    python3 compare_citations.py
"""

from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import pathlib
import textwrap
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE_DIR = pathlib.Path(__file__).parent.parent   # 3_bibliography_analysis/
OUT_DIR  = pathlib.Path(__file__).parent          # comparing_citations/

MAIN_SCRIPT = BASE_DIR / "3_bibliography_analysis.py"
RAW_JSON    = BASE_DIR / "outputs" / "raw_citations.json"

# Reuse the main pipeline's own dedup/title-normalisation logic so the
# canonical works found here match outputs/bibliography.csv exactly.
_spec = importlib.util.spec_from_file_location("biba", str(MAIN_SCRIPT))
biba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(biba)

_BAR_COLOR   = '#0a5c6b'                              # magnitude (matches main pipeline)
_CATEGORICAL = ['#2a78d6', '#1baf7a', '#eda100']       # identity: 3 validated categorical hues

GROUPS: dict[str, list[tuple[str, str]]] = {
    "classics": [
        ("Design with Nature",                                  "mcharg"),
        ("Design with Climate",                                 "olgyay"),
        ("The Architecture of the Well-Tempered Environment",   "banham"),
    ],
    "Energy": [
        ("Carbon Democracy: Political Power in the Age of Oil", "mitchell"),
        ("Petrochemical America",                               "misrach"),
        ("Design and the Green New Deal",                       "flemm"),
    ],
    "Theory": [
        ("Down to Earth: Politics in the New Climatic Regime",  "latour"),
        ("Vibrant Matter: A Political Ecology of Things",       "bennett"),
        ("Hyperobjects: Philosophy and Ecology After the End of the World", "morton"),
    ],
    "Multispecies": [
        ("Staying with the Trouble: Anthropocene, Capitalocene and Chthulucene", "haraway"),
        ("The Mushroom at the End of the World: On the Possibility of Life in Capitalist Ruins", "tsing"),
        ("Lo-TEK: Design by Radical Indigenism",                "watson"),
    ],
    "Urban": [
        ("The Death and Life of Great American Cities",         "jacobs"),
        ("Design with Nature",                                  "mcharg"),
        ("Ecological Urbanism",                                 "mostafavi"),
    ],
    "Temperature": [
        ("Thermal Delight in Architecture",                     "schong"),
        ("After Comfort: A User's Guide",                       "barber"),
        ("Modern Architecture and Climate: Design before Air Conditioning", "barber"),
    ],
    "MaterialEcologies": [
        ("Petrochemical America",                               "misrach"),
        ("Reciprocal Landscapes: material portraits in New York City and elsewhere", "hutton"),
        ("The Cannibal's Cookbook: Mining Myths of Cyclopean Construction", "clifford"),
    ],
}


def load_canonical() -> list[dict]:
    all_raw = json.loads(RAW_JSON.read_text())
    flat = []
    for entry in all_raw:
        for cit in entry.get('citations', []):
            cit['source_pdf']  = entry['pdf']
            cit['source_year'] = entry['year']
            flat.append(cit)
    return biba.deduplicate(flat)


def find_target(canonical: list[dict], label: str, author_hint: str) -> tuple[int, dict]:
    """Match a target work by title + author, excluding chapter-only
    citations like 'Some Essay (in Design with Nature)', and picking the
    most-cited match if more than one candidate qualifies."""
    label_norm = biba.normalise_title(label)
    candidates = []
    for i, rec in enumerate(canonical):
        title = rec.get('title') or ''
        if '(in ' in title.lower():
            continue
        norm = biba.normalise_title(title)
        if not norm:
            continue
        if label_norm in norm or norm in label_norm:
            authors_str = ' '.join(rec.get('authors') or []).lower()
            if author_hint in authors_str:
                candidates.append((i, rec))
    if not candidates:
        raise SystemExit(f"Could not find a canonical record for {label!r}")
    candidates.sort(key=lambda ir: -sum(ir[1].get('year_counts', {}).values()))
    return candidates[0]


def build_syllabus_index(canonical: list[dict]) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    for i, rec in enumerate(canonical):
        for pdf in rec.get('source_syllabi', []):
            idx.setdefault(pdf, []).append(i)
    return idx


def write_overlap_stats(labels: list[str], sets: dict[str, set], outpath: pathlib.Path) -> None:
    rows = []
    for label in labels:
        rows.append(('citing_syllabi_count', label, len(sets[label])))

    for a, b in itertools.combinations(labels, 2):
        inter = sets[a] & sets[b]
        union = sets[a] | sets[b]
        jaccard = len(inter) / len(union) if union else 0.0
        pct_a_also_b = len(inter) / len(sets[a]) if sets[a] else 0.0
        pct_b_also_a = len(inter) / len(sets[b]) if sets[b] else 0.0
        rows.append(('shared_syllabi_count', f'{a} & {b}', len(inter)))
        rows.append(('jaccard_similarity', f'{a} & {b}', round(jaccard, 3)))
        rows.append(('pct_of_A_that_also_cite_B', f'{a} -> {b}', round(100 * pct_a_also_b, 1)))
        rows.append(('pct_of_A_that_also_cite_B', f'{b} -> {a}', round(100 * pct_b_also_a, 1)))

    all_n = set.intersection(*sets.values()) if sets else set()
    rows.append((f'shared_by_all_{len(labels)}_count', ' & '.join(labels), len(all_n)))

    with outpath.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(['metric', 'comparison', 'value'])
        writer.writerows(rows)
    print(f"  -> {outpath.name}")


def format_authors(authors: list[str]) -> str:
    if not authors:
        return ''
    # Some source citations already spell out "et al." as a trailing
    # "author" (e.g. ['Daniel A. Barber', 'et al.']) rather than listing
    # every name; treat that the same as a long author list.
    named = [a for a in authors if a.strip().lower().rstrip('.') != 'et al']
    if len(named) < len(authors) or len(named) > 2:
        return f'{named[0]} et al.' if named else 'et al.'
    if len(named) == 1:
        return named[0]
    return f'{named[0]} & {named[1]}'


def _wrap_with_highlight(prefix: str, highlight: str, suffix: str, width: int) -> list[list[tuple[str, bool]]]:
    """Wrap prefix+highlight+suffix to `width` chars, returning per-line runs
    of (text, is_highlighted) so the highlighted substring can be colored
    even when word-wrapping splits it across lines."""
    full = prefix + highlight + suffix
    hl_start, hl_end = len(prefix), len(prefix) + len(highlight)
    lines, cursor = [], 0
    for line in textwrap.wrap(full, width):
        idx = full.find(line, cursor)
        if idx == -1:
            idx = cursor
        cursor = idx + len(line)
        seg_start = max(hl_start - idx, 0)
        seg_end = min(hl_end - idx, len(line))
        runs = []
        if seg_start > 0:
            runs.append((line[:seg_start], False))
        if seg_end > seg_start:
            runs.append((line[seg_start:seg_end], True))
        if seg_end < len(line):
            runs.append((line[seg_end:], False))
        lines.append(runs or [(line, False)])
    return lines


def _draw_centered_multicolor_title(fig, ax, lines: list[list[tuple[str, bool]]], fontsize: float,
                                     base_color: str, highlight_color: str) -> None:
    """Draw pre-wrapped (text, is_highlighted) runs as a centered title in
    the axes' top margin. matplotlib text can't mix colors in one string, so
    each line is laid out as separate left-to-right runs measured with the
    real renderer, then the whole line is centered over the axes."""
    from matplotlib.font_manager import FontProperties
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer)
    dpi = fig.dpi
    gap_px = 0.12 * dpi
    line_h_px = fontsize * dpi / 72 * 1.3

    n = len(lines)
    for i, runs in enumerate(lines):
        y_px = ax_bbox.ymax + gap_px + (n - 1 - i) * line_h_px + line_h_px * 0.5
        y_frac = (y_px - ax_bbox.y0) / ax_bbox.height

        widths_px = []
        for text, hl in runs:
            fp = FontProperties(size=fontsize, weight='bold' if hl else 'normal')
            w_px, _, _ = renderer.get_text_width_height_descent(text, fp, False)
            widths_px.append(w_px)
        x_px = (ax_bbox.x0 + ax_bbox.x1) / 2 - sum(widths_px) / 2
        for (text, hl), w_px in zip(runs, widths_px):
            x_frac = (x_px - ax_bbox.x0) / ax_bbox.width
            ax.text(x_frac, y_frac, text, transform=ax.transAxes, ha='left', va='center',
                    fontsize=fontsize, color=highlight_color if hl else base_color,
                    fontweight='bold' if hl else 'normal', zorder=5)
            x_px += w_px


def write_venn_diagram(labels: list[str], sets: dict[str, set], authors: dict[str, str],
                        outpath: pathlib.Path) -> None:
    """Hand-drawn 3-circle Venn diagram (no matplotlib-venn dependency):
    equal-sized circles in the classic symmetric layout, each region
    labelled with its syllabi count."""
    a_label, b_label, c_label = labels
    A, B, C = sets[a_label], sets[b_label], sets[c_label]

    only_a = len(A - B - C)
    only_b = len(B - A - C)
    only_c = len(C - A - B)
    ab     = len((A & B) - C)
    ac     = len((A & C) - B)
    bc     = len((B & C) - A)
    abc    = len(A & B & C)

    r = 1.4
    centers = {
        a_label: (-0.736, 0.425),
        b_label: (0.736, 0.425),
        c_label: (0.0, -0.85),
    }
    colors = dict(zip(labels, _CATEGORICAL))

    fig, ax = plt.subplots(figsize=(8, 7.8))
    for label in labels:
        cx, cy = centers[label]
        ax.add_patch(Circle((cx, cy), r, facecolor=colors[label], edgecolor='white',
                             linewidth=2, alpha=0.55, zorder=2))

    region_labels = [
        (only_a, (-1.4, 0.95)),
        (only_b, (1.4, 0.95)),
        (only_c, (0.0, -1.9)),
        (ab,     (0.0, 0.65)),
        (ac,     (-0.55, -0.4)),
        (bc,     (0.55, -0.4)),
        (abc,    (0.0, -0.05)),
    ]
    for count, (x, y) in region_labels:
        ax.text(x, y, str(count), ha='center', va='center', fontsize=14,
                color='#0b0b0b', fontweight='bold', zorder=3)

    # Set-identity labels: a colored swatch + neutral-ink text near each
    # circle, so identity never depends on color alone. Anchors push outward
    # (away from the circle) as titles wrap onto more lines, so long titles
    # don't creep down into the circle they're labelling.
    LINE_H = 0.16
    base_positions = {
        a_label: (-1.5, 2.15, +1),
        b_label: (1.5, 2.15, +1),
        c_label: (0.0, -2.75, -1),
    }
    for label in labels:
        x, y0, direction = base_positions[label]
        wrapped = '\n'.join(textwrap.wrap(label, 24))
        title_lines = wrapped.count('\n') + 1
        y = y0 + direction * max(0, title_lines - 2) * (LINE_H / 2)
        swatch_y = y + title_lines * (LINE_H / 2) + 0.14
        ax.add_patch(Circle((x, swatch_y), 0.09, facecolor=colors[label],
                             edgecolor='none', zorder=3))
        ax.text(x, y, wrapped, ha='center', va='center', fontsize=10.5,
                color='#0b0b0b', zorder=3)
        author = authors.get(label, '')
        if author:
            # Place the author line beneath the (possibly multi-line) title.
            author_y = y - LINE_H * title_lines - 0.14
            ax.text(x, author_y, author, ha='center', va='center', fontsize=9,
                    color='#52514e', style='italic', zorder=3)

    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-3.85, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Syllabi overlap across the three texts', fontsize=13, pad=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  -> {outpath.name}")


def write_cocitation_chart(label: str, target_idx: int, target_pdfs: set[str],
                            syllabus_index: dict[str, list[int]], canonical: list[dict],
                            outpath: pathlib.Path, top_n: int = 10) -> None:
    """For syllabi that cite `label`, what other works do they cite most,
    and how often (as a share of that text's citing syllabi)?"""
    co_counts: Counter = Counter()
    for pdf in target_pdfs:
        for i in syllabus_index.get(pdf, []):
            if i == target_idx:
                continue
            co_counts[i] += 1

    if not co_counts:
        return
    ranked = co_counts.most_common(top_n)
    entries = []
    for i, c in ranked:
        rec = canonical[i]
        title_wrapped = '\n'.join(textwrap.wrap(rec['title'], 48))
        author = format_authors(rec.get('authors') or [])
        entries.append((title_wrapped, author, c))
    counts = [c for _, _, c in entries]
    pct = [round(100 * c / len(target_pdfs), 0) for c in counts]

    # Row height auto-scales to the tallest label in this chart (title
    # lines + an author line) so multi-line titles never collide with
    # neighboring rows, while short titles don't waste vertical space.
    TITLE_FS, AUTHOR_FS = 9.5, 8
    TITLE_LINE_IN  = TITLE_FS * 1.3 / 72
    AUTHOR_LINE_IN = AUTHOR_FS * 1.3 / 72
    GAP_IN, PAD_IN = 0.05, 0.16

    def block_height_in(title_wrapped: str, has_author: bool) -> float:
        n_t = title_wrapped.count('\n') + 1
        h = n_t * TITLE_LINE_IN
        if has_author:
            h += GAP_IN + AUTHOR_LINE_IN
        return h

    max_block_in = max(block_height_in(t, bool(a)) for t, a, _ in entries)
    row_pitch_in = max_block_in + PAD_IN
    data_per_in = 1.0 / row_pitch_in

    fig_h = row_pitch_in * len(entries) + 1.5
    fig, ax = plt.subplots(figsize=(13, max(3.2, fig_h)))
    fig.subplots_adjust(left=0.44, right=0.97, top=1 - 0.9 / fig.get_figheight(),
                         bottom=1.1 / fig.get_figheight())

    ax.barh(range(len(entries)), counts, color=_BAR_COLOR, height=0.6)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([])
    ax.invert_yaxis()

    label_transform = ax.get_yaxis_transform()  # x: axes fraction, y: data
    for i, (title_wrapped, author, _) in enumerate(entries):
        n_t = title_wrapped.count('\n') + 1
        title_h = n_t * TITLE_LINE_IN * data_per_in
        block = title_h
        if author:
            block += (GAP_IN + AUTHOR_LINE_IN) * data_per_in
        top = i - block / 2
        title_y = top + title_h / 2
        ax.text(-0.02, title_y, title_wrapped, transform=label_transform,
                ha='right', va='center', fontsize=TITLE_FS, color='#c8790a',
                fontweight='medium', zorder=3)
        if author:
            author_y = top + title_h + (GAP_IN + AUTHOR_LINE_IN * 0.5) * data_per_in
            ax.text(-0.02, author_y, author, transform=label_transform,
                    ha='right', va='center', fontsize=AUTHOR_FS, color='#52514e',
                    style='italic', zorder=3)

    for i, (c, p) in enumerate(zip(counts, pct)):
        ax.text(c + max(counts) * 0.01, i, f'{c} ({p:.0f}%)', va='center', fontsize=8, color='#52514e')
    xlabel = '\n'.join(textwrap.wrap(
        f'Syllabi also citing this work, of {len(target_pdfs)} that cite "{label}"', 90))
    ax.set_xlabel(xlabel, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', length=0)
    title_lines = _wrap_with_highlight('What co-occurs with "', label, f'" (top {len(entries)})', 70)
    _draw_centered_multicolor_title(fig, ax, title_lines, fontsize=11,
                                     base_color='#0b0b0b', highlight_color='#d03b3b')
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  -> {outpath.name}")


def main():
    print("Loading and deduplicating citations …")
    canonical = load_canonical()
    syllabus_index = build_syllabus_index(canonical)

    for group_name, targets in GROUPS.items():
        print(f"\n=== {group_name} ===")
        group_dir = OUT_DIR / group_name
        group_dir.mkdir(parents=True, exist_ok=True)

        labels, sets, indices, authors = [], {}, {}, {}
        for label, hint in targets:
            idx, rec = find_target(canonical, label, hint)
            n_cit = sum(rec.get('year_counts', {}).values())
            pdfs = set(rec.get('source_syllabi', []))
            print(f'"{label}" -> matched "{rec["title"]}" '
                  f'({n_cit} citations across {len(pdfs)} syllabi)')
            labels.append(label)
            sets[label] = pdfs
            indices[label] = idx
            authors[label] = format_authors(rec.get('authors') or [])

        print("Writing overlap stats …")
        write_overlap_stats(labels, sets, group_dir / 'overlap_stats.csv')

        print("Writing overlap Venn diagram …")
        write_venn_diagram(labels, sets, authors, group_dir / 'overlap_regions.png')

        print("Writing per-text co-citation charts …")
        for label in labels:
            safe = label.lower().replace(' ', '_').replace(':', '').replace(',', '')[:40]
            write_cocitation_chart(
                label, indices[label], sets[label], syllabus_index, canonical,
                group_dir / f'cocitation_{safe}.png',
            )

    print(f"\nDone. Outputs written to {OUT_DIR}")


if __name__ == '__main__':
    main()
