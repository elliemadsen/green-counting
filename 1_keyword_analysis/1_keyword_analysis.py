"""
Step 1: Keyword Analysis
========================
Reads data/syllabi_text.csv (produced by 0_preprocessing.py, which already
applies both stopword lists) and:

  1. Loads keyword groups from data/keywords.txt (comma-separated aliases per line).
  2. Counts keyword frequency per syllabus and year; renders charts.
  3. Optionally runs top-N word analysis and log-likelihood (DO_TOP_N_ANALYSIS flag).
  4. Writes outputs/1_keyword_analysis.md.
"""

import re
import pathlib
import collections

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = pathlib.Path(__file__).parent
INPUT_CSV     = BASE_DIR.parent / "data" / "syllabi_text.csv"
KEYWORDS_FILE = BASE_DIR.parent / "data" / "keywords.txt"
OUT_DIR       = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_MD               = OUT_DIR / "1_keyword_analysis.md"
OUT_TOP100_CSV       = OUT_DIR / "top100_counts.csv"
OUT_LL_CSV           = OUT_DIR / "log_likelihood.csv"
OUT_LL_MD            = OUT_DIR / "log_likelihood_plain_english.md"
CHART_LL             = OUT_DIR / "log_likelihood_top_movers.png"

# Top-N charts (only rendered when DO_TOP_N_ANALYSIS = True)
CHART_HEAT           = OUT_DIR / "top20_per_year_heatmap.png"
CHART_HEAT_NORM      = OUT_DIR / "top20_per_year_heatmap_normalised.png"
CHART_HEAT_NORM_ORD  = OUT_DIR / "top20_per_year_heatmap_normalised_ordered.png"
CHART_BAR            = OUT_DIR / "top20_bar_chart.png"
CHART_RIDGE          = OUT_DIR / "top20_ridgeline.png"
CHART_RIDGE_NORM     = OUT_DIR / "top20_ridgeline_normalised.png"
CHART_RIDGE_NORM_ORD = OUT_DIR / "top20_ridgeline_normalised_ordered.png"

# Keyword charts
OUT_DIR_1                = OUT_DIR / "keywords-1"
OUT_DIR_1.mkdir(exist_ok=True)

CHART_KW_HEAT            = OUT_DIR_1 / "keywords_heatmap.png"
CHART_KW_HEAT_NORM       = OUT_DIR_1 / "keywords_heatmap_normalised.png"
CHART_KW_HEAT_NORM_ORD   = OUT_DIR_1 / "keywords_heatmap_normalised_ordered.png"
CHART_KW_BAR             = OUT_DIR_1 / "keywords_bar_chart.png"
CHART_KW_RIDGE           = OUT_DIR_1 / "keywords_ridgeline.png"
CHART_KW_RIDGE_NORM      = OUT_DIR_1 / "keywords_ridgeline_normalised.png"
CHART_KW_RIDGE_NORM_ORD  = OUT_DIR_1 / "keywords_ridgeline_normalised_ordered.png"

KEYWORDS_FILE_2          = BASE_DIR.parent / "data" / "keywords-2.txt"
OUT_DIR_2                = OUT_DIR / "keywords-2"
OUT_DIR_2.mkdir(exist_ok=True)

CHART_KW2_HEAT           = OUT_DIR_2 / "keywords_heatmap.png"
CHART_KW2_HEAT_NORM      = OUT_DIR_2 / "keywords_heatmap_normalised.png"
CHART_KW2_HEAT_NORM_ORD  = OUT_DIR_2 / "keywords_heatmap_normalised_ordered.png"
CHART_KW2_BAR            = OUT_DIR_2 / "keywords_bar_chart.png"
CHART_KW2_RIDGE          = OUT_DIR_2 / "keywords_ridgeline.png"
CHART_KW2_RIDGE_NORM     = OUT_DIR_2 / "keywords_ridgeline_normalised.png"
CHART_KW2_RIDGE_NORM_ORD = OUT_DIR_2 / "keywords_ridgeline_normalised_ordered.png"

# ── Flags ──────────────────────────────────────────────────────────────────────
# Set True to also run top-N word analysis, LL analysis, and top-N charts.
DO_TOP_N_ANALYSIS = False
TOP_N   = 100
CHART_N = 20

# ── Helpers ────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> list:
    """Tokenize filtered_text (stopwords already removed in step 0)."""
    return [t for t in re.findall(r"[a-z]+", str(text).lower()) if len(t) > 1]


def parse_keywords(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """
    Returns [(label, [alias, ...]), ...].
    Label = first alias (lowercased).  Aliases are all comma-separated tokens.
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


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading CSV …")
df = pd.read_csv(INPUT_CSV, dtype={"year": "Int64"})
df = df.dropna(subset=["filtered_text"])
print(f"  {len(df)} syllabi loaded, years {df['year'].min()}–{df['year'].max()}")

keyword_groups = parse_keywords(KEYWORDS_FILE)
keyword_labels = [label for label, _ in keyword_groups]
print(f"  {len(keyword_groups)} keyword groups loaded from {KEYWORDS_FILE.name}")

keyword_groups_2 = parse_keywords(KEYWORDS_FILE_2)
keyword_labels_2 = [label for label, _ in keyword_groups_2]
print(f"  {len(keyword_groups_2)} keyword groups loaded from {KEYWORDS_FILE_2.name}")

years = sorted(df["year"].dropna().unique())
first_year, last_year = int(min(years)), int(max(years))

# ── Per-year token totals (always needed for normalisation) ────────────────────
print("Computing per-year token totals …")
year_token_totals: dict = {}
for yr, grp in df.groupby("year"):
    year_token_totals[yr] = sum(len(tokenize(t)) for t in grp["filtered_text"])
year_token_series = pd.Series(year_token_totals)

# ── Top-N word analysis (optional) ────────────────────────────────────────────
if DO_TOP_N_ANALYSIS:
    print(f"Counting top-{TOP_N} words across corpus …")
    corpus_counter: collections.Counter = collections.Counter()
    for text in df["filtered_text"]:
        corpus_counter.update(tokenize(text))

    top100 = [word for word, _ in corpus_counter.most_common(TOP_N)]
    print(f"  Top word: '{top100[0]}' ({corpus_counter[top100[0]]:,} occurrences)")

    # Per-syllabus counts for top-N
    rows_top = []
    for _, row in df.iterrows():
        tokens = tokenize(row["filtered_text"])
        c = collections.Counter(tokens)
        syllabus_total = len(tokens)
        for w in top100:
            rows_top.append({
                "word":                  w,
                "pdf_title":             row["pdf_title"],
                "year":                  row["year"],
                "count":                 c.get(w, 0),
                "syllabus_total_tokens": syllabus_total,
            })

    long = pd.DataFrame(rows_top)
    long["freq_per_1k"] = (
        long["count"] / long["syllabus_total_tokens"].replace(0, pd.NA) * 1000
    ).round(4)
    total_map = {w: corpus_counter[w] for w in top100}
    long["corpus_total_count"] = long["word"].map(total_map)
    rank_map = {w: i + 1 for i, w in enumerate(top100)}
    long["corpus_rank"] = long["word"].map(rank_map)
    long = long.sort_values(["corpus_rank", "year", "pdf_title"]).reset_index(drop=True)
    long.to_csv(OUT_TOP100_CSV, index=False)
    print(f"  Written: {OUT_TOP100_CSV.name}  ({len(long):,} rows)")

    year_counts = (
        long.groupby(["year", "word"])["count"]
        .sum()
        .unstack(fill_value=0)
    )
    year_freq = year_counts.div(year_token_series, axis=0) * 1_000

# ── Keyword counts ─────────────────────────────────────────────────────────────
print("Computing keyword counts …")
kw_rows = []
for _, row in df.iterrows():
    tokens = tokenize(row["filtered_text"])
    c = collections.Counter(tokens)
    for label, aliases in keyword_groups:
        cnt = sum(c.get(a, 0) for a in aliases)
        kw_rows.append({
            "word":  label,
            "year":  row["year"],
            "count": cnt,
            "total": len(tokens),
        })

kw_df = pd.DataFrame(kw_rows)
kw_year_counts = kw_df.groupby(["year", "word"])["count"].sum().unstack(fill_value=0)
kw_freq = kw_year_counts.div(year_token_series, axis=0) * 1_000

print("Computing keyword-2 counts …")
kw2_rows = []
for _, row in df.iterrows():
    tokens = tokenize(row["filtered_text"])
    c = collections.Counter(tokens)
    for label, aliases in keyword_groups_2:
        cnt = sum(c.get(a, 0) for a in aliases)
        kw2_rows.append({
            "word":  label,
            "year":  row["year"],
            "count": cnt,
            "total": len(tokens),
        })

kw2_df = pd.DataFrame(kw2_rows)
kw2_year_counts = kw2_df.groupby(["year", "word"])["count"].sum().unstack(fill_value=0)
kw2_freq = kw2_year_counts.div(year_token_series, axis=0) * 1_000

# ── Chart helpers ─────────────────────────────────────────────────────────────
def _render_heatmap(freq_df, word_list, title, outpath):
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    data = freq_df[present].T
    fig, ax = plt.subplots(figsize=(max(8, len(yr_list) * 0.8), max(4, len(present) * 0.45)))
    im = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(yr_list)))
    ax.set_xticklabels([str(y) for y in yr_list], fontsize=10)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=10)
    vmax = data.values.max() or 1
    for yi in range(len(present)):
        for xi in range(len(yr_list)):
            val = data.values[yi, xi]
            ax.text(xi, yi, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color="black" if val < vmax * 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Occurrences per 1 000 words")
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _render_heatmap_normalised(freq_df, word_list, title, outpath):
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    raw = freq_df[present].T.values.astype(float)
    row_min = raw.min(axis=1, keepdims=True)
    row_max = raw.max(axis=1, keepdims=True)
    row_range = np.where(row_max - row_min == 0, 1, row_max - row_min)
    normed = (raw - row_min) / row_range
    fig, ax = plt.subplots(figsize=(max(8, len(yr_list) * 0.8), max(4, len(present) * 0.45)))
    im = ax.imshow(normed, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(yr_list)))
    ax.set_xticklabels([str(y) for y in yr_list], fontsize=10)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=10)
    for yi in range(len(present)):
        for xi in range(len(yr_list)):
            raw_val  = raw[yi, xi]
            norm_val = normed[yi, xi]
            ax.text(xi, yi, f"{raw_val:.1f}", ha="center", va="center",
                    fontsize=7, color="black" if 0.2 < norm_val < 0.8 else "white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Relative change (0 = min, 1 = max per word)", fontsize=8)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _sort_by_shift(freq_df, word_list):
    yr_avail = sorted(freq_df.index)
    y0, y1 = yr_avail[0], yr_avail[-1]
    def shift(w):
        v0 = freq_df.loc[y0, w] if (y0 in freq_df.index and w in freq_df.columns) else 0
        v1 = freq_df.loc[y1, w] if (y1 in freq_df.index and w in freq_df.columns) else 0
        return v1 - v0
    return sorted(word_list, key=shift, reverse=False)


def _sort_by_slope(freq_df, word_list):
    """Sort words by OLS slope (freq/1k ~ year); most negative first."""
    xs = np.array(sorted(freq_df.index), dtype=float)
    def slope(w):
        if w not in freq_df.columns or len(xs) < 2:
            return 0.0
        ys = np.array([freq_df.loc[yr, w] for yr in xs], dtype=float)
        return float(np.polyfit(xs, ys, 1)[0])
    return sorted(word_list, key=slope, reverse=False)


def _ridge_draw(ax, present, yr_list, heights_by_word, _color_norms):
    from matplotlib.patches import Polygon as MplPolygon
    cmap     = plt.get_cmap("YlOrRd")
    N_LEVELS = 40
    MAX_H    = 0.8
    level_h  = MAX_H / N_LEVELS
    bg_colors = [cmap((k + 0.5) / N_LEVELS) for k in range(N_LEVELS)]
    n        = len(present)
    x_pos    = list(range(len(yr_list)))
    x0_lane, x1_lane = x_pos[0], x_pos[-1]
    N_INTERP = 300

    for i, word in enumerate(reversed(present)):
        lane_y  = i
        heights = np.array(heights_by_word[word], dtype=float)
        for k in range(N_LEVELS):
            rect = plt.Rectangle(
                (x0_lane, lane_y + k * level_h), x1_lane - x0_lane, level_h,
                facecolor=bg_colors[k], edgecolor="none")
            ax.add_patch(rect)
        xs_fine = np.linspace(x0_lane, x1_lane, N_INTERP)
        hs_fine = np.interp(xs_fine, x_pos, heights)
        mask_top = lane_y + MAX_H + 0.02
        mask_xs = list(xs_fine) + [xs_fine[-1], x1_lane, x0_lane, xs_fine[0]]
        mask_ys = list(lane_y + hs_fine) + [mask_top, mask_top, mask_top, lane_y + hs_fine[0]]
        ax.add_patch(MplPolygon(list(zip(mask_xs, mask_ys)), closed=True,
                                facecolor="white", edgecolor="none", zorder=3))
        ax.add_patch(plt.Rectangle(
            (x0_lane, lane_y + MAX_H), x1_lane - x0_lane, 0.1,
            facecolor="white", edgecolor="none", zorder=4))
        ax.plot(x_pos, [lane_y + h for h in heights],
                color="#333333", linewidth=1.2, zorder=5)
        ax.plot([x0_lane, x1_lane], [lane_y, lane_y],
                color="#aaaaaa", linewidth=0.5, zorder=5)
        ax.text(-0.35, lane_y + 0.05, word, ha="right", va="bottom",
                fontsize=9, color="#333333", zorder=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(y) for y in yr_list], fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(-0.5, len(yr_list) - 0.5)
    ax.set_ylim(-0.2, n + 0.2)
    ax.set_xlabel("Year", fontsize=11)
    ax.spines[["left", "top", "right"]].set_visible(False)


def _render_ridge(freq_df, word_list, title, outpath):
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    global_max = freq_df[present].values.max() or 1
    scale = 0.8 / global_max
    hbw = {w: [freq_df.loc[yr, w] * scale if yr in freq_df.index else 0
               for yr in yr_list] for w in present}
    fig, ax = plt.subplots(figsize=(max(8, len(yr_list) * 1.1), max(5, len(present) * 0.55)))
    _ridge_draw(ax, present, yr_list, hbw, {})
    ax.set_title(title, fontsize=13)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def _render_ridge_normalised(freq_df, word_list, title, outpath):
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    MAX_H = 0.8
    hbw = {}
    for word in present:
        vals = np.array([freq_df.loc[yr, word] if yr in freq_df.index else 0
                         for yr in yr_list], dtype=float)
        span = vals.max() - vals.min() or 1
        hbw[word] = list((vals - vals.min()) / span * MAX_H)
    fig, ax = plt.subplots(figsize=(max(8, len(yr_list) * 1.1), max(5, len(present) * 0.55)))
    _ridge_draw(ax, present, yr_list, hbw, {})
    ax.set_title(title, fontsize=13)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


def _render_bar(freq_df, word_list, title, outpath):
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    x     = range(len(present))
    width = 0.8 / len(yr_list)
    cmap  = plt.get_cmap("tab10" if len(yr_list) <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(max(12, len(present) * 0.9), 6))
    for i, yr in enumerate(yr_list):
        if yr not in freq_df.index:
            continue
        vals    = [freq_df.loc[yr, w] if w in freq_df.columns else 0 for w in present]
        offsets = [xi + (i - len(yr_list) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width * 0.9, label=str(yr), color=cmap(i / len(yr_list)))
    ax.set_xticks(list(x))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Occurrences per 1 000 words")
    ax.set_title(title, fontsize=13)
    ax.legend(title="Year", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout(); fig.savefig(outpath, dpi=150); plt.close(fig)


# ── Render keyword charts ──────────────────────────────────────────────────────
print("Rendering keyword charts …")
kw_labels_by_freq = sorted(
    keyword_labels,
    key=lambda w: kw_freq[w].sum() if w in kw_freq.columns else 0,
    reverse=True,
)
_render_heatmap(kw_freq, kw_labels_by_freq,
    "Keywords – frequency per 1,000 words, by year", CHART_KW_HEAT)
print(f"  Written: {CHART_KW_HEAT.name}")
_render_heatmap_normalised(kw_freq, keyword_labels,
    "Keywords – row-normalised frequency change by year", CHART_KW_HEAT_NORM)
print(f"  Written: {CHART_KW_HEAT_NORM.name}")
_render_heatmap_normalised(kw_freq, _sort_by_slope(kw_freq, keyword_labels),
    "Keywords – row-normalised frequency change, ordered by slope", CHART_KW_HEAT_NORM_ORD)
print(f"  Written: {CHART_KW_HEAT_NORM_ORD.name}")
_render_bar(kw_freq, keyword_labels,
    "Keywords by year – relative frequency", CHART_KW_BAR)
print(f"  Written: {CHART_KW_BAR.name}")
_render_ridge(kw_freq, keyword_labels,
    "Keywords – frequency over time", CHART_KW_RIDGE)
print(f"  Written: {CHART_KW_RIDGE.name}")
_render_ridge_normalised(kw_freq, keyword_labels,
    "Keywords – row-normalised frequency over time", CHART_KW_RIDGE_NORM)
print(f"  Written: {CHART_KW_RIDGE_NORM.name}")
_render_ridge_normalised(kw_freq, _sort_by_slope(kw_freq, keyword_labels),
    "Keywords – row-normalised frequency over time, ordered by slope", CHART_KW_RIDGE_NORM_ORD)
print(f"  Written: {CHART_KW_RIDGE_NORM_ORD.name}")

# ── Render keyword-2 charts ────────────────────────────────────────────────────
print("Rendering keyword-2 charts …")
kw2_labels_by_freq = sorted(
    keyword_labels_2,
    key=lambda w: kw2_freq[w].sum() if w in kw2_freq.columns else 0,
    reverse=True,
)
_render_heatmap(kw2_freq, kw2_labels_by_freq,
    "Keywords-2 – frequency per 1,000 words, by year", CHART_KW2_HEAT)
print(f"  Written: {CHART_KW2_HEAT.name}")
_render_heatmap_normalised(kw2_freq, keyword_labels_2,
    "Keywords-2 – row-normalised frequency change by year", CHART_KW2_HEAT_NORM)
print(f"  Written: {CHART_KW2_HEAT_NORM.name}")
_render_heatmap_normalised(kw2_freq, _sort_by_slope(kw2_freq, keyword_labels_2),
    "Keywords-2 – row-normalised frequency change, ordered by slope", CHART_KW2_HEAT_NORM_ORD)
print(f"  Written: {CHART_KW2_HEAT_NORM_ORD.name}")
_render_bar(kw2_freq, keyword_labels_2,
    "Keywords-2 by year – relative frequency", CHART_KW2_BAR)
print(f"  Written: {CHART_KW2_BAR.name}")
_render_ridge(kw2_freq, keyword_labels_2,
    "Keywords-2 – frequency over time", CHART_KW2_RIDGE)
print(f"  Written: {CHART_KW2_RIDGE.name}")
_render_ridge_normalised(kw2_freq, keyword_labels_2,
    "Keywords-2 – row-normalised frequency over time", CHART_KW2_RIDGE_NORM)
print(f"  Written: {CHART_KW2_RIDGE_NORM.name}")
_render_ridge_normalised(kw2_freq, _sort_by_slope(kw2_freq, keyword_labels_2),
    "Keywords-2 – row-normalised frequency over time, ordered by slope", CHART_KW2_RIDGE_NORM_ORD)
print(f"  Written: {CHART_KW2_RIDGE_NORM_ORD.name}")

# ── Top-N charts and LL analysis (optional) ────────────────────────────────────
if DO_TOP_N_ANALYSIS:
    top20 = top100[:CHART_N]
    print("Rendering top-N charts …")
    _render_heatmap(year_freq, top20,
        f"Top-{CHART_N} words – frequency per 1,000 words, by year", CHART_HEAT)
    print(f"  Written: {CHART_HEAT.name}")
    _render_heatmap_normalised(year_freq, top20,
        f"Top-{CHART_N} words – row-normalised change by year", CHART_HEAT_NORM)
    print(f"  Written: {CHART_HEAT_NORM.name}")
    _render_heatmap_normalised(year_freq, _sort_by_shift(year_freq, top20),
        f"Top-{CHART_N} words – row-normalised, ordered by 2020→26 shift", CHART_HEAT_NORM_ORD)
    print(f"  Written: {CHART_HEAT_NORM_ORD.name}")
    _render_bar(year_freq, top20,
        f"Top-{CHART_N} words by year – relative frequency", CHART_BAR)
    print(f"  Written: {CHART_BAR.name}")
    _render_ridge(year_freq, top20,
        f"Top-{CHART_N} words – frequency over time (ridgeline)", CHART_RIDGE)
    print(f"  Written: {CHART_RIDGE.name}")
    _render_ridge_normalised(year_freq, top20,
        f"Top-{CHART_N} words – row-normalised ridgeline", CHART_RIDGE_NORM)
    print(f"  Written: {CHART_RIDGE_NORM.name}")
    _render_ridge_normalised(year_freq, _sort_by_shift(year_freq, top20),
        f"Top-{CHART_N} words – row-normalised ridgeline, ordered by 2020→26 shift", CHART_RIDGE_NORM_ORD)
    print(f"  Written: {CHART_RIDGE_NORM_ORD.name}")

    # ── Log-likelihood analysis ────────────────────────────────────────────────
    print("Computing log-likelihood scores …")

    def _log_likelihood(a, b, total_a, total_b):
        N = a + b + (total_a - a) + (total_b - b)
        if N == 0 or (a + b) == 0 or (a + total_a - a) == 0:
            return 0.0, 0
        E1 = total_a * (a + b) / N
        E2 = total_b * (a + b) / N
        g2 = 0.0
        if a > 0 and E1 > 0:
            g2 += 2 * a * np.log(a / E1)
        if b > 0 and E2 > 0:
            g2 += 2 * b * np.log(b / E2)
        direction = 1 if a / total_a >= b / total_b else -1
        return float(g2), direction

    def _p_label(g2):
        if g2 >= 15.13: return "p < 0.0001"
        if g2 >= 10.83: return "p < 0.001"
        if g2 >= 6.63:  return "p < 0.01"
        if g2 >= 3.84:  return "p < 0.05"
        return "not significant"

    all_years    = sorted(year_counts.index)
    corpus_total = int(year_token_series.sum())

    ll_rows = []
    for yr in all_years:
        total_a = int(year_token_series.loc[yr])
        total_b = corpus_total - total_a
        for word in year_counts.columns:
            a = int(year_counts.loc[yr, word])
            b = int(year_counts[word].sum()) - a
            g2, dirn = _log_likelihood(a, b, total_a, total_b)
            ll_rows.append({
                "word":             word,
                "year":             int(yr),
                "count_year":       a,
                "count_other":      b,
                "tokens_year":      total_a,
                "tokens_other":     total_b,
                "freq_year_per1k":  round(a / total_a * 1000, 4) if total_a else 0,
                "freq_other_per1k": round(b / total_b * 1000, 4) if total_b else 0,
                "G2":               round(g2, 3),
                "direction":        "+" if dirn == 1 else "-",
                "p_label":          _p_label(g2),
                "significant":      g2 >= 3.84,
            })

    ll_df = pd.DataFrame(ll_rows)
    ll_df.to_csv(OUT_LL_CSV, index=False)
    print(f"  Written: {OUT_LL_CSV.name}  ({len(ll_df):,} rows)")

    sig       = ll_df[(ll_df["significant"]) & (ll_df["direction"] == "+")].copy()
    top_ll_wds = sig.groupby("word")["G2"].max().nlargest(20).index.tolist()
    sig_top   = sig[sig["word"].isin(top_ll_wds)].sort_values("G2", ascending=False)

    if not sig_top.empty:
        pivot = sig_top.pivot_table(index="word", columns="year", values="G2", fill_value=0)
        pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=True).index]
        fig, ax = plt.subplots(figsize=(max(8, len(all_years) * 0.9), max(5, len(pivot) * 0.45)))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(y) for y in pivot.columns], fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        vmax = pivot.values.max() or 1
        for yi in range(len(pivot.index)):
            for xi in range(len(pivot.columns)):
                val = pivot.values[yi, xi]
                if val > 0:
                    ax.text(xi, yi, f"{val:.0f}", ha="center", va="center",
                            fontsize=7, color="black" if val < vmax * 0.6 else "white")
        plt.colorbar(im, ax=ax, label="G² (log-likelihood score)")
        ax.set_title("Top over-represented words by year (G², p < 0.05)", fontsize=13)
        fig.tight_layout()
        fig.savefig(CHART_LL, dpi=150)
        plt.close(fig)
        print(f"  Written: {CHART_LL.name}")

    # LL plain-English report
    OUT_LL_MD.write_text("(LL report — re-run with DO_TOP_N_ANALYSIS=True to generate)\n",
                         encoding="utf-8")

# ── Markdown report ────────────────────────────────────────────────────────────
print("Writing markdown report …")

def mean_slope(label: str) -> float:
    yr_vals = [(yr, kw_freq.loc[yr, label])
               for yr in sorted(kw_freq.index)
               if label in kw_freq.columns]
    if len(yr_vals) < 2:
        return 0.0
    xs = np.array([y for y, _ in yr_vals], dtype=float)
    ys = np.array([v for _, v in yr_vals], dtype=float)
    return float(np.polyfit(xs, ys, 1)[0])


def mean_slope_2(label: str) -> float:
    yr_vals = [(yr, kw2_freq.loc[yr, label])
               for yr in sorted(kw2_freq.index)
               if label in kw2_freq.columns]
    if len(yr_vals) < 2:
        return 0.0
    xs = np.array([y for y, _ in yr_vals], dtype=float)
    ys = np.array([v for _, v in yr_vals], dtype=float)
    return float(np.polyfit(xs, ys, 1)[0])


md_lines = [
    "# Keyword Analysis",
    "",
    "## Keyword frequency change",
    "",
    f"*(frequency per 1 000 words; slope = OLS trend per year across {first_year}–{last_year})*",
    "",
    "| Keyword | Aliases | Freq " + str(first_year) + " | Freq " + str(last_year) + " | Δ freq/1k | Slope/yr |",
    "|---------|---------|" + "------|" * 4,
]

for label, aliases in keyword_groups:
    alias_str = ", ".join(aliases[1:]) if len(aliases) > 1 else "—"
    f0    = kw_freq.loc[first_year, label] if (first_year in kw_freq.index and label in kw_freq.columns) else 0
    f1    = kw_freq.loc[last_year,  label] if (last_year  in kw_freq.index and label in kw_freq.columns) else 0
    delta = f1 - f0
    slope = mean_slope(label)
    md_lines.append(
        f"| {label} | {alias_str} | {f0:.2f} | {f1:.2f} "
        f"| {'%+.2f' % delta} | {'%+.3f' % slope} |"
    )

md_lines += [
    "",
    "## Keyword-2 frequency change",
    "",
    f"*(frequency per 1 000 words; slope = OLS trend per year across {first_year}–{last_year})*",
    "",
    "| Keyword | Aliases | Freq " + str(first_year) + " | Freq " + str(last_year) + " | Δ freq/1k | Slope/yr |",
    "|---------|---------|" + "------|" * 4,
]

for label, aliases in keyword_groups_2:
    alias_str = ", ".join(aliases[1:]) if len(aliases) > 1 else "—"
    f0    = kw2_freq.loc[first_year, label] if (first_year in kw2_freq.index and label in kw2_freq.columns) else 0
    f1    = kw2_freq.loc[last_year,  label] if (last_year  in kw2_freq.index and label in kw2_freq.columns) else 0
    delta = f1 - f0
    slope = mean_slope_2(label)
    md_lines.append(
        f"| {label} | {alias_str} | {f0:.2f} | {f1:.2f} "
        f"| {'%+.2f' % delta} | {'%+.3f' % slope} |"
    )

if DO_TOP_N_ANALYSIS and first_year in year_freq.index and last_year in year_freq.index:
    top100_delta = year_freq.loc[last_year, top100] - year_freq.loc[first_year, top100]
    rising  = top100_delta.nlargest(10)
    falling = top100_delta.nsmallest(10)

    md_lines += [
        "",
        f"## Biggest movers (top-{TOP_N} words)",
        "",
        f"*(comparing {first_year} → {last_year}, frequency per 1 000 words)*",
        "",
        "### Rising words",
        "| Word | Δ freq/1 000 |",
        "| ------------- | ------------ |",
    ]
    for word, dval in rising.items():
        md_lines.append(f"| {word} | +{dval:.2f} |")

    md_lines += [
        "",
        "### Declining words",
        "| Word | Δ freq/1 000 |",
        "| ------------ | ------------ |",
    ]
    for word, dval in falling.items():
        md_lines.append(f"| {word} | {dval:.2f} |")

md_lines += [
    "",
    "## Output files",
    "",
    "| File | Description |",
    "|------|-------------|",
    "| `keywords-1/keywords_heatmap.png` | Keyword frequency per 1,000 words, by year |",
    "| `keywords-1/keywords_heatmap_normalised.png` | Row-normalised heatmap (relative change) |",
    "| `keywords-1/keywords_ridgeline.png` | Ridgeline chart of keyword frequency over time |",
    "| `keywords-1/keywords_bar_chart.png` | Grouped bar chart of keyword frequency |",
    "| `keywords-2/keywords_heatmap.png` | Keyword-2 frequency per 1,000 words, by year |",
    "| `keywords-2/keywords_heatmap_normalised.png` | Row-normalised heatmap for keyword-2 |",
    "| `keywords-2/keywords_ridgeline.png` | Ridgeline chart of keyword-2 frequency over time |",
    "| `keywords-2/keywords_bar_chart.png` | Grouped bar chart of keyword-2 frequency |",
]
if DO_TOP_N_ANALYSIS:
    md_lines += [
        f"| `top100_counts.csv` | Per-syllabus counts for top-{TOP_N} corpus words |",
        "| `log_likelihood.csv` | G² scores for top-N words × year |",
        "| `log_likelihood_top_movers.png` | Heatmap of top over-represented words |",
    ]

md_lines += [
    "",
    "## Notes",
    "",
    "- Both stopword lists applied in step 0 (PDF preprocessing).",
    "- Frequencies normalised per 1 000 words to account for varying corpus sizes per year.",
    f"- Keyword groups loaded from `data/keywords.txt` and `data/keywords-2.txt`; comma-separated aliases on each line",
    f"  are counted together and displayed under the first alias.",
    f"- Set `DO_TOP_N_ANALYSIS = True` in the script to also produce top-{TOP_N} word charts",
    "  and log-likelihood analysis.",
]

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"  Written: {OUT_MD.name}")

print("\nAll done.")
