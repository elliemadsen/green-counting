"""
Step 1: Keyword Analysis
Reads syllabi_text.csv (produced by 0_pdf_preprocessing.py) and:
  1. Applies a secondary stopword list (stopwords_2.txt) to strip academic/
     institutional boilerplate, so analysis focuses on climate/design language.
  2. Tallies the top 100 most common words across all filtered syllabi.
  3. Produces a single long-format (tidy) CSV:
       top100_counts.csv  —  columns: word | pdf_title | year | count | total_count | freq_per_1k
       One row per (word × syllabus) pair; includes per-year aggregates as extra
       columns so everything lives in one file.
  4. Produces charts in 1_keyword_analysis_output/:
       top20_per_year_heatmap.png  : heatmap of top-20 word frequency per year
       top20_bar_chart.png         : grouped bar chart, top-20 words coloured by year
  5. Writes summary commentary to 1_keyword_analysis_output/1_keyword_analysis.md.
"""

import re
import pathlib
import collections

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all envs)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = pathlib.Path(__file__).parent
INPUT_CSV   = BASE_DIR / "syllabi_text.csv"
SW2_FILE    = BASE_DIR / "stopwords_2.txt"
OUT_DIR     = BASE_DIR / "1_keyword_analysis_output"
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV         = OUT_DIR / "top100_counts.csv"
OUT_MD          = OUT_DIR / "1_keyword_analysis.md"
CHART_HEAT         = OUT_DIR / "top20_per_year_heatmap.png"
CHART_HEAT_NORM    = OUT_DIR / "top20_per_year_heatmap_normalised.png"
CHART_HEAT_NORM_ORD = OUT_DIR / "top20_per_year_heatmap_normalised_ordered.png"
CHART_BAR          = OUT_DIR / "top20_bar_chart.png"
CHART_RIDGE        = OUT_DIR / "top20_ridgeline.png"
CHART_RIDGE_NORM   = OUT_DIR / "top20_ridgeline_normalised.png"
CHART_RIDGE_NORM_ORD = OUT_DIR / "top20_ridgeline_normalised_ordered.png"
CHART_CUSTOM_HEAT  = OUT_DIR / "custom_words_heatmap.png"
CHART_CUSTOM_HEAT_NORM = OUT_DIR / "custom_words_heatmap_normalised.png"
CHART_CUSTOM_HEAT_NORM_ORD = OUT_DIR / "custom_words_heatmap_normalised_ordered.png"
CHART_CUSTOM_BAR   = OUT_DIR / "custom_words_bar_chart.png"
CHART_CUSTOM_RIDGE = OUT_DIR / "custom_words_ridgeline.png"
CHART_CUSTOM_RIDGE_NORM = OUT_DIR / "custom_words_ridgeline_normalised.png"
CHART_CUSTOM_RIDGE_NORM_ORD = OUT_DIR / "custom_words_ridgeline_normalised_ordered.png"
OUT_LL_CSV      = OUT_DIR / "log_likelihood.csv"
OUT_LL_MD       = OUT_DIR / "log_likelihood_plain_english.md"
CHART_LL        = OUT_DIR / "log_likelihood_top_movers.png"

# ── Custom word list ───────────────────────────────────────────────────────────
# Replace these 20 strings with the words you want to track.
# Multi-word phrases are NOT supported here; use single tokens only.
CUSTOM_WORDS = [
    "climate",
    "change",
    "environmental",
    "urban",
    "community",
    "social",
    "energy",
    "material",
    "sustainable",
    "landscape",
    "global",
    "infrastructure",
    "cultural",
    "carbon",
    "critical",
    "justice",
    "history",
    "ecological",
    "water",
    "technology",
]

TOP_N = 100
CHART_N = 20   # words shown in charts

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_stopwords(path: pathlib.Path) -> set:
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words

def tokenize(text: str, extra_stopwords: set) -> list:
    """Split filtered_text into tokens, removing any extra stopwords."""
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in tokens if t not in extra_stopwords and len(t) > 1]


# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading CSV …")
df = pd.read_csv(INPUT_CSV, dtype={"year": "Int64"})
df = df.dropna(subset=["filtered_text"])
print(f"  {len(df)} syllabi loaded, years {df['year'].min()}–{df['year'].max()}")

stopwords_2 = load_stopwords(SW2_FILE)
print(f"  Secondary stopwords loaded: {len(stopwords_2)} words")

# ── 1. Top-100 words across entire corpus ─────────────────────────────────────
print("Counting words across corpus …")
corpus_counter: collections.Counter = collections.Counter()
for text in df["filtered_text"]:
    corpus_counter.update(tokenize(text, stopwords_2))

top100 = [word for word, _ in corpus_counter.most_common(TOP_N)]
print(f"  Top word: '{top100[0]}' ({corpus_counter[top100[0]]:,} occurrences)")

# ── 2. Single long-format CSV ─────────────────────────────────────────────────
# One row per (word × syllabus); includes corpus total and per-1k-word freq
# so the file is self-contained for all downstream analysis.
print("Building per-syllabus counts …")

# Per-syllabus raw counts
rows = []
for _, row in df.iterrows():
    tokens = tokenize(row["filtered_text"], stopwords_2)
    c = collections.Counter(tokens)
    syllabus_total = len(tokens)
    for w in top100:
        cnt = c.get(w, 0)
        rows.append({
            "word":           w,
            "pdf_title":      row["pdf_title"],
            "year":           row["year"],
            "count":          cnt,
            "syllabus_total_tokens": syllabus_total,
        })

long = pd.DataFrame(rows)
long["freq_per_1k"] = (long["count"] / long["syllabus_total_tokens"].replace(0, pd.NA) * 1000).round(4)

# Attach corpus-wide total for each word
total_map = {w: corpus_counter[w] for w in top100}
long["corpus_total_count"] = long["word"].map(total_map)

# Corpus rank (1 = most common)
rank_map = {w: i + 1 for i, w in enumerate(top100)}
long["corpus_rank"] = long["word"].map(rank_map)

long = long.sort_values(["corpus_rank", "year", "pdf_title"]).reset_index(drop=True)
long.to_csv(OUT_CSV, index=False)
print(f"  Written: {OUT_CSV.name}  ({len(long):,} rows)")

# ── 3. Per-year aggregates (used for charts only, not saved separately) ────────
print("Aggregating by year …")
# Total tokens per year (after secondary stopwords)
year_token_totals = (
    long[["year", "pdf_title", "syllabus_total_tokens"]]
    .drop_duplicates()
    .groupby("year")["syllabus_total_tokens"]
    .sum()
)
year_counts = (
    long.groupby(["year", "word"])["count"]
    .sum()
    .unstack(fill_value=0)
)
year_freq = year_counts.div(year_token_totals, axis=0) * 1_000   # per-1000-words

# ── 4. Charts ─────────────────────────────────────────────────────────────────
years = sorted(df["year"].dropna().unique())
top20 = top100[:CHART_N]

# (charts rendered below alongside custom-word charts)

# ── 5. Custom-word charts ────────────────────────────────────────────────────
print("Rendering custom-word charts …")

# Count custom words per syllabus (reuse the per-syllabus token counts)
custom_rows = []
for _, row in df.iterrows():
    tokens = tokenize(row["filtered_text"], stopwords_2)
    c = collections.Counter(tokens)
    syllabus_total = len(tokens)
    for w in CUSTOM_WORDS:
        custom_rows.append({
            "word":    w,
            "year":    row["year"],
            "count":   c.get(w, 0),
            "total":   syllabus_total,
        })

custom_df = pd.DataFrame(custom_rows)
custom_year_counts = custom_df.groupby(["year", "word"])["count"].sum().unstack(fill_value=0)
custom_year_totals = (
    custom_df[["year", "word", "total"]]
    .drop_duplicates(subset=["year", "word"])
    .groupby("year")["total"]
    .first()          # totals are the same for every word in a given year
)
custom_year_totals = df.groupby("year")["filtered_text"].apply(
    lambda texts: sum(len(tokenize(t, stopwords_2)) for t in texts)
)
custom_freq = custom_year_counts.div(custom_year_totals, axis=0) * 1_000

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
    """
    Row-normalised heatmap.  Each word's row is independently scaled
    0–1 (min–max across years), so colour shows relative change over
    time rather than raw frequency magnitude.
    Cell labels show the original freq/1k value for reference.
    """
    import numpy as np
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    raw = freq_df[present].T.values.astype(float)   # shape (words × years)

    # Row-wise min-max normalisation
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
            raw_val = raw[yi, xi]
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
    """Return word_list sorted most-positive → most-negative 2020→2026 shift."""
    years_avail = sorted(freq_df.index)
    y0, y1 = years_avail[0], years_avail[-1]
    def shift(w):
        v0 = freq_df.loc[y0, w] if (y0 in freq_df.index and w in freq_df.columns) else 0
        v1 = freq_df.loc[y1, w] if (y1 in freq_df.index and w in freq_df.columns) else 0
        return v1 - v0
    return sorted(word_list, key=shift, reverse=False)


def _ridge_draw(ax, present, yr_list, heights_by_word, color_norms_by_word):
    """Shared drawing logic for ridgeline charts.
    Each word has a full-lane gradient background (YlOrRd, bottom=yellow, top=burgundy).
    The area above the ridgeline is masked white, so only the gradient below the line shows.
    heights_by_word controls the ridgeline shape.
    color_norms_by_word is accepted but unused (colour is positional/absolute).
    """
    import numpy as np
    from matplotlib.patches import Polygon as MplPolygon

    cmap     = plt.get_cmap("YlOrRd")
    N_LEVELS = 40          # resolution of background gradient bands
    MAX_H    = 0.8
    level_h  = MAX_H / N_LEVELS
    bg_colors = [cmap((k + 0.5) / N_LEVELS) for k in range(N_LEVELS)]

    n     = len(present)
    x_pos = list(range(len(yr_list)))
    x0_lane = x_pos[0]
    x1_lane = x_pos[-1]
    N_INTERP = 300

    for i, word in enumerate(reversed(present)):
        lane_y  = i
        heights = np.array(heights_by_word[word], dtype=float)

        # 1. Draw full-width gradient background for this lane
        for k in range(N_LEVELS):
            band_y0 = lane_y + k * level_h
            band_y1 = lane_y + (k + 1) * level_h
            rect = plt.Rectangle(
                (x0_lane, band_y0), x1_lane - x0_lane, band_y1 - band_y0,
                facecolor=bg_colors[k], edgecolor="none"
            )
            ax.add_patch(rect)

        # 2. Mask the area above the ridgeline with white
        # Extend interpolation to full lane width so no gradient leaks at edges
        xs_fine = np.linspace(x0_lane, x1_lane, N_INTERP)
        hs_fine = np.interp(xs_fine, x_pos, heights)

        # Polygon: left edge top → along ridge top → right edge top → back along ceiling
        mask_top = lane_y + MAX_H + 0.02
        mask_xs  = list(xs_fine) + [xs_fine[-1], x1_lane, x0_lane, xs_fine[0]]
        mask_ys  = list(lane_y + hs_fine) + [mask_top, mask_top, mask_top, lane_y + hs_fine[0]]
        mask = MplPolygon(list(zip(mask_xs, mask_ys)), closed=True,
                          facecolor="white", edgecolor="none", zorder=3)
        ax.add_patch(mask)

        # 3. Clip background to lane bounds (mask above MAX_H and below baseline)
        for clip_y0, clip_y1, clip_x0, clip_x1 in [
            (lane_y + MAX_H, lane_y + MAX_H + 0.1, x0_lane, x1_lane),  # above peak
        ]:
            rect = plt.Rectangle(
                (clip_x0, clip_y0), clip_x1 - clip_x0, clip_y1 - clip_y0,
                facecolor="white", edgecolor="none", zorder=4
            )
            ax.add_patch(rect)

        # 4. Ridgeline outline
        ax.plot(x_pos, [lane_y + h for h in heights],
                color="#333333", linewidth=1.2, zorder=5)
        # Baseline
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
    """
    Ridgeline chart using raw frequency per 1 000 words.
    All words share the same global scale so taller = more frequent.
    """
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    n = len(present)

    all_vals = freq_df[present].values
    global_max = all_vals.max() or 1
    scale = 0.8 / global_max

    x_pos = list(range(len(yr_list)))
    heights_by_word    = {}
    color_norms_by_word = {}
    for word in present:
        vals = [freq_df.loc[yr, word] if yr in freq_df.index else 0 for yr in yr_list]
        heights_by_word[word]     = [v * scale for v in vals]
        color_norms_by_word[word] = [v / global_max for v in vals]  # same scale → matches heatmap

    fig, ax = plt.subplots(figsize=(max(8, len(yr_list) * 1.1), max(5, n * 0.55)))
    _ridge_draw(ax, present, yr_list, heights_by_word, color_norms_by_word)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _render_ridge_normalised(freq_df, word_list, title, outpath):
    """
    Row-normalised ridgeline chart.  Each word is independently scaled
    0–1 (min–max across years), so band height shows relative change
    rather than raw magnitude.  All rows reach the same max height.
    """
    import numpy as np
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    n = len(present)
    MAX_H = 0.8

    heights_by_word    = {}
    color_norms_by_word = {}
    for word in present:
        vals = np.array([freq_df.loc[yr, word] if yr in freq_df.index else 0
                         for yr in yr_list], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        span = vmax - vmin if vmax != vmin else 1
        normed = (vals - vmin) / span
        heights_by_word[word]     = list(normed * MAX_H)
        color_norms_by_word[word] = list(normed)  # same 0-1 scale → matches normalised heatmap

    fig, ax = plt.subplots(figsize=(max(8, len(yr_list) * 1.1), max(5, n * 0.55)))
    _ridge_draw(ax, present, yr_list, heights_by_word, color_norms_by_word)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _render_bar(freq_df, word_list, title, outpath):
    present = [w for w in word_list if w in freq_df.columns]
    yr_list = sorted(freq_df.index)
    x = range(len(present))
    width = 0.8 / len(yr_list)
    cmap = plt.get_cmap("tab10" if len(yr_list) <= 10 else "tab20")
    colors = [cmap(i / len(yr_list)) for i in range(len(yr_list))]
    fig, ax = plt.subplots(figsize=(max(12, len(present) * 0.9), 6))
    for i, (yr, color) in enumerate(zip(yr_list, colors)):
        if yr not in freq_df.index:
            continue
        vals = [freq_df.loc[yr, w] if w in freq_df.columns else 0 for w in present]
        offsets = [xi + (i - len(yr_list) / 2) * width for xi in x]
        ax.bar(offsets, vals, width=width * 0.9, label=str(yr), color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.set_ylabel("Occurrences per 1 000 words")
    ax.set_title(title, fontsize=13)
    ax.legend(title="Year", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# Refactor original top-20 charts to use the helpers
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

_render_heatmap(custom_freq, CUSTOM_WORDS,
    "Select Keywords – frequency per 1,000 words, by year", CHART_CUSTOM_HEAT)
print(f"  Written: {CHART_CUSTOM_HEAT.name}")
_render_heatmap_normalised(custom_freq, CUSTOM_WORDS,
    "Select Keywords – row-normalised frequency change by year", CHART_CUSTOM_HEAT_NORM)
print(f"  Written: {CHART_CUSTOM_HEAT_NORM.name}")
_render_heatmap_normalised(custom_freq, _sort_by_shift(custom_freq, CUSTOM_WORDS),
    "Select Keywords – row-normalised frequency change, ordered by shift", CHART_CUSTOM_HEAT_NORM_ORD)
print(f"  Written: {CHART_CUSTOM_HEAT_NORM_ORD.name}")
_render_bar(custom_freq, CUSTOM_WORDS,
    "Select Keywords by year – relative frequency", CHART_CUSTOM_BAR)
print(f"  Written: {CHART_CUSTOM_BAR.name}")
_render_ridge(custom_freq, CUSTOM_WORDS,
    "Select Keywords – frequency over time", CHART_CUSTOM_RIDGE)
print(f"  Written: {CHART_CUSTOM_RIDGE.name}")
_render_ridge_normalised(custom_freq, CUSTOM_WORDS,
    "Select Keywords – row-normalised frequency over time", CHART_CUSTOM_RIDGE_NORM)
print(f"  Written: {CHART_CUSTOM_RIDGE_NORM.name}")
_render_ridge_normalised(custom_freq, _sort_by_shift(custom_freq, CUSTOM_WORDS),
    "Select Keywords – row-normalised frequency over time, ordered by shift", CHART_CUSTOM_RIDGE_NORM_ORD)
print(f"  Written: {CHART_CUSTOM_RIDGE_NORM_ORD.name}")

# ── 6. Log-likelihood analysis ────────────────────────────────────────────────
print("Computing log-likelihood scores …")

import numpy as np

def _log_likelihood(a, b, total_a, total_b):
    """
    Dunning (1993) G² log-likelihood for a 2×2 contingency table.
    a = observed count of word in focal corpus (year)
    b = observed count of word in reference corpus (rest of years)
    total_a = total tokens in focal corpus
    total_b = total tokens in reference corpus
    Returns (G2, direction) where direction is +1 (over) or -1 (under).
    """
    c = total_a - a
    d = total_b - b
    N = a + b + c + d
    if N == 0 or (a + b) == 0 or (a + c) == 0:
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
    if g2 >= 15.13:  return "p < 0.0001"
    if g2 >= 10.83:  return "p < 0.001"
    if g2 >= 6.63:   return "p < 0.01"
    if g2 >= 3.84:   return "p < 0.05"
    return "not significant"

# Build per-year token totals and word counts over ALL vocabulary (not just top100)
# For LL we need: count of word W in year Y  vs  count of W in all other years.
# Use year_counts (top100 words × years) from step 3.
all_years = sorted(year_counts.index)
corpus_total = int(year_token_totals.sum())

ll_rows = []
for yr in all_years:
    total_a = int(year_token_totals.loc[yr])
    total_b = corpus_total - total_a
    for word in year_counts.columns:
        a = int(year_counts.loc[yr, word])
        b = int(year_counts[word].sum()) - a
        g2, dirn = _log_likelihood(a, b, total_a, total_b)
        ll_rows.append({
            "word":        word,
            "year":        int(yr),
            "count_year":  a,
            "count_other": b,
            "tokens_year": total_a,
            "tokens_other":total_b,
            "freq_year_per1k":  round(a / total_a * 1000, 4) if total_a else 0,
            "freq_other_per1k": round(b / total_b * 1000, 4) if total_b else 0,
            "G2":          round(g2, 3),
            "direction":   "+" if dirn == 1 else "-",
            "p_label":     _p_label(g2),
            "significant": g2 >= 3.84,
        })

ll_df = pd.DataFrame(ll_rows)
ll_df.to_csv(OUT_LL_CSV, index=False)
print(f"  Written: {OUT_LL_CSV.name}  ({len(ll_df):,} rows)")

# ── LL chart: top movers per year (over-represented only, p<0.05) ─────────────
sig = ll_df[(ll_df["significant"]) & (ll_df["direction"] == "+")].copy()
sig_sorted = sig.sort_values("G2", ascending=False)

# Top-N words by max G2 across all years
top_ll_words = sig_sorted.groupby("word")["G2"].max().nlargest(20).index.tolist()
sig_top = sig_sorted[sig_sorted["word"].isin(top_ll_words)]

if not sig_top.empty:
    pivot = sig_top.pivot_table(index="word", columns="year", values="G2", fill_value=0)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=True).index]  # sort by peak G2

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

# ── LL plain-English report ───────────────────────────────────────────────────
print("Writing plain-English log-likelihood report …")

ll_md = [
    "# Log-Likelihood Analysis – Plain English",
    "",
    "## What is log-likelihood?",
    "",
    "Frequency counts alone can be misleading: a word that appears 10 times in a",
    "50-syllabus year looks different from the same count in a 5-syllabus year.",
    "**Log-likelihood (G²)** is a statistical test that asks: *is this word appearing",
    "more (or less) often than we would expect by chance, given the size of each",
    "year’s corpus?*",
    "",
    "The test compares each year against the pooled rest of the corpus.",
    "A higher G² score means a stronger, more reliable signal.",
    "",
    "| G² threshold | Meaning |",
    "|---|---|",
    "| ≥ 3.84 | Significant at p < 0.05 (1-in-20 chance this is random) |",
    "| ≥ 6.63 | Significant at p < 0.01 (1-in-100) |",
    "| ≥ 10.83 | Significant at p < 0.001 (1-in-1 000) |",
    "| ≥ 15.13 | Significant at p < 0.0001 (1-in-10 000) |",
    "",
    "A **`+`** direction means the word is *over-represented* in that year – used more",
    "than the overall corpus rate would predict.  A **`-`** means *under-represented*.",
    "",
    "---",
    "",
    "## Key findings by year",
    "",
]

for yr in all_years:
    yr_df = ll_df[ll_df["year"] == yr].copy()
    over  = yr_df[(yr_df["significant"]) & (yr_df["direction"] == "+")].nlargest(10, "G2")
    under = yr_df[(yr_df["significant"]) & (yr_df["direction"] == "-")].nlargest(5, "G2")

    n_sig = int(yr_df["significant"].sum())
    total_words_yr = int(yr_df["tokens_year"].iloc[0]) if len(yr_df) else 0
    n_syllabi_yr = int(df[df["year"] == yr].shape[0])

    ll_md += [
        f"### {yr}",
        "",
        f"*{n_syllabi_yr} syllabi • {total_words_yr:,} tokens • {n_sig} words significantly over/under-represented*",
        "",
    ]

    if not over.empty:
        ll_md.append("**Significantly over-represented** (words used more than expected in this year):")
        ll_md.append("")
        ll_md.append("| Word | G² | Significance | Freq this year | Freq elsewhere |")
        ll_md.append("|---|---|---|---|---|")
        for _, r in over.iterrows():
            ll_md.append(
                f"| **{r['word']}** | {r['G2']:.1f} | {r['p_label']} "
                f"| {r['freq_year_per1k']:.2f}/1k | {r['freq_other_per1k']:.2f}/1k |"
            )
        ll_md.append("")

    if not under.empty:
        ll_md.append("**Significantly under-represented** (words used less than expected):")
        ll_md.append("")
        ll_md.append("| Word | G² | Significance | Freq this year | Freq elsewhere |")
        ll_md.append("|---|---|---|---|---|")
        for _, r in under.iterrows():
            ll_md.append(
                f"| {r['word']} | {r['G2']:.1f} | {r['p_label']} "
                f"| {r['freq_year_per1k']:.2f}/1k | {r['freq_other_per1k']:.2f}/1k |"
            )
        ll_md.append("")

    if over.empty and under.empty:
        ll_md += ["*No words were significantly over- or under-represented this year.*", ""]

    ll_md.append("---")
    ll_md.append("")

# Summary: biggest movers overall (2020 vs last year)
first_yr, last_yr = all_years[0], all_years[-1]
over_first = ll_df[(ll_df["year"] == first_yr) & (ll_df["significant"]) & (ll_df["direction"] == "+")]
over_last  = ll_df[(ll_df["year"] == last_yr)  & (ll_df["significant"]) & (ll_df["direction"] == "+")]

# Words that flipped: over in last year but NOT in first year
new_in_last = set(over_last["word"]) - set(over_first["word"])
lost_by_last = set(over_first["word"]) - set(over_last["word"])

ll_md += [
    "## Summary: what changed between {} and {}?".format(first_yr, last_yr),
    "",
    "Words that became **significantly over-represented** in {} but were not in {}:".format(last_yr, first_yr),
    "",
]
if new_in_last:
    emerging = over_last[over_last["word"].isin(new_in_last)].nlargest(15, "G2")
    for _, r in emerging.iterrows():
        pct = ((r["freq_year_per1k"] - r["freq_other_per1k"]) / max(r["freq_other_per1k"], 0.001) * 100)
        ll_md.append(f"- **{r['word']}** — G² = {r['G2']:.1f} ({r['p_label']}), "
                     f"{r['freq_year_per1k']:.2f} vs {r['freq_other_per1k']:.2f} per 1k words "
                     f"({'%+.0f' % pct}% relative to rest of corpus)")
else:
    ll_md.append("*None found.*")

ll_md += [
    "",
    "Words that were **significantly over-represented** in {} but are no longer in {}:".format(first_yr, last_yr),
    "",
]
if lost_by_last:
    fading = over_first[over_first["word"].isin(lost_by_last)].nlargest(15, "G2")
    for _, r in fading.iterrows():
        ll_md.append(f"- **{r['word']}** — G² = {r['G2']:.1f} ({r['p_label']}), "
                     f"{r['freq_year_per1k']:.2f} per 1k in {first_yr}")
else:
    ll_md.append("*None found.*")

ll_md += [
    "",
    "---",
    "",
    "## How to read this",
    "",
    "- **G²** is the raw score. Bigger = more surprising departure from the expected rate.",
    "- **Freq this year / Freq elsewhere** shows the actual usage rates so you can see",
    "  the direction and magnitude.",
    "- Words that appear consistently across all years will have low G² even if common,",
    "  because they are not *distinctive* to any year.",
    "- Words appearing in very few syllabi can score high G² by coincidence; always",
    "  cross-check with raw counts.",
    "- The test is run word-by-word (no multiple-comparison correction); treat borderline",
    "  p < 0.05 results with appropriate caution.",
]

OUT_LL_MD.write_text("\n".join(ll_md), encoding="utf-8")
print(f"  Written: {OUT_LL_MD.name}")

# ── 7. Markdown report ────────────────────────────────────────────────────────
print("Writing markdown report …")

# Compute year-on-year biggest risers/fallers (first year vs last year)
first_year, last_year = min(years), max(years)
if first_year in year_freq.index and last_year in year_freq.index:
    delta = year_freq.loc[last_year, top100] - year_freq.loc[first_year, top100]
    rising  = delta.nlargest(10)
    falling = delta.nsmallest(10)
else:
    rising = falling = pd.Series(dtype=float)

md_lines = [
    "# Keyword Analysis – Syllabi 2020–2026",
    "",
    "## Corpus overview",
    "",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Total syllabi | {len(df):,} |",
    f"| Years covered | {first_year} – {last_year} |",
    f"| Total tokens (after stopword filter) | {sum(corpus_counter.values()):,} |",
    f"| Unique tokens | {len(corpus_counter):,} |",
    "",
    "## Top 100 words (corpus-wide)",
    "",
    "| Rank | Word | Total count |",
    "|------|------|-------------|",
]
for rank, (word, cnt) in enumerate(corpus_counter.most_common(TOP_N), 1):
    md_lines.append(f"| {rank} | {word} | {cnt:,} |")

md_lines += [
    "",
    "## Output files",
    "",
    "| File | Description |",
    "|------|-------------|",
    f"| `top100_counts.csv` | Long/tidy format: one row per (word × syllabus). Columns: word, pdf_title, year, count, syllabus_total_tokens, freq_per_1k, corpus_total_count, corpus_rank |",
    f"| `log_likelihood.csv` | G² score, direction, and p-level for every word × year pair |",
    f"| `log_likelihood_plain_english.md` | Plain-English interpretation of significant findings |",
    f"| `log_likelihood_top_movers.png` | Heatmap of top over-represented words by year (G²) |",
    f"| `top20_per_year_heatmap.png` | Heatmap of top-20 word frequency per year (per-1 000-words) |",
    f"| `top20_bar_chart.png` | Grouped bar chart of top-20 words by year |",
    "",
    "## Biggest movers",
    f"*(comparing {first_year} → {last_year}, frequency per 1 000 words)*",
    "",
]

if not rising.empty:
    md_lines += [
        "### Rising words",
        "| Word | Δ freq/1 000 |",
        "|------|--------------|",
    ]
    for word, delta_val in rising.items():
        md_lines.append(f"| {word} | +{delta_val:.2f} |")
    md_lines += [
        "",
        "### Declining words",
        "| Word | Δ freq/1 000 |",
        "|------|--------------|",
    ]
    for word, delta_val in falling.items():
        md_lines.append(f"| {word} | {delta_val:.2f} |")

md_lines += [
    "",
    "## Notes",
    "",
    "- Frequencies are normalised per 1 000 words to account for varying syllabus lengths.",
    "- `stopwords.txt` (general English stopwords) was applied during PDF preprocessing.",
    "- `stopwords_2.txt` (academic/institutional boilerplate) was applied here so results reflect climate/design language rather than university context.",
    "- All output files are in `1_keyword_analysis_output/`.",
    "- Log-likelihood (G²) tests whether each word’s frequency in a given year departs",
    "  significantly from its expected rate; see `log_likelihood_plain_english.md` for interpretation.",
]

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
print(f"  Written: {OUT_MD.name}")

print("\nAll done.")
