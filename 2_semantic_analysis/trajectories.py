"""
Two-Period Semantic Trajectory Analysis
========================================
Splits the corpus into two periods — 2020-2022 and 2023-2026 — and trains a
separate Word2Vec embedding space ("semantic landscape") for each. The
2023-2026 space is aligned onto the 2020-2022 space via orthogonal
Procrustes, so each go-word (data/go-words.txt) ends up with one position
per period in a shared coordinate system. The vector distance between a
word's two positions is its "shift" — how much its usage context changed
between the two periods.

Outputs (outputs/trajectories/):
  shift_magnitudes.csv        Every go-word's shift magnitude, ranked
  landscape_2020-2022.png     Static landscape of go-words, period 1 only
  landscape_2023-2026.png     Static landscape of go-words, period 2 only
  trajectories_pca_arrows.png All go-words as period1→period2 arrows (PCA),
                               colored by shift magnitude; only the biggest
                               movers are labeled to keep it legible
  trajectories_umap_arrows.png Same idea, projected with UMAP instead of PCA
  top_movers_focus.png        Only the biggest movers, fully labeled
  shift_magnitude_bar.png     Ranked bar chart of shift magnitude
  trajectory_quiver_field.png Unlabeled arrow field — the overall directional
                               "flow" of the semantic space between periods
  landscape_*_aligned.png,    Pixel-aligned versions of the two landscapes +
  top_movers_aligned.png      the top-movers arrows — same axes/canvas, so
                               stacking the PNGs lines the words up exactly
  trajectories_before_after_gradient.png
                               Every word labeled at both its before (yellow)
                               and after (red) position, joined by a
                               yellow→red gradient arrow; same aligned frame
  architecture-words/, keywords-2/, top-movers/
                               Aligned landscape pairs with just that list's
                               words highlighted in red, rest grey; plus a
                               before/after gradient plot scoped to just
                               that list's words
  architecture_keywords/      Aligned landscape pair with architecture-words
                               in red and keywords-2 in yellow (rest grey);
                               words in both lists are drawn red
"""

import re
import pathlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

import umap as umap_lib

matplotlib.rcParams["font.family"] = "Roboto"
matplotlib.rcParams["font.weight"] = "normal"
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_DIR         = pathlib.Path(__file__).parent
DATA_PATH        = BASE_DIR.parent / "data" / "syllabi_text.csv"
GO_WORDS_FILE    = BASE_DIR.parent / "data" / "go-words.txt"
ARCH_WORDS_FILE  = BASE_DIR.parent / "data" / "architecture-words.txt"
KEYWORDS2_FILE   = BASE_DIR.parent / "data" / "keywords-2.txt"
OUT_DIR       = BASE_DIR / "outputs" / "trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERIODS = {
    "2020-2022": [2020, 2021, 2022],
    "2023-2026": [2023, 2024, 2025, 2026],
}
PERIOD_LABELS = list(PERIODS.keys())
REF_LABEL, OTHER_LABEL = PERIOD_LABELS

W2V_PARAMS = dict(
    vector_size=100, window=8, min_count=3,
    workers=4, epochs=40, sg=1, seed=42,
)

TOP_MOVERS_N   = 20   # how many words in the fully-labeled "top movers" view
LABEL_TOP_N    = 25   # how many words get labels in the "all words" arrow plots
BAR_TOP_N      = 40   # how many bars in the ranked bar chart
GO_WORDS_COLOR = "#666666"
HIGHLIGHT_COLOR = "#c0392b"   # red — used for the three highlight-recolored landscapes
BEFORE_COLOR    = "#e08e0b"   # orange-yellow — "before" period in the gradient plot
AFTER_COLOR     = "#c0392b"   # red — "after" period in the gradient plot
ENABLE_REPULSION = True
_REPEL_ITERS      = 40
_REPEL_STEP       = 0.5

# ── Helpers ──────────────────────────────────────────────────────────────────
def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", str(text).lower())


def parse_go_words_file(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        term_part, _, aliases_str = line.partition("\t")
        term = term_part.strip().lower()
        if not term:
            continue
        aliases = [a.strip().lower() for a in aliases_str.split(",") if a.strip()]
        groups.append((term, [term] + aliases))
    return groups


def parse_plain_word_list(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """One bare word per line (data/architecture-words.txt)."""
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.append(line)
    return [(w, [w]) for w in words]


def parse_comma_keywords(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """Comma-separated aliases per line, first alias = canonical label (data/keywords-2.txt)."""
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            aliases = [a.strip().lower() for a in line.split(",") if a.strip()]
            if aliases:
                groups.append((aliases[0], aliases))
    return groups


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0


def align_to_reference(source: Word2Vec, ref: Word2Vec) -> dict[str, np.ndarray]:
    shared = list(set(source.wv.index_to_key) & set(ref.wv.index_to_key))
    if not shared:
        return {}
    A = np.array([source.wv[w] for w in shared])
    B = np.array([ref.wv[w]    for w in shared])
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    R, _ = orthogonal_procrustes(A, B)
    out: dict[str, np.ndarray] = {}
    for word in source.wv.index_to_key:
        v = source.wv[word]
        out[word] = (v / (np.linalg.norm(v) + 1e-9)) @ R
    return out


def repel_labels(positions: dict[str, tuple[float, float]],
                  font_sizes: dict[str, int], ax) -> dict[str, tuple[float, float]]:
    """Iteratively push label positions apart in display pixels to reduce overlap."""
    words = list(positions.keys())
    n = len(words)
    if n < 2:
        return positions

    dpi      = ax.figure.dpi
    pt_to_px = dpi / 72.0
    trans    = ax.transData
    inv      = ax.transData.inverted()
    pos = np.array([trans.transform(positions[w]) for w in words], dtype=float)

    hw = np.array([len(w) * font_sizes.get(w, 8) * pt_to_px * 0.30 + 2.0 for w in words])
    hh = np.array([font_sizes.get(w, 8) * pt_to_px * 0.55 + 2.0 for w in words])

    idx_tiebreak = np.sign(np.arange(n)[:, None] - np.arange(n)[None, :])

    for _ in range(_REPEL_ITERS):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        ov_x = np.maximum(0.0, hw[:, None] + hw[None, :] - np.abs(dx))
        ov_y = np.maximum(0.0, hh[:, None] + hh[None, :] - np.abs(dy))
        mask = (ov_x > 0) & (ov_y > 0)
        np.fill_diagonal(mask, False)
        if not mask.any():
            break
        dx_dir = np.where(np.abs(dx) > 1e-6, np.sign(dx), idx_tiebreak)
        dy_dir = np.where(np.abs(dy) > 1e-6, np.sign(dy), idx_tiebreak)
        push_x = np.where(mask, ov_x * dx_dir, 0.0)
        push_y = np.where(mask, ov_y * dy_dir, 0.0)
        pos[:, 0] += _REPEL_STEP * push_x.sum(axis=1)
        pos[:, 1] += _REPEL_STEP * push_y.sum(axis=1)

    return {w: tuple(inv.transform(pos[i])) for i, w in enumerate(words)}


def fit_view_to(ax, points: list[tuple[float, float]]) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    ax.set_xlim(min(xs) - x_range * 0.12 - 0.05, max(xs) + x_range * 0.12 + 0.05)
    ax.set_ylim(min(ys) - y_range * 0.12 - 0.05, max(ys) + y_range * 0.12 + 0.05)


def style_axes(ax, title: str) -> None:
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


# ── Load go-words + the two highlight word lists ──────────────────────────────
gw_groups = parse_go_words_file(GO_WORDS_FILE)
print(f"Loaded {len(gw_groups)} go-words from {GO_WORDS_FILE.name}")

arch_entries = parse_plain_word_list(ARCH_WORDS_FILE)
kw2_entries  = parse_comma_keywords(KEYWORDS2_FILE)
print(f"Loaded {len(arch_entries)} terms from {ARCH_WORDS_FILE.name}")
print(f"Loaded {len(kw2_entries)} terms from {KEYWORDS2_FILE.name}")


def highlight_set_from(entries: list[tuple[str, list[str]]]) -> set[str]:
    """Go-word canonical terms whose label/aliases match any of `entries`.

    The landscape's vocabulary is fixed to go-words.txt, so a highlight list
    can only recolor points that already exist in it — words in `entries`
    with no corresponding go-word simply have nothing to recolor.
    """
    match_pool = {a.lower() for _, aliases in entries for a in aliases}
    return {term for term, aliases in gw_groups
            if term in match_pool or match_pool.intersection(aliases)}

# ── Load data & bucket into the two periods ────────────────────────────────
print("\nLoading data …")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["filtered_text", "year"])
df["year"] = df["year"].astype(int)

period_sentences: dict[str, list[list[str]]] = {}
period_freq: dict[str, Counter] = {}
for label, yrs in PERIODS.items():
    grp = df[df["year"].isin(yrs)]
    sents = []
    for txt in grp["filtered_text"].dropna():
        tokens = tokenise(txt)
        for i in range(0, max(1, len(tokens) - 59), 30):
            chunk = tokens[i:i + 60]
            if len(chunk) >= 5:
                sents.append(chunk)
    period_sentences[label] = sents
    c: Counter = Counter()
    for s in sents:
        c.update(s)
    period_freq[label] = c
    print(f"  {label}: {len(grp)} docs → {len(sents)} training sentences")

# ── Train one Word2Vec model per period — each model IS the "semantic
#    landscape" for that period (100-dimensional) ─────────────────────────────
print("\nTraining per-period Word2Vec models …")
period_models: dict[str, Word2Vec] = {}
for label in PERIOD_LABELS:
    print(f"  {label} …", end=" ")
    m = Word2Vec(sentences=period_sentences[label], **W2V_PARAMS)
    period_models[label] = m
    print(f"vocab={len(m.wv)}")

# ── Align period 2 onto period 1's space (period 1 = reference frame) ─────────
print("\nAligning period-2 space onto period-1 (orthogonal Procrustes) …")
ref_model = period_models[REF_LABEL]
ref_vecs = {
    w: ref_model.wv[w] / (np.linalg.norm(ref_model.wv[w]) + 1e-9)
    for w in ref_model.wv.index_to_key
}
aligned_vecs = {
    REF_LABEL:   ref_vecs,
    OTHER_LABEL: align_to_reference(period_models[OTHER_LABEL], ref_model),
}
print(f"  {REF_LABEL}: {len(aligned_vecs[REF_LABEL])} words (reference)")
print(f"  {OTHER_LABEL}: {len(aligned_vecs[OTHER_LABEL])} aligned words")


def resolve_vec(aliases: list[str], vecs: dict[str, np.ndarray]) -> np.ndarray | None:
    for a in aliases:
        if a in vecs:
            return vecs[a]
    return None


# ── Resolve each go-word's vector per period ──────────────────────────────────
word_period_vecs: dict[str, dict[str, np.ndarray]] = {}
for term, aliases in gw_groups:
    v_ref   = resolve_vec(aliases, aligned_vecs[REF_LABEL])
    v_other = resolve_vec(aliases, aligned_vecs[OTHER_LABEL])
    entry = {}
    if v_ref is not None:
        entry[REF_LABEL] = v_ref
    if v_other is not None:
        entry[OTHER_LABEL] = v_other
    if entry:
        word_period_vecs[term] = entry

present_both = [t for t, d in word_period_vecs.items()
                if REF_LABEL in d and OTHER_LABEL in d]
skipped = [t for t, _ in gw_groups if t not in present_both]
print(f"\n{len(present_both)} / {len(gw_groups)} go-words present in both periods "
      f"(have a real trajectory)")
if skipped:
    print(f"  Skipped (multi-word/hyphenated terms never tokenize as a single "
          f"word, or too rare in one period): {skipped}")

# ── Shift magnitude = cosine distance between the two period positions ────────
shift_rows = []
for term in present_both:
    v1, v2 = word_period_vecs[term][REF_LABEL], word_period_vecs[term][OTHER_LABEL]
    shift_rows.append({
        "term": term,
        "shift": cosine(v1, v2),
        f"freq_{REF_LABEL}":   period_freq[REF_LABEL].get(term, 0),
        f"freq_{OTHER_LABEL}": period_freq[OTHER_LABEL].get(term, 0),
    })

shift_df = pd.DataFrame(shift_rows).sort_values("shift", ascending=False).reset_index(drop=True)
shift_df.to_csv(OUT_DIR / "shift_magnitudes.csv", index=False)
print(f"\nShift magnitudes → {OUT_DIR / 'shift_magnitudes.csv'}")

top_movers  = shift_df.head(TOP_MOVERS_N)["term"].tolist()
label_terms = set(shift_df.head(LABEL_TOP_N)["term"].tolist())
print(f"Top {TOP_MOVERS_N} movers: {top_movers}")

# ── Shared 2-D projections ────────────────────────────────────────────────────
# PCA: fit on each word's average vector (stable shared basis), then project
# each period's actual vector through it — same approach as 2_semantic_analysis.py.
vocab_order = present_both
avg_mat = np.array([np.mean(list(word_period_vecs[t].values()), axis=0) for t in vocab_order])

pca = PCA(n_components=2, random_state=42)
pca.fit(avg_mat)
print(f"\nPCA: {pca.explained_variance_ratio_.sum():.1%} variance explained")

pca_coords: dict[str, dict[str, np.ndarray]] = {label: {} for label in PERIOD_LABELS}
for term in vocab_order:
    for label in PERIOD_LABELS:
        pca_coords[label][term] = pca.transform(
            word_period_vecs[term][label].reshape(1, -1))[0]

# UMAP: fit jointly on the *actual* stacked period-1 + period-2 vectors (not
# the average), so each period's position is a real embedding rather than an
# out-of-sample transform.
print("Fitting UMAP on stacked period vectors …")
stacked = np.array([word_period_vecs[t][REF_LABEL]   for t in vocab_order] +
                    [word_period_vecs[t][OTHER_LABEL] for t in vocab_order])
reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                         metric="cosine", random_state=42)
umap_arr = reducer.fit_transform(stacked)
n = len(vocab_order)
umap_coords: dict[str, dict[str, np.ndarray]] = {
    REF_LABEL:   {t: umap_arr[i]     for i, t in enumerate(vocab_order)},
    OTHER_LABEL: {t: umap_arr[n + i] for i, t in enumerate(vocab_order)},
}

# ── Color scale for shift magnitude ───────────────────────────────────────────
cmap = plt.get_cmap("YlOrRd")
shift_min, shift_max = shift_df["shift"].min(), shift_df["shift"].max()
norm = plt.Normalize(vmin=shift_min, vmax=shift_max)


def shift_color(term: str) -> tuple:
    row = shift_df.loc[shift_df["term"] == term, "shift"]
    return cmap(norm(row.iloc[0])) if len(row) else "#999999"


# ── 1-2: Static per-period landscapes ─────────────────────────────────────────
def draw_landscape(coords: dict[str, np.ndarray], title: str, outpath: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    render = {t: (float(v[0]), float(v[1])) for t, v in coords.items()}
    fit_view_to(ax, list(render.values()))
    if ENABLE_REPULSION:
        font_sizes = {t: 9 for t in render}
        render = repel_labels(render, font_sizes, ax)
        fit_view_to(ax, list(render.values()))
    for term, (x, y) in render.items():
        ax.annotate(term, (x, y), fontsize=9, color=GO_WORDS_COLOR,
                    ha="center", va="center", alpha=0.85, annotation_clip=False)
    style_axes(ax, title)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {outpath.name}")


print("\nDrawing per-period landscapes …")
draw_landscape(pca_coords[REF_LABEL],
               f"Semantic Landscape — {REF_LABEL} (go-words)",
               OUT_DIR / f"landscape_{REF_LABEL}.png")
draw_landscape(pca_coords[OTHER_LABEL],
               f"Semantic Landscape — {OTHER_LABEL} (go-words)",
               OUT_DIR / f"landscape_{OTHER_LABEL}.png")


# ── 3-4 & 6: Arrow trajectory plots (PCA, UMAP, top-movers-only) ─────────────
def draw_arrows(coords: dict[str, dict[str, np.ndarray]], title: str,
                 outpath: pathlib.Path, terms: list[str],
                 label_set: set[str] | None = None, figsize=(11, 9)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    label_set = label_set if label_set is not None else set(terms)

    all_pts = [tuple(coords[REF_LABEL][t]) for t in terms] + \
              [tuple(coords[OTHER_LABEL][t]) for t in terms]
    fit_view_to(ax, all_pts)

    for term in terms:
        start = coords[REF_LABEL][term]
        end   = coords[OTHER_LABEL][term]
        color = shift_color(term)
        is_labeled = term in label_set
        ax.plot(*start, marker="o", markersize=3, color=color,
                alpha=0.9 if is_labeled else 0.5, zorder=2)
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4 if is_labeled else 0.8,
                                     alpha=0.9 if is_labeled else 0.45),
                    zorder=2)

    # Labels for the notable movers only, repelled apart for legibility
    label_pos = {t: (float(coords[OTHER_LABEL][t][0]), float(coords[OTHER_LABEL][t][1]))
                 for t in terms if t in label_set}
    if ENABLE_REPULSION and len(label_pos) >= 2:
        font_sizes = {t: 10 for t in label_pos}
        label_pos = repel_labels(label_pos, font_sizes, ax)
    for term, (x, y) in label_pos.items():
        ax.annotate(term, (x, y), fontsize=10, color=shift_color(term),
                    ha="center", va="center", fontweight="bold",
                    alpha=1.0, zorder=3, annotation_clip=False)

    style_axes(ax, title)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label(f"Shift magnitude ({REF_LABEL} -> {OTHER_LABEL})", fontsize=9)
    fig.tight_layout()
    fig.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {outpath.name}")


print("\nDrawing trajectory arrow plots …")
draw_arrows(pca_coords,
            f"Semantic Trajectories (PCA) — {REF_LABEL} -> {OTHER_LABEL}",
            OUT_DIR / "trajectories_pca_arrows.png", vocab_order, label_terms)
draw_arrows(umap_coords,
            f"Semantic Trajectories (UMAP) — {REF_LABEL} -> {OTHER_LABEL}",
            OUT_DIR / "trajectories_umap_arrows.png", vocab_order, label_terms)
draw_arrows(pca_coords,
            f"Biggest Movers — {REF_LABEL} -> {OTHER_LABEL}",
            OUT_DIR / "top_movers_focus.png", top_movers, set(top_movers), figsize=(9, 8))


# ── 4b: Pixel-aligned overlay set ──────────────────────────────────────────────
# landscape_*_aligned.png and top_movers_aligned.png share one fixed axes
# rect, figure size, and axis-limit window (the union of both periods' full
# 113-word spread), and are saved WITHOUT bbox_inches="tight". That means:
# no per-image auto-cropping, no per-image auto-fit view, so a word's label
# lands on the identical pixel in
# every file — stack the three PNGs and the arrows meet the landscape labels
# exactly. (The original landscape_*.png / top_movers_focus.png each fit
# their own content individually, so they do NOT line up with each other —
# hence separate "_aligned" files rather than reusing those.)
print("\nDrawing pixel-aligned overlay set (landscapes + top movers share one frame) …")
ALIGNED_FIGSIZE = (10, 8)
ALIGNED_RECT    = [0.06, 0.06, 0.88, 0.84]   # [left, bottom, width, height], figure fraction

_all_pts = [tuple(pca_coords[REF_LABEL][t]) for t in vocab_order] + \
           [tuple(pca_coords[OTHER_LABEL][t]) for t in vocab_order]
_xs = [p[0] for p in _all_pts]; _ys = [p[1] for p in _all_pts]
_x_range = max(_xs) - min(_xs) or 1.0
_y_range = max(_ys) - min(_ys) or 1.0
SHARED_XLIM = (min(_xs) - _x_range * 0.12 - 0.05, max(_xs) + _x_range * 0.12 + 0.05)
SHARED_YLIM = (min(_ys) - _y_range * 0.12 - 0.05, max(_ys) + _y_range * 0.12 + 0.05)


def _make_aligned_ax():
    fig = plt.figure(figsize=ALIGNED_FIGSIZE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes(ALIGNED_RECT)
    ax.set_xlim(*SHARED_XLIM)
    ax.set_ylim(*SHARED_YLIM)
    ax.set_facecolor("white")
    return fig, ax


def draw_landscape_aligned(coords: dict[str, np.ndarray], title: str,
                            outpath: pathlib.Path) -> dict[str, tuple[float, float]]:
    fig, ax = _make_aligned_ax()
    render = {t: (float(v[0]), float(v[1])) for t, v in coords.items()}
    if ENABLE_REPULSION:
        render = repel_labels(render, {t: 9 for t in render}, ax)
    for term, (x, y) in render.items():
        ax.annotate(term, (x, y), fontsize=9, color=GO_WORDS_COLOR,
                    ha="center", va="center", alpha=0.85, annotation_clip=False)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(str(outpath), dpi=150)
    plt.close(fig)
    print(f"  → {outpath.name}")
    return render   # final (repelled) label positions actually drawn


repelled_ref = draw_landscape_aligned(
    pca_coords[REF_LABEL], f"Semantic Landscape — {REF_LABEL} (go-words)",
    OUT_DIR / f"landscape_{REF_LABEL}_aligned.png")
repelled_other = draw_landscape_aligned(
    pca_coords[OTHER_LABEL], f"Semantic Landscape — {OTHER_LABEL} (go-words)",
    OUT_DIR / f"landscape_{OTHER_LABEL}_aligned.png")

fig, ax = _make_aligned_ax()
for term in top_movers:
    start, end = repelled_ref[term], repelled_other[term]
    color = shift_color(term)
    ax.plot(*start, marker="o", markersize=4, color=color, alpha=0.9, zorder=2)
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, alpha=0.9), zorder=2)
ax.set_title(f"Biggest Movers (aligned) — {REF_LABEL} -> {OTHER_LABEL}", fontsize=13)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
fig.savefig(str(OUT_DIR / "top_movers_aligned.png"), dpi=150)
plt.close(fig)
print(f"  → top_movers_aligned.png")


# ── 4c: Highlight-recolored landscape pairs ────────────────────────────────────
# Same points, same (repelled_ref / repelled_other) positions as the aligned
# landscapes above — just recolored red for whichever list is being
# highlighted this round, grey otherwise. Each pair gets its own subfolder.
arch_highlight       = highlight_set_from(arch_entries)
kw2_highlight        = highlight_set_from(kw2_entries)
top_movers_highlight = set(top_movers)

print(f"\narchitecture-words.txt: {len(arch_entries)} input terms -> "
      f"{len(arch_highlight)} go-word entries highlighted")
print(f"keywords-2.txt: {len(kw2_entries)} input terms -> "
      f"{len(kw2_highlight)} go-word entries highlighted")


def draw_landscape_recolored(render: dict[str, tuple[float, float]],
                              highlight_set: set[str], title: str,
                              outpath: pathlib.Path) -> None:
    fig, ax = _make_aligned_ax()
    # Grey words first, highlighted words drawn last (on top) and bold.
    for term, (x, y) in render.items():
        if term in highlight_set:
            continue
        ax.annotate(term, (x, y), fontsize=9, color=GO_WORDS_COLOR,
                    ha="center", va="center", alpha=0.75, annotation_clip=False, zorder=2)
    for term in highlight_set:
        if term not in render:
            continue
        x, y = render[term]
        ax.annotate(term, (x, y), fontsize=9, color=HIGHLIGHT_COLOR, fontweight="bold",
                    ha="center", va="center", alpha=1.0, annotation_clip=False, zorder=3)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(str(outpath), dpi=150)
    plt.close(fig)
    print(f"  → {outpath.relative_to(OUT_DIR)}")


print("\nDrawing highlight-recolored landscape pairs …")
for sub_name, hset in [("architecture-words", arch_highlight),
                        ("keywords-2", kw2_highlight),
                        ("top-movers", top_movers_highlight)]:
    sub_dir = OUT_DIR / sub_name
    sub_dir.mkdir(exist_ok=True)
    draw_landscape_recolored(
        repelled_ref, hset,
        f"Semantic Landscape — {REF_LABEL} ({sub_name} in red)",
        sub_dir / f"landscape_{REF_LABEL}_aligned.png")
    draw_landscape_recolored(
        repelled_other, hset,
        f"Semantic Landscape — {OTHER_LABEL} ({sub_name} in red)",
        sub_dir / f"landscape_{OTHER_LABEL}_aligned.png")


# ── 4c-2: Combined architecture-words + keywords-2 landscape pair ─────────────
# architecture-words in red, keywords-2 in yellow, rest grey. A word in both
# lists is drawn red (architecture wins the override).
def draw_landscape_combined(render: dict[str, tuple[float, float]],
                             red_set: set[str], yellow_set: set[str],
                             title: str, outpath: pathlib.Path) -> None:
    fig, ax = _make_aligned_ax()
    for term, (x, y) in render.items():
        if term in red_set or term in yellow_set:
            continue
        ax.annotate(term, (x, y), fontsize=9, color=GO_WORDS_COLOR,
                    ha="center", va="center", alpha=0.75, annotation_clip=False, zorder=2)
    for term in yellow_set - red_set:
        if term not in render:
            continue
        x, y = render[term]
        ax.annotate(term, (x, y), fontsize=9, color=BEFORE_COLOR, fontweight="bold",
                    ha="center", va="center", alpha=1.0, annotation_clip=False, zorder=3)
    for term in red_set:
        if term not in render:
            continue
        x, y = render[term]
        ax.annotate(term, (x, y), fontsize=9, color=HIGHLIGHT_COLOR, fontweight="bold",
                    ha="center", va="center", alpha=1.0, annotation_clip=False, zorder=4)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(str(outpath), dpi=150)
    plt.close(fig)
    print(f"  → {outpath.relative_to(OUT_DIR)}")


print("\nDrawing combined architecture-words + keywords-2 landscape pair …")
combined_dir = OUT_DIR / "architecture_keywords"
combined_dir.mkdir(exist_ok=True)
draw_landscape_combined(
    repelled_ref, arch_highlight, kw2_highlight,
    f"Semantic Landscape — {REF_LABEL} (architecture red, keywords yellow)",
    combined_dir / f"landscape_{REF_LABEL}_aligned.png")
draw_landscape_combined(
    repelled_other, arch_highlight, kw2_highlight,
    f"Semantic Landscape — {OTHER_LABEL} (architecture red, keywords yellow)",
    combined_dir / f"landscape_{OTHER_LABEL}_aligned.png")


# ── 4d: Before/after gradient plot — every word at both positions ─────────────
# "Before" (period 1) label in yellow, "after" (period 2) label in red,
# joined by a yellow→red gradient arrow. Same aligned frame/points as above.
print("\nDrawing before/after gradient plots …")
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

_grad_cmap = mcolors.LinearSegmentedColormap.from_list("before_after", [BEFORE_COLOR, AFTER_COLOR])
_GRAD_SEGMENTS = 24
_shift_lookup = dict(zip(shift_df["term"], shift_df["shift"]))

# Trim this much off each end of the connecting line/arrow (in data units) so
# it stops short of the word labels instead of running through their text.
_GRAD_GAP = 0.028 * min(SHARED_XLIM[1] - SHARED_XLIM[0], SHARED_YLIM[1] - SHARED_YLIM[0])


def draw_before_after_gradient(terms: list[str], title: str, outpath: pathlib.Path,
                                fontsize: int = 8, lw_grad: float = 1.0,
                                lw_arrow: float = 1.2) -> None:
    fig, ax = _make_aligned_ax()
    for term in terms:
        raw_p0 = np.array(repelled_ref[term])
        raw_p1 = np.array(repelled_other[term])
        d = raw_p1 - raw_p0
        dist = np.linalg.norm(d)
        if dist <= 2 * _GRAD_GAP:
            continue   # too short to trim without the connector disappearing
        unit = d / dist
        p0 = raw_p0 + unit * _GRAD_GAP
        p1 = raw_p1 - unit * _GRAD_GAP

        xs = np.linspace(p0[0], p1[0], _GRAD_SEGMENTS)
        ys = np.linspace(p0[1], p1[1], _GRAD_SEGMENTS)
        pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        seg_colors = [_grad_cmap(i / max(_GRAD_SEGMENTS - 2, 1)) for i in range(_GRAD_SEGMENTS - 1)]
        ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=lw_grad, alpha=0.65, zorder=1))
        ax.annotate("", xy=p1, xytext=tuple(pts[-2][0]),
                    arrowprops=dict(arrowstyle="-|>", color=AFTER_COLOR, lw=lw_arrow, alpha=0.9), zorder=2)

    for term in terms:
        ax.annotate(term, repelled_ref[term], fontsize=fontsize, color=BEFORE_COLOR,
                    ha="center", va="center", alpha=0.95, annotation_clip=False, zorder=3)
    for term in terms:
        ax.annotate(term, repelled_other[term], fontsize=fontsize, color=AFTER_COLOR, fontweight="bold",
                    ha="center", va="center", alpha=0.95, annotation_clip=False, zorder=4)

    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(str(outpath), dpi=150)
    plt.close(fig)
    print(f"  → {outpath.relative_to(OUT_DIR)}")


draw_before_after_gradient(
    vocab_order, f"Word Positions & Trajectories — {REF_LABEL} (yellow) -> {OTHER_LABEL} (red)",
    OUT_DIR / "trajectories_before_after_gradient.png")

for sub_name, hset in [("architecture-words", arch_highlight),
                        ("keywords-2", kw2_highlight),
                        ("top-movers", top_movers_highlight)]:
    terms = sorted(hset & set(vocab_order), key=lambda t: -_shift_lookup.get(t, 0.0))
    draw_before_after_gradient(
        terms,
        f"Word Positions & Trajectories ({sub_name}) — {REF_LABEL} (yellow) -> {OTHER_LABEL} (red)",
        OUT_DIR / sub_name / "trajectories_before_after_gradient.png",
        fontsize=10, lw_grad=1.4, lw_arrow=1.8)


# ── 5: Ranked bar chart ────────────────────────────────────────────────────────
print("\nDrawing shift-magnitude bar chart …")
bar_df = shift_df.head(BAR_TOP_N)
fig, ax = plt.subplots(figsize=(9, max(6, len(bar_df) * 0.25)))
colors = [shift_color(t) for t in bar_df["term"]]
ax.barh(bar_df["term"][::-1], bar_df["shift"][::-1], color=colors[::-1],
        edgecolor="white", height=0.7)
ax.set_xlabel(f"Cosine distance magnitude ({REF_LABEL} -> {OTHER_LABEL})", fontsize=11)
ax.set_title(f"Go-Word Shift Magnitude — top {len(bar_df)} of {len(shift_df)}", fontsize=13)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", length=0)
fig.tight_layout()
fig.savefig(str(OUT_DIR / "shift_magnitude_bar.png"), dpi=150)
plt.close(fig)
print(f"  → shift_magnitude_bar.png")


# ── 7: Unlabeled quiver field — overall directional "flow" ────────────────────
print("\nDrawing quiver field …")
fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor("white")
starts = np.array([pca_coords[REF_LABEL][t] for t in vocab_order])
ends   = np.array([pca_coords[OTHER_LABEL][t] for t in vocab_order])
deltas = ends - starts
colors = [shift_color(t) for t in vocab_order]
ax.quiver(starts[:, 0], starts[:, 1], deltas[:, 0], deltas[:, 1],
          angles="xy", scale_units="xy", scale=1, color=colors,
          width=0.003, alpha=0.85)
fit_view_to(ax, [tuple(p) for p in starts] + [tuple(p) for p in ends])
style_axes(ax, f"Semantic Flow Field — {REF_LABEL} -> {OTHER_LABEL}")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
cbar.set_label("Shift magnitude", fontsize=9)
fig.tight_layout()
fig.savefig(str(OUT_DIR / "trajectory_quiver_field.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  → trajectory_quiver_field.png")

print(f"\n✓ Trajectory analysis complete. Outputs → {OUT_DIR}")
