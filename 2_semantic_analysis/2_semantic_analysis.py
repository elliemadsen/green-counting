"""
Syllabi Semantic Analysis — Diachronic Word Embedding Study
===========================================================
Trains a Word2Vec model per year-slice, aligns vector spaces via
orthogonal Procrustes (Hamilton et al. 2016), then produces:

  1. Nearest-neighbour tables for target terms per year
  2. Semantic shift magnitude scores
  3. Per-year landscape PNGs + per-subdir GIF:
       keywords_nearest_neighbors/ — keywords + NNs (top-500 limited)
       keywords/                   — 30 keywords only (dark grey)
       keywords_top100/            — top-100 words, keywords highlighted
       keywords_top500/            — top-500 words, keywords highlighted
  4. Topographic concentric diagrams (top-500 limited) → concentric_diagrams/
  5. Semantic trajectory plot
"""

import math
import re
import warnings
import pathlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.spatial.distance import cosine
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

import umap as umap_lib

matplotlib.rcParams["font.family"] = "Roboto"
matplotlib.rcParams["font.weight"] = "normal"

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR      = pathlib.Path(__file__).parent
DATA_PATH     = BASE_DIR.parent / "data" / "syllabi_text.csv"
KEYWORDS_FILE  = BASE_DIR.parent / "data" / "keywords.txt"
KEYWORDS_FILE_2 = BASE_DIR.parent / "data" / "keywords-2.txt"
TOP500_CSV    = BASE_DIR.parent / "0_preprocessing" / "outputs" / "top500_corpus.csv"
OUT_DIR       = BASE_DIR / "outputs"

KW_DIR               = OUT_DIR / "keywords"
LANDSCAPE_NN_KW_DIR  = KW_DIR / "nearest-neighbors"
LANDSCAPE_KW_DIR     = KW_DIR / "go-words"
LANDSCAPE_TOP100_DIR = KW_DIR / "top-100"
LANDSCAPE_TOP200_DIR = KW_DIR / "top-200"
LANDSCAPE_TOP500_DIR = KW_DIR / "top-500"
CONC_DIAG_DIR        = KW_DIR / "concentric-diagrams"
UMAP_DIR             = KW_DIR / "umap"

for _d in [OUT_DIR, KW_DIR, LANDSCAPE_NN_KW_DIR, LANDSCAPE_KW_DIR,
           LANDSCAPE_TOP100_DIR, LANDSCAPE_TOP200_DIR,
           LANDSCAPE_TOP500_DIR, CONC_DIAG_DIR, UMAP_DIR]:
    _d.mkdir(exist_ok=True)

W2V_PARAMS = dict(
    vector_size=100, window=8, min_count=3,
    workers=4, epochs=40, sg=1, seed=42,
)

TOP_N          = 12    # NNs per term for ring diagrams (4 words × 3 rings)
KW_NN_N        = 20    # NNs per keyword alias for landscape
TOP_WORDS_N    = 100   # corpus-wide top words for keywords_top100
TOP200_WORDS_N = 200   # corpus-wide top words for keywords_top200
MIN_FREQ       = 50    # within-year frequency floor (used for top100/top500 landscapes)
TARGET_COLOR   = "#c0392b"   # red — keywords in landscapes
KEYWORDS_COLOR = "#444444"   # dark grey (kept for backwards compat)
GO_WORDS_COLOR = "#888888"   # grey — go-words in combined landscape
GO_WORDS_FILE  = BASE_DIR.parent / "data" / "go-words.txt"

ENABLE_REPULSION = True   # push labels apart to reduce overlap in landscape outputs
_REPEL_ITERS     = 20     # max iterations of the repulsion loop
_REPEL_STEP      = 0.5    # fraction of each overlap resolved per iteration

# ── Load keywords ──────────────────────────────────────────────────────────────
def parse_keywords(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    groups = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            aliases = [a.strip().lower() for a in line.split(",") if a.strip()]
            if aliases:
                groups.append((aliases[0], aliases))
    return groups


def parse_go_words_file(path: pathlib.Path) -> list[tuple[str, list[str]]]:
    """Parse go-words.txt: tab-separated — term TAB comma-separated aliases."""
    groups = []
    if not path.exists():
        print(f"  [WARN] {path} not found")
        return groups
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


keyword_groups = parse_keywords(KEYWORDS_FILE)
TARGET_TERMS   = [label for label, _ in keyword_groups]
KW_ALL_ALIASES = set(a for _, aliases in keyword_groups for a in aliases)
print(f"Loaded {len(keyword_groups)} keyword groups: {TARGET_TERMS}")

keyword_groups_2 = parse_keywords(KEYWORDS_FILE_2)
TARGET_TERMS_2   = [label for label, _ in keyword_groups_2]
KW_ALL_ALIASES_2 = set(a for _, aliases in keyword_groups_2 for a in aliases)
print(f"Loaded {len(keyword_groups_2)} keyword-2 groups: {TARGET_TERMS_2}")

gw_groups       = parse_go_words_file(GO_WORDS_FILE)
TARGET_TERMS_GW = [label for label, _ in gw_groups]
GW_ALL_ALIASES  = set(a for _, aliases in gw_groups for a in aliases)
print(f"Loaded {len(gw_groups)} go-word groups")

# ── Load top-500 word list from step 0 ────────────────────────────────────────
if TOP500_CSV.exists():
    top_500_words: list[str] = list(pd.read_csv(TOP500_CSV)["word"])
    print(f"Loaded {len(top_500_words)} top-500 words from {TOP500_CSV.name}")
else:
    print(f"Warning: {TOP500_CSV} not found — will compute top-500 from corpus")
    top_500_words = []   # filled in after corpus counts are available
top_500_set: set[str] = set(top_500_words)

# ── 1. Load & tokenise ─────────────────────────────────────────────────────────
print("Loading data …")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["filtered_text", "year"])
df["year"] = df["year"].astype(int)

years = sorted(df["year"].unique())
print(f"Years: {years}  |  Total docs: {len(df)}")

def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())

year_sentences: dict[int, list[list[str]]] = {}
for yr, grp in df.groupby("year"):
    sents = []
    for txt in grp["filtered_text"].dropna():
        tokens = tokenise(txt)
        for i in range(0, max(1, len(tokens) - 59), 30):
            chunk = tokens[i : i + 60]
            if len(chunk) >= 5:
                sents.append(chunk)
    year_sentences[yr] = sents
    print(f"  {yr}: {len(grp)} docs → {len(sents)} training sentences")

all_sentences = [s for sents in year_sentences.values() for s in sents]

# ── Per-year and corpus-wide word frequencies ──────────────────────────────────
word_freq_by_year: dict[int, Counter] = {}
for yr in years:
    c: Counter = Counter()
    for sent in year_sentences[yr]:
        c.update(sent)
    word_freq_by_year[yr] = c

corpus_counts: Counter = Counter()
for c in word_freq_by_year.values():
    corpus_counts.update(c)

top_words:     list[str] = [w for w, _ in corpus_counts.most_common(TOP_WORDS_N)]
top_200_words: list[str] = [w for w, _ in corpus_counts.most_common(TOP200_WORDS_N)]
if not top_500_words:
    top_500_words = [w for w, _ in corpus_counts.most_common(500)]
    top_500_set   = set(top_500_words)
    print(f"Computed top-500 from corpus: {top_500_words[:5]} …")

print(f"\nTop-{TOP_WORDS_N} corpus words: {top_words[:5]} …")
print(f"Top-500 corpus words loaded: {top_500_words[:5]} …")

# ── 2. Train models ────────────────────────────────────────────────────────────
print("\nTraining Word2Vec models …")
print("  Global model …")
global_model = Word2Vec(sentences=all_sentences, **W2V_PARAMS)

year_models: dict[int, Word2Vec] = {}
for yr in years:
    print(f"  {yr} …", end=" ")
    m = Word2Vec(sentences=year_sentences[yr], **W2V_PARAMS)
    year_models[yr] = m
    print(f"vocab={len(m.wv)}")

# ── 3. Procrustes alignment ────────────────────────────────────────────────────
print("\nAligning vector spaces (orthogonal Procrustes) …")

def align_to_reference(source: Word2Vec, ref: Word2Vec) -> dict[str, np.ndarray]:
    shared = list(set(source.wv.index_to_key) & set(ref.wv.index_to_key))
    if not shared:
        return {}
    A = np.array([source.wv[w] for w in shared])
    B = np.array([ref.wv[w]    for w in shared])
    A /= np.linalg.norm(A, axis=0, keepdims=True) + 1e-9
    B /= np.linalg.norm(B, axis=0, keepdims=True) + 1e-9
    R, _ = orthogonal_procrustes(A, B)
    out: dict[str, np.ndarray] = {}
    for word in source.wv.index_to_key:
        v = source.wv[word]
        out[word] = (v / (np.linalg.norm(v) + 1e-9)) @ R
    return out

aligned_vecs: dict[int, dict[str, np.ndarray]] = {}
for yr, model in year_models.items():
    aligned_vecs[yr] = align_to_reference(model, global_model)
    print(f"  {yr}: {len(aligned_vecs[yr])} aligned words")

# ── 4. Nearest-neighbour helpers ───────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-9 else 0.0

def nearest_neighbours(word: str, vecs: dict[str, np.ndarray],
                        n: int = TOP_N) -> list[tuple[str, float]]:
    if word not in vecs:
        return []
    wv = vecs[word]
    sims = [(w, cosine_sim(wv, v)) for w, v in vecs.items() if w != word]
    sims.sort(key=lambda x: -x[1])
    return sims[:n]

def nearest_neighbours_top500(word: str, vecs: dict[str, np.ndarray],
                               n: int = TOP_N) -> list[tuple[str, float]]:
    """NNs restricted to top-500 corpus words."""
    if word not in vecs:
        return []
    wv = vecs[word]
    sims = [(w, cosine_sim(wv, v)) for w, v in vecs.items()
            if w != word and w in top_500_set]
    sims.sort(key=lambda x: -x[1])
    return sims[:n]

# ── 5. Compute NNs for all target terms (top-500 filtered) ────────────────────
print("\n" + "=" * 70)
print("NEAREST NEIGHBOURS BY YEAR  (limited to top-500 corpus words)")
print("=" * 70)

nn_results: dict[str, dict[int, list]] = defaultdict(dict)
for term in TARGET_TERMS:
    print(f"\n── {term.upper()} ──")
    for yr in years:
        nns = nearest_neighbours_top500(term, aligned_vecs[yr])
        nn_results[term][yr] = nns
        if nns:
            print(f"  {yr}: {', '.join(w for w, _ in nns)}")
        else:
            print(f"  {yr}: (not in vocabulary / no top-500 NNs)")

print("\n" + "=" * 70)
print("NEAREST NEIGHBOURS BY YEAR  (keywords-2, limited to top-500 corpus words)")
print("=" * 70)

nn_results_2: dict[str, dict[int, list]] = defaultdict(dict)
for term in TARGET_TERMS_2:
    print(f"\n── {term.upper()} ──")
    for yr in years:
        nns = nearest_neighbours_top500(term, aligned_vecs[yr])
        nn_results_2[term][yr] = nns
        if nns:
            print(f"  {yr}: {', '.join(w for w, _ in nns)}")
        else:
            print(f"  {yr}: (not in vocabulary / no top-500 NNs)")

# ── 6. Semantic shift scores (magnitude only) ──────────────────────────────────
print("\n" + "=" * 70)
print("SEMANTIC SHIFT SCORES  (cosine distance magnitude, first → last year)")
print("=" * 70)

first_yr, last_yr = years[0], years[-1]
shift_scores: dict[str, float] = {}
for term in TARGET_TERMS:
    v0 = aligned_vecs[first_yr].get(term)
    v1 = aligned_vecs[last_yr].get(term)
    if v0 is not None and v1 is not None:
        dist = cosine(v0, v1)
        shift_scores[term] = dist
        print(f"  {term:20s}  |Δ| = {dist:.4f}")
    else:
        print(f"  {term:20s}  (missing in {first_yr} or {last_yr})")

print("\nSEMANTIC SHIFT SCORES  (keywords-2, cosine distance magnitude, first → last year)")
shift_scores_2: dict[str, float] = {}
for term in TARGET_TERMS_2:
    v0 = aligned_vecs[first_yr].get(term)
    v1 = aligned_vecs[last_yr].get(term)
    if v0 is not None and v1 is not None:
        dist = cosine(v0, v1)
        shift_scores_2[term] = dist
        print(f"  {term:20s}  |Δ| = {dist:.4f}")
    else:
        print(f"  {term:20s}  (missing in {first_yr} or {last_yr})")

# ── 7. Save NN table CSV ───────────────────────────────────────────────────────
nn_rows = []
for term in TARGET_TERMS:
    for yr in years:
        for rank, (word, sim) in enumerate(nn_results[term][yr], 1):
            nn_rows.append({"target": term, "year": yr, "rank": rank,
                             "neighbour": word, "cosine_sim": round(sim, 4)})
pd.DataFrame(nn_rows).to_csv(KW_DIR / "nearest_neighbours.csv", index=False)
print(f"\nNearest-neighbour table → {KW_DIR / 'nearest_neighbours.csv'}")

nn2_rows = []
for term in TARGET_TERMS_2:
    for yr in years:
        for rank, (word, sim) in enumerate(nn_results_2[term][yr], 1):
            nn2_rows.append({"target": term, "year": yr, "rank": rank,
                              "neighbour": word, "cosine_sim": round(sim, 4)})
pd.DataFrame(nn2_rows).to_csv(OUT_DIR_2 / "nearest_neighbours.csv", index=False)
print(f"Nearest-neighbour-2 table → {OUT_DIR_2 / 'nearest_neighbours.csv'}")

# ── 8. Semantic shift magnitude bar chart ─────────────────────────────────────
if shift_scores:
    terms_sorted = sorted(shift_scores, key=shift_scores.get, reverse=True)
    vals    = [shift_scores[t] for t in terms_sorted]
    fig, ax = plt.subplots(figsize=(9, max(4, len(terms_sorted) * 0.35)))
    ax.barh(terms_sorted, vals, color="#555555", edgecolor="white", height=0.6)
    ax.set_xlabel(f"Cosine distance magnitude ({first_yr} → {last_yr})", fontsize=11)
    ax.set_title("Semantic Shift Magnitude of Keywords", fontsize=13)
    ax.axvline(np.mean(vals), color="#aaaaaa", linestyle="--", linewidth=1,
               label=f"Mean ({np.mean(vals):.3f})")
    ax.legend(fontsize=9)
    for val, term in zip(vals, terms_sorted):
        ax.text(val + 0.002, terms_sorted.index(term),
                f"{val:.3f}", va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, max(vals) * 1.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    plt.tight_layout()
    plt.savefig(KW_DIR / "semantic_shift_bar.png", dpi=150)
    plt.close()
    print("Shift magnitude bar chart saved")

if shift_scores_2:
    terms_sorted_2 = sorted(shift_scores_2, key=shift_scores_2.get, reverse=True)
    vals_2  = [shift_scores_2[t] for t in terms_sorted_2]
    fig, ax = plt.subplots(figsize=(9, max(4, len(terms_sorted_2) * 0.35)))
    ax.barh(terms_sorted_2, vals_2, color="#555555", edgecolor="white", height=0.6)
    ax.set_xlabel(f"Cosine distance magnitude ({first_yr} → {last_yr})", fontsize=11)
    ax.set_title("Semantic Shift Magnitude of Keywords-2", fontsize=13)
    ax.axvline(np.mean(vals_2), color="#aaaaaa", linestyle="--", linewidth=1,
               label=f"Mean ({np.mean(vals_2):.3f})")
    ax.legend(fontsize=9)
    for val, term in zip(vals_2, terms_sorted_2):
        ax.text(val + 0.002, terms_sorted_2.index(term),
                f"{val:.3f}", va="center", fontsize=9, color="#333333")
    ax.set_xlim(0, max(vals_2) * 1.25)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR_2 / "semantic_shift_bar.png", dpi=150)
    plt.close()
    print("Shift-2 magnitude bar chart saved")

# ── 9. Build shared PCA space ──────────────────────────────────────────────────
print("\nBuilding shared PCA space …")

# Canonical keyword labels as base; search NNs from all aliases but exclude
# non-canonical alias forms (e.g. "materials", "sustainability") from display
kw_nn_vocab: set[str] = set(TARGET_TERMS)
for alias in KW_ALL_ALIASES:
    for yr in years:
        for w, _ in nearest_neighbours(alias, aligned_vecs[yr], n=KW_NN_N):
            if w in top_500_set and w not in KW_ALL_ALIASES:
                kw_nn_vocab.add(w)

kw2_nn_vocab: set[str] = set(TARGET_TERMS_2)
for alias in KW_ALL_ALIASES_2:
    for yr in years:
        for w, _ in nearest_neighbours(alias, aligned_vecs[yr], n=KW_NN_N):
            if w in top_500_set and w not in KW_ALL_ALIASES_2:
                kw2_nn_vocab.add(w)

# Combined vocabulary (keywords + keywords-2 + go-words + their NNs + top-500 + top-100) for one PCA
combined_vocab = sorted(kw_nn_vocab | kw2_nn_vocab | set(TARGET_TERMS_GW) | top_500_set | set(top_words))
print(f"  Combined vocabulary: {len(combined_vocab)} words")

def avg_vec(word: str) -> np.ndarray | None:
    vecs = [aligned_vecs[yr][word] for yr in years if word in aligned_vecs[yr]]
    return np.mean(vecs, axis=0) if vecs else None

valid_vocab = [w for w in combined_vocab if avg_vec(w) is not None]
mat = np.array([avg_vec(w) for w in valid_vocab])

pca = PCA(n_components=2, random_state=42)
pca.fit(mat)

yr_coords: dict[int, dict[str, np.ndarray]] = {}
for yr in years:
    proj: dict[str, np.ndarray] = {}
    for w in valid_vocab:
        if w in aligned_vecs[yr]:
            proj[w] = pca.transform(aligned_vecs[yr][w].reshape(1, -1))[0]
    yr_coords[yr] = proj
print(f"  PCA on {len(valid_vocab)} words — "
      f"{pca.explained_variance_ratio_.sum():.1%} variance explained")

kw_aliases_set  = set(TARGET_TERMS)    # canonical labels only (first form per keywords.txt line)
kw2_aliases_set = set(TARGET_TERMS_2)  # canonical labels only for keywords-2
gw_canonical_set = set(TARGET_TERMS_GW) # canonical labels for go-words
top_words_set   = set(top_words)
top_200_set     = set(top_200_words)

OUT_DIR_2               = OUT_DIR / "keywords-2"
LANDSCAPE_NN_KW_DIR_2   = OUT_DIR_2 / "nearest-neighbors"
LANDSCAPE_KW_DIR_2      = OUT_DIR_2 / "go-words"
LANDSCAPE_TOP100_DIR_2  = OUT_DIR_2 / "top-100"
LANDSCAPE_TOP200_DIR_2  = OUT_DIR_2 / "top-200"
LANDSCAPE_TOP500_DIR_2  = OUT_DIR_2 / "top-500"
CONC_DIAG_DIR_2         = OUT_DIR_2 / "concentric-diagrams"
UMAP_DIR_2              = OUT_DIR_2 / "umap"

for _d2 in [OUT_DIR_2, LANDSCAPE_NN_KW_DIR_2, LANDSCAPE_KW_DIR_2,
            LANDSCAPE_TOP100_DIR_2, LANDSCAPE_TOP200_DIR_2,
            LANDSCAPE_TOP500_DIR_2, CONC_DIAG_DIR_2, UMAP_DIR_2]:
    _d2.mkdir(exist_ok=True)

# ── 10. Label repulsion helper ────────────────────────────────────────────────
def repel_labels(positions: dict[str, tuple[float, float]],
                 font_sizes: dict[str, int],
                 ax) -> dict[str, tuple[float, float]]:
    """Iteratively push label positions apart in display pixels to reduce overlap."""
    words = list(positions.keys())
    n     = len(words)
    if n < 2:
        return positions

    dpi      = ax.figure.dpi
    pt_to_px = dpi / 72.0
    trans    = ax.transData
    inv      = ax.transData.inverted()

    # Convert data coords → display pixels
    pos = np.array([trans.transform(positions[w]) for w in words], dtype=float)

    # Estimated half-extents per label in pixels (with a small margin)
    hw = np.array([len(w) * font_sizes.get(w, 7) * pt_to_px * 0.30 + 2.0
                   for w in words])
    hh = np.array([font_sizes.get(w, 7) * pt_to_px * 0.55 + 2.0
                   for w in words])

    for _ in range(_REPEL_ITERS):
        # Pairwise displacement matrices: dx[i,j] = x_i - x_j
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]  # (n, n)
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]

        # Axis-aligned overlap amounts (>0 only where bboxes intersect)
        ov_x = np.maximum(0.0, hw[:, None] + hw[None, :] - np.abs(dx))
        ov_y = np.maximum(0.0, hh[:, None] + hh[None, :] - np.abs(dy))
        mask = (ov_x > 0) & (ov_y > 0)
        np.fill_diagonal(mask, False)

        if not mask.any():
            break

        # Push each label by half the overlap in the direction it already leans
        push_x = np.where(mask, ov_x * np.sign(dx + 1e-9), 0.0)
        push_y = np.where(mask, ov_y * np.sign(dy + 1e-9), 0.0)

        pos[:, 0] += _REPEL_STEP * push_x.sum(axis=1)
        pos[:, 1] += _REPEL_STEP * push_y.sum(axis=1)

    # Convert display pixels → data coords
    return {w: tuple(inv.transform(pos[i])) for i, w in enumerate(words)}


# ── 11. Landscape drawing helper ──────────────────────────────────────────────
def draw_landscape_panel(ax, year: int, word_coords: dict[str, np.ndarray],
                          highlight_set: set[str],
                          apply_freq_filter: bool = True,
                          highlight_color: str = TARGET_COLOR,
                          second_highlight_set: set[str] | None = None,
                          second_highlight_color: str = GO_WORDS_COLOR) -> None:
    ax.clear()
    freq = word_freq_by_year[year]

    # Collect words that pass the frequency filter
    render: dict[str, tuple[float, float]] = {}
    for word, (x, y) in word_coords.items():
        if apply_freq_filter and freq.get(word, 0) < MIN_FREQ:
            continue
        render[word] = (float(x), float(y))

    if not render:
        ax.set_title(f"Semantic Landscape — {year}", fontsize=12)
        ax.set_facecolor("white")
        return

    xs = [p[0] for p in render.values()]
    ys = [p[1] for p in render.values()]

    # Set axis limits from data BEFORE repulsion so the display transform is calibrated
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    xpad = x_range * 0.12 + 0.05
    ypad = y_range * 0.12 + 0.05
    ax.set_xlim(min(xs) - xpad, max(xs) + xpad)
    ax.set_ylim(min(ys) - ypad, max(ys) + ypad)

    # Optionally spread labels apart to reduce overlap
    if ENABLE_REPULSION and len(render) >= 2:
        font_sizes = {w: (10 if (w in highlight_set or
                                  (second_highlight_set and w in second_highlight_set))
                          else 7) for w in render}
        render = repel_labels(render, font_sizes, ax)

    # Draw labels at final positions
    for word, (x, y) in render.items():
        is_kw = word in highlight_set
        is_gw = second_highlight_set is not None and word in second_highlight_set
        color = highlight_color if is_kw else (
                second_highlight_color if is_gw else "#000000")
        ax.annotate(word, (x, y),
                    fontsize=10 if (is_kw or is_gw) else 7,
                    color=color,
                    ha="center", va="center",
                    alpha=1.0 if (is_kw or is_gw) else 0.6)

    ax.set_title(f"Semantic Landscape — {year}", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")

# ── Static (non-annual) landscape panel — used for UMAP output ────────────────
def draw_static_panel(ax, title: str, word_coords: dict[str, np.ndarray],
                       highlight_set: set[str],
                       highlight_color: str = TARGET_COLOR,
                       second_highlight_set: set[str] | None = None,
                       second_highlight_color: str = GO_WORDS_COLOR) -> None:
    ax.clear()
    render = {w: (float(x), float(y)) for w, (x, y) in word_coords.items()}
    if not render:
        ax.set_title(title, fontsize=12)
        ax.set_facecolor("white")
        return

    xs = [p[0] for p in render.values()]
    ys = [p[1] for p in render.values()]
    x_range = max(xs) - min(xs) or 1.0
    y_range = max(ys) - min(ys) or 1.0
    ax.set_xlim(min(xs) - x_range * 0.12 - 0.05, max(xs) + x_range * 0.12 + 0.05)
    ax.set_ylim(min(ys) - y_range * 0.12 - 0.05, max(ys) + y_range * 0.12 + 0.05)

    if ENABLE_REPULSION and len(render) >= 2:
        font_sizes = {w: (10 if (w in highlight_set or
                                  (second_highlight_set and w in second_highlight_set))
                          else 7) for w in render}
        render = repel_labels(render, font_sizes, ax)

    for word, (x, y) in render.items():
        is_kw = word in highlight_set
        is_gw = second_highlight_set is not None and word in second_highlight_set
        color = highlight_color if is_kw else (
                second_highlight_color if is_gw else "#000000")
        ax.annotate(word, (x, y),
                    fontsize=10 if (is_kw or is_gw) else 7,
                    color=color,
                    ha="center", va="center",
                    alpha=1.0 if (is_kw or is_gw) else 0.6)

    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("white")


# ── Landscape GIF helper ───────────────────────────────────────────────────────
def save_landscape_gif(gif_path: pathlib.Path,
                        coord_fn,
                        highlight_set: set[str],
                        apply_freq_filter: bool = True,
                        highlight_color: str = TARGET_COLOR,
                        second_highlight_set: set[str] | None = None,
                        second_highlight_color: str = GO_WORDS_COLOR) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")

    def _update(frame_idx: int) -> None:
        yr = years[frame_idx]
        draw_landscape_panel(ax, yr, coord_fn(yr), highlight_set,
                             apply_freq_filter, highlight_color,
                             second_highlight_set, second_highlight_color)

    ani = FuncAnimation(fig, _update, frames=len(years), interval=1400, repeat=True)
    ani.save(str(gif_path), writer=PillowWriter(fps=0.7))
    plt.close()
    print(f"  GIF → {gif_path.name}")

# ── 11. keywords_nearest_neighbors/ landscapes ────────────────────────────────
# Consistent across years: NNs limited to top-500, no per-year freq filter
print(f"\nSaving keywords_nearest_neighbors panels …")

def _nn_coords(yr: int) -> dict[str, np.ndarray]:
    return {w: v for w, v in yr_coords[yr].items() if w in kw_nn_vocab}

for yr in years:
    coords = _nn_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw_aliases_set, apply_freq_filter=False)
    p = LANDSCAPE_NN_KW_DIR / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_NN_KW_DIR / "animation.gif",
                   _nn_coords, kw_aliases_set, apply_freq_filter=False)

# ── 12. keywords/ landscapes — keywords (red) + go-words (orange) ─────────────
print("\nSaving keywords panels (keywords in red + go-words in orange) …")

_kw_gw_vocab = kw_aliases_set | gw_canonical_set

def _kw_coords(yr: int) -> dict[str, np.ndarray]:
    return {w: v for w, v in yr_coords[yr].items() if w in _kw_gw_vocab}

for yr in years:
    coords = _kw_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw_aliases_set,
                         apply_freq_filter=False, highlight_color=TARGET_COLOR,
                         second_highlight_set=gw_canonical_set,
                         second_highlight_color=GO_WORDS_COLOR)
    p = LANDSCAPE_KW_DIR / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_KW_DIR / "animation.gif",
                   _kw_coords, kw_aliases_set,
                   apply_freq_filter=False, highlight_color=TARGET_COLOR,
                   second_highlight_set=gw_canonical_set,
                   second_highlight_color=GO_WORDS_COLOR)

# ── 13. keywords_top100/ landscapes ──────────────────────────────────────────
print(f"\nSaving keywords_top100 panels (MIN_FREQ={MIN_FREQ}) …")

def _top100_coords(yr: int) -> dict[str, np.ndarray]:
    return {w: v for w, v in yr_coords[yr].items() if w in top_words_set}

for yr in years:
    coords = _top100_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw_aliases_set)
    p = LANDSCAPE_TOP100_DIR / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_TOP100_DIR / "animation.gif",
                   _top100_coords, kw_aliases_set)

# ── 14. keywords_top200/ landscapes ──────────────────────────────────────────
print(f"\nSaving keywords_top200 panels (MIN_FREQ={MIN_FREQ}) …")

def _top200_coords(yr: int) -> dict[str, np.ndarray]:
    return {w: v for w, v in yr_coords[yr].items() if w in top_200_set}

for yr in years:
    coords = _top200_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw_aliases_set)
    p = LANDSCAPE_TOP200_DIR / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_TOP200_DIR / "animation.gif",
                   _top200_coords, kw_aliases_set)

# ── 16. keywords_top500/ landscapes ──────────────────────────────────────────
print(f"\nSaving keywords_top500 panels (MIN_FREQ={MIN_FREQ}) …")

def _top500_coords(yr: int) -> dict[str, np.ndarray]:
    return {w: v for w, v in yr_coords[yr].items() if w in top_500_set}

for yr in years:
    coords = _top500_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw_aliases_set)
    p = LANDSCAPE_TOP500_DIR / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_TOP500_DIR / "animation.gif",
                   _top500_coords, kw_aliases_set)

# ── 17. UMAP landscape (corpus average vectors — single aggregate view) ────────
print("\nBuilding UMAP embedding (average vectors across all years) …")
reducer = umap_lib.UMAP(
    n_components=2, n_neighbors=15, min_dist=0.1,
    metric="cosine", random_state=42,
)
umap_arr  = reducer.fit_transform(mat)   # mat = avg-vec matrix built for PCA
umap_coords: dict[str, np.ndarray] = {
    w: umap_arr[i] for i, w in enumerate(valid_vocab)
}
print(f"  UMAP fitted on {len(valid_vocab)} words")

_umap_landscapes = [
    ("nearest-neighbors", kw_nn_vocab,   kw_aliases_set, TARGET_COLOR,    None),
    ("go-words",          _kw_gw_vocab,  kw_aliases_set, TARGET_COLOR,    gw_canonical_set),
    ("top-100",           top_words_set, kw_aliases_set, TARGET_COLOR,    None),
    ("top-200",           top_200_set,   kw_aliases_set, TARGET_COLOR,    None),
    ("top-500",           top_500_set,   kw_aliases_set, TARGET_COLOR,    None),
]

for name, word_set, highlight_set, h_color, second_set in _umap_landscapes:
    coords = {w: umap_coords[w] for w in word_set if w in umap_coords}
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    draw_static_panel(ax, f"UMAP Semantic Landscape — {name.replace('-', ' ')}",
                      coords, highlight_set, h_color,
                      second_highlight_set=second_set)
    p = UMAP_DIR / f"{name}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

# ── 19. Topographic concentric diagrams — 2-row layout, fading darkness ───────
print("\nGenerating topographic ring diagrams …")

# Ring configuration
_FEATURE_RADII  = [0.28, 0.55, 0.82]
_RING_GROUPS    = [slice(0, 4), slice(4, 8), slice(8, 12)]
# Inner ring: X shape (π/4) so words are diagonal, not directly L/R of center
_RING_OFFSETS   = [np.pi / 4, 0.0, np.pi / 8]
# Contour step 0.09 puts circles exactly on feature radii (0.28, 0.55, 0.82)
_CONTOUR_RADII  = np.arange(0.10, 0.93, 0.09)
_C_MIN, _C_MAX  = _CONTOUR_RADII[0], _CONTOUR_RADII[-1]

# Per-ring text colours (inner = darkest, outer = lightest)
_RING_TEXT_COLORS = ["#1a1a1a", "#555555", "#888888"]


def draw_ring_diagram(term: str, year_nn: dict, years_to_plot: list,
                       outpath: pathlib.Path) -> None:
    n_panels = len(years_to_plot)
    n_cols   = math.ceil(n_panels / 2) if n_panels > 1 else 1
    n_rows   = 2 if n_panels > 1 else 1

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, 4.6 * n_rows),
                              subplot_kw={"projection": "polar"},
                              gridspec_kw={"hspace": 0.08, "wspace": 0.05})
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, yr in zip(axes_flat, years_to_plot):
        nns = year_nn.get(yr, [])
        theta_full = np.linspace(0, 2 * np.pi, 300)

        # Topographic contour circles — darkness fades with radius
        for r in _CONTOUR_RADII:
            r_norm     = (r - _C_MIN) / (_C_MAX - _C_MIN)   # 0 (inner) → 1 (outer)
            is_feature = any(abs(r - fr) < 0.005 for fr in _FEATURE_RADII)
            if is_feature:
                alpha = 0.55 * (1 - r_norm) + 0.22 * r_norm  # 0.55 → 0.22
                lw    = 0.90 * (1 - r_norm) + 0.45 * r_norm  # 0.90 → 0.45
                color = "#222222"
            else:
                alpha = 0.20 * (1 - r_norm) + 0.07 * r_norm  # 0.20 → 0.07
                lw    = 0.28
                color = "#999999"
            ax.plot(theta_full, [r] * 300,
                    color=color, linewidth=lw, alpha=alpha, zorder=0)

        # Place words at feature ring positions with faded text by ring
        for ring_idx, (ring_r, grp, offset) in enumerate(
                zip(_FEATURE_RADII, _RING_GROUPS, _RING_OFFSETS)):
            words_in_ring = nns[grp]
            n = len(words_in_ring)
            text_color = _RING_TEXT_COLORS[ring_idx]
            for i, (word, _) in enumerate(words_in_ring):
                theta = (2 * np.pi * i / max(n, 1)) + offset
                ax.text(theta, ring_r, word,
                        ha="center", va="center",
                        fontsize=8, color=text_color)

        ax.text(0, 0, term, ha="center", va="center",
                fontsize=10, color="#111111", transform=ax.transData)

        ax.set_ylim(0, 1.0)
        ax.set_yticklabels([]); ax.set_xticklabels([])
        ax.spines["polar"].set_visible(False)
        ax.set_title(str(yr), fontsize=11, pad=6)
        ax.grid(False)

    # Hide unused grid cells
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(f'"{term}" — nearest neighbours across years', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {outpath.name}")


for term in TARGET_TERMS:
    draw_ring_diagram(term, nn_results[term], years,
                      CONC_DIAG_DIR / f"concentric_{term}.png")

print("\n✓ Analysis complete. Outputs →", OUT_DIR)
print("  keywords/")
print("    nearest_neighbours.csv, semantic_shift_bar.png")
print(f"    nearest-neighbors/   ({len(years)} PNGs + animation.gif)")
print(f"    go-words/            ({len(years)} PNGs + animation.gif)  [keywords=red, go-words=grey]")
print(f"    top-100/             ({len(years)} PNGs + animation.gif)")
print(f"    top-200/             ({len(years)} PNGs + animation.gif)")
print(f"    top-500/             ({len(years)} PNGs + animation.gif)")
print(f"    concentric-diagrams/ ({len(TARGET_TERMS)} PNGs)")
print("    umap/                (5 PNGs — aggregate corpus view)")

# ══════════════════════════════════════════════════════════════════════════════
# KEYWORDS-2 ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

# ── keywords-2 nearest_neighbors/ landscapes ─────────────────────────────────
print(f"\nSaving keywords-2 keywords_nearest_neighbors panels …")

def _nn2_coords(yr: int) -> dict[str, np.ndarray]:
    return {w: v for w, v in yr_coords[yr].items() if w in kw2_nn_vocab}

for yr in years:
    coords = _nn2_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw2_aliases_set, apply_freq_filter=False)
    p = LANDSCAPE_NN_KW_DIR_2 / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_NN_KW_DIR_2 / "animation.gif",
                   _nn2_coords, kw2_aliases_set, apply_freq_filter=False)

# ── keywords-2 keywords/ landscapes — keywords-2 (red) + go-words (orange) ────
print("\nSaving keywords-2 keywords panels (keywords-2 in red + go-words in orange) …")

_kw2_gw_vocab = kw2_aliases_set | gw_canonical_set

def _kw2_coords(yr: int) -> dict[str, np.ndarray]:
    return {w: v for w, v in yr_coords[yr].items() if w in _kw2_gw_vocab}

for yr in years:
    coords = _kw2_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw2_aliases_set,
                         apply_freq_filter=False, highlight_color=TARGET_COLOR,
                         second_highlight_set=gw_canonical_set,
                         second_highlight_color=GO_WORDS_COLOR)
    p = LANDSCAPE_KW_DIR_2 / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_KW_DIR_2 / "animation.gif",
                   _kw2_coords, kw2_aliases_set,
                   apply_freq_filter=False, highlight_color=TARGET_COLOR,
                   second_highlight_set=gw_canonical_set,
                   second_highlight_color=GO_WORDS_COLOR)

# ── keywords-2 keywords_top100/ landscapes ────────────────────────────────────
print(f"\nSaving keywords-2 keywords_top100 panels (MIN_FREQ={MIN_FREQ}) …")

for yr in years:
    coords = _top100_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw2_aliases_set)
    p = LANDSCAPE_TOP100_DIR_2 / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_TOP100_DIR_2 / "animation.gif",
                   _top100_coords, kw2_aliases_set)

# ── keywords-2 keywords_top200/ landscapes ────────────────────────────────────
print(f"\nSaving keywords-2 keywords_top200 panels (MIN_FREQ={MIN_FREQ}) …")

for yr in years:
    coords = _top200_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw2_aliases_set)
    p = LANDSCAPE_TOP200_DIR_2 / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_TOP200_DIR_2 / "animation.gif",
                   _top200_coords, kw2_aliases_set)

# ── keywords-2 keywords_top500/ landscapes ────────────────────────────────────
print(f"\nSaving keywords-2 keywords_top500 panels (MIN_FREQ={MIN_FREQ}) …")

for yr in years:
    coords = _top500_coords(yr)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("white")
    draw_landscape_panel(ax, yr, coords, kw2_aliases_set)
    p = LANDSCAPE_TOP500_DIR_2 / f"landscape_{yr}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

save_landscape_gif(LANDSCAPE_TOP500_DIR_2 / "animation.gif",
                   _top500_coords, kw2_aliases_set)

# ── keywords-2 UMAP landscapes ───────────────────────────────────────────────
_umap_landscapes_2 = [
    ("nearest-neighbors", kw2_nn_vocab,    kw2_aliases_set, TARGET_COLOR, None),
    ("go-words",          _kw2_gw_vocab,   kw2_aliases_set, TARGET_COLOR, gw_canonical_set),
    ("top-100",           top_words_set,   kw2_aliases_set, TARGET_COLOR, None),
    ("top-200",           top_200_set,     kw2_aliases_set, TARGET_COLOR, None),
    ("top-500",           top_500_set,     kw2_aliases_set, TARGET_COLOR, None),
]

for name, word_set, highlight_set, h_color, second_set in _umap_landscapes_2:
    coords = {w: umap_coords[w] for w in word_set if w in umap_coords}
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    draw_static_panel(ax, f"UMAP Semantic Landscape (keywords-2) — {name.replace('-', ' ')}",
                      coords, highlight_set, h_color,
                      second_highlight_set=second_set)
    p = UMAP_DIR_2 / f"{name}.png"
    plt.savefig(str(p), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {p.name}")

# ── keywords-2 topographic ring diagrams ──────────────────────────────────────
print("\nGenerating keywords-2 topographic ring diagrams …")

for term in TARGET_TERMS_2:
    draw_ring_diagram(term, nn_results_2[term], years,
                      CONC_DIAG_DIR_2 / f"concentric_{term}.png")

print("\n✓ Keywords-2 analysis complete. Outputs →", OUT_DIR_2)
print("  keywords-2/")
print("    nearest_neighbours.csv, semantic_shift_bar.png")
print(f"    nearest-neighbors/   ({len(years)} PNGs + animation.gif)")
print(f"    go-words/            ({len(years)} PNGs + animation.gif)")
print(f"    top-100/             ({len(years)} PNGs + animation.gif)")
print(f"    top-200/             ({len(years)} PNGs + animation.gif)")
print(f"    top-500/             ({len(years)} PNGs + animation.gif)")
print(f"    concentric-diagrams/ ({len(TARGET_TERMS_2)} PNGs)")
print("    umap/                (5 PNGs — aggregate corpus view)")
