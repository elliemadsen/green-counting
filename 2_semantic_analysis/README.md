# Step 2: Semantic Analysis

## What this step does

This step uses the text of the syllabi to map how the _meaning_ of key words has shifted between 2020 and 2026. It produces visualisations that show where keywords sit relative to each other in embedding space, how that arrangement changes over time, and which words have drifted the most.

Run it once per keyword set:

```
python 2_semantic_analysis.py --keywords 1   # data/keywords.txt   → outputs/keywords-1/
python 2_semantic_analysis.py --keywords 2   # data/keywords-2.txt → outputs/keywords-2/  (default)
```

Each run only trains/plots the selected set — it does not generate both at once.

---

## The premise: word embeddings

The core idea is that words used in similar contexts tend to have similar meanings. By reading all the syllabi for a given year, the model learns a _semantic embedding_ for each word — a point in a multidimensional space where proximity means semantic similarity.

A separate model is trained for each year (2020–2026), so the embedding for a word like _climate_ in 2020 can be compared to 2026. To make the comparison fair, the year-models are mathematically rotated into a common orientation (Procrustes alignment) before any distances are measured.

Two words are "close" if they consistently appear around the same other words in the syllabi. Distance does not mean the words are synonyms — it means they occupy the same conceptual neighbourhood in course content.

---

## Outputs

Everything for a given run lands under `outputs/keywords-<1|2>/`:

### Semantic landscape maps

Each image is a 2-D map of the corpus for one year. The full embedding space has 100 dimensions; PCA compresses it to two axes that capture the most variation. **Axis directions have no fixed meaning** — only relative positions matter.

| Folder               | What it shows                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------- |
| `go-words/`           | The project keywords (red) plus the go-words list (grey)                                                  |
| `nearest-neighbors/`  | Keywords (red) plus the words most similar to each keyword that year, drawn from the top-500 corpus words |
| `top-100/`            | The 100 most frequent corpus words; keywords highlighted in red                                            |
| `top-200/`            | The 200 most frequent corpus words; keywords highlighted in red                                            |
| `top-500/`            | The 500 most frequent corpus words; keywords highlighted in red                                            |

Each folder also contains `animation.gif`, which loops through all years so you can watch the map evolve.

**How to read a landscape:** Words that cluster together are used in similar contexts. A keyword sitting near _infrastructure_ and _energy_ is being taught alongside those topics. If a keyword migrates toward a different cluster between years, its conceptual neighbourhood in the curriculum has changed.

### Aggregate UMAP view (`umap/`)

A single non-annual view of each of the five word sets above (averaged embedding across all years, reduced with UMAP instead of PCA). Useful as a second, differently-shaped projection of the same corpus for cross-checking the PCA landscapes.

### Concentric (ring) diagrams (`concentric-diagrams/`)

One diagram per keyword, covering all years in a two-row grid. Each panel shows the same keyword at its centre, with its 12 nearest neighbours arranged in three rings. **Closer rings = more similar words.** Ring lines fade toward the outside like topographic contour lines — inner rings are darker, indicating stronger semantic proximity.

Neighbours for these diagrams are drawn only from the **go-words list** (`data/go-words.txt`), not the top-500 corpus words — this keeps the rings focused on thematically relevant terms rather than incidental high-frequency words (place names, institution names, etc.).

These diagrams are useful for seeing which specific words are clustering around a keyword in a given year and whether that neighbourhood is stable or shifting.

### Semantic shift magnitude (`semantic_shift_bar.png`)

A bar chart ranking all keywords by how much their embedding moved between the first and last year of the study. The distance is a cosine distance — a pure magnitude with no directional component. A higher bar means the word is being taught in a noticeably different conceptual context in 2026 versus 2020; a lower bar means its usage has stayed stable.

### Semantic trajectories (`term_trajectories.png`)

The same 2-D PCA space as the landscape maps, but showing the _path_ each keyword took year by year. Arrows point toward the most recent year. Words whose paths are long or erratic have shifted more; words with short, straight paths have been stable.

### Nearest-neighbour table (`nearest_neighbours.csv`)

Machine-readable file listing, for each keyword and each year, the top-12 most similar words with their cosine similarity scores (candidates limited to the top-500 corpus words). Useful for detailed inspection or further analysis.

---

## Toggling outputs on/off

Near the top of `2_semantic_analysis.py`, the `OUTPUTS` dict controls which of the above get generated on a given run — flip any entry to `False` to skip it (e.g. set `OUTPUTS["top100_landscape"] = False` to stop generating the top-100 landscape and save time).

---

## Technical notes (brief)

- **Model:** Word2Vec skip-gram, 100-dimensional vectors, trained separately per year on sliding 60-word windows across all syllabi for that year
- **Alignment:** Orthogonal Procrustes rotation to a global reference model (trained on all years combined); word vectors are unit-normalised per word before the rotation is fit
- **Frequency floor:** Words appearing fewer than 50 times in a given year are excluded from the top-100/200/500 landscapes (but not from the `go-words/` or `nearest-neighbors/` outputs)
- **Nearest-neighbour scope:** For the `nearest-neighbors/` landscape and `nearest_neighbours.csv`, neighbours are drawn only from the top-500 most frequent corpus words. For the concentric diagrams, neighbours are drawn only from the go-words list (see above)
