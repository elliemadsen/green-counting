# Bibliography citation network

An interactive co-citation network of the texts in
`3_bibliography_analysis/outputs/bibliography.csv`. Each node is a text cited
in at least two syllabi; node size reflects how many syllabi cite it. Two
texts are linked when a syllabus cites both, and link weight (darker, heavier
lines) counts how many syllabi they co-occur in. Layout is force-directed
(d3-force), so frequently co-cited texts settle into neighborhoods; colors
come from one of two clustering algorithms over the citation links alone (see
below) — no text embeddings or external metadata influence placement.

## Running it

```
python3 build_data.py            # rebuild data.js from bibliography.csv (fast)
python3 build_data.py --covers   # also fetch cover images from Open Library (~10 min first run)
python3 -m http.server 8741      # then open http://localhost:8741/
```

Opening `index.html` directly from the filesystem also works — all assets
(including d3) are local.

- `build_data.py` — filters to texts in ≥2 syllabi, builds weighted co-citation
  edges (with the shared-syllabi list per edge), runs both clustering
  algorithms (below), and writes `data.js`. Open Library lookups are cached in
  `covers_cache.json` and images land in `covers/`, so re-runs are cheap.
  Re-run *without* `--covers` after a cover fetch to refresh `data.js` with the
  downloaded covers.
- `index.html` / `style.css` / `app.js` — the page. Edges render on a canvas
  (there are ~10k, with nearest-segment hit-testing for hover); nodes are SVG
  so covers and text labels stay crisp under zoom.

## View options (sidebar)

- **Min. co-citations per link** — hide weight-1 links to see the skeleton.
- **Min. syllabi per text** — prune to the most-assigned texts.
- **Texts shown as** — covers (with title–author fallback), text only, or dots.
- **Color by** — detected cluster, publication year (sequential ramp), or text
  type. Colors are validated for colorblind separation in both themes
  (all-pairs Machado ΔE; the dark set sits in the 8–12 floor band, which is
  acceptable because every node also carries a visible text label).
- **Cluster algorithm** — only matters when "Color by" is set to cluster:
  - *organic* (weighted label propagation, [build_data.py](build_data.py) —
    `label_propagation()`): each text repeatedly adopts the label held by the
    weighted majority of its neighbors until it stabilizes. This follows the
    real citation structure, so it produces natural but very uneven
    neighborhoods — currently 217/205/114/44/38/20/20 texts across 7 clusters,
    plus 82 texts in smaller communities that didn't make the top-7 cut and
    fold into gray "Other".
  - *balanced* (recursive spectral bisection, `spectral_bisection_clusters()`):
    repeatedly splits the largest partition in half along its Fiedler vector
    (the weakest cut in the normalized graph Laplacian) until every partition
    is ≤150 nodes, unless splitting further would drop a half below 50. On the
    current data this bottoms out at exactly 8 clusters of 92–93 texts each,
    with no leftover bucket. The cost: a forced-even split will sometimes cut
    through a genuinely tight-knit reading group just to hit the size target,
    so cluster boundaries read less like natural neighborhoods than the
    organic mode's.
- **Theme** — dark / light (also `?theme=light` in the URL).

## Covers

Cover images come from the Open Library API: by ISBN where the CSV has one,
otherwise a fuzzy title+author search (subtitle stripped, ≥60% title-token
overlap required before a cover is accepted, to avoid grabbing the wrong
book). Roughly two-thirds of *books* get covers; articles, reports, chapters,
and websites mostly have none and fall back to title–author text, which is
why the text fallback is a first-class display mode.

## Future: including single-syllabus texts

About 4,200 of the ~4,900 texts are cited in exactly one syllabus and are
currently excluded. Word-embedding placement (as Open Syllabus's galaxy does)
is off the table for now — the Open Library metadata hit rate is too low to
embed descriptions/categories reliably. Options that don't need embeddings:

1. **Syllabus-anchored dust.** Every singleton shares a syllabus with mapped
   texts. Place each one near the centroid of its syllabus's mapped texts
   (small jittered dots, no physics). Cheap, and reads like a star field
   around the neighborhoods — the aesthetic reference's look, driven purely by
   the existing link structure.
2. **Bipartite mode.** Add syllabi as a second node type; singletons hang off
   their syllabus node. Honest structure, but doubles visual complexity.
3. **Aggregate nodes.** One "+N more texts" node per syllabus, expandable on
   click into a temporary ring of its singletons.
4. **Local text similarity.** TF-IDF over titles (plus publisher/journal/year)
   computed offline — no API, full coverage — to give singletons weak "soft
   links" into the network. Noisier than embeddings but blocker-free.
5. **Shared-attribute links.** Link singletons by shared author, publisher, or
   journal, which the CSV already has; Open Library subjects could thicken
   this where the cache has them (partial coverage is fine for placement,
   fatal only for embeddings).
6. **On-demand reveal.** Hovering a syllabus name in a tooltip (or a syllabus
   list view) temporarily surfaces its singletons around the cursor, keeping
   the base map clean.
