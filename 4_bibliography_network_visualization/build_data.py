"""Build data.js for the citation network visualization.

Reads 3_bibliography_analysis/outputs/bibliography.csv, keeps texts cited in
>= 2 syllabi, builds co-citation edges (weight = number of shared syllabi),
detects communities with weighted label propagation, and writes data.js.

Cover images come from the Open Library API. Lookups are cached in
covers_cache.json and images are downloaded to covers/ so the page works
offline. Run:

    python3 build_data.py            # network only (fast)
    python3 build_data.py --covers   # also fetch cover images (slow, ~10 min first run)
"""

import csv
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CSV_PATH = HERE.parent / "3_bibliography_analysis" / "outputs" / "bibliography.csv"
CACHE_PATH = HERE / "covers_cache.json"
COVERS_DIR = HERE / "covers"
OUT_PATH = HERE / "data.js"

MIN_SYLLABI = 2  # texts cited in fewer syllabi are excluded (unlinked)
MAX_CLUSTERS = 7  # clusters beyond this fold into "Other" (organic algorithm only)
BALANCED_MIN_SIZE = 50   # soft floor: a partition won't be split below this
BALANCED_MAX_SIZE = 150  # ceiling: any partition over this gets bisected again
UA = {"User-Agent": "buell-center-bibliography-viz/1.0 (research project)"}


def load_rows():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["syllabi"] = sorted({s.strip() for s in r["source_syllabi"].split(";") if s.strip()})
    return [r for r in rows if len(r["syllabi"]) >= MIN_SYLLABI]


def build_edges(rows):
    syl2books = defaultdict(list)
    for i, r in enumerate(rows):
        for s in r["syllabi"]:
            syl2books[s].append(i)
    edges = defaultdict(list)  # (a, b) -> [syllabus, ...]
    for s, books in syl2books.items():
        for a, b in combinations(sorted(books), 2):
            edges[(a, b)].append(s)
    return edges


def label_propagation(n, edges, iterations=30, seed=7):
    """Weighted label propagation; deterministic given the seed."""
    neighbors = defaultdict(list)
    for (a, b), syls in edges.items():
        w = len(syls)
        neighbors[a].append((b, w))
        neighbors[b].append((a, w))
    labels = list(range(n))
    rng = random.Random(seed)
    order = list(range(n))
    for _ in range(iterations):
        rng.shuffle(order)
        changed = 0
        for i in order:
            if not neighbors[i]:
                continue
            votes = Counter()
            for j, w in neighbors[i]:
                votes[labels[j]] += w
            top = max(votes.values())
            # smallest label among the winners keeps ties deterministic
            best = min(l for l, v in votes.items() if v == top)
            if best != labels[i]:
                labels[i] = best
                changed += 1
        if changed == 0:
            break
    # rank communities by size; largest few keep their own id, rest -> -1
    sizes = Counter(labels)
    ranked = [l for l, _ in sizes.most_common()]
    remap = {l: (rank if rank < MAX_CLUSTERS else -1) for rank, l in enumerate(ranked)}
    return [remap[l] for l in labels]


def spectral_bisection_clusters(n, edges, min_size=BALANCED_MIN_SIZE, max_size=BALANCED_MAX_SIZE):
    """Recursive balanced graph partitioning: repeatedly bisect the largest
    partition along its Fiedler vector (2nd-smallest eigenvector of the
    normalized graph Laplacian), splitting at the median so both halves are
    forced to equal size. Keeps splitting until every partition fits under
    max_size, unless a split would leave a half smaller than min_size, in
    which case that partition is accepted as-is. Every node lands in some
    partition — there's no leftover "Other" bucket, unlike label propagation.
    """
    adj = defaultdict(dict)
    for (a, b), syls in edges.items():
        w = len(syls)
        adj[a][b] = w
        adj[b][a] = w

    def bisect(nodes_subset):
        idx = {node: i for i, node in enumerate(nodes_subset)}
        m = len(nodes_subset)
        w = np.zeros((m, m))
        for node in nodes_subset:
            i = idx[node]
            for nb, wt in adj[node].items():
                j = idx.get(nb)
                if j is not None:
                    w[i, j] = wt
        deg = w.sum(axis=1)
        with np.errstate(divide="ignore"):
            dinv = np.where(deg > 0, 1 / np.sqrt(deg), 0)
        laplacian = np.diag(deg) - w
        normalized = (dinv[:, None] * laplacian) * dinv[None, :]
        _, vecs = np.linalg.eigh(normalized)
        fiedler = vecs[:, 1]  # smallest eigenvector (index 0) is ~constant; use the next one
        order = np.argsort(fiedler)
        mid = m // 2
        sorted_nodes = [nodes_subset[i] for i in order]
        return sorted_nodes[:mid], sorted_nodes[mid:]

    partitions = [list(range(n))]
    final = []
    while partitions:
        part = partitions.pop()
        if len(part) <= max_size:
            final.append(part)
            continue
        left, right = bisect(part)
        if min(len(left), len(right)) < min_size:
            final.append(part)  # splitting further would go below the floor
            continue
        partitions.append(left)
        partitions.append(right)

    final.sort(key=len, reverse=True)
    labels = [None] * n
    for label, part in enumerate(final):
        for node in part:
            labels[node] = label
    return labels


def norm_title(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def find_cover(row, cache):
    """Return a cover URL fragment via cache or the Open Library API."""
    key = norm_title(row["title"])[:120]
    if key in cache:
        return cache[key]
    result = None
    isbn = row["isbn"].strip().split(";")[0].strip()
    if isbn:
        url = f"https://covers.openlibrary.org/b/isbn/{urllib.parse.quote(isbn)}-M.jpg?default=false"
        try:
            req = urllib.request.Request(url, headers=UA, method="HEAD")
            with urllib.request.urlopen(req, timeout=15):
                result = {"kind": "isbn", "value": isbn}
        except Exception:
            pass
    if result is None:
        short_title = row["title"].split(":")[0].strip()
        author = row["authors"].split(";")[0].split(",")[0].strip()
        surname = author.split()[-1] if author else ""
        q = urllib.parse.urlencode(
            {"q": f"{short_title} {surname}".strip(), "limit": 5,
             "fields": "cover_i,title"})
        try:
            req = urllib.request.Request("https://openlibrary.org/search.json?" + q, headers=UA)
            with urllib.request.urlopen(req, timeout=15) as resp:
                docs = json.load(resp).get("docs", [])
            want = set(norm_title(short_title).split())
            for doc in docs:
                if not doc.get("cover_i"):
                    continue
                got = set(norm_title(doc.get("title", "")).split())
                # loose title check to avoid grabbing an unrelated book's cover
                if want and len(want & got) / len(want) >= 0.6:
                    result = {"kind": "id", "value": doc["cover_i"]}
                    break
        except Exception:
            pass
    cache[key] = result
    time.sleep(0.35)  # be polite to the API
    return result


def download_cover(node_id, cover, dest):
    if dest.exists():
        return True
    if cover["kind"] == "isbn":
        url = f"https://covers.openlibrary.org/b/isbn/{cover['value']}-M.jpg?default=false"
    else:
        url = f"https://covers.openlibrary.org/b/id/{cover['value']}-M.jpg?default=false"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
        if len(data) > 1000:  # tiny responses are placeholder pixels
            dest.write_bytes(data)
            return True
    except Exception:
        pass
    return False


def main():
    fetch_covers = "--covers" in sys.argv
    rows = load_rows()
    edges = build_edges(rows)
    clusters = label_propagation(len(rows), edges)
    clusters_balanced = spectral_bisection_clusters(len(rows), edges)
    print(f"{len(rows)} nodes, {len(edges)} edges")
    print(f"  organic clusters:  {dict(Counter(clusters).most_common())}")
    print(f"  balanced clusters: {dict(sorted(Counter(clusters_balanced).items()))}")

    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    covers = {}
    if fetch_covers:
        COVERS_DIR.mkdir(exist_ok=True)
        for i, r in enumerate(rows):
            cover = find_cover(r, cache)
            if i % 25 == 0:
                CACHE_PATH.write_text(json.dumps(cache))
                print(f"  covers {i}/{len(rows)}: {sum(1 for v in covers.values() if v)} found")
            if cover and download_cover(i, cover, COVERS_DIR / f"{i}.jpg"):
                covers[i] = True
        CACHE_PATH.write_text(json.dumps(cache))
        print(f"covers found: {sum(1 for v in covers.values() if v)}/{len(rows)}")
    else:
        # keep covers already downloaded on a previous run
        if COVERS_DIR.exists():
            for p in COVERS_DIR.glob("*.jpg"):
                covers[int(p.stem)] = True

    nodes = []
    for i, r in enumerate(rows):
        nodes.append({
            "id": i,
            "title": r["title"],
            "authors": r["authors"],
            "year": r["year_published"],
            "type": r["type"],
            "publisher": r["publisher"],
            "journal": r["journal"] or r["container_title"],
            "citations": len(r["syllabi"]),
            "syllabi": r["syllabi"],
            "cluster": clusters[i],
            "clusterBalanced": clusters_balanced[i],
            "cover": bool(covers.get(i)),
        })
    links = [{"source": a, "target": b, "weight": len(s), "syllabi": sorted(s)}
             for (a, b), s in sorted(edges.items())]

    payload = {"nodes": nodes, "links": links,
               "meta": {"minSyllabi": MIN_SYLLABI, "built": time.strftime("%Y-%m-%d")}}
    OUT_PATH.write_text("const GRAPH = " + json.dumps(payload) + ";\n")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
