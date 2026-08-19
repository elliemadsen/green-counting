"""Extract embedded images from every syllabus PDF for the image gallery.

Reads every PDF in data/ALL2020-2026-full/ (the full submissions -- the same
files ../web/word_associations/drive_ids.js links to, so an image extracted
here always matches what a click-through to Drive shows), pulls out the
raster images placed on each page via pypdfium2, and keeps the ones that look
like real content rather than letterhead/logos/rule lines:

  - placed size on the page must be >= MIN_PT in both dimensions (a lot of
    syllabus templates repeat a small header/footer graphic on every page --
    at 72pt = 1 inch, that's comfortably above logo/icon size but below a
    typical project photo or diagram)
  - aspect ratio (long side / short side) must be <= MAX_ASPECT (drops thin
    banners and rule lines that happen to be wide/tall enough to pass the
    size filter)
  - not near-flat-color (stdev of luminance >= MIN_STDEV; drops solid-color
    spacer rectangles, which have no visual content of their own) -- UNLESS
    listed in include.txt (same format as exclude.txt, below, but the
    opposite effect: a manual allow-list for real content the flat/blank
    filters catch as false positives, e.g. a sparse site plan or scatter
    diagram that's mostly white/empty space by nature. Browse
    images_excluded/flat/ and images_excluded/blank/, add anything worth
    keeping to include.txt, rerun)
  - not mostly-blank (fraction of near-white pixels <= MAX_WHITE_FRAC; drops
    charts/diagrams that are almost entirely empty page) -- also subject to
    the include.txt override above
  - not listed in exclude.txt (a plain filename-per-line manual block list,
    for anything the automatic filters above can't catch -- headshots, most
    notably: there's no reliable automatic signal for "this is a person's
    face" here, so that pass stays manual. Run the extractor, browse
    web/images/, add unwanted filenames to exclude.txt, rerun). An earlier
    version of this script also auto-dropped "typeset text on a white
    background" via OCR, but it flagged too many false positives (real
    diagrams/photos with a caption block); that pass was removed, and its
    true positives were folded into exclude.txt by hand instead.
  - not a byte-identical repeat of an image already kept from the same PDF
    (catches a repeating header graphic that's individually large enough to
    pass the size filter)

That last rule only dedupes WITHIN one PDF -- the same image can and does
turn up across different syllabi (a shared template graphic, or two students
reusing the same photo/diagram), and every one of those occurrences is kept
as its own row here, each with its own syllabus_id/year/page. Cross-syllabus
duplicates are handled after every PDF has been scanned (see
group_duplicates()): two images are grouped as "the same" if either their
exact content hash matches, OR their perceptual hash (dhash(), a 64-bit
brightness-gradient fingerprint) is within PHASH_MAX_DISTANCE bits -- exact
hash alone misses a lot of real duplicates here, since the same source image
is often embedded at a different resolution/JPEG quality by each PDF
producer, which changes its bytes without changing how it looks. Every row's
"group" field is a shared id across everything in that group, so the web
page can recognize cross-syllabus duplicates on its own: the By Color view
groups rows by that id and shows each unique image once, while every other
view still shows one tile per occurrence, per syllabus.

Surviving images are downscaled to MAX_DIM on their long side, saved as JPEG
into ../web/images/photos/ (this has to live under web/ -- it's what the
gallery page actually serves, both locally and in the docs/ GitHub Pages
build, which only copies out of web/), and logged to outputs/images.csv (one
row per image: which syllabus it came from, where in the PDF, and its
average color as HSL -- the "color" axis in the web page's default
arrangement). The same records are also written as ../web/images/data.js,
following the data.js convention used by ../web/bibliography/build_data.py
and ../5_verb_analysis/verb_keyword_per_syllabus.py: the browser reads a
plain embedded JS object instead of fetching/parsing the CSV itself.

Everything rejected past the geometric size/aspect check (i.e. everything
that actually got decoded) is also saved -- to images_excluded/<reason>/,
staying local to this folder since nothing there is ever served -- so the
automatic filters can be spot-checked without rerunning anything.

The "closeup -> open on Drive" link on the web page reuses
../web/word_associations/drive_ids.js directly (loaded as a second <script>
tag) rather than duplicating that mapping here: DRIVE_IDS is keyed by the
"nb" filename stem, so the page looks up DRIVE_IDS[syllabus_id + "nb"].

Orientation and transparency: images are decoded with get_bitmap(render=True),
which composites the object through PDFium's normal rendering path -- both the
page's placement matrix (rotation/flip) AND any soft mask/alpha get applied,
rather than handing back the image's raw stored pixels. That matters for two
reasons: (1) an earlier version used render=False (raw pixels, no matrix) plus
a hand-rolled flip correction based on the matrix's sign, but that couldn't
handle actual rotation, and even its flip cases are just a special case of
what render=True already does correctly; (2) some images are stored with a
transparent background (an SMask) meant to let the page color show through --
render=False silently ignores that mask and exposes whatever garbage pixels
sit underneath it (usually solid black), while render=True composites
correctly, then to_rgb() below flattens the result onto white.

Usage:
  python3 extract_images.py            # all PDFs
  python3 extract_images.py --limit 20 # first 20 only, for a quick smoke test
                                        # (also skips pruning stale photos --
                                        #  see the note above main())

To remove a specific image by hand (headshots, most likely -- see exclude.txt
in this directory), add its filename to exclude.txt and rerun with no
--limit; the CSV, data.js, and web/images/photos/ all update to match. To
rescue one the flat/blank filters wrongly caught (a sparse diagram that's
mostly white/empty by nature, say), add its filename to include.txt instead
and rerun the same way.
"""

import argparse
import colorsys
import csv
import hashlib
import io
import json
import re
import time
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageStat

BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR.parent / "data" / "ALL2020-2026-full"
OUT_CSV = BASE_DIR / "outputs" / "images.csv"
EXCLUDE_FILE = BASE_DIR / "exclude.txt"
INCLUDE_FILE = BASE_DIR / "include.txt"  # overrides the flat/blank filters -- see load_filenames()
EXCLUDED_DIR = BASE_DIR / "images_excluded"  # rejects, sorted by reason -- never served
WEB_DIR = BASE_DIR.parent / "web" / "images"
PHOTOS_DIR = WEB_DIR / "photos"  # kept images -- must live under web/ to be servable
OUT_DATA_JS = WEB_DIR / "data.js"

# Rejection reason -> its images_excluded/ subfolder + the stats/prune key.
REJECT_DIRS = {
    "flat": EXCLUDED_DIR / "flat",       # near-solid-color (MIN_STDEV)
    "manual": EXCLUDED_DIR / "manual",   # listed in exclude.txt
    "blank": EXCLUDED_DIR / "blank",     # mostly-white (MAX_WHITE_FRAC)
}

MIN_PT = 72          # minimum placed width/height on the page, in points (72pt = 1in)
MAX_ASPECT = 6.0      # long side / short side; drops thin banners and rule lines
MIN_STDEV = 3.0       # luminance stdev; drops near-solid-color rectangles
MAX_WHITE_FRAC = 0.90  # near-white pixel fraction above which an image is dropped outright
WHITE_PIXEL_LEVEL = 235  # per-pixel grayscale value (0-255) counted as "near white"
MAX_DIM = 560         # long side of the saved JPEG, in pixels
JPEG_QUALITY = 80
DHASH_SIZE = 8          # -> a DHASH_SIZE**2-bit (64-bit) perceptual hash
PHASH_MAX_DISTANCE = 10  # max Hamming distance to call two images "the same"; true
                         # duplicates measured at 0, unrelated images at ~29+ (wide margin)

FILENAME_RE = re.compile(r"^(\d{4})cprize-(\d+)$")


def to_rgb(pil_img):
    if pil_img.mode == "RGB":
        return pil_img
    if pil_img.mode in ("RGBA", "LA", "PA"):
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.convert("RGBA").split()[-1])
        return bg
    return pil_img.convert("RGB")


def load_filenames(path):
    """Plain filename-per-line list (# comments, blank lines ignored); a
    line may optionally carry a path prefix (only the basename is kept)."""
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(Path(line).name)
    return names


def white_fraction(pil_rgb):
    hist = pil_rgb.convert("L").histogram()
    total = sum(hist)
    return sum(hist[WHITE_PIXEL_LEVEL:]) / total if total else 0.0


def avg_hsl(pil_rgb):
    r, g, b = ImageStat.Stat(pil_rgb).mean
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return h * 360, s, l


def dhash(pil_rgb):
    """Difference hash: resize to (n+1, n) grayscale, then one bit per pixel
    for "is this pixel darker than the one to its right." Robust to the
    resolution/compression differences that break exact-hash matching --
    two independently-exported copies of the same source image reliably
    land at Hamming distance 0, while unrelated images land around 29-36
    (out of 64 bits) -- see PHASH_MAX_DISTANCE."""
    gray = pil_rgb.convert("L").resize((DHASH_SIZE + 1, DHASH_SIZE), Image.LANCZOS)
    px = gray.load()
    bits = 0
    for y in range(DHASH_SIZE):
        for x in range(DHASH_SIZE):
            bits = (bits << 1) | (1 if px[x, y] > px[x + 1, y] else 0)
    return bits


def group_duplicates(rows, phashes):
    """Union-find over exact content hash + perceptual-hash proximity;
    returns a list of group ids, one per row, shared by everything judged
    to be "the same image" regardless of which syllabus it's from."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    by_hash = {}
    for i, row in enumerate(rows):
        by_hash.setdefault(row["hash"], []).append(i)
    for idxs in by_hash.values():
        for i in idxs[1:]:
            union(idxs[0], i)

    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if bin(phashes[i] ^ phashes[j]).count("1") <= PHASH_MAX_DISTANCE:
                union(i, j)

    return [find(i) for i in range(n)]


def save_image(pil_img, filename, dest_dir):
    """Downscale to MAX_DIM and save as JPEG into dest_dir; returns the
    (possibly resized) image, since the caller may still need its pixels."""
    scale = MAX_DIM / max(pil_img.size)
    if scale < 1:
        new_size = (max(1, round(pil_img.width * scale)), max(1, round(pil_img.height * scale)))
        pil_img = pil_img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=JPEG_QUALITY)
    (dest_dir / filename).write_bytes(buf.getvalue())
    return pil_img


def extract_pdf(pdf_path, seen_hashes, image_rows, phashes, excluded, included, rejects):
    stem = pdf_path.stem
    m = FILENAME_RE.match(stem)
    if not m:
        return 0
    year, num = int(m.group(1)), int(m.group(2))
    short_id = f"{year}-{num:02d}"

    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception as e:
        print(f"  [WARN] could not open {pdf_path.name}: {e}")
        return 0

    kept = 0
    seen_here = seen_hashes.setdefault(stem, set())
    for page_i in range(len(pdf)):
        page = pdf[page_i]
        objs = list(page.get_objects(filter=(pdfium.raw.FPDF_PAGEOBJ_IMAGE,)))
        # Index by position among ALL image objects pdfium found on this page
        # (not among only the ones that pass the filters below), so a
        # filename's page/index never shifts as filters get tuned -- that's
        # what keeps exclude.txt's manually-curated entries valid across reruns.
        for obj_idx, obj in enumerate(objs):
            try:
                x0, y0, x1, y1 = obj.get_bounds()
            except Exception:
                continue
            w_pt, h_pt = x1 - x0, y1 - y0
            if w_pt < MIN_PT or h_pt < MIN_PT:
                continue
            if min(w_pt, h_pt) <= 0 or max(w_pt, h_pt) / min(w_pt, h_pt) > MAX_ASPECT:
                continue

            filename = f"{stem}_p{page_i:02d}_i{obj_idx:02d}.jpg"

            try:
                pil_img = obj.get_bitmap(render=True, scale_to_original=True).to_pil()
            except Exception:
                continue
            pil_img = to_rgb(pil_img)

            if filename not in included and ImageStat.Stat(pil_img.convert("L")).stddev[0] < MIN_STDEV:
                save_image(pil_img, filename, REJECT_DIRS["flat"])
                rejects["flat"].add(filename)
                continue
            if filename in excluded:
                save_image(pil_img, filename, REJECT_DIRS["manual"])
                rejects["manual"].add(filename)
                continue

            white_frac = white_fraction(pil_img)
            if filename not in included and white_frac > MAX_WHITE_FRAC:
                save_image(pil_img, filename, REJECT_DIRS["blank"])
                rejects["blank"].add(filename)
                continue

            digest = hashlib.md5(pil_img.tobytes()).hexdigest()
            if digest in seen_here:
                continue
            seen_here.add(digest)

            pil_img = save_image(pil_img, filename, PHOTOS_DIR)
            hue, sat, light = avg_hsl(pil_img)
            hexcolor = "#%02x%02x%02x" % tuple(
                round(c * 255) for c in colorsys.hls_to_rgb(hue / 360, light, sat)
            )

            image_rows.append({
                "syllabus_id": stem,
                "s": short_id,
                "year": year,
                "filename": filename,
                "page": page_i,
                "index_on_page": obj_idx,
                "width": pil_img.width,
                "height": pil_img.height,
                "hex": hexcolor,
                "hue": round(hue, 1),
                "sat": round(sat, 3),
                "light": round(light, 3),
                "hash": digest,
            })
            phashes.append(dhash(pil_img))
            kept += 1
    return kept


def main():
    parser = argparse.ArgumentParser(description="Extract gallery-worthy images from every syllabus PDF.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N PDFs (default: all)")
    args = parser.parse_args()

    PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    for d in REJECT_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")
    if args.limit:
        pdf_files = pdf_files[:args.limit]
    print(f"Scanning {len(pdf_files)} PDFs from {PDF_DIR} …")

    excluded = load_filenames(EXCLUDE_FILE)
    if excluded:
        print(f"Loaded {len(excluded)} manually excluded filename(s) from {EXCLUDE_FILE}")
    included = load_filenames(INCLUDE_FILE)
    if included:
        print(f"Loaded {len(included)} manually included filename(s) from {INCLUDE_FILE}")

    seen_hashes = {}
    image_rows = []
    phashes = []
    rejects = {key: set() for key in REJECT_DIRS}
    t0 = time.time()
    for i, pdf_path in enumerate(pdf_files):
        extract_pdf(pdf_path, seen_hashes, image_rows, phashes, excluded, included, rejects)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pdf_files)} PDFs, {len(image_rows)} images kept so far, "
                  f"{time.time() - t0:.0f}s elapsed")

    print(f"Done: {len(image_rows)} images from {len(pdf_files)} PDFs in {time.time() - t0:.0f}s")
    print(f"  dropped: {len(rejects['flat'])} near-flat-color, {len(rejects['blank'])} mostly-blank, "
          f"{len(rejects['manual'])} manually excluded (saved to {EXCLUDED_DIR}/<reason>/)")

    t1 = time.time()
    groups = group_duplicates(image_rows, phashes)
    for row, gid in zip(image_rows, groups):
        row["group"] = gid
    n_groups = len(set(groups))
    print(f"Grouped {len(image_rows)} images into {n_groups} duplicate-aware groups "
          f"({len(image_rows) - n_groups} cross-syllabus duplicates) in {time.time() - t1:.0f}s")

    # Only prune stale files on a full run -- a --limit smoke test only
    # touches a handful of PDFs and shouldn't delete the rest of a prior
    # full run's output.
    if not args.limit:
        written = {row["filename"] for row in image_rows}
        removed = 0
        for p in PHOTOS_DIR.glob("*.jpg"):
            if p.name not in written:
                p.unlink()
                removed += 1
        for key, reject_dir in REJECT_DIRS.items():
            for p in reject_dir.glob("*.jpg"):
                if p.name not in rejects[key]:
                    p.unlink()
                    removed += 1
        if removed:
            print(f"Removed {removed} stale file(s) no longer produced by this run")

    fieldnames = ["syllabus_id", "s", "year", "filename", "page", "index_on_page",
                  "width", "height", "hex", "hue", "sat", "light", "hash", "group"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)
    print(f"Wrote {OUT_CSV} ({len(image_rows)} rows)")

    payload = {"built": time.strftime("%Y-%m-%d"), "images": image_rows}
    OUT_DATA_JS.write_text("window.IMG_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                            encoding="utf-8")
    print(f"Wrote {OUT_DATA_JS} ({OUT_DATA_JS.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
