/* Image gallery: every kept image from every syllabus PDF (see
   ../../6_images/extract_images.py for how they were extracted and
   filtered), arranged a few different ways. All labels use textContent
   (never innerHTML). */
(function () {
  "use strict";
  const D = window.IMG_DATA;
  const IMAGES = D.images;

  const $ = (sel) => document.querySelector(sel);
  const gallery = $("#gallery");
  const state = { order: [] }; // flat list of images in the CURRENT arrangement's visual order

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Sort key for "color" -- used for both the Year x Color view's y axis
  // and the standalone Color Spectrum view, so the two read consistently:
  // always hue-first, with no separate grayscale bucket. Even a
  // faintly-tinted "grayish" image keeps its own (if faint) hue, so it
  // lands next to true-color images that share that tint instead of being
  // segregated away from them just for having low saturation. Lightness is
  // the tie-break within a hue.
  function spectrumKey(img) {
    return img.hue / 360;
  }

  function documentOrder(a, b) {
    return a.year - b.year || a.syllabus_id.localeCompare(b.syllabus_id) ||
      a.page - b.page || a.index_on_page - b.index_on_page;
  }

  function makeTile(img, idx) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tile";
    btn.dataset.idx = String(idx);
    const im = document.createElement("img");
    im.src = "photos/" + img.filename;
    im.loading = "lazy";
    im.alt = img.s + ", page " + (img.page + 1);
    btn.appendChild(im);
    return btn;
  }

  function yearsPresent() {
    return [...new Set(IMAGES.map((i) => i.year))].sort((a, b) => a - b);
  }

  // ── Arrangements ─────────────────────────────────────────────────────────
  // Default: x = year, y = color. Each year's grid is a fixed height (see
  // .year-grid in style.css: grid-auto-flow: column + auto-filled row
  // tracks), so every year takes the same roughly-one-screen height and
  // wraps into however many columns its image count needs -- fewer columns
  // for a smaller year, more for a bigger one -- instead of every year
  // getting its own height.
  function renderYearColor() {
    gallery.replaceChildren();
    gallery.dataset.arrange = "year-color";
    const order = [];
    for (const y of yearsPresent()) {
      const imgs = IMAGES.filter((i) => i.year === y).sort((a, b) => spectrumKey(a) - spectrumKey(b) || a.light - b.light);
      if (!imgs.length) continue;
      const block = el("div", "year-block");
      const head = el("div", "year-block-head");
      head.appendChild(el("span", null, String(y)));
      head.appendChild(el("span", "n", imgs.length + " images"));
      block.appendChild(head);
      const grid = el("div", "year-grid");
      for (const img of imgs) {
        const idx = order.length;
        order.push(img);
        grid.appendChild(makeTile(img, idx));
      }
      block.appendChild(grid);
      gallery.appendChild(block);
    }
    state.order = order;
  }

  // One continuous rainbow, no year grouping and no separate grayscale
  // bucket -- see spectrumKey(). Also the only view that dedupes: the same
  // image (by "group" -- see extract_images.py's group_duplicates(), which
  // catches both byte-identical repeats and near-duplicates re-exported at
  // a different resolution/quality) can turn up in more than one syllabus,
  // and here it's shown once, as its earliest occurrence, with the rest of
  // the group attached as _dupGroup for the lightbox to list (see
  // renderLbInfo). Every other view intentionally skips this and shows one
  // tile per occurrence, per syllabus/year.
  function renderColorSpectrum() {
    gallery.replaceChildren();
    gallery.dataset.arrange = "color";
    const groups = new Map();
    for (const img of IMAGES) {
      if (!groups.has(img.group)) groups.set(img.group, []);
      groups.get(img.group).push(img);
    }
    const reps = [...groups.values()].map((group) => {
      const rep = [...group].sort(documentOrder)[0];
      return Object.assign({}, rep, { _dupGroup: group });
    });
    reps.sort((a, b) => spectrumKey(a) - spectrumKey(b) || a.light - b.light);
    const grid = el("div", "flow-grid");
    const order = [];
    for (const img of reps) {
      const idx = order.length;
      order.push(img);
      grid.appendChild(makeTile(img, idx));
    }
    gallery.appendChild(grid);
    state.order = order;
  }

  // Plain chronological grid, no color sorting -- a document-browsing baseline.
  function renderByYear() {
    gallery.replaceChildren();
    gallery.dataset.arrange = "year";
    const order = [];
    for (const y of yearsPresent()) {
      const imgs = IMAGES.filter((i) => i.year === y).sort(documentOrder);
      if (!imgs.length) continue;
      const block = el("div", "year-block");
      const head = el("div", "year-block-head");
      head.appendChild(el("span", null, String(y)));
      head.appendChild(el("span", "n", imgs.length + " images"));
      block.appendChild(head);
      const grid = el("div", "flow-grid");
      for (const img of imgs) {
        const idx = order.length;
        order.push(img);
        grid.appendChild(makeTile(img, idx));
      }
      block.appendChild(grid);
      gallery.appendChild(block);
    }
    state.order = order;
  }

  // Small multiples: one cluster per syllabus, in submission order.
  function renderBySyllabus() {
    gallery.replaceChildren();
    gallery.dataset.arrange = "syllabus";
    const order = [];
    const bySyl = new Map();
    for (const img of IMAGES) {
      if (!bySyl.has(img.syllabus_id)) bySyl.set(img.syllabus_id, []);
      bySyl.get(img.syllabus_id).push(img);
    }
    const sylIds = [...bySyl.keys()].sort((a, b) => {
      const ia = bySyl.get(a)[0], ib = bySyl.get(b)[0];
      return ia.year - ib.year || a.localeCompare(b);
    });
    for (const sid of sylIds) {
      const imgs = bySyl.get(sid).sort((a, b) => a.page - b.page || a.index_on_page - b.index_on_page);
      const block = el("div", "syl-block");
      block.id = "syl-" + sid;
      const head = el("div", "syl-block-head");
      head.appendChild(el("span", null, imgs[0].s));
      head.appendChild(el("span", "n", imgs.length + (imgs.length === 1 ? " image" : " images")));
      block.appendChild(head);
      const grid = el("div", "syl-grid");
      for (const img of imgs) {
        const idx = order.length;
        order.push(img);
        grid.appendChild(makeTile(img, idx));
      }
      block.appendChild(grid);
      gallery.appendChild(block);
    }
    state.order = order;
  }

  const RENDERERS = {
    "year-color": renderYearColor,
    "color": renderColorSpectrum,
    "year": renderByYear,
    "syllabus": renderBySyllabus,
  };

  function setArrangement(key) {
    for (const b of document.querySelectorAll("#arrange-toggle button"))
      b.classList.toggle("active", b.dataset.arrange === key);
    RENDERERS[key]();
  }
  for (const btn of document.querySelectorAll("#arrange-toggle button")) {
    btn.addEventListener("click", () => setArrangement(btn.dataset.arrange));
  }
  const zoomSlider = $("#zoom-slider");
  zoomSlider.addEventListener("input", () => {
    gallery.style.setProperty("--tile", zoomSlider.value + "px");
  });

  // Jumps to the By Syllabus view, scrolled to one syllabus's cluster --
  // used by the lightbox's "See all images in this syllabus" button.
  function goToSyllabus(syllabusId) {
    setArrangement("syllabus");
    const target = document.getElementById("syl-" + syllabusId);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  const moreInfoToggle = $("#more-info-toggle");
  const moreInfo = $("#more-info");
  moreInfoToggle.addEventListener("click", () => {
    const showing = moreInfo.hidden;
    moreInfo.hidden = !showing;
    moreInfoToggle.textContent = showing ? "Less information" : "More information";
  });

  // ── Lightbox: same Drive-linking mechanism as the verb/keyword heatmap
  // (../word_associations/app.js's renderSylView) -- DRIVE_IDS is keyed by
  // the "nb" filename stem, so a syllabus without a mapped entry falls back
  // to a Drive search instead of a dead link. ─────────────────────────────
  const lightbox = $("#lightbox");
  const lbImg = $("#lb-img");
  const lbInfo = $("#lb-info");
  let lbIndex = -1;

  function driveLinkInfo(syllabusId) {
    const driveId = (window.DRIVE_IDS || {})[syllabusId + "nb"];
    return {
      href: driveId
        ? "https://drive.google.com/file/d/" + driveId + "/view"
        : "https://drive.google.com/drive/search?q=" + encodeURIComponent(syllabusId),
      title: driveId
        ? "Opens this syllabus PDF directly — requires Drive access"
        : "Searches the shared Google Drive for this syllabus — requires Drive access",
    };
  }

  function showLightboxImage() {
    const img = state.order[lbIndex];
    if (!img) return;
    lbImg.src = "photos/" + img.filename;
    lbImg.alt = img.s + ", page " + (img.page + 1);
    renderLbInfo(img);
  }

  // One identically-formatted row per syllabus this image belongs to: just
  // the current one normally, or every syllabus in the group for a deduped
  // By Color tile (see renderColorSpectrum) -- id, "See all images in this
  // syllabus", "open PDF" all in the same layout either way.
  function renderLbInfo(img) {
    lbInfo.replaceChildren();
    const group = img._dupGroup && img._dupGroup.length > 1 ? img._dupGroup : [img];
    for (const g of group) {
      const row = el("div", "lb-info-row");
      row.appendChild(el("span", "id", g.s));
      const sylBtn = document.createElement("button");
      sylBtn.type = "button";
      sylBtn.className = "drive-link lb-link-btn";
      sylBtn.textContent = "See all images in this syllabus";
      sylBtn.addEventListener("click", () => {
        closeLightbox();
        goToSyllabus(g.syllabus_id);
      });
      row.appendChild(sylBtn);
      const info = driveLinkInfo(g.syllabus_id);
      const driveLink = document.createElement("a");
      driveLink.className = "drive-link";
      driveLink.target = "_blank";
      driveLink.rel = "noopener";
      driveLink.textContent = "open PDF ↗";
      driveLink.href = info.href;
      driveLink.title = info.title;
      row.appendChild(driveLink);
      lbInfo.appendChild(row);
    }
  }
  function openLightbox(idx) {
    lbIndex = idx;
    showLightboxImage();
    lightbox.hidden = false;
  }
  function closeLightbox() {
    lightbox.hidden = true;
    lbIndex = -1;
  }
  function step(delta) {
    if (lbIndex < 0 || !state.order.length) return;
    lbIndex = (lbIndex + delta + state.order.length) % state.order.length;
    showLightboxImage();
  }

  gallery.addEventListener("click", (e) => {
    const tile = e.target.closest(".tile");
    if (!tile) return;
    openLightbox(Number(tile.dataset.idx));
  });
  $("#lb-close").addEventListener("click", closeLightbox);
  $("#lb-prev").addEventListener("click", () => step(-1));
  $("#lb-next").addEventListener("click", () => step(1));
  lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
  document.addEventListener("keydown", (e) => {
    if (lightbox.hidden) return;
    if (e.key === "Escape") closeLightbox();
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
  });

  // Tell the parent shell (if embedded in one -- see web/index.html) to hide
  // its tab bar the moment the user scrolls down at all, and bring it back
  // on any scroll-up, instead of it sitting fixed above the content the
  // whole time. Same mechanism as word_associations/app.js.
  if (window.parent !== window) {
    let lastY = window.scrollY, lastDir = "up", ticking = false, lastFlip = 0;
    // A 1px delta is sensitive enough to react to a small scroll, but real
    // scroll input (trackpad momentum especially, right as it decelerates)
    // isn't perfectly monotonic frame to frame -- without this cooldown,
    // that sub-pixel jitter flips direction back and forth rapidly, which
    // reads as the tab bar hiding, flickering, and reappearing instead of
    // cleanly hiding once. The cooldown doesn't add perceptible delay to the
    // FIRST flip after a real direction change, only to repeated flips.
    const FLIP_COOLDOWN_MS = 200;
    const postDir = (dir) => {
      if (dir === lastDir) return;
      const now = performance.now();
      if (now - lastFlip < FLIP_COOLDOWN_MS) return;
      lastFlip = now;
      lastDir = dir;
      // "*": the parent shell identifies a trusted sender via ev.source
      // (see web/index.html), not by matching this targetOrigin, so this
      // doesn't need to (and shouldn't try to) predict the parent's origin.
      window.parent.postMessage({ type: "gc-scroll", dir }, "*");
    };
    window.addEventListener("scroll", () => {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(() => {
        const y = Math.max(0, window.scrollY);
        // The bar reappears ONLY once the user is actually back at the top
        // (y <= 6), not on any small upward scroll partway down -- so it
        // hides on the way down and stays out of the way even while
        // scrolling back and forth mid-page, only returning when you've
        // scrolled all the way back to the start of the content.
        if (y <= 6) {
          postDir("up");
        } else {
          const delta = y - lastY;
          if (delta > 1) postDir("down");
        }
        lastY = y;
        ticking = false;
      });
    }, { passive: true });
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  const nSyllabi = new Set(IMAGES.map((i) => i.syllabus_id)).size;
  $("#count").textContent = `${IMAGES.length} images from ${nSyllabi} syllabi`;
  gallery.style.setProperty("--tile", $("#zoom-slider").value + "px");
  renderColorSpectrum();
})();
