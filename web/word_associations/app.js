/* Verbs × Keywords explorer, written for a general audience.
   All derived numbers are computed here from the raw sentence tallies in
   data.js; every label is inserted with textContent (never innerHTML). */
(function () {
  "use strict";
  const D = window.VK_DATA;
  const MIN_COOCCUR = 3;     // cells with fewer shared sentences are left unscored
  const MIN_SYL_COOCCUR = 2; // per-course readings need at least this many sentences
  const HIDDEN_KW_CATS = new Set(["local", "universal", "theory"]);
  // "count" lists the texts most of these courses cite; "distinct" ranks by how
  // much MORE often these courses cite a text than the average course does.
  const CITATION_RANKING = "count";

  // Friendlier display names for the verb categories
  const VERB_NAMES = {
    positioning: "positioning",
    collaborating: "collaborating",
    doing: "doing",
    knowledge: "knowing",
    environment_positive: "environmental healing",
    built_environment: "building",
    environment_negative: "environmental harming",
    questioning: "questioning",
  };
  const vName = (v) => VERB_NAMES[v] || v;

  // Alphabetical display order; "local" column dropped
  const VERB_CATS = [...D.verb_cats].sort((a, b) => vName(a).localeCompare(vName(b)));
  const KW_CATS = D.kw_cats.filter((k) => !HIDDEN_KW_CATS.has(k)).sort();

  const state = { view: "corpus", metric: "pmi", cell: null, syl: null };

  const $ = (sel) => document.querySelector(sel);
  const viewCorpus = $("#view-corpus");
  const viewCell = $("#view-cell");
  const viewSyl = $("#view-syllabus");
  const tooltip = $("#tooltip");
  const breadcrumb = $("#breadcrumb");

  const key = (v, k) => v + "|" + k;
  const log2 = (x) => Math.log(x) / Math.LN2;

  // "2.3×", "12×", "0.4×" — how many times more/less than a baseline
  function fmtRatio(r) {
    if (!isFinite(r) || r <= 0) return "—";
    if (r >= 10) return Math.round(r) + "×";
    return (Math.round(r * 10) / 10) + "×";
  }

  // ── Derived stats ────────────────────────────────────────────────────────
  function corpusCell(v, k) {
    const a = D.corpus.c[key(v, k)] || 0;
    const va = D.corpus.v[v] || 0;
    const ka = D.corpus.k[k] || 0;
    const n = D.corpus.total_sentences;
    // "Chance" is measured only over sentences that contain at least one
    // categorized word, so administrative boilerplate (dates, policies)
    // doesn't inflate every pairing above 1×.
    const nA = D.corpus.active_sentences || n;
    let ratio = null; // observed co-occurrence vs pure chance
    if (a > 0 && va > 0 && ka > 0) ratio = (a * nA) / (va * ka);
    return { a, va, ka, n, ratio };
  }

  function sylRate(s, v, k) {
    const a = s.c[key(v, k)] || 0;
    return { a, rate: s.n ? (a / s.n) * 100 : 0 };
  }

  // How this course's use of a pairing compares with the average syllabus (a ratio)
  function sylVsAverage(s, v, k) {
    const a = s.c[key(v, k)] || 0;
    if (a < MIN_SYL_COOCCUR || !s.n) return null;
    const corpus = corpusCell(v, k);
    if (!corpus.a) return null;
    return (a / s.n) / (corpus.a / corpus.n);
  }

  function topSyllabi(v, k, limit) {
    return D.syllabi
      .map((s, i) => ({ s, i, ...sylRate(s, v, k) }))
      .filter((r) => r.a >= MIN_SYL_COOCCUR)
      .sort((x, y) => y.rate - x.rate)
      .slice(0, limit);
  }

  // Corpus-wide citation counts, computed once: how many courses cite each text
  let CITE_BASE = null;
  function citeBase() {
    if (CITE_BASE) return CITE_BASE;
    const counts = new Map();
    let nCourses = 0;
    for (const s of D.syllabi) {
      if (!s.refs || !s.refs.length) continue;
      nCourses++;
      const seen = new Set();
      for (const [title] of s.refs) {
        if (seen.has(title)) continue;
        seen.add(title);
        counts.set(title, (counts.get(title) || 0) + 1);
      }
    }
    CITE_BASE = { counts, nCourses };
    return CITE_BASE;
  }

  // Texts the pairing's courses cite unusually often: ranked by how much more
  // common a text is among these courses than among all courses, so corpus-wide
  // bestsellers don't automatically dominate every pairing.
  function topCitations(v, k, limit) {
    const base = citeBase();
    const counts = new Map();
    let nPair = 0;
    for (const s of D.syllabi) {
      if ((s.c[key(v, k)] || 0) < MIN_SYL_COOCCUR) continue;
      if (!s.refs || !s.refs.length) continue;
      nPair++;
      const seen = new Set();
      for (const [title, author] of s.refs) {
        if (seen.has(title)) continue;
        seen.add(title);
        const e = counts.get(title) || { title, author, n: 0 };
        e.n++;
        counts.set(title, e);
      }
    }
    if (!nPair || !base.nCourses) return [];
    const out = [];
    for (const e of counts.values()) {
      if (e.n < 2) continue;
      const overall = base.counts.get(e.title) || e.n;
      e.nPair = nPair;
      e.ratio = (e.n / nPair) / (overall / base.nCourses);
      out.push(e);
    }
    if (CITATION_RANKING === "distinct")
      return out
        .filter((e) => e.ratio > 1)
        .sort((a, b) => b.ratio - a.ratio || b.n - a.n)
        .slice(0, limit);
    return out
      .sort((a, b) => b.n - a.n || b.ratio - a.ratio)
      .slice(0, limit);
  }

  // ── Color (theme-aware via color-mix on the palette custom properties) ───
  function divColor(t) { // t in [-1, 1]
    const pole = t >= 0 ? "--div-pos" : "--div-neg";
    const pct = Math.min(100, Math.round(Math.abs(t) * 100));
    return `color-mix(in oklab, var(${pole}) ${pct}%, var(--div-mid))`;
  }
  function seqColor(t) { // t in [0, 1]
    const pct = Math.round(6 + Math.min(1, t) * 92);
    return `color-mix(in oklab, var(--accent) ${pct}%, var(--surface-1))`;
  }
  const inkFor = (strength) => (strength > 0.62 ? "#ffffff" : "var(--text-primary)");

  // ── Tooltip ──────────────────────────────────────────────────────────────
  function showTooltip(x, y, build) {
    tooltip.replaceChildren();
    build(tooltip);
    tooltip.hidden = false;
    const pad = 14;
    const r = tooltip.getBoundingClientRect();
    let left = x + pad, top = y + pad;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - pad;
    if (top + r.height > window.innerHeight - 8) top = y - r.height - pad;
    tooltip.style.left = Math.max(4, left) + "px";
    tooltip.style.top = Math.max(4, top) + "px";
  }
  function hideTooltip() { tooltip.hidden = true; }
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTooltip(); });
  document.addEventListener("scroll", hideTooltip, true);

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // Highlighted spans: yellow-orange for verb language, yellow-green for keywords
  const hlVerb = (text) => el("span", "hl-verb", text);
  const hlKw = (text) => el("span", "hl-kw", text);

  function attachHover(node, build) {
    node.addEventListener("pointermove", (e) => showTooltip(e.clientX, e.clientY, build));
    node.addEventListener("pointerleave", hideTooltip);
    node.addEventListener("focus", () => {
      const r = node.getBoundingClientRect();
      showTooltip(r.right, r.top, build);
    });
    node.addEventListener("blur", hideTooltip);
  }

  // ── Category words helpers ───────────────────────────────────────────────
  const topWords = (list, n) => (list || []).slice(0, n).map((w) => w[0]);
  const allWords = (list) => (list || []).map((w) => w[0]);

  function categoryHover(title, words) {
    return (tt) => {
      tt.appendChild(el("div", "tt-value", title));
      tt.appendChild(el("div", "tt-label tt-words",
        words.length ? words.join(", ") : "No words recorded."));
    };
  }

  function headerCell(tag, displayName, wordList, hoverTitle, hlClass) {
    const th = el(tag);
    th.setAttribute("tabindex", "0");
    const name = el("div", "cat-name");
    name.appendChild(el("span", hlClass, displayName));
    th.appendChild(name);
    const top4 = topWords(wordList, 4);
    if (top4.length) th.appendChild(el("div", "cat-words", top4.join(", ") + "…"));
    attachHover(th, categoryHover(hoverTitle, allWords(wordList)));
    return th;
  }

  // ── Breadcrumb + view switching ──────────────────────────────────────────
  // The page has one static width for every view, taken from the corpus
  // (all-categories) heatmap the first time it renders -- that width is
  // stable across renders since it's driven entirely by the heatmap's
  // fixed-width columns. Without this, the verb x keyword pairing view
  // (view-cell) is naturally wider than the two heatmap views, and letting
  // each view size itself made switching views resize the whole page; now
  // .view sections just overflow-x: auto internally if their content
  // exceeds this shared width instead.
  let pageWidth = null;

  function applyPageWidth() {
    if (pageWidth == null) return;
    for (const el of document.querySelectorAll(".view, .page-col"))
      el.style.width = pageWidth + "px";
  }

  function syncPageWidth() {
    if (pageWidth != null) { applyPageWidth(); return; }
    // Two rAFs: one for layout to settle after replaceChildren(), one more
    // in case fonts/webfont metrics shift the measurement.
    requestAnimationFrame(() => requestAnimationFrame(() => {
      const w = viewCorpus.getBoundingClientRect().width;
      if (!w) return;
      pageWidth = w;
      applyPageWidth();
    }));
  }

  function setView(view) {
    state.view = view;
    viewCorpus.hidden = view !== "corpus";
    viewCell.hidden = view !== "cell";
    viewSyl.hidden = view !== "syllabus";
    const toggle = $("#metric-toggle");
    if (toggle) toggle.style.visibility = view === "corpus" ? "visible" : "hidden";
    hideTooltip();
    renderBreadcrumb();
    syncPageWidth();
  }

  function renderBreadcrumb() {
    breadcrumb.replaceChildren();
    const root = el("a", null, "All categories");
    root.setAttribute("tabindex", "0");
    root.addEventListener("click", () => { renderCorpus(); setView("corpus"); });
    root.addEventListener("keydown", (e) => { if (e.key === "Enter") root.click(); });
    breadcrumb.appendChild(root);
    if (state.view === "cell" && state.cell) {
      breadcrumb.appendChild(el("span", "sep", "›"));
      breadcrumb.appendChild(el("span", null, vName(state.cell[0]) + " × " + state.cell[1]));
    } else if (state.view === "syllabus" && state.syl !== null) {
      if (state.cell) {
        breadcrumb.appendChild(el("span", "sep", "›"));
        const cellLink = el("a", null, vName(state.cell[0]) + " × " + state.cell[1]);
        cellLink.setAttribute("tabindex", "0");
        cellLink.addEventListener("click", () => { renderCellView(state.cell[0], state.cell[1]); setView("cell"); });
        cellLink.addEventListener("keydown", (e) => { if (e.key === "Enter") cellLink.click(); });
        breadcrumb.appendChild(cellLink);
      }
      breadcrumb.appendChild(el("span", "sep", "›"));
      breadcrumb.appendChild(el("span", null, "Course " + D.syllabi[state.syl].s));
    }
  }

  // ── Legend helper: gradient bar with labels aligned underneath it ────────
  function legend(gradientCss, leftLabel, midLabel, rightLabel) {
    const wrap = el("div", "legend");
    const bar = el("div", "bar");
    bar.style.background = gradientCss;
    wrap.appendChild(bar);
    const labels = el("div", "legend-labels");
    labels.appendChild(el("span", "l-left", leftLabel));
    if (midLabel !== null) labels.appendChild(el("span", "l-mid", midLabel));
    labels.appendChild(el("span", "l-right", rightLabel));
    wrap.appendChild(labels);
    return wrap;
  }

  // Wrap a heatmap table with visible X ("Keywords") and Y ("Verbs") axis labels
  function withAxes(table) {
    const outer = el("div", "axes-wrap");
    const xLabel = el("div", "axis-x");
    xLabel.appendChild(hlKw("Keywords"));
    xLabel.appendChild(el("span", "axis-sub", " (what the course talks about)"));
    outer.appendChild(xLabel);
    const row = el("div", "axes-row");
    const yLabel = el("div", "axis-y");
    yLabel.appendChild(hlVerb("Verbs"));
    yLabel.appendChild(el("span", "axis-sub", " (what the course asks students to do)"));
    row.appendChild(yLabel);
    row.appendChild(table);
    outer.appendChild(row);
    return outer;
  }

  // ── Main grid (all courses combined) ─────────────────────────────────────
  function renderCorpus() {
    viewCorpus.replaceChildren();
    const isPmi = state.metric === "pmi";
    viewCorpus.appendChild(el("h2", null, isPmi
      ? "Distinctiveness of Association — all 347 courses combined"
      : "Number of Sentences — all 347 courses combined"));
    viewCorpus.appendChild(el("p", "view-sub", isPmi
      ? "Each cell says how many times more often the two groups share a sentence than " +
        "they would by pure coincidence. Red means the pairing happens more than chance " +
        "(2× = twice as often); blue means less; 1× means no relationship either way. " +
        "Hover for the courses that use a pairing most; click to " +
        "open it."
      : "The plain count of sentences where the two groups appear together. Read this " +
        "alongside the distinctiveness view: a striking pairing backed by only a handful " +
        "of sentences is a thinner finding."));

    let vmaxLog = 0, vmaxCount = 0;
    const cells = {};
    for (const v of VERB_CATS) for (const k of KW_CATS) {
      const c = corpusCell(v, k);
      cells[key(v, k)] = c;
      if (c.a >= MIN_COOCCUR && c.ratio !== null)
        vmaxLog = Math.max(vmaxLog, Math.abs(log2(c.ratio)));
      vmaxCount = Math.max(vmaxCount, c.a);
    }
    vmaxLog = vmaxLog || 1;
    vmaxCount = vmaxCount || 1;

    const table = el("table", "heatmap");
    const thead = el("thead");
    const hr = el("tr");
    hr.appendChild(el("th"));
    for (const k of KW_CATS)
      hr.appendChild(headerCell("th", k, D.kw_words[k], "“" + k + "” keywords", "hl-kw"));
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el("tbody");
    for (const v of VERB_CATS) {
      const tr = el("tr");
      tr.appendChild(headerCell("th", vName(v), D.verb_words[v], "“" + vName(v) + "” verbs", "hl-verb"));
      for (const k of KW_CATS) {
        const c = cells[key(v, k)];
        const td = el("td", "cell");
        td.setAttribute("tabindex", "0");
        if (c.a < MIN_COOCCUR || c.ratio === null) {
          td.classList.add("na");
          td.textContent = "·";
        } else if (isPmi) {
          const t = log2(c.ratio) / vmaxLog;
          td.style.background = divColor(t);
          td.style.color = inkFor(Math.abs(t));
          td.textContent = fmtRatio(c.ratio);
        } else {
          const t = c.a / vmaxCount;
          td.style.background = seqColor(t);
          td.style.color = inkFor(t);
          td.textContent = String(c.a);
        }
        const activate = () => { state.cell = [v, k]; renderCellView(v, k); setView("cell"); };
        td.addEventListener("click", activate);
        td.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(); } });
        attachHover(td, (tt) => {
          tt.appendChild(el("div", "tt-value",
            c.ratio === null ? "never appear together"
              : fmtRatio(c.ratio) + " as often as chance · " + c.a + " sentences"));
          tt.appendChild(el("div", "tt-label", vName(v) + " verbs × " + k + " keywords"));
          const top = topSyllabi(v, k, 10);
          if (top.length) {
            tt.appendChild(el("hr"));
            tt.appendChild(el("div", "tt-label", "Courses that use this pairing most (sentences per 100):"));
            const ul = el("ul", "tt-list");
            for (const r of top) {
              const li = el("li");
              li.appendChild(el("span", "name", r.s.s));
              li.appendChild(el("span", null, r.rate.toFixed(1)));
              ul.appendChild(li);
            }
            tt.appendChild(ul);
          }
          const cites = topCitations(v, k, 3);
          if (cites.length) {
            tt.appendChild(el("hr"));
            tt.appendChild(el("div", "tt-label", "Texts these courses often cite:"));
            for (const e of cites) {
              const title = e.title.length > 60 ? e.title.slice(0, 57) + "…" : e.title;
              tt.appendChild(el("div", "tt-label tt-cite", title + " (" + e.n + " courses)"));
            }
          }
        });
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    viewCorpus.appendChild(withAxes(table));

    viewCorpus.appendChild(isPmi
      ? legend(`linear-gradient(to right, var(--div-neg), var(--div-mid) 50%, var(--div-pos))`,
               "less often than chance", "as often as chance", "more often than chance")
      : legend(`linear-gradient(to right, ${seqColor(0)}, ${seqColor(1)})`,
               "0 sentences", null, vmaxCount + " sentences"));
  }

  // ── Pairing view (one verb group × one keyword group, over time) ─────────
  function renderCellView(v, k) {
    viewCell.replaceChildren();
    const c = corpusCell(v, k);
    const h2 = el("h2");
    h2.appendChild(hlVerb(vName(v) + " verbs"));
    h2.appendChild(document.createTextNode(" × "));
    h2.appendChild(hlKw(k + " keywords"));
    viewCell.appendChild(h2);
    viewCell.appendChild(el("p", "view-sub",
      "Across all courses, these two groups share " + c.a + " sentences" +
      (c.ratio !== null ? " — " + fmtRatio(c.ratio) + " as often as pure coincidence" : "") +
      ". Each column is a year and each block is one course that uses this pairing, " +
      "stacked with the heaviest users at the bottom — so a taller, darker column means the " +
      "pairing is spreading. Click a block (or a row in the table) to see that course's full profile."));

    const rates = D.syllabi.map((s, i) => ({ s, i, ...sylRate(s, v, k) }));
    const used = rates.filter((r) => r.a > 0);
    const vmax = Math.max(...used.map((r) => r.rate), 0.001);

    // Chart and word lists sit side by side
    const flex = el("div", "pair-flex");

    const chartCol = el("div", "pair-chart-col");
    const barsRow = el("div", "bars-row");
    const years = [...new Set(D.syllabi.map((s) => s.y))].sort();
    // Popular pairings can involve nearly every course in a year; shrink the
    // blocks so the tallest column stays around 450px instead of growing unbounded.
    const maxPerYear = Math.max(1, ...years.map(
      (y) => used.filter((r) => r.s.y === y).length));
    const segH = Math.max(4, Math.min(13, Math.floor(450 / maxPerYear) - 2));
    const segGap = segH < 7 ? "1px" : "2px";
    for (const y of years) {
      const inYear = used.filter((r) => r.s.y === y).sort((a, b) => b.rate - a.rate);
      const total = D.syllabi.filter((s) => s.y === y).length;
      const col = el("div", "bar-col");
      const bar = el("div", "bar-stack");
      bar.style.gap = segGap;
      for (const r of inYear) {
        const seg = el("button", "seg");
        seg.type = "button";
        seg.style.height = segH + "px";
        seg.style.background = seqColor(r.rate / vmax);
        seg.setAttribute("aria-label",
          "Course " + r.s.s + ": " + r.a + " sentences with this pairing");
        attachHover(seg, (tt) => {
          tt.appendChild(el("div", "tt-value", r.rate.toFixed(1) + " sentences per 100"));
          tt.appendChild(el("div", "tt-label",
            "Course " + r.s.s + " · " + r.a + " of its " + r.s.n + " sentences use this pairing"));
          const vs = sylVsAverage(r.s, v, k);
          if (vs !== null) tt.appendChild(el("div", "tt-label",
            "about " + fmtRatio(vs) + " the average course"));
        });
        seg.addEventListener("click", () => { state.syl = r.i; renderSylView(r.i); setView("syllabus"); });
        bar.appendChild(seg);
      }
      col.appendChild(bar);
      col.appendChild(el("div", "bar-label", String(y)));
      col.appendChild(el("div", "bar-sublabel", inYear.length + " of " + total));
      barsRow.appendChild(col);
    }
    chartCol.appendChild(barsRow);
    chartCol.appendChild(el("p", "view-sub chart-note",
      "Under each year: how many of that year's courses use the pairing at least once. " +
      "Darker blocks use it more heavily (relative to their length)."));
    chartCol.appendChild(legend(
      `linear-gradient(to right, ${seqColor(0)}, ${seqColor(1)})`,
      "rarely", null, "up to " + vmax.toFixed(1) + " sentences per 100"));
    flex.appendChild(chartCol);

    // The words behind the two categories
    const words = el("div", "pair-words");
    const vw = el("p", "word-list");
    const vwLabel = el("strong");
    vwLabel.appendChild(hlVerb("“" + vName(v) + "” verbs"));
    vwLabel.appendChild(document.createTextNode(": "));
    vw.appendChild(vwLabel);
    vw.appendChild(document.createTextNode(allWords(D.verb_words[v]).join(", ") || "—"));
    words.appendChild(vw);
    const kw = el("p", "word-list");
    const kwLabel = el("strong");
    kwLabel.appendChild(hlKw("“" + k + "” keywords"));
    kwLabel.appendChild(document.createTextNode(": "));
    kw.appendChild(kwLabel);
    kw.appendChild(document.createTextNode(allWords(D.kw_words[k]).join(", ") || "—"));
    words.appendChild(kw);

    // Texts these courses cite unusually often, in the same right-hand column
    const cites = topCitations(v, k, 10);
    if (cites.length) {
      words.appendChild(el("h3", "table-title cites-title", "Texts these courses often cite"));
      const ul = el("ul", "refs");
      for (const e of cites) {
        const li = el("li");
        li.appendChild(el("span", "ref-title", e.title));
        if (e.author) li.appendChild(el("span", "ref-author", " — " + e.author));
        li.appendChild(el("span", "ref-count",
          " · cited by " + e.n + " of these " + e.nPair + " courses"));
        ul.appendChild(li);
      }
      words.appendChild(ul);
    }
    flex.appendChild(words);

    viewCell.appendChild(flex);

    const top = rates.filter((r) => r.a >= MIN_SYL_COOCCUR)
      .sort((x, y) => y.rate - x.rate).slice(0, 10);
    if (top.length) {
      viewCell.appendChild(el("h3", "table-title", "Courses that use this pairing most"));
      const table = el("table", "rank");
      const thr = el("tr");
      for (const h of ["Course", "Year", "Sentences with both", "Total sentences", "Per 100 sentences", "Compared to average"])
        thr.appendChild(el("th", null, h));
      table.appendChild(thr);
      for (const r of top) {
        const tr = el("tr");
        const link = el("td", "syl-link", r.s.s);
        link.setAttribute("tabindex", "0");
        const go = () => { state.syl = r.i; renderSylView(r.i); setView("syllabus"); };
        link.addEventListener("click", go);
        link.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
        tr.appendChild(link);
        tr.appendChild(el("td", null, String(r.s.y)));
        tr.appendChild(el("td", null, String(r.a)));
        tr.appendChild(el("td", null, String(r.s.n)));
        tr.appendChild(el("td", null, r.rate.toFixed(1)));
        const vs = sylVsAverage(r.s, v, k);
        tr.appendChild(el("td", null, vs === null ? "—" : fmtRatio(vs)));
        table.appendChild(tr);
      }
      viewCell.appendChild(table);
    }
  }

  // ── Single-course view ───────────────────────────────────────────────────
  function renderSylView(i) {
    viewSyl.replaceChildren();
    const s = D.syllabi[i];
    const h2 = el("h2", null, "Course " + s.s + " ");
    // s.t is the syllabus's filename stem (e.g. "2020cprize-05nb"). drive_ids.js
    // maps it to the Google Drive file ID of the full submission PDF (e.g.
    // "2020cprize-05.pdf" -- no "nb" suffix) when one exists on the shared
    // Drive. If drive_ids.js has been populated for it, link straight to the
    // file; otherwise fall back to a Drive search for the filename.
    const driveId = (window.DRIVE_IDS || {})[s.t];
    const drive = el("a", "drive-link", "open PDF ↗");
    drive.href = driveId
      ? "https://drive.google.com/file/d/" + driveId + "/view"
      : "https://drive.google.com/drive/search?q=" + encodeURIComponent(s.t);
    drive.target = "_blank";
    drive.rel = "noopener";
    drive.title = driveId
      ? "Opens this syllabus PDF directly — requires Drive access"
      : "Searches the shared Google Drive for this syllabus — requires Drive access";
    h2.appendChild(drive);
    viewSyl.appendChild(h2);
    viewSyl.appendChild(el("p", "view-sub",
      "A " + s.y + " course, " + s.n + " sentences long. Each cell compares this course with " +
      "the average syllabus: red pairings appear more often here than average (2× = twice as " +
      "often), blue less. A dot means the pairing shows up too rarely in this course to judge. " +
      "Click any cell to see that pairing across all courses."));

    let vmaxLog = 0;
    const vs = {};
    for (const v of VERB_CATS) for (const k of KW_CATS) {
      const r = sylVsAverage(s, v, k);
      vs[key(v, k)] = r;
      if (r !== null) vmaxLog = Math.max(vmaxLog, Math.abs(log2(r)));
    }
    vmaxLog = vmaxLog || 1;

    const table = el("table", "heatmap");
    const thead = el("thead");
    const hr = el("tr");
    hr.appendChild(el("th"));
    for (const k of KW_CATS)
      hr.appendChild(headerCell("th", k, D.kw_words[k], "“" + k + "” keywords", "hl-kw"));
    thead.appendChild(hr);
    table.appendChild(thead);

    const tbody = el("tbody");
    for (const v of VERB_CATS) {
      const tr = el("tr");
      tr.appendChild(headerCell("th", vName(v), D.verb_words[v], "“" + vName(v) + "” verbs", "hl-verb"));
      for (const k of KW_CATS) {
        const r = vs[key(v, k)];
        const { a, rate } = sylRate(s, v, k);
        const td = el("td", "cell");
        td.setAttribute("tabindex", "0");
        if (r === null) {
          td.classList.add("na");
          td.textContent = "·";
        } else {
          const t = log2(r) / vmaxLog;
          td.style.background = divColor(t);
          td.style.color = inkFor(Math.abs(t));
          td.textContent = fmtRatio(r);
        }
        const activate = () => { state.cell = [v, k]; renderCellView(v, k); setView("cell"); };
        td.addEventListener("click", activate);
        td.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); activate(); } });
        const corpus = corpusCell(v, k);
        attachHover(td, (tt) => {
          tt.appendChild(el("div", "tt-value",
            r === null ? "too rare here to judge" : fmtRatio(r) + " the average course"));
          tt.appendChild(el("div", "tt-label", vName(v) + " verbs × " + k + " keywords"));
          tt.appendChild(el("div", "tt-label",
            a + " sentences here (" + rate.toFixed(1) + " per 100), vs " +
            ((corpus.a / corpus.n) * 100).toFixed(1) + " per 100 across all courses"));
        });
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    viewSyl.appendChild(withAxes(table));

    viewSyl.appendChild(legend(
      `linear-gradient(to right, var(--div-neg), var(--div-mid) 50%, var(--div-pos))`,
      "less than average", "average", "more than average"));

    // Texts this course cites
    viewSyl.appendChild(el("h3", "table-title", "Texts cited in this course"));
    if (s.refs && s.refs.length) {
      const ul = el("ul", "refs");
      for (const [title, author] of s.refs) {
        const li = el("li");
        li.appendChild(el("span", "ref-title", title));
        if (author) li.appendChild(el("span", "ref-author", " — " + author));
        ul.appendChild(li);
      }
      viewSyl.appendChild(ul);
    } else {
      viewSyl.appendChild(el("p", "view-sub", "No reading list could be extracted from this syllabus."));
    }
  }

  // ── Controls ─────────────────────────────────────────────────────────────
  for (const btn of document.querySelectorAll("#metric-toggle button")) {
    btn.addEventListener("click", () => {
      state.metric = btn.dataset.metric;
      for (const b of document.querySelectorAll("#metric-toggle button"))
        b.classList.toggle("active", b === btn);
      renderCorpus();
    });
  }

  const picker = $("#syllabus-picker");
  D.syllabi
    .map((s, i) => ({ s, i }))
    .sort((a, b) => a.s.y - b.s.y || a.s.s.localeCompare(b.s.s))
    .forEach(({ s, i }) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = s.s;
      picker.appendChild(opt);
    });
  picker.addEventListener("change", () => {
    if (picker.value === "") return;
    state.syl = Number(picker.value);
    state.cell = null;
    renderSylView(state.syl);
    setView("syllabus");
  });

  // No resize listener needed: pageWidth is a fixed px value, and .view /
  // .page-col both already have max-width: 100% in the stylesheet, so
  // narrow viewports shrink them with plain CSS.

  // ── Init ─────────────────────────────────────────────────────────────────
  renderCorpus();
  setView("corpus");
})();
