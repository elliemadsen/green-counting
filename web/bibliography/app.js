/* Citation network — d3 force layout, canvas edges + SVG nodes. */
"use strict";

const nodes = GRAPH.nodes;
const links = GRAPH.links.map(l => ({ ...l }));

/* ---------- palettes ---------- */
const PALETTE = ["#e3447c", "#c7522a", "#d68a58", "#e5c185",
                 "#7ca874", "#008585", "#80c2c2", "#b9d3ed"].toReversed();
const CAT = { light: PALETTE.slice(0, 7), dark: PALETTE.slice(0, 7) };
const OTHER_COLOR = PALETTE[7];
const YEAR_FALLBACK = "#8a8a84"; // neutral gray for texts with no usable year
const TYPES = ["book", "article", "chapter", "report", "website", "other", "film"];

const state = {
  theme: "dark",
  colorMode: "cluster",
  clusterAlgo: "organic",
  nodeStyle: "covers",
  minWeight: 2,
  minCites: 2,
  spacing: 1,
};

/* "balanced" clustering has no fixed count, so reuse the brand palette for
   its first 8 slots and only generate extra hues (golden-angle stepped) if
   a rebuild ever produces more than that. */
const balancedClusterCount = d3.max(nodes, d => d.clusterBalanced) + 1;
function balancedColor(idx) {
  if (idx < PALETTE.length) return PALETTE[idx];
  const hue = (idx * 137.508) % 360; // golden angle: maximal spread for overflow slots
  return d3.hsl(hue, 0.55, state.theme === "dark" ? 0.62 : 0.42).formatHex();
}

/* ---------- derived node metrics ---------- */
const years = nodes.map(d => +d.year).filter(y => y >= 1800 && y <= 2026);
const yearExtent = d3.extent(years);
for (const d of nodes) {
  d.r = 5 * Math.sqrt(d.citations);            // base radius, graph units
  d.fs = Math.max(7, Math.min(22, 4 + d.r * 0.72)); // label font size
}

function seqScale() {
  // grayscale ramp: older recedes toward the surface, newer pops
  const range = state.theme === "dark"
    ? ["#4a4a47", "#ffffff"]
    : ["#d8d7d1", "#0b0b0b"];
  return d3.scaleSequential(d3.interpolateRgb(range[0], range[1])).domain(yearExtent);
}
function colorOf(d) {
  const cat = CAT[state.theme];
  if (state.colorMode === "cluster") {
    if (state.clusterAlgo === "balanced") return balancedColor(d.clusterBalanced);
    return d.cluster >= 0 ? cat[d.cluster] : OTHER_COLOR;
  }
  if (state.colorMode === "year") {
    const y = +d.year;
    return y >= 1800 && y <= 2026 ? seqScale()(y) : YEAR_FALLBACK;
  }
  const i = TYPES.indexOf(d.type);
  return i >= 0 && i < 7 ? CAT[state.theme][i] : OTHER_COLOR;
}

/* ---------- DOM ---------- */
const stage = document.getElementById("stage");
const canvas = document.getElementById("edge-canvas");
const ctx = canvas.getContext("2d");
const svg = d3.select("#node-layer");
const gNodes = svg.append("g");
const tooltip = document.getElementById("tooltip");
const dpr = window.devicePixelRatio || 1;

let W = stage.clientWidth, H = stage.clientHeight;
let transform = d3.zoomIdentity;
let hoveredNode = null, hoveredLink = null;
let pinnedNode = null;

function resize() {
  W = stage.clientWidth; H = stage.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  drawEdges();
}
// A plain window "resize" listener only catches the *window* resizing, but
// this page is often embedded in an iframe whose own box can still be
// settling (e.g. while the parent's flex layout resolves) after this script
// starts running -- ResizeObserver reacts to #stage's actual box changing
// for any reason. Resizing the canvas alone isn't enough though: fitView()
// (called once from start(), below) computes the initial zoom/pan transform
// from this same W/H, and if that ran against a stale measurement the view
// stays permanently mis-zoomed/off-center even after the canvas itself
// gets corrected -- which is also what made panning look broken (it was
// working, just on content scaled/positioned way outside the visible area).
// So keep re-fitting on every resize until boot has finished AND the user
// hasn't actually touched the view yet (userInteracted, set in zoom's "zoom"
// handler below) -- once either is true it stops touching their view.
let booted = false;
new ResizeObserver(() => {
  resize();
  if (booted && !userInteracted) fitView();
}).observe(stage);

/* ---------- visibility filters ---------- */
let visibleNodes = [], visibleLinks = [], adjacency = new Map();
function applyFilters() {
  const keep = new Set();
  visibleNodes = nodes.filter(d => d.citations >= state.minCites);
  visibleNodes.forEach(d => keep.add(d.id));
  visibleLinks = links.filter(l =>
    l.weight >= state.minWeight && keep.has(l.source.id) && keep.has(l.target.id));
  adjacency = new Map();
  for (const l of visibleLinks) {
    if (!adjacency.has(l.source.id)) adjacency.set(l.source.id, new Set());
    if (!adjacency.has(l.target.id)) adjacency.set(l.target.id, new Set());
    adjacency.get(l.source.id).add(l.target.id);
    adjacency.get(l.target.id).add(l.source.id);
  }
  document.getElementById("stats").textContent =
    `${visibleNodes.length} texts · ${visibleLinks.length} links shown`;
}

/* ---------- force layout (headless, chunked) ---------- */
const sim = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(l => 26 + 46 / l.weight))
  .force("charge", d3.forceManyBody().strength(-110).theta(0.95))
  .force("collide", d3.forceCollide(d => d.r * 1.7 + 2).iterations(1))
  .force("x", d3.forceX(0).strength(0.045))
  .force("y", d3.forceY(0).strength(0.06))
  .stop();

const TOTAL_TICKS = 320;
let ticksDone = 0;
const loading = document.getElementById("loading");
function layoutChunk() {
  const n = Math.min(15, TOTAL_TICKS - ticksDone);
  for (let i = 0; i < n; i++) sim.tick();
  ticksDone += n;
  loading.textContent = `computing layout… ${Math.round(100 * ticksDone / TOTAL_TICKS)}%`;
  if (ticksDone < TOTAL_TICKS) requestAnimationFrame(layoutChunk);
  else start();
}

function fitView() {
  const xs = d3.extent(nodes, d => d.x), ys = d3.extent(nodes, d => d.y);
  const k = Math.min(2, 0.92 * Math.min(W / (xs[1] - xs[0] + 80), H / (ys[1] - ys[0] + 80)));
  transform = d3.zoomIdentity
    .translate(W / 2 - k * (xs[0] + xs[1]) / 2, H / 2 - k * (ys[0] + ys[1]) / 2)
    .scale(k);
  d3.select(stage).call(zoom.transform, transform);
}

/* ---------- zoom ---------- */
let userInteracted = false;
const zoom = d3.zoom()
  .scaleExtent([0.1, 14])
  .clickDistance(6) // distinguishes a drag-pan from a click, so panning never unfocuses a pin
  .on("zoom", (ev) => {
    // ev.sourceEvent is null for programmatic transforms (fitView calling
    // zoom.transform below) and set for real mouse/touch/wheel input --
    // used to know when to stop auto-refitting on resize (see boot section).
    if (ev.sourceEvent) userInteracted = true;
    transform = ev.transform;
    gNodes.attr("transform", transform);
    drawEdges();
    updateLabelZoom();
    if (pinnedViewIsUncontested()) positionPinnedTooltip();
  });
d3.select(stage).call(zoom).on("dblclick.zoom", null);

/* ---------- edge drawing (screen space) ---------- */
function edgeBase() {
  return state.theme === "dark" ? "195,194,183" : "82,81,78";
}
function drawEdges() {
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const t = transform, rgb = edgeBase();
  const focus = hoveredNode || pinnedNode;
  const focusId = focus ? focus.id : null;
  ctx.lineCap = "round";
  for (const l of visibleLinks) {
    const isFocus = focusId !== null &&
      (l.source.id === focusId || l.target.id === focusId);
    if (focusId !== null && !isFocus && l !== hoveredLink) {
      ctx.strokeStyle = `rgba(${rgb},0.03)`;
      ctx.lineWidth = 0.6;
    } else if (l === hoveredLink || isFocus) {
      ctx.strokeStyle = `rgba(${rgb},0.9)`;
      ctx.lineWidth = 1.2 + 0.7 * Math.min(l.weight, 6);
    } else {
      ctx.strokeStyle = `rgba(${rgb},${Math.min(0.72, 0.1 + 0.11 * (l.weight - 1))})`;
      ctx.lineWidth = 0.6 + 0.45 * Math.min(l.weight - 1, 6);
    }
    ctx.beginPath();
    ctx.moveTo(t.applyX(l.source.x), t.applyY(l.source.y));
    ctx.lineTo(t.applyX(l.target.x), t.applyY(l.target.y));
    ctx.stroke();
  }
}

/* ---------- node rendering ---------- */
function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

/* Counter-scale labels as the user zooms in, and lift truncation once
   there is room: zoomed out shows short labels, zoomed in shows full titles
   at a (relatively) smaller size. */
let labelKey = "";
function updateLabelZoom(force) {
  const k = transform.k;
  const shrink = Math.min(1, Math.pow(k, -0.35)); // only shrink past k=1
  const tier = k < 1.6 ? 0 : k < 2.8 ? 1 : 2;
  const key = tier + ":" + Math.round(shrink * 24);
  if (!force && key === labelKey) return;
  labelKey = key;
  const maxT = tier === 0 ? 36 : tier === 1 ? 64 : 999;
  const maxA = tier === 0 ? 30 : tier === 1 ? 50 : 999;
  gNodes.selectAll("text.title").each(function (d) {
    this.setAttribute("font-size", +this.dataset.base * shrink);
    this.textContent = truncate(d.title, maxT);
  });
  gNodes.selectAll("text.author").each(function (d) {
    this.setAttribute("font-size", +this.dataset.base * shrink);
    this.textContent = truncate(d.authors.split(";")[0], maxA);
  });
}

function renderNodes() {
  const style = state.nodeStyle;
  const sel = gNodes.selectAll("g.node")
    .data(visibleNodes, d => d.id)
    .join("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${d.x},${d.y})`);
  sel.selectAll("*").remove();

  sel.each(function (d) {
    const g = d3.select(this);
    const asCover = style === "covers" && d.cover;
    if (asCover) {
      const w = d.r * 2.6, h = w * 1.45;
      g.append("image")
        .attr("href", `covers/${d.id}.jpg`)
        .attr("x", -w / 2).attr("y", -h / 2)
        .attr("width", w).attr("height", h)
        .attr("preserveAspectRatio", "xMidYMid meet");
      g.append("rect").attr("class", "cover-frame")
        .attr("x", -w / 2).attr("y", -h / 2)
        .attr("width", w).attr("height", h)
        .attr("stroke", colorOf(d)).attr("stroke-width", 0);
    } else if (style === "dots") {
      g.append("circle").attr("r", d.r * 0.62).attr("fill", colorOf(d));
      if (d.citations >= 4)
        g.append("text").attr("class", "title")
          .attr("text-anchor", "middle")
          .attr("y", d.r * 0.62 + d.fs * 0.9)
          .attr("font-size", d.fs * 0.8)
          .attr("data-base", d.fs * 0.8)
          .attr("fill", colorOf(d))
          .text(truncate(d.title, 30));
    } else {
      g.append("text").attr("class", "title")
        .attr("text-anchor", "middle")
        .attr("font-size", d.fs)
        .attr("data-base", d.fs)
        .attr("fill", colorOf(d))
        .text(truncate(d.title, 36));
      if (d.fs >= 10 && d.authors)
        g.append("text").attr("class", "author")
          .attr("text-anchor", "middle")
          .attr("y", d.fs)
          .attr("font-size", d.fs * 0.68)
          .attr("data-base", d.fs * 0.68)
          .text(truncate(d.authors.split(";")[0], 30));
    }
  });
  updateLabelZoom(true);

  sel.on("pointerenter", (ev, d) => {
      if (!canHoverNode(d)) return;
      hoveredNode = d; hoveredLink = null;
      applyFocusHighlight(d);
      showNodeTooltip(ev, d);
    }).on("pointermove", (ev, d) => { if (canHoverNode(d)) positionTooltip(ev); })
    .on("pointerleave", () => {
      hoveredNode = null;
      restorePinnedView();
    })
    .on("click", (ev, d) => {
      ev.stopPropagation();
      setPinned(d, ev);
    });

  if (pinnedNode) applyFocusHighlight(pinnedNode);
}

/* While a node is pinned, hover previews are restricted to that node itself,
   its neighbors, and its own edges — everything else stays inert. */
function canHoverNode(d) {
  if (!pinnedNode) return true;
  if (d.id === pinnedNode.id) return true;
  return (adjacency.get(pinnedNode.id) || new Set()).has(d.id);
}
function applyFocusHighlight(d) {
  svg.classed("dimming", true);
  const neigh = adjacency.get(d.id) || new Set();
  gNodes.selectAll("g.node").classed("active", n => n.id === d.id || neigh.has(n.id));
  drawEdges();
}
function clearFocusHighlight() {
  svg.classed("dimming", false);
  gNodes.selectAll("g.node").classed("active", false);
  drawEdges();
}

/* ---------- click-to-pin ---------- */
function setPinned(d, ev) {
  pinnedNode = d;
  hoveredNode = d;
  hoveredLink = null;
  applyFocusHighlight(d);
  showNodeTooltip(ev, d);
  positionPinnedTooltip();
}
function clearPinned() {
  if (!pinnedNode) return;
  pinnedNode = null;
  hoveredNode = null;
  clearFocusHighlight();
  tooltip.hidden = true;
}
/* True when nothing is currently overriding the pinned display — i.e. no
   OTHER node or a link is being actively hover-previewed. hoveredNode can
   legitimately equal pinnedNode (the mouse simply never left it), which
   must still count as "uncontested" so zoom/pan keep tracking and rescaling
   the pinned tooltip even if the cursor hasn't moved since the click. */
function pinnedViewIsUncontested() {
  return pinnedNode && (!hoveredNode || hoveredNode === pinnedNode) && !hoveredLink;
}
/* The pinned tooltip shrinks a little as the user zooms out, instead of
   staying screen-locked at a fixed size regardless of camera distance.
   Capped at 1 (never grows past normal size zooming in) with a 0.4 floor
   so it stays legible even at the widest zoom-out. Disabled for now. */
const ENABLE_PINNED_TOOLTIP_SCALING = false;
function pinnedTooltipScale() {
  if (!ENABLE_PINNED_TOOLTIP_SCALING) return 1;
  return Math.max(0.4, Math.min(1, Math.pow(transform.k, 0.25)));
}
function positionPinnedTooltip() {
  if (!pinnedNode) return;
  positionTooltip({
    clientX: transform.applyX(pinnedNode.x),
    clientY: transform.applyY(pinnedNode.y),
  });
  tooltip.style.transformOrigin = "top left";
  tooltip.style.transform = `scale(${pinnedTooltipScale()})`;
}
/* Called when a transient hover (node or link) ends: falls back to the
   pinned node's highlight + tooltip, or clears entirely if nothing is pinned. */
function restorePinnedView() {
  if (!pinnedNode) {
    clearFocusHighlight();
    tooltip.hidden = true;
    return;
  }
  applyFocusHighlight(pinnedNode);
  showNodeTooltip({
    clientX: transform.applyX(pinnedNode.x),
    clientY: transform.applyY(pinnedNode.y),
  }, pinnedNode);
  tooltip.style.transformOrigin = "top left";
  tooltip.style.transform = `scale(${pinnedTooltipScale()})`;
}
stage.addEventListener("click", (ev) => {
  if (!pinnedNode) return;
  if (ev.target.closest?.(".node")) return; // the node's own click handler already ran
  clearPinned();
});

function recolorNodes() {
  gNodes.selectAll("g.node").each(function (d) {
    d3.select(this).selectAll("text.title, circle").attr("fill", colorOf(d));
  });
}

/* ---------- tooltips (textContent only) ---------- */
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
function syllabiLine(list) {
  const cap = 24;
  const shown = list.slice(0, cap).join("; ");
  const extra = list.length > cap ? ` … +${list.length - cap} more` : "";
  return el("div", "tt-syllabi", shown + extra);
}
function showNodeTooltip(ev, d) {
  tooltip.style.transform = ""; // full size by default; pinned callers rescale after
  tooltip.replaceChildren();
  const head = el("div", "tt-head");
  if (d.cover) {
    const img = document.createElement("img");
    img.src = `covers/${d.id}.jpg`;
    head.appendChild(img);
  }
  const info = el("div");
  info.appendChild(el("div", "tt-title", d.title));
  if (d.authors) info.appendChild(el("div", "tt-sub", d.authors));
  const bits = [d.year, d.publisher || d.journal, d.type].filter(Boolean);
  info.appendChild(el("div", "tt-meta", bits.join(" · ")));
  head.appendChild(info);
  tooltip.appendChild(head);
  tooltip.appendChild(el("div", "tt-value",
    `Cited in ${d.citations} syllabi`));
  tooltip.appendChild(syllabiLine(d.syllabi));
  tooltip.hidden = false;
  positionTooltip(ev);
}
function showLinkTooltip(ev, l) {
  tooltip.style.transform = "";
  tooltip.replaceChildren();
  const key = el("span", "tt-key");
  key.style.borderTopColor = state.theme === "dark" ? "#c3c2b7" : "#52514e";
  const v = el("div", "tt-value");
  v.appendChild(key);
  v.appendChild(document.createTextNode(
    `Co-cited in ${l.weight} ${l.weight === 1 ? "syllabus" : "syllabi"}`));
  tooltip.appendChild(v);
  tooltip.appendChild(el("div", "tt-sub", l.source.title));
  tooltip.appendChild(el("div", "tt-sub", "↔ " + l.target.title));
  tooltip.appendChild(syllabiLine(l.syllabi));
  tooltip.hidden = false;
  positionTooltip(ev);
}
function positionTooltip(ev) {
  const pad = 14;
  const r = tooltip.getBoundingClientRect();
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - pad;
  tooltip.style.left = x + "px";
  tooltip.style.top = y + "px";
}

/* ---------- edge hover (nearest segment) ---------- */
function distToSegment(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const len2 = dx * dx + dy * dy;
  let t = len2 ? ((px - ax) * dx + (py - ay) * dy) / len2 : 0;
  t = Math.max(0, Math.min(1, t));
  const qx = ax + t * dx, qy = ay + t * dy;
  return Math.hypot(px - qx, py - qy);
}
stage.addEventListener("pointermove", (ev) => {
  if (hoveredNode || ev.target.closest?.(".node")) return;
  const [gx, gy] = transform.invert([ev.clientX, ev.clientY]);
  const tol = 7 / transform.k;
  let best = null, bestD = tol;
  const candidates = pinnedNode
    ? visibleLinks.filter(l => l.source.id === pinnedNode.id || l.target.id === pinnedNode.id)
    : visibleLinks;
  for (const l of candidates) {
    const d = distToSegment(gx, gy, l.source.x, l.source.y, l.target.x, l.target.y);
    if (d < bestD) { bestD = d; best = l; }
  }
  if (best !== hoveredLink) {
    hoveredLink = best;
    drawEdges();
    if (best) showLinkTooltip(ev, best);
    else restorePinnedView();
  } else if (best) positionTooltip(ev);
});
stage.addEventListener("pointerleave", () => {
  if (hoveredLink) { hoveredLink = null; drawEdges(); restorePinnedView(); }
});

/* ---------- legend ---------- */
function renderLegend() {
  const legend = document.getElementById("legend");
  legend.replaceChildren();
  const cat = CAT[state.theme];
  if (state.colorMode === "cluster" && state.clusterAlgo === "balanced") {
    const counts = d3.rollup(nodes, v => v.length, d => d.clusterBalanced);
    for (let c = 0; c < balancedClusterCount; c++) {
      if (!counts.has(c)) continue;
      const row = el("div", "legend-row");
      const sw = el("span", "swatch"); sw.style.background = balancedColor(c);
      row.appendChild(sw);
      row.appendChild(el("span", null, `Cluster ${c + 1} (${counts.get(c)} texts)`));
      legend.appendChild(row);
    }
  } else if (state.colorMode === "cluster") {
    const counts = d3.rollup(nodes, v => v.length, d => d.cluster);
    for (let c = 0; c < 7; c++) {
      if (!counts.has(c)) continue;
      const row = el("div", "legend-row");
      const sw = el("span", "swatch"); sw.style.background = cat[c];
      row.appendChild(sw);
      row.appendChild(el("span", null, `Cluster ${c + 1} (${counts.get(c)} texts)`));
      legend.appendChild(row);
    }
    if (counts.has(-1)) {
      const row = el("div", "legend-row");
      const sw = el("span", "swatch"); sw.style.background = OTHER_COLOR;
      row.appendChild(sw);
      row.appendChild(el("span", null, `Other (${counts.get(-1)} texts)`));
      legend.appendChild(row);
    }
  } else if (state.colorMode === "year") {
    const s = seqScale();
    const stops = d3.range(0, 1.01, 0.1)
      .map(t => s(yearExtent[0] + t * (yearExtent[1] - yearExtent[0])));
    const ramp = el("div", "ramp");
    ramp.style.background = `linear-gradient(to right, ${stops.join(",")})`;
    legend.appendChild(ramp);
    const lab = el("div", "ramp-labels");
    lab.appendChild(el("span", null, String(yearExtent[0])));
    lab.appendChild(el("span", null, String(yearExtent[1])));
    legend.appendChild(lab);
  } else {
    const counts = d3.rollup(nodes, v => v.length, d => d.type);
    TYPES.forEach((t, i) => {
      if (!counts.has(t)) return;
      const row = el("div", "legend-row");
      const sw = el("span", "swatch"); sw.style.background = cat[i];
      row.appendChild(sw);
      row.appendChild(el("span", null, `${t} (${counts.get(t)})`));
      legend.appendChild(row);
    });
  }
}

/* ---------- controls ---------- */
function refresh() {
  applyFilters();
  if (pinnedNode && !visibleNodes.includes(pinnedNode)) clearPinned();
  renderNodes();
  drawEdges();
  renderLegend();
}
/* spacing slider: retune forces and let the layout settle in place */
function relayout() {
  const m = state.spacing;
  sim.force("charge").strength(-110 * m);
  sim.force("collide").radius(d => (d.r * 1.7 + 2) * m);
  sim.force("link").distance(l => (26 + 46 / l.weight) * Math.sqrt(m));
  sim.force("x").strength(0.045 / m);
  sim.force("y").strength(0.06 / m);
  sim.alpha(0.5);
  animateTicks(160);
}
function animateTicks(n) {
  function step(remaining) {
    for (let i = 0; i < 4 && remaining > 0; i++, remaining--) sim.tick();
    gNodes.selectAll("g.node").attr("transform", d => `translate(${d.x},${d.y})`);
    drawEdges();
    if (pinnedViewIsUncontested()) positionPinnedTooltip();
    if (remaining > 0 && sim.alpha() > 0.02) requestAnimationFrame(() => step(remaining));
  }
  requestAnimationFrame(() => step(n));
}
document.getElementById("spacing").addEventListener("input", (e) => {
  document.getElementById("spacing-val").textContent = (+e.target.value).toFixed(1);
});
document.getElementById("spacing").addEventListener("change", (e) => {
  state.spacing = +e.target.value;
  relayout();
});
document.getElementById("edge-weight").addEventListener("input", (e) => {
  state.minWeight = +e.target.value;
  document.getElementById("edge-weight-val").textContent = e.target.value;
  refresh();
});
document.getElementById("min-cites").addEventListener("input", (e) => {
  state.minCites = +e.target.value;
  document.getElementById("min-cites-val").textContent = e.target.value;
  refresh();
});
document.getElementById("node-style").addEventListener("change", (e) => {
  state.nodeStyle = e.target.value;
  renderNodes();
});
document.getElementById("color-mode").addEventListener("change", (e) => {
  state.colorMode = e.target.value;
  recolorNodes();
  renderLegend();
});
document.getElementById("cluster-algo")?.addEventListener("change", (e) => {
  state.clusterAlgo = e.target.value;
  recolorNodes();
  renderLegend();
});
document.getElementById("theme-mode").addEventListener("change", (e) => {
  state.theme = e.target.value;
  document.documentElement.dataset.theme = state.theme;
  recolorNodes();
  drawEdges();
  renderLegend();
});
document.getElementById("sidebar-btn").addEventListener("click", (e) => {
  const sb = document.getElementById("sidebar");
  sb.classList.toggle("collapsed");
  e.currentTarget.setAttribute("aria-expanded", String(!sb.classList.contains("collapsed")));
});
const infoPanel = document.getElementById("info-panel");
document.getElementById("info-btn").addEventListener("click", (e) => {
  infoPanel.hidden = !infoPanel.hidden;
  e.currentTarget.setAttribute("aria-expanded", String(!infoPanel.hidden));
});
document.getElementById("info-close").addEventListener("click", () => {
  infoPanel.hidden = true;
});

/* ---------- boot ---------- */
const urlTheme = new URLSearchParams(location.search).get("theme");
if (urlTheme === "light" || urlTheme === "dark") {
  state.theme = urlTheme;
  document.querySelector(`#theme-mode input[value="${urlTheme}"]`).checked = true;
}
document.documentElement.dataset.theme = state.theme;
resize();
function start() {
  applyFilters();
  fitView();
  renderNodes();
  drawEdges();
  renderLegend();
  loading.classList.add("done");
  booted = true;
}
layoutChunk();
