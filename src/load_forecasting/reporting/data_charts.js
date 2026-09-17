/* Self-contained chart renderer for the data-inspection report.
   Consumes window.__REPORT_DATA__ (see reporting/data_payload.py).

   Layout follows the stacked-panels idea from the earlier ts_forecast
   plot_data_splits(): the target on top, one panel per covariate below, all
   sharing the same time window, with training / validation drawn as coloured
   line segments and meteorological seasons as background bands.

   Depends on globals: d3, Plot (both vendored and inlined).
*/
(function () {
  "use strict";

  const DATA = window.__REPORT_DATA__ || {};
  const Plot = window.Plot;
  if (!Plot) { console.error("Observable Plot did not load"); return; }

  // ---------------------------------------------------------------------
  // Palette
  // ---------------------------------------------------------------------
  const ROLE_COLOR = { training: "#2563eb", validation: "#f97316", none: "#6b7280" };
  const ROLE_LABEL = { training: "Training", validation: "Validation", none: "No split" };
  const SEASON_COLOR = { winter: "#93c5fd", spring: "#86efac", summer: "#fca5a5", fall: "#fcd34d" };
  const SEASON_ORDER = ["winter", "spring", "summer", "fall"];
  const OUTLIER_COLOR = "#b91c1c";
  const SPIKE_COLOR = "#7c3aed";
  const GAP_COLOR = "#9ca3af";
  const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }
  function setText(id, text) { const el = $(id); if (el) el.textContent = text; }
  function clear(el) { while (el && el.firstChild) el.removeChild(el.firstChild); }
  function fmt(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const digits = d === undefined ? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2) : d;
    return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  function fmtDate(s) { return s ? String(s).slice(0, 16).replace("T", " ") : "—"; }
  function fmtDay(s) { return s ? String(s).slice(0, 10) : "—"; }
  function utc(s) { return new Date(s); }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ""; }
  function roleName(code) { return code === 0 ? "training" : code === 1 ? "validation" : "none"; }

  function makeKV(label, value) {
    const kv = document.createElement("div"); kv.className = "kv";
    const k = document.createElement("div"); k.className = "k"; k.textContent = label;
    const v = document.createElement("div"); v.className = "v"; v.textContent = value;
    kv.appendChild(k); kv.appendChild(v);
    return kv;
  }
  function makeTable(headers, rows, numericCols) {
    const table = document.createElement("table"); table.className = "data";
    const thead = document.createElement("thead"); const tr = document.createElement("tr");
    headers.forEach(h => { const th = document.createElement("th"); th.textContent = h; tr.appendChild(th); });
    thead.appendChild(tr); table.appendChild(thead);
    const tbody = document.createElement("tbody");
    rows.forEach(r => {
      const tr2 = document.createElement("tr");
      r.forEach((c, i) => {
        const td = document.createElement("td");
        if (numericCols && numericCols.includes(i)) td.className = "num";
        td.textContent = c; tr2.appendChild(td);
      });
      tbody.appendChild(tr2);
    });
    table.appendChild(tbody);
    return table;
  }

  // Columnar series -> array of {t, v, role, seg} points, split into
  // segments so lines never bridge a gap or a role change.
  const S = DATA.series || { t: [], target: [], role: [], covariates: {} };
  const T = S.t.map(utc);
  const STEP_MS = (() => {
    const f = (DATA.meta || {}).frequency;
    return f === "15min" ? 900000 : f === "30min" ? 1800000 : 3600000;
  })() * (S.stride || 1);

  function toPoints(values, lo, hi) {
    const out = [];
    let seg = 0;
    for (let i = 0; i < T.length; i++) {
      const t = T[i];
      if (lo && t < lo) continue;
      if (hi && t > hi) break;
      const v = values[i];
      const role = roleName(S.role[i]);
      if (out.length) {
        const prev = out[out.length - 1];
        if (prev.role !== role || (t - prev.t) > STEP_MS * 1.5) seg++;
      }
      if (v === null || v === undefined || !isFinite(v)) { seg++; continue; }
      out.push({ t, v, role, seg });
    }
    return out;
  }

  // ---------------------------------------------------------------------
  // Time-window state (range buttons)
  // ---------------------------------------------------------------------
  const FULL = { label: "Full range", start: T[0], end: T[T.length - 1] };
  const state = { range: FULL };

  function seasonSpans() {
    return (DATA.seasons || []).map(sp => ({
      season: sp.season, start: utc(sp.start), end: utc(sp.end),
    }));
  }

  function rangeButtons() {
    const buttons = [FULL];
    const spans = seasonSpans();
    const counts = {};
    spans.forEach(sp => { counts[sp.season] = (counts[sp.season] || 0) + 1; });
    const seen = {};
    spans.forEach(sp => {
      seen[sp.season] = (seen[sp.season] || 0) + 1;
      const m1 = MONTHS[sp.start.getUTCMonth()], m2 = MONTHS[sp.end.getUTCMonth()];
      const months = m1 === m2 ? m1 : `${m1}–${m2}`;
      const label = counts[sp.season] > 1 ? `${cap(sp.season)} (${months})` : cap(sp.season);
      buttons.push({ label, start: sp.start, end: sp.end });
    });
    (DATA.windows || []).forEach(w => buttons.push({ label: w.label, start: utc(w.start), end: utc(w.end) }));
    return buttons;
  }

  function renderRangeBar() {
    const bar = $("range-bar"); clear(bar);
    const lbl = document.createElement("span"); lbl.className = "lbl"; lbl.textContent = "Window:";
    bar.appendChild(lbl);
    rangeButtons().forEach(b => {
      const btn = document.createElement("button");
      btn.textContent = b.label;
      if (b.label === state.range.label) btn.classList.add("active");
      btn.addEventListener("click", () => {
        state.range = b;
        renderRangeBar(); renderOverview(); renderCovariates();
      });
      bar.appendChild(btn);
    });
  }

  // ---------------------------------------------------------------------
  // Shared marks: season bands, split boundaries, gaps
  // ---------------------------------------------------------------------
  function bandMarks(lo, hi, yDomain, withLabels) {
    const marks = [];
    const spans = seasonSpans().filter(sp => sp.end >= lo && sp.start <= hi)
      .map(sp => ({ ...sp, start: sp.start < lo ? lo : sp.start, end: sp.end > hi ? hi : sp.end }));
    if (spans.length) {
      marks.push(Plot.rect(spans, {
        x1: "start", x2: "end", y1: yDomain[0], y2: yDomain[1],
        fill: d => SEASON_COLOR[d.season], fillOpacity: 0.13,
      }));
      if (withLabels) {
        marks.push(Plot.text(spans, {
          x: d => new Date((d.start.getTime() + d.end.getTime()) / 2), y: yDomain[1],
          text: d => cap(d.season), dy: -6, fontSize: 11, fill: "#6b7280", fontWeight: 600,
        }));
      }
    }
    const gaps = (DATA.gaps || []).map(g => ({ start: utc(g.start), end: utc(g.end), n: g.n_missing_steps }))
      .filter(g => g.end >= lo && g.start <= hi);
    if (gaps.length) {
      marks.push(Plot.rect(gaps, {
        x1: "start", x2: "end", y1: yDomain[0], y2: yDomain[1],
        fill: GAP_COLOR, fillOpacity: 0.35, stroke: GAP_COLOR, strokeOpacity: 0.6,
        title: d => `Gap: ${d.n} missing step(s)`,
      }));
    }
    const segs = (DATA.split && DATA.split.segments) || [];
    const bounds = [];
    for (let i = 1; i < segs.length; i++) {
      if (segs[i].role !== segs[i - 1].role) bounds.push(utc(segs[i].start));
    }
    const vis = bounds.filter(b => b >= lo && b <= hi);
    if (vis.length) marks.push(Plot.ruleX(vis, { stroke: "#6b7280", strokeDasharray: "3,3", strokeOpacity: 0.7 }));
    return marks;
  }

  function yExtent(points, pad) {
    if (!points.length) return [0, 1];
    let lo = Infinity, hi = -Infinity;
    for (const p of points) { if (p.v < lo) lo = p.v; if (p.v > hi) hi = p.v; }
    if (lo === hi) { lo -= 1; hi += 1; }
    const span = hi - lo;
    return [lo - span * (pad || 0.05), hi + span * (pad || 0.08)];
  }

  function fullWidth() {
    const host = document.querySelector(".container") || document.body;
    return Math.min(1100, Math.max(600, host.clientWidth - 40));
  }

  function seriesChart(points, opts) {
    const lo = state.range.start, hi = state.range.end;
    const dom = yExtent(points, opts.pad);
    const marks = bandMarks(lo, hi, dom, opts.labels);
    marks.push(Plot.line(points, {
      x: "t", y: "v", z: "seg", stroke: d => ROLE_COLOR[d.role],
      strokeWidth: opts.strokeWidth || 1, strokeOpacity: opts.opacity || 0.95,
    }));
    if (opts.outliers) {
      const val = (DATA.outliers.value || []).map(o => ({ t: utc(o.t), v: o.v })).filter(o => o.t >= lo && o.t <= hi);
      const spk = (DATA.outliers.spikes || []).map(o => ({ t: utc(o.t), v: o.v, step: o.step })).filter(o => o.t >= lo && o.t <= hi);
      if (spk.length) marks.push(Plot.dot(spk, { x: "t", y: "v", r: 2, stroke: SPIKE_COLOR, strokeOpacity: 0.7, title: d => `Step change ${fmt(d.step)}` }));
      if (val.length) marks.push(Plot.dot(val, { x: "t", y: "v", r: 2.6, fill: OUTLIER_COLOR, fillOpacity: 0.85, title: d => `Outlier ${fmt(d.v)}` }));
    }
    marks.push(Plot.tip(points, Plot.pointerX({
      x: "t", y: "v",
      title: d => `${fmtDate(d.t.toISOString())}\n${opts.name}: ${fmt(d.v)}\n${ROLE_LABEL[d.role]}`,
    })));
    return Plot.plot({
      width: fullWidth(),
      height: opts.height || 320,
      marginLeft: 64, marginRight: 16, marginTop: opts.labels ? 22 : 10,
      x: { type: "utc", domain: [lo, hi], label: null },
      y: { domain: dom, label: opts.yLabel || null, grid: true, nice: false },
      marks,
    });
  }

  // ---------------------------------------------------------------------
  // Sections
  // ---------------------------------------------------------------------
  function renderHeader() {
    const m = DATA.meta || {};
    setText("report-subtitle", `${m.csv_path || ""} · ${fmt(m.n_rows, 0)} rows · ${m.frequency || "?"} · ${fmtDay(m.start)} → ${fmtDay(m.end)} (${fmt(m.days, 0)} days)`);
    setText("footer-generated", `Generated ${fmtDate(DATA.generated_at)} UTC · STLF-MCP-Server data report`);
  }

  function renderSummary() {
    const m = DATA.meta || {}; const ins = DATA.inspection || {}; const ts = m.target_stats || {};
    const grid = $("summary-grid"); clear(grid);
    grid.appendChild(makeKV("Target", m.target || "—"));
    grid.appendChild(makeKV("Mean / std", `${fmt(ts.mean)} / ${fmt(ts.std)}`));
    grid.appendChild(makeKV("Min / max", `${fmt(ts.min)} / ${fmt(ts.max)}`));
    grid.appendChild(makeKV("Missing target values", fmt(ts.n_missing, 0)));
    grid.appendChild(makeKV("Timestamp gaps", `${(DATA.gaps || []).length}`));
    grid.appendChild(makeKV("Covariates", `${(m.covariates || []).length}${(m.unmapped_covariates || []).length ? ` (${m.unmapped_covariates.length} unmapped)` : ""}`));
    const sp = DATA.split || {};
    grid.appendChild(makeKV("Split", sp.strategy === "none" ? "none" : `${sp.strategy} · ${Math.round((sp.validation_split || 0) * 100)}% validation`));
    if (ins.time_range && ins.time_range.coverage_pct !== undefined) grid.appendChild(makeKV("Coverage", `${fmt(ins.time_range.coverage_pct, 1)}%`));

    const badge = $("ready-badge");
    if (ins.ready_to_train === true) { badge.textContent = "ready to train"; badge.className = "badge improved"; }
    else if (ins.ready_to_train === false) { badge.textContent = "blocked"; badge.className = "badge degraded"; }
    else { badge.textContent = ""; badge.className = "badge"; }

    const flags = $("summary-flags"); clear(flags);
    (ins.blocking_issues || []).forEach(f => { const s = document.createElement("span"); s.className = "flag block"; s.textContent = f; flags.appendChild(s); });
    (ins.quality_flags || []).forEach(f => { const s = document.createElement("span"); s.className = "flag"; s.textContent = f; flags.appendChild(s); });
    if (!(ins.blocking_issues || []).length && !(ins.quality_flags || []).length && ins.ready_to_train !== undefined) {
      const s = document.createElement("span"); s.className = "flag ok"; s.textContent = "No quality flags"; flags.appendChild(s);
    }
  }

  function renderLegend() {
    const el = $("overview-legend"); clear(el);
    const items = [];
    const hasSplit = (DATA.split && DATA.split.segments && DATA.split.segments.length);
    if (hasSplit) {
      items.push([ROLE_COLOR.training, "Training", ""], [ROLE_COLOR.validation, "Validation", ""]);
    } else {
      items.push([ROLE_COLOR.none, "Target", ""]);
    }
    SEASON_ORDER.forEach(s => { if ((DATA.seasons || []).some(sp => sp.season === s)) items.push([SEASON_COLOR[s], cap(s), "band"]); });
    if ((DATA.gaps || []).length) items.push([GAP_COLOR, "Gap", "band"]);
    if ((DATA.outliers.value || []).length) items.push([OUTLIER_COLOR, "±3σ outlier", "dot"]);
    if ((DATA.outliers.spikes || []).length) items.push([SPIKE_COLOR, "Step change", "dot"]);
    items.forEach(([color, label, kind]) => {
      const span = document.createElement("span");
      const sw = document.createElement("span"); sw.className = "sw " + kind; sw.style.background = color;
      if (kind === "dot") { sw.style.width = "9px"; sw.style.height = "9px"; sw.style.borderRadius = "50%"; }
      span.appendChild(sw); span.appendChild(document.createTextNode(label));
      el.appendChild(span);
    });
  }

  function renderOverview() {
    const host = $("chart-overview"); clear(host);
    const pts = toPoints(S.target, state.range.start, state.range.end);
    host.appendChild(seriesChart(pts, {
      name: DATA.meta.target, yLabel: DATA.meta.target, height: 340, labels: true, outliers: true, strokeWidth: 1.1,
    }));
    const note = [];
    if (S.downsampled) note.push(`Overview series drawn at every ${S.stride}th point (${fmt(S.n_points, 0)} of ${fmt(DATA.meta.n_rows, 0)} rows); every statistic uses the full data.`);
    const w = (DATA.windows || []).find(x => x.label === state.range.label);
    if (w) note.push(`Window centred on the ${w.season} peak: ${fmt(w.peak_value)} at ${fmtDate(w.peak_time)}.`);
    setText("overview-note", note.join(" "));
  }

  function renderSplit() {
    const sp = DATA.split || {}; const segs = sp.segments || [];
    const note = $("split-note");
    if (!segs.length) {
      note.textContent = sp.requested === "none" ? "No split requested." : "No split could be computed.";
      clear($("chart-split-timeline")); clear($("split-table")); return;
    }
    const sum = sp.summary || {};
    let txt = `${cap(sp.strategy)} split, ${Math.round(sp.validation_split * 100)}% validation: ${fmt(sum.training_rows, 0)} training rows, ${fmt(sum.validation_rows, 0)} validation rows in ${sum.n_segments} segments.`;
    if (sp.strategy === "seasonal") txt += " Each meteorological season contributes its last portion to validation (Li et al. 2025); note that winter spans Dec + Jan–Feb, so its blocks are non-contiguous within a calendar year.";
    if (sp.note) txt += " " + sp.note;
    note.textContent = txt;

    const rows = segs.map((s, i) => ({ i, role: s.role, season: s.season || "", start: utc(s.start), end: utc(s.end), n: s.n_rows }));
    const host = $("chart-split-timeline"); clear(host);
    host.appendChild(Plot.plot({
      width: fullWidth(),
      height: 110, marginLeft: 80, marginRight: 16,
      x: { type: "utc", domain: [T[0], T[T.length - 1]], label: null },
      y: { domain: ["training", "validation"], label: null, tickFormat: d => ROLE_LABEL[d] },
      marks: [
        Plot.rect(rows, { x1: "start", x2: "end", y: "role", fill: d => ROLE_COLOR[d.role], fillOpacity: 0.85, inset: 3,
          title: d => `${ROLE_LABEL[d.role]}${d.season ? " · " + cap(d.season) : ""}\n${fmtDate(d.start.toISOString())} → ${fmtDate(d.end.toISOString())}\n${fmt(d.n, 0)} rows` }),
        Plot.text(rows.filter(d => d.season), { x: d => new Date((d.start.getTime() + d.end.getTime()) / 2), y: "role", text: d => cap(d.season).slice(0, 3), fill: "#fff", fontSize: 10, fontWeight: 600 }),
      ],
    }));

    const tableHost = $("split-table"); clear(tableHost);
    tableHost.appendChild(makeTable(
      ["#", "Role", "Season", "Start", "End", "Rows"],
      rows.map(r => [String(r.i + 1), ROLE_LABEL[r.role], cap(r.season) || "—", fmtDate(r.start.toISOString()), fmtDate(r.end.toISOString()), fmt(r.n, 0)]),
      [5]
    ));
  }

  function renderCovariates() {
    const host = $("covariate-panels"); clear(host);
    const covs = (DATA.meta && DATA.meta.covariates) || [];
    const note = $("covariates-note");
    if (!covs.length) { note.textContent = "No numeric covariate columns in this file."; return; }
    const unmapped = covs.filter(c => c.role === "unmapped").map(c => c.name);
    note.textContent = unmapped.length
      ? `Columns tagged "unmapped" are numeric but not recognised as covariates by the column mapping — a model trained without an explicit column_mapping will ignore them: ${unmapped.join(", ")}.`
      : "All numeric columns are mapped as covariates.";
    covs.forEach(c => {
      const panel = document.createElement("div"); panel.className = "cov-panel";
      const title = document.createElement("div"); title.className = "cov-title";
      const nm = document.createElement("span"); nm.textContent = c.name; title.appendChild(nm);
      const tag = document.createElement("span");
      tag.className = "tag" + (c.role === "unmapped" ? " unmapped" : c.role === "future_covariate" ? " future" : "");
      tag.textContent = c.role === "past_covariate" ? "past covariate" : c.role === "future_covariate" ? "future covariate" : "unmapped";
      title.appendChild(tag);
      const r = document.createElement("span"); r.className = "r";
      r.textContent = `r = ${fmt(c.pearson_r, 2)} vs target · mean ${fmt(c.mean)} · ${fmt(c.n_missing, 0)} missing`;
      title.appendChild(r);
      panel.appendChild(title);
      const pts = toPoints(S.covariates[c.name] || [], state.range.start, state.range.end);
      panel.appendChild(seriesChart(pts, { name: c.name, yLabel: c.name, height: 150, labels: false, opacity: 0.8, strokeWidth: 0.9 }));
      host.appendChild(panel);
    });
  }

  function profileChart(rows, opts) {
    return Plot.plot({
      width: 520, height: 240, marginLeft: 60,
      x: opts.x, y: { label: DATA.meta.target, grid: true, nice: true },
      color: opts.color,
      marks: opts.marks(rows),
    });
  }

  function renderProfiles() {
    const P = DATA.profiles || {};
    const h1 = $("chart-profile-season"); clear(h1);
    const bySeason = (P.hour_of_day_by_season || []);
    if (bySeason.length) {
      h1.appendChild(Plot.plot({
        width: 520, height: 240, marginLeft: 60,
        x: { label: "Hour of day", domain: [0, 23], ticks: 12 },
        y: { label: DATA.meta.target, grid: true },
        color: { domain: SEASON_ORDER, range: SEASON_ORDER.map(s => SEASON_COLOR[s]).map(c => d3.color(c).darker(0.8).formatHex()), legend: true },
        marks: [
          Plot.areaY(bySeason, { x: "hour", y1: "p25", y2: "p75", fill: "season", fillOpacity: 0.12 }),
          Plot.line(bySeason, { x: "hour", y: "mean", stroke: "season", strokeWidth: 2, tip: true }),
        ],
      }));
    }
    const h2 = $("chart-profile-weekday"); clear(h2);
    const wk = (P.hour_of_day_weekday || []).map(r => ({ ...r, kind: "Weekday" }))
      .concat((P.hour_of_day_weekend || []).map(r => ({ ...r, kind: "Weekend" })));
    if (wk.length) {
      h2.appendChild(Plot.plot({
        width: 520, height: 240, marginLeft: 60,
        x: { label: "Hour of day", domain: [0, 23], ticks: 12 }, y: { label: DATA.meta.target, grid: true },
        color: { domain: ["Weekday", "Weekend"], range: ["#1f2937", "#0ea5e9"], legend: true },
        marks: [
          Plot.areaY(wk, { x: "hour", y1: "p25", y2: "p75", fill: "kind", fillOpacity: 0.12 }),
          Plot.line(wk, { x: "hour", y: "mean", stroke: "kind", strokeWidth: 2, tip: true }),
        ],
      }));
    }
    const h3 = $("chart-profile-dow"); clear(h3);
    const dow = (P.day_of_week || []).map(r => ({ ...r, label: DOW[r.dow] }));
    if (dow.length) {
      h3.appendChild(Plot.plot({
        width: 520, height: 220, marginLeft: 60,
        x: { label: null, domain: DOW }, y: { label: "Mean " + DATA.meta.target, grid: true },
        marks: [Plot.barY(dow, { x: "label", y: "mean", fill: d => d.dow >= 5 ? "#0ea5e9" : "#2563eb", fillOpacity: 0.85, tip: true }), Plot.ruleY([0])],
      }));
    }
    const h4 = $("chart-profile-month"); clear(h4);
    const mo = (P.month || []).map(r => ({ ...r, label: MONTHS[r.month - 1] }));
    if (mo.length) {
      h4.appendChild(Plot.plot({
        width: 520, height: 220, marginLeft: 60,
        x: { label: null, domain: MONTHS.filter(m => mo.some(r => r.label === m)) }, y: { label: "Mean " + DATA.meta.target, grid: true },
        marks: [Plot.barY(mo, { x: "label", y: "mean", fill: d => SEASON_COLOR[["winter","winter","spring","spring","spring","summer","summer","summer","fall","fall","fall","winter"][d.month - 1]], stroke: "#374151", strokeWidth: 0.4, tip: true }), Plot.ruleY([0])],
      }));
    }
  }

  function renderHeatmap() {
    const host = $("chart-heatmap"); clear(host);
    const cells = DATA.heatmap || [];
    if (!cells.length) return;
    const dates = Array.from(new Set(cells.map(c => c.date))).sort();
    const monthStarts = dates.filter((d, i) => i === 0 || d.slice(0, 7) !== dates[i - 1].slice(0, 7));
    host.appendChild(Plot.plot({
      width: fullWidth(),
      height: 330, marginLeft: 44, marginBottom: 40,
      x: { domain: dates, ticks: monthStarts, tickFormat: d => d.slice(0, 7), label: null, tickRotate: 0 },
      y: { domain: d3.range(23, -1, -1), label: "Hour", ticks: [0, 6, 12, 18, 23] },
      color: { scheme: "YlOrRd", legend: true, label: DATA.meta.target },
      marks: [Plot.cell(cells, { x: "date", y: "hour", fill: "v", inset: 0, title: d => `${d.date} ${String(d.hour).padStart(2, "0")}:00\n${fmt(d.v)}` })],
    }));
  }

  function renderRelations() {
    const host = $("relation-panels"); clear(host);
    const R = DATA.relations || {}; const names = Object.keys(R);
    if (!names.length) { host.textContent = "No covariates."; return; }
    const row = document.createElement("div"); row.className = "chart-row"; host.appendChild(row);
    names.forEach(name => {
      const rel = R[name];
      const wrap = document.createElement("div"); wrap.className = "chart";
      const h = document.createElement("h3"); h.style.marginTop = "0";
      h.textContent = `${DATA.meta.target} vs ${name}` + (rel.pearson_r !== null && rel.pearson_r !== undefined ? ` (r = ${fmt(rel.pearson_r, 2)})` : "");
      wrap.appendChild(h);
      if (!rel.scatter || !rel.scatter.length) { const p = document.createElement("div"); p.className = "note"; p.textContent = "Not enough variation to plot."; wrap.appendChild(p); row.appendChild(wrap); return; }
      wrap.appendChild(Plot.plot({
        width: 520, height: 300, marginLeft: 60,
        x: { label: name, grid: true }, y: { label: DATA.meta.target, grid: true },
        color: { type: "cyclical", scheme: "rainbow", domain: [0, 24], label: "Hour of day", legend: true },
        marks: [
          Plot.dot(rel.scatter, { x: "x", y: "y", fill: "hour", r: 1.8, fillOpacity: 0.45 }),
          Plot.areaY(rel.binned, { x: "x", y1: "p25", y2: "p75", fill: "#1f2937", fillOpacity: 0.12 }),
          Plot.line(rel.binned, { x: "x", y: "mean", stroke: "#1f2937", strokeWidth: 2.2, tip: true }),
        ],
      }));
      row.appendChild(wrap);
    });
  }

  function renderQuality() {
    const gh = $("gaps-table"); clear(gh);
    const gaps = DATA.gaps || [];
    if (!gaps.length) { gh.textContent = "No gaps — every expected timestamp is present."; }
    else {
      gh.appendChild(makeTable(["Start", "End", "Missing steps", "Hours"],
        gaps.slice(0, 25).map(g => [fmtDate(g.start), fmtDate(g.end), fmt(g.n_missing_steps, 0), fmt(g.hours, 1)]), [2, 3]));
      if (gaps.length > 25) { const n = document.createElement("div"); n.className = "note"; n.textContent = `… and ${gaps.length - 25} more.`; gh.appendChild(n); }
    }
    const oh = $("outliers-table"); clear(oh);
    const O = DATA.outliers || {}; const th = O.thresholds || {};
    const info = document.createElement("div"); info.className = "note";
    info.textContent = `${(O.value || []).length} value outlier(s) beyond ±3σ [${fmt(th.value_low)}, ${fmt(th.value_high)}]; ${(O.spikes || []).length} step change(s) beyond ${fmt(th.step_change)}. Same rules as inspect_data.`;
    oh.appendChild(info);
    const top = (O.value || []).slice().sort((a, b) => Math.abs(b.v) - Math.abs(a.v)).slice(0, 10);
    if (top.length) oh.appendChild(makeTable(["Timestamp", "Value"], top.map(o => [fmtDate(o.t), fmt(o.v)]), [1]));
    const sl = $("suggestions-list"); clear(sl);
    const ins = DATA.inspection || {};
    const items = (ins.suggestions || []);
    if (!items.length) { const li = document.createElement("li"); li.textContent = ins.ready_to_train === undefined ? "Run inspect_data for feature suggestions." : "None."; sl.appendChild(li); }
    items.forEach(s => { const li = document.createElement("li"); li.textContent = s; sl.appendChild(li); });
  }

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------
  try {
    renderHeader();
    renderSummary();
    renderRangeBar();
    renderLegend();
    renderOverview();
    renderSplit();
    renderCovariates();
    renderProfiles();
    renderHeatmap();
    renderRelations();
    renderQuality();

    // Full-width charts re-flow when the window is resized (debounced).
    let resizeTimer = null, lastWidth = fullWidth();
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        const w = fullWidth();
        if (w === lastWidth) return;
        lastWidth = w;
        renderOverview(); renderSplit(); renderCovariates(); renderHeatmap();
      }, 150);
    });
  } catch (err) {
    console.error("Data report render failed", err);
    const el = document.createElement("pre"); el.textContent = "Render error: " + (err && err.stack || err);
    document.body.appendChild(el);
  }
})();
