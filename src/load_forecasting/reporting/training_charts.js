/* Self-contained chart renderer for the training report.
   Consumes window.__REPORT_DATA__ (see reporting/training_payload.py).
   Depends on globals: d3, Plot (vendored and inlined).
*/
(function () {
  "use strict";

  const DATA = window.__REPORT_DATA__ || {};
  const Plot = window.Plot;
  if (!Plot) { console.error("Observable Plot did not load"); return; }

  const MODELS = DATA.models || [];
  const PALETTE = ["#2563eb", "#f97316", "#059669", "#7c3aed", "#db2777", "#0891b2", "#b45309", "#4b5563"];
  const colorOf = {}, tagOf = {};
  MODELS.forEach((m, i) => { colorOf[m.model_id] = PALETTE[i % PALETTE.length]; tagOf[m.model_id] = `#${i + 1} ${m.model_type}`; });
  const TRAIN_STYLE = { stroke: "#2563eb" }, VAL_STYLE = { stroke: "#f97316" };

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }
  function clear(el) { while (el && el.firstChild) el.removeChild(el.firstChild); }
  function setText(id, t) { const el = $(id); if (el) el.textContent = t; }
  function fmt(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    const digits = d === undefined ? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 2 : 4) : d;
    return Number(v).toLocaleString(undefined, { maximumFractionDigits: digits });
  }
  function fmtDate(s) { return s ? String(s).slice(0, 16).replace("T", " ") : "—"; }
  function shortId(id) { return id.length > 34 ? id.slice(0, 16) + "…" + id.slice(-14) : id; }
  function fullWidth() { const h = document.querySelector(".container") || document.body; return Math.min(1100, Math.max(600, h.clientWidth - 40)); }
  function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
  function makeKV(k, v) { const kv = el("div", "kv"); kv.appendChild(el("div", "k", k)); kv.appendChild(el("div", "v", v)); return kv; }
  function makeTable(headers, rows, numericCols, bestRow) {
    const t = el("table", "data"); const thead = el("thead"); const tr = el("tr");
    headers.forEach(h => tr.appendChild(el("th", null, h))); thead.appendChild(tr); t.appendChild(thead);
    const tb = el("tbody");
    rows.forEach((r, ri) => {
      const tr2 = el("tr", ri === bestRow ? "best" : null);
      r.forEach((c, i) => { const td = el("td", numericCols && numericCols.includes(i) ? "num" : null); if (c instanceof Node) td.appendChild(c); else td.textContent = c; tr2.appendChild(td); });
      tb.appendChild(tr2);
    });
    t.appendChild(tb); return t;
  }
  function kwargsText(kw) {
    const keys = Object.keys(kw || {}).filter(k => !k.startsWith("_"));
    if (!keys.length) return "defaults";
    const show = v => typeof v === "number" ? (Number.isInteger(v) ? String(v) : Number(v.toPrecision(4)).toString()) : typeof v === "object" ? JSON.stringify(v) : String(v);
    return keys.map(k => `${k}=${show(kw[k])}`).join(", ");
  }

  const state = { log: false };

  // ---------------------------------------------------------------------
  // Header + cards
  // ---------------------------------------------------------------------
  function renderHeader() {
    const m = DATA.meta || {};
    setText("report-subtitle", `${m.n_models} model${m.n_models === 1 ? "" : "s"} · ranked by ${(DATA.comparison || {}).basis || "validation CV-RMSE"} · generated ${fmtDate(DATA.generated_at)} UTC`);
    setText("footer-generated", "STLF-MCP-Server training report");
  }

  function renderCards() {
    const host = $("model-cards"); clear(host);
    const best = (DATA.comparison || {}).best_model_id;
    MODELS.forEach(m => {
      const card = el("div", "model-card");
      const h = el("h3");
      const sw = el("span", "swatch"); sw.style.background = colorOf[m.model_id]; h.appendChild(sw);
      h.appendChild(el("span", "pill", tagOf[m.model_id]));
      h.appendChild(el("span", null, m.model_id));
      if (m.config.tuned) h.appendChild(el("span", "pill tuned", "tuned"));
      if (m.model_id === best && MODELS.length > 1) h.appendChild(el("span", "pill best", "best validation"));
      card.appendChild(h);
      const g = el("div", "kv-grid");
      g.appendChild(makeKV("Data", `${m.data.csv_name || "—"} · ${m.data.target || "—"}`));
      g.appendChild(makeKV("Rows / freq", `${fmt(m.data.n_rows, 0)} · ${m.data.frequency || "—"}`));
      g.appendChild(makeKV("Lookback / horizon", `${m.config.lookback_hours ?? "—"} h / ${m.config.horizon_hours ?? "—"} h`));
      g.appendChild(makeKV("Split", `${m.split.strategy || "—"} · ${Math.round((m.config.validation_split || 0) * 100)}% val`));
      g.appendChild(makeKV("Covariates", `${m.data.past_covariates.length} past${m.data.generated_covariates.length ? ` (+${m.data.generated_covariates.length} generated)` : ""} · ${m.data.future_covariates.length} future`));
      g.appendChild(makeKV("Device", m.config.device || "n/a (non-Torch)"));
      g.appendChild(makeKV("Training time", m.training.time_s !== null ? `${fmt(m.training.time_s, 1)} s` : "—"));
      g.appendChild(makeKV(m.training.x_label ? `${m.training.x_label}s recorded` : "Curve", m.training.n_points !== null ? String(m.training.n_points) : "none"));
      card.appendChild(g);
      const kw = el("div", "note"); kw.style.marginTop = "8px";
      kw.appendChild(el("span", null, "kwargs: ")); kw.appendChild(el("code", "kw", kwargsText(m.config.model_kwargs)));
      card.appendChild(kw);
      const created = el("div", "note", `created ${fmtDate(m.created_at)}`); created.style.marginTop = "4px"; card.appendChild(created);
      host.appendChild(card);
    });
  }

  // ---------------------------------------------------------------------
  // Learning curves
  // ---------------------------------------------------------------------
  function curveChart(m, width, height) {
    const c = m.curve;
    const pts = [];
    c.points.forEach(p => {
      if (p.train !== null && p.train !== undefined) pts.push({ x: p.x, v: p.train, series: "training" });
      if (p.val !== null && p.val !== undefined) pts.push({ x: p.x, v: p.val, series: "validation" });
    });
    const usable = state.log ? pts.filter(p => p.v > 0) : pts;
    const marks = [
      Plot.line(usable, { x: "x", y: "v", stroke: "series", strokeWidth: 1.8, curve: "linear" }),
      Plot.dot(usable.filter(() => usable.length <= 60), { x: "x", y: "v", fill: "series", r: 2.2 }),
    ];
    if (c.best) {
      marks.push(Plot.ruleX([c.best.x], { stroke: "#059669", strokeDasharray: "4,3" }));
      marks.push(Plot.text([c.best], { x: "x", y: "val", text: d => `best ${c.x_label} ${d.x}`, dy: -10, dx: 4, textAnchor: "start", fill: "#059669", fontSize: 11 }));
    }
    marks.push(Plot.tip(usable, Plot.pointerX({ x: "x", y: "v", title: d => `${c.x_label} ${d.x}\n${d.series}: ${fmt(d.v, 5)}` })));
    return Plot.plot({
      width, height, marginLeft: 64,
      x: { label: c.x_label, nice: true },
      y: { label: `${c.metric}${state.log ? " (log)" : ""}`, grid: true, type: state.log ? "log" : "linear" },
      color: { domain: ["training", "validation"].filter(s => (s === "training" ? c.has_train : c.has_val)),
               range: ["training", "validation"].filter(s => (s === "training" ? c.has_train : c.has_val)).map(s => s === "training" ? TRAIN_STYLE.stroke : VAL_STYLE.stroke), legend: true },
      marks,
    });
  }

  function renderCurveToolbar() {
    const bar = $("curve-toolbar"); clear(bar);
    bar.appendChild(el("span", null, "Scale:"));
    [["linear", false], ["log", true]].forEach(([label, log]) => {
      const b = el("button", state.log === log ? "active" : null, label);
      b.addEventListener("click", () => { state.log = log; renderCurveToolbar(); renderCurves(); });
      bar.appendChild(b);
    });
    const n = MODELS.filter(m => m.curve).length;
    bar.appendChild(el("span", null, `· ${n} of ${MODELS.length} model${MODELS.length === 1 ? "" : "s"} recorded a curve`));
  }

  function renderCurves() {
    const host = $("curve-panels"); clear(host);
    const overlay = $("curve-overlay"); clear(overlay);
    const withCurves = MODELS.filter(m => m.curve);
    if (!withCurves.length) { host.appendChild(el("div", "note", "No learning curves: none of these models records per-epoch history (LinearRegression, ARIMA and the naive baselines fit in one shot; TimesFM is excluded by design).")); return; }

    if (withCurves.filter(m => m.curve.has_val).length > 1) {
      // Overlay of validation curves relative to each model's first value, so
      // models with different loss functions / scales can still be compared on shape.
      const rows = [];
      withCurves.forEach(m => {
        const vals = m.curve.points.filter(p => p.val !== null);
        if (!vals.length) return;
        const base = vals[0].val || 1;
        vals.forEach(p => rows.push({ x: p.x, rel: p.val / base, model: m.model_id, label: `${m.model_type} · ${shortId(m.model_id)}` }));
      });
      const wrap = el("div", "curve-panel");
      const t = el("div", "panel-title"); t.appendChild(el("span", null, "Validation loss relative to its first epoch — all models"));
      t.appendChild(el("span", "sub", "shape comparison; absolute values differ by loss function")); wrap.appendChild(t);
      wrap.appendChild(Plot.plot({
        width: fullWidth(), height: 260, marginLeft: 64,
        x: { label: "epoch / iteration" }, y: { label: "val / val[first]", grid: true, type: state.log ? "log" : "linear" },
        color: { domain: withCurves.map(m => m.model_id), range: withCurves.map(m => colorOf[m.model_id]), legend: true, tickFormat: id => tagOf[id] },
        marks: [Plot.ruleY([1], { stroke: "#9ca3af", strokeDasharray: "2,3" }), Plot.line(rows, { x: "x", y: "rel", stroke: "model", strokeWidth: 1.6, tip: true })],
      }));
      overlay.appendChild(wrap);
    }

    withCurves.forEach(m => {
      const c = m.curve;
      const wrap = el("div", "curve-panel");
      const t = el("div", "panel-title");
      const sw = el("span", "swatch"); sw.style.background = colorOf[m.model_id]; t.appendChild(sw);
      t.appendChild(el("span", null, `${tagOf[m.model_id]} · ${m.model_id}`));
      const bits = [];
      if (c.best) bits.push(`best ${c.x_label} ${c.best.x} (val ${fmt(c.best.val, 5)})`);
      bits.push(`final train ${fmt(c.final.train, 5)} · val ${fmt(c.final.val, 5)}`);
      if (c.gap_ratio !== null) bits.push(`val/train ${fmt(c.gap_ratio, 2)}×`);
      t.appendChild(el("span", "sub", bits.join(" · ")));
      wrap.appendChild(t);
      wrap.appendChild(curveChart(m, fullWidth(), 280));
      c.diagnostics.forEach(d => { const f = el("span", "flag", d); const box = el("div"); box.style.marginTop = "6px"; box.appendChild(f); wrap.appendChild(box); });
      host.appendChild(wrap);
    });
  }

  // ---------------------------------------------------------------------
  // Metrics
  // ---------------------------------------------------------------------
  function renderMetrics() {
    const rows = [];
    MODELS.forEach(m => {
      ["training", "validation"].forEach(split => {
        const mt = m.metrics[split];
        rows.push({ model: m.model_id, label: tagOf[m.model_id], split, cv_rmse: mt.cv_rmse, mape: mt.mape, r2: mt.r_squared });
      });
    });
    const labels = MODELS.map(m => tagOf[m.model_id]);
    const h1 = $("chart-cvrmse"); clear(h1);
    h1.appendChild(Plot.plot({
      width: 520, height: 280, marginLeft: 56, marginBottom: 70,
      fx: { domain: labels, label: null, padding: 0.2 },
      x: { domain: ["training", "validation"], axis: null, paddingInner: 0.15 },
      y: { label: "CV-RMSE (%)", grid: true },
      color: { domain: ["training", "validation"], range: [TRAIN_STYLE.stroke, VAL_STYLE.stroke], legend: true },
      marks: [Plot.barY(rows.filter(r => r.cv_rmse !== null), { fx: "label", x: "split", y: "cv_rmse", fill: "split", tip: true, title: d => `${d.split}\nCV-RMSE ${fmt(d.cv_rmse, 2)}%` }), Plot.ruleY([0])],
    }));
    const h2 = $("chart-mape-r2"); clear(h2);
    const val = rows.filter(r => r.split === "validation");
    h2.appendChild(Plot.plot({
      width: 520, height: 280, marginLeft: 56, marginBottom: 70,
      x: { domain: labels, label: null }, y: { label: "validation MAPE (%)", grid: true },
      marks: [
        Plot.barY(val.filter(r => r.mape !== null), { x: "label", y: "mape", fill: "#f97316", fillOpacity: 0.8, tip: true, title: d => `MAPE ${fmt(d.mape, 2)}%\nR² ${fmt(d.r2, 3)}` }),
        Plot.ruleY([0]),
      ],
    }));

    const th = $("metrics-table"); clear(th);
    const best = (DATA.comparison || {}).rows || [];
    const bestIdx = MODELS.findIndex(m => m.model_id === (DATA.comparison || {}).best_model_id);
    th.appendChild(makeTable(
      ["#", "Model", "Split", "RMSE", "MAE", "MAPE %", "CV-RMSE %", "R²"],
      MODELS.flatMap((m, i) => ["training", "validation"].map(s => {
        const mt = m.metrics[s];
        return [s === "training" ? tagOf[m.model_id] : "", s === "training" ? m.model_id : "", s, fmt(mt.rmse, 3), fmt(mt.mae, 3), fmt(mt.mape, 2), fmt(mt.cv_rmse, 2), fmt(mt.r_squared, 3)];
      })),
      [3, 4, 5, 6, 7],
      bestIdx >= 0 ? bestIdx * 2 + 1 : -1
    ));
  }

  // ---------------------------------------------------------------------
  // Splits
  // ---------------------------------------------------------------------
  const ROLE_COLOR = { training: "#2563eb", validation: "#f97316" };
  function renderSplits() {
    const host = $("split-panels"); clear(host);
    MODELS.forEach(m => {
      const wrap = el("div", "curve-panel");
      const t = el("div", "panel-title");
      const sw = el("span", "swatch"); sw.style.background = colorOf[m.model_id]; t.appendChild(sw);
      t.appendChild(el("span", null, `${tagOf[m.model_id]} · ${shortId(m.model_id)}`));
      const segs = m.split.segments || [];
      if (!segs.length) {
        t.appendChild(el("span", "sub", `${m.split.strategy || "unknown"} split · ${fmt(m.split.train_steps, 0)} train / ${fmt(m.split.validation_steps, 0)} val steps · no date segments recorded (trained before segment provenance existed)`));
        wrap.appendChild(t); host.appendChild(wrap); return;
      }
      const s = m.split.summary || {};
      t.appendChild(el("span", "sub", `${m.split.strategy} · ${fmt(s.training_rows, 0)} train / ${fmt(s.validation_rows, 0)} val rows · ${segs.length} segments`));
      wrap.appendChild(t);
      const rows = segs.map(x => ({ role: x.role, season: x.season || "", start: new Date(x.start), end: new Date(x.end), n: x.n_rows }));
      wrap.appendChild(Plot.plot({
        width: fullWidth(), height: 100, marginLeft: 80, marginRight: 16,
        x: { type: "utc", label: null }, y: { domain: ["training", "validation"], label: null },
        marks: [
          Plot.rect(rows, { x1: "start", x2: "end", y: "role", fill: d => ROLE_COLOR[d.role], fillOpacity: 0.85, inset: 3, title: d => `${d.role}${d.season ? " · " + d.season : ""}\n${fmtDate(d.start.toISOString())} → ${fmtDate(d.end.toISOString())}\n${fmt(d.n, 0)} rows` }),
          Plot.text(rows.filter(d => d.season), { x: d => new Date((d.start.getTime() + d.end.getTime()) / 2), y: "role", text: d => d.season.slice(0, 3), fill: "#fff", fontSize: 10, fontWeight: 600 }),
        ],
      }));
      host.appendChild(wrap);
    });
  }

  // ---------------------------------------------------------------------
  // Tuning
  // ---------------------------------------------------------------------
  function renderTuning() {
    const host = $("tuning-panels"); clear(host);
    const tuned = MODELS.filter(m => m.tuning);
    if (!tuned.length) { host.appendChild(el("div", "note", "None of these models came from tune_model.")); return; }
    tuned.forEach(m => {
      const T = m.tuning;
      const wrap = el("div", "curve-panel");
      const t = el("div", "panel-title");
      const sw = el("span", "swatch"); sw.style.background = colorOf[m.model_id]; t.appendChild(sw);
      t.appendChild(el("span", null, `${tagOf[m.model_id]} · ${m.model_id}`));
      t.appendChild(el("span", "sub", `${T.n_completed} trial${T.n_completed === 1 ? "" : "s"}${T.n_failed ? ` (${T.n_failed} failed)` : ""} · best trial ${T.best_trial} · CV-RMSE ${fmt(T.best_cv_rmse, 3)}%`));
      wrap.appendChild(t);

      // Convergence: cv_rmse per trial + best-so-far
      const ok = T.trials.filter(x => x.cv_rmse !== null);
      wrap.appendChild(Plot.plot({
        width: fullWidth(), height: 220, marginLeft: 64,
        x: { label: "trial", nice: true }, y: { label: "validation CV-RMSE (%)", grid: true },
        marks: [
          Plot.line(T.best_so_far.filter(b => b.best !== null), { x: "trial", y: "best", stroke: "#059669", strokeWidth: 1.6, curve: "step-after" }),
          Plot.dot(ok, { x: "trial", y: "cv_rmse", r: 4, fill: d => d.trial === T.best_trial ? "#059669" : "#7c3aed", fillOpacity: 0.85, tip: true, title: d => `trial ${d.trial}\nCV-RMSE ${fmt(d.cv_rmse, 3)}%\n${kwargsText(d.params)}\n${fmt(d.duration_s, 1)} s` }),
        ],
      }));

      // Parameter scatter small multiples
      if (T.numeric_params.length) {
        const row = el("div", "chart-row"); row.style.marginTop = "10px";
        T.numeric_params.forEach(name => {
          const pts = ok.filter(x => typeof x.params[name] === "number").map(x => ({ v: x.params[name], cv: x.cv_rmse, best: x.trial === T.best_trial, trial: x.trial }));
          const box = el("div", "chart"); box.appendChild(el("h3", null, name)).style.marginTop = "0";
          box.appendChild(Plot.plot({
            width: 340, height: 200, marginLeft: 56,
            x: { label: name, nice: true }, y: { label: "CV-RMSE (%)", grid: true },
            marks: [Plot.dot(pts, { x: "v", y: "cv", r: d => d.best ? 5 : 3.5, fill: d => d.best ? "#059669" : "#7c3aed", fillOpacity: 0.8, tip: true, title: d => `trial ${d.trial}: ${name}=${d.v} → ${fmt(d.cv, 3)}%` })],
          }));
          row.appendChild(box);
        });
        wrap.appendChild(row);
      }

      // Per-trial train-loss curves (Lightning models)
      if (T.has_trial_curves) {
        const rows = [];
        T.trials.forEach(x => { (x.train_loss || []).forEach((v, i) => rows.push({ trial: x.trial, epoch: i, v, best: x.trial === T.best_trial })); });
        const box = el("div"); box.style.marginTop = "10px";
        box.appendChild(el("h3", null, "Train loss per epoch, per trial")).style.marginBottom = "4px";
        box.appendChild(Plot.plot({
          width: fullWidth(), height: 220, marginLeft: 64,
          x: { label: "epoch" }, y: { label: "train loss", grid: true, type: state.log ? "log" : "linear" },
          marks: [Plot.line(rows, { x: "epoch", y: "v", z: "trial", stroke: d => d.best ? "#059669" : "#9ca3af", strokeWidth: d => d.best ? 2.2 : 1, strokeOpacity: d => d.best ? 1 : 0.6, tip: true, title: d => `trial ${d.trial} · epoch ${d.epoch}: ${fmt(d.v, 5)}` })],
        }));
        wrap.appendChild(box);
      }

      // Top trials table
      const top = ok.slice().sort((a, b) => a.cv_rmse - b.cv_rmse).slice(0, 8);
      const tbl = el("div"); tbl.style.marginTop = "10px";
      tbl.appendChild(makeTable(["Trial", "CV-RMSE %", "MAPE %", "R²", "Duration s", "Parameters"],
        top.map(x => [String(x.trial), fmt(x.cv_rmse, 3), fmt(x.metrics.mape, 2), fmt(x.metrics.r_squared, 3), fmt(x.duration_s, 1), kwargsText(x.params)]),
        [1, 2, 3, 4], top.findIndex(x => x.trial === T.best_trial)));
      wrap.appendChild(tbl);
      const ss = el("div", "note"); ss.style.marginTop = "6px"; ss.textContent = "search space: " + JSON.stringify(T.search_space);
      wrap.appendChild(ss);
      host.appendChild(wrap);
    });
  }

  // ---------------------------------------------------------------------
  // Compute + findings
  // ---------------------------------------------------------------------
  function renderCompute() {
    const rows = MODELS.map(m => ({ label: tagOf[m.model_id], time: m.training.time_s, energy: m.training.energy_kwh, id: m.model_id }));
    const h1 = $("chart-time"); clear(h1);
    h1.appendChild(Plot.plot({ width: 520, height: 220, marginLeft: 60, marginBottom: 60, x: { label: null, domain: MODELS.map(m => tagOf[m.model_id]) }, y: { label: "seconds", grid: true },
      marks: [Plot.barY(rows.filter(r => r.time !== null), { x: "label", y: "time", fill: d => colorOf[d.id], tip: true, sort: null }), Plot.ruleY([0])] }));
    const h2 = $("chart-energy"); clear(h2);
    h2.appendChild(Plot.plot({ width: 520, height: 220, marginLeft: 70, marginBottom: 60, x: { label: null, domain: MODELS.map(m => tagOf[m.model_id]) }, y: { label: "kWh (estimated)", grid: true },
      marks: [Plot.barY(rows.filter(r => r.energy !== null), { x: "label", y: "energy", fill: d => colorOf[d.id], tip: true, sort: null }), Plot.ruleY([0])] }));
  }

  function renderFlags() {
    const host = $("flags-list"); clear(host);
    let any = false;
    MODELS.forEach(m => {
      if (!m.flags.length) return;
      any = true;
      const box = el("div"); box.style.marginBottom = "8px";
      box.appendChild(el("strong", null, `${tagOf[m.model_id]} · ${shortId(m.model_id)}: `));
      m.flags.forEach(f => box.appendChild(el("span", "flag" + (f.startsWith("OVERFIT") ? " bad" : f.startsWith("NO_CURVE") ? " info" : ""), f)));
      host.appendChild(box);
    });
    if (!any) host.appendChild(el("div", "note", "No findings — every model has a curve and no overfitting signal."));
  }

  try {
    renderHeader(); renderCards(); renderCurveToolbar(); renderCurves(); renderMetrics(); renderSplits(); renderTuning(); renderCompute(); renderFlags();
    let timer = null, last = fullWidth();
    window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => { const w = fullWidth(); if (w !== last) { last = w; renderCurves(); renderSplits(); renderTuning(); } }, 150); });
  } catch (err) {
    console.error("Training report render failed", err);
    const pre = document.createElement("pre"); pre.textContent = "Render error: " + (err && err.stack || err); document.body.appendChild(pre);
  }
})();
