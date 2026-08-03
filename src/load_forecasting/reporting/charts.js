/* Self-contained chart renderer for the evaluation report.
   Consumes window.__REPORT_DATA__ and renders every section.

   Depends on globals provided by the embedded scripts:
     - d3        (from d3.min.js)
     - Plot      (from observable-plot.min.js, UMD bundle exposes window.Plot)
*/
(function () {
  "use strict";

  const DATA = window.__REPORT_DATA__ || {};
  const Plot = window.Plot;
  if (!Plot) {
    console.error("Observable Plot did not load");
    return;
  }

  // ---------------------------------------------------------------------
  // Small helpers
  // ---------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  function fmtNumber(v, digits) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    digits = (digits === undefined) ? 3 : digits;
    return Number(v).toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: 0,
    });
  }

  function fmtDate(s) {
    if (!s) return "—";
    const d = new Date(s);
    if (isNaN(d)) return s;
    return d.toLocaleString();
  }

  function parsePoints(arr, valueKey) {
    if (!Array.isArray(arr)) return [];
    const out = [];
    for (const row of arr) {
      const t = row.t ? new Date(row.t) : null;
      const v = row[valueKey];
      if (t && !isNaN(t) && v !== null && v !== undefined && isFinite(v)) {
        out.push({ t: t, v: +v });
      }
    }
    return out;
  }

  function setText(id, text) {
    const el = $(id);
    if (el) el.textContent = (text === null || text === undefined) ? "—" : text;
  }

  function append(parent, node) {
    const p = (typeof parent === "string") ? $(parent) : parent;
    if (p && node) p.appendChild(node);
  }

  function makeKV(label, value) {
    const wrap = document.createElement("div");
    wrap.className = "kv";
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = label;
    const v = document.createElement("span");
    v.className = "v";
    v.textContent = (value === null || value === undefined || value === "") ? "—" : value;
    wrap.appendChild(k); wrap.appendChild(v);
    return wrap;
  }

  function makeMetricCard(label, value, compare) {
    const card = document.createElement("div");
    card.className = "metric-card";
    const lab = document.createElement("div");
    lab.className = "label"; lab.textContent = label;
    const val = document.createElement("div");
    val.className = "value"; val.textContent = value;
    card.appendChild(lab); card.appendChild(val);
    if (compare) {
      const c = document.createElement("div");
      c.className = "compare"; c.textContent = compare;
      card.appendChild(c);
    }
    return card;
  }

  // ---------------------------------------------------------------------
  // Header + metadata
  // ---------------------------------------------------------------------
  function renderHeader() {
    const meta = DATA.meta || {};
    setText("report-title", meta.title || "Evaluation report");
    setText("report-subtitle",
      "Generated " + fmtDate(DATA.generated_at) +
      " · model " + (meta.model_id || "?"));

    const grid = $("meta-grid");
    if (!grid) return;
    grid.innerHTML = "";
    grid.appendChild(makeKV("Model ID", meta.model_id));
    grid.appendChild(makeKV("Model type", meta.model_type));
    grid.appendChild(makeKV("Building", meta.building_name || "—"));
    grid.appendChild(makeKV("Created", fmtDate(meta.created_at)));
    grid.appendChild(makeKV("Lookback (hrs)", meta.lookback_hours));
    grid.appendChild(makeKV("Horizon (hrs)", meta.horizon_hours));
    grid.appendChild(makeKV("Frequency", meta.frequency));
    grid.appendChild(makeKV("Validation split", meta.validation_split));
    const trange = meta.train_data_range || {};
    grid.appendChild(makeKV("Train start", fmtDate(trange.start)));
    grid.appendChild(makeKV("Train end", fmtDate(trange.end)));
    grid.appendChild(makeKV("Train samples", fmtNumber(trange.samples, 0)));
    const testr = meta.test_data_range || {};
    grid.appendChild(makeKV("Test start", fmtDate(testr.start_date)));
    grid.appendChild(makeKV("Test end", fmtDate(testr.end_date)));
    grid.appendChild(makeKV("Test samples", fmtNumber(testr.test_samples, 0)));
  }

  // ---------------------------------------------------------------------
  // Metric cards
  // ---------------------------------------------------------------------
  function renderMetrics() {
    const m = DATA.metrics || {};
    const t = m.test || {};
    const v = m.validation || {};
    const cmp = m.comparison || {};

    const cards = $("metric-cards");
    if (!cards) return;
    cards.innerHTML = "";

    const order = [
      ["RMSE",     "rmse",     ""],
      ["MAE",      "mae",      ""],
      ["MAPE",     "mape",     "%"],
      ["CV-RMSE",  "cv_rmse",  "%"],
      ["R²",       "r_squared", ""],
    ];
    for (const [label, key, unit] of order) {
      const val = t[key];
      const valStr = (val === null || val === undefined) ? "—"
        : fmtNumber(val, 3) + (unit || "");
      let cmpStr = null;
      if (v[key] !== null && v[key] !== undefined) {
        cmpStr = "val: " + fmtNumber(v[key], 3) + (unit || "");
      }
      cards.appendChild(makeMetricCard(label, valStr, cmpStr));
    }

    // Peak headline cards — surfaced alongside the standard metrics when
    // peak_dates were supplied (full per-day detail lives in #peak-metrics).
    const peakOrder = [
      ["Peak MAPE",         "peak_mape",                "%"],
      ["Peak timing error", "peak_timing_error_hours",  " h"],
    ];
    for (const [label, key, unit] of peakOrder) {
      const val = t[key];
      if (val === null || val === undefined) continue;
      cards.appendChild(makeMetricCard(label, fmtNumber(val, 2) + (unit || ""), null));
    }

    // Comparison badge
    const badge = $("compare-badge");
    if (badge) {
      const status = cmp.performance_status;
      if (status) {
        badge.className = "badge " + status;
        badge.textContent = status;
        const diff = cmp.cv_rmse_diff;
        if (diff !== null && diff !== undefined) {
          badge.textContent += "  (Δcv_rmse " +
            (diff >= 0 ? "+" : "") + fmtNumber(diff, 2) + ")";
        }
      } else {
        badge.style.display = "none";
      }
    }
  }

  // ---------------------------------------------------------------------
  // Charts
  // ---------------------------------------------------------------------
  function renderActualVsPredicted() {
    const preds = DATA.series && DATA.series.predictions;
    if (!preds || !preds.length) return;
    const actual = parsePoints(preds, "actual").map(p => ({ ...p, series: "Actual" }));
    const predicted = parsePoints(preds, "predicted").map(p => ({ ...p, series: "Predicted" }));
    const all = actual.concat(predicted);

    const chart = Plot.plot({
      width: 1100,
      height: 320,
      marginLeft: 56,
      marginBottom: 36,
      x: { type: "utc", label: "Time" },
      y: { label: "Load", grid: true },
      color: {
        legend: true,
        domain: ["Actual", "Predicted"],
        range: ["#1f2937", "#2563eb"],
      },
      marks: [
        Plot.ruleY([0], { stroke: "#e5e7eb" }),
        Plot.line(all, { x: "t", y: "v", stroke: "series", strokeWidth: 1.2 }),
        Plot.tip(all, Plot.pointerX({ x: "t", y: "v", stroke: "series", channels: { series: "series" } })),
      ],
    });
    append("chart-actual-vs-pred", chart);
  }

  function renderResidualsOverTime() {
    const preds = DATA.series && DATA.series.predictions;
    if (!preds || !preds.length) return;
    const pts = parsePoints(preds, "residual");
    const chart = Plot.plot({
      width: 1100,
      height: 220,
      marginLeft: 56,
      marginBottom: 36,
      x: { type: "utc", label: "Time" },
      y: { label: "Residual (actual − predicted)", grid: true },
      marks: [
        Plot.ruleY([0], { stroke: "#b91c1c", strokeOpacity: 0.6 }),
        Plot.line(pts, { x: "t", y: "v", stroke: "#475569", strokeWidth: 0.8 }),
        Plot.tip(pts, Plot.pointerX({ x: "t", y: "v" })),
      ],
    });
    append("chart-residuals", chart);
  }

  function renderResidualHistogram() {
    const preds = DATA.series && DATA.series.predictions;
    if (!preds || !preds.length) return;
    const residuals = preds
      .map(r => r.residual)
      .filter(v => v !== null && v !== undefined && isFinite(v));
    if (!residuals.length) return;

    const mean = d3.mean(residuals);
    const std = d3.deviation(residuals) || 0;

    const chart = Plot.plot({
      width: 540,
      height: 260,
      marginLeft: 56,
      marginBottom: 36,
      x: { label: "Residual", grid: true },
      y: { label: "Count" },
      marks: [
        Plot.rectY(residuals, Plot.binX({ y: "count" }, { x: d => d, fill: "#93c5fd", stroke: "#1d4ed8", strokeWidth: 0.5 })),
        Plot.ruleX([mean], { stroke: "#b91c1c", strokeWidth: 1.5 }),
      ],
    });
    append("chart-residual-hist", chart);

    setText("residual-hist-stats",
      "mean = " + fmtNumber(mean, 4) + ",  std = " + fmtNumber(std, 4) +
      ",  n = " + residuals.length);
  }

  function renderParity() {
    const preds = DATA.series && DATA.series.predictions;
    if (!preds || !preds.length) return;
    const pts = preds
      .filter(r => isFinite(r.actual) && isFinite(r.predicted))
      .map(r => ({ a: +r.actual, p: +r.predicted }));
    if (!pts.length) return;

    const lo = Math.min(d3.min(pts, d => d.a), d3.min(pts, d => d.p));
    const hi = Math.max(d3.max(pts, d => d.a), d3.max(pts, d => d.p));
    const diag = [{ x: lo, y: lo }, { x: hi, y: hi }];

    const chart = Plot.plot({
      width: 540,
      height: 540,
      marginLeft: 56,
      marginBottom: 40,
      x: { label: "Actual", domain: [lo, hi], grid: true },
      y: { label: "Predicted", domain: [lo, hi], grid: true },
      marks: [
        Plot.line(diag, { x: "x", y: "y", stroke: "#b91c1c", strokeDasharray: "4,3", strokeOpacity: 0.7 }),
        Plot.dot(pts, { x: "a", y: "p", r: 1.6, fill: "#2563eb", fillOpacity: 0.45 }),
      ],
    });
    append("chart-parity", chart);
  }

  function renderHourlyMAE() {
    const agg = (DATA.aggregations || {}).hour_of_day_mae;
    if (!agg || !agg.length) return;
    const chart = Plot.plot({
      width: 540,
      height: 220,
      marginLeft: 56,
      marginBottom: 36,
      x: { label: "Hour of day", tickFormat: d => d, domain: d3.range(24) },
      y: { label: "Mean abs. error", grid: true },
      marks: [
        Plot.barY(agg, { x: "hour", y: "mae", fill: "#2563eb", fillOpacity: 0.8 }),
        Plot.ruleY([0]),
      ],
    });
    append("chart-hourly-mae", chart);
  }

  function renderDOWMAE() {
    const agg = (DATA.aggregations || {}).day_of_week_mae;
    if (!agg || !agg.length) return;
    const chart = Plot.plot({
      width: 540,
      height: 220,
      marginLeft: 56,
      marginBottom: 36,
      x: { label: "Day of week", domain: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"] },
      y: { label: "Mean abs. error", grid: true },
      marks: [
        Plot.barY(agg, { x: "label", y: "mae", fill: "#047857", fillOpacity: 0.8 }),
        Plot.ruleY([0]),
      ],
    });
    append("chart-dow-mae", chart);
  }

  function renderInputPreview() {
    const target = DATA.series && DATA.series.input_target;
    const covs = (DATA.series && DATA.series.covariates) || {};
    const summary = DATA.input_summary || {};

    if (target && target.length) {
      const pts = parsePoints(target, "value");
      const chart = Plot.plot({
        width: 1100,
        height: 220,
        marginLeft: 56,
        marginBottom: 36,
        x: { type: "utc", label: "Time" },
        y: { label: "Target", grid: true },
        marks: [
          Plot.line(pts, { x: "t", y: "v", stroke: "#1f2937", strokeWidth: 1 }),
          Plot.tip(pts, Plot.pointerX({ x: "t", y: "v" })),
        ],
      });
      append("chart-input-target", chart);
    } else {
      setText("input-target-note", "Input target series not available.");
    }

    const covWrap = $("chart-input-covariates");
    if (covWrap) {
      covWrap.innerHTML = "";
      for (const [name, points] of Object.entries(covs)) {
        if (!points || !points.length) continue;
        const pts = parsePoints(points, "value");
        const h = document.createElement("h3");
        h.textContent = name;
        covWrap.appendChild(h);
        const chart = Plot.plot({
          width: 1100,
          height: 160,
          marginLeft: 56,
          marginBottom: 28,
          x: { type: "utc", label: null },
          y: { label: name, grid: true },
          marks: [
            Plot.line(pts, { x: "t", y: "v", stroke: "#0ea5e9", strokeWidth: 0.9 }),
          ],
        });
        covWrap.appendChild(chart);
      }
    }

    // Stats table
    const tbody = $("input-stats-body");
    if (tbody) {
      tbody.innerHTML = "";
      const rows = [];
      if (summary.target) rows.push(Object.assign({ role: "target" }, summary.target));
      const covSummary = summary.covariates || {};
      for (const [name, st] of Object.entries(covSummary)) {
        rows.push(Object.assign({ role: "covariate" }, st));
      }
      for (const r of rows) {
        const tr = document.createElement("tr");
        const cells = [r.role, r.column, r.n, r.mean, r.std, r.min, r.max, r.p5, r.p95];
        for (let i = 0; i < cells.length; i++) {
          const td = document.createElement("td");
          const v = cells[i];
          td.textContent = (i <= 2) ? (v === null || v === undefined ? "—" : v)
                                    : fmtNumber(v, 3);
          tr.appendChild(td);
        }
        tbody.appendChild(tr);
      }
      if (!rows.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 9; td.className = "note";
        td.textContent = "No input statistics available.";
        tr.appendChild(td); tbody.appendChild(tr);
      }
    }
  }

  function renderPeakMetrics() {
    const section = $("peak-metrics");
    const pm = DATA.peak_metrics || {};
    const perDay = Array.isArray(pm.per_day_results) ? pm.per_day_results : [];
    // Hide the whole section if peak metrics were not requested / are empty.
    const hasSummary = pm.peak_mape !== undefined && pm.peak_mape !== null;
    if (!hasSummary && !perDay.length) {
      if (section) section.style.display = "none";
      return;
    }
    if (section) section.style.display = "";

    // Summary cards
    const cards = $("peak-metric-cards");
    if (cards) {
      cards.innerHTML = "";
      cards.appendChild(makeMetricCard(
        "Peak MAPE",
        (pm.peak_mape === null || pm.peak_mape === undefined)
          ? "—" : fmtNumber(pm.peak_mape, 2) + "%",
        null
      ));
      cards.appendChild(makeMetricCard(
        "Peak Timing Error",
        (pm.peak_timing_error_hours === null || pm.peak_timing_error_hours === undefined)
          ? "—" : fmtNumber(pm.peak_timing_error_hours, 2) + " h",
        null
      ));
    }

    // Counts kv-grid
    const counts = $("peak-metric-counts");
    if (counts) {
      counts.innerHTML = "";
      counts.appendChild(makeKV("Peak days evaluated", fmtNumber(pm.n_peak_days_evaluated, 0)));
      counts.appendChild(makeKV("Peak days skipped", fmtNumber(pm.n_peak_days_skipped, 0)));
    }

    // Per-day table
    const tbody = $("peak-metrics-body");
    if (tbody) {
      tbody.innerHTML = "";
      if (!perDay.length) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = 7; td.className = "note";
        td.textContent = "No per-day peak results available.";
        tr.appendChild(td); tbody.appendChild(tr);
      } else {
        for (const r of perDay) {
          const tr = document.createElement("tr");
          const cells = [
            r.date,
            fmtNumber(r.actual_peak_value, 3),
            fmtNumber(r.predicted_peak_value, 3),
            (r.actual_peak_hour === null || r.actual_peak_hour === undefined)
              ? "—" : String(r.actual_peak_hour).padStart(2, "0") + ":00",
            (r.predicted_peak_hour === null || r.predicted_peak_hour === undefined)
              ? "—" : String(r.predicted_peak_hour).padStart(2, "0") + ":00",
            fmtNumber(r.peak_magnitude_error_pct, 2),
            fmtNumber(r.peak_timing_error_hours, 2),
          ];
          for (const v of cells) {
            const td = document.createElement("td");
            td.textContent = (v === null || v === undefined) ? "—" : v;
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        }
      }
    }

    // Grouped bar chart: actual vs predicted peak value per date
    const chartWrap = $("chart-peak-values");
    if (chartWrap && perDay.length) {
      chartWrap.innerHTML = "";
      const rows = [];
      for (const r of perDay) {
        if (r.actual_peak_value !== null && r.actual_peak_value !== undefined) {
          rows.push({ date: r.date, series: "Actual", value: +r.actual_peak_value });
        }
        if (r.predicted_peak_value !== null && r.predicted_peak_value !== undefined) {
          rows.push({ date: r.date, series: "Predicted", value: +r.predicted_peak_value });
        }
      }
      if (rows.length) {
        const chart = Plot.plot({
          width: 1100,
          height: 260,
          marginLeft: 56,
          marginBottom: 40,
          x: { label: "Date" },
          y: { label: "Peak load", grid: true },
          color: {
            legend: true,
            domain: ["Actual", "Predicted"],
            range: ["#1f2937", "#2563eb"],
          },
          marks: [
            Plot.barY(rows, {
              x: "date",
              y: "value",
              fill: "series",
              fx: "date",
            }),
            Plot.ruleY([0]),
          ],
        });
        chartWrap.appendChild(chart);
      }
    }
  }

  function renderResidualAnalysis() {
    const ra = DATA.residual_analysis || {};
    const grid = $("residual-analysis-grid");
    if (!grid) return;
    grid.innerHTML = "";
    grid.appendChild(makeKV("Mean residual", fmtNumber(ra.mean_residual, 4)));
    grid.appendChild(makeKV("Std residual", fmtNumber(ra.std_residual, 4)));
    grid.appendChild(makeKV("Lag-1 autocorrelation", fmtNumber(ra.autocorrelation_lag1, 4)));
  }

  // ---------------------------------------------------------------------
  // Training curve (train/val loss per epoch) — Torch models only.
  // Hides the section gracefully when no history was captured.
  // ---------------------------------------------------------------------
  function renderTrainingCurve() {
    const section = $("training-curve");
    const hist = (DATA.meta || {}).training_history || {};
    const points = Array.isArray(hist.points) ? hist.points : [];
    if (!points.length) {
      if (section) section.style.display = "none";
      return;
    }
    if (section) section.style.display = "";

    // Long-form rows: one entry per (epoch, series) with a finite loss.
    const rows = [];
    for (const p of points) {
      if (p.train_loss !== null && p.train_loss !== undefined && isFinite(p.train_loss)) {
        rows.push({ epoch: +p.epoch, loss: +p.train_loss, series: "Train loss" });
      }
      if (p.val_loss !== null && p.val_loss !== undefined && isFinite(p.val_loss)) {
        rows.push({ epoch: +p.epoch, loss: +p.val_loss, series: "Validation loss" });
      }
    }
    if (!rows.length) {
      if (section) section.style.display = "none";
      return;
    }

    const domain = hist.has_val ? ["Train loss", "Validation loss"] : ["Train loss"];
    const chart = Plot.plot({
      width: 1100,
      height: 300,
      marginLeft: 64,
      marginBottom: 40,
      x: { label: "Epoch", tickFormat: d => d, grid: true },
      y: { label: "Loss", grid: true },
      color: {
        legend: true,
        domain: domain,
        range: ["#2563eb", "#dc2626"],
      },
      marks: [
        Plot.ruleY([0], { stroke: "#e5e7eb" }),
        Plot.line(rows, { x: "epoch", y: "loss", stroke: "series", strokeWidth: 1.8 }),
        Plot.dot(rows, { x: "epoch", y: "loss", fill: "series", r: 2.5 }),
        Plot.tip(rows, Plot.pointerX({
          x: "epoch", y: "loss", stroke: "series",
          channels: { series: "series" },
        })),
      ],
    });
    const container = $("chart-training-curve");
    if (container) { container.innerHTML = ""; container.appendChild(chart); }
  }

  // ---------------------------------------------------------------------
  // Drive it
  // ---------------------------------------------------------------------
  function renderAll() {
    try { renderHeader(); } catch (e) { console.error(e); }
    try { renderMetrics(); } catch (e) { console.error(e); }
    try { renderTrainingCurve(); } catch (e) { console.error(e); }
    try { renderActualVsPredicted(); } catch (e) { console.error(e); }
    try { renderResidualsOverTime(); } catch (e) { console.error(e); }
    try { renderResidualHistogram(); } catch (e) { console.error(e); }
    try { renderParity(); } catch (e) { console.error(e); }
    try { renderHourlyMAE(); } catch (e) { console.error(e); }
    try { renderDOWMAE(); } catch (e) { console.error(e); }
    try { renderPeakMetrics(); } catch (e) { console.error(e); }
    try { renderInputPreview(); } catch (e) { console.error(e); }
    try { renderResidualAnalysis(); } catch (e) { console.error(e); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderAll);
  } else {
    renderAll();
  }
})();
