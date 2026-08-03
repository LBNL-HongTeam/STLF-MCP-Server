/* Self-contained chart renderer for the backtest report.
   Consumes window.__BACKTEST_DATA__ and renders every section.

   Depends on globals provided by the embedded scripts:
     - d3        (from d3.min.js)
     - Plot      (from observable-plot.min.js, UMD bundle exposes window.Plot)
*/
(function () {
  "use strict";

  const DATA = window.__BACKTEST_DATA__ || {};
  const Plot = window.Plot;
  if (!Plot) {
    console.error("Observable Plot did not load");
    return;
  }

  // -------------------------------------------------------------------------
  // Small helpers (identical to charts.js)
  // -------------------------------------------------------------------------
  function $(id) { return document.getElementById(id); }

  function fmtNumber(v, digits) {
    if (v === null || v === undefined || !isFinite(v)) return "\u2014";
    digits = (digits === undefined) ? 3 : digits;
    return Number(v).toLocaleString(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: 0,
    });
  }

  function fmtDate(s) {
    if (!s) return "\u2014";
    const d = new Date(s);
    if (isNaN(d)) return s;
    return d.toLocaleString();
  }

  function setText(id, text) {
    const el = $(id);
    if (el) el.textContent = (text === null || text === undefined) ? "\u2014" : text;
  }

  function append(parent, node) {
    const p = (typeof parent === "string") ? $(parent) : parent;
    if (p && node) p.appendChild(node);
  }

  function clearEl(id) {
    const el = $(id);
    if (el) el.innerHTML = "";
  }

  function makeKV(label, value) {
    const wrap = document.createElement("div");
    wrap.className = "kv";
    const k = document.createElement("span");
    k.className = "k"; k.textContent = label;
    const v = document.createElement("span");
    v.className = "v";
    v.textContent = (value === null || value === undefined || value === "") ? "\u2014" : value;
    wrap.appendChild(k); wrap.appendChild(v);
    return wrap;
  }

  function makeSummaryCard(label, value) {
    const card = document.createElement("div");
    card.className = "summary-card";
    const lab = document.createElement("div");
    lab.className = "label"; lab.textContent = label;
    const val = document.createElement("div");
    val.className = "value"; val.textContent = value;
    card.appendChild(lab); card.appendChild(val);
    return card;
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

  // -------------------------------------------------------------------------
  // Header + metadata
  // -------------------------------------------------------------------------
  function renderHeader() {
    const meta = DATA.meta || {};
    setText("report-title", meta.title || "Backtest report");
    setText("report-subtitle",
      "Generated " + fmtDate(DATA.generated_at) +
      " \xb7 model " + (meta.model_id || "?"));

    const grid = $("meta-grid");
    if (!grid) return;
    grid.innerHTML = "";
    grid.appendChild(makeKV("Model ID", meta.model_id));
    grid.appendChild(makeKV("Model type", meta.model_type));
    grid.appendChild(makeKV("Building", meta.building_name || "\u2014"));
    grid.appendChild(makeKV("Created", fmtDate(meta.created_at)));
    grid.appendChild(makeKV("Lookback (hrs)", meta.lookback_hours));
    grid.appendChild(makeKV("Horizon (hrs)", meta.horizon_hours));
    grid.appendChild(makeKV("Frequency", meta.frequency));
    grid.appendChild(makeKV("Validation split", meta.validation_split));
    const trange = meta.train_data_range || {};
    grid.appendChild(makeKV("Train start", fmtDate(trange.start)));
    grid.appendChild(makeKV("Train end", fmtDate(trange.end)));
    grid.appendChild(makeKV("Train samples", fmtNumber(trange.samples, 0)));
  }

  // -------------------------------------------------------------------------
  // Backtest summary cards
  // -------------------------------------------------------------------------
  function renderBacktestSummary() {
    const s = DATA.backtest_summary || {};
    const cards = $("bt-summary-cards");
    if (!cards) return;
    cards.innerHTML = "";
    cards.appendChild(makeSummaryCard("Windows", fmtNumber(s.n_windows, 0)));
    cards.appendChild(makeSummaryCard("Stride (hrs)", fmtNumber(s.stride_hours, 0)));
    cards.appendChild(makeSummaryCard("Start fraction", fmtNumber(s.start_fraction, 2)));
    cards.appendChild(makeSummaryCard("Backtest start", fmtDate(s.backtest_start_date)));
    cards.appendChild(makeSummaryCard("Backtest end", fmtDate(s.backtest_end_date)));
    cards.appendChild(makeSummaryCard("Total samples", fmtNumber(s.total_samples, 0)));
  }

  // -------------------------------------------------------------------------
  // Metrics cards
  // -------------------------------------------------------------------------
  function renderMetrics() {
    const m = DATA.metrics || {};
    const bt = m.backtest || {};
    const v = m.validation || {};
    const cmp = m.comparison || {};

    const cards = $("metric-cards");
    if (!cards) return;
    cards.innerHTML = "";

    const order = [
      ["RMSE",    "rmse",      ""],
      ["MAE",     "mae",       ""],
      ["MAPE",    "mape",      "%"],
      ["CV-RMSE", "cv_rmse",   "%"],
      ["R\u00b2", "r_squared", ""],
    ];
    for (const [label, key, unit] of order) {
      const val = bt[key];
      const valStr = (val === null || val === undefined) ? "\u2014"
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
      ["Peak MAPE",         "peak_mape",               "%"],
      ["Peak timing error", "peak_timing_error_hours", " h"],
    ];
    for (const [label, key, unit] of peakOrder) {
      const val = bt[key];
      if (val === null || val === undefined) continue;
      cards.appendChild(makeMetricCard(label, fmtNumber(val, 2) + (unit || ""), null));
    }

    const badge = $("compare-badge");
    if (badge) {
      const status = cmp.performance_status;
      if (status) {
        badge.className = "badge " + status;
        badge.textContent = status;
        const diff = cmp.cv_rmse_diff;
        if (diff !== null && diff !== undefined) {
          badge.textContent += "  (\u0394cv_rmse " +
            (diff >= 0 ? "+" : "") + fmtNumber(diff, 2) + ")";
        }
      } else {
        badge.style.display = "none";
      }
    }
  }

  // -------------------------------------------------------------------------
  // Forecast playback slider
  // -------------------------------------------------------------------------
  const playbackState = {
    currentIdx: 0,
    playTimer: null,
    playInterval: 800, // ms per step during auto-play
  };

  function parseActual() {
    return (DATA.full_actual || [])
      .filter(r => r.t && r.value !== null && isFinite(r.value))
      .map(r => ({ t: new Date(r.t), v: +r.value }));
  }

  function renderPlaybackSlider() {
    const windows = DATA.windows || [];
    if (!windows.length) {
      const el = $("chart-playback");
      if (el) el.innerHTML = "<p class='note'>No window data available.</p>";
      return;
    }

    const slider = $("bt-slider");
    if (slider) {
      slider.max = String(windows.length - 1);
      slider.value = "0";
      slider.addEventListener("input", () => {
        playbackState.currentIdx = parseInt(slider.value, 10);
        renderPlaybackWindow(playbackState.currentIdx);
      });
    }

    const prevBtn = $("bt-prev");
    const nextBtn = $("bt-next");
    const playPauseBtn = $("bt-play-pause");

    if (prevBtn) {
      prevBtn.addEventListener("click", () => {
        stopPlay();
        seekTo(playbackState.currentIdx - 1);
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        stopPlay();
        seekTo(playbackState.currentIdx + 1);
      });
    }
    if (playPauseBtn) {
      playPauseBtn.addEventListener("click", () => {
        if (playbackState.playTimer) {
          stopPlay();
        } else {
          startPlay();
        }
      });
    }

    // Keyboard navigation (left/right arrow keys when slider is focused)
    if (slider) {
      slider.addEventListener("keydown", (e) => {
        if (e.key === "ArrowLeft") { stopPlay(); seekTo(playbackState.currentIdx - 1); e.preventDefault(); }
        if (e.key === "ArrowRight") { stopPlay(); seekTo(playbackState.currentIdx + 1); e.preventDefault(); }
      });
    }

    // Initial render
    renderPlaybackWindow(0);
  }

  function seekTo(idx) {
    const windows = DATA.windows || [];
    idx = Math.max(0, Math.min(windows.length - 1, idx));
    playbackState.currentIdx = idx;
    const slider = $("bt-slider");
    if (slider) slider.value = String(idx);
    renderPlaybackWindow(idx);
  }

  function startPlay() {
    const playPauseBtn = $("bt-play-pause");
    if (playPauseBtn) playPauseBtn.textContent = "\u23f8 Pause";
    playbackState.playTimer = setInterval(() => {
      const windows = DATA.windows || [];
      let next = playbackState.currentIdx + 1;
      if (next >= windows.length) {
        stopPlay();
        return;
      }
      seekTo(next);
    }, playbackState.playInterval);
  }

  function stopPlay() {
    if (playbackState.playTimer) {
      clearInterval(playbackState.playTimer);
      playbackState.playTimer = null;
    }
    const playPauseBtn = $("bt-play-pause");
    if (playPauseBtn) playPauseBtn.textContent = "\u25b6 Play";
  }

  // Cache the parsed actual series so we only parse it once
  let _cachedActual = null;
  function getActual() {
    if (!_cachedActual) _cachedActual = parseActual();
    return _cachedActual;
  }

  function renderPlaybackWindow(idx) {
    const windows = DATA.windows || [];
    if (!windows.length) return;

    // Update slider label
    const label = $("bt-slider-label");
    if (label) label.textContent = "Window " + (idx + 1) + " / " + windows.length;

    const win = windows[idx];
    if (!win) return;

    // Update info panel
    renderWindowInfo(win, idx);

    // Build the playback chart
    const actual = getActual();
    if (!actual.length && !win.steps.length) return;

    // Parse window steps for predicted overlay
    const predicted = (win.steps || [])
      .filter(s => s.t && s.predicted !== null && isFinite(s.predicted))
      .map(s => ({ t: new Date(s.t), v: +s.predicted }));

    // Also parse actual points within this window for the "window actual" overlay
    const winActual = (win.steps || [])
      .filter(s => s.t && s.actual !== null && isFinite(s.actual))
      .map(s => ({ t: new Date(s.t), v: +s.actual }));

    // Determine x-domain: show a context window around the forecast
    // Show [origin - context_before, origin + 2*horizon] where context_before = horizon
    const originTime = new Date(win.forecast_origin);
    const horizonMs = predicted.length > 1
      ? (predicted[predicted.length - 1].t - predicted[0].t) + (predicted[0].t - predicted[0].t /* 0 */)
      : 0;

    // Compute step size in ms from first two predicted points (if available)
    let stepMs = 3600000; // 1h default
    if (predicted.length >= 2) {
      stepMs = predicted[1].t - predicted[0].t;
    }
    const horizonDuration = predicted.length * stepMs;
    const contextBefore = Math.max(horizonDuration * 2, stepMs * 24);  // at least 24 steps back

    const xLo = new Date(originTime - contextBefore);
    const xHi = new Date(originTime.getTime() + horizonDuration + stepMs);

    // Clip actual series to display domain
    const actualClipped = actual.filter(p => p.t >= xLo && p.t <= xHi);

    // Build combined array for color legend
    const actualTagged = actualClipped.map(p => ({ ...p, series: "Actual" }));
    const predictedTagged = predicted.map(p => ({ ...p, series: "Predicted" }));
    const combined = actualTagged.concat(predictedTagged);

    // Vertical rule at forecast origin
    const originRule = [{ t: originTime }];

    const chart = Plot.plot({
      width: 1100,
      height: 320,
      marginLeft: 56,
      marginBottom: 36,
      x: {
        type: "utc",
        label: "Time",
        domain: [xLo, xHi],
      },
      y: { label: "Load", grid: true },
      color: {
        legend: true,
        domain: ["Actual", "Predicted"],
        range: ["#6b7280", "#2563eb"],
      },
      marks: [
        Plot.ruleY([0], { stroke: "#e5e7eb" }),
        // Full actual context line (grey)
        Plot.line(actualTagged, {
          x: "t", y: "v", stroke: "series",
          strokeWidth: 1.4,
        }),
        // Predicted trajectory (blue, thicker)
        Plot.line(predictedTagged, {
          x: "t", y: "v", stroke: "series",
          strokeWidth: 2.2,
        }),
        // Dot at each predicted step
        Plot.dot(predictedTagged, {
          x: "t", y: "v", fill: "#2563eb", r: 3,
        }),
        // Vertical rule at forecast origin
        Plot.ruleX(originRule, {
          x: "t",
          stroke: "#b91c1c",
          strokeDasharray: "5,3",
          strokeWidth: 1.5,
        }),
        // Tooltip
        Plot.tip(combined, Plot.pointerX({
          x: "t", y: "v", stroke: "series",
          channels: { series: "series" },
        })),
      ],
    });

    const container = $("chart-playback");
    if (container) {
      container.innerHTML = "";
      container.appendChild(chart);
    }
  }

  function renderWindowInfo(win, idx) {
    const infoEl = $("bt-window-info");
    if (!infoEl) return;
    infoEl.innerHTML = "";

    const items = [
      ["Forecast origin", fmtDate(win.forecast_origin)],
      ["Window", (idx + 1) + " / " + (DATA.windows || []).length],
      ["Window RMSE", fmtNumber(win.window_rmse, 4)],
      ["Window MAE",  fmtNumber(win.window_mae, 4)],
      ["Horizon steps", (win.steps || []).length],
    ];
    for (const [k, v] of items) {
      infoEl.appendChild(makeKV(k, v));
    }
  }

  // -------------------------------------------------------------------------
  // H-step-ahead RMSE bar chart
  // -------------------------------------------------------------------------
  function renderHorizonRMSE() {
    const hm = DATA.horizon_metrics || [];
    if (!hm.length) {
      const el = $("chart-horizon-rmse");
      if (el) el.innerHTML = "<p class='note'>No horizon metrics available.</p>";
      return;
    }

    const valid = hm.filter(d => d.rmse !== null && isFinite(d.rmse));
    if (!valid.length) return;

    const chart = Plot.plot({
      width: 1100,
      height: 260,
      marginLeft: 56,
      marginBottom: 36,
      x: {
        label: "Horizon step h",
        tickFormat: d => d,
        domain: hm.map(d => d.h),
      },
      y: { label: "RMSE", grid: true },
      marks: [
        Plot.ruleY([0]),
        Plot.barY(valid, {
          x: "h",
          y: "rmse",
          fill: "#2563eb",
          fillOpacity: 0.8,
          title: d =>
            "h=" + d.h +
            "\nRMSE: " + fmtNumber(d.rmse, 4) +
            "\nMAE: " + fmtNumber(d.mae, 4) +
            (d.mape !== null ? "\nMAPE: " + fmtNumber(d.mape, 2) + "%" : ""),
        }),
        Plot.tip(valid, Plot.pointerX({
          x: "h", y: "rmse",
          title: d =>
            "h=" + d.h +
            "  RMSE: " + fmtNumber(d.rmse, 4) +
            "  MAE: " + fmtNumber(d.mae, 4),
        })),
      ],
    });
    append("chart-horizon-rmse", chart);
  }

  // -------------------------------------------------------------------------
  // Hour-of-day MAE and Day-of-week MAE (same as eval report)
  // -------------------------------------------------------------------------
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

  // -------------------------------------------------------------------------
  // Residual analysis
  // -------------------------------------------------------------------------
  function renderPeakMetrics() {
    const section = $("peak-metrics");
    const pm = DATA.peak_metrics || {};
    const perDay = Array.isArray(pm.per_day_results) ? pm.per_day_results : [];
    const hasSummary = pm.peak_mape !== undefined && pm.peak_mape !== null;
    if (!hasSummary && !perDay.length) {
      if (section) section.style.display = "none";
      return;
    }
    if (section) section.style.display = "";

    const cards = $("peak-metric-cards");
    if (cards) {
      cards.innerHTML = "";
      cards.appendChild(makeMetricCard(
        "Peak MAPE",
        (pm.peak_mape === null || pm.peak_mape === undefined)
          ? "\u2014" : fmtNumber(pm.peak_mape, 2) + "%",
        null
      ));
      cards.appendChild(makeMetricCard(
        "Peak Timing Error",
        (pm.peak_timing_error_hours === null || pm.peak_timing_error_hours === undefined)
          ? "\u2014" : fmtNumber(pm.peak_timing_error_hours, 2) + " h",
        null
      ));
    }

    const counts = $("peak-metric-counts");
    if (counts) {
      counts.innerHTML = "";
      counts.appendChild(makeKV("Peak days evaluated", fmtNumber(pm.n_peak_days_evaluated, 0)));
      counts.appendChild(makeKV("Peak days skipped", fmtNumber(pm.n_peak_days_skipped, 0)));
    }

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
              ? "\u2014" : String(r.actual_peak_hour).padStart(2, "0") + ":00",
            (r.predicted_peak_hour === null || r.predicted_peak_hour === undefined)
              ? "\u2014" : String(r.predicted_peak_hour).padStart(2, "0") + ":00",
            fmtNumber(r.peak_magnitude_error_pct, 2),
            fmtNumber(r.peak_timing_error_hours, 2),
          ];
          for (const v of cells) {
            const td = document.createElement("td");
            td.textContent = (v === null || v === undefined) ? "\u2014" : v;
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        }
      }
    }

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

    const hasSomething = ra.mean_residual !== undefined
      || ra.std_residual !== undefined
      || ra.autocorrelation_lag1 !== undefined;

    if (!hasSomething) {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = "Residual analysis not requested (set include_residual_analysis=True).";
      grid.appendChild(note);
      return;
    }

    grid.appendChild(makeKV("Mean residual", fmtNumber(ra.mean_residual, 4)));
    grid.appendChild(makeKV("Std residual", fmtNumber(ra.std_residual, 4)));
    grid.appendChild(makeKV("Lag-1 autocorrelation", fmtNumber(ra.autocorrelation_lag1, 4)));
  }

  // -------------------------------------------------------------------------
  // Drive everything
  // -------------------------------------------------------------------------
  function renderAll() {
    try { renderHeader(); }           catch (e) { console.error("renderHeader:", e); }
    try { renderBacktestSummary(); }  catch (e) { console.error("renderBacktestSummary:", e); }
    try { renderMetrics(); }          catch (e) { console.error("renderMetrics:", e); }
    try { renderPlaybackSlider(); }   catch (e) { console.error("renderPlaybackSlider:", e); }
    try { renderHorizonRMSE(); }      catch (e) { console.error("renderHorizonRMSE:", e); }
    try { renderHourlyMAE(); }        catch (e) { console.error("renderHourlyMAE:", e); }
    try { renderDOWMAE(); }           catch (e) { console.error("renderDOWMAE:", e); }
    try { renderPeakMetrics(); }      catch (e) { console.error("renderPeakMetrics:", e); }
    try { renderResidualAnalysis(); } catch (e) { console.error("renderResidualAnalysis:", e); }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderAll);
  } else {
    renderAll();
  }
})();
