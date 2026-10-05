/* =============================================================================
   inference_charts.js — Live Inference Dashboard
   Renders the load forecast chart with live Open-Meteo weather overlay.
   Depends on: d3 v7, Observable Plot v0.6, window.__INFERENCE_DATA__
   ============================================================================= */

(function () {
  "use strict";

  /* --------------------------------------------------------------------------
     1. Helpers
     -------------------------------------------------------------------------- */

  const DATA = window.__INFERENCE_DATA__ || {};

  function fmt(v, decimals = 1) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    return Number(v).toFixed(decimals);
  }

  function fmtDt(iso) {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      return d.toLocaleString(undefined, {
        month: "short", day: "numeric",
        hour: "2-digit", minute: "2-digit",
      });
    } catch (_) { return iso; }
  }

  function el(id) { return document.getElementById(id); }

  function setStatus(msg, isError) {
    const s = el("refresh-status");
    if (!s) return;
    s.textContent = msg;
    s.className = "refresh-status " + (isError ? "error" : "ok");
  }

  /* --------------------------------------------------------------------------
     2. Metadata header
     -------------------------------------------------------------------------- */

  function renderMeta() {
    const meta = DATA.meta || {};
    const grid = el("meta-grid");
    if (!grid) return;

    const fields = [
      ["Model ID",       meta.model_id],
      ["Model type",     meta.model_type],
      ["Building",       meta.building_name || "—"],
      ["Frequency",      meta.frequency],
      ["Lookback",       meta.lookback_hours != null ? meta.lookback_hours + "h" : "—"],
      ["Horizon",        meta.horizon_hours  != null ? meta.horizon_hours  + "h" : "—"],
      ["Location",       meta.latitude != null ? `${meta.latitude}, ${meta.longitude}` : "—"],
      ["Timezone",       meta.timezone || "—"],
      ["Generated at",   fmtDt(meta.generated_at)],
    ];

    grid.innerHTML = fields.map(([k, v]) =>
      `<div class="kv"><span class="k">${k}</span><span class="v">${v ?? "—"}</span></div>`
    ).join("");
  }

  /* --------------------------------------------------------------------------
     3. Summary cards
     -------------------------------------------------------------------------- */

  function renderSummaryCards(forecast) {
    const cards = el("summary-cards");
    if (!cards || !forecast || !forecast.length) return;

    const vals = forecast.map(r => r.predicted_load).filter(v => v !== null && v !== undefined);
    if (!vals.length) return;

    const peakVal  = Math.max(...vals);
    const minVal   = Math.min(...vals);
    const avgVal   = vals.reduce((s, v) => s + v, 0) / vals.length;
    const peakRow  = forecast.find(r => r.predicted_load === peakVal);
    const totalKwh = vals.reduce((s, v) => s + v, 0) *
                     (DATA.meta?.frequency === "15min" ? 0.25 :
                      DATA.meta?.frequency === "30min" ? 0.5  : 1.0);

    const items = [
      { label: "Peak load",   value: fmt(peakVal, 1) + " kWh", sub: peakRow ? fmtDt(peakRow.datetime) : "" },
      { label: "Min load",    value: fmt(minVal,  1) + " kWh", sub: "" },
      { label: "Avg load",    value: fmt(avgVal,  1) + " kWh", sub: "" },
      { label: "Total energy",value: fmt(totalKwh, 0) + " kWh", sub: DATA.meta?.horizon_hours + "h horizon" },
      { label: "Horizon",     value: (DATA.meta?.horizon_hours ?? "—") + " h", sub: forecast.length + " steps" },
    ];

    cards.innerHTML = items.map(({ label, value, sub }) => `
      <div class="metric-card">
        <div class="label">${label}</div>
        <div class="value">${value}</div>
        ${sub ? `<div class="compare">${sub}</div>` : ""}
      </div>`
    ).join("");
  }

  /* --------------------------------------------------------------------------
     4. Load forecast chart (Observable Plot)
     -------------------------------------------------------------------------- */

  function renderForecastChart(forecast, liveWeather) {
    const container = el("chart-forecast");
    if (!container || !forecast || !forecast.length) return;
    container.innerHTML = "";

    const parsedForecast = forecast.map(r => ({
      ...r,
      dt: new Date(r.datetime),
      load: r.predicted_load,
    }));

    // Context (past) actual load if available
    const context = (DATA.context_series || []).map(r => ({
      dt: new Date(r.datetime),
      load: r.actual_load,
    }));

    // Compute x domain from actual data — never from wall-clock "now"
    const allDts = [
      ...context.map(r => r.dt),
      ...parsedForecast.map(r => r.dt),
    ];
    const xMin = new Date(Math.min(...allDts));
    const xMax = new Date(Math.max(...allDts));

    // Draw a "transition" rule at the context/forecast boundary if both exist
    const boundaryMarks = [];
    if (context.length && parsedForecast.length) {
      const boundary = new Date(Math.max(...context.map(r => r.dt)));
      boundaryMarks.push(
        Plot.ruleX([boundary], { stroke: "#ef4444", strokeWidth: 1.5, strokeDasharray: "4,3" })
      );
    }

    // Prediction band (probabilistic models only). Rows carry per-level
    // "q<level>" keys straight from generate_forecast.
    const band = (DATA.meta || {}).band;
    const bandRows = band
      ? parsedForecast
          .map(r => ({ dt: r.dt, lo: r[band.lower_key], hi: r[band.upper_key] }))
          .filter(r => r.lo !== null && r.lo !== undefined && isFinite(r.lo) &&
                       r.hi !== null && r.hi !== undefined && isFinite(r.hi))
      : [];

    const marks = [
      Plot.areaY(parsedForecast, {
        x: "dt", y: "load",
        fill: "#dbeafe", stroke: "none",
        curve: "monotone-x",
      }),
      Plot.lineY(parsedForecast, {
        x: "dt", y: "load",
        stroke: "#2563eb", strokeWidth: 2,
        curve: "monotone-x",
        tip: true,
        title: d => `${fmtDt(d.datetime)}\n${fmt(d.load, 2)} kWh`,
      }),
      ...boundaryMarks,
    ];

    if (bandRows.length) {
      // Insert after the decorative fill but before the forecast line so the
      // median stays legible on top of the shaded interval.
      marks.splice(1, 0, Plot.areaY(bandRows, {
        x: "dt", y1: "lo", y2: "hi",
        fill: "#2563eb", fillOpacity: 0.18, stroke: "none",
        curve: "monotone-x",
      }));
    }

    if (context.length) {
      marks.unshift(
        Plot.lineY(context, {
          x: "dt", y: "load",
          stroke: "#6b7280", strokeWidth: 1.5,
          strokeDasharray: "3,2",
          curve: "monotone-x",
          tip: true,
          title: d => `${fmtDt(d.dt.toISOString())}\n${fmt(d.load, 2)} kWh (actual)`,
        })
      );
    }

    const plot = Plot.plot({
      width: container.clientWidth || 900,
      height: 280,
      marginLeft: 60,
      marginRight: 20,
      x: { domain: [xMin, xMax], label: null, tickFormat: d => fmtDt(d.toISOString()) },
      y: { label: "Load (kWh)", grid: true },
      marks,
    });

    container.appendChild(plot);

    // Legend
    const hasBoundary = context.length && parsedForecast.length;
    const legend = document.createElement("div");
    legend.className = "chart-legend";
    const bandLabel = bandRows.length
      ? "P" + Math.round(band.lower * 100) + "–P" + Math.round(band.upper * 100) + " interval"
      : "";
    legend.innerHTML = `
      <span class="legend-item"><span class="dot" style="background:#2563eb"></span>Forecast${bandRows.length ? " (median)" : ""}</span>
      ${bandRows.length ? '<span class="legend-item"><span class="dot" style="background:#2563eb;opacity:0.25"></span>' + bandLabel + '</span>' : ""}
      ${context.length ? '<span class="legend-item"><span class="dash" style="background:#6b7280"></span>Context (actual)</span>' : ""}
      ${hasBoundary ? '<span class="legend-item"><span class="dash" style="background:#ef4444"></span>Forecast start</span>' : ""}`;
    container.appendChild(legend);
  }

  /* --------------------------------------------------------------------------
     5. Weather chart
     -------------------------------------------------------------------------- */

  function renderWeatherChart(weatherData, liveRows) {
    const container = el("chart-weather");
    if (!container) return;
    container.innerHTML = "";

    const rows = (liveRows || weatherData || []).map(r => ({
      dt: new Date(r.datetime),
      temperature: r.temperature ?? r.temperature_2m ?? null,
      solar:       r.solar_radiation ?? r.shortwave_radiation ?? null,
      humidity:    r.relative_humidity ?? r.relative_humidity_2m ?? null,
    })).filter(r => r.temperature !== null);

    if (!rows.length) {
      container.innerHTML = '<p class="note">No weather data available.</p>';
      return;
    }

    const tempMark = Plot.lineY(rows, {
      x: "dt", y: "temperature",
      stroke: "#ef4444", strokeWidth: 2,
      curve: "monotone-x",
      tip: true,
      title: d => `${fmtDt(d.dt.toISOString())}\n${fmt(d.temperature, 1)} °C`,
    });

    // Only draw "now" rule if it falls within the weather data range
    const wxMin = new Date(Math.min(...rows.map(r => r.dt)));
    const wxMax = new Date(Math.max(...rows.map(r => r.dt)));
    const nowDt  = new Date();
    const nowMarks = (nowDt >= wxMin && nowDt <= wxMax)
      ? [Plot.ruleX([nowDt], { stroke: "#94a3b8", strokeDasharray: "4,3" })]
      : [];

    const plot = Plot.plot({
      width: container.clientWidth || 900,
      height: 200,
      marginLeft: 55,
      marginRight: 80,
      x: { domain: [wxMin, wxMax], label: null },
      y: { label: "Temperature (°C)", grid: true },
      marks: [tempMark, ...nowMarks],
    });
    container.appendChild(plot);

    // Solar overlay if present
    const solarRows = rows.filter(r => r.solar !== null);
    if (solarRows.length) {
      const container2 = el("chart-solar");
      if (container2) {
        container2.innerHTML = "";
        container2.appendChild(
          Plot.plot({
            width: container2.clientWidth || 900,
            height: 160,
            marginLeft: 55,
            marginRight: 80,
            x: { domain: [wxMin, wxMax], label: null },
            y: { label: "Solar (W/m²)", grid: true },
            marks: [
              Plot.areaY(solarRows, { x: "dt", y: "solar", fill: "#fef08a", curve: "monotone-x" }),
              Plot.lineY(solarRows, { x: "dt", y: "solar", stroke: "#d97706", strokeWidth: 1.5, curve: "monotone-x" }),
              ...nowMarks,
            ],
          })
        );
      }
    }
  }

  /* --------------------------------------------------------------------------
     6. Weather table
     -------------------------------------------------------------------------- */

  function renderWeatherTable(rows) {
    const tbody = el("weather-table-body");
    if (!tbody || !rows || !rows.length) return;
    tbody.innerHTML = rows.slice(0, 48).map(r => {
      const temp = r.temperature ?? r.temperature_2m ?? null;
      const hum  = r.relative_humidity ?? r.relative_humidity_2m ?? null;
      const sol  = r.solar_radiation ?? r.shortwave_radiation ?? null;
      const wind = r.wind_speed ?? r.wind_speed_10m ?? null;
      return `<tr>
        <td>${fmtDt(r.datetime)}</td>
        <td>${fmt(temp, 1)}</td>
        <td>${fmt(hum, 0)}</td>
        <td>${fmt(sol, 0)}</td>
        <td>${fmt(wind, 1)}</td>
      </tr>`;
    }).join("");
  }

  /* --------------------------------------------------------------------------
     7. Forecast table
     -------------------------------------------------------------------------- */

  function renderForecastTable(forecast) {
    const tbody = el("forecast-table-body");
    if (!tbody || !forecast || !forecast.length) return;
    tbody.innerHTML = forecast.map((r, i) => `
      <tr>
        <td>${i + 1}</td>
        <td>${fmtDt(r.datetime)}</td>
        <td>${fmt(r.predicted_load, 2)}</td>
      </tr>`
    ).join("");
  }

  /* --------------------------------------------------------------------------
     8. Live Open-Meteo fetch
     -------------------------------------------------------------------------- */

  async function fetchLiveWeather() {
    const meta = DATA.meta || {};
    const lat  = meta.latitude;
    const lon  = meta.longitude;
    if (lat == null || lon == null) {
      throw new Error("No latitude/longitude in dashboard data — cannot fetch live weather.");
    }

    const variables = [
      "temperature_2m",
      "relative_humidity_2m",
      "apparent_temperature",
      "shortwave_radiation",
      "wind_speed_10m",
      "cloud_cover",
      "precipitation",
    ].join(",");

    const tz       = encodeURIComponent(meta.timezone || "UTC");
    const fhours   = Math.max(meta.horizon_hours || 24, 24);
    const pastH    = Math.min(meta.lookback_hours || 24, 92);
    const url      = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&hourly=${variables}&timezone=${tz}&timeformat=iso8601&forecast_hours=${fhours}&past_hours=${pastH}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Open-Meteo HTTP ${res.status}: ${res.statusText}`);
    const json = await res.json();

    if (json.error) throw new Error(`Open-Meteo API error: ${json.reason}`);

    const hourly = json.hourly;
    if (!hourly || !hourly.time) throw new Error("Open-Meteo response missing hourly data.");

    // Rename to short names matching ForecastingDataLoader convention
    const renameMap = {
      temperature_2m:      "temperature",
      relative_humidity_2m:"relative_humidity",
      apparent_temperature:"apparent_temperature",
      shortwave_radiation: "solar_radiation",
      wind_speed_10m:      "wind_speed",
      cloud_cover:         "cloud_cover",
      precipitation:       "precipitation",
    };

    return hourly.time.map((t, i) => {
      const row = { datetime: t };
      for (const [src, dst] of Object.entries(renameMap)) {
        if (hourly[src] !== undefined) row[dst] = hourly[src][i];
      }
      return row;
    });
  }

  /* --------------------------------------------------------------------------
     9. Refresh button handler
     -------------------------------------------------------------------------- */

  async function handleRefresh() {
    const btn = el("refresh-btn");
    if (btn) { btn.disabled = true; btn.textContent = "Fetching…"; }
    setStatus("Fetching live weather from Open-Meteo…", false);

    try {
      const liveWeather = await fetchLiveWeather();
      window.__LIVE_WEATHER__ = liveWeather;

      renderWeatherChart(DATA.weather_forecast, liveWeather);
      renderWeatherTable(liveWeather);

      const ts = new Date().toLocaleTimeString();
      setStatus(`Weather updated at ${ts}. Forecast below uses model snapshot — retrain or re-run server tool for updated load predictions.`, false);

      // Update last-refresh display
      const refreshEl = el("last-refresh");
      if (refreshEl) refreshEl.textContent = `Last weather refresh: ${ts}`;

    } catch (err) {
      setStatus(`Live weather fetch failed: ${err.message}`, true);
      console.error("Open-Meteo fetch error:", err);
    } finally {
      if (btn) { btn.disabled = false; btn.textContent = "Refresh Weather"; }
    }
  }

  /* --------------------------------------------------------------------------
     10. Auto-refresh toggle
     -------------------------------------------------------------------------- */

  let _autoRefreshInterval = null;

  function setupAutoRefresh() {
    const toggle = el("auto-refresh-toggle");
    if (!toggle) return;
    toggle.addEventListener("change", () => {
      if (toggle.checked) {
        handleRefresh();
        _autoRefreshInterval = setInterval(handleRefresh, 5 * 60 * 1000); // every 5 min
        setStatus("Auto-refresh enabled (every 5 minutes).", false);
      } else {
        clearInterval(_autoRefreshInterval);
        _autoRefreshInterval = null;
        setStatus("Auto-refresh disabled.", false);
      }
    });
  }

  /* --------------------------------------------------------------------------
     11. Main entry point
     -------------------------------------------------------------------------- */

  function main() {
    const forecast = DATA.forecast || [];
    const weather  = DATA.weather_forecast || [];

    renderMeta();
    renderSummaryCards(forecast);
    renderForecastChart(forecast, null);
    renderWeatherChart(weather, null);
    renderWeatherTable(weather);
    renderForecastTable(forecast);

    // Wire up refresh button
    const btn = el("refresh-btn");
    if (btn) btn.addEventListener("click", handleRefresh);
    setupAutoRefresh();

    // Subtitle
    const sub = el("dashboard-subtitle");
    if (sub) {
      const meta = DATA.meta || {};
      sub.textContent =
        `${meta.model_type || "Model"} · Horizon: ${meta.horizon_hours || "?"}h · ` +
        `Generated: ${fmtDt(meta.generated_at)}`;
    }

    // Initial live fetch if coords are present
    if ((DATA.meta || {}).latitude != null) {
      handleRefresh().catch(() => {});
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", main);
  } else {
    main();
  }

})();
