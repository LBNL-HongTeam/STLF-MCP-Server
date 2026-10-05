"""
Assemble the self-contained inference dashboard HTML.

Reads the Jinja template, CSS, vendored JS libraries, and the live-fetch
chart JS from disk and inlines them into a single HTML string.  The
resulting file works when opened via file:// — charts render from the
baked-in snapshot; the Refresh Weather button fetches live data from
Open-Meteo in the browser.
"""

from __future__ import annotations

from datetime import datetime

from ._render import read_asset, render_report, to_html_json


def build_inference_dashboard_html(
    *,
    model_id: str,
    model_metadata: dict,
    forecast: list[dict],
    context_series: list[dict],
    weather_forecast: list[dict],
    latitude: float | None,
    longitude: float | None,
    timezone: str,
    title: str | None = None,
    quantiles: list | None = None,
    num_samples: int | None = None,
) -> str:
    """
    Render a self-contained inference dashboard HTML document.

    Args:
        model_id:        Registry ID of the model that produced the forecast.
        model_metadata:  Full metadata dict from ModelRegistry.load_model().
        forecast:        List of {datetime, predicted_load} dicts (forecast steps).
        context_series:  List of {datetime, actual_load} dicts (context window
                         shown as dashed grey line in the chart).
        weather_forecast: List of weather dicts from Open-Meteo (baked in;
                          refreshed live by the browser).
        latitude:        Decimal latitude used for live weather refresh.
        longitude:       Decimal longitude used for live weather refresh.
        timezone:        IANA timezone string (e.g. "America/Los_Angeles").
        title:           Optional dashboard title.
        quantiles:       Quantile levels emitted by a probabilistic model.
                         When two levels straddle the median, the outermost
                         pair is shaded as a prediction band in the chart.
                         None / empty for point models.
        num_samples:     Monte-Carlo sample count behind the quantiles;
                         surfaced in the band caption.

    Returns:
        Complete self-contained HTML document as a string.
    """
    config = model_metadata.get("config", {})
    data_info = model_metadata.get("data_info", {})

    effective_title = title or f"Load Forecast — {model_id}"

    # Widest central band: outermost quantile pair straddling the median.
    qs = sorted(float(q) for q in (quantiles or []))
    lows = [q for q in qs if q < 0.5]
    highs = [q for q in qs if q > 0.5]
    band = (
        {
            "lower": lows[0],
            "upper": highs[-1],
            "lower_key": f"q{lows[0]}",
            "upper_key": f"q{highs[-1]}",
            "nominal": round(highs[-1] - lows[0], 4),
        }
        if lows and highs
        else None
    )

    payload = {
        "meta": {
            "model_id":      model_id,
            "model_type":    model_metadata.get("model_type", ""),
            "building_name": model_metadata.get("building_name"),
            "frequency":     config.get("frequency", "h"),
            "lookback_hours": config.get("lookback_hours"),
            "horizon_hours":  config.get("horizon_hours"),
            "latitude":  latitude,
            "longitude": longitude,
            "timezone":  timezone,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "training_start": data_info.get("start_date"),
            "training_end":   data_info.get("end_date"),
            "probabilistic":  bool(band),
            "quantiles":      qs,
            "num_samples":    num_samples,
            "band":           band,
        },
        "forecast":         forecast,
        "context_series":   context_series,
        "weather_forecast": weather_forecast,
    }

    return render_report(
        "inference_template.html",
        title=effective_title,
        n_steps=len(forecast),
        charts_js=read_asset("inference_charts.js"),
        payload_json=to_html_json(payload),
    )
