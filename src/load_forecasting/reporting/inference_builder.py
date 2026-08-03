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

    Returns:
        Complete self-contained HTML document as a string.
    """
    config = model_metadata.get("config", {})
    data_info = model_metadata.get("data_info", {})

    effective_title = title or f"Load Forecast — {model_id}"

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
