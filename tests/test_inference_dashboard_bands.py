"""
Tests for prediction-band rendering in the inference dashboard.

Covers the band-level selection in ``build_inference_dashboard_html`` and the
resulting payload that ``inference_charts.js`` consumes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.reporting import build_inference_dashboard_html


@pytest.fixture
def metadata():
    return {
        "model_type": "XGBoost",
        "building_name": "TestBuilding",
        "config": {"frequency": "h", "lookback_hours": 24, "horizon_hours": 6},
        "data_info": {
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-12-31T23:00:00",
        },
    }


def _forecast(quantiles=(0.1, 0.5, 0.9), n=6):
    rows = []
    for i in range(n):
        r = {"datetime": f"2024-01-01T{i:02d}:00:00", "predicted_load": 100.0 + i}
        for q in quantiles:
            r[f"q{q}"] = r["predicted_load"] + (q - 0.5) * 40.0
        rows.append(r)
    return rows


def _payload(html: str) -> dict:
    m = re.search(r"window\.__DASHBOARD_DATA__\s*=\s*(.*?);\s*\n", html)
    if m is None:
        # Fall back to whatever global the template assigns.
        m = re.search(r"window\.__\w+__\s*=\s*(\{.*?\});\s*\n", html, re.S)
    assert m, "could not locate embedded dashboard payload"
    return json.loads(m.group(1).replace("<\\/", "</"))


def _build(metadata, **kwargs):
    defaults = dict(
        model_id="m1",
        model_metadata=metadata,
        forecast=_forecast(),
        context_series=[{"datetime": "2023-12-31T23:00:00", "actual_load": 99.0}],
        weather_forecast=[],
        latitude=45.5,
        longitude=-122.6,
        timezone="UTC",
    )
    defaults.update(kwargs)
    return build_inference_dashboard_html(**defaults)


class TestInferenceDashboardBands:
    def test_point_model_has_no_band(self, metadata):
        html = _build(metadata, forecast=[
            {"datetime": "2024-01-01T00:00:00", "predicted_load": 100.0}
        ], quantiles=None)
        meta = _payload(html)["meta"]
        assert meta["probabilistic"] is False
        assert meta["band"] is None

    def test_band_uses_outermost_pair(self, metadata):
        html = _build(
            metadata,
            forecast=_forecast(quantiles=(0.05, 0.1, 0.5, 0.9, 0.95)),
            quantiles=[0.05, 0.1, 0.5, 0.9, 0.95],
            num_samples=200,
        )
        meta = _payload(html)["meta"]
        assert meta["probabilistic"] is True
        assert meta["band"]["lower"] == 0.05
        assert meta["band"]["upper"] == 0.95
        assert meta["band"]["nominal"] == 0.9
        # Keys must match the generate_forecast row convention exactly.
        assert meta["band"]["lower_key"] == "q0.05"
        assert meta["band"]["upper_key"] == "q0.95"
        assert meta["num_samples"] == 200

    def test_band_keys_resolve_against_forecast_rows(self, metadata):
        quantiles = [0.1, 0.5, 0.9]
        html = _build(metadata, quantiles=quantiles)
        payload = _payload(html)
        band = payload["meta"]["band"]
        for row in payload["forecast"]:
            assert band["lower_key"] in row
            assert band["upper_key"] in row
            assert row[band["lower_key"]] < row["predicted_load"] < row[band["upper_key"]]

    def test_median_only_yields_no_band(self, metadata):
        html = _build(metadata, quantiles=[0.5])
        assert _payload(html)["meta"]["band"] is None

    def test_html_is_self_contained(self, metadata):
        html = _build(metadata, quantiles=[0.1, 0.5, 0.9])
        assert "<script" in html
        assert "http://" not in html.replace("http://www.w3.org", "")
