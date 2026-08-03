"""
Integration test for the generate_backtest_report MCP tool.

Trains a small NaiveSeasonal model on synthetic data, then exercises the
backtest report tool end-to-end and asserts the resulting HTML is well-formed,
self-contained, and contains the expected sections.

Also exercises calculate_horizon_metrics directly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.tools import (
    train_forecast_model,
    generate_backtest_report,
)


@pytest.fixture
def sample_hourly_csv():
    """Synthetic hourly data — 600 points (~25 days) with a daily cycle."""
    n_points = 600
    rng = np.random.RandomState(7)
    idx = pd.date_range("2024-01-01", periods=n_points, freq="h")
    hour = idx.hour
    base = 100 + 25 * np.sin(2 * np.pi * hour / 24)
    noise = rng.normal(0, 2, n_points)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "electricity_kwh": base + noise,
        }
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


class TestGenerateBacktestReport:
    @pytest.mark.asyncio
    async def test_end_to_end_html_report(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        # 1. Train
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            building_name="bt_rpt_test",
        )
        assert train["success"], train.get("error")
        model_id = train["model_id"]

        # 2. Generate backtest report
        out_path = tmp_path / "backtest_report.html"
        result = generate_backtest_report(
            model_id=model_id,
            csv_path=sample_hourly_csv,
            output_html_path=str(out_path),
            include_residual_analysis=True,
            title="Backtest integration test",
        )

        # 3. Validate tool response
        assert result["success"], result.get("error")
        assert result["output_html_path"] == str(out_path)
        assert result["n_windows"] > 0
        assert result["n_points"] > 0
        assert result["model_id"] == model_id
        assert result["model_type"] == "NaiveSeasonal"
        assert result["backtest_metrics"]["rmse"] is not None
        assert result["file_size_bytes"] > 400_000  # vendored JS alone

        # 4. Validate file on disk
        assert out_path.exists()
        html = out_path.read_text(encoding="utf-8")
        assert html.lower().startswith("<!doctype html>")
        assert "</html>" in html

        # Self-contained: no external CDN references
        assert "script src=" not in html.lower()
        assert "cdn." not in html.lower()

        # Embedded libraries
        assert "d3js.org" in html
        assert "observablehq/plot" in html

        # Expected sections
        for section_id in [
            'id="meta"',
            'id="bt-summary"',
            'id="metrics"',
            'id="playback"',
            'id="horizon-error"',
            'id="error-profile"',
            'id="residual-analysis"',
        ]:
            assert section_id in html, f"Missing section: {section_id}"

        # Title override propagated
        assert "Backtest integration test" in html

        # Data marker
        assert "window.__BACKTEST_DATA__" in html

        # Slider element present
        assert 'id="bt-slider"' in html

    @pytest.mark.asyncio
    async def test_peak_metrics_in_report(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        """peak_dates → peak_metrics in tool response AND in the embedded HTML payload."""
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train["success"]
        model_id = train["model_id"]

        # Series covers 2024-01-01 → ~2024-01-25; the backtest starts at
        # start_fraction=0.2 (~day 5). Pick dates deep in the backtest region.
        peak_dates = ["2024-01-18", "2024-01-19"]

        out_path = tmp_path / "bt_peak.html"
        result = generate_backtest_report(
            model_id=model_id,
            csv_path=sample_hourly_csv,
            output_html_path=str(out_path),
            peak_dates=peak_dates,
        )
        assert result["success"], result.get("error")

        pm = result.get("peak_metrics")
        assert pm is not None
        assert pm["n_peak_days_evaluated"] >= 1
        assert pm["peak_mape"] is not None

        html = out_path.read_text(encoding="utf-8")
        assert 'id="peak-metrics"' in html
        assert "2024-01-18" in html

        import json, re
        m = re.search(r"window\.__BACKTEST_DATA__\s*=\s*(\{.*?\});\s*<\/script>", html, re.S)
        assert m
        payload = json.loads(m.group(1).replace("<\\/", "</"))
        payload_pm = payload.get("peak_metrics") or {}
        assert payload_pm.get("peak_mape") is not None
        assert isinstance(payload_pm.get("per_day_results"), list)
        assert len(payload_pm["per_day_results"]) == pm["n_peak_days_evaluated"]

    @pytest.mark.asyncio
    async def test_backtest_summary_fields(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train["success"]
        model_id = train["model_id"]

        result = generate_backtest_report(
            model_id=model_id,
            csv_path=sample_hourly_csv,
            output_html_path=str(tmp_path / "bt2.html"),
            stride_hours=6,
            start_fraction=0.3,
        )
        assert result["success"], result.get("error")

        summary = result["backtest_summary"]
        assert summary["stride_hours"] == 6
        assert summary["start_fraction"] == 0.3
        assert summary["n_windows"] > 0
        assert summary["backtest_start_date"] is not None

    @pytest.mark.asyncio
    async def test_missing_model_returns_error(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        result = generate_backtest_report(
            model_id="does_not_exist",
            csv_path=sample_hourly_csv,
            output_html_path=str(tmp_path / "x.html"),
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_bad_extension(self, temp_model_dir, tmp_path):
        result = generate_backtest_report(
            model_id="any",
            csv_path="anything.csv",
            output_html_path=str(tmp_path / "report.txt"),
        )
        assert result["success"] is False
        assert ".html" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_output_path_rejected(self, temp_model_dir):
        result = generate_backtest_report(
            model_id="any",
            csv_path="anything.csv",
            output_html_path="",
        )
        assert result["success"] is False
        assert "required" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_start_fraction_rejected(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train["success"]

        result = generate_backtest_report(
            model_id=train["model_id"],
            csv_path=sample_hourly_csv,
            output_html_path=str(tmp_path / "bad.html"),
            start_fraction=0.99,
        )
        assert result["success"] is False
        assert "start_fraction" in result["error"]


class TestCalculateHorizonMetrics:
    """Unit-level tests for calculate_horizon_metrics via the tool path."""

    @pytest.mark.asyncio
    async def test_horizon_metrics_present_in_response(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        """The HTML payload must contain horizon_metrics with one entry per step."""
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train["success"]
        model_id = train["model_id"]
        horizon_hours = 6

        result = generate_backtest_report(
            model_id=model_id,
            csv_path=sample_hourly_csv,
            output_html_path=str(tmp_path / "hm.html"),
        )
        assert result["success"], result.get("error")

        # Read the embedded JSON payload from the HTML
        import json, re
        html = Path(result["output_html_path"]).read_text(encoding="utf-8")
        # Extract the payload JSON from window.__BACKTEST_DATA__ = {...};
        m = re.search(r"window\.__BACKTEST_DATA__\s*=\s*(\{.*?\});\s*<\/script>", html, re.S)
        assert m, "Could not find __BACKTEST_DATA__ in HTML"
        payload = json.loads(m.group(1).replace("<\\/", "</"))

        hm = payload.get("horizon_metrics", [])
        assert len(hm) == horizon_hours, f"Expected {horizon_hours} horizon steps, got {len(hm)}"
        for entry in hm:
            assert "h" in entry
            assert "rmse" in entry
            assert "mae" in entry
            assert entry["h"] >= 1

    @pytest.mark.asyncio
    async def test_windows_in_payload(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        """Each window in the payload must have forecast_origin and steps."""
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train["success"]

        result = generate_backtest_report(
            model_id=train["model_id"],
            csv_path=sample_hourly_csv,
            output_html_path=str(tmp_path / "wnd.html"),
        )
        assert result["success"], result.get("error")

        import json, re
        html = Path(result["output_html_path"]).read_text(encoding="utf-8")
        m = re.search(r"window\.__BACKTEST_DATA__\s*=\s*(\{.*?\});\s*<\/script>", html, re.S)
        assert m
        payload = json.loads(m.group(1).replace("<\\/", "</"))

        windows = payload.get("windows", [])
        assert len(windows) > 0
        w0 = windows[0]
        assert "forecast_origin" in w0
        assert "steps" in w0
        assert len(w0["steps"]) == 6  # horizon_hours
        step0 = w0["steps"][0]
        assert step0["h"] == 1
        assert "t" in step0
        assert "predicted" in step0
