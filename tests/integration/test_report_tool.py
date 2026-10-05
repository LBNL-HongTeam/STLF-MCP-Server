"""
Integration test for the generate_evaluation_report MCP tool.

Trains a small NaiveSeasonal model on synthetic data, then exercises the
report tool end-to-end and asserts the resulting HTML is well-formed,
self-contained, and contains the expected sections.
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
    generate_evaluation_report,
)


@pytest.fixture
def sample_hourly_csv():
    """Synthetic hourly data with daily seasonality + slow trend + weather."""
    n_points = 600  # ~25 days
    rng = np.random.RandomState(42)
    idx = pd.date_range("2024-01-01", periods=n_points, freq="h")
    hour = idx.hour
    base = 100 + 25 * np.sin(2 * np.pi * hour / 24)
    noise = rng.normal(0, 2, n_points)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "electricity_kwh": base + noise,
            "outdoor_temp": 50 + 10 * np.sin(2 * np.pi * hour / 24),
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


class TestGenerateEvaluationReport:
    @pytest.mark.asyncio
    async def test_end_to_end_html_report(self, sample_hourly_csv, temp_model_dir, tmp_path):
        # 1. Train a quick baseline model
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            building_name="rpt_test",
        )
        assert train["success"], train.get("error")
        model_id = train["model_id"]

        # 2. Generate the report
        out_path = tmp_path / "report.html"
        result = generate_evaluation_report(
            model_id=model_id,
            csv_path=sample_hourly_csv,
            output_html_path=str(out_path),
            include_residual_analysis=True,
            title="Integration test report",
        )

        # 3. Validate tool response
        assert result["success"], result.get("error")
        assert result["output_html_path"] == str(out_path)
        assert result["n_points"] > 0
        assert result["model_id"] == model_id
        assert result["model_type"] == "NaiveSeasonal"
        assert result["test_metrics"]["rmse"] is not None
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
            'id="metrics"',
            'id="actual-vs-pred"',
            'id="residuals"',
            'id="error-profile"',
            'id="input-preview"',
            'id="residual-analysis"',
        ]:
            assert section_id in html, f"Missing section {section_id}"

        # Title override propagated to <title> and header
        assert "Integration test report" in html

        # Data marker present
        assert "window.__REPORT_DATA__" in html

    @pytest.mark.asyncio
    async def test_peak_metrics_in_report(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        """When peak_dates is provided, peak_metrics must appear in the response
        and a Peak metrics section must be rendered into the HTML payload."""
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            building_name="peak_rpt_test",
        )
        assert train["success"], train.get("error")
        model_id = train["model_id"]

        # Series covers 2024-01-01 to ~2024-01-25; pick a couple of days that
        # fall well inside the evaluation window (after the lookback prefix).
        peak_dates = ["2024-01-15", "2024-01-16"]

        out_path = tmp_path / "peak_report.html"
        result = generate_evaluation_report(
            model_id=model_id,
            csv_path=sample_hourly_csv,
            output_html_path=str(out_path),
            peak_dates=peak_dates,
        )
        assert result["success"], result.get("error")

        # Response must include peak_metrics
        pm = result.get("peak_metrics")
        assert pm is not None, "peak_metrics missing from response"
        assert pm["n_peak_days_evaluated"] >= 1
        assert pm["peak_mape"] is not None
        assert isinstance(pm["per_day"], list)
        assert len(pm["per_day"]) >= 1

        # HTML must include the section + embedded payload data
        html = out_path.read_text(encoding="utf-8")
        assert 'id="peak-metrics"' in html
        # Peak dates should appear as data in the embedded payload
        assert "2024-01-15" in html

        # The payload JSON must contain peak_metrics with per_day_results
        import json, re
        m = re.search(r"window\.__REPORT_DATA__\s*=\s*(\{.*?\});\s*<\/script>", html, re.S)
        assert m, "Could not locate window.__REPORT_DATA__ in HTML"
        payload = json.loads(m.group(1).replace("<\\/", "</"))
        payload_pm = payload.get("peak_metrics") or {}
        assert payload_pm.get("peak_mape") is not None
        assert isinstance(payload_pm.get("per_day_results"), list)
        assert len(payload_pm["per_day_results"]) == pm["n_peak_days_evaluated"]

    @pytest.mark.asyncio
    async def test_no_peak_metrics_when_not_requested(
        self, sample_hourly_csv, temp_model_dir, tmp_path
    ):
        """Without peak_dates, response has no peak_metrics key and the payload's
        peak_metrics dict is empty (JS hides the section)."""
        train = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train["success"]

        out_path = tmp_path / "no_peak.html"
        result = generate_evaluation_report(
            model_id=train["model_id"],
            csv_path=sample_hourly_csv,
            output_html_path=str(out_path),
        )
        assert result["success"]
        assert "peak_metrics" not in result

        html = out_path.read_text(encoding="utf-8")
        # Section markup is always present; JS hides it client-side.
        assert 'id="peak-metrics"' in html

        import json, re
        m = re.search(r"window\.__REPORT_DATA__\s*=\s*(\{.*?\});\s*<\/script>", html, re.S)
        assert m
        payload = json.loads(m.group(1).replace("<\\/", "</"))
        assert payload.get("peak_metrics") == {}

    @pytest.mark.asyncio
    async def test_missing_model_returns_error(self, sample_hourly_csv, temp_model_dir, tmp_path):
        result = generate_evaluation_report(
            model_id="does_not_exist",
            csv_path=sample_hourly_csv,
            output_html_path=str(tmp_path / "x.html"),
        )
        assert result["success"] is False
        assert "Evaluation step failed" in result["error"]

    @pytest.mark.asyncio
    async def test_rejects_bad_extension(self, temp_model_dir, tmp_path):
        result = generate_evaluation_report(
            model_id="any",
            csv_path="anything.csv",
            output_html_path=str(tmp_path / "report.txt"),
        )
        assert result["success"] is False
        assert ".html" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_output_path_rejected(self, temp_model_dir):
        result = generate_evaluation_report(
            model_id="any",
            csv_path="anything.csv",
            output_html_path="",
        )
        assert result["success"] is False
        assert "required" in result["error"].lower()
