"""
Tests for prediction-interval support in the backtest report.

Focus is the per-horizon-step coverage decomposition, which is the reason the
backtest is the right home for interval diagnostics: pooled coverage can hide
an over-covered near horizon averaging out an under-covered far horizon.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.reporting import build_backtest_payload
from load_forecasting.reporting.backtest_builder import build_backtest_report_html
from load_forecasting.tools.backtest import _band_levels


HORIZON = 3
N_WINDOWS = 2


@pytest.fixture
def actual_series():
    idx = pd.date_range("2024-01-01", periods=12, freq="h")
    return TimeSeries.from_times_and_values(idx, np.full(len(idx), 100.0))


@pytest.fixture
def windows_raw():
    out = []
    for w in range(N_WINDOWS):
        idx = pd.date_range(f"2024-01-01 {w * HORIZON:02d}:00", periods=HORIZON, freq="h")
        out.append(TimeSeries.from_times_and_values(idx, np.full(HORIZON, 99.0)))
    return out


@pytest.fixture
def metadata():
    return {
        "model_type": "XGBoost",
        "building_name": "B",
        "created_at": "2024-01-01T00:00:00",
        "config": {
            "lookback_hours": 24,
            "horizon_hours": HORIZON,
            "frequency": "h",
            "probabilistic": True,
            "quantiles": [0.1, 0.5, 0.9],
        },
        "data_info": {"start_date": "2023-01-01", "end_date": "2023-12-31", "total_samples": 100},
        "metrics": {"validation": {"rmse": 1.0}},
    }


@pytest.fixture
def backtest_result():
    return {
        "backtest_metrics": {"rmse": 1.0, "mae": 1.0, "mape": 1.0, "cv_rmse": 1.0, "r_squared": 0.9},
        "comparison_to_validation": {"performance_status": "similar", "cv_rmse_diff": 0.0},
        "backtest_summary": {"n_windows": N_WINDOWS, "stride_hours": HORIZON},
        "predictions": [],
        "residual_analysis": {},
        "peak_metrics": None,
    }


def _probabilistic_block(windows_raw):
    """Synthetic stochastic-pass output matching the window timestamps."""
    window_bands = []
    for w in windows_raw:
        window_bands.append([
            {"t": ts.isoformat(), "lower": 95.0, "upper": 105.0}
            for ts in w.time_index
        ])
    return {
        "quantiles": [0.1, 0.5, 0.9],
        "num_samples": 100,
        "n_windows": len(windows_raw),
        "pinball_loss": 1.5,
        "coverage": 0.9,
        "nominal_coverage": 0.8,
        "mean_interval_width": 10.0,
        "per_quantile_pinball": {"0.1": 1.0, "0.5": 2.0, "0.9": 1.5},
        "band": {"lower": 0.1, "upper": 0.9, "nominal": 0.8},
        # Deliberately degrading: fine at h=1, badly under-covered by h=3.
        "horizon_coverage": [
            {"h": 1, "coverage": 0.97, "mean_interval_width": 4.0, "n": 2},
            {"h": 2, "coverage": 0.85, "mean_interval_width": 8.0, "n": 2},
            {"h": 3, "coverage": 0.62, "mean_interval_width": 12.0, "n": 2},
        ],
        "window_bands": window_bands,
    }


def _build(metadata, backtest_result, windows_raw, actual_series, prob):
    return build_backtest_payload(
        model_id="m",
        model_metadata=metadata,
        backtest_result=backtest_result,
        windows_raw=windows_raw,
        actual_series_inv=actual_series,
        horizon_metrics=[{"h": h, "rmse": 1.0, "mae": 1.0, "mape": 1.0} for h in range(1, HORIZON + 1)],
        input_df=None,
        column_mapping={"target": "kwh"},
        probabilistic=prob,
    )


class TestBandLevels:
    def test_outermost_pair(self):
        assert _band_levels([0.05, 0.1, 0.5, 0.9, 0.95]) == (0.05, 0.95)

    def test_median_only_is_none(self):
        assert _band_levels([0.5]) is None

    def test_empty_is_none(self):
        assert _band_levels([]) is None
        assert _band_levels(None) is None

    def test_one_sided_is_none(self):
        assert _band_levels([0.5, 0.9]) is None


class TestBacktestProbabilisticPayload:
    def test_absent_for_point_model(
        self, metadata, backtest_result, windows_raw, actual_series
    ):
        payload = _build(metadata, backtest_result, windows_raw, actual_series, None)
        assert payload.probabilistic == {}
        assert "lower" not in payload.windows[0]["steps"][0]

    def test_horizon_coverage_preserved(
        self, metadata, backtest_result, windows_raw, actual_series
    ):
        payload = _build(
            metadata, backtest_result, windows_raw, actual_series,
            _probabilistic_block(windows_raw),
        )
        hc = payload.probabilistic["horizon_coverage"]
        assert [r["h"] for r in hc] == [1, 2, 3]
        # The degradation must survive into the payload intact — this is the
        # signal the pooled number hides.
        assert hc[0]["coverage"] == 0.97
        assert hc[-1]["coverage"] == 0.62
        assert hc[0]["coverage"] > hc[-1]["coverage"]

    def test_bands_zipped_onto_window_steps(
        self, metadata, backtest_result, windows_raw, actual_series
    ):
        payload = _build(
            metadata, backtest_result, windows_raw, actual_series,
            _probabilistic_block(windows_raw),
        )
        assert payload.probabilistic["has_band"] is True
        assert payload.probabilistic["n_banded_steps"] == N_WINDOWS * HORIZON
        for win in payload.windows:
            for step in win["steps"]:
                assert step["lower"] == 95.0
                assert step["upper"] == 105.0

    def test_mismatched_band_timestamps_are_skipped(
        self, metadata, backtest_result, windows_raw, actual_series
    ):
        prob = _probabilistic_block(windows_raw)
        # Shift every band timestamp so nothing aligns.
        for wb in prob["window_bands"]:
            for row in wb:
                row["t"] = row["t"].replace("2024-01", "2030-01")
        payload = _build(metadata, backtest_result, windows_raw, actual_series, prob)
        assert payload.probabilistic["n_banded_steps"] == 0
        assert payload.probabilistic["has_band"] is False
        assert "lower" not in payload.windows[0]["steps"][0]

    def test_html_contains_interval_section(
        self, metadata, backtest_result, windows_raw, actual_series
    ):
        payload = _build(
            metadata, backtest_result, windows_raw, actual_series,
            _probabilistic_block(windows_raw),
        )
        html = build_backtest_report_html(payload)
        assert 'id="probabilistic"' in html
        assert "chart-horizon-coverage" in html
        assert "chart-horizon-width" in html
        assert "chart-bt-pinball" in html
