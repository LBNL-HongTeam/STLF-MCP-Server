"""
Unit tests for probabilistic (quantile) forecasting building blocks:

  * ``validate_quantiles`` normalisation / validation
  * ``create_model`` likelihood wiring for supported / unsupported models
  * ``calculate_probabilistic_metrics`` pinball / coverage / width maths
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.trainer import (
    validate_quantiles,
    create_model,
    DEFAULT_QUANTILES,
    PROBABILISTIC_MODELS,
)
from load_forecasting.core.evaluator import calculate_probabilistic_metrics


# ---------------------------------------------------------------------------
# validate_quantiles
# ---------------------------------------------------------------------------

class TestValidateQuantiles:
    def test_default_when_none(self):
        assert validate_quantiles(None) == list(DEFAULT_QUANTILES)

    def test_median_always_included(self):
        assert 0.5 in validate_quantiles([0.1, 0.9])

    def test_sorted_and_deduped(self):
        assert validate_quantiles([0.9, 0.1, 0.1, 0.5]) == [0.1, 0.5, 0.9]

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            validate_quantiles([0.0, 0.5])
        with pytest.raises(ValueError):
            validate_quantiles([0.5, 1.0])

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_quantiles([])

    def test_rejects_non_numeric(self):
        with pytest.raises(ValueError):
            validate_quantiles(["a", "b"])


# ---------------------------------------------------------------------------
# create_model likelihood wiring
# ---------------------------------------------------------------------------

class TestProbabilisticModelCreation:
    @pytest.mark.parametrize("model_type", sorted(PROBABILISTIC_MODELS))
    def test_supported_models_become_probabilistic(self, model_type):
        m = create_model(
            model_type, lookback=12, horizon=6,
            probabilistic=True, quantiles=[0.1, 0.9],
            accelerator="cpu",
        )
        assert m.supports_probabilistic_prediction is True

    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError):
            create_model("LinearRegression", probabilistic=True)

    def test_point_model_unchanged(self):
        # A non-probabilistic XGBoost is still deterministic.
        m = create_model("XGBoost", lookback=12, horizon=6)
        assert m.supports_probabilistic_prediction is False


# ---------------------------------------------------------------------------
# calculate_probabilistic_metrics
# ---------------------------------------------------------------------------

def _ts(values, start="2023-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="h")
    return TimeSeries.from_times_and_values(idx, np.asarray(values, dtype=float))


class TestProbabilisticMetrics:
    def test_perfect_median_zero_pinball_component(self):
        actual = _ts([10, 20, 30, 40])
        # Median exactly equals actual → its pinball contribution is 0.
        q = {0.5: _ts([10, 20, 30, 40])}
        res = calculate_probabilistic_metrics(actual, q)
        assert res["per_quantile_pinball"]["0.5"] == 0.0
        assert res["pinball_loss"] == 0.0

    def test_full_coverage(self):
        actual = _ts([10, 20, 30, 40])
        q = {
            0.1: _ts([0, 0, 0, 0]),
            0.5: _ts([10, 20, 30, 40]),
            0.9: _ts([100, 100, 100, 100]),
        }
        res = calculate_probabilistic_metrics(actual, q)
        assert res["coverage"] == 1.0
        assert res["nominal_coverage"] == pytest.approx(0.8)
        assert res["mean_interval_width"] == pytest.approx(100.0)

    def test_zero_coverage_when_band_misses(self):
        actual = _ts([10, 20, 30, 40])
        q = {
            0.1: _ts([100, 100, 100, 100]),
            0.9: _ts([200, 200, 200, 200]),
        }
        res = calculate_probabilistic_metrics(actual, q)
        assert res["coverage"] == 0.0

    def test_empty_returns_nulls(self):
        actual = _ts([1, 2, 3])
        res = calculate_probabilistic_metrics(actual, {})
        assert res["pinball_loss"] is None
        assert res["coverage"] is None


# ---------------------------------------------------------------------------
# _attach_quantiles_to_predictions — merging quantile series onto pred rows
# ---------------------------------------------------------------------------

from load_forecasting.tools.evaluation import _attach_quantiles_to_predictions


def _ts(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="h")
    return TimeSeries.from_times_and_values(idx, np.array(values, dtype=float))


class TestAttachQuantilesToPredictions:
    def _rows(self, n=3, start="2024-01-01"):
        idx = pd.date_range(start, periods=n, freq="h")
        return [
            {"timestamp": t.isoformat(), "actual": 10.0, "predicted": 9.0, "residual": 1.0}
            for t in idx
        ]

    def test_adds_quantile_keys(self):
        rows = self._rows(3)
        _attach_quantiles_to_predictions(
            rows, {0.1: _ts([1, 2, 3]), 0.9: _ts([7, 8, 9])}, scaler=None
        )
        assert [r["q0.1"] for r in rows] == [1.0, 2.0, 3.0]
        assert [r["q0.9"] for r in rows] == [7.0, 8.0, 9.0]

    def test_partial_overlap_leaves_other_rows_untouched(self):
        rows = self._rows(3)
        # Quantile series starts one hour later -> only 2 rows can be matched.
        _attach_quantiles_to_predictions(
            rows, {0.1: _ts([5, 6], start="2024-01-01 01:00")}, scaler=None
        )
        assert "q0.1" not in rows[0]
        assert rows[1]["q0.1"] == 5.0
        assert rows[2]["q0.1"] == 6.0

    def test_no_overlap_is_a_noop(self):
        rows = self._rows(2)
        _attach_quantiles_to_predictions(
            rows, {0.1: _ts([1, 2], start="2030-01-01")}, scaler=None
        )
        assert all("q0.1" not in r for r in rows)

    def test_bad_series_is_swallowed(self):
        rows = self._rows(2)
        # A non-TimeSeries value must not blow up the whole evaluation.
        _attach_quantiles_to_predictions(rows, {0.1: object()}, scaler=None)
        assert all("q0.1" not in r for r in rows)


# ---------------------------------------------------------------------------
# calculate_horizon_coverage — per-h-step interval calibration
# ---------------------------------------------------------------------------

from load_forecasting.core.evaluator import calculate_horizon_coverage


def _stochastic_window(start, samples):
    """Build a stochastic TimeSeries from a (n_times, n_samples) array."""
    arr = np.asarray(samples, dtype=float)
    idx = pd.date_range(start, periods=arr.shape[0], freq="h")
    return TimeSeries.from_times_and_values(idx, arr.reshape(arr.shape[0], 1, arr.shape[1]))


class TestHorizonCoverage:
    def test_empty_windows(self):
        assert calculate_horizon_coverage([], _ts([1, 2, 3]), 0.1, 0.9) == []

    def test_one_row_per_horizon_step(self):
        # 2 windows x 3 horizon steps, 100 samples each.
        rng = np.random.default_rng(0)
        w1 = _stochastic_window("2024-01-01", rng.normal(10, 1, (3, 100)))
        w2 = _stochastic_window("2024-01-01 03:00", rng.normal(10, 1, (3, 100)))
        actual = _ts([10.0] * 6)
        out = calculate_horizon_coverage([w1, w2], actual, 0.1, 0.9)
        assert [r["h"] for r in out] == [1, 2, 3]
        assert all(r["n"] == 2 for r in out)

    def test_detects_degrading_coverage(self):
        # h=1 band is wide (always covers); h=2 band is narrow (never covers).
        # Pooled coverage would be a meaningless 50%.
        wide = np.tile(np.linspace(0.0, 20.0, 100), (1, 1))     # spans actual=10
        narrow = np.tile(np.linspace(99.0, 100.0, 100), (1, 1))  # misses actual=10
        windows = []
        for k in range(4):
            samples = np.vstack([wide, narrow])
            windows.append(_stochastic_window(f"2024-01-0{k+1}", samples))
        actual_idx = pd.date_range("2024-01-01", periods=96, freq="h")
        actual = TimeSeries.from_times_and_values(
            actual_idx, np.full(len(actual_idx), 10.0)
        )
        out = calculate_horizon_coverage(windows, actual, 0.1, 0.9)
        assert out[0]["coverage"] == 1.0, "h=1 band should cover"
        assert out[1]["coverage"] == 0.0, "h=2 band should miss"
        # Width must reflect the difference in sharpness.
        assert out[0]["mean_interval_width"] > out[1]["mean_interval_width"]

    def test_insufficient_pairs_returns_none(self):
        w = _stochastic_window("2024-01-01", np.random.default_rng(1).normal(10, 1, (2, 50)))
        # Actual covers only the first timestamp -> h=2 has n=0.
        actual = _ts([10.0])
        out = calculate_horizon_coverage([w], actual, 0.1, 0.9)
        assert out[1]["coverage"] is None
        assert out[1]["n"] == 0

    def test_no_timestamp_overlap(self):
        w = _stochastic_window("2030-01-01", np.random.default_rng(2).normal(10, 1, (3, 50)))
        out = calculate_horizon_coverage([w], _ts([10.0] * 3), 0.1, 0.9)
        assert all(r["coverage"] is None for r in out)
