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
