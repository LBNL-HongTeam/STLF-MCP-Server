"""Unit tests for the TimesFMResidualHybrid class.

These tests exercise the covariate-feature-engineering, Ridge fitting, and
save/load logic *without* invoking the real TimesFM 2.5 backbone (which
requires an ~800MB HuggingFace download).  A dummy TimesFM stand-in is
injected via monkeypatch.  Real end-to-end training against the actual
foundation model is covered in a separate opt-in integration test.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from darts import TimeSeries

from load_forecasting.core import hybrid_timesfm as hyb


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


class _DummyTimesFM:
    """Minimal stand-in for TimesFM2p5Model used only in unit tests.

    ``fit`` records the training length; ``predict`` returns the mean of the
    training series repeated ``n`` times so residuals are non-trivial when the
    real signal has structure.  ``historical_forecasts`` runs a rudimentary
    rolling forecast so the hybrid's residual pass has data to fit on.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.input_chunk_length = kwargs.get("input_chunk_length", 24)
        self.output_chunk_length = kwargs.get("output_chunk_length", 6)
        self._train_series = None
        self.train_sample = (np.zeros(1, dtype=np.float32),)

    def fit(self, series):
        self._train_series = series
        return self

    def predict(self, n, series=None):
        if series is None:
            series = self._train_series
        last_time = series.time_index[-1]
        freq = series.freq or pd.Timedelta(hours=1)
        times = pd.date_range(last_time + freq, periods=n, freq=freq)
        # Simple mean-repeat prediction, with a tiny drift so it's not perfect
        mean_val = float(series.values().mean())
        vals = np.full(n, mean_val, dtype=np.float64)
        return TimeSeries.from_times_and_values(times, vals.reshape(-1, 1))

    def historical_forecasts(
        self,
        series,
        start,
        forecast_horizon,
        stride,
        retrain,
        last_points_only,
        verbose,
    ):
        preds_times: list[pd.Timestamp] = []
        preds_vals: list[float] = []
        idx = int(start)
        n = len(series)
        while idx + forecast_horizon <= n:
            window = series[:idx]
            future = self.predict(forecast_horizon, series=window)
            if last_points_only:
                preds_times.append(future.time_index[-1])
                preds_vals.append(float(future.values()[-1, 0]))
            else:
                for t, v in zip(future.time_index, future.values().flatten()):
                    preds_times.append(t)
                    preds_vals.append(float(v))
            idx += stride
        if not preds_times:
            return None
        return TimeSeries.from_times_and_values(
            pd.DatetimeIndex(preds_times),
            np.asarray(preds_vals).reshape(-1, 1),
        )

    def save(self, path):
        # Write a tiny JSON blob capturing kwargs
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "DummyTimesFM", "kwargs": self.kwargs}, f)

    @classmethod
    def load(cls, path, weights_only=False):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload.get("kwargs", {}))


@pytest.fixture(autouse=True)
def _patch_timesfm(monkeypatch):
    """Swap in the dummy TimesFM for the duration of each unit test."""
    monkeypatch.setattr(hyb, "TimesFM2p5Model", _DummyTimesFM)
    monkeypatch.setattr(hyb, "_HAS_TIMESFM", True)
    yield


def _make_series(n: int = 400) -> tuple[TimeSeries, TimeSeries, TimeSeries]:
    """Build (target, past_covariates, future_covariates) of length n."""
    times = pd.date_range("2023-01-01", periods=n, freq="h")
    # Target: daily seasonality + temperature-driven quadratic effect
    temp = 15 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24)
    load = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + 0.5 * (temp - 20) ** 2
    target = TimeSeries.from_times_and_values(times, load.reshape(-1, 1))
    # Past covariates: humidity, solar (unrelated noise)
    rh = 50 + 10 * np.random.default_rng(0).normal(size=n)
    solar = np.clip(500 * np.sin(np.arange(n) * 2 * np.pi / 24), 0, None)
    past = TimeSeries.from_times_and_values(
        times,
        np.column_stack([rh, solar]),
        columns=["humidity", "solar_radiation"],
    )
    # Future covariates: temperature (the driver) + a cyclic calendar feature
    hour_sin = np.sin(np.arange(n) * 2 * np.pi / 24)
    fut = TimeSeries.from_times_and_values(
        times,
        np.column_stack([temp, hour_sin]),
        columns=["temperature", "cal_hour_sin"],
    )
    return target, past, fut


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTemperatureDetection:
    def test_exact_match(self):
        assert hyb._detect_temperature_column(["humidity", "temperature"]) == "temperature"

    def test_substring_match_prefers_temp(self):
        cols = ["outdoor_temp", "humidity"]
        assert hyb._detect_temperature_column(cols) == "outdoor_temp"

    def test_ignores_calendar_columns(self):
        cols = ["cal_temp_sin", "temperature"]
        assert hyb._detect_temperature_column(cols) == "temperature"

    def test_none_when_absent(self):
        assert hyb._detect_temperature_column(["humidity", "wind"]) is None


class TestFeatureEngineering:
    def test_hdd_cdd_shape(self):
        temps = np.array([0.0, 18.0, 22.0, 30.0])
        knots = np.array([10.0, 20.0, 28.0])
        feats, names = hyb._build_temperature_features(temps, knots, 18.0, 22.0)
        assert feats.shape == (4, 2 + 3)
        assert names[:2] == ["temp_hdd", "temp_cdd"]
        # HDD at 0°C = 18, HDD at 22°C = 0
        assert feats[0, 0] == pytest.approx(18.0)
        assert feats[2, 0] == pytest.approx(0.0)
        # CDD at 30°C = 8
        assert feats[3, 1] == pytest.approx(8.0)

    def test_spline_basis_nonnegative(self):
        x = np.linspace(-5, 40, 50)
        knots = np.array([10.0, 20.0, 30.0])
        basis = hyb._spline_basis(x, knots)
        assert basis.shape == (50, 3)
        assert (basis >= 0).all()


class TestHybridFitPredict:
    def test_fit_selects_temperature_column(self):
        target, past, fut = _make_series(400)
        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        model.fit(target, past_covariates=past, future_covariates=fut)
        assert model.temperature_col == "temperature"
        assert model.knots is not None
        assert len(model.feature_names) > 0

    def test_predict_shapes(self):
        target, past, fut = _make_series(400)
        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        model.fit(target, past_covariates=past, future_covariates=fut)
        pred = model.predict(
            n=6,
            series=target[:300],
            past_covariates=past,
            future_covariates=fut,
        )
        assert len(pred) == 6
        assert pred.values().shape[-1] == 1

    def test_predict_without_covariates_still_works(self):
        """If model was fit without covariates, predict returns base TimesFM."""
        target, _, _ = _make_series(400)
        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        model.fit(target)  # no covariates
        pred = model.predict(n=6, series=target[:300])
        assert len(pred) == 6

    def test_residual_correction_changes_prediction(self):
        target, past, fut = _make_series(400)
        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        model.fit(target, past_covariates=past, future_covariates=fut)

        base_only = model.timesfm.predict(n=6, series=target[:300])
        corrected = model.predict(
            n=6,
            series=target[:300],
            past_covariates=past,
            future_covariates=fut,
        )
        # If Ridge learned any signal at all, corrected != base_only.
        # Allow the "identical" case only when Ridge coefficients are all zero.
        diff = np.abs(corrected.values() - base_only.values()).sum()
        if np.allclose(model.ridge.coef_, 0):
            assert diff == pytest.approx(0.0)
        else:
            assert diff > 0.0


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        target, past, fut = _make_series(400)
        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        model.fit(target, past_covariates=past, future_covariates=fut)

        save_path = tmp_path / "hybrid_model.pkl"
        model.save(str(save_path))

        # Sentinel + sibling files present
        assert save_path.exists()
        assert (save_path.parent / (save_path.name + ".timesfm.ckpt")).exists()
        assert (save_path.parent / (save_path.name + ".ridge.joblib")).exists()
        assert (save_path.parent / (save_path.name + ".meta.json")).exists()

        loaded = hyb.TimesFMResidualHybrid.load(str(save_path))
        assert loaded.temperature_col == model.temperature_col
        assert loaded.feature_names == model.feature_names
        np.testing.assert_allclose(loaded.knots, model.knots)
        np.testing.assert_allclose(loaded.ridge.coef_, model.ridge.coef_)

        # Predictions match
        p1 = model.predict(
            n=6, series=target[:300], past_covariates=past, future_covariates=fut
        )
        p2 = loaded.predict(
            n=6, series=target[:300], past_covariates=past, future_covariates=fut
        )
        np.testing.assert_allclose(p1.values(), p2.values(), rtol=1e-6)

    def test_save_before_fit_raises(self, tmp_path):
        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "no_fit.pkl"))


class TestCovariateAlignment:
    def test_missing_temperature_column_at_predict(self):
        """A covariate column present at fit time but missing at predict
        should not crash — the feature engineering emits zeros in its place."""
        target, past, fut = _make_series(400)
        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        model.fit(target, past_covariates=past, future_covariates=fut)

        # Drop temperature column from future covariates
        fut_no_temp = fut.drop_columns(["temperature"])
        pred = model.predict(
            n=6,
            series=target[:300],
            past_covariates=past,
            future_covariates=fut_no_temp,
        )
        assert len(pred) == 6
