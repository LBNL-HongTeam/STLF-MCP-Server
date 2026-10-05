"""Tests for model_registry module."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from darts import TimeSeries

from load_forecasting.core.model_registry import (
    ModelRegistry,
    ModelNotFoundError,
)
from load_forecasting.core import hybrid_timesfm as hyb


class _DummyTimesFM:
    """Minimal TimesFM stand-in mirroring the one in test_hybrid_timesfm.py."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.input_chunk_length = kwargs.get("input_chunk_length", 24)
        self.output_chunk_length = kwargs.get("output_chunk_length", 6)
        self._train_series = None
        self.train_sample = (np.zeros(1, dtype=np.float32),)

    def fit(self, series):
        self._train_series = series

    def predict(self, n, series=None):
        if series is None:
            series = self._train_series
        last_time = series.time_index[-1]
        freq = series.freq or pd.Timedelta(hours=1)
        times = pd.date_range(last_time + freq, periods=n, freq=freq)
        mean_val = float(series.values().mean())
        return TimeSeries.from_times_and_values(
            times, np.full(n, mean_val).reshape(-1, 1)
        )

    def historical_forecasts(self, series, start, forecast_horizon, stride,
                             retrain, last_points_only, verbose):
        preds_times, preds_vals = [], []
        idx = int(start)
        while idx + forecast_horizon <= len(series):
            fut = self.predict(forecast_horizon, series=series[:idx])
            preds_times.append(fut.time_index[-1])
            preds_vals.append(float(fut.values()[-1, 0]))
            idx += stride
        if not preds_times:
            return None
        return TimeSeries.from_times_and_values(
            pd.DatetimeIndex(preds_times),
            np.asarray(preds_vals).reshape(-1, 1),
        )

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"kind": "DummyTimesFM", "kwargs": self.kwargs}, f)

    @classmethod
    def load(cls, path, weights_only=False):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(**payload.get("kwargs", {}))


@pytest.fixture
def temp_registry():
    """Create a temporary registry for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(base_dir=tmpdir)
        yield registry


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_generate_model_id(self, temp_registry):
        """Test model ID generation."""
        model_id = temp_registry.generate_model_id("Building_33", "LinearRegression")

        assert "Building_33" in model_id
        assert "LinearRegression" in model_id
        assert len(model_id.split("_")) >= 4  # building_model_date_time

    def test_generate_model_id_no_building(self, temp_registry):
        """Test model ID generation without building name."""
        model_id = temp_registry.generate_model_id(None, "NaiveMean")

        assert "unnamed" in model_id
        assert "NaiveMean" in model_id

    def test_list_models_empty(self, temp_registry):
        """Test listing models from empty registry."""
        models, count = temp_registry.list_models()

        assert models == []
        assert count == 0

    def test_model_not_found(self, temp_registry):
        """Test loading non-existent model."""
        with pytest.raises(ModelNotFoundError):
            temp_registry.load_model("nonexistent_model_id")

    def test_init_registry(self, temp_registry):
        """Test registry initialization."""
        assert temp_registry.registry_path.exists()

        registry = temp_registry._load_registry()
        assert "version" in registry
        assert "models" in registry
        assert registry["models"] == []

    def test_training_info_persisted_to_metadata(self, temp_registry):
        """training_info (incl. training_history) is written into metadata.json."""
        from darts.models import NaiveMean

        n = 200
        times = pd.date_range("2023-01-01", periods=n, freq="h")
        series = TimeSeries.from_times_and_values(
            times, np.linspace(10, 20, n).reshape(-1, 1)
        )
        model = NaiveMean()
        model.fit(series)

        training_info = {
            "training_time_seconds": 4.2,
            "energy_kwh": 0.01,
            "training_history": {
                "epochs": [0, 1, 2],
                "train_loss": [0.9, 0.5, 0.3],
                "val_loss": [1.0, 0.6, 0.45],
            },
        }
        model_id = "unnamed_NaiveMean_traininfo_test"
        temp_registry.save_model(
            model=model,
            model_id=model_id,
            model_type="NaiveMean",
            building_name=None,
            config={"lookback_hours": 24, "horizon_hours": 6, "frequency": "h"},
            column_mapping={"target": "load"},
            data_info={"total_samples": n},
            training_metrics={},
            validation_metrics={"cv_rmse": None},
            training_info=training_info,
        )

        meta_path = Path(temp_registry.base_dir) / model_id / "metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["training_info"]["training_history"]["epochs"] == [0, 1, 2]
        assert meta["training_info"]["training_history"]["train_loss"] == [0.9, 0.5, 0.3]

        _, loaded_meta, _ = temp_registry.load_model(model_id)
        assert loaded_meta["training_info"]["energy_kwh"] == 0.01

    def test_training_info_defaults_to_empty(self, temp_registry):
        """Omitting training_info leaves an empty dict (backward compatible)."""
        from darts.models import NaiveMean

        n = 200
        times = pd.date_range("2023-01-01", periods=n, freq="h")
        series = TimeSeries.from_times_and_values(
            times, np.linspace(10, 20, n).reshape(-1, 1)
        )
        model = NaiveMean()
        model.fit(series)

        model_id = "unnamed_NaiveMean_noinfo_test"
        temp_registry.save_model(
            model=model,
            model_id=model_id,
            model_type="NaiveMean",
            building_name=None,
            config={"lookback_hours": 24, "horizon_hours": 6, "frequency": "h"},
            column_mapping={"target": "load"},
            data_info={"total_samples": n},
            training_metrics={},
            validation_metrics={"cv_rmse": None},
        )
        _, meta, _ = temp_registry.load_model(model_id)
        assert meta["training_info"] == {}


class TestTimesFMResidualRoundtrip:
    """Save/load a TimesFM+Residual hybrid through the registry."""

    def test_roundtrip(self, temp_registry, monkeypatch):
        # Patch TimesFM at both the hybrid module and the registry-imported
        # module reference so the load path picks up the dummy too.
        monkeypatch.setattr(hyb, "TimesFM2p5Model", _DummyTimesFM)
        monkeypatch.setattr(hyb, "_HAS_TIMESFM", True)
        # The registry checks _HAS_TIMESFM through its own module import
        from load_forecasting.core import model_registry as mr
        monkeypatch.setattr(mr, "_HAS_TIMESFM", True)

        n = 400
        times = pd.date_range("2023-01-01", periods=n, freq="h")
        temp = 15 + 10 * np.sin(np.arange(n) * 2 * np.pi / 24)
        load = 100 + 20 * np.sin(np.arange(n) * 2 * np.pi / 24) + 0.5 * (temp - 20) ** 2
        target = TimeSeries.from_times_and_values(times, load.reshape(-1, 1))
        fut = TimeSeries.from_times_and_values(
            times, temp.reshape(-1, 1), columns=["temperature"],
        )

        model = hyb.TimesFMResidualHybrid(
            timesfm_kwargs={"input_chunk_length": 48, "output_chunk_length": 6},
            horizon=6,
        )
        model.fit(target, future_covariates=fut)

        model_id = "unnamed_TimesFM+Residual_test"
        temp_registry.save_model(
            model=model,
            model_id=model_id,
            model_type="TimesFM+Residual",
            building_name=None,
            config={"lookback_hours": 48, "horizon_hours": 6, "frequency": "h"},
            column_mapping={"target": "load", "future_covariates": ["temperature"]},
            data_info={"total_samples": n},
            training_metrics={},
            validation_metrics={"cv_rmse": None},
        )

        loaded_model, metadata, scalers = temp_registry.load_model(model_id)
        assert metadata["model_type"] == "TimesFM+Residual"
        assert loaded_model.temperature_col == "temperature"

        p1 = model.predict(n=6, series=target[:300], future_covariates=fut)
        p2 = loaded_model.predict(n=6, series=target[:300], future_covariates=fut)
        np.testing.assert_allclose(p1.values(), p2.values(), rtol=1e-6)
