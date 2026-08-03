"""Integration tests for MCP tools."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import asyncio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.tools import (
    train_forecast_model,
    evaluate_forecast_model,
    list_models,
)
from load_forecasting.core.model_registry import ModelRegistry


@pytest.fixture
def sample_hourly_csv():
    """Create sample hourly data for testing."""
    # Need enough data for training: lookback + horizon + validation
    n_points = 500  # About 3 weeks of hourly data

    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_points, freq="h"),
        "electricity_kwh": [
            100 + 20 * (i % 24) / 24 + 5 * (i % 168) / 168
            for i in range(n_points)
        ],
        "outdoor_temp": [50 + 10 * (i % 24) / 24 for i in range(n_points)],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    Path(f.name).unlink()


@pytest.fixture
def temp_model_dir(monkeypatch):
    """Set up temporary model directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


class TestTrainForecastModel:
    """Tests for train_forecast_model tool."""

    @pytest.mark.asyncio
    async def test_train_linear_regression(self, sample_hourly_csv, temp_model_dir):
        """Test training a linear regression model."""
        result = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_building",
        )

        assert result["success"] is True
        assert result["model_id"] is not None
        assert "test_building" in result["model_id"]
        assert result["model_type"] == "LinearRegression"
        assert "validation_metrics" in result
        assert result["validation_metrics"]["cv_rmse"] is not None

    @pytest.mark.asyncio
    async def test_train_naive_mean(self, sample_hourly_csv, temp_model_dir):
        """Test training a naive mean model."""
        result = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveMean",
            building_name="test_building",
        )

        assert result["success"] is True
        assert result["model_type"] == "NaiveMean"

    @pytest.mark.asyncio
    async def test_train_invalid_model_type(self, sample_hourly_csv, temp_model_dir):
        """Test error on invalid model type."""
        result = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="InvalidModel",
        )

        assert result["success"] is False
        assert "Unknown model type" in result["error"]

    @pytest.mark.asyncio
    async def test_train_file_not_found(self, temp_model_dir):
        """Test error on missing file."""
        result = train_forecast_model(
            csv_path="/nonexistent/file.csv",
        )

        assert result["success"] is False
        assert "not found" in result["error"]


class TestListModels:
    """Tests for list_models tool."""

    @pytest.mark.asyncio
    async def test_list_empty(self, temp_model_dir):
        """Test listing empty registry."""
        result = list_models()

        assert result["success"] is True
        assert result["models"] == []
        assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_list_after_training(self, sample_hourly_csv, temp_model_dir):
        """Test listing after training a model."""
        # Train a model first
        train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveMean",
            building_name="test_building",
        )

        # List models
        result = list_models()

        assert result["success"] is True
        assert result["total_count"] == 1
        assert len(result["models"]) == 1
        assert result["models"][0]["building_name"] == "test_building"


# ---------------------------------------------------------------------------
# Fixtures for future covariate tests
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv_with_future_covariates():
    """Hourly CSV with a past covariate and an explicit future covariate."""
    n_points = 500
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_points, freq="h"),
        "electricity_kwh": [
            100 + 20 * (i % 24) / 24 + 5 * (i % 168) / 168
            for i in range(n_points)
        ],
        "outdoor_temp": [50 + 10 * (i % 24) / 24 for i in range(n_points)],
        # Simulate a weather forecast column — available for future windows
        "temp_forecast": [51 + 10 * (i % 24) / 24 for i in range(n_points)],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    Path(f.name).unlink()


class TestFutureCovariatesIntegration:
    """Integration tests for future covariates support."""

    @pytest.mark.asyncio
    async def test_train_with_future_covariates(
        self, sample_csv_with_future_covariates, temp_model_dir
    ):
        """Train LinearRegression with explicit future covariates."""
        result = train_forecast_model(
            csv_path=sample_csv_with_future_covariates,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_building",
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "past_covariates": ["outdoor_temp"],
                "future_covariates": ["temp_forecast"],
            },
        )

        assert result["success"] is True, result.get("error")
        assert result["model_type"] == "LinearRegression"
        # Future covariate columns should appear in the data summary
        assert "future_covariate_columns" in result["data_summary"]
        assert "temp_forecast" in result["data_summary"]["future_covariate_columns"]
        assert result["validation_metrics"]["cv_rmse"] is not None

    @pytest.mark.asyncio
    async def test_future_covariate_scaler_persisted(
        self, sample_csv_with_future_covariates, temp_model_dir
    ):
        """future_covariate_scaler must be saved to the model registry."""
        result = train_forecast_model(
            csv_path=sample_csv_with_future_covariates,
            model_type="LinearRegression",
            building_name="test_building",
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "past_covariates": ["outdoor_temp"],
                "future_covariates": ["temp_forecast"],
            },
        )
        assert result["success"] is True, result.get("error")

        # Load the saved scalers via the registry
        registry = ModelRegistry()
        _, _, scalers = registry.load_model(result["model_id"])
        assert "future_covariate_scaler" in scalers
        assert scalers["future_covariate_scaler"] is not None

    @pytest.mark.asyncio
    async def test_evaluate_with_future_covariates(
        self, sample_csv_with_future_covariates, temp_model_dir
    ):
        """evaluate_forecast_model should work when the model was trained with future covariates."""
        column_mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": ["outdoor_temp"],
            "future_covariates": ["temp_forecast"],
        }
        train_result = train_forecast_model(
            csv_path=sample_csv_with_future_covariates,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_building",
            column_mapping=column_mapping,
        )
        assert train_result["success"] is True, train_result.get("error")

        eval_result = evaluate_forecast_model(
            model_id=train_result["model_id"],
            csv_path=sample_csv_with_future_covariates,
        )
        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["rmse"] is not None

    @pytest.mark.asyncio
    async def test_train_without_future_covariates_unchanged(
        self, sample_hourly_csv, temp_model_dir
    ):
        """Training without future covariates still works (no regressions)."""
        result = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_building",
        )
        assert result["success"] is True, result.get("error")
        assert result["data_summary"]["future_covariate_columns"] == []


# ---------------------------------------------------------------------------
# Pre-flight inspect_data gate tests
# ---------------------------------------------------------------------------

def _make_blocking_csv(tmp_path: Path) -> Path:
    """CSV whose datetime column cannot be detected → loader_error → blocking."""
    df = pd.DataFrame({"measurement_id": ["a", "b", "c"], "value": [1, 2, 3]})
    p = tmp_path / "bad_cols.csv"
    df.to_csv(p, index=False)
    return p


def _make_good_csv(tmp_path: Path, n: int = 500) -> Path:
    """Minimal valid CSV with 500 hourly rows (enough to train)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "load_kw":   [100.0 + i * 0.1 for i in range(n)],
    })
    p = tmp_path / "good.csv"
    df.to_csv(p, index=False)
    return p


class TestInspectDataGate:
    """inspect_data is called before training/tuning; blocking issues abort the run."""

    @pytest.mark.asyncio
    async def test_train_includes_data_inspection_on_success(
        self, sample_hourly_csv, temp_model_dir
    ):
        """Successful training response includes data_inspection key."""
        result = train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="gate_test",
        )
        assert result["success"] is True, result.get("error")
        assert "data_inspection" in result, "data_inspection missing from response"
        di = result["data_inspection"]
        assert "quality_flags" in di
        assert "suggestions" in di

    @pytest.mark.asyncio
    async def test_train_blocked_by_inspection_blocking_issue(self, tmp_path, temp_model_dir):
        """Train is blocked when inspect_data reports a blocking issue."""
        bad_csv = _make_blocking_csv(tmp_path)
        result = train_forecast_model(
            csv_path=str(bad_csv),
            model_type="LinearRegression",
        )
        assert result["success"] is False
        assert "blocking" in result["error"].lower() or "inspection" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_train_blocked_when_file_missing(self, temp_model_dir):
        """Train returns failure (via inspection gate) when CSV doesn't exist."""
        result = train_forecast_model(
            csv_path="/nonexistent/path/data.csv",
            model_type="LinearRegression",
        )
        assert result["success"] is False


# ---------------------------------------------------------------------------
# TimesFM+Residual hybrid — uses a dummy TimesFM backbone to avoid the ~800MB
# HuggingFace weights download.  This exercises the full tool-layer wiring:
# train_forecast_model -> ForecastingDataLoader -> trainer.create_model ->
# TimesFMResidualHybrid.fit -> _manual_walk_forward_with_covariates ->
# ModelRegistry.save_model -> ModelRegistry.load_model -> generate_forecast.
# ---------------------------------------------------------------------------

class _DummyTimesFM:
    """Tool-layer stand-in mirroring test_hybrid_timesfm._DummyTimesFM."""

    def __init__(self, **kwargs):
        import numpy as _np
        self.kwargs = kwargs
        self.input_chunk_length = kwargs.get("input_chunk_length", 24)
        self.output_chunk_length = kwargs.get("output_chunk_length", 6)
        self._train_series = None
        self.train_sample = (_np.zeros(1, dtype=_np.float32),)

    def fit(self, series):
        self._train_series = series

    def predict(self, n, series=None):
        import numpy as _np
        if series is None:
            series = self._train_series
        last_time = series.time_index[-1]
        freq = series.freq or pd.Timedelta(hours=1)
        times = pd.date_range(last_time + freq, periods=n, freq=freq)
        mean_val = float(series.values().mean())
        from darts import TimeSeries as _TS
        return _TS.from_times_and_values(
            times, _np.full(n, mean_val).reshape(-1, 1)
        )

    def historical_forecasts(self, series, start, forecast_horizon, stride,
                             retrain, last_points_only, verbose):
        import numpy as _np
        from darts import TimeSeries as _TS
        preds_times, preds_vals = [], []
        idx = int(start)
        while idx + forecast_horizon <= len(series):
            fut = self.predict(forecast_horizon, series=series[:idx])
            if last_points_only:
                preds_times.append(fut.time_index[-1])
                preds_vals.append(float(fut.values()[-1, 0]))
            else:
                for t, v in zip(fut.time_index, fut.values().flatten()):
                    preds_times.append(t)
                    preds_vals.append(float(v))
            idx += stride
        if not preds_times:
            return None
        return _TS.from_times_and_values(
            pd.DatetimeIndex(preds_times),
            _np.asarray(preds_vals).reshape(-1, 1),
        )

    def save(self, path):
        import json as _json
        with open(path, "w", encoding="utf-8") as f:
            _json.dump({"kwargs": self.kwargs}, f)

    @classmethod
    def load(cls, path, weights_only=False):
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            payload = _json.load(f)
        return cls(**payload.get("kwargs", {}))


@pytest.fixture
def _patch_timesfm_hybrid(monkeypatch):
    """Patch TimesFM at all relevant modules so tool-layer code picks up the dummy."""
    from load_forecasting.core import hybrid_timesfm as _hyb
    from load_forecasting.core import model_registry as _mr
    from load_forecasting.core import trainer as _trn

    monkeypatch.setattr(_hyb, "TimesFM2p5Model", _DummyTimesFM)
    monkeypatch.setattr(_hyb, "_HAS_TIMESFM", True)
    monkeypatch.setattr(_mr, "_HAS_TIMESFM", True)
    monkeypatch.setattr(_trn, "_HAS_TIMESFM", True)
    yield


class TestTimesFMResidualHybrid:
    """End-to-end tool-layer tests for TimesFM+Residual."""

    @pytest.mark.asyncio
    async def test_train_and_forecast(
        self,
        sample_csv_with_future_covariates,
        temp_model_dir,
        _patch_timesfm_hybrid,
    ):
        result = train_forecast_model(
            csv_path=sample_csv_with_future_covariates,
            model_type="TimesFM+Residual",
            lookback_hours=48,
            horizon_hours=6,
            building_name="hybrid_test",
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "past_covariates": ["outdoor_temp"],
                "future_covariates": ["temp_forecast"],
            },
        )
        assert result["success"] is True, result.get("error")
        assert result["model_type"] == "TimesFM+Residual"
        model_id = result["model_id"]

        # Registry round-trip: model loads with correct type
        registry = ModelRegistry()
        loaded_model, metadata, scalers = registry.load_model(model_id)
        assert metadata["model_type"] == "TimesFM+Residual"
        # Feature engineering picked up temperature
        assert loaded_model.temperature_col is not None

        # Generate a forecast: build a context CSV that includes horizon rows
        # with NaN target and valid future covariate values.
        from load_forecasting.tools import generate_forecast

        df = pd.read_csv(sample_csv_with_future_covariates)
        # Keep first 200 rows as context, then append 12 horizon rows with
        # NaN target but valid future covariates (past covariate may be NaN).
        context = df.iloc[:200].copy()
        horizon_rows = df.iloc[200:212].copy()
        horizon_rows["electricity_kwh"] = float("nan")
        horizon_rows["outdoor_temp"] = float("nan")
        combined = pd.concat([context, horizon_rows], ignore_index=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tf:
            combined.to_csv(tf.name, index=False)
            inference_csv = tf.name

        try:
            fresult = generate_forecast(
                model_id=model_id,
                csv_path=inference_csv,
                horizon_hours=6,
            )
            assert fresult["success"] is True, fresult.get("error")
            assert len(fresult["predictions"]) == 6
        finally:
            Path(inference_csv).unlink()
