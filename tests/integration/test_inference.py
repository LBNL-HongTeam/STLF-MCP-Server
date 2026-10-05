"""
Integration tests for generate_forecast and backtest_model MCP tools.

These tests train a real model using synthetic data and then run inference /
backtesting on that model. They write real model artefacts to a temporary
directory (monkeypatched via LOAD_FORECASTING_MODEL_DIR).
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.tools import (
    train_forecast_model,
    generate_forecast,
    backtest_model,
)


# ==============================================================================
# Shared fixtures
# ==============================================================================

def _make_synthetic_csv(n_hours: int = 600, with_future_cov: bool = False) -> str:
    """
    Create a temporary synthetic hourly CSV and return its path.

    The data has a realistic diurnal + weekly pattern so the models can
    actually learn something.
    """
    t = np.arange(n_hours)
    load = (
        100
        + 20 * np.sin(2 * np.pi * t / 24)       # diurnal
        + 5  * np.sin(2 * np.pi * t / 168)       # weekly
        + np.random.default_rng(42).normal(0, 2, n_hours)
    )
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=n_hours, freq="h"),
        "electricity_kwh": load,
        "outdoor_temp": 50 + 10 * np.sin(2 * np.pi * t / 24),
    }
    if with_future_cov:
        data["T_out_forecast"] = data["outdoor_temp"] + np.random.default_rng(7).normal(0, 0.5, n_hours)

    df = pd.DataFrame(data)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


@pytest.fixture
def sample_csv():
    """Synthetic hourly CSV without future covariates (600 h ≈ 25 days)."""
    path = _make_synthetic_csv(n_hours=600)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_with_future_cov():
    """Synthetic hourly CSV WITH a future-covariate column."""
    path = _make_synthetic_csv(n_hours=600, with_future_cov=True)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_dir(monkeypatch):
    """Isolated temporary model directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


# ==============================================================================
# TestGenerateForecast
# ==============================================================================

class TestGenerateForecast:
    """Tests for the generate_forecast MCP tool."""

    async def _train_lr(self, csv_path: str) -> str:
        """Helper: train a LinearRegression model and return model_id."""
        result = train_forecast_model(
            csv_path=csv_path,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_bldg",
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        return result["model_id"]

    async def _train_naive(self, csv_path: str, model_type: str = "NaiveSeasonal") -> str:
        result = train_forecast_model(
            csv_path=csv_path,
            model_type=model_type,
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_bldg",
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        return result["model_id"]

    async def test_basic_linear_regression(self, sample_csv, temp_model_dir):
        """Train a LinearRegression model and generate a forward forecast."""
        model_id = await self._train_lr(sample_csv)

        # Use the last 48 h of the CSV as context
        df = pd.read_csv(sample_csv)
        context_df = df.tail(48)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        context_df.to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is True, f"generate_forecast failed: {result.get('error')}"
        assert result["model_id"] == model_id
        assert result["model_type"] == "LinearRegression"
        assert result["forecast_horizon_hours"] == 6
        assert len(result["predictions"]) == 6
        # Each prediction should have datetime and predicted_load
        for pred in result["predictions"]:
            assert "datetime" in pred
            assert "predicted_load" in pred
            assert isinstance(pred["predicted_load"], float)
        assert result["forecast_start"] is not None
        assert result["forecast_end"] is not None

    async def test_horizon_override(self, sample_csv, temp_model_dir):
        """horizon_hours override should produce the requested number of steps."""
        model_id = await self._train_lr(sample_csv)

        df = pd.read_csv(sample_csv)
        context_df = df.tail(48)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        context_df.to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
                horizon_hours=3,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is True
        assert result["forecast_horizon_hours"] == 3
        assert len(result["predictions"]) == 3

    async def test_output_csv_path(self, sample_csv, temp_model_dir, tmp_path):
        """Predictions should be written to output_csv_path."""
        model_id = await self._train_lr(sample_csv)

        df = pd.read_csv(sample_csv)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(48).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        out_path = str(tmp_path / "forecast.csv")
        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
                output_csv_path=out_path,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is True
        assert Path(out_path).exists()
        saved = pd.read_csv(out_path)
        assert len(saved) == 6
        assert "predicted_load" in saved.columns

    async def test_naive_seasonal_refits_on_context(self, sample_csv, temp_model_dir):
        """
        NaiveSeasonal is a local model and must refit on the provided context.
        generate_forecast should succeed and return correct number of predictions.
        """
        model_id = await self._train_naive(sample_csv, "NaiveSeasonal")

        df = pd.read_csv(sample_csv)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(48).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is True, f"Expected success, got: {result.get('error')}"
        assert len(result["predictions"]) == 6

    async def test_naive_mean_refits_on_context(self, sample_csv, temp_model_dir):
        """NaiveMean also refits. Should succeed."""
        model_id = await self._train_naive(sample_csv, "NaiveMean")

        df = pd.read_csv(sample_csv)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(48).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is True
        assert len(result["predictions"]) == 6

    async def test_short_context_global_model_returns_error(self, sample_csv, temp_model_dir):
        """
        For GlobalForecastingModels, context shorter than lookback should
        return a clear error rather than a cryptic Darts exception.
        """
        model_id = await self._train_lr(sample_csv)

        # Provide only 5 rows — much less than lookback=24
        df = pd.read_csv(sample_csv)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(5).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is False
        assert "context" in result["error"].lower() or "lookback" in result["error"].lower()

    async def test_invalid_model_id_returns_error(self, sample_csv, temp_model_dir):
        """generate_forecast with a non-existent model_id should fail gracefully."""
        df = pd.read_csv(sample_csv)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(48).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id="nonexistent_model_id_xyz",
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_invalid_horizon_returns_error(self, sample_csv, temp_model_dir):
        """horizon_hours outside [1, 48] should return a validation error."""
        model_id = await self._train_lr(sample_csv)

        df = pd.read_csv(sample_csv)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(48).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
                horizon_hours=99,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is False
        assert "horizon_hours" in result["error"]

    async def test_context_warning_short_context(self, sample_csv, temp_model_dir):
        """
        Contexts shorter than 168 h should include a context_warning field
        but still succeed (lookback is met).
        """
        model_id = await self._train_lr(sample_csv)

        # Provide 30h — less than 168h but >= lookback=24h
        df = pd.read_csv(sample_csv)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(30).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is True
        assert "context_warning" in result

    async def test_future_covariates_extended_csv(self, sample_csv_with_future_cov, temp_model_dir):
        """
        Model trained with future covariates: context CSV must have
        future_cov columns extending horizon rows beyond last target.
        """
        # Train with T_out_forecast as future covariate
        result_train = train_forecast_model(
            csv_path=sample_csv_with_future_cov,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_bldg",
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "past_covariates": ["outdoor_temp"],
                "future_covariates": ["T_out_forecast"],
            },
        )
        assert result_train["success"], f"Training failed: {result_train.get('error')}"
        model_id = result_train["model_id"]

        # Build extended context CSV: 48 context rows + 6 future_cov-only rows
        df = pd.read_csv(sample_csv_with_future_cov)
        context_df = df.tail(48).copy()

        last_ts = pd.to_datetime(context_df["timestamp"].iloc[-1])
        horizon_rows = pd.DataFrame({
            "timestamp": pd.date_range(last_ts + pd.Timedelta(hours=1), periods=6, freq="h"),
            "electricity_kwh": [np.nan] * 6,           # target unknown
            "outdoor_temp": [np.nan] * 6,               # past cov unknown
            "T_out_forecast": [7.5, 7.0, 6.8, 6.5, 6.2, 6.0],
        })
        extended_df = pd.concat([context_df, horizon_rows], ignore_index=True)

        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        extended_df.to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is True, f"generate_forecast failed: {result.get('error')}"
        assert len(result["predictions"]) == 6

    async def test_future_covariates_missing_horizon_rows_returns_error(
        self, sample_csv_with_future_cov, temp_model_dir
    ):
        """
        Model trained with future covariates but context CSV has no horizon
        rows → should return a clear error for global models.
        """
        result_train = train_forecast_model(
            csv_path=sample_csv_with_future_cov,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_bldg",
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "past_covariates": ["outdoor_temp"],
                "future_covariates": ["T_out_forecast"],
            },
        )
        assert result_train["success"]
        model_id = result_train["model_id"]

        # Context CSV with NO horizon rows
        df = pd.read_csv(sample_csv_with_future_cov)
        ctx_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(48).to_csv(ctx_f.name, index=False)
        ctx_f.close()

        try:
            result = generate_forecast(
                model_id=model_id,
                csv_path=ctx_f.name,
            )
        finally:
            Path(ctx_f.name).unlink(missing_ok=True)

        assert result["success"] is False
        assert "future covariate" in result["error"].lower() or "horizon" in result["error"].lower()


# ==============================================================================
# TestBacktestModel
# ==============================================================================

class TestBacktestModel:
    """Tests for the backtest_model MCP tool."""

    async def _train_lr(self, csv_path: str) -> str:
        result = train_forecast_model(
            csv_path=csv_path,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_bldg",
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        return result["model_id"]

    async def test_basic_backtest(self, sample_csv, temp_model_dir):
        """Train then backtest on the same CSV — should return metrics."""
        model_id = await self._train_lr(sample_csv)

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
        )

        assert result["success"] is True, f"backtest_model failed: {result.get('error')}"
        assert result["model_id"] == model_id
        assert result["model_type"] == "LinearRegression"
        assert "backtest_metrics" in result
        metrics = result["backtest_metrics"]
        assert metrics["rmse"] is not None
        assert metrics["cv_rmse"] is not None
        assert "comparison_to_validation" in result
        assert "backtest_summary" in result
        summary = result["backtest_summary"]
        assert summary["n_windows"] > 0
        assert summary["stride_hours"] == 6  # default = training horizon

    async def test_backtest_returns_predictions(self, sample_csv, temp_model_dir):
        """return_predictions=True should include per-step predictions."""
        model_id = await self._train_lr(sample_csv)

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            return_predictions=True,
        )

        assert result["success"] is True
        assert "predictions" in result
        assert len(result["predictions"]) > 0
        pred = result["predictions"][0]
        assert "timestamp" in pred
        assert "actual" in pred
        assert "predicted" in pred
        assert "residual" in pred

    async def test_backtest_no_predictions(self, sample_csv, temp_model_dir):
        """return_predictions=False should omit the predictions list."""
        model_id = await self._train_lr(sample_csv)

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            return_predictions=False,
        )

        assert result["success"] is True
        assert "predictions" not in result

    async def test_backtest_residual_analysis(self, sample_csv, temp_model_dir):
        """include_residual_analysis=True should add residual_analysis key."""
        model_id = await self._train_lr(sample_csv)

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            include_residual_analysis=True,
        )

        assert result["success"] is True
        assert "residual_analysis" in result
        ra = result["residual_analysis"]
        assert "mean_residual" in ra
        assert "std_residual" in ra
        assert "autocorrelation_lag1" in ra

    async def test_backtest_custom_stride(self, sample_csv, temp_model_dir):
        """stride_hours=1 should produce more windows than stride_hours=6."""
        model_id = await self._train_lr(sample_csv)

        result_default = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            return_predictions=False,
        )
        result_stride1 = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            stride_hours=1,
            return_predictions=False,
        )

        assert result_default["success"] is True
        assert result_stride1["success"] is True
        # Stride-1 should yield more windows
        assert (
            result_stride1["backtest_summary"]["n_windows"]
            > result_default["backtest_summary"]["n_windows"]
        )
        assert result_stride1["backtest_summary"]["stride_hours"] == 1

    async def test_backtest_custom_start_fraction(self, sample_csv, temp_model_dir):
        """start_fraction=0.5 should start later → fewer windows."""
        model_id = await self._train_lr(sample_csv)

        result_02 = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            start_fraction=0.2,
            return_predictions=False,
        )
        result_05 = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            start_fraction=0.5,
            return_predictions=False,
        )

        assert result_02["success"] is True
        assert result_05["success"] is True
        assert (
            result_02["backtest_summary"]["n_windows"]
            > result_05["backtest_summary"]["n_windows"]
        )

    async def test_backtest_output_csv(self, sample_csv, temp_model_dir, tmp_path):
        """output_csv_path should write a CSV with prediction rows."""
        model_id = await self._train_lr(sample_csv)
        out_path = str(tmp_path / "backtest_preds.csv")

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            output_csv_path=out_path,
        )

        assert result["success"] is True
        assert Path(out_path).exists()
        saved = pd.read_csv(out_path)
        assert "timestamp" in saved.columns
        assert "actual" in saved.columns
        assert "predicted" in saved.columns
        assert len(saved) > 0

    async def test_backtest_invalid_model_id(self, sample_csv, temp_model_dir):
        """Non-existent model_id should return a clear error."""
        result = backtest_model(
            model_id="nonexistent_model_xyz",
            csv_path=sample_csv,
        )

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    async def test_backtest_invalid_stride(self, sample_csv, temp_model_dir):
        """stride_hours outside [1, 48] should return a validation error."""
        model_id = await self._train_lr(sample_csv)

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            stride_hours=100,
        )

        assert result["success"] is False
        assert "stride_hours" in result["error"]

    async def test_backtest_invalid_start_fraction(self, sample_csv, temp_model_dir):
        """start_fraction outside [0.1, 0.5] should return a validation error."""
        model_id = await self._train_lr(sample_csv)

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            start_fraction=0.9,
        )

        assert result["success"] is False
        assert "start_fraction" in result["error"]

    async def test_backtest_naive_seasonal(self, sample_csv, temp_model_dir):
        """NaiveSeasonal (local model) should also work with backtest_model."""
        result_train = train_forecast_model(
            csv_path=sample_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_bldg",
        )
        assert result_train["success"]
        model_id = result_train["model_id"]

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            return_predictions=False,
        )

        assert result["success"] is True
        assert result["backtest_metrics"]["rmse"] is not None

    async def test_backtest_comparison_to_validation(self, sample_csv, temp_model_dir):
        """comparison_to_validation should always be present with a status."""
        model_id = await self._train_lr(sample_csv)

        result = backtest_model(
            model_id=model_id,
            csv_path=sample_csv,
            return_predictions=False,
        )

        assert result["success"] is True
        comp = result["comparison_to_validation"]
        assert "cv_rmse_diff" in comp
        assert "performance_status" in comp
        assert comp["performance_status"] in ("similar", "degraded", "improved")
