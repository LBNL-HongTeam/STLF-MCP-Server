"""Integration tests for MCP tools."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import asyncio

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.tools.forecasting_tools import (
    train_forecast_model,
    evaluate_forecast_model,
    list_models,
)


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
        result = await train_forecast_model(
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
        result = await train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveMean",
            building_name="test_building",
        )

        assert result["success"] is True
        assert result["model_type"] == "NaiveMean"

    @pytest.mark.asyncio
    async def test_train_invalid_model_type(self, sample_hourly_csv, temp_model_dir):
        """Test error on invalid model type."""
        result = await train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="InvalidModel",
        )

        assert result["success"] is False
        assert "Unknown model type" in result["error"]

    @pytest.mark.asyncio
    async def test_train_file_not_found(self, temp_model_dir):
        """Test error on missing file."""
        result = await train_forecast_model(
            csv_path="/nonexistent/file.csv",
        )

        assert result["success"] is False
        assert "not found" in result["error"]


class TestListModels:
    """Tests for list_models tool."""

    @pytest.mark.asyncio
    async def test_list_empty(self, temp_model_dir):
        """Test listing empty registry."""
        result = await list_models()

        assert result["success"] is True
        assert result["models"] == []
        assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_list_after_training(self, sample_hourly_csv, temp_model_dir):
        """Test listing after training a model."""
        # Train a model first
        await train_forecast_model(
            csv_path=sample_hourly_csv,
            model_type="NaiveMean",
            building_name="test_building",
        )

        # List models
        result = await list_models()

        assert result["success"] is True
        assert result["total_count"] == 1
        assert len(result["models"]) == 1
        assert result["models"][0]["building_name"] == "test_building"
