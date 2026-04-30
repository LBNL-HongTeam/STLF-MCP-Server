"""Tests for data_loader module."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.data_loader import ForecastingDataLoader, DataLoadError


@pytest.fixture
def sample_csv():
    """Create a sample CSV file for testing."""
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="h"),
        "electricity_kwh": [100 + i * 0.5 for i in range(100)],
        "outdoor_temp": [50 + i * 0.1 for i in range(100)],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    Path(f.name).unlink()


class TestForecastingDataLoader:
    """Tests for ForecastingDataLoader class."""

    def test_load_csv_auto_detect(self, sample_csv):
        """Test loading CSV with auto-detection."""
        loader = ForecastingDataLoader(csv_path=sample_csv)

        assert loader.column_mapping["datetime"] == "timestamp"
        assert loader.column_mapping["target"] == "electricity_kwh"
        assert "outdoor_temp" in loader.column_mapping.get("past_covariates", [])

    def test_load_csv_explicit_mapping(self, sample_csv):
        """Test loading CSV with explicit column mapping."""
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": [],
        }
        loader = ForecastingDataLoader(csv_path=sample_csv, column_mapping=mapping)

        assert loader.column_mapping["target"] == "electricity_kwh"
        assert loader.column_mapping["past_covariates"] == []

    def test_get_data_summary(self, sample_csv):
        """Test data summary generation."""
        loader = ForecastingDataLoader(csv_path=sample_csv)
        summary = loader.get_data_summary()

        assert summary["total_samples"] == 100
        assert summary["target_column"] == "electricity_kwh"
        assert "start_date" in summary
        assert "end_date" in summary

    def test_split_train_val(self, sample_csv):
        """Test train/validation split."""
        loader = ForecastingDataLoader(csv_path=sample_csv)
        train_loader, val_loader = loader.split_train_val(validation_split=0.2)

        assert len(train_loader.df) == 80
        assert len(val_loader.df) == 20

    def test_file_not_found(self):
        """Test error on missing file."""
        with pytest.raises(DataLoadError, match="not found"):
            ForecastingDataLoader(csv_path="/nonexistent/path.csv")

    def test_to_darts_series(self, sample_csv):
        """Test conversion to Darts TimeSeries."""
        loader = ForecastingDataLoader(csv_path=sample_csv)
        target_series, cov_series = loader.to_darts_series()

        assert target_series is not None
        assert len(target_series) == 100
