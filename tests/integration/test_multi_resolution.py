"""
Subtask 1.4 — Multi-resolution pipeline validation.

Validates that the full pipeline (data loading, feature engineering, training,
and evaluation) works correctly at each supported temporal resolution:

  - 15min  → feeder-level data (4 steps/h, 96-step daily seasonality)
  - 30min  → substation-level data (2 steps/h, 48-step daily seasonality)
  - h      → district-level data (1 step/h, 24-step daily seasonality)

Synthetic CSVs are used so no external data is required.  The tests are
structured so that swapping in real CSVs later requires only changing the
fixture path.

Checked per resolution:
  1. inspect_data infers the correct frequency
  2. Lag features shift by the right step count
  3. NaiveSeasonal K equals one day in steps (seasonality period)
  4. hours_to_steps conversion is correct
  5. train_forecast_model succeeds and returns valid metrics
  6. evaluate_forecast_model succeeds on the same CSV
  7. Data summary covariate columns are populated
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.core.data_loader import ForecastingDataLoader
from load_forecasting.core.frequency_utils import (
    hours_to_steps,
    get_seasonality_steps,
    FREQ_TO_STEPS_PER_HOUR,
)
from load_forecasting.core.trainer import create_model
from load_forecasting.tools.forecasting_tools import (
    inspect_data,
    train_forecast_model,
    evaluate_forecast_model,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _make_synthetic_csv(
    freq: str,
    n_days: int = 100,
    target_col: str = "electricity_kwh",
    dt_col: str = "timestamp",
    add_temp: bool = True,
) -> str:
    """
    Generate a synthetic building-load CSV at the given frequency.

    Pattern: diurnal cycle + weekly modulation + small noise seed, so
    the signal is realistic enough for LinearRegression to learn.
    """
    steps_per_hour = FREQ_TO_STEPS_PER_HOUR[freq]
    n_points = n_days * 24 * steps_per_hour

    index = pd.date_range("2023-01-01", periods=n_points, freq=freq)

    # Diurnal pattern: peak around 14:00, trough at 03:00
    hour_frac = index.hour.to_numpy(dtype=float) + index.minute.to_numpy(dtype=float) / 60
    diurnal = 20 * np.sin(np.pi * (hour_frac - 3) / 12) ** 2

    # Weekly modulation: weekdays ~10% higher than weekends
    weekly = 10 * (index.dayofweek.to_numpy() < 5).astype(float)

    load = 100 + diurnal + weekly

    data = {dt_col: index, target_col: load}
    if add_temp:
        # Temperature correlated with hour (warm during day)
        data["outdoor_temp"] = 15 + np.pi * (hour_frac - 6) / 24 * 10 + 5

    df = pd.DataFrame(data)

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_15min():
    path = _make_synthetic_csv("15min")
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def csv_30min():
    path = _make_synthetic_csv("30min")
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def csv_hourly():
    path = _make_synthetic_csv("h")
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


# ---------------------------------------------------------------------------
# Parametrised helpers
# ---------------------------------------------------------------------------

RESOLUTIONS = [
    pytest.param("15min", 4, 96, id="feeder_15min"),
    pytest.param("30min", 2, 48, id="substation_30min"),
    pytest.param("h",     1, 24, id="district_hourly"),
]


# ---------------------------------------------------------------------------
# Unit-level: frequency utility correctness
# ---------------------------------------------------------------------------

class TestFrequencyUtils:
    """Verify step-conversion arithmetic at each resolution."""

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_steps_per_hour(self, freq, steps_per_hour, daily_steps):
        assert FREQ_TO_STEPS_PER_HOUR[freq] == steps_per_hour

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_hours_to_steps_24h(self, freq, steps_per_hour, daily_steps):
        assert hours_to_steps(24, freq) == daily_steps

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_hours_to_steps_6h(self, freq, steps_per_hour, daily_steps):
        expected = 6 * steps_per_hour
        assert hours_to_steps(6, freq) == expected

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_seasonality_steps_equals_daily(self, freq, steps_per_hour, daily_steps):
        """NaiveSeasonal K should equal one day of steps."""
        assert get_seasonality_steps(freq) == daily_steps

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_naive_seasonal_k(self, freq, steps_per_hour, daily_steps):
        """create_model sets K=daily_steps for NaiveSeasonal."""
        model = create_model("NaiveSeasonal", lookback=daily_steps, horizon=6,
                             frequency=freq)
        assert model.K == daily_steps


# ---------------------------------------------------------------------------
# Data loader: lag features at each resolution
# ---------------------------------------------------------------------------

class TestLagFeaturesMultiResolution:
    """Lag feature step counts must match the frequency."""

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_lag_24h_step_count(self, freq, steps_per_hour, daily_steps, tmp_path):
        path = _make_synthetic_csv(freq, n_days=10)
        try:
            loader = ForecastingDataLoader(
                csv_path=path,
                frequency=freq,
                add_calendar_features=False,
                lag_hours=[24],
            )
            target = loader.df["electricity_kwh"].values
            lag = loader.df["lag_24h"].values
            # After bfill prefix, row daily_steps should equal target[0]
            assert abs(lag[daily_steps] - target[0]) < 1e-6
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_lag_168h_step_count(self, freq, steps_per_hour, daily_steps, tmp_path):
        """lag_168h (one week) must be 168 × steps_per_hour steps."""
        path = _make_synthetic_csv(freq, n_days=15)
        weekly_steps = 168 * steps_per_hour
        try:
            loader = ForecastingDataLoader(
                csv_path=path,
                frequency=freq,
                add_calendar_features=False,
                lag_hours=[168],
            )
            target = loader.df["electricity_kwh"].values
            lag = loader.df["lag_168h"].values
            assert abs(lag[weekly_steps] - target[0]) < 1e-6
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.parametrize("freq,steps_per_hour,daily_steps", RESOLUTIONS)
    def test_no_nans_in_lag_features(self, freq, steps_per_hour, daily_steps):
        path = _make_synthetic_csv(freq, n_days=10)
        try:
            loader = ForecastingDataLoader(
                csv_path=path,
                frequency=freq,
                lag_hours=[24, 48, 168],
            )
            for h in [24, 48, 168]:
                col = f"lag_{h}h"
                assert not loader.df[col].isna().any(), f"NaN in {col} at freq={freq}"
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# inspect_data: frequency inference
# ---------------------------------------------------------------------------

class TestInspectDataFrequency:
    """inspect_data frequency inference for resolutions not covered by test_inspect_data.py."""

    @pytest.mark.asyncio
    async def test_30min_inferred(self, csv_30min):
        """30-min is only tested at this resolution level."""
        result = await inspect_data(csv_path=csv_30min)
        assert result["success"] is True
        assert result["frequency"]["inferred"] == "30min"

    @pytest.mark.asyncio
    async def test_frequency_mismatch_flag(self, csv_15min):
        """Declaring the wrong frequency should produce a FREQUENCY_MISMATCH flag."""
        result = await inspect_data(csv_path=csv_15min, frequency="h")
        assert result["success"] is True
        assert any("FREQUENCY_MISMATCH" in f for f in result["quality_flags"])


# ---------------------------------------------------------------------------
# End-to-end: train + evaluate at each resolution
# ---------------------------------------------------------------------------

class TestTrainEvalMultiResolution:
    """Full train → evaluate pipeline at each temporal resolution."""

    @pytest.mark.asyncio
    async def test_train_and_evaluate_15min(self, csv_15min, temp_model_dir):
        train = await train_forecast_model(
            csv_path=csv_15min,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="15min",
            building_name="feeder",
        )
        assert train["success"] is True, train.get("error")
        assert train["validation_metrics"]["cv_rmse"] is not None
        # Training info step counts must reflect 15-min resolution
        assert train["training_info"]["lookback_hours"] == 96   # 24h × 4
        assert train["training_info"]["horizon_hours"] == 24    # 6h × 4

        result = await evaluate_forecast_model(
            model_id=train["model_id"],
            csv_path=csv_15min,
        )
        assert result["success"] is True, result.get("error")
        assert result["test_metrics"]["rmse"] is not None

    @pytest.mark.asyncio
    async def test_train_and_evaluate_30min(self, csv_30min, temp_model_dir):
        train = await train_forecast_model(
            csv_path=csv_30min,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="30min",
            building_name="substation",
        )
        assert train["success"] is True, train.get("error")
        assert train["validation_metrics"]["cv_rmse"] is not None
        assert train["training_info"]["lookback_hours"] == 48   # 24h × 2
        assert train["training_info"]["horizon_hours"] == 12    # 6h × 2

        result = await evaluate_forecast_model(
            model_id=train["model_id"],
            csv_path=csv_30min,
        )
        assert result["success"] is True, result.get("error")
        assert result["test_metrics"]["rmse"] is not None

    @pytest.mark.asyncio
    async def test_train_and_evaluate_hourly(self, csv_hourly, temp_model_dir):
        train = await train_forecast_model(
            csv_path=csv_hourly,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="district",
        )
        assert train["success"] is True, train.get("error")
        assert train["validation_metrics"]["cv_rmse"] is not None
        assert train["training_info"]["lookback_hours"] == 24
        assert train["training_info"]["horizon_hours"] == 6

        result = await evaluate_forecast_model(
            model_id=train["model_id"],
            csv_path=csv_hourly,
        )
        assert result["success"] is True, result.get("error")
        assert result["test_metrics"]["rmse"] is not None

    @pytest.mark.asyncio
    async def test_naive_seasonal_15min(self, csv_15min, temp_model_dir):
        """NaiveSeasonal should train successfully at 15-min resolution."""
        result = await train_forecast_model(
            csv_path=csv_15min,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            frequency="15min",
            building_name="feeder",
        )
        assert result["success"] is True, result.get("error")

    @pytest.mark.asyncio
    async def test_naive_seasonal_30min(self, csv_30min, temp_model_dir):
        result = await train_forecast_model(
            csv_path=csv_30min,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            frequency="30min",
            building_name="substation",
        )
        assert result["success"] is True, result.get("error")


# ---------------------------------------------------------------------------
# Data summary: covariate columns present at each resolution
# ---------------------------------------------------------------------------

class TestDataSummaryMultiResolution:
    """Data summary returned by train must reflect correct features and sample counts."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("freq,building", [
        ("15min", "feeder"),
        ("30min", "substation"),
        ("h",     "district"),
    ])
    async def test_train_data_summary(self, freq, building, tmp_path, monkeypatch):
        """Covariate columns and total sample count must be correct at each resolution."""
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", str(tmp_path))
        n_days = 100
        steps_per_hour = FREQ_TO_STEPS_PER_HOUR[freq]
        expected_samples = n_days * 24 * steps_per_hour

        path = _make_synthetic_csv(freq, n_days=n_days)
        try:
            result = await train_forecast_model(
                csv_path=path,
                model_type="LinearRegression",
                lookback_hours=24,
                horizon_hours=6,
                frequency=freq,
                building_name=building,
            )
            assert result["success"] is True, result.get("error")
            covs = result["data_summary"]["covariate_columns"]
            assert any("cal_" in c for c in covs), f"No calendar features at {freq}"
            assert any("lag_" in c for c in covs), f"No lag features at {freq}"
            assert "outdoor_temp" in covs, f"outdoor_temp missing at {freq}"
            assert result["data_summary"]["total_samples"] == expected_samples
        finally:
            Path(path).unlink(missing_ok=True)
