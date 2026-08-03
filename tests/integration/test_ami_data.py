"""
Integration tests: pipeline validation on real AMI data at three spatial resolutions.

Covers subtask 1.2 (future covariates with real weather data) and subtask 1.4
(multi-resolution pipeline validation) by running inspect_data → train_forecast_model
→ evaluate_forecast_model on:

  - District level:   data/examples/AMI/2021_city_level.csv
  - Substation level: data/examples/AMI/2021_substation_level.csv
  - Feeder level:     data/examples/AMI/2021_feeder_level-GLENDOVEER_substation.csv

Weather data (2021-2023_openmeteo_weather.csv) is merged with each load dataset
to supply past_covariates (T_out, RH_out) and, in one dedicated test, as
future_covariates to validate the 1.2 end-to-end path with real data.
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.tools import (
    train_forecast_model,
    evaluate_forecast_model,
    inspect_data,
)

# ---------------------------------------------------------------------------
# Paths to the real AMI data files
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent.parent
_AMI_DIR = _REPO_ROOT / "data" / "examples" / "AMI"

_CITY_CSV = _AMI_DIR / "2021_city_level.csv"
_SUBSTATION_CSV = _AMI_DIR / "2021_substation_level.csv"
_FEEDER_CSV = _AMI_DIR / "2021_feeder_level-GLENDOVEER_substation.csv"
_WEATHER_CSV = _AMI_DIR / "2021-2023_openmeteo_weather.csv"

# Specific load columns used per level
_CITY_TARGET = "City (n=41703)"
_SUBSTATION_TARGET = "GLENDOVEER (n=9214)"
_FEEDER_TARGET = "GLENDOVEER-13599 (n=1910)"

# Weather columns used as covariates
_PAST_WEATHER_COLS = ["T_out", "RH_out"]
_FUTURE_WEATHER_COLS = ["T_out"]  # temperature as proxy for forecast (1.2 test)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_weather() -> pd.DataFrame:
    """Load weather CSV and set DatetimeIndex."""
    w = pd.read_csv(_WEATHER_CSV)
    w.index = pd.to_datetime(w["date"])
    w.index.name = "Datetime"
    return w.drop(columns=["date"])


def _build_merged_csv(load_csv: Path, target_col: str, tmp_dir: Path) -> str:
    """
    Merge one load CSV with weather data and write to a temp CSV.

    Returns the path to the merged temp CSV.
    """
    load_df = pd.read_csv(load_csv)
    load_df.index = pd.to_datetime(load_df["Datetime"])
    load_df.index.name = "Datetime"
    load_df = load_df[[target_col]]  # keep only the chosen target

    weather_df = _load_weather()

    merged = load_df.join(weather_df[_PAST_WEATHER_COLS + _FUTURE_WEATHER_COLS],
                          how="inner")
    merged = merged.reset_index()  # restore Datetime column

    out_path = tmp_dir / f"{load_csv.stem}_merged.csv"
    merged.to_csv(out_path, index=False)
    return str(out_path)


# ---------------------------------------------------------------------------
# Module-scoped model directory (isolates these tests from others)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ami_model_dir():
    """Temporary model registry directory for all AMI tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.environ.get("LOAD_FORECASTING_MODEL_DIR")
        os.environ["LOAD_FORECASTING_MODEL_DIR"] = tmpdir
        yield tmpdir
        if old is not None:
            os.environ["LOAD_FORECASTING_MODEL_DIR"] = old
        else:
            os.environ.pop("LOAD_FORECASTING_MODEL_DIR", None)


# ---------------------------------------------------------------------------
# Module-scoped merged CSV fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def city_merged_csv(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("city")
    return _build_merged_csv(_CITY_CSV, _CITY_TARGET, tmpdir)


@pytest.fixture(scope="module")
def substation_merged_csv(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("substation")
    return _build_merged_csv(_SUBSTATION_CSV, _SUBSTATION_TARGET, tmpdir)


@pytest.fixture(scope="module")
def feeder_merged_csv(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("feeder")
    return _build_merged_csv(_FEEDER_CSV, _FEEDER_TARGET, tmpdir)


# ---------------------------------------------------------------------------
# Column mappings
# ---------------------------------------------------------------------------

def _city_mapping(future_covs=None):
    m = {
        "datetime": "Datetime",
        "target": _CITY_TARGET,
        "past_covariates": _PAST_WEATHER_COLS,
    }
    if future_covs:
        m["past_covariates"] = [c for c in _PAST_WEATHER_COLS if c not in future_covs]
        m["future_covariates"] = future_covs
    return m


def _substation_mapping():
    return {
        "datetime": "Datetime",
        "target": _SUBSTATION_TARGET,
        "past_covariates": _PAST_WEATHER_COLS,
    }


def _feeder_mapping():
    return {
        "datetime": "Datetime",
        "target": _FEEDER_TARGET,
        "past_covariates": _PAST_WEATHER_COLS,
    }


# ===========================================================================
# District / City level
# ===========================================================================

class TestCityLevel:
    """Pipeline validation on district-level (city aggregate) AMI data."""

    async def test_inspect_data(self, city_merged_csv, ami_model_dir):
        result = inspect_data(
            csv_path=city_merged_csv,
            column_mapping=_city_mapping(),
        )
        assert result["success"] is True, result.get("error")
        assert result["ready_to_train"] is True
        assert result["frequency"]["inferred"] == "h"
        assert result["columns"]["target"] == _CITY_TARGET
        assert "T_out" in result["columns"]["past_covariates"]
        assert "RH_out" in result["columns"]["past_covariates"]
        # A full year at hourly frequency should have excellent coverage
        assert result["time_range"]["coverage_pct"] >= 99.0

    async def test_train(self, city_merged_csv, ami_model_dir):
        result = train_forecast_model(
            csv_path=city_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="city",
            column_mapping=_city_mapping(),
        )
        assert result["success"] is True, result.get("error")
        assert result["model_id"] is not None
        assert "city" in result["model_id"]
        assert result["model_type"] == "LinearRegression"
        # Validation metrics should be populated
        val = result["validation_metrics"]
        assert val["cv_rmse"] is not None
        assert val["cv_rmse"] > 0
        # Data summary should include weather past covariates
        summary = result["data_summary"]
        assert "T_out" in summary["covariate_columns"]
        assert "RH_out" in summary["covariate_columns"]
        assert summary["future_covariate_columns"] == []

    async def test_train_and_evaluate(self, city_merged_csv, ami_model_dir):
        train_result = train_forecast_model(
            csv_path=city_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="city_eval",
            column_mapping=_city_mapping(),
        )
        assert train_result["success"] is True, train_result.get("error")

        eval_result = evaluate_forecast_model(
            model_id=train_result["model_id"],
            csv_path=city_merged_csv,
        )
        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["rmse"] is not None
        assert eval_result["test_metrics"]["rmse"] > 0
        assert eval_result["test_metrics"]["cv_rmse"] is not None


# ===========================================================================
# Substation level
# ===========================================================================

class TestSubstationLevel:
    """Pipeline validation on substation-level AMI data (GLENDOVEER substation)."""

    async def test_inspect_data(self, substation_merged_csv, ami_model_dir):
        result = inspect_data(
            csv_path=substation_merged_csv,
            column_mapping=_substation_mapping(),
        )
        assert result["success"] is True, result.get("error")
        assert result["ready_to_train"] is True
        assert result["frequency"]["inferred"] == "h"
        assert result["columns"]["target"] == _SUBSTATION_TARGET
        assert result["time_range"]["coverage_pct"] >= 99.0

    async def test_train(self, substation_merged_csv, ami_model_dir):
        result = train_forecast_model(
            csv_path=substation_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="substation",
            column_mapping=_substation_mapping(),
        )
        assert result["success"] is True, result.get("error")
        assert result["validation_metrics"]["cv_rmse"] is not None
        assert result["data_summary"]["target_column"] == _SUBSTATION_TARGET

    async def test_train_and_evaluate(self, substation_merged_csv, ami_model_dir):
        train_result = train_forecast_model(
            csv_path=substation_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="substation_eval",
            column_mapping=_substation_mapping(),
        )
        assert train_result["success"] is True, train_result.get("error")

        eval_result = evaluate_forecast_model(
            model_id=train_result["model_id"],
            csv_path=substation_merged_csv,
        )
        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["rmse"] > 0

    async def test_naive_baseline(self, substation_merged_csv, ami_model_dir):
        """NaiveSeasonal should also train successfully on real data."""
        result = train_forecast_model(
            csv_path=substation_merged_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="substation_naive",
            column_mapping=_substation_mapping(),
        )
        assert result["success"] is True, result.get("error")
        assert result["validation_metrics"]["cv_rmse"] is not None


# ===========================================================================
# Feeder level
# ===========================================================================

class TestFeederLevel:
    """Pipeline validation on feeder-level AMI data (GLENDOVEER-13599 circuit)."""

    async def test_inspect_data(self, feeder_merged_csv, ami_model_dir):
        result = inspect_data(
            csv_path=feeder_merged_csv,
            column_mapping=_feeder_mapping(),
        )
        assert result["success"] is True, result.get("error")
        assert result["ready_to_train"] is True
        assert result["frequency"]["inferred"] == "h"
        assert result["columns"]["target"] == _FEEDER_TARGET
        assert result["time_range"]["coverage_pct"] >= 99.0

    async def test_train(self, feeder_merged_csv, ami_model_dir):
        result = train_forecast_model(
            csv_path=feeder_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="feeder",
            column_mapping=_feeder_mapping(),
        )
        assert result["success"] is True, result.get("error")
        assert result["validation_metrics"]["cv_rmse"] is not None
        assert result["data_summary"]["target_column"] == _FEEDER_TARGET

    async def test_train_and_evaluate(self, feeder_merged_csv, ami_model_dir):
        train_result = train_forecast_model(
            csv_path=feeder_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="feeder_eval",
            column_mapping=_feeder_mapping(),
        )
        assert train_result["success"] is True, train_result.get("error")

        eval_result = evaluate_forecast_model(
            model_id=train_result["model_id"],
            csv_path=feeder_merged_csv,
        )
        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["rmse"] > 0


# ===========================================================================
# Subtask 1.2: Future covariates with real weather data
# ===========================================================================

class TestWeatherFutureCovariates:
    """
    Validates subtask 1.2 end-to-end with real AMI + weather data.

    T_out is used as a future covariate (simulating a perfect temperature
    forecast), while RH_out remains a past covariate.  Tests that:
    - The model trains successfully with future covariates from real data.
    - future_covariate_scaler is persisted to the registry.
    - Evaluation re-uses the persisted scaler correctly.
    """

    async def test_train_with_temperature_forecast(
        self, city_merged_csv, ami_model_dir
    ):
        """Train LinearRegression using T_out as a future covariate."""
        mapping = _city_mapping(future_covs=_FUTURE_WEATHER_COLS)
        result = train_forecast_model(
            csv_path=city_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="city_fc",
            column_mapping=mapping,
        )
        assert result["success"] is True, result.get("error")
        # Future covariate column must appear in data summary
        assert _FUTURE_WEATHER_COLS[0] in result["data_summary"]["future_covariate_columns"]
        # Past covariates should NOT include T_out
        assert _FUTURE_WEATHER_COLS[0] not in result["data_summary"]["covariate_columns"]
        assert result["validation_metrics"]["cv_rmse"] is not None

    async def test_future_covariate_scaler_persisted(
        self, city_merged_csv, ami_model_dir
    ):
        """future_covariate_scaler must be saved to the registry after training."""
        from load_forecasting.core.model_registry import ModelRegistry

        mapping = _city_mapping(future_covs=_FUTURE_WEATHER_COLS)
        train_result = train_forecast_model(
            csv_path=city_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="city_fc_scaler",
            column_mapping=mapping,
        )
        assert train_result["success"] is True, train_result.get("error")

        registry = ModelRegistry()
        _, _, scalers = registry.load_model(train_result["model_id"])
        assert "future_covariate_scaler" in scalers
        assert scalers["future_covariate_scaler"] is not None

    async def test_evaluate_with_future_covariates(
        self, city_merged_csv, ami_model_dir
    ):
        """evaluate_forecast_model must work when model was trained with future covariates."""
        mapping = _city_mapping(future_covs=_FUTURE_WEATHER_COLS)
        train_result = train_forecast_model(
            csv_path=city_merged_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            building_name="city_fc_eval",
            column_mapping=mapping,
        )
        assert train_result["success"] is True, train_result.get("error")

        eval_result = evaluate_forecast_model(
            model_id=train_result["model_id"],
            csv_path=city_merged_csv,
            # column_mapping loaded from metadata; future covariate mapping is reused
        )
        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["rmse"] is not None
        assert eval_result["test_metrics"]["rmse"] > 0
