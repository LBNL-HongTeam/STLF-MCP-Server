"""Tests for data_loader module."""

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.data_loader import (
    DataLoadError,
    ForecastingDataLoader,
    _infer_frequency,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_csv(
    periods: int = 200,
    freq: str = "h",
    target_col: str = "electricity_kwh",
    dt_col: str = "timestamp",
    add_temp: bool = True,
    add_negatives: bool = False,
    null_fraction: float = 0.0,
    duplicate_rows: int = 0,
) -> str:
    """Write a CSV to a temp file and return its path."""
    index = pd.date_range("2023-01-01", periods=periods, freq=freq)
    data = {
        dt_col: index,
        target_col: [100.0 + i * 0.5 for i in range(periods)],
    }
    if add_temp:
        data["outdoor_temp"] = [50.0 + i * 0.1 for i in range(periods)]

    df = pd.DataFrame(data)

    if add_negatives:
        df.loc[df.index[:5], target_col] = -10.0

    if null_fraction > 0:
        n_null = int(periods * null_fraction)
        null_idx = df.sample(n=n_null, random_state=42).index
        df.loc[null_idx, target_col] = np.nan

    if duplicate_rows:
        extra = df.iloc[:duplicate_rows].copy()
        df = pd.concat([df, extra]).sort_values(dt_col).reset_index(drop=True)

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


@pytest.fixture
def sample_csv():
    """Hourly CSV, 200 rows, with a temperature covariate."""
    path = _make_csv()
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def sample_csv_15min():
    """15-minute CSV, 200 rows."""
    path = _make_csv(freq="15min")
    yield path
    Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------

class TestColumnDetection:

    def test_auto_detect_datetime_and_target(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        assert loader.column_mapping["datetime"] == "timestamp"
        assert loader.column_mapping["target"] == "electricity_kwh"

    def test_auto_detect_covariate(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        assert "outdoor_temp" in loader.column_mapping["past_covariates"]

    def test_explicit_mapping_no_features(self, sample_csv):
        """When calendar and lag features are disabled, past_covariates stays empty."""
        mapping = {"datetime": "timestamp", "target": "electricity_kwh", "past_covariates": []}
        loader = ForecastingDataLoader(
            csv_path=sample_csv,
            column_mapping=mapping,
            add_calendar_features=False,
            lag_hours=[],
        )
        assert loader.column_mapping["past_covariates"] == []

    def test_explicit_mapping_features_still_added_by_default(self, sample_csv):
        """When features are enabled (default), they are appended even to an explicit mapping."""
        mapping = {"datetime": "timestamp", "target": "electricity_kwh", "past_covariates": []}
        loader = ForecastingDataLoader(csv_path=sample_csv, column_mapping=mapping)
        assert "cal_hour" in loader.column_mapping["past_covariates"]

    def test_explicit_mapping_without_past_covariates_key(self, sample_csv):
        """past_covariates is optional in the provided mapping."""
        mapping = {"datetime": "timestamp", "target": "electricity_kwh"}
        loader = ForecastingDataLoader(csv_path=sample_csv, column_mapping=mapping)
        assert "past_covariates" in loader.column_mapping


# ---------------------------------------------------------------------------
# Basic loading and data summary
# ---------------------------------------------------------------------------

class TestBasicLoading:

    def test_get_data_summary_fields(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        summary = loader.get_data_summary()
        for key in ("total_samples", "start_date", "end_date", "target_column",
                    "target_mean", "target_std", "target_min", "target_max",
                    "missing_value_count"):
            assert key in summary, f"Missing key: {key}"

    def test_total_samples(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        assert loader.get_data_summary()["total_samples"] == 200

    def test_split_train_val(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        train, val = loader.split_train_val(validation_split=0.2)
        assert len(train.df) == 160
        assert len(val.df) == 40

    def test_to_darts_series_returns_target(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        target, _, _ = loader.to_darts_series()
        assert target is not None
        assert len(target) == 200

    def test_to_darts_series_returns_covariates(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        _, cov, _ = loader.to_darts_series()
        # calendar + lag + temperature -> covariates must be non-None
        assert cov is not None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:

    def test_file_not_found(self):
        with pytest.raises(DataLoadError, match="not found"):
            ForecastingDataLoader(csv_path="/nonexistent/path.csv")

    def test_missing_datetime_column_raises(self, tmp_path):
        """A CSV with unrecognisable columns should raise DataLoadError."""
        csv = tmp_path / "bad.csv"
        # Use a column name that is neither a datetime pattern nor a target
        # pattern, so auto-detection fails for both.
        pd.DataFrame({"measurement_id": ["a", "b", "c"]}).to_csv(csv, index=False)
        with pytest.raises(DataLoadError):
            ForecastingDataLoader(csv_path=str(csv))

    def test_missing_target_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=3, freq="h")}).to_csv(
            csv, index=False
        )
        with pytest.raises(DataLoadError, match="target"):
            ForecastingDataLoader(csv_path=str(csv))

    def test_low_coverage_raises(self, tmp_path):
        """> 10 % nulls should raise DataLoadError."""
        path = _make_csv(periods=200, null_fraction=0.15)
        try:
            with pytest.raises(DataLoadError, match="coverage"):
                ForecastingDataLoader(csv_path=path)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_value_strategy_error_raises(self, tmp_path):
        # Create CSV with a NaN in a valid file (small null fraction < 10 %)
        path = _make_csv(periods=200, null_fraction=0.02)
        try:
            with pytest.raises(DataLoadError, match="missing values"):
                ForecastingDataLoader(csv_path=path, missing_value_strategy="error")
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Duplicate timestamps
# ---------------------------------------------------------------------------

class TestDuplicates:

    def test_duplicates_are_removed(self):
        path = _make_csv(periods=100, duplicate_rows=5)
        try:
            loader = ForecastingDataLoader(csv_path=path)
            # After dedup the index should be unique
            assert not loader.df.index.duplicated().any()
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

class TestCalendarFeatures:

    def test_calendar_columns_present(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv, add_calendar_features=True)
        for col in ("cal_hour", "cal_dow", "cal_month", "cal_is_weekend",
                    "cal_hour_sin", "cal_hour_cos"):
            assert col in loader.df.columns, f"Missing calendar column: {col}"

    def test_calendar_columns_in_mapping(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv, add_calendar_features=True)
        covs = loader.column_mapping["past_covariates"]
        for col in ("cal_hour", "cal_dow", "cal_month", "cal_is_weekend",
                    "cal_hour_sin", "cal_hour_cos"):
            assert col in covs, f"Calendar column not in past_covariates mapping: {col}"

    def test_calendar_disabled(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv, add_calendar_features=False)
        for col in ("cal_hour", "cal_dow"):
            assert col not in loader.df.columns

    def test_hour_sin_cos_range(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        assert loader.df["cal_hour_sin"].between(-1, 1).all()
        assert loader.df["cal_hour_cos"].between(-1, 1).all()

    def test_hour_sin_cos_identity(self, sample_csv):
        """sin² + cos² should equal 1 for every row."""
        loader = ForecastingDataLoader(csv_path=sample_csv)
        sq_sum = loader.df["cal_hour_sin"] ** 2 + loader.df["cal_hour_cos"] ** 2
        assert (sq_sum - 1.0).abs().max() < 1e-10

    def test_is_weekend_binary(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv)
        vals = loader.df["cal_is_weekend"].unique()
        assert set(vals).issubset({0.0, 1.0})


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

class TestLagFeatures:

    def test_lag_columns_present(self, sample_csv):
        loader = ForecastingDataLoader(
            csv_path=sample_csv, lag_hours=[24, 48, 168]
        )
        for h in [24, 48, 168]:
            assert f"lag_{h}h" in loader.df.columns, f"Missing lag column: lag_{h}h"

    def test_lag_columns_in_mapping(self, sample_csv):
        loader = ForecastingDataLoader(
            csv_path=sample_csv, lag_hours=[24, 48]
        )
        covs = loader.column_mapping["past_covariates"]
        for h in [24, 48]:
            assert f"lag_{h}h" in covs

    def test_lag_values_correct(self, sample_csv):
        """lag_24h at row i should equal the target value 24 steps earlier."""
        loader = ForecastingDataLoader(
            csv_path=sample_csv,
            lag_hours=[24],
            add_calendar_features=False,  # isolate lag test
        )
        target = loader.df["electricity_kwh"].values
        lag = loader.df["lag_24h"].values
        # After bfill prefix, row 24 onward should match shifted target
        assert abs(lag[24] - target[0]) < 1e-9

    def test_lag_disabled(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv, lag_hours=[])
        for h in [24, 48, 168]:
            assert f"lag_{h}h" not in loader.df.columns

    def test_no_nans_in_lag_after_bfill(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv, lag_hours=[24, 48, 168])
        for h in [24, 48, 168]:
            col = f"lag_{h}h"
            assert not loader.df[col].isna().any(), f"NaN found in {col}"

    def test_lag_15min(self, sample_csv_15min):
        """Lag features should work at 15-minute resolution."""
        loader = ForecastingDataLoader(
            csv_path=sample_csv_15min, frequency="15min", lag_hours=[24]
        )
        # 24h at 15-min = 96 steps
        target = loader.df["electricity_kwh"].values
        lag = loader.df["lag_24h"].values
        assert abs(lag[96] - target[0]) < 1e-9


# ---------------------------------------------------------------------------
# Frequency inference
# ---------------------------------------------------------------------------

class TestFrequencyInference:

    def test_infer_hourly(self):
        idx = pd.date_range("2023-01-01", periods=24, freq="h")
        assert _infer_frequency(idx) == "h"

    def test_infer_15min(self):
        idx = pd.date_range("2023-01-01", periods=96, freq="15min")
        assert _infer_frequency(idx) == "15min"

    def test_infer_30min(self):
        idx = pd.date_range("2023-01-01", periods=48, freq="30min")
        assert _infer_frequency(idx) == "30min"

    def test_infer_unknown_returns_none(self):
        idx = pd.DatetimeIndex(["2023-01-01 00:00", "2023-01-01 00:07"])
        assert _infer_frequency(idx) is None

    def test_mismatch_logs_warning(self, sample_csv, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="load_forecasting.core.data_loader"):
            ForecastingDataLoader(csv_path=sample_csv, frequency="15min")
        assert any("does not match" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Negative value warning
# ---------------------------------------------------------------------------

class TestNegativeValues:

    def test_negative_target_logs_warning(self, caplog):
        import logging
        path = _make_csv(periods=200, add_negatives=True)
        try:
            with caplog.at_level(logging.WARNING, logger="load_forecasting.core.data_loader"):
                ForecastingDataLoader(csv_path=path)
            assert any("negative" in r.message for r in caplog.records)
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Future covariates
# ---------------------------------------------------------------------------

def _make_csv_with_future_covariates(
    periods: int = 300,
    freq: str = "h",
) -> str:
    """CSV with a past covariate (outdoor_temp) and a future covariate (temp_forecast)."""
    index = pd.date_range("2023-01-01", periods=periods, freq=freq)
    df = pd.DataFrame({
        "timestamp": index,
        "electricity_kwh": [100.0 + i * 0.5 for i in range(periods)],
        "outdoor_temp": [50.0 + i * 0.1 for i in range(periods)],
        "temp_forecast": [52.0 + i * 0.1 for i in range(periods)],
    })
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


@pytest.fixture
def csv_with_future_covariates():
    path = _make_csv_with_future_covariates()
    yield path
    Path(path).unlink(missing_ok=True)


class TestFutureCovariates:

    def test_future_covariates_not_auto_detected(self, csv_with_future_covariates):
        """Future covariates are never auto-detected; temp_forecast stays unrecognised."""
        loader = ForecastingDataLoader(csv_path=csv_with_future_covariates)
        assert loader.column_mapping.get("future_covariates") == []
        # temp_forecast matches no auto-detect future pattern — it ends up in
        # past_covariates because "temp" matches the past pattern
        assert "temp_forecast" in loader.column_mapping.get("past_covariates", [])

    def test_explicit_future_covariates_in_mapping(self, csv_with_future_covariates):
        """Explicitly listed future covariates are stored in column_mapping."""
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": ["outdoor_temp"],
            "future_covariates": ["temp_forecast"],
        }
        loader = ForecastingDataLoader(
            csv_path=csv_with_future_covariates,
            column_mapping=mapping,
            add_calendar_features=False,
            lag_hours=[],
        )
        assert loader.column_mapping["future_covariates"] == ["temp_forecast"]
        assert loader.column_mapping["past_covariates"] == ["outdoor_temp"]

    def test_future_covariates_key_always_present(self, sample_csv):
        """future_covariates key must always exist in column_mapping, even when empty."""
        loader = ForecastingDataLoader(csv_path=sample_csv)
        assert "future_covariates" in loader.column_mapping

    def test_future_covariate_returns_third_series(self, csv_with_future_covariates):
        """to_darts_series should return a non-None 3rd element for future covariates."""
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": [],
            "future_covariates": ["temp_forecast"],
        }
        loader = ForecastingDataLoader(
            csv_path=csv_with_future_covariates,
            column_mapping=mapping,
            add_calendar_features=False,
            lag_hours=[],
        )
        _, past_cov, future_cov = loader.to_darts_series(fit_scalers=True)
        assert past_cov is None
        assert future_cov is not None
        assert len(future_cov) == 300

    def test_future_covariate_scaler_fitted(self, csv_with_future_covariates):
        """future_covariate_scaler should be fitted after to_darts_series."""
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": [],
            "future_covariates": ["temp_forecast"],
        }
        loader = ForecastingDataLoader(
            csv_path=csv_with_future_covariates,
            column_mapping=mapping,
            add_calendar_features=False,
            lag_hours=[],
        )
        loader.to_darts_series(fit_scalers=True)
        assert loader.future_covariate_scaler is not None

    def test_future_covariate_scaler_reused_on_transform(self, csv_with_future_covariates):
        """fit_scalers=False should apply the existing future_covariate_scaler."""
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": [],
            "future_covariates": ["temp_forecast"],
        }
        loader = ForecastingDataLoader(
            csv_path=csv_with_future_covariates,
            column_mapping=mapping,
            add_calendar_features=False,
            lag_hours=[],
        )
        _, _, fut_cov_fit = loader.to_darts_series(fit_scalers=True)
        # Now reuse the scaler (fit_scalers=False)
        _, _, fut_cov_transform = loader.to_darts_series(fit_scalers=False)
        # Both should produce the same values since scaler is already fitted
        import numpy as np
        np.testing.assert_allclose(
            fut_cov_fit.values(), fut_cov_transform.values(), rtol=1e-5
        )

    def test_split_train_val_copies_future_scaler(self, csv_with_future_covariates):
        """split_train_val sub-loaders should propagate the future_covariate_scaler."""
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": [],
            "future_covariates": ["temp_forecast"],
        }
        loader = ForecastingDataLoader(
            csv_path=csv_with_future_covariates,
            column_mapping=mapping,
            add_calendar_features=False,
            lag_hours=[],
        )
        train_loader, val_loader = loader.split_train_val(validation_split=0.2)
        train_loader.to_darts_series(fit_scalers=True)
        # Manually propagate scalers (as the tool does)
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, _, val_fut = val_loader.to_darts_series(fit_scalers=False)
        assert val_fut is not None

    def test_missing_future_covariate_column_warns(self, csv_with_future_covariates, caplog):
        """A future covariate column that's missing in the CSV should warn and be dropped."""
        import logging
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": [],
            "future_covariates": ["nonexistent_col"],
        }
        with caplog.at_level(logging.WARNING, logger="load_forecasting.core.data_loader"):
            loader = ForecastingDataLoader(
                csv_path=csv_with_future_covariates,
                column_mapping=mapping,
                add_calendar_features=False,
                lag_hours=[],
            )
        assert loader.column_mapping["future_covariates"] == []
        assert any("nonexistent_col" in r.message for r in caplog.records)

    def test_get_data_summary_includes_future_covariates(self, csv_with_future_covariates):
        """data_summary must include future_covariate_columns key."""
        mapping = {
            "datetime": "timestamp",
            "target": "electricity_kwh",
            "past_covariates": [],
            "future_covariates": ["temp_forecast"],
        }
        loader = ForecastingDataLoader(
            csv_path=csv_with_future_covariates,
            column_mapping=mapping,
            add_calendar_features=False,
            lag_hours=[],
        )
        summary = loader.get_data_summary()
        assert "future_covariate_columns" in summary
        assert "temp_forecast" in summary["future_covariate_columns"]
