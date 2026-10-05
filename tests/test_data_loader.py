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
                    "cal_hour_sin", "cal_hour_cos",
                    "cal_dow_sin", "cal_dow_cos",
                    "cal_month_sin", "cal_month_cos"):
            assert col in loader.df.columns, f"Missing calendar column: {col}"

    def test_calendar_columns_in_mapping(self, sample_csv):
        loader = ForecastingDataLoader(csv_path=sample_csv, add_calendar_features=True)
        covs = loader.column_mapping["past_covariates"]
        for col in ("cal_hour", "cal_dow", "cal_month", "cal_is_weekend",
                    "cal_hour_sin", "cal_hour_cos",
                    "cal_dow_sin", "cal_dow_cos",
                    "cal_month_sin", "cal_month_cos"):
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

    def test_dow_sin_cos_identity(self, sample_csv):
        """sin²(dow) + cos²(dow) should equal 1 for every row (Li et al. 2025, Table 6)."""
        loader = ForecastingDataLoader(csv_path=sample_csv)
        sq_sum = loader.df["cal_dow_sin"] ** 2 + loader.df["cal_dow_cos"] ** 2
        assert (sq_sum - 1.0).abs().max() < 1e-10

    def test_month_sin_cos_identity(self, sample_csv):
        """sin²(month) + cos²(month) should equal 1 for every row (Li et al. 2025, Table 6)."""
        loader = ForecastingDataLoader(csv_path=sample_csv)
        sq_sum = loader.df["cal_month_sin"] ** 2 + loader.df["cal_month_cos"] ** 2
        assert (sq_sum - 1.0).abs().max() < 1e-10

    def test_month_cycle_wraps_dec_jan(self, tmp_path):
        """Dec 31 and Jan 1 should be adjacent on the unit circle (month-1 offset check)."""
        import math
        # Build a minimal two-row DataFrame spanning the year boundary
        idx = pd.date_range("2023-12-31 23:00", periods=2, freq="h", tz="UTC")
        df = pd.DataFrame({"timestamp": idx, "electricity_kwh": [1.0, 1.0]})
        csv = tmp_path / "wrap.csv"
        df.to_csv(csv, index=False)
        loader = ForecastingDataLoader(
            csv_path=str(csv),
            column_mapping={"datetime": "timestamp", "target": "electricity_kwh"},
            add_calendar_features=True,
            lag_hours=[],
        )
        dec_sin = loader.df["cal_month_sin"].iloc[0]
        dec_cos = loader.df["cal_month_cos"].iloc[0]
        jan_sin = loader.df["cal_month_sin"].iloc[1]
        jan_cos = loader.df["cal_month_cos"].iloc[1]
        # Euclidean distance on the unit circle between Dec (month=12, month0=11)
        # and Jan (month=1, month0=0) should be small (2*sin(π/12) ≈ 0.518)
        # and strictly less than the distance between, say, Jun and Dec (≈ √2).
        dist_dec_jan = math.sqrt((dec_sin - jan_sin) ** 2 + (dec_cos - jan_cos) ** 2)
        dist_jun_dec = math.sqrt(
            (math.sin(5 * 2 * math.pi / 12) - math.sin(11 * 2 * math.pi / 12)) ** 2
            + (math.cos(5 * 2 * math.pi / 12) - math.cos(11 * 2 * math.pi / 12)) ** 2
        )
        assert dist_dec_jan < dist_jun_dec, (
            f"Dec→Jan distance {dist_dec_jan:.4f} should be less than "
            f"Jun→Dec distance {dist_jun_dec:.4f}"
        )


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


# ---------------------------------------------------------------------------
# Seasonal split tests
# ---------------------------------------------------------------------------

def _make_full_year_csv(year: int = 2023, freq: str = "h") -> str:
    """Write a full-year hourly CSV to a temp file and return its path.

    Covers all four meteorological seasons (winter/spring/summer/fall).
    """
    index = pd.date_range(f"{year}-01-01", periods=8760, freq=freq)
    data = {
        "timestamp": index,
        "electricity_kwh": [100.0 + i * 0.01 for i in range(len(index))],
        "outdoor_temp": [15.0 + 10.0 * pd.Timestamp(ts).month for ts in index],
    }
    df = pd.DataFrame(data)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


def _make_partial_year_csv(months: list[int] = None, freq: str = "h") -> str:
    """Write a CSV covering only the specified months (< 4 seasons)."""
    if months is None:
        months = [6, 7, 8]  # summer only → missing 3 seasons
    timestamps = []
    for m in months:
        start = pd.Timestamp(f"2023-{m:02d}-01")
        end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)
        timestamps.extend(pd.date_range(start, end, freq=freq).tolist())
    n = len(timestamps)
    data = {
        "timestamp": timestamps,
        "electricity_kwh": [100.0 + i * 0.01 for i in range(n)],
    }
    df = pd.DataFrame(data)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


@pytest.fixture
def full_year_csv():
    path = _make_full_year_csv()
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def partial_year_csv():
    path = _make_partial_year_csv(months=[6, 7, 8])  # summer only
    yield path
    Path(path).unlink(missing_ok=True)


class TestSeasonalSplit:
    """Tests for ForecastingDataLoader.split_train_val_seasonal()."""

    def test_total_rows_preserved(self, full_year_csv):
        """Train + val rows must equal original row count."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        total = len(loader.df)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)
        assert len(train.df) + len(val.df) == total

    def test_approximate_split_ratio(self, full_year_csv):
        """Val fraction should be close to requested split across all seasons."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        total = len(loader.df)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)
        # Allow ±2 percentage points due to integer truncation per season
        assert abs(len(val.df) / total - 0.2) < 0.02

    def test_all_four_seasons_in_val(self, full_year_csv):
        """Validation set must include rows from all four meteorological seasons."""
        _MONTH_TO_SEASON = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring",  4: "spring", 5: "spring",
            6: "summer",  7: "summer", 8: "summer",
            9: "fall",   10: "fall",  11: "fall",
        }
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        _, val = loader.split_train_val_seasonal(validation_split=0.2)
        seasons_in_val = set(val.df.index.month.map(_MONTH_TO_SEASON).unique())
        assert seasons_in_val == {"winter", "spring", "summer", "fall"}

    def test_all_four_seasons_in_train(self, full_year_csv):
        """Training set must also include rows from all four seasons."""
        _MONTH_TO_SEASON = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring",  4: "spring", 5: "spring",
            6: "summer",  7: "summer", 8: "summer",
            9: "fall",   10: "fall",  11: "fall",
        }
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, _ = loader.split_train_val_seasonal(validation_split=0.2)
        seasons_in_train = set(train.df.index.month.map(_MONTH_TO_SEASON).unique())
        assert seasons_in_train == {"winter", "spring", "summer", "fall"}

    def test_no_overlap_between_train_and_val(self, full_year_csv):
        """Train and val indices must be disjoint."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)
        overlap = train.df.index.intersection(val.df.index)
        assert len(overlap) == 0

    def test_both_sorted_chronologically(self, full_year_csv):
        """Both sub-loaders must have time-sorted indices."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)
        assert train.df.index.is_monotonic_increasing
        assert val.df.index.is_monotonic_increasing

    def test_metadata_preserved(self, full_year_csv):
        """Sub-loaders must carry the same column_mapping, frequency, csv_path."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)
        for sub in (train, val):
            assert sub.column_mapping == loader.column_mapping
            assert sub.frequency == loader.frequency
            assert sub.csv_path == loader.csv_path
            assert sub.target_scaler is None
            assert sub.covariate_scaler is None
            assert sub.future_covariate_scaler is None

    def test_fallback_on_partial_year_warns(self, partial_year_csv, caplog):
        """When fewer than 4 seasons are present a WARNING is logged and the
        result matches the sequential split."""
        import logging
        loader = ForecastingDataLoader(csv_path=partial_year_csv)
        with caplog.at_level(logging.WARNING, logger="load_forecasting.core.data_loader"):
            train_s, val_s = loader.split_train_val_seasonal(validation_split=0.2)

        # At least one warning mentioning the fallback
        assert any(
            "Falling back" in r.message or "falling back" in r.message.lower()
            for r in caplog.records
        ), "Expected a fallback warning but none was emitted"

        # Result must match sequential split
        train_seq, val_seq = loader.split_train_val(validation_split=0.2)
        assert len(train_s.df) == len(train_seq.df)
        assert len(val_s.df) == len(val_seq.df)

    def test_val_rows_are_latest_within_each_season(self, full_year_csv):
        """Within each season, validation rows must be temporally after all
        training rows of the same season."""
        _MONTH_TO_SEASON = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring",  4: "spring", 5: "spring",
            6: "summer",  7: "summer", 8: "summer",
            9: "fall",   10: "fall",  11: "fall",
        }
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)

        for season in ("winter", "spring", "summer", "fall"):
            train_season = train.df[train.df.index.month.map(_MONTH_TO_SEASON) == season]
            val_season = val.df[val.df.index.month.map(_MONTH_TO_SEASON) == season]
            if train_season.empty or val_season.empty:
                continue
            # Last training timestamp must be before first validation timestamp
            assert train_season.index.max() < val_season.index.min(), (
                f"Season '{season}': val rows overlap with or precede train rows"
            )


class TestGapFreeChunkConversion:
    """Tests for contiguous_chunks() / to_darts_series_chunks().

    A seasonal split is non-contiguous by construction.  Feeding that frame
    through ``to_darts_series`` reindexes it to a regular frequency and
    interpolates straight-line values across the multi-week holes the split
    just created.  ``to_darts_series_chunks`` splits at the gaps instead so
    that no synthetic data is ever fabricated.
    """

    def test_contiguous_frame_is_one_chunk(self, full_year_csv):
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        chunks = loader.contiguous_chunks()
        assert len(chunks) == 1
        assert len(chunks[0]) == len(loader.df)

    def test_seasonal_split_frame_is_multi_chunk(self, full_year_csv):
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)
        assert len(train.contiguous_chunks()) > 1
        assert len(val.contiguous_chunks()) > 1

    def test_chunks_preserve_every_row_and_add_none(self, full_year_csv):
        """The regression this method exists for: zero synthetic rows."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)

        train_series, _, _ = train.to_darts_series_chunks(fit_scalers=True)
        for tgt in (val,):
            tgt.target_scaler = train.target_scaler
            tgt.covariate_scaler = train.covariate_scaler
            tgt.future_covariate_scaler = train.future_covariate_scaler
        val_series, _, _ = val.to_darts_series_chunks(fit_scalers=False)

        assert sum(len(s) for s in train_series) == len(train.df)
        assert sum(len(s) for s in val_series) == len(val.df)

    def test_single_series_conversion_does_interpolate(self, full_year_csv):
        """Documents the behaviour that motivates the chunked path."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, _ = loader.split_train_val_seasonal(validation_split=0.2)
        series, _, _ = train.to_darts_series(fit_scalers=True)
        # The gap-filled single series is strictly longer than the real rows.
        assert len(series) > len(train.df)

    def test_chunks_are_internally_contiguous(self, full_year_csv):
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, _ = loader.split_train_val_seasonal(validation_split=0.2)
        for s in train.to_darts_series_chunks(fit_scalers=True)[0]:
            deltas = pd.Series(s.time_index).diff().dropna().unique()
            assert len(deltas) == 1, "chunk contains a time discontinuity"

    def test_scaler_is_shared_across_chunks(self, full_year_csv):
        """global_fit=True: one scaling applies to all chunks.

        Per-series fitting (the Darts default for sequence input) would map
        every season independently onto [0, 1] and destroy the between-season
        level differences the model needs.
        """
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, _ = loader.split_train_val_seasonal(validation_split=0.2)
        chunks = train.to_darts_series_chunks(fit_scalers=True)[0]
        mins = [float(s.values().min()) for s in chunks]
        maxs = [float(s.values().max()) for s in chunks]
        # Exactly one chunk should touch each end of the global [0, 1] range.
        assert min(mins) == pytest.approx(0.0, abs=1e-9)
        assert max(maxs) == pytest.approx(1.0, abs=1e-9)
        assert sum(m == pytest.approx(0.0, abs=1e-9) for m in mins) == 1
        assert sum(m == pytest.approx(1.0, abs=1e-9) for m in maxs) == 1

    def test_val_chunks_use_train_scaler(self, full_year_csv):
        """Validation must not be rescaled to its own range."""
        loader = ForecastingDataLoader(csv_path=full_year_csv)
        train, val = loader.split_train_val_seasonal(validation_split=0.2)
        train.to_darts_series_chunks(fit_scalers=True)
        val.target_scaler = train.target_scaler
        val_chunks = val.to_darts_series_chunks(fit_scalers=False)[0]
        # Not forced onto [0, 1] — it inherits the training scale.
        spans_unit_range = (
            min(float(s.values().min()) for s in val_chunks) == pytest.approx(0.0, abs=1e-9)
            and max(float(s.values().max()) for s in val_chunks) == pytest.approx(1.0, abs=1e-9)
        )
        assert not spans_unit_range
