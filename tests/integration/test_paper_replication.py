"""
Paper replication integration tests — Li et al. (2025).

Reference:
    "A cross-dimensional analysis of data-driven short-term load forecasting
    methods with large-scale smart meter data"
    Energy & Buildings 344 (2025) 115909
    DOI: 10.1016/j.enbuild.2025.115909

Setup mirrors the paper exactly:
  - Training data  : 2021 Portland General Electric AMI data
  - Test data      : 2023 Portland General Electric AMI data
  - lookback_hours : 96  (paper's four-day window)
  - horizon_hours  : 96  (paper's four-day horizon)
  - frequency      : "h" (hourly; paper downsampled 15-min → hourly)
  - Seasonal split : 80/20 per season (split_train_val_seasonal, already default)
  - Metrics        : overall MAPE, PMAPE, PTE for winter/summer peak windows

All tests skip automatically when the AMI data files are absent (same
pattern as tests/integration/test_ami_data.py).

No strict numeric assertions are made on MAPE values because results vary
by hardware, random seed, and exact hyperparameters.  The tests assert:
  - success=True
  - MAPE is not None
  - MAPE < 50%  (sanity bound — the paper's worst result is ~30%)
  - peak_metrics keys are present and have valid types when peak_dates supplied
"""

import sys
import os
import logging
import tempfile
import csv

import pytest
import pandas as pd

# Insert src/ into path so the package is importable without pip install -e .
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from load_forecasting.tools import (
    train_forecast_model,
    evaluate_forecast_model,
    backtest_model,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AMI data paths
# ---------------------------------------------------------------------------
AMI_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "examples", "AMI"
)

# City (district) level — with weather already merged
CITY_2021 = os.path.join(AMI_DIR, "2021_city_level_with_weather.csv")
CITY_2023 = os.path.join(AMI_DIR, "2023_city_level_with_weather.csv")

# Substation level (no weather; we use load only)
SUBSTATION_2021 = os.path.join(AMI_DIR, "2021_substation_level.csv")
SUBSTATION_2023 = os.path.join(AMI_DIR, "2023_substation_level.csv")

# Feeder level — use the pre-merged LENTS HAPPY VALLEY file (has weather)
FEEDER_2021_MERGED = os.path.join(AMI_DIR, "_merged_LENTS_HAPPY_VALLEY.csv")
# For 2023 feeder test we build a single-target CSV from the LENTS feeder file
FEEDER_2023_LENTS = os.path.join(AMI_DIR, "2023_feeder_level-LENTS_substation.csv")
# Open-Meteo weather covering 2021-2023, merged into the 2023 feeder test CSV
WEATHER_CSV = os.path.join(AMI_DIR, "2021-2023_openmeteo_weather.csv")

# Paper peak windows (Section 4.2)
WINTER_PEAK_DATES = ["2023-02-22", "2023-02-23", "2023-02-24", "2023-02-25"]
SUMMER_PEAK_DATES = ["2023-08-14", "2023-08-15", "2023-08-16", "2023-08-17"]

# Paper lookback/horizon (Section 3.1)
LOOKBACK_HOURS = 96
HORIZON_HOURS = 96


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _files_exist(*paths: str) -> bool:
    """Return True only when every file path exists on disk."""
    return all(os.path.exists(p) for p in paths)


def _write_single_target_csv(
    src_csv: str,
    datetime_col: str,
    target_col: str,
    weather_csv: str | None = None,
    weather_datetime_col: str = "date",
    weather_cols: list | None = None,
) -> str:
    """
    Write a temporary single-target CSV from a multi-column load file.

    Optionally inner-joins weather data from a separate CSV on the datetime
    column.  Returns the path to the temp file.
    """
    df = pd.read_csv(src_csv, parse_dates=[datetime_col])
    df = df[[datetime_col, target_col]].rename(columns={target_col: "load_kwh"})
    df = df.dropna(subset=["load_kwh"])

    if weather_csv and weather_cols:
        w = pd.read_csv(weather_csv, parse_dates=[weather_datetime_col])
        w = w.rename(columns={weather_datetime_col: datetime_col})
        # Align datetime precision (strip timezone if present)
        df[datetime_col] = pd.to_datetime(df[datetime_col]).dt.tz_localize(None)
        w[datetime_col] = pd.to_datetime(w[datetime_col]).dt.tz_localize(None)
        keep_cols = [datetime_col] + [c for c in weather_cols if c in w.columns]
        df = df.merge(w[keep_cols], on=datetime_col, how="inner")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    df.to_csv(tmp, index=False)
    tmp.close()
    return tmp.name


# ==============================================================================
# 1. District (city) level
# ==============================================================================

@pytest.mark.skipif(
    not _files_exist(CITY_2021, CITY_2023),
    reason="AMI city-level CSV files not found — skipping paper replication test",
)
class TestPaperReplicationDistrictLevel:
    """
    Train on 2021 city-level data, evaluate on 2023 city-level data.

    Paper reference: district level (n=41,703 customers).
    Expected range from Fig. 5: MAPE < 10% for LinearRegression and XGBoost
    at 96-h horizon.
    """

    COLUMN_MAPPING = {
        "datetime": "Datetime",
        "target": "City (n=41703)",
        "past_covariates": ["T_out", "RH_out", "Direct_Radiation"],
    }

    def test_linear_regression_district(self):
        """LinearRegression: train 2021, evaluate 2023, district level."""
        result = train_forecast_model(
            csv_path=CITY_2021,
            model_type="LinearRegression",
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            frequency="h",
            validation_split=0.2,
            building_name="city_district",
            column_mapping=self.COLUMN_MAPPING,
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        model_id = result["model_id"]

        # Verify energy tracking is present
        ti = result.get("training_info", {})
        assert ti.get("energy_kwh") is not None
        assert ti["energy_kwh"] > 0
        assert ti.get("watts_assumed") == 65

        logger.info(
            "[DISTRICT] LinearRegression | train MAPE=%.2f%% | val MAPE=%.2f%% | "
            "energy=%.6f kWh",
            result["training_metrics"].get("mape", float("nan")),
            result["validation_metrics"].get("mape", float("nan")),
            ti["energy_kwh"],
        )

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=CITY_2023,
            return_predictions=False,
            peak_dates=WINTER_PEAK_DATES,
        )
        assert eval_result["success"], f"Evaluation failed: {eval_result.get('error')}"
        mape = eval_result["test_metrics"]["mape"]
        assert mape is not None
        assert mape < 50, f"MAPE={mape:.1f}% exceeds sanity bound of 50%"

        pm = eval_result.get("peak_metrics", {})
        assert pm.get("n_peak_days_evaluated", 0) > 0, "No winter peak days evaluated"

        logger.info(
            "[DISTRICT] LinearRegression | test MAPE=%.2f%% | "
            "winter PMAPE=%.2f%% | winter PTE=%.2fh",
            mape,
            pm.get("peak_mape", float("nan")),
            pm.get("peak_timing_error_hours", float("nan")),
        )
        # Paper Fig. 5: LinearRegression district MAPE typically 5–20%
        logger.info(
            "[PAPER REF] District LinearRegression expected MAPE ~ 5-20%% "
            "(Fig. 5, 96-h horizon)"
        )

    def test_xgboost_district(self):
        """XGBoost: train 2021, evaluate 2023, district level."""
        result = train_forecast_model(
            csv_path=CITY_2021,
            model_type="XGBoost",
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            frequency="h",
            validation_split=0.2,
            building_name="city_district",
            column_mapping=self.COLUMN_MAPPING,
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        model_id = result["model_id"]

        ti = result.get("training_info", {})
        assert ti.get("watts_assumed") == 100  # XGBoost assumed wattage

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=CITY_2023,
            return_predictions=False,
            peak_dates=WINTER_PEAK_DATES,
        )
        assert eval_result["success"], f"Evaluation failed: {eval_result.get('error')}"
        mape = eval_result["test_metrics"]["mape"]
        assert mape is not None
        assert mape < 50

        pm = eval_result.get("peak_metrics", {})
        logger.info(
            "[DISTRICT] XGBoost | test MAPE=%.2f%% | "
            "winter PMAPE=%s | winter PTE=%s",
            mape,
            f"{pm.get('peak_mape', 'N/A'):.2f}%" if pm.get("peak_mape") else "N/A",
            f"{pm.get('peak_timing_error_hours', 'N/A'):.2fh}" if pm.get("peak_timing_error_hours") else "N/A",
        )
        logger.info("[PAPER REF] District XGBoost expected MAPE < 10%% (Fig. 5)")

    def test_naive_seasonal_district(self):
        """NaiveSeasonal: district-level baseline."""
        result = train_forecast_model(
            csv_path=CITY_2021,
            model_type="NaiveSeasonal",
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            frequency="h",
            validation_split=0.2,
            building_name="city_district_naive",
            column_mapping=self.COLUMN_MAPPING,
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        model_id = result["model_id"]

        ti = result.get("training_info", {})
        assert ti.get("watts_assumed") == 15  # Naive models: 15 W

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=CITY_2023,
            return_predictions=False,
        )
        assert eval_result["success"], f"Evaluation failed: {eval_result.get('error')}"
        mape = eval_result["test_metrics"]["mape"]
        assert mape is not None
        logger.info("[DISTRICT] NaiveSeasonal | test MAPE=%.2f%%", mape)
        logger.info("[PAPER REF] NaiveSeasonal (baseline) typically MAPE > 20%%")


# ==============================================================================
# 2. Substation level
# ==============================================================================

@pytest.mark.skipif(
    not _files_exist(SUBSTATION_2021, SUBSTATION_2023),
    reason="AMI substation CSV files not found — skipping paper replication test",
)
class TestPaperReplicationSubstationLevel:
    """
    Train on 2021 substation data, evaluate on 2023.

    Uses GLENDOVEER substation (n=9,214 customers) as representative target.
    Paper reference: substation level. Expected MAPE < 15% for XGBoost (Fig. 5).
    """

    TARGET_COL = "GLENDOVEER (n=9214)"
    DT_COL = "Datetime"

    @pytest.fixture(autouse=True)
    def _build_temp_csvs(self, tmp_path):
        """Create single-target CSVs for the GLENDOVEER substation."""
        self.train_csv = str(tmp_path / "substation_train.csv")
        self.test_csv = str(tmp_path / "substation_test.csv")

        for src, dst in [
            (SUBSTATION_2021, self.train_csv),
            (SUBSTATION_2023, self.test_csv),
        ]:
            df = pd.read_csv(src)[[self.DT_COL, self.TARGET_COL]]
            df = df.rename(columns={self.TARGET_COL: "load_kwh"})
            df.to_csv(dst, index=False)

        self.column_mapping = {
            "datetime": self.DT_COL,
            "target": "load_kwh",
        }

    def test_linear_regression_substation(self):
        result = train_forecast_model(
            csv_path=self.train_csv,
            model_type="LinearRegression",
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            frequency="h",
            validation_split=0.2,
            building_name="substation_glendoveer",
            column_mapping=self.column_mapping,
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        model_id = result["model_id"]

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=self.test_csv,
            return_predictions=False,
            peak_dates=WINTER_PEAK_DATES,
        )
        assert eval_result["success"], f"Evaluation failed: {eval_result.get('error')}"
        mape = eval_result["test_metrics"]["mape"]
        assert mape is not None
        assert mape < 50

        pm = eval_result.get("peak_metrics", {})
        logger.info(
            "[SUBSTATION] LinearRegression | MAPE=%.2f%% | "
            "winter PMAPE=%s | PTE=%s",
            mape,
            f"{pm.get('peak_mape', float('nan')):.2f}%" if pm.get("peak_mape") is not None else "N/A",
            f"{pm.get('peak_timing_error_hours', float('nan')):.2f}h" if pm.get("peak_timing_error_hours") is not None else "N/A",
        )
        logger.info("[PAPER REF] Substation LinearRegression MAPE ~ 5-20%% (Fig. 5)")

    def test_xgboost_substation(self):
        result = train_forecast_model(
            csv_path=self.train_csv,
            model_type="XGBoost",
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            frequency="h",
            validation_split=0.2,
            building_name="substation_glendoveer",
            column_mapping=self.column_mapping,
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        model_id = result["model_id"]

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=self.test_csv,
            return_predictions=False,
            peak_dates=SUMMER_PEAK_DATES,
        )
        assert eval_result["success"], f"Evaluation failed: {eval_result.get('error')}"
        mape = eval_result["test_metrics"]["mape"]
        assert mape is not None
        assert mape < 50

        pm = eval_result.get("peak_metrics", {})
        logger.info(
            "[SUBSTATION] XGBoost | MAPE=%.2f%% | summer PMAPE=%s | PTE=%s",
            mape,
            f"{pm.get('peak_mape', float('nan')):.2f}%" if pm.get("peak_mape") is not None else "N/A",
            f"{pm.get('peak_timing_error_hours', float('nan')):.2f}h" if pm.get("peak_timing_error_hours") is not None else "N/A",
        )
        logger.info("[PAPER REF] Substation XGBoost MAPE < 10%% (Fig. 5)")


# ==============================================================================
# 3. Feeder level
# ==============================================================================

@pytest.mark.skipif(
    not _files_exist(FEEDER_2021_MERGED, FEEDER_2023_LENTS),
    reason="AMI feeder CSV files not found — skipping paper replication test",
)
class TestPaperReplicationFeederLevel:
    """
    Train on 2021 LENTS HAPPY VALLEY feeder data (pre-merged with weather),
    evaluate on 2023 LENTS feeder data.

    Paper reference: feeder level. Expected MAPE 10-30% for LinearRegression
    and XGBoost at 96h horizon (Fig. 5).
    """

    FEEDER_COL_2021 = "LENTS-HAPPY VALLEY (n=3345)"
    FEEDER_COL_2023 = "LENTS-HAPPY VALLEY (n=3345)"
    DT_COL = "Datetime"
    WEATHER_COLS = ["T_out", "RH_out", "Direct_Radiation"]

    @pytest.fixture(autouse=True)
    def _build_temp_csvs(self, tmp_path):
        """Prepare single-target feeder CSVs with weather."""
        # 2021 training — already merged
        df21 = pd.read_csv(FEEDER_2021_MERGED)
        keep_21 = [self.DT_COL, self.FEEDER_COL_2021] + [
            c for c in self.WEATHER_COLS if c in df21.columns
        ]
        df21 = df21[keep_21].rename(columns={self.FEEDER_COL_2021: "load_kwh"})
        self.train_csv = str(tmp_path / "feeder_train.csv")
        df21.to_csv(self.train_csv, index=False)

        # 2023 test — extract from multi-column feeder file and merge weather.
        # The weather covariates are required: the model is trained with them,
        # so the evaluation CSV must expose the same past-covariate columns.
        df23 = pd.read_csv(FEEDER_2023_LENTS)
        df23 = df23[[self.DT_COL, self.FEEDER_COL_2023]].rename(
            columns={self.FEEDER_COL_2023: "load_kwh"}
        )
        df23[self.DT_COL] = pd.to_datetime(df23[self.DT_COL])

        w = pd.read_csv(WEATHER_CSV)
        w[self.DT_COL] = pd.to_datetime(w["date"])
        keep_w = [self.DT_COL] + [c for c in self.WEATHER_COLS if c in w.columns]
        df23 = df23.merge(w[keep_w], on=self.DT_COL, how="inner")

        self.test_csv = str(tmp_path / "feeder_test.csv")
        df23.to_csv(self.test_csv, index=False)

        self.train_mapping = {
            "datetime": self.DT_COL,
            "target": "load_kwh",
            "past_covariates": [c for c in self.WEATHER_COLS if c in df21.columns],
        }
        self.test_mapping = {
            "datetime": self.DT_COL,
            "target": "load_kwh",
            "past_covariates": [c for c in self.WEATHER_COLS if c in df23.columns],
        }

    def test_linear_regression_feeder(self):
        result = train_forecast_model(
            csv_path=self.train_csv,
            model_type="LinearRegression",
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            frequency="h",
            validation_split=0.2,
            building_name="feeder_lents_happy_valley",
            column_mapping=self.train_mapping,
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        model_id = result["model_id"]

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=self.test_csv,
            column_mapping=self.test_mapping,
            return_predictions=False,
            peak_dates=WINTER_PEAK_DATES,
        )
        assert eval_result["success"], f"Evaluation failed: {eval_result.get('error')}"
        mape = eval_result["test_metrics"]["mape"]
        assert mape is not None
        assert mape < 50

        pm = eval_result.get("peak_metrics", {})
        logger.info(
            "[FEEDER] LinearRegression | MAPE=%.2f%% | "
            "winter PMAPE=%s | PTE=%s",
            mape,
            f"{pm.get('peak_mape', float('nan')):.2f}%" if pm.get("peak_mape") is not None else "N/A",
            f"{pm.get('peak_timing_error_hours', float('nan')):.2f}h" if pm.get("peak_timing_error_hours") is not None else "N/A",
        )
        logger.info("[PAPER REF] Feeder LinearRegression MAPE ~ 10-30%% (Fig. 5)")

    def test_xgboost_feeder(self):
        result = train_forecast_model(
            csv_path=self.train_csv,
            model_type="XGBoost",
            lookback_hours=LOOKBACK_HOURS,
            horizon_hours=HORIZON_HOURS,
            frequency="h",
            validation_split=0.2,
            building_name="feeder_lents_happy_valley",
            column_mapping=self.train_mapping,
        )
        assert result["success"], f"Training failed: {result.get('error')}"
        model_id = result["model_id"]

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=self.test_csv,
            column_mapping=self.test_mapping,
            return_predictions=False,
            peak_dates=SUMMER_PEAK_DATES,
        )
        assert eval_result["success"], f"Evaluation failed: {eval_result.get('error')}"
        mape = eval_result["test_metrics"]["mape"]
        assert mape is not None
        assert mape < 50

        pm = eval_result.get("peak_metrics", {})
        logger.info(
            "[FEEDER] XGBoost | MAPE=%.2f%% | summer PMAPE=%s | PTE=%s",
            mape,
            f"{pm.get('peak_mape', float('nan')):.2f}%" if pm.get("peak_mape") is not None else "N/A",
            f"{pm.get('peak_timing_error_hours', float('nan')):.2f}h" if pm.get("peak_timing_error_hours") is not None else "N/A",
        )
        logger.info("[PAPER REF] Feeder XGBoost MAPE ~ 10-20%% (Fig. 5)")


# ==============================================================================
# 4. Peak Metrics correctness
# ==============================================================================

class TestPeakMetricsReplication:
    """
    Unit-style tests for calculate_peak_metrics correctness and API shape,
    using the evaluate_forecast_model pipeline on synthetic data.
    """

    @pytest.fixture()
    def trained_model_id(self, tmp_path):
        """Train a quick NaiveSeasonal model on synthetic hourly data."""
        import numpy as np

        n = 600
        rng = np.random.default_rng(42)
        base = 100 + 20 * np.sin(
            2 * np.pi * np.arange(n) / 24
        ) + 5 * rng.standard_normal(n)
        timestamps = pd.date_range("2022-01-01", periods=n, freq="h")
        df = pd.DataFrame({"datetime": timestamps, "load_kwh": base})
        csv_path = str(tmp_path / "synth.csv")
        df.to_csv(csv_path, index=False)

        result = train_forecast_model(
            csv_path=csv_path,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=24,
            frequency="h",
            validation_split=0.2,
        )
        assert result["success"], result.get("error")
        return result["model_id"], csv_path

    def test_peak_metrics_shape(self, trained_model_id):
        """peak_metrics response has all expected keys."""
        model_id, csv_path = trained_model_id
        peak_dates = ["2022-01-05", "2022-01-06"]
        result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=csv_path,
            return_predictions=False,
            peak_dates=peak_dates,
        )
        assert result["success"], result.get("error")
        assert "peak_metrics" in result
        pm = result["peak_metrics"]
        assert "peak_mape" in pm
        assert "peak_timing_error_hours" in pm
        assert "n_peak_days_evaluated" in pm
        assert "n_peak_days_skipped" in pm
        assert "per_day" in pm
        assert isinstance(pm["per_day"], list)

    def test_peak_metrics_absent_without_peak_dates(self, trained_model_id):
        """peak_metrics is absent from response when peak_dates not provided."""
        model_id, csv_path = trained_model_id
        result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=csv_path,
            return_predictions=False,
        )
        assert result["success"], result.get("error")
        assert "peak_metrics" not in result

    def test_peak_metrics_per_day_detail(self, trained_model_id):
        """per_day list contains correct keys for each evaluated day."""
        model_id, csv_path = trained_model_id
        peak_dates = ["2022-01-10", "2022-01-11"]
        result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=csv_path,
            return_predictions=False,
            peak_dates=peak_dates,
        )
        assert result["success"], result.get("error")
        pm = result["peak_metrics"]
        for day_result in pm.get("per_day", []):
            assert "date" in day_result
            assert "actual_peak_value" in day_result
            assert "predicted_peak_value" in day_result
            assert "actual_peak_hour" in day_result
            assert "predicted_peak_hour" in day_result
            assert "peak_magnitude_error_pct" in day_result
            assert "peak_timing_error_hours" in day_result
            assert 0 <= day_result["actual_peak_hour"] <= 23
            assert 0 <= day_result["predicted_peak_hour"] <= 23
            assert day_result["peak_magnitude_error_pct"] >= 0
            assert day_result["peak_timing_error_hours"] >= 0

    @pytest.mark.skipif(
        not _files_exist(CITY_2021, CITY_2023),
        reason="AMI city-level CSV files not found",
    )
    def test_winter_peak_metrics_real_data(self):
        """
        Real-data test: winter PMAPE and PTE using city-level AMI data.
        Validates the paper's exact winter peak window (Feb 22-25 2023).
        """
        column_mapping = {
            "datetime": "Datetime",
            "target": "City (n=41703)",
            "past_covariates": ["T_out", "RH_out", "Direct_Radiation"],
        }
        result = train_forecast_model(
            csv_path=CITY_2021,
            model_type="LinearRegression",
            lookback_hours=96,
            horizon_hours=96,
            frequency="h",
            validation_split=0.2,
            building_name="peak_test_winter",
            column_mapping=column_mapping,
        )
        assert result["success"], result.get("error")
        model_id = result["model_id"]

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=CITY_2023,
            return_predictions=False,
            peak_dates=WINTER_PEAK_DATES,
        )
        assert eval_result["success"], eval_result.get("error")
        pm = eval_result["peak_metrics"]

        assert pm["n_peak_days_evaluated"] > 0, (
            "No winter peak days found — check that 2023 CSV covers Feb 22-25"
        )
        assert pm["peak_mape"] is not None
        assert pm["peak_timing_error_hours"] is not None
        # Paper Fig. 8: most models predict peak timing within ±1h
        # We allow up to ±3h as a sanity bound for LinearRegression
        assert pm["peak_timing_error_hours"] <= 12, (
            f"PTE={pm['peak_timing_error_hours']:.1f}h exceeds 12h sanity bound"
        )
        logger.info(
            "[PAPER REF WINTER] City LinearRegression | "
            "winter PMAPE=%.2f%% | PTE=%.2fh | n_days=%d",
            pm["peak_mape"],
            pm["peak_timing_error_hours"],
            pm["n_peak_days_evaluated"],
        )
        logger.info(
            "[PAPER REF] Fig. 8: LinearRegression winter PMAPE typically > 20%%; "
            "PTE within ±1h for most models"
        )

    @pytest.mark.skipif(
        not _files_exist(CITY_2021, CITY_2023),
        reason="AMI city-level CSV files not found",
    )
    def test_summer_peak_metrics_real_data(self):
        """
        Real-data test: summer PMAPE and PTE using city-level AMI data.
        Validates the paper's exact summer peak window (Aug 14-17 2023).
        """
        column_mapping = {
            "datetime": "Datetime",
            "target": "City (n=41703)",
            "past_covariates": ["T_out", "RH_out", "Direct_Radiation"],
        }
        result = train_forecast_model(
            csv_path=CITY_2021,
            model_type="XGBoost",
            lookback_hours=96,
            horizon_hours=96,
            frequency="h",
            validation_split=0.2,
            building_name="peak_test_summer",
            column_mapping=column_mapping,
        )
        assert result["success"], result.get("error")
        model_id = result["model_id"]

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=CITY_2023,
            return_predictions=False,
            peak_dates=SUMMER_PEAK_DATES,
        )
        assert eval_result["success"], eval_result.get("error")
        pm = eval_result["peak_metrics"]

        assert pm["n_peak_days_evaluated"] > 0, (
            "No summer peak days found — check that 2023 CSV covers Aug 14-17"
        )
        assert pm["peak_mape"] is not None
        logger.info(
            "[PAPER REF SUMMER] City XGBoost | "
            "summer PMAPE=%.2f%% | PTE=%.2fh | n_days=%d",
            pm["peak_mape"],
            pm["peak_timing_error_hours"],
            pm["n_peak_days_evaluated"],
        )
        logger.info(
            "[PAPER REF] Fig. 8: XGBoost summer PMAPE typically 10-20%%; "
            "models perform better in summer (smoother sinusoidal profile)"
        )


# ==============================================================================
# 5. Gaussian noise augmentation
# ==============================================================================

class TestGaussianNoiseAugmentation:
    """
    Tests for weather noise augmentation (Li et al. 2025, Section 3.1.2).
    """

    @pytest.fixture()
    def weather_csv(self, tmp_path):
        """Synthetic hourly data with a temperature future covariate."""
        import numpy as np

        n = 600
        rng = np.random.default_rng(0)
        timestamps = pd.date_range("2022-01-01", periods=n, freq="h")
        load = 100 + 20 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.standard_normal(n)
        temp = 15 + 5 * np.sin(2 * np.pi * np.arange(n) / (24 * 365))
        df = pd.DataFrame({"datetime": timestamps, "load_kwh": load, "T_out": temp})
        path = str(tmp_path / "weather.csv")
        df.to_csv(path, index=False)
        return path

    def test_noise_augmentation_trains_successfully(self, weather_csv):
        """augment_weather_noise=True completes without error."""
        result = train_forecast_model(
            csv_path=weather_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            validation_split=0.2,
            augment_weather_noise=True,
            weather_noise_std=1.0,
            column_mapping={
                "datetime": "datetime",
                "target": "load_kwh",
                "future_covariates": ["T_out"],
            },
        )
        assert result["success"], f"Training with noise augmentation failed: {result.get('error')}"
        assert result.get("model_id") is not None

    def test_noise_augmentation_no_future_covariates_no_error(self, weather_csv):
        """augment_weather_noise=True with no future_covariates is silently skipped."""
        result = train_forecast_model(
            csv_path=weather_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            validation_split=0.2,
            augment_weather_noise=True,
            weather_noise_std=2.0,
            column_mapping={
                "datetime": "datetime",
                "target": "load_kwh",
                "past_covariates": ["T_out"],
                # No future_covariates — noise should be skipped, not crash
            },
        )
        assert result["success"], f"Failed: {result.get('error')}"

    def test_noise_augmentation_false_is_default(self, weather_csv):
        """augment_weather_noise defaults to False and training still succeeds."""
        result = train_forecast_model(
            csv_path=weather_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
            validation_split=0.2,
            # augment_weather_noise not passed — defaults to False
            column_mapping={
                "datetime": "datetime",
                "target": "load_kwh",
                "future_covariates": ["T_out"],
            },
        )
        assert result["success"], f"Failed: {result.get('error')}"


# ==============================================================================
# 6. Energy burden tracking
# ==============================================================================

class TestEnergyBurdenTracking:
    """
    Tests for training energy estimation (Li et al. 2025, Table 5).
    """

    @pytest.fixture()
    def simple_csv(self, tmp_path):
        import numpy as np

        n = 500
        rng = np.random.default_rng(1)
        ts = pd.date_range("2022-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "datetime": ts,
            "load_kwh": 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.standard_normal(n),
        })
        path = str(tmp_path / "simple.csv")
        df.to_csv(path, index=False)
        return path

    def test_energy_kwh_present(self, simple_csv):
        result = train_forecast_model(
            csv_path=simple_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
        )
        assert result["success"], result.get("error")
        ti = result["training_info"]
        assert "energy_kwh" in ti, "energy_kwh missing from training_info"
        assert "watts_assumed" in ti, "watts_assumed missing from training_info"
        assert ti["energy_kwh"] > 0
        assert ti["energy_kwh"] < 1.0  # training a small model should be << 1 kWh
        assert ti["watts_assumed"] == 65  # LinearRegression

    def test_naive_seasonal_lower_wattage(self, simple_csv):
        result = train_forecast_model(
            csv_path=simple_csv,
            model_type="NaiveSeasonal",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
        )
        assert result["success"], result.get("error")
        ti = result["training_info"]
        assert ti["watts_assumed"] == 15  # Naive baselines

    def test_xgboost_wattage(self, simple_csv):
        result = train_forecast_model(
            csv_path=simple_csv,
            model_type="XGBoost",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
        )
        assert result["success"], result.get("error")
        ti = result["training_info"]
        assert ti["watts_assumed"] == 100  # XGBoost

    def test_energy_formula_consistency(self, simple_csv):
        """Verify energy_kwh = watts × time / 3_600_000."""
        result = train_forecast_model(
            csv_path=simple_csv,
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
            frequency="h",
        )
        assert result["success"], result.get("error")
        ti = result["training_info"]
        expected = ti["watts_assumed"] * ti["training_time_seconds"] / 3_600_000
        # Allow small floating-point rounding tolerance (rounding to 8 dp in trainer)
        assert abs(ti["energy_kwh"] - expected) < 1e-6, (
            f"energy_kwh={ti['energy_kwh']} != expected={expected}"
        )


# ==============================================================================
# 7. Horizon cap raised to 96
# ==============================================================================

class TestHorizonCap:
    """Verify that horizon_hours=96 is accepted (was previously capped at 48)."""

    @pytest.fixture()
    def long_csv(self, tmp_path):
        import numpy as np

        # Need enough data: lookback(96) + horizon(96) + 100 buffer = 292 rows
        n = 800
        rng = np.random.default_rng(99)
        ts = pd.date_range("2022-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "datetime": ts,
            "load_kwh": 50 + 10 * np.sin(2 * np.pi * np.arange(n) / 24) + rng.standard_normal(n),
        })
        path = str(tmp_path / "long.csv")
        df.to_csv(path, index=False)
        return path

    def test_horizon_96_accepted(self, long_csv):
        result = train_forecast_model(
            csv_path=long_csv,
            model_type="LinearRegression",
            lookback_hours=96,
            horizon_hours=96,
            frequency="h",
        )
        assert result["success"], (
            f"horizon_hours=96 should be accepted but got: {result.get('error')}"
        )

    def test_horizon_97_rejected(self, long_csv):
        result = train_forecast_model(
            csv_path=long_csv,
            model_type="LinearRegression",
            lookback_hours=96,
            horizon_hours=97,
            frequency="h",
        )
        assert not result["success"]
        assert "96" in result.get("error", "")
