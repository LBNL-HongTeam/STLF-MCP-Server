"""Column auto-detection: patterns, target exclusions, content-aware datetime
pick, and the role hints inspect_data derives from them.

One assertion per row of the "uncertain covariates" review, so a future
pattern change has to consciously update the expectation here.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.data_loader import (
    DEFAULT_COVARIATE_PATTERNS,
    DEFAULT_TARGET_EXCLUDE_PATTERNS,
    DEFAULT_TARGET_PATTERNS,
    ForecastingDataLoader,
    auto_detect_columns,
)
from load_forecasting.tools import inspect_data


def _frame(**cols) -> pd.DataFrame:
    n = 48
    base = {"timestamp": pd.date_range("2023-06-01", periods=n, freq="h").astype(str)}
    for k, v in cols.items():
        base[k] = v if hasattr(v, "__len__") and not isinstance(v, str) else [v] * n
    return pd.DataFrame(base)


# ---------------------------------------------------------------------------
# Covariate patterns -- section A/B/C of the review
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("column", [
    # server's own bundle / Open-Meteo names
    "T_out", "T_out_app", "RH_out", "Direct_Radiation", "Diffuse_Radiation", "Cloud_Cover", "Rain",
    "Sunshine_Duration", "surface_pressure", "pressure_msl", "snowfall", "snow_depth", "precipitation",
    "dew_point", "wind_speed_10m", "shortwave_radiation", "direct_normal_irradiance",
    # other weather sources
    "hdd", "cdd", "heating_degree_days", "cooling_degree_hours", "wet_bulb", "wetbulb_temp", "enthalpy",
    "global_horizontal", "direct_normal", "diffuse_horizontal",
    # occupancy
    "occupancy", "occupants", "headcount", "people_count",
])
def test_recognised_as_past_covariate(column):
    df = _frame(load_kwh=1.0, **{column: 2.0})
    assert auto_detect_columns(df)["past_covariates"] == [column]


@pytest.mark.parametrize("column", [
    "Weather_Code", "weathercode", "Is_Daytime", "is_day", "uv_index", "visibility",
    "et0_fao_evapotranspiration", "illuminance", "price", "tariff", "is_holiday", "is_weekend", "misc_signal",
])
def test_deliberately_not_a_covariate(column):
    df = _frame(load_kwh=1.0, **{column: 2.0})
    assert auto_detect_columns(df)["past_covariates"] == []


def test_normalized_load_is_not_swept_in_by_direct_normal():
    """'direct_normal' is a full phrase; bare 'normal' must not match."""
    df = _frame(load_kwh=1.0, normalized_load=0.5)
    r = auto_detect_columns(df)
    assert r["target"] == "load_kwh" and r["past_covariates"] == []


# ---------------------------------------------------------------------------
# Target detection -- section D traps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("decoy", ["load_forecast", "forecast_kwh", "da_fcst_kw", "pred_load", "solar_kwh", "pv_kw", "generation_kwh"])
def test_forecast_and_generation_columns_never_become_target(decoy):
    # Decoy comes FIRST, which used to win under first-match.
    df = _frame(**{decoy: 5.0, "site_load_kw": 3.0})
    r = auto_detect_columns(df)
    assert r["target"] == "site_load_kw"
    assert decoy in r["target_excluded"]


def test_excluded_generation_column_can_still_be_a_covariate():
    df = _frame(solar_kwh=5.0, site_load_kw=3.0)
    r = auto_detect_columns(df)
    assert r["past_covariates"] == ["solar_kwh"]        # matches "solar"


@pytest.mark.parametrize("name", ["Total_kW", "MW", "site_mwh", "consumption", "usage_kwh", "demand_kw"])
def test_utility_style_target_names(name):
    df = _frame(T_out=1.0, **{name: 3.0})
    assert auto_detect_columns(df)["target"] == name


def test_text_column_with_target_name_is_skipped():
    df = _frame(load_type="base", load_kwh=3.0)
    assert auto_detect_columns(df)["target"] == "load_kwh"


def test_all_candidates_excluded_falls_back_to_first_numeric():
    df = _frame(load_forecast=5.0)
    r = auto_detect_columns(df)
    assert r["target"] == "load_forecast" and r["target_excluded"] == []


# ---------------------------------------------------------------------------
# Datetime detection -- the Is_Daytime trap
# ---------------------------------------------------------------------------

def test_numeric_daytime_flag_does_not_hijack_datetime():
    n = 48
    df = pd.DataFrame({
        "Is_Daytime": [1, 0] * (n // 2),
        "ts": pd.date_range("2023-06-01", periods=n, freq="h").astype(str),
        "load_kwh": 1.0,
    })
    r = auto_detect_columns(df)
    assert r["datetime"] == "ts"
    assert r["target"] == "load_kwh"
    assert "Is_Daytime" not in r["past_covariates"]


def test_datetime_by_content_when_no_name_matches():
    df = pd.DataFrame({"when": pd.date_range("2023-06-01", periods=48, freq="h").astype(str), "load_kwh": 1.0})
    assert auto_detect_columns(df)["datetime"] == "when"


def test_epoch_style_numeric_datetime_still_accepted_last():
    df = pd.DataFrame({"timestamp": np.arange(48) * 3600 + 1_700_000_000, "load_kwh": 1.0})
    assert auto_detect_columns(df)["datetime"] == "timestamp"


def test_loader_uses_shared_detector(tmp_path):
    n = 400
    df = pd.DataFrame({
        "Is_Daytime": [1, 0] * (n // 2),
        "ts": pd.date_range("2023-06-01", periods=n, freq="h"),
        "load_forecast": 5.0,
        "site_load_kw": np.random.default_rng(0).normal(100, 5, n),
        "T_out": 10.0,
    })
    df.to_csv(tmp_path / "t.csv", index=False)
    ldr = ForecastingDataLoader(str(tmp_path / "t.csv"), frequency="h", add_calendar_features=False, lag_hours=[])
    assert ldr.column_mapping["datetime"] == "ts"
    assert ldr.column_mapping["target"] == "site_load_kw"
    assert ldr.column_mapping["past_covariates"] == ["T_out"]
    assert ldr.target_excluded == ["load_forecast"]


def test_explicit_mapping_bypasses_detection(tmp_path):
    df = pd.DataFrame({"ts": pd.date_range("2023-06-01", periods=400, freq="h"), "a": 1.0, "b": 2.0})
    df.to_csv(tmp_path / "e.csv", index=False)
    ldr = ForecastingDataLoader(str(tmp_path / "e.csv"), column_mapping={"datetime": "ts", "target": "b"}, frequency="h",
                                add_calendar_features=False, lag_hours=[])
    assert ldr.column_mapping["target"] == "b" and ldr.target_excluded == []


# ---------------------------------------------------------------------------
# inspect_data hints -- section C mechanism
# ---------------------------------------------------------------------------

@pytest.fixture
def hint_csv(tmp_path):
    n = 400
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "ts": pd.date_range("2023-06-01", periods=n, freq="h"),
        "load_forecast": rng.normal(100, 5, n),
        "site_load_kw": rng.normal(100, 5, n),
        "T_out": rng.normal(20, 3, n),
        "Weather_Code": 61,
        "is_holiday": 0,
        "is_weekend": 0,
        "schedule_flag": 1,
        "misc_signal": rng.normal(0, 1, n),
    })
    p = tmp_path / "hints.csv"; df.to_csv(p, index=False); return p


def test_inspect_data_role_hints(hint_csv):
    r = inspect_data(str(hint_csv))
    assert r["success"], r.get("error")
    c = r["columns"]
    assert c["target"] == "site_load_kw" and c["past_covariates"] == ["T_out"]
    assert c["target_excluded"] == ["load_forecast"]
    assert set(c["future_covariate_candidates"]) == {"load_forecast", "is_holiday", "schedule_flag"}
    assert c["redundant_calendar_columns"] == ["is_weekend"]
    assert c["categorical_code_columns"] == ["Weather_Code"]

    flags = " ".join(r["quality_flags"])
    assert "CATEGORICAL_CODE" in flags and "Weather_Code" in flags

    sugg = r["suggestions"]
    assert any("skipped as target candidates" in s and "load_forecast" in s for s in sugg)
    assert any("future covariates" in s and "is_holiday" in s for s in sugg)
    assert any("duplicate calendar flags" in s and "is_weekend" in s for s in sugg)
    # The generic 'not mapped' suggestion only lists what the hints did not explain.
    leftover = [s for s in sugg if "were not mapped to any role" in s]
    assert len(leftover) == 1 and "misc_signal" in leftover[0]
    for explained in ("is_holiday", "is_weekend", "Weather_Code", "schedule_flag"):
        assert explained not in leftover[0]


def test_categorical_code_mapped_as_covariate_is_called_out(tmp_path):
    n = 400
    df = pd.DataFrame({"ts": pd.date_range("2023-06-01", periods=n, freq="h"), "load_kwh": 1.0, "temp_code": 3})
    p = tmp_path / "c.csv"; df.to_csv(p, index=False)
    r = inspect_data(str(p))
    assert "temp_code" in r["columns"]["past_covariates"]      # 'temp' matched
    flag = next(f for f in r["quality_flags"] if f.startswith("CATEGORICAL_CODE"))
    assert "currently mapped as covariates" in flag


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------

def test_pattern_lists_are_lowercase_and_unique():
    for lst in (DEFAULT_COVARIATE_PATTERNS, DEFAULT_TARGET_PATTERNS, DEFAULT_TARGET_EXCLUDE_PATTERNS):
        assert lst == [p.lower() for p in lst]
        assert len(lst) == len(set(lst))
