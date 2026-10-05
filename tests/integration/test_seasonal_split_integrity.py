"""Integration tests: the training split must never fabricate data.

``split_train_val_seasonal`` produces a non-contiguous frame by construction
(the held-out 20% of each season leaves a multi-week hole).  Converting that
frame to a single regular-frequency Darts TimeSeries interpolates straight-line
load across those holes, which at hourly resolution fabricated ~16% of the
training year and ~69% of the validation year.

``train_forecast_model`` therefore routes global models through
``to_darts_series_chunks`` (a list of gap-free per-season series, which Darts
global models fit natively) and falls back to a contiguous sequential split for
single-series models.  Either way the number of interpolated steps must be zero.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from load_forecasting.core.trainer import MULTI_SERIES_MODELS  # noqa: E402
from load_forecasting.tools import train_forecast_model  # noqa: E402


@pytest.fixture
def temp_model_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


@pytest.fixture
def full_year_csv():
    """One full year of hourly load + a weather column, all four seasons."""
    idx = pd.date_range("2021-01-01", periods=8760, freq="h")
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()
    temp = 12 + 10 * np.sin(2 * np.pi * (doy - 100) / 365) + 4 * np.sin(2 * np.pi * hour / 24)
    load = (
        50000
        + 8000 * np.sin(2 * np.pi * (hour - 8) / 24)
        + 6000 * np.cos(2 * np.pi * doy / 365)
        + 300 * temp
        + np.random.default_rng(0).normal(0, 800, len(idx))
    )
    df = pd.DataFrame({"Datetime": idx, "load": load, "T_out": temp})
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    yield path
    os.unlink(path)


MAPPING = {"datetime": "Datetime", "target": "load", "future_covariates": ["T_out"]}


@pytest.mark.parametrize("model_type", ["LinearRegression", "XGBoost"])
def test_global_models_use_gap_free_seasonal_chunks(
    model_type, full_year_csv, temp_model_dir
):
    result = train_forecast_model(
        csv_path=full_year_csv,
        model_type=model_type,
        lookback_hours=48,
        horizon_hours=24,
        frequency="h",
        column_mapping=MAPPING,
    )
    assert result["success"], result.get("error")
    split = result["data_summary"]["split"]
    assert split["strategy"] == "seasonal_chunked"
    assert split["train_chunks"] > 1
    assert split["validation_chunks"] > 1
    assert split["interpolated_train_steps"] == 0
    assert split["interpolated_validation_steps"] == 0


@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["NaiveSeasonal", "NaiveMean"])
def test_single_series_models_fall_back_to_contiguous_split(
    model_type, full_year_csv, temp_model_dir
):
    result = train_forecast_model(
        csv_path=full_year_csv,
        model_type=model_type,
        lookback_hours=48,
        horizon_hours=24,
        frequency="h",
        column_mapping=MAPPING,
    )
    assert result["success"], result.get("error")
    split = result["data_summary"]["split"]
    assert split["strategy"] == "sequential"
    assert split["train_chunks"] == 1
    assert split["interpolated_train_steps"] == 0
    assert split["interpolated_validation_steps"] == 0


@pytest.mark.slow
def test_no_model_type_silently_interpolates(full_year_csv, temp_model_dir):
    """Whichever branch a model takes, synthetic rows must be zero."""
    for model_type in ["LinearRegression", "NaiveMean"]:
        result = train_forecast_model(
            csv_path=full_year_csv,
            model_type=model_type,
            lookback_hours=48,
            horizon_hours=24,
            frequency="h",
            column_mapping=MAPPING,
        )
        assert result["success"], result.get("error")
        split = result["data_summary"]["split"]
        assert split["interpolated_train_steps"] == 0, model_type
        assert split["interpolated_validation_steps"] == 0, model_type


def test_validation_metrics_are_computed_on_real_data(full_year_csv, temp_model_dir):
    """A model that fits this smooth synthetic series should validate well.

    Under the old interpolating path the validation series was ~69% synthetic
    straight lines, so this metric measured mostly ramp-following.
    """
    result = train_forecast_model(
        csv_path=full_year_csv,
        model_type="LinearRegression",
        lookback_hours=48,
        horizon_hours=24,
        frequency="h",
        column_mapping=MAPPING,
    )
    assert result["success"], result.get("error")
    cv_rmse = result["validation_metrics"]["cv_rmse"]
    assert cv_rmse is not None
    assert 0 < cv_rmse < 50


@pytest.mark.slow
def test_torch_model_accepts_chunk_lists_and_runs_val_loop(
    full_year_csv, temp_model_dir
):
    """Torch models must fit on a sequence and validate on a sequence.

    Darts raises if any series in a val_series sequence is shorter than
    lookback + horizon, and the covariates must be broadcast to match the
    number of target chunks — this exercises both.  A recorded learning curve
    proves the Lightning validation loop actually ran on the chunk list.
    """
    result = train_forecast_model(
        csv_path=full_year_csv,
        model_type="TiDE",
        lookback_hours=48,
        horizon_hours=24,
        frequency="h",
        column_mapping=MAPPING,
        device="cpu",
    )
    assert result["success"], result.get("error")
    split = result["data_summary"]["split"]
    assert split["strategy"] == "seasonal_chunked"
    assert split["train_chunks"] > 1
    assert split["interpolated_train_steps"] == 0
    assert split["interpolated_validation_steps"] == 0
    history = (result.get("training_info") or {}).get("training_history", {})
    assert history.get("epochs"), "Lightning validation loop did not run"


@pytest.mark.slow
def test_multi_series_gate_matches_split_strategy(full_year_csv, temp_model_dir):
    """MULTI_SERIES_MODELS membership must decide the split strategy."""
    for model_type in ["XGBoost", "NaiveMean"]:
        result = train_forecast_model(
            csv_path=full_year_csv,
            model_type=model_type,
            lookback_hours=48,
            horizon_hours=24,
            frequency="h",
            column_mapping=MAPPING,
        )
        assert result["success"], result.get("error")
        expected = (
            "seasonal_chunked" if model_type in MULTI_SERIES_MODELS else "sequential"
        )
        assert result["data_summary"]["split"]["strategy"] == expected
