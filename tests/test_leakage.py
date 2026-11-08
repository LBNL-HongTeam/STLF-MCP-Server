"""Time-series leakage tests.

Information from the validation (or any later) period must not reach the
data a model is fitted on. These tests check the four places where that
could happen in the training pipeline:

  1. the train/validation split (sequential and seasonal),
  2. the scalers, which must be fitted on training rows only,
  3. the lag features, which must use only past target values,
  4. the post-split hook used for weather-noise augmentation, which must
     see training rows only.

A strictly increasing synthetic load is used so that any validation value
is larger than every training value; a scaler fitted on validation rows
would then map the training maximum below 1.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.data_loader import ForecastingDataLoader  # noqa: E402
from load_forecasting.tools.train import _prepare_training_series  # noqa: E402

SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}


def _trend_frame(periods: int = 24 * 365) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=periods, freq="h")
    t = np.arange(periods, dtype=float)
    return pd.DataFrame({
        "timestamp": idx,
        "load_kwh": 100.0 + 0.01 * t + 5.0 * np.sin(2 * np.pi * t / 24),
        "temp_c": 10.0 + 0.001 * t,
    })


def _write(df: pd.DataFrame) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


MAPPING = {"datetime": "timestamp", "target": "load_kwh", "future_covariates": ["temp_c"]}


@pytest.fixture(scope="module")
def year_csv():
    return _write(_trend_frame())


def _loader(path, **kw):
    return ForecastingDataLoader(csv_path=path, column_mapping=dict(MAPPING), **kw)


def _values(series) -> np.ndarray:
    if isinstance(series, (list, tuple)):
        return np.concatenate([s.values().ravel() for s in series])
    return series.values().ravel()


# 1. Splits ---------------------------------------------------------------

def test_sequential_split_validation_strictly_after_training(year_csv):
    train, val = _loader(year_csv).split_train_val(0.2)
    assert train.df.index.max() < val.df.index.min()
    assert train.df.index.intersection(val.df.index).empty


def test_seasonal_split_validation_after_training_in_each_season(year_csv):
    train, val = _loader(year_csv).split_train_val_seasonal(0.2)
    assert train.df.index.intersection(val.df.index).empty
    for season in ("winter", "spring", "summer", "fall"):
        tr = train.df.index[train.df.index.month.map(SEASON) == season]
        va = val.df.index[val.df.index.month.map(SEASON) == season]
        assert len(tr) and len(va)
        assert tr.max() < va.min(), season


# 2. Scalers --------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["LinearRegression", "TiDE"])
def test_scalers_fitted_on_training_rows_only(year_csv, model_type):
    """Training rows span [0, 1] after scaling, and validation rows exceed 1.

    LinearRegression and TiDE use the seasonal chunked split. Local models
    such as ARIMA use the sequential split, covered by the next test.
    """
    prep = _prepare_training_series(_loader(year_csv), 0.2, model_type=model_type)
    tr, va = _values(prep["train_series"]), _values(prep["val_series"])
    assert tr.min() == pytest.approx(0.0, abs=1e-9)
    assert tr.max() == pytest.approx(1.0, abs=1e-9)
    assert va.max() > 1.0  # a scaler refitted on validation data would give 1.0

    fut_tr = _values(prep["train_future_covariates"])
    assert fut_tr.min() == pytest.approx(0.0, abs=1e-9)
    assert fut_tr.max() == pytest.approx(1.0, abs=1e-9)

    # The scaler stored with the model is the training scaler.
    assert prep["scaler"] is prep["train_loader"].target_scaler
    assert prep["val_loader"].target_scaler is prep["train_loader"].target_scaler


def test_sequential_split_scaler_fitted_on_training_rows_only(year_csv):
    loader = _loader(year_csv)
    train, val = loader.split_train_val(0.2)
    tr, _, _ = train.to_darts_series(fit_scalers=True)
    val.target_scaler = train.target_scaler
    va, _, _ = val.to_darts_series(fit_scalers=False)
    assert _values(tr).max() == pytest.approx(1.0, abs=1e-9)
    assert _values(va).max() > 1.0  # a scaler refitted on validation data would give 1.0


# 3. Lag features ---------------------------------------------------------

def test_lag_features_unchanged_by_future_values():
    """Changing load after time k must not change any feature at or before k."""
    df = _trend_frame(24 * 30)
    k = 24 * 20
    altered = df.copy()
    altered.loc[k + 1:, "load_kwh"] *= 3.0

    a = _loader(_write(df), lag_hours=[24, 48, 168]).df
    b = _loader(_write(altered), lag_hours=[24, 48, 168]).df
    cols = [c for c in a.columns if c.startswith("lag_")]
    assert cols
    cutoff = a.index[k]
    pd.testing.assert_frame_equal(a.loc[:cutoff, cols], b.loc[:cutoff, cols])


def test_lag_feature_equals_past_target():
    df = _trend_frame(24 * 30)
    a = _loader(_write(df), lag_hours=[24], add_calendar_features=False).df
    np.testing.assert_allclose(
        a["lag_24h"].iloc[24:].to_numpy(), a["load_kwh"].iloc[:-24].to_numpy()
    )


# 4. Post-split hook (weather-noise augmentation) -------------------------

@pytest.mark.parametrize("model_type", ["LinearRegression", "ARIMA"])
def test_post_split_hook_sees_training_rows_only(year_csv, model_type):
    seen = {}

    def hook(train_loader):
        seen["index"] = train_loader.df.index.copy()

    prep = _prepare_training_series(
        _loader(year_csv), 0.2, post_split_hook=hook, model_type=model_type
    )
    assert seen["index"].equals(prep["train_loader"].df.index)
    assert seen["index"].intersection(prep["val_loader"].df.index).empty
