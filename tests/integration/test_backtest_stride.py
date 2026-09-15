"""Backtest metrics must not depend on the stride.

Rolling windows overlap whenever stride < horizon.  The original implementation
flattened them with ``darts_concatenate(..., ignore_time_axis=True)``, which
lays every window end-to-end on a fabricated time axis of length
n_windows x horizon.  At stride == horizon the windows tile the axis exactly so
the fabricated axis coincided with the real one and the metrics were right; at
any shorter stride ``calculate_metrics`` then compared actual against predicted
at mismatched timestamps and returned nonsense (MAPE 9% -> 48% on real data).

Metrics are now pooled per window on real timestamps, so a shorter stride adds
forecast origins without changing the error level.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from load_forecasting.tools import backtest_model, train_forecast_model  # noqa: E402


@pytest.fixture
def temp_model_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


@pytest.fixture
def hourly_csv():
    idx = pd.date_range("2021-01-01", periods=4000, freq="h")
    hour = idx.hour.to_numpy()
    day = idx.dayofyear.to_numpy()
    load = (
        1000
        + 200 * np.sin(2 * np.pi * (hour - 8) / 24)
        + 80 * np.cos(2 * np.pi * day / 365)
        + np.random.default_rng(3).normal(0, 12, len(idx))
    )
    df = pd.DataFrame({"Datetime": idx, "load": load})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
    yield path
    os.unlink(path)


MAPPING = {"datetime": "Datetime", "target": "load"}


@pytest.fixture
def trained_model(hourly_csv, temp_model_dir):
    r = train_forecast_model(
        csv_path=hourly_csv,
        model_type="LinearRegression",
        lookback_hours=48,
        horizon_hours=24,
        frequency="h",
        column_mapping=MAPPING,
    )
    assert r["success"], r.get("error")
    return r["model_id"]


def _backtest(model_id, csv, stride):
    r = backtest_model(
        model_id=model_id,
        csv_path=csv,
        column_mapping=MAPPING,
        stride_hours=stride,
        start_fraction=0.2,
        return_predictions=False,
    )
    assert r["success"], r.get("error")
    return r


@pytest.mark.slow
@pytest.mark.parametrize("stride", [24, 12, 6, 1])
def test_backtest_metrics_are_sane_at_every_stride(
    stride, trained_model, hourly_csv
):
    m = _backtest(trained_model, hourly_csv, stride)["backtest_metrics"]
    assert m["mape"] is not None
    # The pre-fix bug produced MAPE in the tens of percent and negative R2 on a
    # series this predictable.
    assert m["mape"] < 15, f"stride={stride} MAPE={m['mape']}"
    assert m["r_squared"] > 0.5, f"stride={stride} R2={m['r_squared']}"


@pytest.mark.slow
def test_backtest_metrics_stride_invariant(trained_model, hourly_csv):
    """Shorter strides add origins; they must not shift the error level."""
    ref = _backtest(trained_model, hourly_csv, 24)["backtest_metrics"]
    for stride in (12, 6, 1):
        m = _backtest(trained_model, hourly_csv, stride)["backtest_metrics"]
        assert m["mape"] == pytest.approx(ref["mape"], rel=0.35), (
            f"stride={stride} MAPE {m['mape']} vs stride=24 {ref['mape']}"
        )
        assert m["cv_rmse"] == pytest.approx(ref["cv_rmse"], rel=0.35), (
            f"stride={stride} CV-RMSE {m['cv_rmse']} vs stride=24 {ref['cv_rmse']}"
        )


def test_shorter_stride_yields_more_windows(trained_model, hourly_csv):
    n24 = _backtest(trained_model, hourly_csv, 24)["backtest_summary"]["n_windows"]
    n6 = _backtest(trained_model, hourly_csv, 6)["backtest_summary"]["n_windows"]
    assert n6 > n24


@pytest.mark.slow
def test_flattened_predictions_stay_on_the_real_time_axis(
    trained_model, hourly_csv
):
    """Overlapping windows must not inflate the flattened prediction span."""
    r = backtest_model(
        model_id=trained_model,
        csv_path=hourly_csv,
        column_mapping=MAPPING,
        stride_hours=1,
        start_fraction=0.2,
        return_predictions=True,
    )
    assert r["success"], r.get("error")
    preds = r.get("predictions") or r.get("predictions_flat") or []
    if not preds:
        pytest.skip("tool did not return a flattened prediction list")
    ts = pd.to_datetime([p["t"] if "t" in p else p["timestamp"] for p in preds])
    source_end = pd.read_csv(hourly_csv, parse_dates=["Datetime"])["Datetime"].max()
    # Pre-fix, n_windows x horizon points were laid end-to-end and ran far past
    # the end of the source data.
    assert ts.max() <= source_end
    assert not ts.duplicated().any()
