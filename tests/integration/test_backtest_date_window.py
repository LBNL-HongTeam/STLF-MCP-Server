"""Date-bounded training, evaluation and backtesting end to end.

Scenario from a real session: models trained on one year, backtested on 2023
"excluding November and December".  Before start_date/end_date and
backtest_start_date existed, the only control was start_fraction (skip 10-50%
of the series from the front), so the backtest always ran through Dec 31.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from load_forecasting.tools import (  # noqa: E402
    backtest_model,
    evaluate_forecast_model,
    generate_backtest_report,
    train_forecast_model,
)

MAPPING = {"datetime": "Datetime", "target": "load"}
LOOKBACK_H = 24
HORIZON_H = 24


def _frame(start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h")
    hour = idx.hour.to_numpy()
    day = idx.dayofyear.to_numpy()
    load = (
        1000
        + 200 * np.sin(2 * np.pi * (hour - 8) / 24)
        + 80 * np.cos(2 * np.pi * day / 365)
        + np.random.default_rng(7).normal(0, 12, len(idx))
    )
    df = pd.DataFrame({"Datetime": idx, "load": load})
    # Nov-Dec 2023 is unusable (missing meter data), as in the real file.
    df.loc[df["Datetime"] >= "2023-11-01", "load"] = np.nan
    return df


def _write(df: pd.DataFrame) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        return f.name


@pytest.fixture(scope="module")
def model_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.environ.get("LOAD_FORECASTING_MODEL_DIR")
        os.environ["LOAD_FORECASTING_MODEL_DIR"] = tmpdir
        yield tmpdir
        if old is None:
            os.environ.pop("LOAD_FORECASTING_MODEL_DIR", None)
        else:
            os.environ["LOAD_FORECASTING_MODEL_DIR"] = old


@pytest.fixture(scope="module")
def two_year_csv():
    path = _write(_frame("2022-01-01", "2023-12-31 23:00"))
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def year_2023_csv():
    path = _write(_frame("2023-01-01", "2023-12-31 23:00"))
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def model_id(two_year_csv, model_dir):
    # Train on 2022 only, from a file that also contains 2023.
    r = train_forecast_model(
        csv_path=two_year_csv,
        model_type="LinearRegression",
        lookback_hours=LOOKBACK_H,
        horizon_hours=HORIZON_H,
        column_mapping=MAPPING,
        end_date="2022-12-31",
    )
    assert r["success"], r.get("error")
    assert r["data_summary"]["end_date"].startswith("2022-12-31T23:00")
    assert r["data_summary"]["date_window"]["end_date"] == "2022-12-31"
    return r["model_id"]


def _naive(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(None) if t.tzinfo else t


def _bt(model_id, csv, **kw):
    return backtest_model(
        model_id=model_id, csv_path=csv, column_mapping=MAPPING,
        return_predictions=True, **kw,
    )


def test_full_2023_fails_without_window(model_id, year_2023_csv):
    # The excluded tail is bad data: without end_date the load is rejected.
    r = _bt(model_id, year_2023_csv)
    assert not r["success"]
    assert "non-null" in r["error"]


def test_end_date_excludes_nov_dec(model_id, year_2023_csv):
    r = _bt(model_id, year_2023_csv, end_date="2023-10-31", start_fraction=0.1)
    assert r["success"], r.get("error")
    summary = r["backtest_summary"]
    assert summary["backtest_end_date"].startswith("2023-10-31T23:00")
    assert summary["date_window"]["rows_excluded"] == 61 * 24
    assert summary["start_mode"] == "fraction"
    last = max(_naive(p["timestamp"]) for p in r["predictions"])
    assert last <= pd.Timestamp("2023-10-31 23:00")


def test_backtest_start_date_too_early_names_earliest(model_id, year_2023_csv):
    # File starts Jan 1, so lag_168h is forward-filled for the first week;
    # first forecast needs that week plus the 24 h lookback -> Jan 9 00:00.
    r = _bt(model_id, year_2023_csv, end_date="2023-10-31", backtest_start_date="2023-01-01")
    assert not r["success"]
    assert "2023-01-09T00:00" in r["error"]


def test_backtest_start_date_at_earliest(model_id, year_2023_csv):
    r = _bt(model_id, year_2023_csv, end_date="2023-10-31", backtest_start_date="2023-01-09")
    assert r["success"], r.get("error")
    s = r["backtest_summary"]
    assert s["start_mode"] == "date"
    assert s["start_fraction"] is None
    assert s["backtest_start_date"].startswith("2023-01-09T00:00")
    assert s["earliest_valid_backtest_start_date"].startswith("2023-01-09T00:00")
    first = min(_naive(p["timestamp"]) for p in r["predictions"])
    assert first == pd.Timestamp("2023-01-09 00:00")


def test_start_date_buffer_allows_earlier_backtest(model_id, two_year_csv):
    # With 2022 history in the file, start_date keeps a lag buffer before
    # 2023-01-01, so only the lookback is needed: earliest is Jan 2 00:00.
    r = _bt(
        model_id, two_year_csv,
        start_date="2023-01-01", end_date="2023-10-31",
        backtest_start_date="2023-01-02",
    )
    assert r["success"], r.get("error")
    s = r["backtest_summary"]
    assert s["earliest_valid_backtest_start_date"].startswith("2023-01-02T00:00")
    assert s["date_window"]["first_timestamp"].startswith("2023-01-01T00:00")
    # Scored on 2023 only even though the model's training year is in the file.
    first = min(_naive(p["timestamp"]) for p in r["predictions"])
    assert first == pd.Timestamp("2023-01-02 00:00")


def test_both_start_controls_rejected(model_id, year_2023_csv):
    r = _bt(
        model_id, year_2023_csv, end_date="2023-10-31",
        start_fraction=0.2, backtest_start_date="2023-02-01",
    )
    assert not r["success"]
    assert "not both" in r["error"]


def test_default_start_fraction_unchanged(model_id, year_2023_csv):
    r = _bt(model_id, year_2023_csv, end_date="2023-10-31")
    assert r["success"], r.get("error")
    assert r["backtest_summary"]["start_fraction"] == 0.2


def test_backtest_start_after_data_errors(model_id, year_2023_csv):
    r = _bt(model_id, year_2023_csv, end_date="2023-10-31", backtest_start_date="2023-11-15")
    assert not r["success"]
    assert "after the last timestamp" in r["error"]


def test_evaluate_with_window(model_id, year_2023_csv):
    r = evaluate_forecast_model(
        model_id=model_id, csv_path=year_2023_csv, column_mapping=MAPPING,
        return_predictions=False, start_date="2023-03-01", end_date="2023-10-31",
    )
    assert r["success"], r.get("error")
    ts = r["test_summary"]
    assert ts["start_date"].startswith("2023-03-01T00:00")
    assert ts["end_date"].startswith("2023-10-31T23:00")
    assert ts["date_window"]["start_date"] == "2023-03-01"


def test_backtest_report_with_window(model_id, year_2023_csv, tmp_path):
    out = tmp_path / "bt.html"
    r = generate_backtest_report(
        model_id=model_id, csv_path=year_2023_csv, output_html_path=str(out),
        column_mapping=MAPPING, end_date="2023-10-31", backtest_start_date="2023-01-09",
    )
    assert r["success"], r.get("error")
    assert out.exists() and out.stat().st_size > 0
