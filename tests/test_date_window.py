"""start_date / end_date selection (core.date_window + ForecastingDataLoader + inspect_data).

The motivating case: "backtest on 2023 excluding November and December", where
the excluded tail is also bad data (missing load, forward-filled weather).  The
window must (a) include the whole end day, (b) drop the tail *before* any
validation so it cannot fail the load, and (c) keep lag features real at the
start of the window by computing them from the history just before it.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from load_forecasting.core.data_loader import DataLoadError, ForecastingDataLoader  # noqa: E402
from load_forecasting.core.date_window import DateWindow, DateWindowError  # noqa: E402
from load_forecasting.tools import inspect_data  # noqa: E402

MAPPING = {"datetime": "Datetime", "target": "load"}


def _year_frame(nan_from: str | None = None) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
    load = 1000 + 200 * np.sin(2 * np.pi * idx.hour.to_numpy() / 24) + np.arange(len(idx)) * 0.01
    df = pd.DataFrame({"Datetime": idx, "load": load})
    if nan_from:
        df.loc[df["Datetime"] >= nan_from, "load"] = np.nan
    return df


@pytest.fixture
def csv_factory():
    paths = []

    def make(df: pd.DataFrame) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            df.to_csv(f.name, index=False)
            paths.append(f.name)
            return f.name

    yield make
    for p in paths:
        os.unlink(p)


# ---------------------------------------------------------------------------
# DateWindow semantics
# ---------------------------------------------------------------------------

class TestDateWindow:
    idx = pd.date_range("2023-10-31 20:00", "2023-11-01 03:00", freq="h", tz="UTC")

    def test_no_bounds_is_none(self):
        assert DateWindow.from_args(None, None) is None
        assert DateWindow.from_args("", "") is None

    def test_date_only_end_includes_whole_day(self):
        w = DateWindow.from_args(None, "2023-10-31")
        kept = self.idx[w.mask(self.idx)]
        assert kept.max() == pd.Timestamp("2023-10-31 23:00", tz="UTC")
        assert pd.Timestamp("2023-11-01 00:00", tz="UTC") not in kept

    def test_datetime_end_is_inclusive_instant(self):
        w = DateWindow.from_args(None, "2023-10-31 22:00")
        assert self.idx[w.mask(self.idx)].max() == pd.Timestamp("2023-10-31 22:00", tz="UTC")

    def test_iso_t_separator_counts_as_time(self):
        w = DateWindow.from_args(None, "2023-10-31T22:00")
        assert self.idx[w.mask(self.idx)].max() == pd.Timestamp("2023-10-31 22:00", tz="UTC")

    def test_start_inclusive(self):
        w = DateWindow.from_args("2023-11-01", None)
        assert self.idx[w.mask(self.idx)].min() == pd.Timestamp("2023-11-01 00:00", tz="UTC")

    def test_naive_index(self):
        naive = self.idx.tz_localize(None)
        w = DateWindow.from_args("2023-10-31 21:00", "2023-10-31")
        kept = naive[w.mask(naive)]
        assert kept.min() == pd.Timestamp("2023-10-31 21:00")
        assert kept.max() == pd.Timestamp("2023-10-31 23:00")

    def test_offset_bound_is_converted(self):
        # 2023-10-31 16:00-07:00 == 23:00 UTC
        w = DateWindow.from_args(None, "2023-10-31T16:00-07:00")
        assert self.idx[w.mask(self.idx)].max() == pd.Timestamp("2023-10-31 23:00", tz="UTC")

    def test_start_offset_keeps_history(self):
        w = DateWindow.from_args("2023-11-01", None)
        kept = self.idx[w.mask(self.idx, start_offset=pd.Timedelta(hours=2))]
        assert kept.min() == pd.Timestamp("2023-10-31 22:00", tz="UTC")

    def test_start_after_end_rejected(self):
        with pytest.raises(DateWindowError, match="empty"):
            DateWindow.from_args("2023-11-01", "2023-10-31")

    def test_same_day_window_allowed(self):
        assert DateWindow.from_args("2023-10-31", "2023-10-31") is not None

    def test_invalid_string_rejected(self):
        with pytest.raises(DateWindowError, match="not a valid date"):
            DateWindow.from_args("Halloween", None)


# ---------------------------------------------------------------------------
# ForecastingDataLoader
# ---------------------------------------------------------------------------

class TestLoaderWindow:
    def test_end_date_trims_and_reports(self, csv_factory):
        path = csv_factory(_year_frame())
        loader = ForecastingDataLoader(path, column_mapping=MAPPING, end_date="2023-10-31")
        assert loader.df.index.max() == pd.Timestamp("2023-10-31 23:00", tz="UTC")
        info = loader.get_data_summary()["date_window"]
        assert info["rows_in_window"] == len(loader.df) == 304 * 24
        assert info["rows_excluded"] == (365 - 304) * 24
        assert info["end_date"] == "2023-10-31"
        assert info["file_end"].startswith("2023-12-31T23:00")

    def test_bad_excluded_tail_does_not_fail_load(self, csv_factory):
        # Nov-Dec load missing -> 83% coverage, below the 90% minimum.
        path = csv_factory(_year_frame(nan_from="2023-11-01"))
        with pytest.raises(DataLoadError, match="non-null"):
            ForecastingDataLoader(path, column_mapping=MAPPING)
        loader = ForecastingDataLoader(path, column_mapping=MAPPING, end_date="2023-10-31")
        assert not loader.df["load"].isna().any()

    def test_start_date_lags_use_real_history(self, csv_factory):
        df = _year_frame()
        path = csv_factory(df)
        loader = ForecastingDataLoader(path, column_mapping=MAPPING, start_date="2023-03-01")
        first = loader.df.index.min()
        assert first == pd.Timestamp("2023-03-01 00:00", tz="UTC")
        raw = df.set_index("Datetime")["load"]
        expected = raw[pd.Timestamp("2023-03-01") - pd.Timedelta(hours=168)]
        assert loader.df["lag_168h"].iloc[0] == pytest.approx(expected)
        # Lag features are real from the first kept row.
        assert loader.lag_warmup_until <= first

    def test_no_window_leaves_data_untouched(self, csv_factory):
        path = csv_factory(_year_frame())
        loader = ForecastingDataLoader(path, column_mapping=MAPPING)
        assert len(loader.df) == 365 * 24
        assert "date_window" not in loader.get_data_summary()
        assert loader.lag_warmup_until == pd.Timestamp("2023-01-08 00:00", tz="UTC")

    def test_window_outside_data_errors(self, csv_factory):
        path = csv_factory(_year_frame())
        with pytest.raises(DataLoadError, match="No rows fall inside"):
            ForecastingDataLoader(path, column_mapping=MAPPING, start_date="2024-01-01")

    def test_invalid_date_is_data_load_error(self, csv_factory):
        path = csv_factory(_year_frame())
        with pytest.raises(DataLoadError, match="not a valid date"):
            ForecastingDataLoader(path, column_mapping=MAPPING, end_date="soon")


# ---------------------------------------------------------------------------
# inspect_data
# ---------------------------------------------------------------------------

class TestInspectWindow:
    def test_window_unblocks_bad_tail(self, csv_factory):
        path = csv_factory(_year_frame(nan_from="2023-11-01"))
        full = inspect_data(path, column_mapping=MAPPING)
        assert full["success"] and full["blocking_issues"]

        win = inspect_data(path, column_mapping=MAPPING, end_date="2023-10-31")
        assert win["success"], win.get("error")
        assert win["blocking_issues"] == []
        assert win["n_rows_raw"] == 304 * 24
        assert win["date_window"]["rows_excluded"] == (365 - 304) * 24
        assert win["time_range"]["end"].startswith("2023-10-31T23:00")
        # Excluded rows are not reported as duplicates.
        assert win["time_range"]["n_duplicates_removed"] == 0

    def test_window_auto_detects_datetime_column(self, csv_factory):
        path = csv_factory(_year_frame())
        r = inspect_data(path, start_date="2023-06-01", end_date="2023-06-30")
        assert r["success"], r.get("error")
        assert r["n_rows_raw"] == 30 * 24

    def test_empty_window_errors(self, csv_factory):
        path = csv_factory(_year_frame())
        r = inspect_data(path, column_mapping=MAPPING, start_date="2030-01-01")
        assert not r["success"] and "No rows fall inside" in r["error"]

    def test_invalid_window_errors(self, csv_factory):
        path = csv_factory(_year_frame())
        r = inspect_data(path, column_mapping=MAPPING, start_date="2023-12-01", end_date="2023-01-01")
        assert not r["success"] and "empty" in r["error"]
