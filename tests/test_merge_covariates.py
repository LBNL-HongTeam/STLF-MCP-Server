"""Tests for the merge_covariates MCP tool."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.tools import merge_covariates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(df: pd.DataFrame) -> str:
    """Write a DataFrame to a temp CSV and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


def _output_path() -> str:
    """Return a temp path for the merged output."""
    f = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    f.close()
    return f.name


def _make_load_df(periods: int = 24, freq: str = "h", start: str = "2021-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({"Datetime": idx, "load_kwh": range(periods)})


def _make_weather_df(periods: int = 24, freq: str = "h", start: str = "2021-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq)
    return pd.DataFrame({
        "date": idx,
        "T_out": [10.0 + i * 0.5 for i in range(periods)],
        "RH_out": [60.0 - i * 0.1 for i in range(periods)],
    })


# ---------------------------------------------------------------------------
# Basic success path
# ---------------------------------------------------------------------------

class TestMergeCovariatesBasic:

    @pytest.mark.asyncio
    async def test_simple_merge_returns_success(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
        )

        assert result["success"] is True
        assert result["error"] is None
        assert result["n_rows"] == 24
        assert result["n_missing_filled"] == 0
        assert "T_out" in result["covariate_columns"]
        assert "RH_out" in result["covariate_columns"]

    @pytest.mark.asyncio
    async def test_output_file_is_written(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
        )

        df = pd.read_csv(out_path)
        assert len(df) == 24
        assert "T_out" in df.columns
        assert "RH_out" in df.columns
        assert "load_kwh" in df.columns

    @pytest.mark.asyncio
    async def test_all_response_keys_present(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
        )

        for key in (
            "success", "error", "output_csv_path", "n_rows",
            "covariate_columns", "n_missing_filled", "start_date", "end_date",
            "timezone_conversion", "dst_duplicates_dropped",
        ):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------

class TestMergeCovariatesColumnSelection:

    @pytest.mark.asyncio
    async def test_select_subset_of_covariate_columns(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            covariate_columns=["T_out"],
        )

        assert result["success"] is True
        assert result["covariate_columns"] == ["T_out"]

        df = pd.read_csv(out_path)
        assert "T_out" in df.columns
        assert "RH_out" not in df.columns

    @pytest.mark.asyncio
    async def test_invalid_covariate_column_returns_error(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            covariate_columns=["nonexistent_col"],
        )

        assert result["success"] is False
        assert "nonexistent_col" in result["error"]


# ---------------------------------------------------------------------------
# Explicit datetime column names
# ---------------------------------------------------------------------------

class TestMergeCovariatesDatetimeCols:

    @pytest.mark.asyncio
    async def test_explicit_datetime_cols(self):
        load_df = _make_load_df().rename(columns={"Datetime": "ts"})
        cov_df = _make_weather_df().rename(columns={"date": "weather_ts"})
        load_path = _write_csv(load_df)
        cov_path = _write_csv(cov_df)
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            load_datetime_col="ts",
            covariate_datetime_col="weather_ts",
        )

        assert result["success"] is True
        assert result["n_rows"] == 24

    @pytest.mark.asyncio
    async def test_wrong_load_datetime_col_returns_error(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            load_datetime_col="wrong_col",
        )

        assert result["success"] is False
        assert "wrong_col" in result["error"]


# ---------------------------------------------------------------------------
# Missing files
# ---------------------------------------------------------------------------

class TestMergeCovariatesMissingFiles:

    @pytest.mark.asyncio
    async def test_missing_load_file_returns_error(self):
        result = merge_covariates(
            load_csv_path="/nonexistent/load.csv",
            covariate_csv_path=_write_csv(_make_weather_df()),
            output_csv_path=_output_path(),
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_covariate_file_returns_error(self):
        result = merge_covariates(
            load_csv_path=_write_csv(_make_load_df()),
            covariate_csv_path="/nonexistent/weather.csv",
            output_csv_path=_output_path(),
        )
        assert result["success"] is False
        assert "not found" in result["error"].lower()


# ---------------------------------------------------------------------------
# Timezone conversion
# ---------------------------------------------------------------------------

class TestMergeCovariatesTimezone:

    @pytest.mark.asyncio
    async def test_utc_to_pacific_conversion(self):
        """Weather in UTC (starting 2021-01-01 08:00 UTC = 2021-01-01 00:00 PST)."""
        # Load file: naive local timestamps, midnight PST
        load_df = _make_load_df(periods=24, start="2021-01-01 00:00")
        # Weather file: UTC timestamps, 8 hours ahead of PST
        weather_df = _make_weather_df(periods=32, start="2021-01-01 08:00")

        load_path = _write_csv(load_df)
        cov_path = _write_csv(weather_df)
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            covariate_timezone="UTC",
            load_timezone="America/Los_Angeles",
        )

        assert result["success"] is True
        assert result["timezone_conversion"] == "UTC → America/Los_Angeles"
        assert result["n_rows"] == 24

        df = pd.read_csv(out_path)
        # T_out values should be aligned to local time, not 8h shifted
        assert df["T_out"].notna().all()

    @pytest.mark.asyncio
    async def test_no_timezone_when_covariate_timezone_not_set(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
        )

        assert result["success"] is True
        assert result["timezone_conversion"] is None
        assert result["dst_duplicates_dropped"] == 0

    @pytest.mark.asyncio
    async def test_covariate_timezone_without_load_timezone_returns_error(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            covariate_timezone="UTC",
            # load_timezone intentionally omitted
        )

        assert result["success"] is False
        assert "load_timezone" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_timezone_returns_error(self):
        load_path = _write_csv(_make_load_df())
        cov_path = _write_csv(_make_weather_df())
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            covariate_timezone="NotATimezone/Invalid",
            load_timezone="America/Los_Angeles",
        )

        assert result["success"] is False
        assert "timezone" in result["error"].lower()


# ---------------------------------------------------------------------------
# DST handling
# ---------------------------------------------------------------------------

class TestMergeCovariatesDST:

    @pytest.mark.asyncio
    async def test_dst_fall_back_duplicate_dropped(self):
        """
        Simulate a DST fall-back: weather file has a repeated local hour
        (e.g. 01:00 appears twice). After UTC→local conversion the duplicate
        should be dropped and dst_duplicates_dropped should be 1.
        """
        # Build weather timestamps in UTC that include the 2021 fall-back:
        # 2021-11-07 08:00 UTC = 01:00 PDT, 2021-11-07 09:00 UTC = 01:00 PST
        utc_times = pd.date_range("2021-11-07 07:00", periods=6, freq="h")
        weather_df = pd.DataFrame({
            "date": utc_times,
            "T_out": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        })
        # Load is in local time — the local clock goes: 00, 01(PDT), 01(PST), 02, 03, 04
        # After conversion+dedup the weather has 5 unique local rows, not 6.
        local_times = [
            "2021-11-07 00:00",
            "2021-11-07 01:00",
            "2021-11-07 02:00",
            "2021-11-07 03:00",
            "2021-11-07 04:00",
        ]
        load_df = pd.DataFrame({
            "Datetime": pd.to_datetime(local_times),
            "load_kwh": range(5),
        })

        load_path = _write_csv(load_df)
        cov_path = _write_csv(weather_df)
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            covariate_timezone="UTC",
            load_timezone="America/Los_Angeles",
        )

        assert result["success"] is True
        assert result["dst_duplicates_dropped"] == 1

    @pytest.mark.asyncio
    async def test_dst_spring_forward_gap_is_filled(self):
        """
        Simulate a DST spring-forward: weather file has no row for the
        missing local hour (e.g. 02:00 doesn't exist). The load file does
        have that row. The gap should be forward-filled.
        """
        # Load has 2021-03-14 00:00 through 05:00 (6 rows incl. the missing 02:00)
        local_times = pd.date_range("2021-03-14 00:00", periods=6, freq="h")
        load_df = pd.DataFrame({
            "Datetime": local_times,
            "load_kwh": range(6),
        })

        # Weather in UTC: 2021-03-14 08:00 UTC = 00:00 PST
        # Spring-forward: clocks skip from 02:00 to 03:00, so 09:00 UTC maps to 01:00,
        # then 10:00 UTC maps to 03:00 (no 02:00 local exists in UTC source).
        utc_times = [
            "2021-03-14 08:00",  # → 00:00 local
            "2021-03-14 09:00",  # → 01:00 local
            # 10:00 UTC → 03:00 local (02:00 skipped)
            "2021-03-14 10:00",  # → 03:00 local
            "2021-03-14 11:00",  # → 04:00 local
            "2021-03-14 12:00",  # → 05:00 local
        ]
        weather_df = pd.DataFrame({
            "date": pd.to_datetime(utc_times),
            "T_out": [5.0, 6.0, 8.0, 9.0, 10.0],
        })

        load_path = _write_csv(load_df)
        cov_path = _write_csv(weather_df)
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
            covariate_timezone="UTC",
            load_timezone="America/Los_Angeles",
        )

        assert result["success"] is True
        assert result["n_missing_filled"] == 1  # the spring-forward hour

        df = pd.read_csv(out_path)
        assert df["T_out"].notna().all(), "Spring-forward gap was not filled"
        # The 02:00 row should be forward-filled from 01:00's value (6.0)
        assert df.loc[2, "T_out"] == 6.0


# ---------------------------------------------------------------------------
# Left-join preserves all load rows
# ---------------------------------------------------------------------------

class TestMergeCovariatesLeftJoin:

    @pytest.mark.asyncio
    async def test_load_rows_always_preserved(self):
        """Load has 48 hours; weather only covers first 24. All 48 rows kept."""
        load_df = _make_load_df(periods=48)
        cov_df = _make_weather_df(periods=24)  # only covers first 24h

        load_path = _write_csv(load_df)
        cov_path = _write_csv(cov_df)
        out_path = _output_path()

        result = merge_covariates(
            load_csv_path=load_path,
            covariate_csv_path=cov_path,
            output_csv_path=out_path,
        )

        assert result["success"] is True
        assert result["n_rows"] == 48
        # The second 24h had missing weather → filled
        assert result["n_missing_filled"] == 24

        df = pd.read_csv(out_path)
        assert len(df) == 48
        assert df["T_out"].notna().all()
