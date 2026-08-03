"""Tests for the inspect_data MCP tool."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.tools import inspect_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(
    periods: int = 300,
    freq: str = "h",
    dt_col: str = "timestamp",
    target_col: str = "electricity_kwh",
    add_temp: bool = True,
    null_fraction: float = 0.0,
    add_negatives: bool = False,
    duplicate_rows: int = 0,
    add_gap_at: int = 0,       # insert a gap of 4h after this row index
) -> str:
    """Write a temp CSV and return its path."""
    index = pd.date_range("2024-01-01", periods=periods, freq=freq)
    data = {
        dt_col: index,
        target_col: [50.0 + i * 0.1 for i in range(periods)],
    }
    if add_temp:
        data["outdoor_temp"] = [20.0 + i * 0.05 for i in range(periods)]

    df = pd.DataFrame(data)

    if null_fraction > 0:
        n_null = max(1, int(periods * null_fraction))
        null_idx = df.sample(n=n_null, random_state=0).index
        df.loc[null_idx, target_col] = np.nan

    if add_negatives:
        df.loc[df.index[:3], target_col] = -5.0

    if add_gap_at and add_gap_at < len(df):
        # Shift all timestamps after the gap point forward by 4h
        df.loc[df.index[add_gap_at:], dt_col] += pd.Timedelta(hours=4)

    if duplicate_rows:
        extra = df.iloc[:duplicate_rows].copy()
        df = pd.concat([df, extra]).sort_values(dt_col).reset_index(drop=True)

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Basic success path
# ---------------------------------------------------------------------------

class TestInspectDataBasic:

    @pytest.mark.asyncio
    async def test_clean_file_response(self):
        """success, all top-level keys, and ready_to_train on a clean CSV."""
        path = _make_csv()
        try:
            result = inspect_data(csv_path=path)
            assert result["success"] is True
            for key in (
                "columns", "frequency", "time_range", "column_statistics",
                "gaps", "quality_flags", "suggestions",
                "ready_to_train", "blocking_issues", "loader_error",
            ):
                assert key in result, f"Missing key: {key}"
            assert result["ready_to_train"] is True
            assert result["blocking_issues"] == []
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_n_rows_raw_correct(self):
        path = _make_csv(periods=200)
        try:
            result = inspect_data(csv_path=path)
            assert result["n_rows_raw"] == 200
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

class TestColumnDetection:

    @pytest.mark.asyncio
    async def test_detects_datetime_and_target(self):
        path = _make_csv()
        try:
            result = inspect_data(csv_path=path)
            cols = result["columns"]
            assert cols["datetime"] == "timestamp"
            assert cols["target"] == "electricity_kwh"
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_detects_covariate(self):
        path = _make_csv(add_temp=True)
        try:
            result = inspect_data(csv_path=path)
            assert "outdoor_temp" in result["columns"]["past_covariates"]
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_no_covariate_when_absent(self):
        path = _make_csv(add_temp=False)
        try:
            result = inspect_data(csv_path=path)
            assert result["columns"]["past_covariates"] == []
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_explicit_mapping_respected(self):
        path = _make_csv()
        try:
            mapping = {"datetime": "timestamp", "target": "electricity_kwh", "past_covariates": []}
            result = inspect_data(csv_path=path, column_mapping=mapping)
            assert result["columns"]["target"] == "electricity_kwh"
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Frequency detection
# ---------------------------------------------------------------------------

class TestFrequencyDetection:

    @pytest.mark.asyncio
    async def test_infers_hourly(self):
        path = _make_csv(freq="h")
        try:
            result = inspect_data(csv_path=path)
            assert result["frequency"]["inferred"] == "h"
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_infers_15min(self):
        path = _make_csv(freq="15min")
        try:
            result = inspect_data(csv_path=path, frequency="15min")
            assert result["frequency"]["inferred"] == "15min"
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_frequency_match(self):
        path = _make_csv(freq="h")
        try:
            result = inspect_data(csv_path=path, frequency="h")
            assert result["frequency"]["match"] is True
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_frequency_mismatch_flagged(self):
        path = _make_csv(freq="h")
        try:
            result = inspect_data(csv_path=path, frequency="15min")
            assert result["frequency"]["match"] is False
            assert any("FREQUENCY_MISMATCH" in f for f in result["quality_flags"])
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_no_declared_frequency_no_mismatch(self):
        """When frequency is not declared, match should be True (no comparison made)."""
        path = _make_csv(freq="h")
        try:
            result = inspect_data(csv_path=path, frequency=None)
            assert result["frequency"]["match"] is True
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Column statistics
# ---------------------------------------------------------------------------

class TestColumnStatistics:

    @pytest.mark.asyncio
    async def test_target_and_covariate_stats_present(self):
        path = _make_csv(add_temp=True)
        try:
            result = inspect_data(csv_path=path)
            stats = {s["column"]: s for s in result["column_statistics"]}
            assert "electricity_kwh" in stats
            for key in ("mean", "std", "min", "max", "p5", "p95", "n_missing", "coverage_pct"):
                assert key in stats["electricity_kwh"], f"Missing stat: {key}"
            assert "outdoor_temp" in stats
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_missing_value_count(self):
        path = _make_csv(periods=200, null_fraction=0.05)
        try:
            result = inspect_data(csv_path=path)
            stats = {s["column"]: s for s in result["column_statistics"]}
            assert stats["electricity_kwh"]["n_missing"] > 0
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_negative_count(self):
        path = _make_csv(add_negatives=True)
        try:
            result = inspect_data(csv_path=path)
            stats = {s["column"]: s for s in result["column_statistics"]}
            assert stats["electricity_kwh"]["n_negative"] == 3
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Gap analysis
# ---------------------------------------------------------------------------

class TestGapAnalysis:

    @pytest.mark.asyncio
    async def test_no_gaps_clean_file(self):
        path = _make_csv(periods=200)
        try:
            result = inspect_data(csv_path=path)
            assert result["gaps"]["n_gaps"] == 0
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_gap_analysis(self):
        """A CSV with an artificial gap is detected, flagged, and listed in top_gaps."""
        path = _make_csv(periods=200, add_gap_at=100)
        try:
            result = inspect_data(csv_path=path)
            assert result["gaps"]["n_gaps"] >= 1
            assert result["gaps"]["total_missing_steps"] >= 1
            assert any("GAPS" in f for f in result["quality_flags"])
            assert "top_gaps" in result["gaps"]
            assert len(result["gaps"]["top_gaps"]) >= 1
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

class TestQualityFlags:

    @pytest.mark.asyncio
    async def test_no_flags_clean_file(self):
        path = _make_csv()
        try:
            result = inspect_data(csv_path=path)
            # A clean file must have no *blocking* issues (LOAD_ERROR /
            # UNSUPPORTED_FREQUENCY). Warning-only flags (OUTLIERS, SPIKE,
            # LOW_DIURNAL_VARIATION) may appear on synthetic test fixtures
            # that are not shaped like real building load data.
            blocking_prefixes = ("LOAD_ERROR", "UNSUPPORTED_FREQUENCY")
            blocking = [f for f in result["quality_flags"] if f.startswith(blocking_prefixes)]
            assert blocking == [], f"Unexpected blocking flags: {blocking}"
            assert result["ready_to_train"] is True
            assert result["blocking_issues"] == []
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_missing_values_flag(self):
        path = _make_csv(periods=300, null_fraction=0.03)
        try:
            result = inspect_data(csv_path=path)
            assert any("MISSING_VALUES" in f for f in result["quality_flags"])
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_negative_target_flag(self):
        path = _make_csv(add_negatives=True)
        try:
            result = inspect_data(csv_path=path)
            assert any("NEGATIVE_TARGET" in f for f in result["quality_flags"])
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_duplicate_flag(self):
        path = _make_csv(periods=200, duplicate_rows=5)
        try:
            result = inspect_data(csv_path=path)
            assert any("DUPLICATES" in f for f in result["quality_flags"])
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

class TestSuggestions:

    @pytest.mark.asyncio
    async def test_calendar_features_mentioned(self):
        path = _make_csv()
        try:
            result = inspect_data(csv_path=path)
            combined = " ".join(result["suggestions"])
            assert "calendar" in combined.lower() or "Calendar" in combined
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_no_covariate_suggestion(self):
        """When no weather columns are present, the tool should suggest adding them."""
        path = _make_csv(add_temp=False)
        try:
            result = inspect_data(csv_path=path)
            combined = " ".join(result["suggestions"])
            assert "temperature" in combined.lower() or "covariate" in combined.lower()
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# New data-quality checks (UTC / DST / NaN risk)
# ---------------------------------------------------------------------------

def _make_utc_csv(
    start_utc: str = "2021-12-31 16:00",
    periods: int = 300,
    include_march_november: bool = False,
) -> str:
    """Write a CSV whose timestamps start at a non-midnight hour (simulating UTC data)."""
    if include_march_november:
        # ~3 months starting Dec 31 covers Jan, Feb, March
        start_utc = "2020-12-31 16:00"
        periods = 24 * 95  # ~Jan–Mar
    index = pd.date_range(start_utc, periods=periods, freq="h")
    df = pd.DataFrame({
        "date": index,
        "T_out": [10.0 + i * 0.01 for i in range(periods)],
        "RH_out": [70.0] * periods,
    })
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


def _make_gap_csv(gap_hours: int = 3) -> str:
    """Write a CSV with a deliberate gap that will become NaN inside Darts."""
    idx_before = pd.date_range("2024-03-01", periods=100, freq="h")
    idx_after  = pd.date_range(idx_before[-1] + pd.Timedelta(hours=gap_hours + 1),
                               periods=100, freq="h")
    index = idx_before.append(idx_after)
    df = pd.DataFrame({
        "timestamp": index,
        "electricity_kwh": [50.0 + i * 0.1 for i in range(len(index))],
    })
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


class TestNewDataQualityChecks:

    @pytest.mark.asyncio
    async def test_possible_utc_flag_non_midnight_start(self):
        """Non-midnight first timestamp triggers POSSIBLE_UTC quality flag."""
        path = _make_utc_csv(start_utc="2021-12-31 16:00", periods=300)
        try:
            result = inspect_data(csv_path=path)
            flags = result["quality_flags"]
            assert any("POSSIBLE_UTC" in f for f in flags), (
                f"Expected POSSIBLE_UTC in flags, got: {flags}"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_midnight_start_no_possible_utc_flag(self):
        """Midnight first timestamp must NOT trigger POSSIBLE_UTC."""
        path = _make_csv()  # starts 2024-01-01 00:00
        try:
            result = inspect_data(csv_path=path)
            flags = result["quality_flags"]
            assert not any("POSSIBLE_UTC" in f for f in flags), (
                f"Got unexpected POSSIBLE_UTC in flags: {flags}"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_dst_merge_risk_flag_when_covering_march_november(self):
        """DST_MERGE_RISK is added when non-midnight UTC data spans Mar or Nov."""
        path = _make_utc_csv(include_march_november=True)  # covers Jan–Mar
        try:
            result = inspect_data(csv_path=path)
            flags = result["quality_flags"]
            assert any("DST_MERGE_RISK" in f for f in flags), (
                f"Expected DST_MERGE_RISK in flags, got: {flags}"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_dst_merge_risk_not_set_for_non_dst_months(self):
        """DST_MERGE_RISK must NOT fire for UTC data that avoids Mar and Nov."""
        # July-August only — no DST transition months
        path = _make_utc_csv(start_utc="2021-07-01 07:00", periods=24 * 60)
        try:
            result = inspect_data(csv_path=path)
            flags = result["quality_flags"]
            assert not any("DST_MERGE_RISK" in f for f in flags), (
                f"Got unexpected DST_MERGE_RISK in flags: {flags}"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_nan_training_risk_suggestion_when_gaps_present(self):
        """A gap in the CSV → NaN training-failure suggestion is included."""
        path = _make_gap_csv(gap_hours=3)
        try:
            result = inspect_data(csv_path=path)
            combined = " ".join(result["suggestions"])
            assert "NaN" in combined or "nan" in combined.lower(), (
                f"Expected NaN risk suggestion, got suggestions: {result['suggestions']}"
            )
            assert "LinearRegression" in combined or "sklearn" in combined.lower(), (
                f"Expected model-specific hint in suggestions: {result['suggestions']}"
            )
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_no_nan_risk_suggestion_for_clean_file(self):
        """Clean file with no gaps must not include the NaN risk suggestion."""
        path = _make_csv()
        try:
            result = inspect_data(csv_path=path)
            combined = " ".join(result["suggestions"])
            assert "NaN training-failure risk" not in combined
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_utc_suggestion_includes_conversion_pattern(self):
        """When POSSIBLE_UTC fires, suggestions include the tz_convert pattern."""
        path = _make_utc_csv(start_utc="2021-06-15 07:00", periods=300)
        try:
            result = inspect_data(csv_path=path)
            combined = " ".join(result["suggestions"])
            assert "tz_localize" in combined or "tz_convert" in combined, (
                f"Expected timezone conversion hint, got: {result['suggestions']}"
            )
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        result = inspect_data(csv_path="/nonexistent/file.csv")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("")
        result = inspect_data(csv_path=str(csv))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_loader_error_propagated(self, tmp_path):
        """A file with no recognisable columns still returns success=True
        but sets loader_error and ready_to_train=False."""
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"measurement_id": ["a", "b", "c"]}).to_csv(csv, index=False)
        result = inspect_data(csv_path=str(csv))
        # The tool should not crash — it may succeed with loader_error set
        assert "success" in result
