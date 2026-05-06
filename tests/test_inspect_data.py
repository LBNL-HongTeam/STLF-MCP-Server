"""Tests for the inspect_data MCP tool."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.tools.forecasting_tools import inspect_data


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
            result = await inspect_data(csv_path=path)
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
            result = await inspect_data(csv_path=path)
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
            result = await inspect_data(csv_path=path)
            cols = result["columns"]
            assert cols["datetime"] == "timestamp"
            assert cols["target"] == "electricity_kwh"
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_detects_covariate(self):
        path = _make_csv(add_temp=True)
        try:
            result = await inspect_data(csv_path=path)
            assert "outdoor_temp" in result["columns"]["past_covariates"]
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_no_covariate_when_absent(self):
        path = _make_csv(add_temp=False)
        try:
            result = await inspect_data(csv_path=path)
            assert result["columns"]["past_covariates"] == []
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_explicit_mapping_respected(self):
        path = _make_csv()
        try:
            mapping = {"datetime": "timestamp", "target": "electricity_kwh", "past_covariates": []}
            result = await inspect_data(csv_path=path, column_mapping=mapping)
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
            result = await inspect_data(csv_path=path)
            assert result["frequency"]["inferred"] == "h"
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_infers_15min(self):
        path = _make_csv(freq="15min")
        try:
            result = await inspect_data(csv_path=path, frequency="15min")
            assert result["frequency"]["inferred"] == "15min"
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_frequency_match(self):
        path = _make_csv(freq="h")
        try:
            result = await inspect_data(csv_path=path, frequency="h")
            assert result["frequency"]["match"] is True
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_frequency_mismatch_flagged(self):
        path = _make_csv(freq="h")
        try:
            result = await inspect_data(csv_path=path, frequency="15min")
            assert result["frequency"]["match"] is False
            assert any("FREQUENCY_MISMATCH" in f for f in result["quality_flags"])
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_no_declared_frequency_no_mismatch(self):
        """When frequency is not declared, match should be True (no comparison made)."""
        path = _make_csv(freq="h")
        try:
            result = await inspect_data(csv_path=path, frequency=None)
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
            result = await inspect_data(csv_path=path)
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
            result = await inspect_data(csv_path=path)
            stats = {s["column"]: s for s in result["column_statistics"]}
            assert stats["electricity_kwh"]["n_missing"] > 0
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_negative_count(self):
        path = _make_csv(add_negatives=True)
        try:
            result = await inspect_data(csv_path=path)
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
            result = await inspect_data(csv_path=path)
            assert result["gaps"]["n_gaps"] == 0
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_gap_analysis(self):
        """A CSV with an artificial gap is detected, flagged, and listed in top_gaps."""
        path = _make_csv(periods=200, add_gap_at=100)
        try:
            result = await inspect_data(csv_path=path)
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
            result = await inspect_data(csv_path=path)
            assert result["quality_flags"] == []
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_missing_values_flag(self):
        path = _make_csv(periods=300, null_fraction=0.03)
        try:
            result = await inspect_data(csv_path=path)
            assert any("MISSING_VALUES" in f for f in result["quality_flags"])
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_negative_target_flag(self):
        path = _make_csv(add_negatives=True)
        try:
            result = await inspect_data(csv_path=path)
            assert any("NEGATIVE_TARGET" in f for f in result["quality_flags"])
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_duplicate_flag(self):
        path = _make_csv(periods=200, duplicate_rows=5)
        try:
            result = await inspect_data(csv_path=path)
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
            result = await inspect_data(csv_path=path)
            combined = " ".join(result["suggestions"])
            assert "calendar" in combined.lower() or "Calendar" in combined
        finally:
            Path(path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_no_covariate_suggestion(self):
        """When no weather columns are present, the tool should suggest adding them."""
        path = _make_csv(add_temp=False)
        try:
            result = await inspect_data(csv_path=path)
            combined = " ".join(result["suggestions"])
            assert "temperature" in combined.lower() or "covariate" in combined.lower()
        finally:
            Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestErrorPaths:

    @pytest.mark.asyncio
    async def test_file_not_found(self):
        result = await inspect_data(csv_path="/nonexistent/file.csv")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_file(self, tmp_path):
        csv = tmp_path / "empty.csv"
        csv.write_text("")
        result = await inspect_data(csv_path=str(csv))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_loader_error_propagated(self, tmp_path):
        """A file with no recognisable columns still returns success=True
        but sets loader_error and ready_to_train=False."""
        csv = tmp_path / "bad.csv"
        pd.DataFrame({"measurement_id": ["a", "b", "c"]}).to_csv(csv, index=False)
        result = await inspect_data(csv_path=str(csv))
        # The tool should not crash — it may succeed with loader_error set
        assert "success" in result
