"""Tests for dataset discovery, data-path resolution, and training provenance.

Covers the three agent-facing gaps these features close:

1. ``list_datasets`` — the agent can enumerate available files with absolute
   paths instead of concluding that no data exists.
2. ``resolve_data_path`` — relative ``csv_path`` values work regardless of the
   server process's working directory (Claude Desktop launches from ``/``).
3. Provenance — ``data_info`` and the registry index record the training CSV.
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core import paths as paths_mod
from load_forecasting.core.paths import (
    ENV_DATA_DIR,
    bundled_examples_dir,
    default_dataset_dir,
    not_found_hint,
    resolve_data_path,
)
from load_forecasting.core.data_loader import ForecastingDataLoader, DataLoadError
from load_forecasting.tools import inspect_data, list_datasets, list_models, train_forecast_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_csv(path: Path, periods: int, freq: str, dt_col: str = "timestamp") -> None:
    df = pd.DataFrame(
        {
            dt_col: pd.date_range("2023-01-01", periods=periods, freq=freq),
            "load_kwh": [100 + (i % 24) for i in range(periods)],
            "outdoor_temp": [50 + (i % 12) for i in range(periods)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


@pytest.fixture
def dataset_dir(tmp_path):
    """A directory tree with a mix of data files, nested dirs, and noise."""
    _write_csv(tmp_path / "hourly.csv", periods=240, freq="h")
    _write_csv(tmp_path / "quarter.csv", periods=96 * 3, freq="15min", dt_col="Datetime")
    _write_csv(tmp_path / "nested" / "deep.csv", periods=48, freq="30min")
    _write_csv(tmp_path / ".hidden" / "secret.csv", periods=48, freq="h")
    (tmp_path / "notes.txt").write_text("not a dataset\n")
    (tmp_path / "no_time.csv").write_text("a,b\n1,2\n3,4\n")
    return tmp_path


@pytest.fixture
def temp_model_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


@pytest.fixture
def foreign_cwd(tmp_path, monkeypatch):
    """Run the test from a directory unrelated to the repo (like Claude Desktop does)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


requires_bundled_examples = pytest.mark.skipif(
    bundled_examples_dir() is None, reason="bundled data/examples not present"
)


# ---------------------------------------------------------------------------
# resolve_data_path
# ---------------------------------------------------------------------------

class TestResolveDataPath:
    def test_absolute_path_passthrough(self, tmp_path):
        target = tmp_path / "x.csv"
        assert resolve_data_path(str(target)) == target

    def test_relative_to_cwd_is_made_absolute(self, dataset_dir, monkeypatch):
        monkeypatch.chdir(dataset_dir)
        resolved = resolve_data_path("hourly.csv")
        assert resolved.is_absolute()
        assert resolved == (dataset_dir / "hourly.csv").resolve()

    def test_relative_to_env_data_dir(self, dataset_dir, foreign_cwd, monkeypatch):
        monkeypatch.setenv(ENV_DATA_DIR, str(dataset_dir))
        assert resolve_data_path("nested/deep.csv") == (dataset_dir / "nested" / "deep.csv").resolve()

    def test_missing_path_returned_unchanged(self, foreign_cwd, monkeypatch):
        monkeypatch.delenv(ENV_DATA_DIR, raising=False)
        assert resolve_data_path("does/not/exist.csv") == Path("does/not/exist.csv")

    def test_tilde_expanded(self):
        assert resolve_data_path("~/x.csv") == Path.home() / "x.csv"

    @requires_bundled_examples
    def test_repo_relative_from_foreign_cwd(self, foreign_cwd):
        resolved = resolve_data_path("data/examples/sample_building_load.csv")
        assert resolved.exists() and resolved.is_absolute()

    @requires_bundled_examples
    def test_examples_relative_from_foreign_cwd(self, foreign_cwd):
        assert resolve_data_path("AMI/2021_city_level.csv").exists()

    def test_not_found_hint_mentions_list_datasets(self):
        assert "list_datasets" in not_found_hint("x.csv")


class TestDefaultDatasetDir:
    def test_env_var_wins(self, dataset_dir, monkeypatch):
        monkeypatch.setenv(ENV_DATA_DIR, str(dataset_dir))
        path, source = default_dataset_dir()
        assert path == dataset_dir and source == ENV_DATA_DIR

    def test_falls_back_to_bundled(self, monkeypatch):
        monkeypatch.delenv(ENV_DATA_DIR, raising=False)
        path, source = default_dataset_dir()
        if bundled_examples_dir() is None:
            assert path is None and source == "none"
        else:
            assert path == bundled_examples_dir() and source == "bundled_examples"


# ---------------------------------------------------------------------------
# list_datasets tool
# ---------------------------------------------------------------------------

class TestListDatasets:
    def test_lists_files_with_stats(self, dataset_dir):
        result = list_datasets(directory=str(dataset_dir))
        assert result["success"] is True, result.get("error")
        assert result["source"] == "argument"
        assert result["directory"] == str(dataset_dir.resolve())

        by_rel = {d["relative_path"]: d for d in result["datasets"]}
        # Hidden dirs and non-data files are excluded; nested dirs included.
        assert set(by_rel) == {"hourly.csv", "quarter.csv", "no_time.csv", "nested/deep.csv"}
        assert result["total_count"] == 4 and result["truncated"] is False

        hourly = by_rel["hourly.csv"]
        assert Path(hourly["path"]).is_absolute()
        assert hourly["format"] == "csv"
        assert hourly["rows"] == 240
        assert hourly["datetime_column"] == "timestamp"
        assert hourly["frequency"] == "h"
        assert hourly["columns"] == ["timestamp", "load_kwh", "outdoor_temp"]
        assert hourly["numeric_columns"] == ["load_kwh", "outdoor_temp"]
        assert hourly["start"].startswith("2023-01-01T00:00:00")
        assert hourly["end"].startswith("2023-01-10T23:00:00")

        assert by_rel["quarter.csv"]["frequency"] == "15min"
        assert by_rel["quarter.csv"]["datetime_column"] == "Datetime"
        assert by_rel["nested/deep.csv"]["frequency"] == "30min"

    def test_file_without_datetime_still_listed(self, dataset_dir):
        result = list_datasets(directory=str(dataset_dir))
        rec = next(d for d in result["datasets"] if d["name"] == "no_time.csv")
        assert rec["datetime_column"] is None
        assert rec["rows"] == 2
        assert "frequency" not in rec

    def test_non_recursive(self, dataset_dir):
        result = list_datasets(directory=str(dataset_dir), recursive=False)
        names = {d["name"] for d in result["datasets"]}
        assert "deep.csv" not in names and "hourly.csv" in names

    def test_include_stats_false_is_name_only(self, dataset_dir):
        result = list_datasets(directory=str(dataset_dir), include_stats=False)
        rec = result["datasets"][0]
        assert {"name", "path", "relative_path", "format", "size_bytes", "size_mb", "modified"} <= set(rec)
        assert "rows" not in rec and "columns" not in rec

    def test_limit_and_truncation(self, dataset_dir):
        result = list_datasets(directory=str(dataset_dir), limit=2)
        assert result["returned"] == 2
        assert result["total_count"] == 4
        assert result["truncated"] is True
        assert any("limit" in n for n in result["notes"])

    def test_limit_is_clamped(self, dataset_dir):
        assert list_datasets(directory=str(dataset_dir), limit=0)["returned"] == 1

    def test_parquet_without_engine_is_graceful(self, tmp_path):
        # A bogus .parquet file: whatever engine state pandas has, the record
        # must come back with a stats_error rather than raising.
        (tmp_path / "x.parquet").write_bytes(b"not really parquet")
        result = list_datasets(directory=str(tmp_path))
        assert result["success"] is True
        rec = result["datasets"][0]
        assert rec["format"] == "parquet"
        assert "stats_error" in rec
        assert any("Parquet" in n for n in result["notes"])

    def test_directory_not_found(self, tmp_path):
        result = list_datasets(directory=str(tmp_path / "missing"))
        assert result["success"] is False
        assert "Directory not found" in result["error"]

    def test_directory_is_a_file(self, dataset_dir):
        result = list_datasets(directory=str(dataset_dir / "hourly.csv"))
        assert result["success"] is False
        assert "Not a directory" in result["error"]
        assert "inspect_data" in result["error"]

    def test_default_uses_env_data_dir(self, dataset_dir, monkeypatch):
        monkeypatch.setenv(ENV_DATA_DIR, str(dataset_dir))
        result = list_datasets()
        assert result["success"] is True
        assert result["source"] == ENV_DATA_DIR
        assert result["total_count"] == 4

    def test_relative_directory_resolved_against_env_data_dir(self, dataset_dir, foreign_cwd, monkeypatch):
        monkeypatch.setenv(ENV_DATA_DIR, str(dataset_dir))
        result = list_datasets(directory="nested")
        assert result["success"] is True
        assert [d["name"] for d in result["datasets"]] == ["deep.csv"]

    def test_no_default_available_is_an_error(self, monkeypatch):
        monkeypatch.delenv(ENV_DATA_DIR, raising=False)
        monkeypatch.setattr(paths_mod, "bundled_examples_dir", lambda: None)
        result = list_datasets()
        assert result["success"] is False
        assert ENV_DATA_DIR in result["error"]

    def test_empty_directory_has_helpful_note(self, tmp_path):
        result = list_datasets(directory=str(tmp_path))
        assert result["success"] is True and result["total_count"] == 0
        assert any("No .csv" in n for n in result["notes"])

    @requires_bundled_examples
    def test_bundled_examples_from_foreign_cwd(self, foreign_cwd, monkeypatch):
        monkeypatch.delenv(ENV_DATA_DIR, raising=False)
        result = list_datasets()
        assert result["success"] is True
        assert result["source"] == "bundled_examples"
        names = {d["name"]: d for d in result["datasets"]}
        assert "sample_building_load.csv" in names
        assert names["sample_building_load.csv"]["frequency"] == "15min"
        assert all(Path(d["path"]).is_absolute() for d in result["datasets"])


# ---------------------------------------------------------------------------
# Relative csv_path through the real entry points, from a foreign cwd
# ---------------------------------------------------------------------------

@requires_bundled_examples
class TestRelativeCsvPathFromForeignCwd:
    def test_loader_resolves_repo_relative_path(self, foreign_cwd):
        loader = ForecastingDataLoader(
            "data/examples/sample_building_load.csv",
            frequency="15min",
            add_calendar_features=False,
            lag_hours=[],
        )
        assert loader.csv_path.is_absolute()
        assert len(loader.df) == 8640

    def test_inspect_data_resolves_examples_relative_path(self, foreign_cwd):
        result = inspect_data("AMI/2021_city_level.csv")
        assert result["success"] is True, result.get("error")

    def test_missing_file_error_carries_hint(self, foreign_cwd):
        with pytest.raises(DataLoadError, match="list_datasets"):
            ForecastingDataLoader("nope/missing.csv")
        result = inspect_data("nope/missing.csv")
        assert result["success"] is False
        assert "list_datasets" in result["error"]


# ---------------------------------------------------------------------------
# Provenance: what was this model trained on?
# ---------------------------------------------------------------------------

class TestTrainingProvenance:
    def test_metadata_and_registry_record_csv_path(self, tmp_path, temp_model_dir, monkeypatch):
        _write_csv(tmp_path / "train.csv", periods=500, freq="h")
        monkeypatch.chdir(tmp_path)

        # Relative path on purpose: the recorded path must still be absolute.
        result = train_forecast_model(
            csv_path="train.csv",
            model_type="NaiveMean",
            building_name="prov_test",
        )
        assert result["success"] is True, result.get("error")
        expected = str((tmp_path / "train.csv").resolve())
        assert result["data_summary"]["csv_path"] == expected

        metadata = json.loads(
            (Path(temp_model_dir) / result["model_id"] / "metadata.json").read_text()
        )
        assert metadata["data_info"]["csv_path"] == expected

        listed = list_models()
        assert listed["total_count"] == 1
        entry = listed["models"][0]
        assert entry["csv_path"] == expected
        assert entry["target_column"] == "load_kwh"
        assert entry["frequency"] == "h"


# ---------------------------------------------------------------------------
# Server handshake instructions
# ---------------------------------------------------------------------------

def test_server_instructions_advertise_datasets():
    from load_forecasting.server import mcp

    text = mcp.instructions or ""
    assert "list_datasets" in text
    assert "csv_path" in text
    if bundled_examples_dir() is not None:
        assert str(bundled_examples_dir()) in text
