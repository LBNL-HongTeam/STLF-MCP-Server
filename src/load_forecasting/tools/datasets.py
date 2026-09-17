"""
Dataset discovery tool.

The MCP client cannot see the server's filesystem, so an agent has no way to
learn that bundled sample data exists (or what a user dropped into
``LOAD_FORECASTING_DATA_DIR``) unless the server tells it.  ``list_datasets``
is that mechanism: a cheap, read-only enumeration of CSV/Parquet files under a
directory, each with an **absolute path** the agent can hand straight to
``inspect_data`` / ``train_forecast_model``, plus just enough metadata (rows,
columns, date range, inferred frequency) to pick the right file without a
full ``inspect_data`` call on each one.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import logging

import pandas as pd

from ..core.data_loader import _infer_frequency
from ..core.paths import (
    ENV_DATA_DIR,
    default_dataset_dir,
    resolve_data_path,
    search_roots,
)
from ._common import create_success_response, create_error_response

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".csv": "csv", ".parquet": "parquet"}

# Directories never descended into during a recursive scan.
_SKIP_DIR_NAMES = {"__pycache__", "node_modules", ".git", ".venv", "venv"}

# Above this size we still list the file but skip the row/date statistics,
# which require reading the datetime column end to end.
_STATS_SIZE_LIMIT_BYTES = 500 * 1024 * 1024

# Same name hints the data loader and merge_covariates use for auto-detection.
_DATETIME_HINTS = ("datetime", "timestamp", "date", "time", "dt")

_MAX_LIMIT = 500
_MAX_DEPTH = 4


def list_datasets(
    directory: Optional[str] = None,
    recursive: bool = True,
    include_stats: bool = True,
    limit: int = 50,
) -> dict:
    """
    List CSV/Parquet datasets available to the server, with absolute paths.

    Call this first when the user asks what data is available, or before
    guessing a ``csv_path``.  With no ``directory`` it scans
    ``LOAD_FORECASTING_DATA_DIR`` if set, otherwise the sample datasets
    bundled with the server (``data/examples`` in a source checkout).

    Args:
        directory: Directory to scan.  Absolute, or relative to the same roots
            ``csv_path`` resolves against (working directory,
            ``LOAD_FORECASTING_DATA_DIR``, repository root, bundled examples).
            Omit to scan the default location.
        recursive: Descend into subdirectories (max depth 4).  Default True.
        include_stats: Read each file's header and datetime column to report
            rows, columns, date range and inferred frequency.  Default True.
            Set False for a fast name-only listing of a large directory.
        limit: Maximum number of files to return (1-500).  Default 50.

    Returns:
        Dict with ``directory`` (absolute path scanned), ``source`` (how the
        directory was chosen), ``datasets`` (list of file records, each with
        ``path`` ready to pass as ``csv_path``), ``total_count``,
        ``truncated`` and ``notes``.
    """
    try:
        limit = max(1, min(int(limit), _MAX_LIMIT))

        # ------------------------------------------------------------------
        # Resolve the directory to scan
        # ------------------------------------------------------------------
        if directory:
            scan_dir = resolve_data_path(directory)
            source = "argument"
        else:
            scan_dir, source = default_dataset_dir()
            if scan_dir is None:
                return create_error_response(
                    "No dataset directory is configured and no bundled examples "
                    f"were found. Set {ENV_DATA_DIR} to a directory containing CSV "
                    "files, or pass `directory` explicitly."
                )

        if not scan_dir.exists():
            roots = ", ".join(str(r) for r in search_roots()) or "(none)"
            return create_error_response(
                f"Directory not found: {directory or scan_dir}. "
                f"Relative directories are resolved against the working directory and: {roots}."
            )
        if not scan_dir.is_dir():
            return create_error_response(
                f"Not a directory: {scan_dir}. To profile a single file use inspect_data."
            )
        scan_dir = scan_dir.resolve()

        # ------------------------------------------------------------------
        # Enumerate files
        # ------------------------------------------------------------------
        files = _find_data_files(scan_dir, recursive=recursive)
        total_count = len(files)
        truncated = total_count > limit
        files = files[:limit]

        datasets = []
        for path in files:
            record = _describe_file(path, scan_dir, include_stats=include_stats)
            datasets.append(record)

        # ------------------------------------------------------------------
        # Notes that steer the agent's next call
        # ------------------------------------------------------------------
        notes = [
            "Pass a record's `path` verbatim as csv_path to inspect_data, "
            "train_forecast_model, or batch_train_forecast_models.",
        ]
        if any(d["format"] == "parquet" for d in datasets):
            notes.append(
                "Parquet files are listed for completeness only; the forecasting "
                "tools read CSV. Convert Parquet to CSV before use."
            )
        if any(d.get("n_columns", 0) > 3 for d in datasets if d["format"] == "csv"):
            notes.append(
                "Wide files with several numeric columns are one series per "
                "column (e.g. one feeder or substation each); pick a target with "
                "column_mapping, or train them all with batch_train_forecast_models."
            )
        if truncated:
            notes.append(
                f"Only the first {limit} of {total_count} files are shown; raise "
                "`limit` or pass a narrower `directory`."
            )
        if total_count == 0:
            notes.append(
                "No .csv or .parquet files found. Check the directory, or set "
                f"{ENV_DATA_DIR} to point at your data."
            )

        return create_success_response(
            directory=str(scan_dir),
            source=source,
            datasets=datasets,
            total_count=total_count,
            returned=len(datasets),
            truncated=truncated,
            notes=notes,
        )

    except Exception as e:
        logger.exception("list_datasets failed")
        return create_error_response(f"Failed to list datasets: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_data_files(root: Path, recursive: bool) -> list[Path]:
    """Return supported data files under ``root``, sorted by relative path."""
    found: list[Path] = []
    if recursive:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(root).parts[:-1]
            if len(rel_parts) > _MAX_DEPTH:
                continue
            if any(part.startswith(".") or part in _SKIP_DIR_NAMES for part in rel_parts):
                continue
            if path.name.startswith(".") or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            found.append(path)
    else:
        for path in root.iterdir():
            if (
                path.is_file()
                and not path.name.startswith(".")
                and path.suffix.lower() in SUPPORTED_SUFFIXES
            ):
                found.append(path)
    return sorted(found, key=lambda p: str(p.relative_to(root)).lower())


def _describe_file(path: Path, root: Path, include_stats: bool) -> dict:
    """Build the per-file record: identity, size, and (optionally) shape/time stats."""
    stat = path.stat()
    record: dict = {
        "name": path.name,
        "path": str(path.resolve()),
        "relative_path": str(path.relative_to(root)),
        "format": SUPPORTED_SUFFIXES[path.suffix.lower()],
        "size_bytes": stat.st_size,
        "size_mb": round(stat.st_size / (1024 * 1024), 2),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    if not include_stats:
        return record

    if stat.st_size > _STATS_SIZE_LIMIT_BYTES:
        record["stats_error"] = (
            f"File is {record['size_mb']} MB; statistics skipped above "
            f"{_STATS_SIZE_LIMIT_BYTES // (1024 * 1024)} MB. Use inspect_data."
        )
        return record

    try:
        if record["format"] == "csv":
            record.update(_profile_csv(path))
        else:
            record.update(_profile_parquet(path))
    except ImportError as e:
        # Parquet without pyarrow/fastparquet lands here.
        record["stats_error"] = f"Could not read {record['format']}: {e}"
    except Exception as e:
        record["stats_error"] = f"Could not compute statistics: {e}"
    return record


def _detect_datetime_column(columns: list[str], sample: pd.DataFrame) -> Optional[str]:
    """Pick the datetime column by name hint, else the first column that parses as datetimes."""
    for col in columns:
        if any(hint in str(col).lower() for hint in _DATETIME_HINTS):
            return col
    for col in columns:
        series = sample[col]
        if pd.api.types.is_numeric_dtype(series):
            continue
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() > 0.9:
            return col
    return None


def _time_stats(values: pd.Series) -> dict:
    """Row count, date range, and inferred frequency from a datetime column."""
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    valid = parsed.dropna()
    out: dict = {"rows": int(len(values))}
    if valid.empty:
        out["datetime_parse_error"] = "no parseable timestamps"
        return out
    idx = pd.DatetimeIndex(valid.sort_values())
    out["start"] = idx.min().isoformat()
    out["end"] = idx.max().isoformat()
    out["frequency"] = _infer_frequency(idx)
    if len(valid) != len(values):
        out["unparseable_timestamps"] = int(len(values) - len(valid))
    return out


def _profile_csv(path: Path) -> dict:
    """Header + datetime-column read only; never loads every column of a large file."""
    sample = pd.read_csv(path, nrows=200)
    columns = [str(c) for c in sample.columns]
    out: dict = {
        "columns": columns,
        "n_columns": len(columns),
        "numeric_columns": [
            str(c) for c in sample.columns if pd.api.types.is_numeric_dtype(sample[c])
        ],
    }
    dt_col = _detect_datetime_column(columns, sample)
    out["datetime_column"] = dt_col
    if dt_col is None:
        # Still report the row count cheaply.
        out["rows"] = int(sum(1 for _ in open(path, "rb")) - 1)
        return out
    dt_values = pd.read_csv(path, usecols=[dt_col])[dt_col]
    out.update(_time_stats(dt_values))
    return out


def _profile_parquet(path: Path) -> dict:
    """Parquet profile; raises ImportError when no engine is installed."""
    df = pd.read_parquet(path)
    columns = [str(c) for c in df.columns]
    out: dict = {
        "columns": columns,
        "n_columns": len(columns),
        "numeric_columns": [
            str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ],
        "rows": int(len(df)),
    }
    if isinstance(df.index, pd.DatetimeIndex):
        out["datetime_column"] = df.index.name or "(index)"
        out.update(_time_stats(pd.Series(df.index)))
        return out
    dt_col = _detect_datetime_column(columns, df.head(200))
    out["datetime_column"] = dt_col
    if dt_col is not None:
        out.update(_time_stats(df[dt_col]))
    return out
