"""
Filesystem path resolution for data inputs.

The MCP server process usually runs with a working directory that has nothing
to do with this repository: Claude Desktop launches servers from ``/``, Codex
uses whatever ``cwd`` its config sets, and an HTTP deployment may run from
anywhere.  A bare ``Path(csv_path)`` therefore resolves relative paths against
an arbitrary directory, and an agent that was told "data/examples/x.csv" gets
"file not found" with no explanation.

``resolve_data_path`` fixes that by trying an ordered set of roots:

1. the path exactly as given (absolute, or relative to the process cwd);
2. ``LOAD_FORECASTING_DATA_DIR`` when that variable is set;
3. the repository root (source checkouts only);
4. the bundled examples directory ``<repo>/data/examples`` (source checkouts
   only), so that ``AMI/2021_city_level.csv`` works too.

The first candidate that exists wins.  When none exist the *original* path is
returned unchanged so downstream "file not found" messages name exactly what
the caller passed.

The same roots feed ``list_datasets`` (default scan directory) and the server
``instructions`` string emitted during the MCP handshake.
"""

import os
from pathlib import Path
from typing import Optional

ENV_DATA_DIR = "LOAD_FORECASTING_DATA_DIR"

# Depth from this file to the repository root:
#   core/paths.py -> core -> load_forecasting -> src -> <repo>
_REPO_ROOT_DEPTH = 3


def repo_root() -> Optional[Path]:
    """Return the source-checkout root, or None when running from an installed wheel.

    Detection is by the presence of ``pyproject.toml`` at the expected depth;
    a wheel installs the package under ``site-packages`` where no such file
    exists.
    """
    try:
        candidate = Path(__file__).resolve().parents[_REPO_ROOT_DEPTH]
    except IndexError:  # pragma: no cover - only if installed at filesystem root
        return None
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return None


def bundled_examples_dir() -> Optional[Path]:
    """Return ``<repo>/data/examples`` when it exists, else None."""
    root = repo_root()
    if root is None:
        return None
    examples = root / "data" / "examples"
    return examples if examples.is_dir() else None


def configured_data_dir() -> Optional[Path]:
    """Return ``LOAD_FORECASTING_DATA_DIR`` as a Path when set (existence not checked)."""
    value = os.getenv(ENV_DATA_DIR)
    if not value:
        return None
    return Path(value).expanduser()


def default_dataset_dir() -> tuple[Optional[Path], str]:
    """Directory that ``list_datasets`` scans when called without an argument.

    Returns:
        ``(path, source)`` where ``source`` is one of
        ``"LOAD_FORECASTING_DATA_DIR"``, ``"bundled_examples"`` or ``"none"``.
    """
    configured = configured_data_dir()
    if configured is not None:
        return configured, ENV_DATA_DIR
    examples = bundled_examples_dir()
    if examples is not None:
        return examples, "bundled_examples"
    return None, "none"


def search_roots() -> list[Path]:
    """Ordered roots that relative data paths are resolved against (after cwd)."""
    roots: list[Path] = []
    configured = configured_data_dir()
    if configured is not None:
        roots.append(configured)
    root = repo_root()
    if root is not None:
        roots.append(root)
    examples = bundled_examples_dir()
    if examples is not None:
        roots.append(examples)
    return roots


def resolve_data_path(path) -> Path:
    """Resolve a user-supplied data path to an absolute ``Path`` when possible.

    See the module docstring for the search order.  Absolute paths are returned
    as-is (after ``~`` expansion).  A relative path that exists under one of
    the search roots is returned as an absolute path.  A path that cannot be
    found anywhere is returned unchanged so the caller's error message stays
    faithful to the input.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    for root in search_roots():
        candidate = root / p
        if candidate.exists():
            return candidate.resolve()
    return p


def not_found_hint(path) -> str:
    """Human/agent-readable explanation appended to "file not found" errors."""
    roots = [str(r) for r in search_roots()]
    searched = ", ".join([str(Path.cwd())] + roots) if roots else str(Path.cwd())
    return (
        f"Relative paths are resolved against: {searched}. "
        "Pass an absolute path, or call list_datasets to discover available "
        "files with their absolute paths."
    )
