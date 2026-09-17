"""Drift-guard tests: keep YAML specs in sync with registered MCP tools.

The tool function signatures are the runtime I/O contract (FastMCP derives the
JSON schema from them). The YAML specs in ``src/load_forecasting/specs/`` are a
SEPARATE, hand-authored source used only for agent discovery via the
``get_algorithm_specifications`` tool.

These tests ensure that second source cannot silently drift from the set of
registered tools again (the historical "9 specs vs 13 tools" problem).
"""

import asyncio
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.server import mcp
from load_forecasting.core.spec_loader import get_all_specs, load_spec, validate_spec
from load_forecasting.core.data_loader import (
    DEFAULT_DATETIME_PATTERNS,
    DEFAULT_TARGET_PATTERNS,
    DEFAULT_TARGET_EXCLUDE_PATTERNS,
    DEFAULT_COVARIATE_PATTERNS,
)

# Discovery tools are intentionally NOT given a spec: they are the mechanisms
# that surface specs and skills, not forecasting algorithms to be discovered.
SPEC_EXEMPT_TOOLS = {"get_algorithm_specifications", "list_skills", "get_skill"}

SPECS_DIR = Path(__file__).parent.parent / "src" / "load_forecasting" / "specs"


def _registered_tool_names() -> set[str]:
    """Return the set of tool names registered on the FastMCP server."""
    tools = asyncio.run(mcp.list_tools())
    return {t.name for t in tools}


def _spec_tool_names() -> set[str]:
    """Return the set of mcp_tool names declared across all YAML specs."""
    return {spec["mcp_tool"] for spec in get_all_specs()}


def test_every_registered_tool_has_a_spec():
    """Each registered tool (except exempt ones) must have exactly one spec."""
    registered = _registered_tool_names()
    expected = registered - SPEC_EXEMPT_TOOLS
    specced = _spec_tool_names()

    missing = expected - specced
    assert not missing, (
        f"Registered tools without a YAML spec: {sorted(missing)}. "
        f"Add src/load_forecasting/specs/<tool>.yaml (and register it in "
        f"pyproject.toml force-include)."
    )


def test_no_orphan_specs():
    """Every spec must map to a currently-registered tool (no stale specs)."""
    registered = _registered_tool_names()
    orphans = _spec_tool_names() - registered
    assert not orphans, (
        f"YAML specs referencing tools that are not registered: {sorted(orphans)}."
    )


def test_exempt_tools_have_no_spec():
    """The discovery tool must NOT have its own spec (keeps the mapping 1:1)."""
    overlap = SPEC_EXEMPT_TOOLS & _spec_tool_names()
    assert not overlap, (
        f"Exempt tools should not have a YAML spec: {sorted(overlap)}."
    )


@pytest.mark.parametrize("spec_file", sorted(SPECS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_spec_filename_matches_mcp_tool(spec_file):
    """Spec filename stem must equal its mcp_tool (load_spec relies on this)."""
    spec = load_spec(spec_file.stem)
    assert spec is not None, f"load_spec({spec_file.stem!r}) returned None"
    assert spec["mcp_tool"] == spec_file.stem, (
        f"{spec_file.name}: mcp_tool={spec['mcp_tool']!r} != filename stem "
        f"{spec_file.stem!r}"
    )


@pytest.mark.parametrize("spec_file", sorted(SPECS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_spec_has_required_fields(spec_file):
    """Every spec must satisfy the required-field contract in spec_loader."""
    spec = load_spec(spec_file.stem)
    assert validate_spec(spec), (
        f"{spec_file.name} is missing one of the required fields "
        f"(id, version, mcp_server, mcp_tool)."
    )


@pytest.mark.parametrize("spec_name", ["train_forecast_model", "inspect_data"])
def test_auto_detect_patterns_match_code(spec_name):
    """The YAML auto_detect_patterns are documentation of the code constants.

    ForecastingDataLoader reads train_forecast_model.yaml at runtime and falls
    back to the constants only if the spec is missing, so the two must agree
    or auto-detection would differ between source checkouts and wheels.
    """
    spec = load_spec(spec_name)
    props = next(i for i in spec["inputs"] if i["name"] == "column_mapping")["properties"]
    assert props["datetime"]["auto_detect_patterns"] == DEFAULT_DATETIME_PATTERNS
    assert props["target"]["auto_detect_patterns"] == DEFAULT_TARGET_PATTERNS
    assert props["target"]["auto_detect_exclude_patterns"] == DEFAULT_TARGET_EXCLUDE_PATTERNS
    assert props["past_covariates"]["auto_detect_patterns"] == DEFAULT_COVARIATE_PATTERNS
