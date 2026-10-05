"""
Load YAML specs as single source of truth for validation.

Specs define:
- Input/output schemas for tools
- Data constraints and validation rules
- Column auto-detection patterns
- Algorithm metadata for AI discovery

Specs live in load_forecasting/specs/ (inside the package) so they are
accessible whether the package is run from source or installed via pip.
importlib.resources is used for path resolution to guarantee correctness
in both environments.
"""

import yaml
import logging
from functools import lru_cache
from importlib.resources import files
from typing import Optional

logger = logging.getLogger(__name__)

# Package-relative path to the specs directory
_SPECS_PACKAGE = "load_forecasting.specs"


def _specs_dir():
    """Return the importlib.resources traversable for the specs package."""
    return files(_SPECS_PACKAGE)


@lru_cache(maxsize=32)
def load_spec(tool_name: str) -> Optional[dict]:
    """
    Load YAML spec by MCP tool name.

    Args:
        tool_name: Name of the MCP tool (e.g., "train_forecast_model")

    Returns:
        Parsed YAML spec dict, or None if not found
    """
    spec_file = _specs_dir().joinpath(f"{tool_name}.yaml")

    try:
        content = spec_file.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        logger.warning("Spec not found: %s.yaml", tool_name)
        return None
    except Exception as e:
        logger.error("Error reading spec %s.yaml: %s", tool_name, e)
        return None

    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error("Error parsing spec %s.yaml: %s", tool_name, e)
        return None


@lru_cache(maxsize=1)
def get_all_specs() -> list[dict]:
    """
    Load all specs for get_algorithm_specifications tool.

    Cached: specs are static files; the cache is valid for the process lifetime.

    Returns:
        List of all parsed YAML spec dicts
    """
    specs = []
    specs_dir = _specs_dir()

    try:
        entries = sorted(
            (e for e in specs_dir.iterdir() if e.name.endswith(".yaml")),
            key=lambda e: e.name,
        )
    except Exception as e:
        logger.warning("Could not list specs directory: %s", e)
        return specs

    for entry in entries:
        try:
            content = entry.read_text(encoding="utf-8")
            spec = yaml.safe_load(content)
            if spec:
                specs.append(spec)
        except yaml.YAMLError as e:
            logger.error("Error parsing spec %s: %s", entry.name, e)

    return specs


def get_required_columns(tool_name: str) -> dict:
    """
    Extract column requirements from spec for data validation.

    Args:
        tool_name: Name of the MCP tool

    Returns:
        Dict with expected_columns and constraints
    """
    spec = load_spec(tool_name)
    if not spec:
        return {}

    for input_def in spec.get("inputs", []):
        if input_def.get("name") == "csv_path":
            return {
                "expected_columns": input_def.get("expected_columns", []),
                "constraints": input_def.get("constraints", {}),
            }

    return {}


def get_auto_detect_patterns(tool_name: str) -> dict:
    """
    Get column auto-detection patterns from spec.

    Args:
        tool_name: Name of the MCP tool

    Returns:
        Dict mapping column roles to detection patterns
    """
    spec = load_spec(tool_name)
    if not spec:
        return {}

    patterns = {}

    for input_def in spec.get("inputs", []):
        if input_def.get("name") == "column_mapping":
            properties = input_def.get("properties", {})
            for prop_name, prop_def in properties.items():
                if "auto_detect_patterns" in prop_def:
                    patterns[prop_name] = prop_def["auto_detect_patterns"]

    return patterns


def get_preprocessing_config(tool_name: str) -> dict:
    """
    Get preprocessing configuration from spec.

    Args:
        tool_name: Name of the MCP tool

    Returns:
        Dict with datetime_parsing, missing_values, normalization config
    """
    spec = load_spec(tool_name)
    if not spec:
        return {}

    return spec.get("preprocessing", {})


_REQUIRED_SPEC_FIELDS: frozenset = frozenset({"id", "version", "mcp_server", "mcp_tool"})


def validate_spec(spec_data: dict) -> bool:
    """
    Validate that a spec has required fields.

    Args:
        spec_data: Parsed spec dict

    Returns:
        True if valid, False otherwise
    """
    for field in _REQUIRED_SPEC_FIELDS:
        if field not in spec_data:
            logger.warning("Spec missing required field: %s", field)
            return False

    return True
