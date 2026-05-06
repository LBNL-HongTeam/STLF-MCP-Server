"""
Load YAML specs as single source of truth for validation.

Specs define:
- Input/output schemas for tools
- Data constraints and validation rules
- Column auto-detection patterns
- Algorithm metadata for AI discovery
"""

import yaml
from pathlib import Path
from functools import lru_cache
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Specs directory relative to this file
SPECS_DIR = Path(__file__).parent.parent.parent.parent / "specs"


@lru_cache(maxsize=32)
def load_spec(tool_name: str) -> Optional[dict]:
    """
    Load YAML spec by MCP tool name.

    Args:
        tool_name: Name of the MCP tool (e.g., "train_forecast_model")

    Returns:
        Parsed YAML spec dict, or None if not found
    """
    spec_path = SPECS_DIR / f"{tool_name}.yaml"

    if not spec_path.exists():
        logger.warning(f"Spec not found: {spec_path}")
        return None

    try:
        with open(spec_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing spec {spec_path}: {e}")
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

    if not SPECS_DIR.exists():
        logger.warning(f"Specs directory not found: {SPECS_DIR}")
        return specs

    for spec_file in sorted(SPECS_DIR.glob("*.yaml")):
        try:
            with open(spec_file, encoding="utf-8") as f:
                spec = yaml.safe_load(f)
                if spec:
                    specs.append(spec)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing spec {spec_file}: {e}")

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
            logger.warning(f"Spec missing required field: {field}")
            return False

    return True
