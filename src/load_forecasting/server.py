"""
FastMCP Server Configuration and Tool Registration
"""

import logging
import os
from pathlib import Path

from fastmcp import FastMCP

from .core.spec_loader import get_all_specs
from .core.paths import ENV_DATA_DIR, default_dataset_dir
from .tools import (
    train_forecast_model,
    evaluate_forecast_model,
    list_models,
    inspect_data,
    list_datasets,
    tune_model,
    generate_forecast,
    backtest_model,
    generate_evaluation_report,
    generate_backtest_report,
    merge_covariates,
    fetch_weather_forecast,
    generate_inference_dashboard,
    batch_train_forecast_models,
    batch_generate_forecast,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_instructions() -> str:
    """Server-level guidance delivered to the client in the MCP initialize handshake.

    Most MCP hosts (Claude Desktop, Claude Code, Codex) place this text in the
    model's context, so it is the cheapest way to tell an agent that sample
    data exists and how paths are resolved -- facts it cannot discover by
    itself because the server's filesystem is invisible to the client.
    """
    dataset_dir, source = default_dataset_dir()
    model_dir = Path(os.getenv("LOAD_FORECASTING_MODEL_DIR", "models")).expanduser().resolve()

    lines = [
        "Short-term load forecasting server: train, tune, evaluate, backtest and "
        "forecast electrical load with Darts models.",
        "Typical workflow: list_datasets -> inspect_data -> train_forecast_model -> "
        "evaluate_forecast_model or backtest_model -> generate_forecast.",
    ]
    if dataset_dir is not None:
        label = "bundled sample datasets" if source == "bundled_examples" else f"datasets ({source})"
        lines.append(
            f"This server has {label} at {dataset_dir}. Call list_datasets to "
            "enumerate them with absolute paths before telling the user no data is "
            "available."
        )
    else:
        lines.append(
            f"No dataset directory is configured; set {ENV_DATA_DIR} or pass absolute "
            "csv_path values."
        )
    lines.append(
        "csv_path arguments accept absolute paths, or paths relative to "
        f"{ENV_DATA_DIR}, the repository root, or the bundled examples directory "
        "(e.g. data/examples/AMI/2021_city_level.csv). Prefer the absolute `path` "
        "values returned by list_datasets."
    )
    lines.append(
        f"Trained models are stored under {model_dir}. list_models reports the "
        "csv_path, target_column and frequency each model was trained on."
    )
    return "\n".join(lines)


# Initialize FastMCP server
mcp = FastMCP(name="load_forecasting", instructions=_build_instructions())

# Register tools directly — no wrapper functions needed
mcp.add_tool(train_forecast_model)
mcp.add_tool(evaluate_forecast_model)
mcp.add_tool(list_models)
mcp.add_tool(inspect_data)
mcp.add_tool(list_datasets)
mcp.add_tool(tune_model)
mcp.add_tool(generate_forecast)
mcp.add_tool(backtest_model)
mcp.add_tool(generate_evaluation_report)
mcp.add_tool(generate_backtest_report)
mcp.add_tool(merge_covariates)
mcp.add_tool(fetch_weather_forecast)
mcp.add_tool(generate_inference_dashboard)
mcp.add_tool(batch_train_forecast_models)
mcp.add_tool(batch_generate_forecast)


def get_algorithm_specifications() -> dict:
    """
    Return all algorithm specifications for AI agent discovery.

    Loads and returns all YAML specs from the specs/ directory.
    Used by AlphaBuilding-Agents SpecRegistry for algorithm selection.

    Returns:
        Dict with list of algorithm specifications
    """
    specs = get_all_specs()
    return {
        "success": True,
        "specs": specs,
        "count": len(specs),
    }


mcp.add_tool(get_algorithm_specifications)


def run_server(transport_mode: str = "stdio", port: int = 8003) -> None:
    """Run the MCP server with the specified transport."""
    if transport_mode == "http":
        logger.info("Starting LoadForecasting-MCP server in HTTP mode on port %d", port)
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        logger.info("Starting LoadForecasting-MCP server in STDIO mode")
        mcp.run(transport="stdio")
