"""
FastMCP Server Configuration and Tool Registration
"""

import logging
import os

from fastmcp import FastMCP

from .core.spec_loader import get_all_specs
from .tools import (
    train_forecast_model,
    evaluate_forecast_model,
    list_models,
    inspect_data,
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

# Initialize FastMCP server
mcp = FastMCP(name="load_forecasting")

# Register tools directly — no wrapper functions needed
mcp.add_tool(train_forecast_model)
mcp.add_tool(evaluate_forecast_model)
mcp.add_tool(list_models)
mcp.add_tool(inspect_data)
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
