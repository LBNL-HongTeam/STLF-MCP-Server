"""
FastMCP Server Configuration and Tool Registration

This module sets up the MCP server and registers all forecasting tools.
"""

import logging
import os

from fastmcp import FastMCP

from .core.spec_loader import get_all_specs
from .tools.forecasting_tools import (
    train_forecast_model,
    evaluate_forecast_model,
    list_models,
    inspect_data,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="load_forecasting",
)


# Register tools
@mcp.tool(name="train_forecast_model")
async def _train_forecast_model(
    csv_path: str,
    model_type: str = "LinearRegression",
    lookback_hours: int = 24,
    horizon_hours: int = 6,
    frequency: str = "h",
    validation_split: float = 0.2,
    building_name: str | None = None,
    model_name: str | None = None,
    column_mapping: dict | None = None,
) -> dict:
    """
    Train a time-series forecasting model on historical building load data.

    Evaluates on a validation split and persists the trained model for later use.
    Supports naive baselines and linear regression models.

    Args:
        csv_path: Path to CSV file with datetime index and load data
        model_type: Model type (NaiveMean, NaiveSeasonal, NaiveMovingAverage, LinearRegression)
        lookback_hours: Hours of history to use as model input (1-168)
        horizon_hours: Hours ahead to forecast (1-48)
        frequency: Data frequency (15min, 30min, h)
        validation_split: Fraction of data for validation (0.1-0.3)
        building_name: Building identifier for model naming
        model_name: Custom model name (auto-generated if not provided)
        column_mapping: Map CSV columns to roles (datetime, target, past_covariates)

    Returns:
        Dict with model_id, metrics, and data summary
    """
    return await train_forecast_model(
        csv_path=csv_path,
        model_type=model_type,
        lookback_hours=lookback_hours,
        horizon_hours=horizon_hours,
        frequency=frequency,
        validation_split=validation_split,
        building_name=building_name,
        model_name=model_name,
        column_mapping=column_mapping,
    )


@mcp.tool(name="evaluate_forecast_model")
async def _evaluate_forecast_model(
    model_id: str,
    csv_path: str,
    column_mapping: dict | None = None,
    return_predictions: bool = True,
    output_csv_path: str | None = None,
    include_residual_analysis: bool = False,
) -> dict:
    """
    Evaluate a trained forecasting model on new/test data.

    Loads a previously trained model and generates predictions with accuracy metrics.
    Uses column mapping from training if not explicitly provided.

    Args:
        model_id: ID of trained model (from train_forecast_model)
        csv_path: Path to CSV with test data
        column_mapping: Column mapping (uses training mapping if not provided)
        return_predictions: Include predicted values in output
        output_csv_path: Optional path to save predictions as CSV
        include_residual_analysis: Include residual statistics

    Returns:
        Dict with test metrics, comparison to validation, and predictions
    """
    return await evaluate_forecast_model(
        model_id=model_id,
        csv_path=csv_path,
        column_mapping=column_mapping,
        return_predictions=return_predictions,
        output_csv_path=output_csv_path,
        include_residual_analysis=include_residual_analysis,
    )


@mcp.tool(name="list_models")
async def _list_models(
    building_name: str | None = None,
    model_type: str | None = None,
    sort_by: str = "created_at",
    limit: int = 20,
) -> dict:
    """
    List all trained models in the registry with their metadata and metrics.

    Args:
        building_name: Filter models by building name
        model_type: Filter by model type (e.g., LinearRegression)
        sort_by: Sort order (created_at, validation_cv_rmse, model_type)
        limit: Maximum number of models to return

    Returns:
        Dict with list of models and total count
    """
    return await list_models(
        building_name=building_name,
        model_type=model_type,
        sort_by=sort_by,
        limit=limit,
    )


@mcp.tool(name="inspect_data")
async def _inspect_data(
    csv_path: str,
    column_mapping: dict | None = None,
    frequency: str | None = None,
) -> dict:
    """
    Inspect a CSV file before training — detect columns, frequency, gaps,
    statistics, quality issues, and feature suggestions.

    Run this before train_forecast_model to understand the data and catch
    problems early.

    Args:
        csv_path: Path to CSV file to inspect
        column_mapping: Optional explicit column roles (datetime, target,
            past_covariates). Auto-detected if not provided.
        frequency: Expected data frequency ('15min', '30min', 'h').
            Inferred from timestamps if not provided.

    Returns:
        Dict with detected columns, frequency, statistics, gap analysis,
        quality flags, feature suggestions, and a ready_to_train flag.
    """
    return await inspect_data(
        csv_path=csv_path,
        column_mapping=column_mapping,
        frequency=frequency,
    )


@mcp.tool(name="get_algorithm_specifications")
async def _get_algorithm_specifications() -> dict:
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


def run_server(transport_mode: str = "stdio"):
    """Run the MCP server with specified transport."""
    port = int(os.getenv("MCP_HTTP_PORT", "8003"))

    if transport_mode == "http":
        logger.info(f"Starting LoadForecasting-MCP server in HTTP mode on port {port}")
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        logger.info("Starting LoadForecasting-MCP server in STDIO mode")
        mcp.run(transport="stdio")


def main():
    """Entry point for console script."""
    transport_mode = os.getenv("MCP_TRANSPORT", "stdio").lower()
    run_server(transport_mode)


if __name__ == "__main__":
    main()
