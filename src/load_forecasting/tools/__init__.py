"""
MCP Tool Wrappers for Load Forecasting.

These functions are registered as MCP tools in server.py.
"""

from .forecasting_tools import (
    train_forecast_model,
    evaluate_forecast_model,
    list_models,
)

__all__ = [
    "train_forecast_model",
    "evaluate_forecast_model",
    "list_models",
]
