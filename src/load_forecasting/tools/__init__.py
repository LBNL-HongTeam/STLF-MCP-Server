"""
MCP Tool implementations for load forecasting.

All tool functions are registered as MCP tools in server.py. They are split
across per-workflow modules (train, tune, evaluation, backtest, inference,
inspection, datasets, skills, reports, data_prep) with shared helpers in ``_common``.
"""

from .train import train_forecast_model
from .tune import tune_model
from .evaluation import evaluate_forecast_model
from .backtest import backtest_model
from .inference import generate_forecast
from .inspection import inspect_data
from .datasets import list_datasets
from .skills import list_skills, get_skill
from .reports import (
    generate_evaluation_report,
    generate_backtest_report,
    generate_inference_dashboard,
)
from .data_prep import (
    list_models,
    merge_covariates,
    fetch_weather_forecast,
)
from .batch import (
    batch_train_forecast_models,
    batch_generate_forecast,
)

__all__ = [
    "train_forecast_model",
    "tune_model",
    "evaluate_forecast_model",
    "backtest_model",
    "generate_forecast",
    "inspect_data",
    "list_datasets",
    "list_skills",
    "get_skill",
    "generate_evaluation_report",
    "generate_backtest_report",
    "generate_inference_dashboard",
    "list_models",
    "merge_covariates",
    "fetch_weather_forecast",
    "batch_train_forecast_models",
    "batch_generate_forecast",
]
