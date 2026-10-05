"""
HTML reporting package for load forecasting evaluations.

Produces a single, self-contained HTML file that embeds its own JS
(d3 + Observable Plot) and its own data, so the report can be opened
in any browser without a server.
"""

from .builder import build_report_html
from .payload import build_report_payload, ReportPayload
from .backtest_builder import build_backtest_report_html
from .backtest_payload import build_backtest_payload, BacktestReportPayload
from .inference_builder import build_inference_dashboard_html
from .data_builder import build_data_report_html
from .data_payload import build_data_report_payload, DataReportPayload, SPLIT_STRATEGIES
from .training_builder import build_training_report_html
from .training_payload import build_training_report_payload, TrainingReportPayload

__all__ = [
    "build_report_html",
    "build_report_payload",
    "ReportPayload",
    "build_backtest_report_html",
    "build_backtest_payload",
    "BacktestReportPayload",
    "build_inference_dashboard_html",
    "build_data_report_html",
    "build_data_report_payload",
    "DataReportPayload",
    "SPLIT_STRATEGIES",
    "build_training_report_html",
    "build_training_report_payload",
    "TrainingReportPayload",
]
