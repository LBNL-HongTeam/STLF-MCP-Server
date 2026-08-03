"""
Build the JSON-serialisable payload that drives the HTML backtest report.

The backtest payload extends the evaluation payload with:
  - per-window forecast data (for the playback slider)
  - h-step-ahead horizon metrics (RMSE/MAE/MAPE per horizon step)
  - overall backtest summary metadata
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._render import to_html_json
from .payload import (
    _ReportJSONEncoder,
    _round,
    _round_peak_metrics,
    _safe_iso,
    build_error_aggregations,
    build_input_preview,
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestReportPayload:
    """Top-level payload consumed by the backtest report JS."""

    meta: dict
    metrics: dict
    backtest_summary: dict
    # Per-window data for the playback slider
    windows: list
    # Full actual series (background context for the slider chart)
    full_actual: list
    # H-step-ahead RMSE / MAE / MAPE
    horizon_metrics: list
    # Hour-of-day MAE and day-of-week MAE (same as eval report)
    aggregations: dict
    # Optional residual analysis
    residual_analysis: dict
    # Optional input data summary
    input_summary: dict
    # Optional peak-day metrics (PMAPE / PTE + per-day breakdown)
    peak_metrics: dict = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return asdict(self)


def backtest_payload_to_json(payload: BacktestReportPayload) -> str:
    """Serialise a BacktestReportPayload to a JSON string safe to embed in HTML."""
    return to_html_json(payload.to_dict(), encoder=_ReportJSONEncoder)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_backtest_payload(
    *,
    model_id: str,
    model_metadata: dict,
    backtest_result: dict,
    windows_raw: list,  # list[TimeSeries] from historical_forecasts(last_points_only=False)
    actual_series_inv: Any,  # inverse-transformed actual TimeSeries (backtest region)
    horizon_metrics: list,
    input_df: Optional[pd.DataFrame],
    column_mapping: Optional[dict],
    title: Optional[str] = None,
) -> BacktestReportPayload:
    """
    Build a BacktestReportPayload from a backtest result and per-window data.

    Args:
        model_id: Model identifier.
        model_metadata: Dict from ModelRegistry.load_model (metadata portion).
        backtest_result: Dict returned by the backtest logic containing
            ``backtest_metrics``, ``backtest_summary``, optionally
            ``residual_analysis`` and ``predictions``.
        windows_raw: List of inverse-transformed Darts TimeSeries, one per
            forecast window (from historical_forecasts(last_points_only=False)).
            Each has length == horizon_steps.
        actual_series_inv: Inverse-transformed actual TimeSeries covering the
            backtest region (from backtest start index onward).
        horizon_metrics: List of dicts ``[{"h": 1, "rmse": ..., ...}, ...]``
            from ``calculate_horizon_metrics``.
        input_df: Optional raw test DataFrame for the input preview section.
        column_mapping: Resolved column mapping.
        title: Optional report title.

    Returns:
        BacktestReportPayload ready to be JSON-serialised and embedded in HTML.
    """
    config = model_metadata.get("config", {}) or {}
    building_name = model_metadata.get("building_name")
    model_type = model_metadata.get("model_type") or backtest_result.get("model_type")
    created_at = model_metadata.get("created_at")
    train_data_info = model_metadata.get("data_info", {}) or {}

    # ---- meta -----------------------------------------------------------
    meta = {
        "title": title or f"Backtest report — {model_id}",
        "model_id": model_id,
        "model_type": model_type,
        "building_name": building_name,
        "created_at": created_at,
        "lookback_hours": config.get("lookback_hours"),
        "horizon_hours": config.get("horizon_hours"),
        "frequency": config.get("frequency"),
        "validation_split": config.get("validation_split"),
        "train_data_range": {
            "start": train_data_info.get("start_date"),
            "end": train_data_info.get("end_date"),
            "samples": train_data_info.get("total_samples"),
        },
        "column_mapping": column_mapping or {},
    }

    # ---- metrics --------------------------------------------------------
    bt_metrics = backtest_result.get("backtest_metrics", {}) or {}
    metrics = {
        "backtest": {k: _round(v, 4) for k, v in bt_metrics.items()},
        "validation": {
            k: _round(v, 4)
            for k, v in (model_metadata.get("metrics", {}) or {})
            .get("validation", {})
            .items()
        },
        "comparison": backtest_result.get("comparison_to_validation", {}) or {},
    }

    # ---- backtest summary -----------------------------------------------
    backtest_summary = backtest_result.get("backtest_summary", {}) or {}

    # ---- per-window data for the slider ---------------------------------
    windows_payload: list = []
    actual_pd = actual_series_inv.to_dataframe().iloc[:, 0]

    for idx, window in enumerate(windows_raw):
        origin_ts = window.start_time()
        steps_data: list = []
        window_actuals: list = []
        window_predicted: list = []

        for i in range(len(window)):
            step_ts = window.time_index[i]
            pred_val = float(window.univariate_values()[i])
            actual_val: Optional[float] = None
            if step_ts in actual_pd.index:
                actual_val = float(actual_pd.loc[step_ts])
                window_actuals.append(actual_val)
                window_predicted.append(pred_val)

            steps_data.append(
                {
                    "h": i + 1,
                    "t": _safe_iso(step_ts),
                    "actual": _round(actual_val, 4),
                    "predicted": _round(pred_val, 4),
                }
            )

        # Per-window aggregate metrics
        if len(window_actuals) >= 1:
            a = np.array(window_actuals)
            p = np.array(window_predicted)
            res = a - p
            w_mae = float(np.abs(res).mean())
            w_rmse = float(np.sqrt((res ** 2).mean()))
        else:
            w_mae = None
            w_rmse = None

        windows_payload.append(
            {
                "window_id": idx,
                "forecast_origin": _safe_iso(origin_ts),
                "steps": steps_data,
                "window_mae": _round(w_mae, 4),
                "window_rmse": _round(w_rmse, 4),
            }
        )

    # ---- full actual series (context behind the slider) -----------------
    full_actual: list = [
        {"t": _safe_iso(ts), "value": _round(float(v), 4)}
        for ts, v in actual_pd.items()
        if math.isfinite(float(v))
    ]

    # ---- aggregations (hour-of-day MAE, day-of-week MAE) ---------------
    aggregations = build_error_aggregations(backtest_result.get("predictions") or [])

    # ---- residual analysis (optional) ----------------------------------
    residual_analysis = backtest_result.get("residual_analysis") or {}

    # ---- input preview -------------------------------------------------
    series_input_target, series_covariates, input_summary = build_input_preview(
        input_df, column_mapping
    )

    # ---- peak metrics (optional) ---------------------------------------
    peak_metrics = _round_peak_metrics(backtest_result.get("peak_metrics"))

    return BacktestReportPayload(
        meta=meta,
        metrics=metrics,
        backtest_summary=backtest_summary,
        windows=windows_payload,
        full_actual=full_actual,
        horizon_metrics=horizon_metrics,
        aggregations=aggregations,
        residual_analysis=residual_analysis,
        input_summary=input_summary,
        peak_metrics=peak_metrics,
    )
