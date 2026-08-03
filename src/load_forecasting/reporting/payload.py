"""
Build the JSON-serialisable payload that drives the HTML evaluation report.

The payload bundles model metadata, evaluation metrics, prediction series,
input data preview, and pre-computed aggregations (hour-of-day MAE, day-of-week
MAE) so the embedded JS does not have to do any heavy grouping client-side.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from ._render import to_html_json


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReportPayload:
    """Top-level payload consumed by the report's JS via window.__REPORT_DATA__."""

    meta: dict
    metrics: dict
    residual_analysis: dict
    series: dict
    aggregations: dict
    input_summary: dict
    peak_metrics: dict = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------


class _ReportJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars, pandas Timestamps and NaN."""

    def default(self, obj: Any) -> Any:  # noqa: D401
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def payload_to_json(payload: ReportPayload) -> str:
    """Serialise a ReportPayload to a JSON string safe to embed in HTML.

    The output is escaped so a literal ``</script>`` in any string value
    cannot terminate the surrounding <script> block in the rendered page.
    """
    return to_html_json(payload.to_dict(), encoder=_ReportJSONEncoder)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _round(v: Any, n: int = 4) -> Optional[float]:
    """Round a numeric value, returning None for non-finite or non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, n)


def _round_peak_metrics(pm: Optional[dict]) -> dict:
    """Round the numeric fields of a peak_metrics dict for JSON embedding.

    Returns an empty dict when ``pm`` is falsy so downstream JS can treat
    "no peak metrics requested" and "peak metrics failed" uniformly.
    """
    if not pm:
        return {}
    out = {
        "peak_mape": _round(pm.get("peak_mape"), 2),
        "peak_timing_error_hours": _round(pm.get("peak_timing_error_hours"), 2),
        "n_peak_days_evaluated": int(pm.get("n_peak_days_evaluated", 0) or 0),
        "n_peak_days_skipped": int(pm.get("n_peak_days_skipped", 0) or 0),
    }
    # ``calculate_peak_metrics`` uses the key "per_day"; the Pydantic
    # response schema exposes it as "per_day_results".  Accept either
    # so this helper works regardless of the caller's convention.
    per_day = pm.get("per_day") or pm.get("per_day_results") or []
    out["per_day_results"] = [
        {
            "date": r.get("date"),
            "actual_peak_value": _round(r.get("actual_peak_value"), 4),
            "predicted_peak_value": _round(r.get("predicted_peak_value"), 4),
            "actual_peak_hour": r.get("actual_peak_hour"),
            "predicted_peak_hour": r.get("predicted_peak_hour"),
            "peak_magnitude_error_pct": _round(r.get("peak_magnitude_error_pct"), 2),
            "peak_timing_error_hours": _round(r.get("peak_timing_error_hours"), 2),
        }
        for r in per_day
    ]
    return out


def _safe_iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return None


def _build_training_history(model_metadata: dict) -> dict:
    """Extract a rounded per-epoch learning curve from model metadata.

    Reads ``training_info.training_history`` (recorded for Torch models during
    training).  Returns ``{}`` when no history was captured (non-Torch models
    or models trained before the feature existed), so the report JS can hide
    the section gracefully.
    """
    hist = (model_metadata.get("training_info", {}) or {}).get("training_history")
    if not hist:
        return {}
    epochs = hist.get("epochs") or []
    if not epochs:
        return {}
    train_loss = hist.get("train_loss") or []
    val_loss = hist.get("val_loss") or []
    points = []
    for i, ep in enumerate(epochs):
        tl = _round(train_loss[i], 6) if i < len(train_loss) else None
        vl = _round(val_loss[i], 6) if i < len(val_loss) else None
        points.append({"epoch": int(ep), "train_loss": tl, "val_loss": vl})
    has_val = any(p["val_loss"] is not None for p in points)
    return {"points": points, "has_val": has_val, "n_epochs": len(points)}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_report_payload(
    *,
    model_id: str,
    model_metadata: dict,
    eval_result: dict,
    input_df: Optional[pd.DataFrame],
    column_mapping: Optional[dict],
    title: Optional[str] = None,
) -> ReportPayload:
    """
    Build a ReportPayload from an evaluation result.

    Args:
        model_id: Model identifier.
        model_metadata: Dict returned by ModelRegistry.load_model (the metadata
            portion).  Used for header info: model_type, building_name, config.
        eval_result: Dict returned by ``evaluate_forecast_model``.  Must contain
            ``predictions`` (list of {timestamp, actual, predicted, residual})
            and ``test_metrics``.  May contain ``residual_analysis`` and
            ``comparison_to_validation``.
        input_df: Optional raw test DataFrame (datetime-indexed) used to render
            the "input data preview" section.  Pass None to skip that section.
        column_mapping: Resolved column mapping (target + covariates).
        title: Optional human-readable report title override.

    Returns:
        A ReportPayload ready to be JSON-serialised and embedded in the HTML
        template.
    """
    if not eval_result.get("success", False):
        raise ValueError(
            f"Cannot build report from failed evaluation: {eval_result.get('error')}"
        )

    predictions = eval_result.get("predictions") or []
    if not predictions:
        raise ValueError(
            "Evaluation result has no predictions; call evaluate_forecast_model "
            "with return_predictions=True."
        )

    config = model_metadata.get("config", {}) or {}
    building_name = model_metadata.get("building_name")
    model_type = model_metadata.get("model_type") or eval_result.get("model_type")
    created_at = model_metadata.get("created_at")
    train_data_info = model_metadata.get("data_info", {}) or {}

    # ---- meta -----------------------------------------------------------
    meta = {
        "title": title or f"Evaluation report — {model_id}",
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
        "test_data_range": eval_result.get("test_summary", {}),
        "column_mapping": column_mapping or {},
        "training_history": _build_training_history(model_metadata),
    }

    # ---- metrics --------------------------------------------------------
    test_metrics = eval_result.get("test_metrics", {}) or {}
    metrics = {
        "test": {k: _round(v, 4) for k, v in test_metrics.items()},
        "validation": {
            k: _round(v, 4)
            for k, v in (model_metadata.get("metrics", {}) or {})
            .get("validation", {})
            .items()
        },
        "comparison": eval_result.get("comparison_to_validation", {}) or {},
    }

    # ---- residual analysis ---------------------------------------------
    residual_analysis = eval_result.get("residual_analysis") or {}

    # ---- predictions series --------------------------------------------
    # Normalise types: ensure floats are JSON-finite, ISO strings for ts.
    series_predictions = []
    for row in predictions:
        series_predictions.append(
            {
                "t": row.get("timestamp"),
                "actual": _round(row.get("actual"), 4),
                "predicted": _round(row.get("predicted"), 4),
                "residual": _round(row.get("residual"), 4),
            }
        )

    # ---- input target + covariates -------------------------------------
    series_input_target, series_covariates, input_summary = build_input_preview(
        input_df, column_mapping
    )

    # ---- aggregations: hour-of-day MAE, day-of-week MAE -----------------
    aggregations = build_error_aggregations(predictions)

    # ---- peak metrics (optional) ---------------------------------------
    peak_metrics = _round_peak_metrics(eval_result.get("peak_metrics"))

    return ReportPayload(
        meta=meta,
        metrics=metrics,
        residual_analysis=residual_analysis,
        series={
            "predictions": series_predictions,
            "input_target": series_input_target,
            "covariates": series_covariates,
        },
        aggregations=aggregations,
        input_summary=input_summary,
        peak_metrics=peak_metrics,
    )


def _column_stats(name: str, s: pd.Series) -> dict:
    """Compute small set of stats for the input-data preview table."""
    if len(s) == 0:
        return {"column": name, "n": 0}
    return {
        "column": name,
        "n": int(len(s)),
        "mean": _round(s.mean(), 4),
        "std": _round(s.std(), 4),
        "min": _round(s.min(), 4),
        "max": _round(s.max(), 4),
        "p5": _round(np.percentile(s, 5), 4),
        "p95": _round(np.percentile(s, 95), 4),
    }


_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_error_aggregations(predictions: list[dict]) -> dict:
    """Compute hour-of-day and day-of-week MAE aggregations from predictions."""
    aggregations: dict = {}
    if not predictions:
        return aggregations
    pred_df = pd.DataFrame(predictions)
    if pred_df.empty or not {"timestamp", "actual", "predicted"}.issubset(pred_df.columns):
        return aggregations

    pred_df = pred_df.copy()
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"], errors="coerce")
    pred_df = pred_df.dropna(subset=["timestamp"])
    pred_df["abs_err"] = (pred_df["actual"] - pred_df["predicted"]).abs()
    pred_df["hour"] = pred_df["timestamp"].dt.hour
    pred_df["dow"] = pred_df["timestamp"].dt.dayofweek

    by_hour = pred_df.groupby("hour")["abs_err"].mean().reindex(range(24)).reset_index()
    aggregations["hour_of_day_mae"] = [
        {"hour": int(r.hour), "mae": _round(r.abs_err, 4)}
        for r in by_hour.itertuples(index=False)
    ]

    by_dow = pred_df.groupby("dow")["abs_err"].mean().reindex(range(7)).reset_index()
    aggregations["day_of_week_mae"] = [
        {"dow": int(r.dow), "label": _DOW_LABELS[int(r.dow)], "mae": _round(r.abs_err, 4)}
        for r in by_dow.itertuples(index=False)
    ]
    return aggregations


def build_input_preview(
    input_df: Optional[pd.DataFrame], column_mapping: Optional[dict]
) -> tuple[list, dict, dict]:
    """Build (input_target_series, covariate_series, input_summary) preview blocks."""
    series_input_target: list = []
    series_covariates: dict = {}
    input_summary: dict = {}

    if input_df is None or not column_mapping:
        return series_input_target, series_covariates, input_summary

    target_col = column_mapping.get("target")
    past_covs = column_mapping.get("past_covariates", []) or []
    future_covs = column_mapping.get("future_covariates", []) or []

    if target_col and target_col in input_df.columns:
        target_series = input_df[target_col].dropna()
        series_input_target = [
            {"t": _safe_iso(ts), "value": _round(val, 4)}
            for ts, val in target_series.items()
        ]
        input_summary["target"] = _column_stats(target_col, target_series)

    for col in list(past_covs) + list(future_covs):
        if col in input_df.columns:
            s = input_df[col].dropna()
            series_covariates[col] = [
                {"t": _safe_iso(ts), "value": _round(val, 4)} for ts, val in s.items()
            ]
            input_summary.setdefault("covariates", {})[col] = _column_stats(col, s)

    return series_input_target, series_covariates, input_summary
