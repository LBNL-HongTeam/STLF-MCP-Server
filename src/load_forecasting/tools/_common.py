"""
Shared helpers for the load-forecasting MCP tool implementations.

This module holds response wrappers, model-loading utilities, ML best-practice
guardrail checks, and small series/DataFrame helpers used across the
per-workflow tool modules (train, tune, evaluation, backtest, inference,
inspection, reports, data_prep).
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd

from ..core.data_loader import ForecastingDataLoader, DataLoadError
from ..core.evaluator import (
    calculate_peak_metrics,
)
from ..core.model_registry import (
    ModelRegistry,
    ModelNotFoundError,
    ModelCorruptedError,
)
from ..core.frequency_utils import hours_to_steps, FREQ_TO_STEPS_PER_HOUR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ML best-practice guardrail helpers
# ---------------------------------------------------------------------------

def _check_frequency_mismatch(loader: "ForecastingDataLoader", model_frequency: str) -> Optional[str]:
    """
    Return an error string if the loaded CSV frequency differs from the model's
    training frequency, or None if they match (or frequency cannot be inferred).

    A mismatch is a hard-block condition: hours_to_steps conversions would be
    wrong, producing metrics computed on an incorrectly-aligned series.
    """
    inferred = getattr(loader, "inferred_frequency", None)
    if inferred is not None and inferred != model_frequency:
        return (
            f"Data frequency mismatch: the CSV appears to be '{inferred}' frequency "
            f"but the model was trained on '{model_frequency}' frequency. "
            f"Provide a '{model_frequency}' CSV or retrain a model on '{inferred}' data. "
            "Using mismatched frequencies produces incorrect metrics."
        )
    return None


def _check_covariate_mismatch(
    loader: "ForecastingDataLoader",
    metadata: dict,
) -> Optional[str]:
    """
    Return an error string if the covariate columns available in the loaded CSV
    do not match the ones the model was trained on, or None if they match.

    This is a hard-block condition. The fitted covariate scalers expect an exact
    column count and order; feeding a different set raises an opaque sklearn
    error ("X has N features, but MinMaxScaler is expecting M features") deep
    inside Darts. Checking up front lets us name the offending columns instead.

    Note that ``to_darts_series`` silently drops mapped columns that are absent
    from the DataFrame, so the comparison uses the intersection of the mapping
    and the actual DataFrame columns.
    """
    trained_mapping = metadata.get("column_mapping") or {}
    problems: list = []

    for key, label in (
        ("past_covariates", "past covariate"),
        ("future_covariates", "future covariate"),
    ):
        expected = list(trained_mapping.get(key) or [])
        if not expected:
            continue
        available = [
            c for c in (loader.column_mapping.get(key) or [])
            if c in loader.df.columns
        ]
        missing = [c for c in expected if c not in available]
        extra = [c for c in available if c not in expected]
        if not missing and not extra:
            continue

        detail = f"{label} columns do not match the model"
        if missing:
            detail += f"; missing from the CSV: {missing}"
        if extra:
            detail += f"; unexpected extra columns: {extra}"
        problems.append(detail)

    if not problems:
        return None

    return (
        "Covariate mismatch: " + "; ".join(problems) + ". "
        "The model was trained with a fixed covariate set and its fitted scalers "
        "cannot be applied to a different one. Supply a CSV containing the same "
        "covariate columns (and a column_mapping that declares them), or retrain "
        "a model on the covariates you have."
    )


def _check_training_data_overlap(
    loader: "ForecastingDataLoader",
    metadata: dict,
) -> Optional[str]:
    """
    Return a warning string if the test CSV appears to be the same data used
    for training (heuristic: date range is fully contained within the training
    period AND row count is within 10% of the training total).

    Evaluating on training data produces optimistic, unreliable metrics.
    """
    data_info = metadata.get("data_info", {})
    train_start_str = data_info.get("start_date")
    train_end_str = data_info.get("end_date")
    train_total = data_info.get("total_samples")

    if not (train_start_str and train_end_str and train_total):
        return None

    try:
        train_start = pd.Timestamp(train_start_str)
        train_end = pd.Timestamp(train_end_str)
        test_start = loader.df.index.min()
        test_end = loader.df.index.max()
        test_total = len(loader.df)

        # Make timestamps timezone-naive for comparison
        if train_start.tzinfo is not None:
            train_start = train_start.tz_localize(None)
        if train_end.tzinfo is not None:
            train_end = train_end.tz_localize(None)
        if test_start.tzinfo is not None:
            test_start = test_start.tz_localize(None)
        if test_end.tzinfo is not None:
            test_end = test_end.tz_localize(None)

        date_overlap = test_start >= train_start and test_end <= train_end
        row_similarity = abs(test_total - train_total) / max(train_total, 1) <= 0.10

        if date_overlap and row_similarity:
            return (
                "The test CSV date range and row count closely match the training data "
                f"(training: {train_start.date()} to {train_end.date()}, {train_total} rows; "
                f"test: {test_start.date()} to {test_end.date()}, {test_total} rows). "
                "Evaluating on training data produces optimistic metrics that do not reflect "
                "real-world performance. Use a held-out CSV from a different time period."
            )
    except Exception:
        pass  # If comparison fails, skip the check silently

    return None


def _check_short_test_data(
    loader: "ForecastingDataLoader",
    lookback_steps: int,
    frequency: str,
) -> Optional[str]:
    """
    Return a warning if the effective test window (after burning lookback rows)
    is less than 168 hourly-equivalent steps (1 week).
    """
    steps_per_hour = FREQ_TO_STEPS_PER_HOUR.get(frequency, 1)
    effective_points = len(loader.df) - lookback_steps
    min_recommended = 168 * steps_per_hour
    if effective_points < min_recommended:
        effective_hours = effective_points / steps_per_hour
        return (
            f"Short test window: only {effective_points} usable steps "
            f"({effective_hours:.0f}h) after the {lookback_steps}-step lookback burn-in. "
            "At least 168h (1 week) of test data is recommended for statistically "
            "meaningful metrics. Current metrics may not generalise."
        )
    return None


def _check_interpolation(loader: "ForecastingDataLoader") -> Optional[str]:
    """
    Return a warning if the loader interpolated missing target values.
    The loader stores pre/post counts in data_summary; we check missing_value_count.
    """
    summary = loader.get_data_summary()
    n_missing = summary.get("missing_value_count", 0)
    if n_missing and n_missing > 0:
        return (
            f"{n_missing} missing target value(s) were interpolated before evaluation. "
            "Metrics computed on interpolated rows reflect estimated, not observed, actuals "
            "and will be artificially good in those windows."
        )
    return None


def _check_overfitting(training_metrics: dict, validation_metrics: dict) -> Optional[str]:
    """
    Return a warning if validation CV-RMSE is more than 2× the training CV-RMSE,
    indicating the model has likely overfit the training data.
    """
    train_cv = training_metrics.get("cv_rmse")
    val_cv = validation_metrics.get("cv_rmse")
    if train_cv and val_cv and train_cv > 0:
        ratio = val_cv / train_cv
        if ratio > 2.0:
            return (
                f"Possible overfitting: validation CV-RMSE ({val_cv:.2f}%) is "
                f"{ratio:.1f}× higher than training CV-RMSE ({train_cv:.2f}%). "
                "Consider reducing model complexity, adding regularisation, or "
                "increasing training data."
            )
    return None


def _check_horizon_lookback(horizon_hours: int, lookback_hours: int) -> Optional[str]:
    """Return a warning if horizon exceeds lookback — unusual and likely misconfigured."""
    if horizon_hours > lookback_hours:
        return (
            f"horizon_hours ({horizon_hours}) exceeds lookback_hours ({lookback_hours}). "
            "Forecasting further ahead than the input window is unusual and typically "
            "degrades accuracy. Consider setting lookback_hours >= horizon_hours."
        )
    return None


def _check_high_step_count(horizon_steps: int, frequency: str) -> Optional[str]:
    """Return a warning when the effective step count is high for non-hourly data."""
    if frequency != "h" and horizon_steps > 96:
        return (
            f"horizon_steps={horizon_steps} at {frequency} frequency is a long forecast "
            "horizon. Accuracy typically degrades with longer horizons. "
            "Consider reducing horizon_hours or using a more powerful model (TFT, TiDE, TSMixer, XGBoost)."
        )
    return None


def _check_arima_lookback(model_type: str, lookback_hours: int) -> Optional[str]:
    """Return a warning when ARIMA is used with an explicit lookback_hours (which it ignores)."""
    if model_type == "ARIMA" and lookback_hours != 24:
        return (
            f"ARIMA ignores lookback_hours (got {lookback_hours}h). "
            "ARIMA's effective lookback is controlled by the AR order p. "
            "Set p in the search_space when using tune_model, or pass model_kwargs "
            "with {'p': <value>} to train_forecast_model."
        )
    return None


# ---------------------------------------------------------------------------
# Response wrappers
# ---------------------------------------------------------------------------

def create_success_response(**data) -> dict:
    """Wrap successful response."""
    return {
        "success": True,
        "error": None,
        **data,
    }


def create_error_response(error: str, **partial_data) -> dict:
    """Wrap error response with any partial data collected."""
    return {
        "success": False,
        "error": error,
        **partial_data,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

class _ModelLoadError(Exception):
    """Raised by _load_model_context when a model cannot be loaded."""

    def __init__(self, response: dict):
        self.response = response


def _load_model_context(model_id: str, column_mapping: Optional[dict]) -> dict:
    """Load a model + metadata + scalers and extract common config.

    Returns a context dict with keys: registry, model, metadata, scalers,
    model_type, column_mapping, config, frequency, lookback_hours,
    horizon_hours, lookback_steps, horizon_steps.

    Raises _ModelLoadError (carrying an error-response dict) on failure so
    callers can ``except _ModelLoadError as e: return e.response``.
    """
    registry = ModelRegistry()
    try:
        model, metadata, scalers = registry.load_model(model_id)
    except ModelNotFoundError:
        raise _ModelLoadError(create_error_response(f"Model not found: {model_id}"))
    except ModelCorruptedError as e:
        raise _ModelLoadError(create_error_response(str(e)))

    if column_mapping is None:
        column_mapping = metadata.get("column_mapping")

    config = metadata.get("config", {})
    frequency = config.get("frequency", "h")
    lookback_hours = config.get("lookback_hours", 24)
    horizon_hours = config.get("horizon_hours", 6)
    return {
        "registry": registry,
        "model": model,
        "metadata": metadata,
        "scalers": scalers,
        "model_type": metadata.get("model_type", ""),
        "column_mapping": column_mapping,
        "config": config,
        "frequency": frequency,
        "lookback_hours": lookback_hours,
        "horizon_hours": horizon_hours,
        "lookback_steps": hours_to_steps(lookback_hours, frequency),
        "horizon_steps": hours_to_steps(horizon_hours, frequency),
    }


# ---------------------------------------------------------------------------
# Warning collection utilities
# ---------------------------------------------------------------------------

def _collect_warnings(*checks) -> list:
    """Run zero-arg check callables and collect their non-empty string results."""
    warnings: list = []
    for check in checks:
        w = check()
        if w:
            warnings.append(w)
    return warnings


def _append_warning(warnings: list, value) -> None:
    """Append ``value`` to ``warnings`` if it is truthy."""
    if value:
        warnings.append(value)


def _peak_metrics_safe(actual, predicted, peak_dates, scaler) -> Optional[dict]:
    """Compute peak metrics, returning None (and logging) on failure."""
    if not peak_dates:
        return None
    try:
        return calculate_peak_metrics(
            actual=actual, predicted=predicted, peak_dates=peak_dates, scaler=scaler
        )
    except Exception as e:
        logger.warning("Failed to compute peak metrics: %s", e)
        return None


def _merge_peak_headline(metrics: dict, peak_metrics: Optional[dict]) -> None:
    """Surface the peak headline numbers alongside the standard metrics.

    Adds ``peak_mape`` and ``peak_timing_error_hours`` into the standard
    ``metrics`` dict (in-place) so they appear next to rmse/mae/mape/cv_rmse/
    r_squared. The full ``peak_metrics`` dict (per-day detail, counts) is left
    untouched as a separate top-level response key. No-op when peak metrics
    were not computed.
    """
    if not peak_metrics:
        return
    metrics["peak_mape"] = peak_metrics.get("peak_mape")
    metrics["peak_timing_error_hours"] = peak_metrics.get("peak_timing_error_hours")


# ---------------------------------------------------------------------------
# Series / DataFrame helpers
# ---------------------------------------------------------------------------

def _df_to_records(df: "pd.DataFrame", datetime_col: str) -> list:
    """Convert a DataFrame to a list of JSON-safe dicts (ISO datetimes, py scalars)."""
    records = []
    for row in df.itertuples(index=False):
        rec = {}
        for col, val in zip(df.columns, row):
            if col == datetime_col and hasattr(val, "isoformat"):
                rec[col] = val.isoformat()
            elif hasattr(val, "item"):
                rec[col] = val.item()
            else:
                rec[col] = val
        records.append(rec)
    return records


def _validate_html_output_path(output_html_path: str) -> Optional[str]:
    """Validate an HTML output path; return an error string or None."""
    if not output_html_path.lower().endswith((".html", ".htm")):
        return "output_html_path must end with .html or .htm"
    Path(output_html_path).parent.mkdir(parents=True, exist_ok=True)
    return None


def _records_from_series(pred_series, actual_series) -> list:
    """Build a list of {timestamp, actual, predicted, residual} dicts from two series."""
    pred_times = pred_series.time_index
    pred_values = pred_series.values().flatten()
    actual_values = actual_series.values().flatten()
    min_len = min(len(pred_times), len(pred_values), len(actual_values))
    return [
        {
            "timestamp": t.isoformat(),
            "actual": float(a),
            "predicted": float(p),
            "residual": float(a - p),
        }
        for t, a, p in zip(
            pred_times[:min_len], actual_values[:min_len], pred_values[:min_len]
        )
    ]


def _load_and_scale_series(
    csv_path: str,
    column_mapping: dict,
    scalers: dict,
    frequency: str,
    dataframe=None,
    metadata: Optional[dict] = None,
) -> tuple:
    """
    Load a CSV (or pre-loaded DataFrame) and apply pre-trained scalers.

    This is the shared data-loading core used by evaluate_forecast_model,
    generate_forecast, and backtest_model.

    Args:
        csv_path: Path to CSV (used for metadata even when dataframe is provided).
        column_mapping: Column role mapping (resolved from model metadata).
        scalers: Dict with 'target_scaler', 'covariate_scaler',
            'future_covariate_scaler' (values may be None).
        frequency: Data frequency string ('h', '30min', '15min').
        dataframe: Optional pre-loaded DataFrame to pass directly to the
            ForecastingDataLoader, skipping file I/O. Useful when the caller
            has already filtered out horizon rows before loading.
        metadata: Optional model metadata. When supplied, the CSV's covariate
            columns are validated against the training covariate set before the
            fitted scalers are applied, so a mismatch produces a readable error
            instead of an opaque sklearn feature-count failure.

    Returns:
        Tuple of (loader, target_series, past_covariate_series or None,
        future_covariate_series or None).

    Raises:
        DataLoadError: Propagated from ForecastingDataLoader on bad data.
        _ModelLoadError: When the covariate set does not match the model's.
    """
    loader = ForecastingDataLoader(
        csv_path=csv_path,
        column_mapping=column_mapping,
        frequency=frequency,
        dataframe=dataframe,
    )
    if metadata is not None:
        cov_err = _check_covariate_mismatch(loader, metadata)
        if cov_err:
            raise _ModelLoadError(create_error_response(cov_err))

    if "target_scaler" in scalers and scalers["target_scaler"] is not None:
        loader.target_scaler = scalers["target_scaler"]
    if "covariate_scaler" in scalers and scalers["covariate_scaler"] is not None:
        loader.covariate_scaler = scalers["covariate_scaler"]
    if "future_covariate_scaler" in scalers and scalers["future_covariate_scaler"] is not None:
        loader.future_covariate_scaler = scalers["future_covariate_scaler"]

    target_series, past_cov, future_cov = loader.to_darts_series(fit_scalers=False)
    return loader, target_series, past_cov, future_cov


def _parse_raw_csv_for_inference(csv_path: str, column_mapping: dict) -> pd.DataFrame:
    """
    Load raw CSV with minimal processing, keeping datetime as a regular column.

    Used by generate_forecast to split a CSV into context rows and future
    horizon rows before passing through ForecastingDataLoader.  Datetime is
    kept as a column (not set as index) so the DataFrame can be passed
    directly to ForecastingDataLoader, which handles its own index parsing.

    Args:
        csv_path: Path to the CSV file.
        column_mapping: Resolved column mapping (must have 'datetime' key).

    Returns:
        DataFrame with an integer index. The datetime column is parsed to
        Timestamp but remains a column.  Rows are sorted by datetime.

    Raises:
        DataLoadError: If the file cannot be read or datetime cannot be parsed.
    """
    try:
        raw_df = pd.read_csv(csv_path)
    except Exception as e:
        raise DataLoadError(f"Failed to read CSV '{csv_path}': {e}")

    datetime_col = column_mapping.get("datetime")
    if datetime_col and datetime_col in raw_df.columns:
        raw_df[datetime_col] = pd.to_datetime(raw_df[datetime_col], utc=False)
        # Strip timezone to stay consistent with ForecastingDataLoader
        if raw_df[datetime_col].dt.tz is not None:
            raw_df[datetime_col] = raw_df[datetime_col].dt.tz_localize(None)
        raw_df = raw_df.sort_values(datetime_col).reset_index(drop=True)
    else:
        # Positional fallback — try first column
        first_col = raw_df.columns[0]
        try:
            raw_df[first_col] = pd.to_datetime(raw_df[first_col], utc=False)
            if raw_df[first_col].dt.tz is not None:
                raw_df[first_col] = raw_df[first_col].dt.tz_localize(None)
            raw_df = raw_df.sort_values(first_col).reset_index(drop=True)
        except Exception:
            raise DataLoadError(
                "Could not parse a datetime column from the CSV. "
                "Provide column_mapping={'datetime': '<column_name>', ...}."
            )

    return raw_df
