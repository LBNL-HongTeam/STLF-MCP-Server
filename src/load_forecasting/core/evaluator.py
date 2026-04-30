"""
Metrics calculation for forecasting models.

Computes:
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error (%)
- CV-RMSE: Coefficient of Variation of RMSE (%)
- R-squared: Coefficient of determination
"""

import numpy as np
from darts import TimeSeries
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(
    actual: TimeSeries,
    predicted: TimeSeries,
    scaler=None,
) -> dict:
    """
    Calculate all metrics on original scale.

    Args:
        actual: Ground truth time series
        predicted: Model predictions
        scaler: If provided, inverse transform before computing metrics

    Returns:
        Dictionary with rmse, mae, mape, cv_rmse, r_squared
    """
    # Inverse transform if scaled
    if scaler is not None:
        try:
            actual = scaler.inverse_transform(actual)
            predicted = scaler.inverse_transform(predicted)
        except Exception as e:
            logger.warning(f"Failed to inverse transform: {e}")

    # Align time series (they may have different lengths after historical_forecasts)
    actual_vals, predicted_vals = _align_series(actual, predicted)

    if len(actual_vals) == 0:
        logger.warning("No overlapping data points for metrics calculation")
        return {
            "rmse": None,
            "mae": None,
            "mape": None,
            "cv_rmse": None,
            "r_squared": None,
        }

    # RMSE
    rmse_val = float(np.sqrt(np.mean((actual_vals - predicted_vals) ** 2)))

    # MAE
    mae_val = float(np.mean(np.abs(actual_vals - predicted_vals)))

    # MAPE (handle zero values)
    non_zero_mask = actual_vals != 0
    if non_zero_mask.any():
        mape_val = float(
            np.mean(
                np.abs(
                    (actual_vals[non_zero_mask] - predicted_vals[non_zero_mask])
                    / actual_vals[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape_val = float("inf")

    # CV-RMSE
    mean_actual = float(np.mean(actual_vals))
    cv_rmse_val = (rmse_val / mean_actual) * 100 if mean_actual != 0 else float("inf")

    # R-squared
    ss_res = float(np.sum((actual_vals - predicted_vals) ** 2))
    ss_tot = float(np.sum((actual_vals - mean_actual) ** 2))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "rmse": round(rmse_val, 4),
        "mae": round(mae_val, 4),
        "mape": round(mape_val, 2),
        "cv_rmse": round(cv_rmse_val, 2),
        "r_squared": round(r_squared, 4),
    }


def _align_series(
    actual: TimeSeries, predicted: TimeSeries
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two time series to overlapping time range.

    Args:
        actual: Ground truth series
        predicted: Predicted series

    Returns:
        Tuple of aligned numpy arrays
    """
    # Get time indices
    actual_times = actual.time_index
    predicted_times = predicted.time_index

    # Find overlapping range
    start_time = max(actual_times.min(), predicted_times.min())
    end_time = min(actual_times.max(), predicted_times.max())

    # Slice both series to overlap
    actual_slice = actual.slice(start_time, end_time)
    predicted_slice = predicted.slice(start_time, end_time)

    # Convert to numpy
    actual_vals = actual_slice.values().flatten()
    predicted_vals = predicted_slice.values().flatten()

    # Ensure same length (take minimum)
    min_len = min(len(actual_vals), len(predicted_vals))
    actual_vals = actual_vals[:min_len]
    predicted_vals = predicted_vals[:min_len]

    return actual_vals, predicted_vals


def calculate_residual_analysis(
    actual: TimeSeries,
    predicted: TimeSeries,
    scaler=None,
) -> dict:
    """
    Calculate residual statistics for model diagnostics.

    Args:
        actual: Ground truth series
        predicted: Predicted series
        scaler: If provided, inverse transform first

    Returns:
        Dict with mean_residual, std_residual, autocorrelation_lag1
    """
    # Inverse transform if scaled
    if scaler is not None:
        try:
            actual = scaler.inverse_transform(actual)
            predicted = scaler.inverse_transform(predicted)
        except Exception:
            pass

    actual_vals, predicted_vals = _align_series(actual, predicted)

    if len(actual_vals) < 2:
        return {
            "mean_residual": None,
            "std_residual": None,
            "autocorrelation_lag1": None,
        }

    residuals = actual_vals - predicted_vals

    # Mean residual (should be ~0 for unbiased model)
    mean_residual = float(np.mean(residuals))

    # Std residual
    std_residual = float(np.std(residuals))

    # Autocorrelation at lag 1 (should be ~0 for good model)
    if len(residuals) > 1:
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        autocorrelation_lag1 = float(autocorr) if not np.isnan(autocorr) else 0.0
    else:
        autocorrelation_lag1 = 0.0

    return {
        "mean_residual": round(mean_residual, 4),
        "std_residual": round(std_residual, 4),
        "autocorrelation_lag1": round(autocorrelation_lag1, 4),
    }


def compare_to_validation(
    test_metrics: dict,
    validation_metrics: dict,
    threshold: float = 5.0,
) -> dict:
    """
    Compare test metrics to validation metrics.

    Args:
        test_metrics: Metrics from test set
        validation_metrics: Metrics from validation set
        threshold: CV-RMSE difference threshold for "similar" status

    Returns:
        Dict with cv_rmse_diff and performance_status
    """
    test_cv_rmse = test_metrics.get("cv_rmse")
    val_cv_rmse = validation_metrics.get("cv_rmse")

    if test_cv_rmse is None or val_cv_rmse is None:
        return {
            "cv_rmse_diff": None,
            "performance_status": "unknown",
        }

    cv_rmse_diff = test_cv_rmse - val_cv_rmse

    if abs(cv_rmse_diff) <= threshold:
        status = "similar"
    elif cv_rmse_diff > threshold:
        status = "degraded"
    else:
        status = "improved"

    return {
        "cv_rmse_diff": round(cv_rmse_diff, 2),
        "performance_status": status,
    }
