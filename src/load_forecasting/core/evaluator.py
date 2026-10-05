"""
Metrics calculation for forecasting models.

Computes:
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error
- MAPE: Mean Absolute Percentage Error (%)
- CV-RMSE: Coefficient of Variation of RMSE (%)
- R-squared: Coefficient of determination
- Peak MAPE (PMAPE): Error on daily peak magnitude (Li et al. 2025)
- Peak Timing Error (PTE): Error on daily peak hour (Li et al. 2025)
"""

import numpy as np
import pandas as pd
from darts import TimeSeries
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def _maybe_inverse(series: TimeSeries, scaler):
    """Inverse-transform a series if a scaler is given; return it unchanged on error."""
    if scaler is None:
        return series
    try:
        return scaler.inverse_transform(series)
    except Exception as e:
        logger.warning(f"Failed to inverse transform: {e}")
        return series


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
    actual = _maybe_inverse(actual, scaler)
    predicted = _maybe_inverse(predicted, scaler)

    # Align time series (they may have different lengths after historical_forecasts)
    actual_vals, predicted_vals = _align_series(actual, predicted)
    return metrics_from_arrays(actual_vals, predicted_vals)


def metrics_from_arrays(
    actual_vals: np.ndarray, predicted_vals: np.ndarray
) -> dict:
    """Compute the standard point metrics from already-aligned value arrays.

    Split out of :py:func:`calculate_metrics` so that callers holding several
    disjoint (actual, predicted) pairs — e.g. the per-season chunks of a
    seasonal train/validation split — can pool the residuals into a single set
    of metrics instead of averaging per-chunk metrics, which would weight short
    chunks equally with long ones.

    Both arrays must already be on the original (un-scaled) axis and aligned
    element-wise.
    """
    actual_vals = np.asarray(actual_vals, dtype=float)
    predicted_vals = np.asarray(predicted_vals, dtype=float)

    if len(actual_vals) == 0:
        logger.warning("No overlapping data points for metrics calculation")
        return {
            "rmse": None,
            "mae": None,
            "mape": None,
            "cv_rmse": None,
            "r_squared": None,
        }

    # Shared residual arrays — computed once, reused across all metrics
    residuals = actual_vals - predicted_vals
    sq_residuals = residuals ** 2

    # RMSE
    rmse_val = float(np.sqrt(sq_residuals.mean()))

    # MAE
    mae_val = float(np.abs(residuals).mean())

    # MAPE (handle zero values)
    non_zero_mask = actual_vals != 0
    if non_zero_mask.any():
        mape_val = float(
            np.mean(
                np.abs(residuals[non_zero_mask] / actual_vals[non_zero_mask])
            )
            * 100
        )
    else:
        mape_val = float("inf")

    # CV-RMSE
    mean_actual = float(actual_vals.mean())
    cv_rmse_val = (rmse_val / mean_actual) * 100 if mean_actual != 0 else float("inf")

    # R-squared
    ss_res = float(sq_residuals.sum())
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
    Align two time series by matching timestamps exactly.

    Uses slice_intersect so only timestamps present in BOTH series are
    compared.  This is essential when the predicted series has a different
    stride than the actual series (e.g. historical_forecasts with
    last_points_only=True returns one point per stride block while the actual
    series is hourly).

    Args:
        actual: Ground truth series
        predicted: Predicted series (may have coarser or offset time index)

    Returns:
        Tuple of aligned numpy arrays (matching timestamps only)
    """
    # Use Darts slice_intersect to keep only common timestamps
    try:
        actual_aligned = actual.slice_intersect(predicted)
        predicted_aligned = predicted.slice_intersect(actual)
    except Exception:
        # Fallback: time-range slice for edge cases (e.g. integer-indexed series)
        actual_times = actual.time_index
        predicted_times = predicted.time_index
        start_time = max(actual_times.min(), predicted_times.min())
        end_time = min(actual_times.max(), predicted_times.max())
        actual_aligned = actual.slice(start_time, end_time)
        predicted_aligned = predicted.slice(start_time, end_time)

    actual_vals = actual_aligned.values().flatten()
    predicted_vals = predicted_aligned.values().flatten()

    # Final safety truncation (should be equal length after slice_intersect)
    min_len = min(len(actual_vals), len(predicted_vals))
    return actual_vals[:min_len], predicted_vals[:min_len]


def calculate_peak_metrics(
    actual: TimeSeries,
    predicted: TimeSeries,
    peak_dates: list,
    scaler=None,
) -> dict:
    """
    Calculate peak load prediction metrics for specified peak days.

    Implements the Peak MAPE (PMAPE) and Peak Timing Error (PTE) metrics
    from Li et al. (2025) "A cross-dimensional analysis of data-driven
    short-term load forecasting methods with large-scale smart meter data"
    (Energy & Buildings, 344, Table 5 / Section 4.2).

    For each date in ``peak_dates`` the function:
      1. Slices actual and predicted to the calendar day (midnight-to-midnight).
      2. Finds the hour of the actual daily peak → ``t_actual``, value → ``L_actual``.
      3. Finds the hour of the predicted daily peak → ``t_predicted``, value → ``L_predicted``.
      4. Computes per-day PMAPE = |L_actual - L_predicted| / L_actual × 100.
      5. Computes per-day PTE   = |t_actual - t_predicted| in hours.

    Results are averaged across all days for which both series have coverage.

    Args:
        actual: Ground truth TimeSeries.
        predicted: Predicted TimeSeries.
        peak_dates: List of date strings ("YYYY-MM-DD") identifying peak days
            to evaluate.  Typically 3–4 consecutive winter or summer peak days
            as used in the paper.
        scaler: Optional Darts Scaler used to inverse-transform before computing
            metrics (both series are inverse-transformed together).

    Returns:
        Dict with keys:
            peak_mape               – mean PMAPE across evaluated days (%)
            peak_timing_error_hours – mean PTE across evaluated days (hours)
            n_peak_days_evaluated   – number of days with valid coverage
            n_peak_days_skipped     – days without sufficient data
            per_day                 – list of per-day dicts with detailed results
    """
    # Inverse transform if scaled
    actual = _maybe_inverse(actual, scaler)
    predicted = _maybe_inverse(predicted, scaler)

    # Convert both series to pandas Series indexed by timestamp for easy slicing.
    # Darts ≥0.27 removed pd_series()/pd_dataframe() in favour of to_dataframe().
    actual_pd = actual.to_dataframe().iloc[:, 0]
    predicted_pd = predicted.to_dataframe().iloc[:, 0]

    per_day_results: list[dict] = []
    pmape_values: list[float] = []
    pte_values: list[float] = []
    n_skipped = 0

    for date_str in peak_dates:
        try:
            day_start = pd.Timestamp(date_str)
            day_end = day_start + pd.Timedelta(hours=23)

            # Slice to this calendar day (inclusive on both ends)
            actual_day = actual_pd.loc[
                (actual_pd.index >= day_start) & (actual_pd.index <= day_end)
            ]
            predicted_day = predicted_pd.loc[
                (predicted_pd.index >= day_start) & (predicted_pd.index <= day_end)
            ]

            if actual_day.empty or predicted_day.empty:
                logger.debug(
                    "calculate_peak_metrics: no data for date %s — skipping",
                    date_str,
                )
                n_skipped += 1
                continue

            # Find peak hour in actual series for this day
            actual_peak_idx = actual_day.idxmax()
            actual_peak_value = float(actual_day[actual_peak_idx])
            actual_peak_hour = actual_peak_idx.hour

            # Find peak hour in predicted series for this day
            predicted_peak_idx = predicted_day.idxmax()
            predicted_peak_value = float(predicted_day[predicted_peak_idx])
            predicted_peak_hour = predicted_peak_idx.hour

            if actual_peak_value == 0:
                logger.warning(
                    "calculate_peak_metrics: actual peak is zero on %s — skipping PMAPE",
                    date_str,
                )
                n_skipped += 1
                continue

            day_pmape = abs(actual_peak_value - predicted_peak_value) / actual_peak_value * 100
            day_pte = abs(actual_peak_hour - predicted_peak_hour)

            pmape_values.append(day_pmape)
            pte_values.append(day_pte)

            per_day_results.append({
                "date": date_str,
                "actual_peak_value": round(actual_peak_value, 4),
                "predicted_peak_value": round(predicted_peak_value, 4),
                "actual_peak_hour": actual_peak_hour,
                "predicted_peak_hour": predicted_peak_hour,
                "peak_magnitude_error_pct": round(day_pmape, 2),
                "peak_timing_error_hours": round(float(day_pte), 2),
            })

        except Exception as e:
            logger.warning(
                "calculate_peak_metrics: error processing date %s: %s",
                date_str, e,
            )
            n_skipped += 1

    if not pmape_values:
        logger.warning(
            "calculate_peak_metrics: no valid peak days found in %s",
            peak_dates,
        )
        return {
            "peak_mape": None,
            "peak_timing_error_hours": None,
            "n_peak_days_evaluated": 0,
            "n_peak_days_skipped": n_skipped,
            "per_day": [],
        }

    return {
        "peak_mape": round(float(np.mean(pmape_values)), 2),
        "peak_timing_error_hours": round(float(np.mean(pte_values)), 2),
        "n_peak_days_evaluated": len(pmape_values),
        "n_peak_days_skipped": n_skipped,
        "per_day": per_day_results,
    }


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
    actual = _maybe_inverse(actual, scaler)
    predicted = _maybe_inverse(predicted, scaler)

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


def calculate_horizon_metrics(
    predictions_raw: list,
    actual: "TimeSeries",
    scaler=None,
) -> list:
    """
    Compute h-step-ahead error metrics across all rolling forecast windows.

    For each horizon step h (1-indexed, 1 = one-step-ahead), this function
    collects all predicted values at position h across every window produced by
    ``historical_forecasts(last_points_only=False)``, aligns them with ground
    truth at the corresponding timestamps, and computes RMSE, MAE, and MAPE.

    This follows the rolling/recursive forecast evaluation convention: the model
    re-forecasts at each stride, and we evaluate the h-step-ahead prediction
    against the actual value that occurs h steps after the forecast origin.

    Args:
        predictions_raw: List of ``TimeSeries`` objects, one per forecast window,
            as returned by ``historical_forecasts(last_points_only=False)``.
            Each window has length ``horizon_steps``.
        actual: Full ground-truth ``TimeSeries`` (same time axis as the
            original data, starting from the backtest start index).
        scaler: Optional Darts ``Scaler``.  When provided both ``predictions_raw``
            elements and ``actual`` are inverse-transformed before metrics are
            computed.

    Returns:
        List of dicts, one per horizon step, sorted by h::

            [
                {"h": 1, "rmse": float, "mae": float, "mape": float},
                {"h": 2, "rmse": float, "mae": float, "mape": float},
                ...
            ]

        Steps for which fewer than 2 aligned pairs exist return ``None``
        for all metrics.
    """
    if not predictions_raw:
        return []

    # Inverse-transform actual once (expensive; predictions below are per-step slices)
    actual_inv = actual
    if scaler is not None:
        try:
            actual_inv = scaler.inverse_transform(actual)
        except Exception as e:
            logger.warning("calculate_horizon_metrics: failed to inverse-transform actual: %s", e)

    # Convert actual to pandas for fast timestamp lookup
    actual_pd = actual_inv.to_dataframe().iloc[:, 0]

    horizon_steps = len(predictions_raw[0]) if predictions_raw else 0

    # Inverse-transform each window once up front (O(W) scaler calls, not O(W×H))
    windows_inv: list = []
    for window in predictions_raw:
        if scaler is not None:
            try:
                windows_inv.append(scaler.inverse_transform(window))
            except Exception:
                windows_inv.append(window)
        else:
            windows_inv.append(window)

    results: list = []

    for h in range(1, horizon_steps + 1):
        h_actuals: list = []
        h_predicted: list = []

        for window in windows_inv:
            if len(window) < h:
                continue
            # Step h is 0-indexed as h-1
            step_ts = window.time_index[h - 1]
            step_pred = float(window.univariate_values()[h - 1])

            # Look up actual at this timestamp
            if step_ts in actual_pd.index:
                step_actual = float(actual_pd.loc[step_ts])
                h_actuals.append(step_actual)
                h_predicted.append(step_pred)

        if len(h_actuals) < 2:
            results.append({"h": h, "rmse": None, "mae": None, "mape": None})
            continue

        a = np.array(h_actuals)
        p = np.array(h_predicted)
        residuals = a - p

        rmse_h = float(np.sqrt((residuals ** 2).mean()))
        mae_h = float(np.abs(residuals).mean())

        non_zero = a != 0
        if non_zero.any():
            mape_h = float(np.mean(np.abs(residuals[non_zero] / a[non_zero])) * 100)
        else:
            mape_h = None

        results.append({
            "h": h,
            "rmse": round(rmse_h, 4),
            "mae": round(mae_h, 4),
            "mape": round(mape_h, 2) if mape_h is not None else None,
        })

    return results


def calculate_horizon_coverage(
    stochastic_windows: list,
    actual: "TimeSeries",
    lower_q: float,
    upper_q: float,
    scaler=None,
) -> list:
    """
    Compute prediction-interval coverage and width per h-step-ahead position.

    Pooled coverage across a multi-step horizon is misleading: a 1-step-ahead
    band and a 24-step-ahead band are different objects, and a model that is
    over-covered near the origin and under-covered far out can average to a
    perfect-looking number.  This decomposes coverage by horizon step so that
    degradation with lead time is visible.

    Mirrors ``calculate_horizon_metrics``: for each step h it gathers the
    predicted interval at position h across every rolling window and aligns it
    with ground truth at the corresponding timestamp.

    Args:
        stochastic_windows: List of *stochastic* ``TimeSeries`` (n_samples > 1),
            one per forecast window, as returned by
            ``historical_forecasts(last_points_only=False, num_samples=N)``.
        actual: Full ground-truth ``TimeSeries`` from the backtest start index.
        lower_q: Lower quantile level of the band (e.g. 0.1).
        upper_q: Upper quantile level of the band (e.g. 0.9).
        scaler: Optional Darts ``Scaler`` applied to both sides.

    Returns:
        List of dicts sorted by h::

            [{"h": 1, "coverage": 0.91, "mean_interval_width": 812.4, "n": 340},
             ...]

        Steps with fewer than 2 aligned pairs return ``None`` metrics and the
        observed ``n``.
    """
    if not stochastic_windows:
        return []

    actual_inv = _maybe_inverse(actual, scaler)
    try:
        actual_pd = actual_inv.to_dataframe().iloc[:, 0]
    except Exception as e:
        logger.warning("calculate_horizon_coverage: cannot read actual series: %s", e)
        return []

    # Collapse each stochastic window to its lower/upper quantile pair once,
    # then inverse-transform — O(W) scaler calls rather than O(W x H).
    bands: list = []
    for window in stochastic_windows:
        try:
            lo = _maybe_inverse(window.quantile(lower_q), scaler)
            hi = _maybe_inverse(window.quantile(upper_q), scaler)
        except Exception:
            continue
        bands.append((lo, hi))

    if not bands:
        return []

    horizon_steps = max(len(lo) for lo, _ in bands)
    results: list = []

    for h in range(1, horizon_steps + 1):
        inside = 0
        widths: list = []
        n = 0
        for lo, hi in bands:
            if len(lo) < h or len(hi) < h:
                continue
            step_ts = lo.time_index[h - 1]
            if step_ts not in actual_pd.index:
                continue
            lo_v = float(lo.univariate_values()[h - 1])
            hi_v = float(hi.univariate_values()[h - 1])
            a_v = float(actual_pd.loc[step_ts])
            n += 1
            widths.append(hi_v - lo_v)
            if lo_v <= a_v <= hi_v:
                inside += 1

        if n < 2:
            results.append(
                {"h": h, "coverage": None, "mean_interval_width": None, "n": n}
            )
            continue

        results.append({
            "h": h,
            "coverage": round(inside / n, 4),
            "mean_interval_width": round(float(np.mean(widths)), 4),
            "n": n,
        })

    return results


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


def calculate_probabilistic_metrics(
    actual: TimeSeries,
    quantile_forecasts: dict,
    scaler=None,
) -> dict:
    """Compute probabilistic (interval) forecast metrics.

    Scores a set of quantile forecasts against the realised values using
    metrics standard in probabilistic load forecasting:

      * ``pinball_loss`` — mean pinball (quantile) loss averaged over all
        supplied quantile levels and time steps.  Lower is better; this is the
        proper scoring rule that quantile models are trained on.
      * ``coverage`` — for the widest symmetric central interval that can be
        formed from the supplied quantiles (e.g. P10–P90 → nominal 80%), the
        empirical fraction of actuals that fall inside it.  Well-calibrated
        forecasts have coverage close to the nominal level.
      * ``nominal_coverage`` — the nominal level of that interval (e.g. 0.8).
      * ``mean_interval_width`` — average width of that central interval on the
        original load scale (a proxy for forecast sharpness).
      * ``per_quantile_pinball`` — pinball loss broken out per quantile level.

    Args:
        actual: Ground-truth ``TimeSeries`` (scaled or original).
        quantile_forecasts: Mapping ``{quantile_level: TimeSeries}`` where each
            series is the forecast for that quantile.  Levels are floats in
            (0, 1).  Series may be scaled; ``scaler`` inverse-transforms them.
        scaler: Optional scaler applied (inverse) to both actuals and every
            quantile series before scoring, so metrics are on the original
            load scale.

    Returns:
        Dict of probabilistic metrics (see above).  Returns a dict of ``None``
        values when there are no overlapping points.
    """
    null_result = {
        "pinball_loss": None,
        "coverage": None,
        "nominal_coverage": None,
        "mean_interval_width": None,
        "per_quantile_pinball": {},
    }
    if not quantile_forecasts:
        return null_result

    actual_inv = _maybe_inverse(actual, scaler)

    # Align every quantile series to the actuals and stack into a matrix.
    levels = sorted(quantile_forecasts.keys())
    aligned_actual = None
    q_arrays: dict = {}
    for q in levels:
        q_series = _maybe_inverse(quantile_forecasts[q], scaler)
        a_vals, q_vals = _align_series(actual_inv, q_series)
        if len(a_vals) == 0:
            continue
        # Keep the shortest common actual vector so all quantiles line up.
        if aligned_actual is None or len(a_vals) < len(aligned_actual):
            aligned_actual = a_vals
        q_arrays[q] = q_vals

    if aligned_actual is None or not q_arrays:
        return null_result

    n = len(aligned_actual)
    a = aligned_actual[:n]

    # Pinball loss per quantile, then averaged.
    per_q_pinball: dict = {}
    total = 0.0
    for q, q_vals in q_arrays.items():
        qv = q_vals[:n]
        diff = a - qv
        loss = np.where(diff >= 0, q * diff, (q - 1) * diff)
        pl = float(np.mean(loss))
        per_q_pinball[str(q)] = round(pl, 4)
        total += pl
    pinball_loss = total / len(q_arrays)

    # Widest symmetric central interval from the supplied quantiles.
    lower_q = min(q_arrays.keys())
    upper_q = max(q_arrays.keys())
    coverage_val = None
    nominal = None
    mean_width = None
    if upper_q > lower_q:
        lo = q_arrays[lower_q][:n]
        hi = q_arrays[upper_q][:n]
        inside = (a >= lo) & (a <= hi)
        coverage_val = float(np.mean(inside))
        nominal = float(upper_q - lower_q)
        mean_width = float(np.mean(hi - lo))

    return {
        "pinball_loss": round(pinball_loss, 4),
        "coverage": round(coverage_val, 4) if coverage_val is not None else None,
        "nominal_coverage": round(nominal, 4) if nominal is not None else None,
        "mean_interval_width": round(mean_width, 4) if mean_width is not None else None,
        "per_quantile_pinball": per_q_pinball,
    }
