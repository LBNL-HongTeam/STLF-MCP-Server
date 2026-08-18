"""
MCP tool: backtest_model — rolling-window backtest of a trained model.

Also hosts the shared rolling-backtest pipeline (``_run_backtest_core``) used
by both ``backtest_model`` and ``generate_backtest_report`` (in reports.py).
"""

from typing import Optional
import logging

import pandas as pd

from ..core.data_loader import DataLoadError
from ..core.trainer import generate_predictions
from ..core.evaluator import (
    calculate_residual_analysis,
    compare_to_validation,
    calculate_metrics,
    calculate_horizon_metrics,
    calculate_horizon_coverage,
    calculate_probabilistic_metrics,
    _maybe_inverse,
)
from ..core.frequency_utils import hours_to_steps
from ._common import (
    create_success_response,
    create_error_response,
    _ModelLoadError,
    _load_model_context,
    _collect_warnings,
    _peak_metrics_safe,
    _merge_peak_headline,
    _check_frequency_mismatch,
    _check_training_data_overlap,
    _check_short_test_data,
    _check_interpolation,
    _load_and_scale_series,
    _records_from_series,
)

logger = logging.getLogger(__name__)


def _band_levels(quantiles: list) -> Optional[tuple]:
    """Return the outermost (lower, upper) quantile pair straddling the median.

    None when fewer than two non-median levels exist (nothing to shade).
    """
    qs = sorted(float(q) for q in (quantiles or []))
    lows = [q for q in qs if q < 0.5]
    highs = [q for q in qs if q > 0.5]
    if not lows or not highs:
        return None
    return lows[0], highs[-1]


def _run_probabilistic_backtest(
    *,
    model,
    metadata: dict,
    model_type: str,
    target_series,
    past_cov,
    future_cov,
    lookback_steps: int,
    horizon_steps: int,
    stride_steps: int,
    start_idx: int,
    target_scaler,
    num_samples: int,
) -> dict:
    """Run a stochastic rolling pass and derive interval quality metrics.

    Returns ``{}`` for point models, for models that cannot sample, or when the
    stochastic pass fails — the deterministic backtest is unaffected either way.

    The returned dict carries pooled interval metrics plus the per-horizon-step
    coverage decomposition and the per-window quantile bands used by the HTML
    report's playback chart.
    """
    from darts import concatenate as darts_concatenate

    config = metadata.get("config", {}) or {}
    if not config.get("probabilistic"):
        return {}
    if not getattr(model, "supports_probabilistic_prediction", False):
        return {}

    quantiles = sorted({float(q) for q in (config.get("quantiles") or [])} | {0.5})
    band = _band_levels(quantiles)
    n_samples = max(1, int(num_samples))

    try:
        stochastic_raw = generate_predictions(
            model, target_series, past_cov, lookback_steps, horizon_steps,
            future_covariates=future_cov, model_type=model_type,
            stride=stride_steps, start=start_idx, last_points_only=False,
            num_samples=n_samples,
        )
    except Exception as e:
        logger.warning("Probabilistic backtest pass failed: %s", e)
        return {}

    stochastic_windows = (
        stochastic_raw if isinstance(stochastic_raw, list) else [stochastic_raw]
    )
    stochastic_windows = [
        w for w in stochastic_windows if getattr(w, "n_samples", 1) > 1
    ]
    if not stochastic_windows:
        logger.warning("Probabilistic backtest produced no stochastic samples")
        return {}

    out: dict = {
        "quantiles": quantiles,
        "num_samples": n_samples,
        "n_windows": len(stochastic_windows),
    }

    # Pooled interval metrics over the whole backtest span.
    try:
        concat = darts_concatenate(stochastic_windows, ignore_time_axis=True)
        quantile_forecasts = {q: concat.quantile(q) for q in quantiles}
        pooled = calculate_probabilistic_metrics(
            target_series[start_idx:], quantile_forecasts, scaler=target_scaler
        )
        out.update(pooled)
    except Exception as e:
        logger.warning("Pooled probabilistic backtest metrics failed: %s", e)

    if band is None:
        return out

    lower_q, upper_q = band
    out["band"] = {
        "lower": lower_q,
        "upper": upper_q,
        "nominal": round(upper_q - lower_q, 4),
    }

    # The headline addition: coverage as a function of lead time.
    try:
        out["horizon_coverage"] = calculate_horizon_coverage(
            stochastic_windows,
            target_series[start_idx:],
            lower_q,
            upper_q,
            scaler=target_scaler,
        )
    except Exception as e:
        logger.warning("Horizon coverage failed: %s", e)
        out["horizon_coverage"] = []

    # Per-window bands for the report's forecast-playback overlay.
    windows_bands: list = []
    for w in stochastic_windows:
        try:
            lo = _maybe_inverse(w.quantile(lower_q), target_scaler)
            hi = _maybe_inverse(w.quantile(upper_q), target_scaler)
            windows_bands.append(
                [
                    {
                        "t": ts.isoformat(),
                        "lower": float(lv),
                        "upper": float(hv),
                    }
                    for ts, lv, hv in zip(
                        lo.time_index,
                        lo.univariate_values(),
                        hi.univariate_values(),
                    )
                ]
            )
        except Exception:
            windows_bands.append([])
    out["window_bands"] = windows_bands

    return out


def _run_backtest_core(
    model_id: str,
    csv_path: str,
    column_mapping: Optional[dict],
    stride_hours: Optional[int],
    start_fraction: float,
    include_residual_analysis: bool,
    peak_dates: Optional[list],
    *,
    reconstruct_windows: bool = False,
    num_samples: int = 200,
) -> dict:
    """Shared rolling-backtest pipeline for backtest_model + generate_backtest_report.

    Loads the model, validates params, runs rolling predictions and computes
    metrics/summary. When ``reconstruct_windows`` is True the result also
    includes per-window series, inverse-transformed series, horizon metrics and
    a raw input preview frame (needed by the HTML report).

    Raises _ModelLoadError (carrying an error-response dict) for any early exit.
    Returns a dict of all computed artifacts on success.
    """
    from darts import concatenate as darts_concatenate

    # Validate parameters (fail fast before loading model/data)
    if stride_hours is not None and not 1 <= stride_hours <= 96:
        raise _ModelLoadError(create_error_response("stride_hours must be between 1 and 96"))
    if not 0.1 <= start_fraction <= 0.5:
        raise _ModelLoadError(
            create_error_response("start_fraction must be between 0.1 and 0.5")
        )

    column_mapping_source = "provided" if column_mapping else "from_model_metadata"
    ctx = _load_model_context(model_id, column_mapping)
    model = ctx["model"]
    metadata = ctx["metadata"]
    scalers = ctx["scalers"]
    model_type = ctx["model_type"]
    column_mapping = ctx["column_mapping"]
    frequency = ctx["frequency"]
    horizon_hours_cfg = ctx["horizon_hours"]
    lookback_steps = ctx["lookback_steps"]
    horizon_steps = ctx["horizon_steps"]

    resolved_mapping = column_mapping or {}
    try:
        loader, target_series, past_cov, future_cov = _load_and_scale_series(
            csv_path=csv_path,
            column_mapping=resolved_mapping or None,
            scalers=scalers,
            frequency=frequency,
            metadata=metadata,
        )
    except DataLoadError as e:
        raise _ModelLoadError(create_error_response(f"Failed to load data: {e}"))

    freq_err = _check_frequency_mismatch(loader, frequency)
    if freq_err:
        raise _ModelLoadError(create_error_response(freq_err))

    ml_warnings = _collect_warnings(
        lambda: _check_training_data_overlap(loader, metadata),
        lambda: _check_short_test_data(loader, lookback_steps, frequency),
        lambda: _check_interpolation(loader),
    )
    resolved_mapping = loader.column_mapping or resolved_mapping

    stride_steps = (
        hours_to_steps(stride_hours, frequency) if stride_hours is not None else horizon_steps
    )
    effective_stride_hours = stride_hours if stride_hours is not None else horizon_hours_cfg

    start_idx = max(lookback_steps, int(start_fraction * len(target_series)))
    if start_idx + horizon_steps > len(target_series):
        raise _ModelLoadError(
            create_error_response(
                f"Not enough data for backtest: series has {len(target_series)} steps but "
                f"start_idx={start_idx} + horizon={horizon_steps} exceeds series length. "
                "Reduce start_fraction or use a longer CSV."
            )
        )

    # Rolling predictions (list of per-window TimeSeries)
    try:
        predictions_raw = generate_predictions(
            model, target_series, past_cov, lookback_steps, horizon_steps,
            future_covariates=future_cov, model_type=model_type,
            stride=stride_steps, start=start_idx, last_points_only=False,
        )
    except Exception as e:
        raise _ModelLoadError(create_error_response(f"Rolling backtest failed: {e}"))

    # Reconstruct per-window list + concatenated series. Local models return a
    # flat concatenated series; ML models return a list of windows.
    if isinstance(predictions_raw, list):
        windows_list = predictions_raw
        predictions_concat = darts_concatenate(predictions_raw, ignore_time_axis=True)
    else:
        predictions_concat = predictions_raw
        windows_list = []
        if reconstruct_windows:
            total_steps = len(predictions_concat)
            for w_start in range(0, total_steps - horizon_steps + 1, stride_steps):
                try:
                    windows_list.append(predictions_concat[w_start:w_start + horizon_steps])
                except Exception:
                    break

    target_scaler = loader.target_scaler
    backtest_metrics = calculate_metrics(
        target_series[start_idx:], predictions_concat, scaler=target_scaler
    )
    validation_metrics = metadata.get("metrics", {}).get("validation", {})
    comparison = compare_to_validation(backtest_metrics, validation_metrics)

    n_windows = (
        len(windows_list)
        if windows_list
        else (len(predictions_concat) // horizon_steps if horizon_steps > 0 else 0)
    )
    backtest_summary = {
        "n_windows": n_windows,
        "stride_hours": effective_stride_hours,
        "stride_steps": stride_steps,
        "start_fraction": start_fraction,
        "backtest_start_date": (
            loader.df.index[start_idx].isoformat() if start_idx < len(loader.df) else None
        ),
        "backtest_end_date": loader.df.index.max().isoformat(),
        "total_samples": len(loader.df),
        "column_mapping_source": column_mapping_source,
    }

    # ---- probabilistic (interval) backtest -------------------------------
    # A second, stochastic rolling pass for quantile-trained models.  Point
    # models skip this entirely, so they pay nothing.  The per-window
    # structure is what makes coverage-by-horizon-step possible — that
    # decomposition is unavailable from a single pooled test split.
    prob_backtest = _run_probabilistic_backtest(
        model=model,
        metadata=metadata,
        model_type=model_type,
        target_series=target_series,
        past_cov=past_cov,
        future_cov=future_cov,
        lookback_steps=lookback_steps,
        horizon_steps=horizon_steps,
        stride_steps=stride_steps,
        start_idx=start_idx,
        target_scaler=target_scaler,
        num_samples=num_samples,
    )

    result = {
        "model": model,
        "metadata": metadata,
        "model_type": model_type,
        "loader": loader,
        "target_series": target_series,
        "start_idx": start_idx,
        "predictions_concat": predictions_concat,
        "windows_list": windows_list,
        "target_scaler": target_scaler,
        "backtest_metrics": backtest_metrics,
        "comparison": comparison,
        "backtest_summary": backtest_summary,
        "n_windows": n_windows,
        "ml_warnings": ml_warnings,
        "resolved_mapping": resolved_mapping,
        "frequency": frequency,
        "probabilistic": prob_backtest,
    }

    if not reconstruct_windows:
        return result

    # Extra artifacts for the HTML report
    if target_scaler is not None:
        pred_inv = target_scaler.inverse_transform(predictions_concat)
        actual_inv = target_scaler.inverse_transform(target_series[start_idx:])
        windows_inv = []
        for w in windows_list:
            try:
                windows_inv.append(target_scaler.inverse_transform(w))
            except Exception:
                windows_inv.append(w)
    else:
        pred_inv = predictions_concat
        actual_inv = target_series[start_idx:]
        windows_inv = list(windows_list)

    horizon_metrics: list = []
    if windows_list:
        try:
            horizon_metrics = calculate_horizon_metrics(
                predictions_raw=windows_list,
                actual=target_series[start_idx:],
                scaler=target_scaler,
            )
        except Exception as e:
            logger.warning("calculate_horizon_metrics failed: %s", e)

    result.update(
        pred_inv=pred_inv,
        actual_inv=actual_inv,
        windows_inv=windows_inv,
        horizon_metrics=horizon_metrics,
        predictions_flat=_records_from_series(pred_inv, actual_inv),
    )
    return result


def backtest_model(
    model_id: str,
    csv_path: str,
    column_mapping: Optional[dict] = None,
    stride_hours: Optional[int] = None,
    start_fraction: float = 0.2,
    return_predictions: bool = True,
    output_csv_path: Optional[str] = None,
    include_residual_analysis: bool = False,
    peak_dates: Optional[list] = None,
    num_samples: int = 200,
) -> dict:
    """
    Run a rolling-window backtest of a trained model on historical data.

    Similar to evaluate_forecast_model but with configurable stride and
    start position, making it easy to assess model performance across many
    historical windows rather than a single test split.

    Args:
        model_id: ID of a trained model (from train_forecast_model).
        csv_path: Path to CSV with historical data.
        column_mapping: Column role mapping. Uses model metadata if not provided.
        stride_hours: Step size between forecast windows in hours. Defaults to
            the model's training horizon (non-overlapping windows). Smaller
            values produce overlapping windows and more prediction points.
        start_fraction: Fraction of the series to skip before starting the
            rolling forecast.  Ensures the model has enough history to warm up.
            Must be between 0.1 and 0.5.  Default: 0.2.
        return_predictions: If True, include per-step predictions in the
            response.  Can be large for long series.
        output_csv_path: Optional path to write predictions as CSV.
        include_residual_analysis: If True, include residual statistics.
        num_samples: Monte-Carlo sample count used when the model was trained
            with probabilistic=True.  Drives a second, stochastic rolling pass
            that yields ``probabilistic_metrics`` — pooled pinball/coverage/
            width plus ``horizon_coverage``, the per-h-step-ahead coverage
            decomposition.  Ignored for point models (default 200).

    Returns:
        Dict with success, model_id, model_type, backtest_metrics,
        comparison_to_validation, backtest_summary, and optionally
        predictions, residual_analysis, peak_metrics, probabilistic_metrics,
        output_csv_path.  peak_metrics is only present when peak_dates is
        provided; probabilistic_metrics only for quantile-trained models.
    """
    try:
        try:
            core = _run_backtest_core(
                model_id=model_id,
                csv_path=csv_path,
                column_mapping=column_mapping,
                stride_hours=stride_hours,
                start_fraction=start_fraction,
                include_residual_analysis=include_residual_analysis,
                peak_dates=peak_dates,
                num_samples=num_samples,
            )
        except _ModelLoadError as e:
            return e.response

        target_series = core["target_series"]
        start_idx = core["start_idx"]
        predictions = core["predictions_concat"]
        target_scaler = core["target_scaler"]

        response_data = {
            "model_id": model_id,
            "model_type": core["model_type"],
            "backtest_metrics": core["backtest_metrics"],
            "comparison_to_validation": core["comparison"],
            "backtest_summary": core["backtest_summary"],
        }

        if return_predictions:
            if target_scaler is not None:
                pred_original = target_scaler.inverse_transform(predictions)
                actual_original = target_scaler.inverse_transform(target_series[start_idx:])
            else:
                pred_original = predictions
                actual_original = target_series[start_idx:]
            response_data["predictions"] = _records_from_series(
                pred_original, actual_original
            )

        if include_residual_analysis:
            response_data["residual_analysis"] = calculate_residual_analysis(
                target_series[start_idx:], predictions, scaler=target_scaler
            )

        peak_metrics = _peak_metrics_safe(
            target_series[start_idx:], predictions, peak_dates, target_scaler
        )
        if peak_metrics is not None:
            response_data["peak_metrics"] = peak_metrics
            # Surface the peak headline (peak_mape, peak_timing_error_hours)
            # alongside the standard metrics; full detail stays in peak_metrics.
            _merge_peak_headline(response_data["backtest_metrics"], peak_metrics)

        prob = core.get("probabilistic") or {}
        if prob:
            # Drop the per-window band payload — it exists for the HTML report
            # and would bloat the tool response for long backtests.
            response_data["probabilistic_metrics"] = {
                k: v for k, v in prob.items() if k != "window_bands"
            }

        if output_csv_path and return_predictions:
            pd.DataFrame(response_data["predictions"]).to_csv(output_csv_path, index=False)
            response_data["output_csv_path"] = output_csv_path

        if core["ml_warnings"]:
            response_data["ml_warnings"] = core["ml_warnings"]

        return create_success_response(**response_data)

    except Exception as e:
        logger.exception("backtest_model failed")
        return create_error_response(f"Backtest failed: {str(e)}")
