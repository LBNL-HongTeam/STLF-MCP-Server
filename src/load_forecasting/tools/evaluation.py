"""
MCP tool: evaluate_forecast_model — evaluate a trained model on held-out data.
"""

from typing import Optional
import logging

import pandas as pd

from ..core.data_loader import ForecastingDataLoader, DataLoadError
from ..core.trainer import generate_predictions
from ..core.evaluator import (
    calculate_residual_analysis,
    compare_to_validation,
    calculate_metrics,
    calculate_probabilistic_metrics,
    _maybe_inverse,
)
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
)

logger = logging.getLogger(__name__)


def _attach_quantiles_to_predictions(
    pred_list: list[dict],
    quantile_forecasts: dict,
    scaler=None,
) -> None:
    """Merge quantile forecast values into prediction rows, in place.

    Each row gains a ``"q<level>"`` key (e.g. ``"q0.1"``) matching the
    ``generate_forecast`` convention.  Values are inverse-transformed back to
    original units so they are directly comparable with ``actual`` /
    ``predicted``.

    Rows whose timestamp has no corresponding quantile value are left
    untouched, so a partial overlap degrades to a partially-banded chart
    rather than an error.

    Args:
        pred_list: Prediction rows from evaluate_forecast_model; each has a
            ``"timestamp"`` ISO string.  Mutated in place.
        quantile_forecasts: Mapping of quantile level -> TimeSeries, as
            produced by ``stochastic.quantile(q)``.
        scaler: Optional target scaler used during training.
    """
    for q, series in quantile_forecasts.items():
        try:
            s = _maybe_inverse(series, scaler)
            values = s.values().flatten()
            by_ts = {
                ts.isoformat(): float(v) for ts, v in zip(s.time_index, values)
            }
        except Exception as e:
            logger.warning("Failed to attach quantile %s to predictions: %s", q, e)
            continue

        key = f"q{q}"
        for row in pred_list:
            v = by_ts.get(row.get("timestamp"))
            if v is not None:
                row[key] = v


def evaluate_forecast_model(
    model_id: str,
    csv_path: str,
    column_mapping: Optional[dict] = None,
    return_predictions: bool = True,
    output_csv_path: Optional[str] = None,
    include_residual_analysis: bool = False,
    peak_dates: Optional[list] = None,
    num_samples: int = 200,
) -> dict:
    """
    Evaluate a trained model on test data.

    Args:
        model_id: ID of trained model
        csv_path: Path to test data CSV
        column_mapping: Column mapping (uses training mapping if not provided)
        return_predictions: Include predictions in output
        output_csv_path: Optional path to save predictions CSV
        include_residual_analysis: Include residual statistics
        peak_dates: Optional list of date strings ("YYYY-MM-DD") identifying
            peak demand days to evaluate with PMAPE and PTE metrics following
            Li et al. (2025), Table 5.  Example:
            ["2023-02-22", "2023-02-23", "2023-02-24", "2023-02-25"]
            for the winter peak window used in the paper.
            When provided, a ``peak_metrics`` key is added to the response.
        num_samples: Monte-Carlo sample count used when the model was trained
            with probabilistic=True.  Drives the pinball-loss / coverage /
            interval-width metrics reported under ``probabilistic_metrics``.
            Ignored for point models (default 200).

    Returns:
        Dict with test metrics and predictions.  For probabilistic models a
        ``probabilistic_metrics`` block (pinball loss, empirical coverage,
        mean interval width, per-quantile pinball) is included.
    """
    try:
        column_mapping_source = "provided" if column_mapping else "from_model_metadata"
        # Load model + config context from registry
        try:
            ctx = _load_model_context(model_id, column_mapping)
        except _ModelLoadError as e:
            return e.response
        model = ctx["model"]
        metadata = ctx["metadata"]
        scalers = ctx["scalers"]
        column_mapping = ctx["column_mapping"]
        config = ctx["config"]
        frequency = ctx["frequency"]
        lookback = ctx["lookback_hours"]
        horizon = ctx["horizon_hours"]
        lookback_steps = ctx["lookback_steps"]
        horizon_steps = ctx["horizon_steps"]

        # Load test data
        try:
            loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=column_mapping,
                frequency=frequency,
            )
        except DataLoadError as e:
            return create_error_response(str(e))

        # Hard block: frequency mismatch between model and test CSV
        freq_err = _check_frequency_mismatch(loader, frequency)
        if freq_err:
            return create_error_response(freq_err)

        # Use scalers from training
        if "target_scaler" in scalers:
            loader.target_scaler = scalers["target_scaler"]
        if "covariate_scaler" in scalers:
            loader.covariate_scaler = scalers["covariate_scaler"]
        if "future_covariate_scaler" in scalers:
            loader.future_covariate_scaler = scalers["future_covariate_scaler"]

        # Convert to Darts TimeSeries (use existing scalers)
        test_series, test_covariates, test_future_covariates = (
            loader.to_darts_series(fit_scalers=False)
        )

        # Collect soft ML warnings
        ml_warnings = _collect_warnings(
            lambda: _check_training_data_overlap(loader, metadata),
            lambda: _check_short_test_data(loader, lookback_steps, frequency),
            lambda: _check_interpolation(loader),
        )
        if config.get("tuned"):
            ml_warnings.append(
                "This model was hyperparameter-tuned. If evaluating on the same CSV used "
                "for tuning, reported metrics are optimistic (the validation split was used "
                "to select hyperparameters). Use a fully held-out CSV for unbiased evaluation."
            )

        # Generate predictions.
        # Use last_points_only=False so the full forecast trajectory is
        # available for every horizon step — required for peak-day evaluation
        # where peak_dates may fall on any day inside a multi-step window.
        # Overall metrics are computed after aligning actual vs predicted on
        # their shared time index, so they are unaffected by this choice.
        from darts import concatenate as darts_concatenate
        predictions_raw = generate_predictions(
            model, test_series, test_covariates, lookback_steps, horizon_steps,
            future_covariates=test_future_covariates,
            last_points_only=False,
        )
        # last_points_only=False returns a list of TimeSeries (one per window).
        # Concatenate into a single series, keeping only the first occurrence
        # of any duplicated timestamps (windows overlap when stride < horizon).
        if isinstance(predictions_raw, list):
            predictions = darts_concatenate(predictions_raw, ignore_time_axis=False)
        else:
            predictions = predictions_raw

        # Calculate metrics
        test_metrics = calculate_metrics(
            test_series[lookback_steps:],
            predictions,
            scaler=loader.target_scaler,
        )

        # Compare to validation metrics
        validation_metrics = metadata.get("metrics", {}).get("validation", {})
        comparison = compare_to_validation(test_metrics, validation_metrics)

        # Test summary
        test_summary = {
            "test_samples": len(loader.df),
            "start_date": loader.df.index.min().isoformat(),
            "end_date": loader.df.index.max().isoformat(),
            "column_mapping_source": column_mapping_source,
        }

        # Build response
        response_data = {
            "model_id": model_id,
            "model_type": metadata.get("model_type"),
            "test_metrics": test_metrics,
            "comparison_to_validation": comparison,
            "test_summary": test_summary,
        }

        # Include predictions if requested
        if return_predictions:
            # Inverse transform predictions for output
            if loader.target_scaler:
                pred_original = loader.target_scaler.inverse_transform(predictions)
                actual_original = loader.target_scaler.inverse_transform(
                    test_series[lookback_steps:]
                )
            else:
                pred_original = predictions
                actual_original = test_series[lookback_steps:]

            # Build predictions list — vectorised to avoid per-row Python overhead
            pred_times = pred_original.time_index
            pred_values = pred_original.values().flatten()
            actual_values = actual_original.values().flatten()

            min_len = min(len(pred_times), len(pred_values), len(actual_values))
            pred_times_s = pred_times[:min_len]
            pred_values_s = pred_values[:min_len]
            actual_values_s = actual_values[:min_len]
            residuals_s = actual_values_s - pred_values_s

            pred_list = [
                {
                    "timestamp": t.isoformat(),
                    "actual": float(a),
                    "predicted": float(p),
                    "residual": float(r),
                }
                for t, a, p, r in zip(
                    pred_times_s, actual_values_s, pred_values_s, residuals_s
                )
            ]

            response_data["predictions"] = pred_list

        # Include residual analysis if requested
        if include_residual_analysis:
            residual_analysis = calculate_residual_analysis(
                test_series[lookback_steps:],
                predictions,
                scaler=loader.target_scaler,
            )
            response_data["residual_analysis"] = residual_analysis

        # Include peak metrics if peak_dates are provided (Li et al. 2025)
        peak_metrics = _peak_metrics_safe(
            test_series[lookback_steps:], predictions, peak_dates, loader.target_scaler
        )
        if peak_metrics is not None:
            response_data["peak_metrics"] = peak_metrics
            # Surface the peak headline (peak_mape, peak_timing_error_hours)
            # alongside the standard metrics; full detail stays in peak_metrics.
            _merge_peak_headline(test_metrics, peak_metrics)

        # Probabilistic (interval) metrics for quantile-trained models.
        if config.get("probabilistic") and getattr(
            model, "supports_probabilistic_prediction", False
        ):
            trained_quantiles = sorted(
                {float(q) for q in (config.get("quantiles") or [])} | {0.5}
            )
            try:
                stochastic_raw = generate_predictions(
                    model, test_series, test_covariates, lookback_steps, horizon_steps,
                    future_covariates=test_future_covariates,
                    last_points_only=False,
                    num_samples=max(1, int(num_samples)),
                )
                if isinstance(stochastic_raw, list):
                    stochastic = darts_concatenate(
                        stochastic_raw, ignore_time_axis=False
                    )
                else:
                    stochastic = stochastic_raw

                if getattr(stochastic, "n_samples", 1) > 1:
                    quantile_forecasts = {
                        q: stochastic.quantile(q) for q in trained_quantiles
                    }
                    prob_metrics = calculate_probabilistic_metrics(
                        test_series[lookback_steps:],
                        quantile_forecasts,
                        scaler=loader.target_scaler,
                    )
                    prob_metrics["quantiles"] = trained_quantiles
                    prob_metrics["num_samples"] = int(num_samples)
                    response_data["probabilistic_metrics"] = prob_metrics

                    # Attach the per-timestamp quantile values to the
                    # prediction rows so downstream consumers (HTML report,
                    # output CSV) can draw prediction bands.  Keys match the
                    # generate_forecast convention: "q0.1", "q0.9", ...
                    if return_predictions and response_data.get("predictions"):
                        _attach_quantiles_to_predictions(
                            response_data["predictions"],
                            quantile_forecasts,
                            loader.target_scaler,
                        )
            except Exception as e:
                logger.warning("Failed to compute probabilistic metrics: %s", e)

        # Save to CSV if requested
        if output_csv_path and return_predictions:
            pred_df = pd.DataFrame(response_data["predictions"])
            pred_df.to_csv(output_csv_path, index=False)
            response_data["output_csv_path"] = output_csv_path

        if ml_warnings:
            response_data["ml_warnings"] = ml_warnings

        return create_success_response(**response_data)

    except Exception as e:
        logger.exception("Evaluation failed")
        return create_error_response(f"Evaluation failed: {str(e)}")
