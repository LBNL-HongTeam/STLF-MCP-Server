"""
MCP tool: generate_forecast — forward inference from a trained model.
"""

from typing import Optional
import logging

import pandas as pd

from ..core.data_loader import DataLoadError
from ..core.trainer import (
    LOCAL_MODEL_NAMES,
    FUTURE_ONLY_MODELS,
)
from ..core.frequency_utils import hours_to_steps, FREQ_TO_STEPS_PER_HOUR
from ._common import (
    create_success_response,
    create_error_response,
    _ModelLoadError,
    _load_model_context,
    _append_warning,
    _check_horizon_lookback,
    _check_frequency_mismatch,
    _load_and_scale_series,
    _parse_raw_csv_for_inference,
)

logger = logging.getLogger(__name__)


def generate_forecast(
    model_id: str,
    csv_path: str,
    column_mapping: Optional[dict] = None,
    horizon_hours: Optional[int] = None,
    output_csv_path: Optional[str] = None,
    num_samples: int = 200,
) -> dict:
    """
    Generate a forward forecast from a trained model using recent context data.

    Loads a trained model and runs inference on the provided context window,
    returning predictions for the next ``horizon_hours`` hours.

    For models trained **without future covariates**: supply a CSV with at
    least ``lookback_hours`` of recent history for the target (and any past
    covariate) columns.

    For models trained **with future covariates** (e.g. weather forecasts):
    supply a single extended CSV where the future-covariate columns extend
    ``horizon_hours`` rows beyond the last non-null target row.  Target and
    past-covariate columns may be absent or NaN for those future rows.

    Args:
        model_id: ID of a trained model (from train_forecast_model).
        csv_path: Path to CSV with recent context data.
        column_mapping: Column role mapping. If not provided, the mapping
            stored in the model metadata from training is used.
        horizon_hours: Forecast horizon in hours.  Overrides the horizon the
            model was trained with (must be 1–48).  If not provided, uses the
            training horizon.
        output_csv_path: Optional path to write predictions as CSV.
        num_samples: Number of Monte-Carlo samples to draw when the model was
            trained with probabilistic=True.  Ignored for point models.  More
            samples give smoother quantile estimates at higher cost
            (default 200).

    Returns:
        Dict with success, model_id, model_type, forecast_horizon_hours,
        predictions (list of {datetime, predicted_load}), forecast_start,
        forecast_end, context_summary, and optionally output_csv_path.

        For models trained with probabilistic=True, each prediction row also
        carries the fitted quantiles as ``q<level>`` keys (e.g. ``q0.1``,
        ``q0.9``) and the response includes ``probabilistic: True`` plus the
        ``quantiles`` list.  The ``predicted_load`` value is the median (P50).
    """
    from darts import concatenate as darts_concatenate, TimeSeries as DartsTimeSeries

    try:
        # Load model + config context from registry
        try:
            ctx = _load_model_context(model_id, column_mapping)
        except _ModelLoadError as e:
            return e.response
        model = ctx["model"]
        metadata = ctx["metadata"]
        scalers = ctx["scalers"]
        model_type = ctx["model_type"]
        column_mapping = ctx["column_mapping"]
        if column_mapping is None:
            return create_error_response(
                "No column_mapping provided and none stored in model metadata."
            )
        frequency = ctx["frequency"]
        lookback_hours_cfg = ctx["lookback_hours"]
        horizon_hours_cfg = ctx["horizon_hours"]
        lookback_steps = ctx["lookback_steps"]

        # Probabilistic (quantile) config, if the model was trained with it.
        config = ctx["config"]
        is_probabilistic = bool(config.get("probabilistic"))
        trained_quantiles = config.get("quantiles") or []

        # Apply horizon_hours override if provided
        if horizon_hours is not None:
            if not 1 <= horizon_hours <= 96:
                return create_error_response("horizon_hours must be between 1 and 96")
            horizon_steps = hours_to_steps(horizon_hours, frequency)
            effective_horizon_hours = horizon_hours
        else:
            horizon_steps = hours_to_steps(horizon_hours_cfg, frequency)
            effective_horizon_hours = horizon_hours_cfg

        # Soft warning: horizon exceeds lookback
        forecast_ml_warnings: list[str] = []
        _append_warning(
            forecast_ml_warnings,
            _check_horizon_lookback(effective_horizon_hours, lookback_hours_cfg),
        )

        # ------------------------------------------------------------------
        # Parse raw CSV to split context rows from horizon rows
        # ------------------------------------------------------------------
        try:
            raw_df = _parse_raw_csv_for_inference(csv_path, column_mapping)
        except DataLoadError as e:
            return create_error_response(str(e))

        target_col = column_mapping.get("target")
        if not target_col or target_col not in raw_df.columns:
            return create_error_response(
                f"Target column '{target_col}' not found in CSV. "
                "Check column_mapping or CSV column names."
            )

        future_cov_cols = column_mapping.get("future_covariates", []) or []
        datetime_col = column_mapping.get("datetime")

        # Split: rows with non-NaN target = context; trailing NaN rows = horizon.
        # raw_df has an integer index (reset by _parse_raw_csv_for_inference).
        last_valid_pos = raw_df[target_col].last_valid_index()
        if last_valid_pos is None:
            return create_error_response(
                f"Target column '{target_col}' has no valid (non-NaN) values in the CSV."
            )

        # context_df: rows 0 .. last_valid_pos inclusive (datetime still a column)
        context_df = raw_df.iloc[: last_valid_pos + 1].copy()
        # horizon_df: rows after last_valid_pos (future_cov columns only)
        horizon_df = (
            raw_df.iloc[last_valid_pos + 1:][
                ([datetime_col] if datetime_col and datetime_col in raw_df.columns else [])
                + [c for c in future_cov_cols if c in raw_df.columns]
            ].copy()
            if future_cov_cols
            else pd.DataFrame()
        )

        # ------------------------------------------------------------------
        # Load context through ForecastingDataLoader with saved scalers
        # ------------------------------------------------------------------
        try:
            loader, target_series, past_cov, context_future_cov = _load_and_scale_series(
                csv_path=csv_path,
                column_mapping=column_mapping,
                scalers=scalers,
                frequency=frequency,
                dataframe=context_df,
                metadata=ctx["metadata"],
            )
        except DataLoadError as e:
            return create_error_response(f"Failed to load context data: {e}")
        except _ModelLoadError as e:
            return e.response

        context_rows = len(loader.df)
        context_start = loader.df.index.min().isoformat()
        context_end = loader.df.index.max().isoformat()

        # Hard block: frequency mismatch between model and context CSV
        freq_err = _check_frequency_mismatch(loader, frequency)
        if freq_err:
            return create_error_response(freq_err)

        # Warn if lag features may be unreliable (less than 168h of context)
        steps_per_hour = FREQ_TO_STEPS_PER_HOUR.get(frequency, 1)
        min_recommended_steps = 168 * steps_per_hour
        context_warning = None
        if context_rows < min_recommended_steps:
            context_warning = (
                f"Context has only {context_rows} rows "
                f"({context_rows / steps_per_hour:.0f}h). "
                "Lag features (24h/48h/168h) may be approximate. "
                f"Provide at least 168h ({min_recommended_steps} rows) for best accuracy."
            )

        # ------------------------------------------------------------------
        # Short-context check for GlobalForecastingModels
        # ------------------------------------------------------------------
        is_local_model = model_type in LOCAL_MODEL_NAMES or model_type == "ARIMA"
        if not is_local_model and len(target_series) < lookback_steps:
            return create_error_response(
                f"Model requires at least {lookback_steps} steps "
                f"({lookback_hours_cfg}h at {frequency}) of context, "
                f"but only {len(target_series)} steps "
                f"({len(target_series) / steps_per_hour:.1f}h) were provided. "
                "Extend the context CSV to cover the full lookback window."
            )

        # ------------------------------------------------------------------
        # Build full future_covariates series (context + horizon rows)
        # ------------------------------------------------------------------
        full_future_cov = None
        if future_cov_cols:
            if context_future_cov is None:
                return create_error_response(
                    f"Model was trained with future covariates {future_cov_cols} "
                    "but none were found in the provided CSV."
                )

            if len(horizon_df) > 0:
                # Scale the horizon rows using the saved future_covariate_scaler.
                # horizon_df has datetime as a column (not index).
                future_cov_scaler = scalers.get("future_covariate_scaler")
                hcols = [c for c in future_cov_cols if c in horizon_df.columns]
                time_col_in_horizon = datetime_col if (datetime_col and datetime_col in horizon_df.columns) else None
                try:
                    if time_col_in_horizon:
                        horizon_ts = DartsTimeSeries.from_dataframe(
                            horizon_df[[time_col_in_horizon] + hcols],
                            time_col=time_col_in_horizon,
                            value_cols=hcols,
                            freq=frequency,
                            fill_missing_dates=True,
                            fillna_value=None,
                        )
                    else:
                        # Fallback: set integer index as DatetimeIndex using freq
                        horizon_ts = DartsTimeSeries.from_dataframe(
                            horizon_df[hcols],
                            freq=frequency,
                        )
                    if future_cov_scaler is not None:
                        horizon_ts = future_cov_scaler.transform(horizon_ts)
                    full_future_cov = darts_concatenate(
                        [context_future_cov, horizon_ts], axis=0
                    )
                except Exception as e:
                    return create_error_response(
                        f"Failed to process future covariate horizon rows: {e}"
                    )
            else:
                # No horizon rows — future_cov only covers context period.
                # For global models this will likely cause a Darts error at
                # predict() time; surface a clear message here instead.
                if not is_local_model:
                    return create_error_response(
                        f"Model uses future covariates {future_cov_cols}. "
                        f"The CSV must include {horizon_steps} rows "
                        f"({effective_horizon_hours}h) of future covariate data "
                        "beyond the last non-null target row. "
                        "Add those rows with NaN target and valid covariate values."
                    )
                full_future_cov = context_future_cov
        else:
            full_future_cov = None

        # ------------------------------------------------------------------
        # Run inference
        # ------------------------------------------------------------------
        try:
            if model_type in LOCAL_MODEL_NAMES:
                # Naive baselines: refit on context, then predict
                model.fit(target_series)
                predictions = model.predict(n=horizon_steps)

            elif model_type == "ARIMA":
                # ARIMA: pass series to predict() for temporary state update
                predictions = model.predict(
                    n=horizon_steps,
                    series=target_series,
                    future_covariates=full_future_cov,
                )

            else:
                # GlobalForecastingModels (LinearRegression, XGBoost, LSTM,
                # TFT, TiDE, TSMixer, TimesFM) and hybrid (TimesFM+Residual).
                predict_kwargs: dict = {
                    "n": horizon_steps,
                    "series": target_series,
                }
                if past_cov is not None and model_type not in FUTURE_ONLY_MODELS:
                    predict_kwargs["past_covariates"] = past_cov
                if full_future_cov is not None:
                    predict_kwargs["future_covariates"] = full_future_cov
                # Draw Monte-Carlo samples for probabilistic models so we can
                # extract quantile bands below.  Point models keep num_samples=1.
                if is_probabilistic and getattr(
                    model, "supports_probabilistic_prediction", False
                ):
                    predict_kwargs["num_samples"] = max(1, int(num_samples))

                predictions = model.predict(**predict_kwargs)

        except Exception as e:
            return create_error_response(
                f"Model inference failed: {e}. "
                "Ensure the context window is long enough and covariates are aligned."
            )

        # ------------------------------------------------------------------
        # Inverse-scale and format output
        # ------------------------------------------------------------------
        target_scaler = scalers.get("target_scaler")

        # Detect whether we actually obtained a stochastic (multi-sample)
        # forecast we can slice quantiles from.
        emit_quantiles = (
            is_probabilistic
            and getattr(predictions, "is_stochastic", False)
            and getattr(predictions, "n_samples", 1) > 1
        )

        def _inverse(series):
            if target_scaler is None:
                return series
            return target_scaler.inverse_transform(series)

        try:
            if emit_quantiles:
                # Use the trained quantile levels; fall back to a default band.
                q_levels = sorted(
                    {float(q) for q in trained_quantiles} | {0.5}
                ) or [0.1, 0.5, 0.9]
                # Extract + inverse-transform each quantile as a deterministic
                # series before flattening to values.
                q_value_map = {}
                for q in q_levels:
                    q_series = _inverse(predictions.quantile(q))
                    q_value_map[q] = q_series.values().flatten()
                predictions_original = _inverse(predictions.quantile(0.5))
                pred_times = predictions_original.time_index
                median_values = q_value_map[0.5]
            else:
                predictions_original = _inverse(predictions)
                pred_times = predictions_original.time_index
                median_values = predictions_original.values().flatten()
                q_value_map = {}
        except Exception as e:
            return create_error_response(f"Failed to inverse-transform predictions: {e}")

        pred_list = []
        for i, (t, v) in enumerate(zip(pred_times, median_values)):
            row = {"datetime": t.isoformat(), "predicted_load": float(v)}
            for q, vals in q_value_map.items():
                if i < len(vals):
                    row[f"q{q}"] = float(vals[i])
            pred_list.append(row)

        forecast_start = pred_times[0].isoformat() if len(pred_times) > 0 else None
        forecast_end = pred_times[-1].isoformat() if len(pred_times) > 0 else None

        response_data = {
            "model_id": model_id,
            "model_type": model_type,
            "forecast_horizon_hours": effective_horizon_hours,
            "predictions": pred_list,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "context_summary": {
                "context_rows": context_rows,
                "context_start": context_start,
                "context_end": context_end,
                "lookback_steps_required": lookback_steps,
            },
        }

        if emit_quantiles:
            response_data["probabilistic"] = True
            response_data["quantiles"] = sorted(q_value_map.keys())
            response_data["num_samples"] = int(num_samples)

        if context_warning:
            response_data["context_warning"] = context_warning

        if forecast_ml_warnings:
            response_data["ml_warnings"] = forecast_ml_warnings

        if output_csv_path:
            pred_df = pd.DataFrame(pred_list)
            pred_df.to_csv(output_csv_path, index=False)
            response_data["output_csv_path"] = output_csv_path

        return create_success_response(**response_data)

    except Exception as e:
        logger.exception("generate_forecast failed")
        return create_error_response(f"Forecast generation failed: {str(e)}")
