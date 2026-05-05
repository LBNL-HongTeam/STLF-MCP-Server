"""
MCP Tool implementations for load forecasting.

These functions are wrapped and registered in server.py.
"""

from typing import Optional
import logging

import numpy as np
import pandas as pd

from ..core.data_loader import ForecastingDataLoader, DataLoadError, _infer_frequency
from ..core.trainer import train_model as _train_model, get_available_models
from ..core.evaluator import calculate_residual_analysis, compare_to_validation
from ..core.model_registry import (
    ModelRegistry,
    ModelNotFoundError,
    ModelCorruptedError,
)
from ..core.frequency_utils import hours_to_steps, FREQ_TO_STEPS_PER_HOUR

logger = logging.getLogger(__name__)


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


async def train_forecast_model(
    csv_path: str,
    model_type: str = "LinearRegression",
    lookback_hours: int = 24,
    horizon_hours: int = 6,
    frequency: str = "h",
    validation_split: float = 0.2,
    building_name: Optional[str] = None,
    model_name: Optional[str] = None,
    column_mapping: Optional[dict] = None,
) -> dict:
    """
    Train a forecasting model on historical building load data.

    Args:
        csv_path: Path to CSV file with datetime and load data
        model_type: Model type to train
        lookback_hours: Hours of history for model input
        horizon_hours: Hours ahead to forecast
        frequency: Data frequency (15min, 30min, h)
        validation_split: Fraction for validation
        building_name: Building identifier
        model_name: Custom model name
        column_mapping: Map CSV columns to roles

    Returns:
        Dict with model_id, metrics, and data summary
    """
    data_summary = {}

    try:
        # Validate model type
        available_models = get_available_models()
        if model_type not in available_models:
            return create_error_response(
                f"Unknown model type: {model_type}. Available: {available_models}"
            )

        # Validate parameters
        if not 1 <= lookback_hours <= 168:
            return create_error_response("lookback_hours must be between 1 and 168")

        if not 1 <= horizon_hours <= 48:
            return create_error_response("horizon_hours must be between 1 and 48")

        if not 0.1 <= validation_split <= 0.3:
            return create_error_response("validation_split must be between 0.1 and 0.3")

        # Load and validate data
        try:
            loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=column_mapping,
                frequency=frequency,
            )
        except DataLoadError as e:
            return create_error_response(str(e))

        data_summary = loader.get_data_summary()

        # Convert hours to time steps based on data frequency
        lookback_steps = hours_to_steps(lookback_hours, frequency)
        horizon_steps = hours_to_steps(horizon_hours, frequency)

        # Check minimum data requirements (using steps, not hours)
        min_samples = lookback_steps + horizon_steps + 100  # Need buffer
        if data_summary["total_samples"] < min_samples:
            return create_error_response(
                f"Insufficient data: {data_summary['total_samples']} samples. "
                f"Need at least {min_samples} for lookback={lookback_hours}h "
                f"({lookback_steps} steps), horizon={horizon_hours}h "
                f"({horizon_steps} steps) at frequency={frequency}"
            )

        # Split into train/validation
        train_loader, val_loader = loader.split_train_val(validation_split)

        # Convert to Darts TimeSeries
        train_series, train_covariates, train_future_covariates = (
            train_loader.to_darts_series(fit_scalers=True)
        )

        # Copy fitted scalers to val_loader for consistent scaling
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, _, _ = val_loader.to_darts_series(fit_scalers=False)

        # Get full covariates for evaluation (needed for historical_forecasts)
        # Darts needs covariates to extend beyond the target series for predictions
        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_covariates, full_future_covariates = loader.to_darts_series(fit_scalers=False)

        # Update data summary with split info
        data_summary["training_samples"] = len(train_loader.df)
        data_summary["validation_samples"] = len(val_loader.df)

        logger.info(
            f"Training with lookback={lookback_hours}h ({lookback_steps} steps), "
            f"horizon={horizon_hours}h ({horizon_steps} steps) "
            f"at frequency={frequency}"
        )

        # Train model (pass full covariates for proper evaluation).
        # We pass the full-dataset covariate series (not just train-split) so
        # that historical_forecasts can look up covariates in the validation
        # window without index out-of-bounds.
        model, training_metrics, validation_metrics, training_info = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type=model_type,
            lookback=lookback_steps,
            horizon=horizon_steps,
            train_covariates=full_covariates,
            val_covariates=full_covariates,
            train_future_covariates=full_future_covariates,
            val_future_covariates=full_future_covariates,
            scaler=train_loader.target_scaler,
            frequency=frequency,
        )

        # Save to registry
        registry = ModelRegistry()
        model_id = model_name or registry.generate_model_id(building_name, model_type)

        config = {
            "lookback_hours": lookback_hours,
            "horizon_hours": horizon_hours,
            "frequency": frequency,
            "validation_split": validation_split,
        }

        model_path = registry.save_model(
            model=model,
            model_id=model_id,
            model_type=model_type,
            building_name=building_name,
            config=config,
            column_mapping=loader.column_mapping,
            data_info=data_summary,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            scalers={
                "target_scaler": train_loader.target_scaler,
                "covariate_scaler": train_loader.covariate_scaler,
                "future_covariate_scaler": train_loader.future_covariate_scaler,
            },
        )

        return create_success_response(
            model_id=model_id,
            model_path=model_path,
            model_type=model_type,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            data_summary=data_summary,
            training_info=training_info,
        )

    except Exception as e:
        logger.exception("Training failed")
        return create_error_response(
            f"Training failed: {str(e)}",
            data_summary=data_summary if data_summary else None,
        )


async def evaluate_forecast_model(
    model_id: str,
    csv_path: str,
    column_mapping: Optional[dict] = None,
    return_predictions: bool = True,
    output_csv_path: Optional[str] = None,
    include_residual_analysis: bool = False,
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

    Returns:
        Dict with test metrics and predictions
    """
    try:
        # Load model from registry
        registry = ModelRegistry()

        try:
            model, metadata, scalers = registry.load_model(model_id)
        except ModelNotFoundError:
            return create_error_response(f"Model not found: {model_id}")
        except ModelCorruptedError as e:
            return create_error_response(str(e))

        # Use column mapping from training if not provided
        if column_mapping is None:
            column_mapping = metadata.get("column_mapping")
            column_mapping_source = "from_model_metadata"
        else:
            column_mapping_source = "provided"

        # Get config from metadata
        config = metadata.get("config", {})
        lookback = config.get("lookback_hours", 24)
        horizon = config.get("horizon_hours", 6)
        frequency = config.get("frequency", "h")

        # Load test data
        try:
            loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=column_mapping,
                frequency=frequency,
            )
        except DataLoadError as e:
            return create_error_response(str(e))

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

        # Convert hours to time steps based on data frequency
        lookback_steps = hours_to_steps(lookback, frequency)
        horizon_steps = hours_to_steps(horizon, frequency)

        # Generate predictions
        from ..core.trainer import _generate_predictions

        predictions = _generate_predictions(
            model, test_series, test_covariates, lookback_steps, horizon_steps,
            future_covariates=test_future_covariates,
        )

        # Calculate metrics
        from ..core.evaluator import calculate_metrics

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

            # Build predictions list
            pred_list = []
            pred_times = pred_original.time_index
            pred_values = pred_original.values().flatten()
            actual_values = actual_original.values().flatten()

            min_len = min(len(pred_times), len(pred_values), len(actual_values))
            for i in range(min_len):
                pred_list.append({
                    "timestamp": pred_times[i].isoformat(),
                    "actual": float(actual_values[i]),
                    "predicted": float(pred_values[i]),
                    "residual": float(actual_values[i] - pred_values[i]),
                })

            response_data["predictions"] = pred_list

        # Include residual analysis if requested
        if include_residual_analysis:
            residual_analysis = calculate_residual_analysis(
                test_series[lookback_steps:],
                predictions,
                scaler=loader.target_scaler,
            )
            response_data["residual_analysis"] = residual_analysis

        # Save to CSV if requested
        if output_csv_path and return_predictions:
            import pandas as pd

            pred_df = pd.DataFrame(response_data["predictions"])
            pred_df.to_csv(output_csv_path, index=False)
            response_data["output_csv_path"] = output_csv_path

        return create_success_response(**response_data)

    except Exception as e:
        logger.exception("Evaluation failed")
        return create_error_response(f"Evaluation failed: {str(e)}")


async def list_models(
    building_name: Optional[str] = None,
    model_type: Optional[str] = None,
    sort_by: str = "created_at",
    limit: int = 20,
) -> dict:
    """
    List trained models in the registry.

    Args:
        building_name: Filter by building
        model_type: Filter by model type
        sort_by: Sort field
        limit: Maximum results

    Returns:
        Dict with models list and total count
    """
    try:
        registry = ModelRegistry()

        models, total_count = registry.list_models(
            building_name=building_name,
            model_type=model_type,
            sort_by=sort_by,
            limit=limit,
        )

        return create_success_response(
            models=models,
            total_count=total_count,
            filters_applied={
                "building_name": building_name,
                "model_type": model_type,
                "sort_by": sort_by,
                "limit": limit,
            },
        )

    except Exception as e:
        logger.exception("List models failed")
        return create_error_response(f"Failed to list models: {str(e)}")


async def inspect_data(
    csv_path: str,
    column_mapping: Optional[dict] = None,
    frequency: Optional[str] = None,
) -> dict:
    """
    Inspect a CSV file and report everything an agent needs before training.

    Reads the file, auto-detects or validates column roles, infers the data
    frequency, computes per-column statistics, identifies gaps and anomalies,
    and returns actionable feature suggestions.

    Args:
        csv_path: Path to CSV file to inspect.
        column_mapping: Optional explicit column roles (datetime, target,
            past_covariates).  Auto-detected if not provided.
        frequency: Expected data frequency ('15min', '30min', 'h').  If
            omitted, the tool infers it from the timestamps.

    Returns:
        Dict with detected columns, frequency, statistics, gap analysis,
        quality flags, and feature suggestions.
    """
    try:
        from pathlib import Path

        path = Path(csv_path)
        if not path.exists():
            return create_error_response(
                f"File not found: {csv_path}\n"
                "Check that the path is correct and the file is accessible."
            )

        # ------------------------------------------------------------------
        # Read raw CSV (no preprocessing yet — we want the raw picture)
        # ------------------------------------------------------------------
        try:
            raw_df = pd.read_csv(csv_path)
        except Exception as e:
            return create_error_response(f"Failed to read CSV: {e}")

        if raw_df.empty:
            return create_error_response("CSV file is empty.")

        n_rows_raw = len(raw_df)
        columns_in_file = raw_df.columns.tolist()

        # ------------------------------------------------------------------
        # Run the data loader (auto-detect columns, parse datetimes, dedup)
        # Use add_calendar_features=False / lag_hours=[] so we inspect only
        # what the user actually has in their file.
        # ------------------------------------------------------------------
        detected_frequency = frequency
        loader_error: Optional[str] = None
        loader: Optional[ForecastingDataLoader] = None

        # Determine frequency to pass to loader
        if detected_frequency is None:
            # Sniff from raw datetime column without full loader initialisation
            detected_frequency = _sniff_frequency(raw_df)

        try:
            loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=column_mapping,
                frequency=detected_frequency or "h",
                add_calendar_features=False,
                lag_hours=[],
            )
        except DataLoadError as e:
            loader_error = str(e)

        # ------------------------------------------------------------------
        # Column detection report
        # ------------------------------------------------------------------
        if loader is not None:
            resolved_mapping = loader.column_mapping
            dt_col = resolved_mapping.get("datetime")
            target_col = resolved_mapping.get("target")
            cov_cols = resolved_mapping.get("past_covariates", [])
            fut_cov_cols = resolved_mapping.get("future_covariates", [])
            inferred_freq = _infer_frequency(loader.df.index)
            actual_frequency = inferred_freq or detected_frequency or "unknown"
        else:
            # Loader failed — do best-effort column sniffing from raw df
            resolved_mapping = _sniff_columns(raw_df, column_mapping)
            dt_col = resolved_mapping.get("datetime")
            target_col = resolved_mapping.get("target")
            cov_cols = resolved_mapping.get("past_covariates", [])
            fut_cov_cols = resolved_mapping.get("future_covariates", [])
            actual_frequency = detected_frequency or "unknown"

        all_mapped = [dt_col, target_col] + cov_cols + fut_cov_cols
        column_report = {
            "datetime": dt_col,
            "target": target_col,
            "past_covariates": cov_cols,
            "future_covariates": fut_cov_cols,
            "unrecognised": [
                c for c in columns_in_file
                if c not in all_mapped
            ],
            "all_columns": columns_in_file,
        }

        # ------------------------------------------------------------------
        # Frequency report
        # ------------------------------------------------------------------
        freq_report = {
            "declared": frequency,
            "inferred": actual_frequency,
            "match": (frequency is None) or (actual_frequency == frequency),
            "supported": actual_frequency in FREQ_TO_STEPS_PER_HOUR,
        }
        if not freq_report["match"]:
            freq_report["recommendation"] = (
                f"Set frequency='{actual_frequency}' to match the data."
            )

        # ------------------------------------------------------------------
        # Row / time range summary
        # ------------------------------------------------------------------
        time_range: dict = {}
        n_rows_clean = n_rows_raw
        if loader is not None:
            n_rows_clean = len(loader.df)
            time_range = {
                "start": loader.df.index.min().isoformat(),
                "end": loader.df.index.max().isoformat(),
                "n_rows_raw": n_rows_raw,
                "n_rows_after_dedup": n_rows_clean,
                "n_duplicates_removed": n_rows_raw - n_rows_clean,
            }
            # Expected row count at detected frequency
            if actual_frequency in FREQ_TO_STEPS_PER_HOUR:
                steps_per_hour = FREQ_TO_STEPS_PER_HOUR[actual_frequency]
                duration_hours = (
                    loader.df.index.max() - loader.df.index.min()
                ).total_seconds() / 3600
                expected_rows = int(duration_hours * steps_per_hour) + 1
                time_range["expected_rows_at_frequency"] = expected_rows
                time_range["coverage_pct"] = round(
                    100.0 * n_rows_clean / max(expected_rows, 1), 2
                )

        # ------------------------------------------------------------------
        # Per-column statistics
        # We use raw_df for missing-value counts (pre-interpolation) so that
        # the report reflects what the user actually has in their file.
        # For numeric stats (mean, std, …) we use the cleaned loader.df when
        # available, falling back to raw_df otherwise.
        # ------------------------------------------------------------------
        column_stats: list[dict] = []
        work_df = loader.df if loader is not None else raw_df
        raw_df_indexed = raw_df.set_index(raw_df.columns[0]) if loader is not None else raw_df

        # Build the list of columns to report stats for
        _stat_cols: list = []
        if target_col:
            _stat_cols.append(target_col)
        _stat_cols.extend(cov_cols)
        _stat_cols.extend(fut_cov_cols)

        for col in _stat_cols:
            if col not in work_df.columns:
                continue
            s = work_df[col]
            # Count nulls from raw data (pre-interpolation) when possible
            raw_col = raw_df[col] if col in raw_df.columns else s
            n_null = int(raw_col.isna().sum())
            n_total_raw = len(raw_col)
            non_null = s.dropna()
            _role = (
                "target" if col == target_col
                else "future_covariate" if col in fut_cov_cols
                else "past_covariate"
            )
            stat: dict = {
                "column": col,
                "role": _role,
                "n_total": n_total_raw,
                "n_missing": n_null,
                "coverage_pct": round(100.0 * (n_total_raw - n_null) / max(n_total_raw, 1), 2),
                "dtype": str(s.dtype),
            }
            if pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
                stat.update({
                    "mean": round(float(non_null.mean()), 4),
                    "std": round(float(non_null.std()), 4),
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "p5": round(float(np.percentile(non_null, 5)), 4),
                    "p95": round(float(np.percentile(non_null, 95)), 4),
                    "n_negative": int((non_null < 0).sum()),
                    "n_zero": int((non_null == 0).sum()),
                })
            column_stats.append(stat)

        # ------------------------------------------------------------------
        # Gap analysis (only possible when loader succeeded)
        # ------------------------------------------------------------------
        gap_report: dict = {"analysis_available": loader is not None}
        if loader is not None and actual_frequency in FREQ_TO_STEPS_PER_HOUR:
            expected_td = pd.Timedelta(hours=1) / FREQ_TO_STEPS_PER_HOUR[actual_frequency]
            diffs = loader.df.index.to_series().diff().dropna()
            gaps = diffs[diffs > expected_td * 1.5]  # 50% tolerance
            gap_report["n_gaps"] = len(gaps)
            gap_report["total_missing_steps"] = int(
                sum((g / expected_td) - 1 for g in gaps)
            )
            if len(gaps) > 0:
                largest = gaps.max()
                gap_report["largest_gap"] = str(largest)
                gap_report["largest_gap_start"] = gaps.idxmax().isoformat()
                gap_list = []
                for ts, dur in gaps.sort_values(ascending=False).head(5).items():
                    gap_list.append({
                        "start": ts.isoformat(),
                        "duration": str(dur),
                        "missing_steps": int(dur / expected_td) - 1,
                    })
                gap_report["top_gaps"] = gap_list

        # ------------------------------------------------------------------
        # Quality flags
        # ------------------------------------------------------------------
        quality_flags: list[str] = []
        if loader_error:
            quality_flags.append(f"LOAD_ERROR: {loader_error}")
        if not freq_report["match"]:
            quality_flags.append(
                f"FREQUENCY_MISMATCH: declared={frequency}, "
                f"inferred={actual_frequency}"
            )
        if not freq_report["supported"]:
            quality_flags.append(
                f"UNSUPPORTED_FREQUENCY: '{actual_frequency}' — "
                "use 15min, 30min, or h"
            )
        if time_range.get("n_duplicates_removed", 0) > 0:
            quality_flags.append(
                f"DUPLICATES: {time_range['n_duplicates_removed']} "
                "duplicate timestamps removed"
            )
        if time_range.get("coverage_pct", 100) < 90:
            quality_flags.append(
                f"LOW_COVERAGE: {time_range.get('coverage_pct')}% row coverage "
                f"(< 90% threshold)"
            )
        if gap_report.get("n_gaps", 0) > 0:
            quality_flags.append(
                f"GAPS: {gap_report['n_gaps']} gap(s) totalling "
                f"{gap_report.get('total_missing_steps', '?')} missing steps"
            )
        for stat in column_stats:
            if stat.get("n_missing", 0) > 0:
                quality_flags.append(
                    f"MISSING_VALUES: column '{stat['column']}' has "
                    f"{stat['n_missing']} null(s) ({100 - stat['coverage_pct']:.1f}%)"
                )
            if stat.get("n_negative", 0) > 0 and stat["role"] == "target":
                quality_flags.append(
                    f"NEGATIVE_TARGET: column '{stat['column']}' has "
                    f"{stat['n_negative']} negative value(s)"
                )

        # ------------------------------------------------------------------
        # Feature suggestions
        # ------------------------------------------------------------------
        suggestions: list[str] = []
        if target_col:
            suggestions.append(
                "Calendar features (hour, day-of-week, month, is_weekend, "
                "hour_sin/cos) are auto-added by the data loader — no action needed."
            )
            suggestions.append(
                "Lag features at 24h, 48h, and 168h are auto-added by the "
                "data loader — no action needed."
            )
        if not cov_cols:
            suggestions.append(
                "No weather/covariate columns detected. Adding outdoor "
                "temperature (column name containing 'temp' or 'temperature') "
                "typically improves accuracy."
            )
        if column_report["unrecognised"]:
            suggestions.append(
                f"Columns {column_report['unrecognised']} were not mapped to "
                "any role. If they contain useful signals, pass them explicitly "
                "via column_mapping={'past_covariates': [...]}."
            )
        if time_range.get("coverage_pct", 100) < 95:
            suggestions.append(
                "Coverage is below 95%. Gaps up to 2h are interpolated "
                "automatically; larger gaps will remain as NaN and may hurt "
                "model accuracy."
            )
        if loader is not None and n_rows_clean < 500:
            suggestions.append(
                f"Only {n_rows_clean} rows after deduplication. At least ~500 "
                "rows (ideally 3+ months) are recommended for reliable training."
            )
        if target_col and loader is not None:
            # Check if data is long enough for default lookback/horizon
            min_needed = hours_to_steps(24, actual_frequency if actual_frequency in FREQ_TO_STEPS_PER_HOUR else "h") + \
                         hours_to_steps(6, actual_frequency if actual_frequency in FREQ_TO_STEPS_PER_HOUR else "h") + 100
            if n_rows_clean < min_needed:
                suggestions.append(
                    f"Data has {n_rows_clean} rows, which may be insufficient "
                    f"for default parameters (lookback=24h, horizon=6h). "
                    f"Minimum recommended: {min_needed} rows."
                )
            else:
                suggestions.append(
                    f"Data length ({n_rows_clean} rows) is sufficient for "
                    "training with default parameters."
                )

        # ------------------------------------------------------------------
        # Readiness verdict
        # ------------------------------------------------------------------
        blocking = [f for f in quality_flags if f.startswith(("LOAD_ERROR", "UNSUPPORTED_FREQUENCY"))]
        ready_to_train = loader_error is None and freq_report["supported"] and bool(target_col)

        return create_success_response(
            file_path=csv_path,
            n_rows_raw=n_rows_raw,
            columns=column_report,
            frequency=freq_report,
            time_range=time_range,
            column_statistics=column_stats,
            gaps=gap_report,
            quality_flags=quality_flags,
            suggestions=suggestions,
            ready_to_train=ready_to_train,
            blocking_issues=blocking,
            loader_error=loader_error,
        )

    except Exception as e:
        logger.exception("inspect_data failed")
        return create_error_response(f"Inspection failed: {str(e)}")


# ---------------------------------------------------------------------------
# Private helpers for inspect_data
# ---------------------------------------------------------------------------

def _sniff_frequency(raw_df: pd.DataFrame) -> Optional[str]:
    """
    Attempt to infer frequency from the first column that parses as datetimes.
    Returns a frequency string or None.
    """
    for col in raw_df.columns:
        try:
            parsed = pd.to_datetime(raw_df[col], utc=True)
            idx = pd.DatetimeIndex(parsed)
            return _infer_frequency(idx)
        except Exception:
            continue
    return None


def _sniff_columns(raw_df: pd.DataFrame, mapping: Optional[dict]) -> dict:
    """
    Best-effort column role detection without running the full loader.
    Used as fallback when ForecastingDataLoader.__init__ fails.
    """
    if mapping:
        result = dict(mapping)
        if "past_covariates" not in result:
            result["past_covariates"] = []
        if "future_covariates" not in result:
            result["future_covariates"] = []
        return result

    dt_patterns = ["datetime", "timestamp", "date", "time", "dt"]
    target_patterns = ["kwh", "load", "power", "energy", "electricity", "demand"]
    cov_patterns = ["temp", "temperature", "rh", "humidity", "solar", "wind"]

    result: dict = {"past_covariates": [], "future_covariates": []}
    for col in raw_df.columns:
        cl = col.lower()
        if "datetime" not in result and any(p in cl for p in dt_patterns):
            result["datetime"] = col
        elif "target" not in result and any(p in cl for p in target_patterns):
            result["target"] = col
        elif any(p in cl for p in cov_patterns):
            result["past_covariates"].append(col)
    return result
