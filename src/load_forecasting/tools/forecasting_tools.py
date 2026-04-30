"""
MCP Tool implementations for load forecasting.

These functions are wrapped and registered in server.py.
"""

from typing import Optional
import logging

from ..core.data_loader import ForecastingDataLoader, DataLoadError
from ..core.trainer import train_model as _train_model, get_available_models
from ..core.evaluator import calculate_residual_analysis, compare_to_validation
from ..core.model_registry import (
    ModelRegistry,
    ModelNotFoundError,
    ModelCorruptedError,
)
from ..core.frequency_utils import hours_to_steps

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
        train_series, train_covariates = train_loader.to_darts_series(fit_scalers=True)

        # Copy fitted scalers to val_loader for consistent scaling
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_series, _ = val_loader.to_darts_series(fit_scalers=False)

        # Get full covariates for evaluation (needed for historical_forecasts)
        # Darts needs covariates to extend beyond the target series for predictions
        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        _, full_covariates = loader.to_darts_series(fit_scalers=False)

        # Update data summary with split info
        data_summary["training_samples"] = len(train_loader.df)
        data_summary["validation_samples"] = len(val_loader.df)

        logger.info(
            f"Training with lookback={lookback_hours}h ({lookback_steps} steps), "
            f"horizon={horizon_hours}h ({horizon_steps} steps) "
            f"at frequency={frequency}"
        )

        # Train model (pass full covariates for proper evaluation)
        model, training_metrics, validation_metrics, training_info = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type=model_type,
            lookback=lookback_steps,
            horizon=horizon_steps,
            train_covariates=full_covariates,
            val_covariates=full_covariates,
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

        # Convert to Darts TimeSeries (use existing scalers)
        test_series, test_covariates = loader.to_darts_series(fit_scalers=False)

        # Convert hours to time steps based on data frequency
        lookback_steps = hours_to_steps(lookback, frequency)
        horizon_steps = hours_to_steps(horizon, frequency)

        # Generate predictions
        from ..core.trainer import _generate_predictions

        predictions = _generate_predictions(
            model, test_series, test_covariates, lookback_steps, horizon_steps
        )

        # Calculate metrics
        from ..core.evaluator import calculate_metrics

        test_metrics = calculate_metrics(
            test_series.slice_n_points_after(start=lookback_steps),
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
                    test_series.slice_n_points_after(start=lookback)
                )
            else:
                pred_original = predictions
                actual_original = test_series.slice_n_points_after(start=lookback)

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
                test_series.slice_n_points_after(start=lookback),
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
