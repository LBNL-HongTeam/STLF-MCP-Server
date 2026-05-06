"""
Model training wrapper using Darts.

Supports:
- NaiveMean: Predicts mean of training data
- NaiveSeasonal: Repeats pattern from K periods ago
- NaiveMovingAverage: Moving average baseline
- LinearRegression: Regression on lagged features
"""

from typing import Optional, Any
import time
import logging

from darts import TimeSeries
from darts.models import (
    NaiveMean,
    NaiveSeasonal,
    NaiveMovingAverage,
    RegressionModel,
)
from sklearn.linear_model import LinearRegression

from .evaluator import calculate_metrics
from .frequency_utils import get_seasonality_steps

logger = logging.getLogger(__name__)

# Frozenset for O(1) local-model membership test
_LOCAL_MODEL_NAMES: frozenset = frozenset(
    {"NaiveMean", "NaiveSeasonal", "NaiveMovingAverage"}
)

# Model class mapping
MODEL_CLASSES = {
    "NaiveMean": NaiveMean,
    "NaiveSeasonal": NaiveSeasonal,
    "NaiveMovingAverage": NaiveMovingAverage,
    "LinearRegression": "RegressionModel",  # Special handling
}

# Darts class names for metadata
DARTS_CLASS_NAMES = {
    "NaiveMean": "darts.models.NaiveMean",
    "NaiveSeasonal": "darts.models.NaiveSeasonal",
    "NaiveMovingAverage": "darts.models.NaiveMovingAverage",
    "LinearRegression": "darts.models.RegressionModel",
}


def get_available_models() -> list[str]:
    """Return list of available model types."""
    return list(MODEL_CLASSES.keys())


def create_model(
    model_type: str,
    lookback: int = 24,
    horizon: int = 6,
    use_past_covariates: bool = False,
    use_future_covariates: bool = False,
    frequency: str = "h",
    # Legacy alias kept for backwards compatibility
    use_covariates: bool = False,
) -> Any:
    """
    Create a model instance.

    Args:
        model_type: Type of model to create
        lookback: Input window size in TIME STEPS (not hours)
        horizon: Output horizon in TIME STEPS (not hours)
        use_past_covariates: Whether past covariates will be used
        use_future_covariates: Whether future covariates will be used
        frequency: Data frequency for NaiveSeasonal seasonality
        use_covariates: Deprecated alias for use_past_covariates

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_type is unknown
    """
    # Handle legacy alias
    use_past_covariates = use_past_covariates or use_covariates

    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_CLASSES.keys())}"
        )

    if model_type == "LinearRegression":
        # lags_future_covariates: cover current step through the full horizon
        # so the model sees the future covariate at every predicted timestep.
        lags_future = list(range(0, horizon)) if use_future_covariates else None
        model = RegressionModel(
            lags=lookback,
            lags_past_covariates=lookback if use_past_covariates else None,
            lags_future_covariates=lags_future,
            output_chunk_length=horizon,
            model=LinearRegression(),
        )
    elif model_type == "NaiveSeasonal":
        # K = daily seasonality in steps (96 for 15min, 24 for hourly)
        seasonality_k = get_seasonality_steps(frequency)
        model = NaiveSeasonal(K=seasonality_k)
    elif model_type == "NaiveMovingAverage":
        # Use input_chunk_length for moving average window
        model = NaiveMovingAverage(input_chunk_length=lookback)
    else:
        model_class = MODEL_CLASSES[model_type]
        model = model_class()

    return model


def train_model(
    train_series: TimeSeries,
    val_series: TimeSeries,
    model_type: str = "LinearRegression",
    lookback: int = 24,
    horizon: int = 6,
    train_covariates: Optional[TimeSeries] = None,
    val_covariates: Optional[TimeSeries] = None,
    train_future_covariates: Optional[TimeSeries] = None,
    val_future_covariates: Optional[TimeSeries] = None,
    scaler=None,
    frequency: str = "h",
) -> tuple[Any, dict, dict, dict]:
    """
    Train a forecasting model.

    Args:
        train_series: Training time series (scaled)
        val_series: Validation time series (scaled)
        model_type: Type of model to train
        lookback: Input window size in TIME STEPS (not hours)
        horizon: Forecast horizon in TIME STEPS (not hours)
        train_covariates: Training past covariates (optional)
        val_covariates: Validation past covariates (optional)
        train_future_covariates: Training future covariates (optional).
            Must cover at least the training period plus the forecast horizon.
        val_future_covariates: Validation future covariates (optional).
        scaler: Scaler for inverse transform when computing metrics
        frequency: Data frequency for seasonality calculation

    Returns:
        Tuple of (model, training_metrics, validation_metrics, info)
    """
    start_time = time.time()

    # Create model
    use_past_covariates = train_covariates is not None
    use_future_covariates = train_future_covariates is not None
    model = create_model(
        model_type, lookback, horizon,
        use_past_covariates=use_past_covariates,
        use_future_covariates=use_future_covariates,
        frequency=frequency,
    )
    logger.info(
        f"Created {model_type} model with lookback={lookback}, "
        f"horizon={horizon}, past_covariates={use_past_covariates}, "
        f"future_covariates={use_future_covariates}"
    )

    # Train model
    try:
        if model_type == "LinearRegression":
            fit_kwargs: dict = {}
            if train_covariates is not None:
                fit_kwargs["past_covariates"] = train_covariates
            if train_future_covariates is not None:
                fit_kwargs["future_covariates"] = train_future_covariates
            model.fit(train_series, **fit_kwargs)
        else:
            model.fit(train_series)
        logger.info("Model training completed")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

    training_time = time.time() - start_time

    # Evaluate on training data
    try:
        train_pred = _generate_predictions(
            model, train_series, train_covariates, lookback, horizon,
            future_covariates=train_future_covariates,
        )
        training_metrics = calculate_metrics(
            train_series[lookback:],
            train_pred,
            scaler=scaler,
        )
    except Exception as e:
        logger.warning(f"Failed to compute training metrics: {e}")
        training_metrics = {
            "rmse": None, "mae": None, "mape": None,
            "cv_rmse": None, "r_squared": None
        }

    # Evaluate on validation data
    try:
        val_pred = _generate_predictions(
            model, val_series, val_covariates, lookback, horizon,
            future_covariates=val_future_covariates,
        )
        validation_metrics = calculate_metrics(
            val_series[lookback:],
            val_pred,
            scaler=scaler,
        )
    except Exception as e:
        logger.warning(f"Failed to compute validation metrics: {e}")
        validation_metrics = {
            "rmse": None, "mae": None, "mape": None,
            "cv_rmse": None, "r_squared": None
        }

    # Training info
    training_info = {
        "training_time_seconds": round(training_time, 2),
        "darts_model_class": DARTS_CLASS_NAMES.get(model_type, "unknown"),
        "lookback_hours": lookback,
        "horizon_hours": horizon,
    }

    return model, training_metrics, validation_metrics, training_info


def _generate_predictions(
    model: Any,
    series: TimeSeries,
    covariates: Optional[TimeSeries],
    lookback: int,
    horizon: int,
    future_covariates: Optional[TimeSeries] = None,
) -> TimeSeries:
    """
    Generate predictions using historical_forecasts.

    Args:
        model: Trained model
        series: Input series
        covariates: Optional past covariates
        lookback: Lookback window
        horizon: Forecast horizon
        future_covariates: Optional future covariates (columns known at
            forecast time, e.g. weather forecasts).

    Returns:
        Concatenated predictions as TimeSeries
    """
    # Check if this is a LocalForecastingModel (naive baselines)
    model_class_name = model.__class__.__name__
    is_local_model = model_class_name in _LOCAL_MODEL_NAMES

    if is_local_model:
        # LocalForecastingModels don't support historical_forecasts
        # with retrain=False. Implement manual walk-forward validation
        logger.info(
            f"Using manual walk-forward validation for {model_class_name}"
        )
        return _manual_walk_forward(model, series, lookback, horizon)

    # For GlobalForecastingModels, use historical_forecasts
    start = lookback

    try:
        hf_kwargs: dict = {
            "start": start,
            "forecast_horizon": horizon,
            "stride": horizon,
            "retrain": False,
            "verbose": False,
        }
        if covariates is not None and hasattr(model, "past_covariates"):
            hf_kwargs["past_covariates"] = covariates
        if future_covariates is not None and hasattr(model, "future_covariates"):
            hf_kwargs["future_covariates"] = future_covariates

        predictions = model.historical_forecasts(series, **hf_kwargs)
    except Exception as e:
        logger.warning(f"historical_forecasts failed: {e}")
        raise

    return predictions


def _manual_walk_forward(
    model: Any,
    series: TimeSeries,
    lookback: int,
    horizon: int,
) -> TimeSeries:
    """
    Perform manual walk-forward validation for LocalForecastingModels.

    LocalForecastingModels need to be refit at each step since they don't
    support historical_forecasts with retrain=False.

    Args:
        model: Trained LocalForecastingModel
        series: Input series
        lookback: Minimum lookback window
        horizon: Forecast horizon

    Returns:
        Concatenated predictions as TimeSeries
    """
    from darts import concatenate

    predictions_list = []
    start_idx = lookback

    # Allocate a single reusable model instance before the loop so we avoid
    # repeated object construction on every stride step (O(n) instead of O(n²)).
    model_class = model.__class__
    if hasattr(model, 'K'):  # NaiveSeasonal
        walk_model = model_class(K=model.K)
    elif hasattr(model, 'input_chunk_length'):  # NaiveMovingAverage
        walk_model = model_class(input_chunk_length=model.input_chunk_length)
    else:  # NaiveMean
        walk_model = model_class()

    # Generate predictions at regular intervals
    while start_idx + horizon <= len(series):
        # Get training data up to current point
        train_series = series[:start_idx]

        # Refit and predict
        walk_model.fit(train_series)
        pred = walk_model.predict(n=horizon)
        predictions_list.append(pred)

        # Move forward by horizon steps
        start_idx += horizon

    # Concatenate all predictions
    if predictions_list:
        return concatenate(predictions_list, axis=0)
    else:
        # If no predictions, return empty prediction
        return model.predict(n=horizon)


def evaluate_model(
    model: Any,
    test_series: TimeSeries,
    test_covariates: Optional[TimeSeries] = None,
    test_future_covariates: Optional[TimeSeries] = None,
    lookback: int = 24,
    horizon: int = 6,
    scaler=None,
) -> tuple[dict, TimeSeries]:
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained model
        test_series: Test time series
        test_covariates: Test past covariates (optional)
        test_future_covariates: Test future covariates (optional)
        lookback: Lookback window
        horizon: Forecast horizon
        scaler: Scaler for inverse transform

    Returns:
        Tuple of (metrics_dict, predictions_series)
    """
    predictions = _generate_predictions(
        model, test_series, test_covariates, lookback, horizon,
        future_covariates=test_future_covariates,
    )

    metrics = calculate_metrics(
        test_series[lookback:],
        predictions,
        scaler=scaler,
    )

    return metrics, predictions
