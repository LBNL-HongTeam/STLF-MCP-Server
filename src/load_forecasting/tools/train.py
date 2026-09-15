"""
MCP tool: train_forecast_model — train a forecasting model on historical data.

Also hosts the training-series preparation and model-persistence helpers
(``_prepare_training_series``, ``_save_trained_model``) that are shared with
``tune.py``.
"""

from typing import Optional
import logging

import numpy as np

from ..core.data_loader import ForecastingDataLoader, DataLoadError
from ..core.trainer import (
    train_model as _train_model,
    get_available_models,
    _normalize_device,
    MULTI_SERIES_MODELS,
    PROBABILISTIC_MODELS,
    validate_quantiles,
)
from ..core.frequency_utils import hours_to_steps
from ..core.model_registry import ModelRegistry
from ._common import (
    create_success_response,
    create_error_response,
    _collect_warnings,
    _append_warning,
    _check_horizon_lookback,
    _check_arima_lookback,
    _check_high_step_count,
    _check_overfitting,
)
from .inspection import inspect_data

logger = logging.getLogger(__name__)

# Keyword arguments train_forecast_model computes itself and forwards to the
# trainer.  Allowing a caller to override them through model_kwargs would let
# the tool's own validated arguments be silently contradicted.
_RESERVED_MODEL_KWARGS = frozenset({
    "train_series", "val_series", "model_type", "lookback", "horizon",
    "train_covariates", "val_covariates", "train_future_covariates",
    "val_future_covariates", "scaler", "frequency", "accelerator",
    "probabilistic", "quantiles",
})

# Learning-rate schedulers addressable by name.  MCP arguments arrive as JSON,
# so a scheduler cannot be passed as a Python class; callers name it instead
# and the class is resolved here.  Extend this map rather than importing
# arbitrary attributes by name, which would be an code-execution vector.
_LR_SCHEDULERS = (
    "CosineAnnealingLR", "CosineAnnealingWarmRestarts", "StepLR",
    "MultiStepLR", "ExponentialLR", "ReduceLROnPlateau", "OneCycleLR",
    "LinearLR", "ConstantLR",
)


def _resolve_model_kwargs(model_kwargs: Optional[dict]) -> dict:
    """
    Validate ``model_kwargs`` and resolve JSON-friendly aliases.

    Raises ValueError with an actionable message on bad input.
    """
    if model_kwargs is None:
        return {}
    if not isinstance(model_kwargs, dict):
        raise ValueError(
            f"model_kwargs must be a dict of keyword arguments, got "
            f"{type(model_kwargs).__name__}."
        )

    clashes = sorted(set(model_kwargs) & _RESERVED_MODEL_KWARGS)
    if clashes:
        raise ValueError(
            f"model_kwargs may not override arguments that "
            f"train_forecast_model sets itself: {clashes}. "
            f"Use the tool's own parameters instead (for example device= "
            f"rather than accelerator=, probabilistic= rather than likelihood "
            f"settings)."
        )

    resolved = dict(model_kwargs)

    # Accept a scheduler named as a string, e.g.
    #   {"lr_scheduler_cls": "CosineAnnealingLR",
    #    "lr_scheduler_kwargs": {"T_max": 50}}
    sched = resolved.get("lr_scheduler_cls")
    if isinstance(sched, str):
        if sched not in _LR_SCHEDULERS:
            raise ValueError(
                f"Unknown lr_scheduler_cls '{sched}'. Supported: "
                f"{list(_LR_SCHEDULERS)}."
            )
        try:
            import torch.optim.lr_scheduler as _sched_mod
        except ImportError as e:  # pragma: no cover - torch is a hard dep
            raise ValueError(
                f"lr_scheduler_cls requires PyTorch, which is not "
                f"importable: {e}"
            ) from e
        resolved["lr_scheduler_cls"] = getattr(_sched_mod, sched)

    return resolved


def _apply_gaussian_noise(
    df: "np.ndarray",
    covariate_cols: list,
    noise_std: float,
    rng: "np.random.Generator",
):
    """
    Add zero-mean Gaussian noise to the specified columns of a DataFrame.

    Implements the weather forecast uncertainty simulation described in
    Li et al. (2025) Section 3.1.2:
        "we generated synthetic forecast scenarios by superimposing Gaussian
        noise onto the curated outdoor temperature measurements"

    The noise is applied in the raw (pre-scaling) value space so the MinMax
    scaler sees a slightly perturbed distribution.  Applied only during
    training; evaluation and inference use the unperturbed data.

    Args:
        df: DataFrame whose index is the DatetimeIndex (post-preprocessing).
        covariate_cols: List of column names to perturb.
        noise_std: Standard deviation of the noise in raw covariate units
            (e.g. °C for temperature).
        rng: NumPy random Generator for reproducible noise.

    Returns:
        A copy of df with noise applied to the specified columns.
    """
    df = df.copy()
    for col in covariate_cols:
        if col not in df.columns:
            continue
        noise = rng.normal(loc=0.0, scale=noise_std, size=len(df))
        df[col] = df[col] + noise
    return df


def _prepare_training_series(
    loader,
    validation_split: float,
    post_split_hook=None,
    model_type: str = "LinearRegression",
) -> dict:
    """Split a loader seasonally, fit scalers, and build all Darts series.

    ``post_split_hook`` (if given) is called with the train_loader after the
    split but before scalers are fit — used by train_forecast_model to inject
    weather-noise augmentation into the training DataFrame only.

    Split strategy depends on ``model_type``:

    * **Global models** (``MULTI_SERIES_MODELS``: LinearRegression, XGBoost,
      LSTM, TFT, TiDE, TSMixer) get the seasonal split as a *list* of gap-free
      per-season chunks.  A seasonal split is non-contiguous by construction,
      and collapsing it into one regular-frequency series would interpolate
      straight-line load across the multi-week holes it creates (~16% of a
      training year, ~69% of a validation year at hourly resolution).  Darts
      global models fit on a sequence of series natively, so the gaps are simply
      never bridged.
    * **Local / single-series models** (Naive*, ARIMA, TimesFM*) cannot fit a
      sequence, so they fall back to the sequential (contiguous) split, which
      also introduces no synthetic data.  Seasonal coverage is traded for split
      integrity; this is logged.

    Returns a dict with train_loader, val_loader, train_series, val_series,
    train_covariates, train_future_covariates, full_covariates,
    full_future_covariates, scaler, and split_info.
    """
    use_chunks = model_type in MULTI_SERIES_MODELS

    if use_chunks:
        train_loader, val_loader = loader.split_train_val_seasonal(validation_split)
        split_strategy = "seasonal_chunked"
    else:
        train_loader, val_loader = loader.split_train_val(validation_split)
        split_strategy = "sequential"
        logger.info(
            "%s cannot fit a sequence of series; using a contiguous sequential "
            "split instead of the seasonal split to avoid interpolating across "
            "split-induced gaps.",
            model_type,
        )

    if post_split_hook is not None:
        post_split_hook(train_loader)

    to_series = (
        (lambda ldr, fit: ldr.to_darts_series_chunks(fit_scalers=fit))
        if use_chunks
        else (lambda ldr, fit: ldr.to_darts_series(fit_scalers=fit))
    )

    train_series, train_covariates, train_future_covariates = to_series(
        train_loader, True
    )

    # Propagate fitted scalers to val + full loaders for consistent scaling
    for tgt in (val_loader, loader):
        tgt.target_scaler = train_loader.target_scaler
        tgt.covariate_scaler = train_loader.covariate_scaler
        tgt.future_covariate_scaler = train_loader.future_covariate_scaler

    val_series, _, _ = to_series(val_loader, False)
    # Full-dataset covariates so historical_forecasts can read into the
    # validation window without index out-of-bounds.  ``loader`` is the
    # unsplit frame and is therefore contiguous, so the single-series
    # conversion is correct here regardless of the split strategy.
    _, full_covariates, full_future_covariates = loader.to_darts_series(fit_scalers=False)

    def _n_steps(s):
        return sum(len(x) for x in s) if isinstance(s, (list, tuple)) else len(s)

    split_info = {
        "strategy": split_strategy,
        "train_chunks": len(train_series) if isinstance(train_series, list) else 1,
        "validation_chunks": len(val_series) if isinstance(val_series, list) else 1,
        "train_steps": _n_steps(train_series),
        "validation_steps": _n_steps(val_series),
        "interpolated_train_steps": _n_steps(train_series) - len(train_loader.df),
        "interpolated_validation_steps": _n_steps(val_series) - len(val_loader.df),
    }

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_series": train_series,
        "val_series": val_series,
        "train_covariates": train_covariates,
        "train_future_covariates": train_future_covariates,
        "full_covariates": full_covariates,
        "full_future_covariates": full_future_covariates,
        "scaler": train_loader.target_scaler,
        "split_info": split_info,
    }


def _save_trained_model(
    registry, model, model_id, model_type, building_name, config,
    loader, train_loader, data_summary, training_metrics, validation_metrics,
    training_info=None,
) -> str:
    """Persist a trained model + its scalers to the registry, returning the path."""
    return registry.save_model(
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
        training_info=training_info,
    )


def train_forecast_model(
    csv_path: str,
    model_type: str = "LinearRegression",
    lookback_hours: int = 24,
    horizon_hours: int = 6,
    frequency: str = "h",
    validation_split: float = 0.2,
    building_name: Optional[str] = None,
    model_name: Optional[str] = None,
    column_mapping: Optional[dict] = None,
    augment_weather_noise: bool = False,
    weather_noise_std: float = 1.0,
    device: Optional[str] = None,
    probabilistic: bool = False,
    quantiles: Optional[list] = None,
    model_kwargs: Optional[dict] = None,
) -> dict:
    """
    Train a forecasting model on historical building load data.

    Args:
        csv_path: Path to CSV file with datetime and load data
        model_type: Model type to train
        lookback_hours: Hours of history for model input
        horizon_hours: Hours ahead to forecast (1–96; use 96 to match Li et al. 2025)
        frequency: Data frequency (15min, 30min, h)
        validation_split: Fraction for validation
        building_name: Building identifier
        model_name: Custom model name
        column_mapping: Map CSV columns to roles
        augment_weather_noise: If True, add Gaussian noise to future covariate columns
            during training to simulate weather forecast uncertainty (Li et al. 2025,
            Section 3.1.2). Has no effect when no future_covariates are provided.
        weather_noise_std: Standard deviation of the noise in raw covariate units
            (e.g. °C for temperature). Only used when augment_weather_noise=True.
            Default 1.0 °C matches typical NWP forecast uncertainty.
        device: Compute device for PyTorch-backed models (LSTM, TFT, TiDE,
            TSMixer, TimesFM, TimesFM+Residual). One of "cuda" (NVIDIA GPU),
            "mps" (Apple Silicon GPU), "cpu", or "auto"/None to auto-detect
            (priority: cuda > mps > cpu). If the requested device is unavailable
            the trainer falls back to CPU with a warning. Ignored by CPU-only
            models (LinearRegression, XGBoost, ARIMA, Naive*).
        probabilistic: If True, fit a quantile (probabilistic) model that can
            emit prediction intervals at inference time.  Only supported for
            XGBoost, LSTM, TFT, TiDE, and TSMixer; other model types return an
            error.  Enables pinball-loss / coverage / interval-width metrics in
            evaluate_forecast_model and prediction bands in generate_forecast.
        quantiles: Quantile levels to fit when probabilistic=True (floats in
            (0, 1)).  Defaults to [0.1, 0.5, 0.9] (P10/P50/P90 — a median plus
            an 80% central interval).  0.5 is always included so a median point
            forecast is available.
        model_kwargs: Algorithm-specific settings forwarded to the underlying
            Darts model constructor, for settings the tool does not expose as
            named parameters.  Accepted keys depend on the model type; unknown
            keys raise an error from Darts rather than being ignored.  Common
            uses for the PyTorch models are ``n_epochs``, ``batch_size``,
            ``dropout``, ``optimizer_kwargs``, and early stopping via
            ``pl_trainer_kwargs``.  For XGBoost, ``n_estimators`` and
            ``max_depth``.

            Because MCP arguments arrive as JSON, a learning-rate scheduler is
            named as a string and resolved to the corresponding
            ``torch.optim.lr_scheduler`` class.  For example, the training
            schedule of Li et al. (2025) is:

                model_kwargs={
                    "n_epochs": 50,
                    "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 1e-5},
                    "lr_scheduler_cls": "CosineAnnealingLR",
                    "lr_scheduler_kwargs": {"T_max": 50},
                }

            Keys that train_forecast_model sets itself (``accelerator``,
            ``probabilistic``, ``quantiles``, the series and covariate
            arguments, ``lookback``, ``horizon``, ``frequency``) are rejected;
            use the tool's own parameters for those.  Use tune_model instead
            when the aim is to search over these values rather than to set
            them.

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

        # Validate/resolve compute device (cuda > mps > cpu when auto).
        try:
            resolved_device = _normalize_device(device)
        except ValueError as e:
            return create_error_response(str(e))

        # Validate/resolve algorithm-specific passthrough settings.
        try:
            resolved_model_kwargs = _resolve_model_kwargs(model_kwargs)
        except ValueError as e:
            return create_error_response(str(e))

        # Validate/resolve probabilistic (quantile) configuration.
        resolved_quantiles = None
        if probabilistic:
            if model_type not in PROBABILISTIC_MODELS:
                return create_error_response(
                    f"probabilistic=True is not supported for model type "
                    f"'{model_type}'. Supported: {sorted(PROBABILISTIC_MODELS)}. "
                    "Use one of those, or set probabilistic=False for a point forecast."
                )
            try:
                resolved_quantiles = validate_quantiles(quantiles)
            except ValueError as e:
                return create_error_response(str(e))

        # Validate parameters
        if not 1 <= lookback_hours <= 168:
            return create_error_response("lookback_hours must be between 1 and 168")

        if not 1 <= horizon_hours <= 96:
            return create_error_response("horizon_hours must be between 1 and 96")

        # Collect soft warnings for ML best-practice issues
        ml_warnings = _collect_warnings(
            lambda: _check_horizon_lookback(horizon_hours, lookback_hours),
            lambda: _check_arima_lookback(model_type, lookback_hours),
        )

        if not 0.1 <= validation_split <= 0.3:
            return create_error_response("validation_split must be between 0.1 and 0.3")

        # ------------------------------------------------------------------
        # Pre-flight data inspection — always run before training
        # ------------------------------------------------------------------
        inspection = inspect_data(
            csv_path=csv_path,
            column_mapping=column_mapping,
            frequency=frequency,
        )
        if not inspection.get("success"):
            return create_error_response(
                f"Data inspection failed before training: {inspection.get('error')}"
            )
        if inspection.get("blocking_issues"):
            return create_error_response(
                "Data has blocking issues that must be resolved before training:\n"
                + "\n".join(f"  - {issue}" for issue in inspection["blocking_issues"])
            )

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

        _append_warning(ml_warnings, _check_high_step_count(horizon_steps, frequency))

        # Check minimum data requirements (using steps, not hours)
        min_samples = lookback_steps + horizon_steps + 100  # Need buffer
        if data_summary["total_samples"] < min_samples:
            return create_error_response(
                f"Insufficient data: {data_summary['total_samples']} samples. "
                f"Need at least {min_samples} for lookback={lookback_hours}h "
                f"({lookback_steps} steps), horizon={horizon_hours}h "
                f"({horizon_steps} steps) at frequency={frequency}"
            )

        # Weather noise augmentation (Li et al. 2025, Section 3.1.2): add
        # Gaussian noise to future covariate columns of the *training* frame
        # only.  Validation and evaluation data are never perturbed.
        def _augment(train_loader):
            if not augment_weather_noise:
                return
            fut_cov_cols = train_loader.column_mapping.get("future_covariates", [])
            if fut_cov_cols:
                train_loader.df = _apply_gaussian_noise(
                    train_loader.df,
                    covariate_cols=fut_cov_cols,
                    noise_std=weather_noise_std,
                    rng=np.random.default_rng(),
                )
                logger.info(
                    "Weather noise augmentation applied to %d future covariate "
                    "column(s) with std=%.3f: %s",
                    len(fut_cov_cols), weather_noise_std, fut_cov_cols,
                )
            else:
                logger.info(
                    "augment_weather_noise=True but no future_covariates found — "
                    "noise augmentation skipped."
                )

        # Seasonally-stratified split (falls back to sequential if < 4 seasons)
        # plus scaler fitting and full-covariate construction.
        prep = _prepare_training_series(
            loader, validation_split, post_split_hook=_augment, model_type=model_type
        )
        train_loader = prep["train_loader"]
        val_loader = prep["val_loader"]
        train_series = prep["train_series"]
        val_series = prep["val_series"]
        full_covariates = prep["full_covariates"]
        full_future_covariates = prep["full_future_covariates"]

        # Update data summary with split info
        data_summary["training_samples"] = len(train_loader.df)
        data_summary["validation_samples"] = len(val_loader.df)
        data_summary["split"] = prep["split_info"]

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
            accelerator=resolved_device,
            probabilistic=probabilistic,
            quantiles=resolved_quantiles,
            **resolved_model_kwargs,
        )

        # Save to registry
        registry = ModelRegistry()
        model_id = model_name or registry.generate_model_id(building_name, model_type)

        config = {
            "lookback_hours": lookback_hours,
            "horizon_hours": horizon_hours,
            "frequency": frequency,
            "validation_split": validation_split,
            "probabilistic": probabilistic,
            "quantiles": resolved_quantiles,
            # Stored as supplied by the caller (JSON-safe) rather than after
            # resolution, so the metadata stays serialisable and the recorded
            # value can be fed straight back into a later call.
            "model_kwargs": model_kwargs or None,
        }

        model_path = _save_trained_model(
            registry, model, model_id, model_type, building_name, config,
            loader, train_loader, data_summary, training_metrics, validation_metrics,
            training_info=training_info,
        )

        # Overfitting check
        _append_warning(ml_warnings, _check_overfitting(training_metrics, validation_metrics))

        response = create_success_response(
            model_id=model_id,
            model_path=model_path,
            model_type=model_type,
            training_metrics=training_metrics,
            validation_metrics=validation_metrics,
            data_summary=data_summary,
            training_info=training_info,
            data_inspection={
                "quality_flags": inspection.get("quality_flags", []),
                "suggestions": inspection.get("suggestions", []),
            },
        )
        if ml_warnings:
            response["ml_warnings"] = ml_warnings
        return response

    except Exception as e:
        logger.exception("Training failed")
        return create_error_response(
            f"Training failed: {str(e)}",
            data_summary=data_summary if data_summary else None,
        )
