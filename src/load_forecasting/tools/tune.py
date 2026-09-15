"""
MCP tool: tune_model — Optuna hyperparameter tuning + best-model registration.
"""

from typing import Optional
import logging

from ..core.data_loader import ForecastingDataLoader, DataLoadError
from ..core.trainer import (
    train_model as _train_model,
    create_model as _create_model,
    get_available_models,
    generate_predictions,
    _cast_series_to_float32,
    _normalize_device,
    _eval_split,
    _match_covariates,
    MIXED_COVARIATE_MODELS,
    _TORCH_MODEL_NAMES,
)
from ..core.tuning import DEFAULT_SEARCH_SPACES, sample_params
from ..core.evaluator import calculate_metrics
from ..core.model_registry import ModelRegistry
from ..core.frequency_utils import hours_to_steps
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
from .train import _prepare_training_series, _save_trained_model

logger = logging.getLogger(__name__)


# DEFAULT_SEARCH_SPACES and sample_params live in core/tuning.py (imported above)


def tune_model(
    csv_path: str,
    model_type: str,
    frequency: str = "h",
    lookback_hours: int = 24,
    horizon_hours: int = 6,
    validation_split: float = 0.2,
    n_trials: int = 20,
    search_space: Optional[dict] = None,
    building_name: Optional[str] = None,
    column_mapping: Optional[dict] = None,
    device: Optional[str] = None,
) -> dict:
    """
    Hyperparameter-tune a model using Optuna and register the best result.

    Loads training data once, then runs ``n_trials`` Optuna trials. Each trial
    samples hyperparameters from ``search_space`` (or the built-in default for
    the model type), trains the model on the train split, and scores it on the
    validation split by CV-RMSE (lower is better).

    After all trials the best parameter set is used to train a final model on
    the combined train+val data, which is then saved to the model registry and
    returned as ``model_id``.

    Args:
        csv_path: Path to CSV file with datetime index and load data
        model_type: Model type to tune (XGBoost, LSTM, LinearRegression, …)
        frequency: Data frequency (15min, 30min, h)
        lookback_hours: Hours of history to use as input window
        horizon_hours: Hours ahead to forecast
        validation_split: Fraction of data for validation (0.1–0.3)
        n_trials: Number of Optuna trials (more = better tuning, slower)
        search_space: Custom search space dict. If None the built-in default
            for the model type is used. Keys are hyperparameter names; values
            are dicts with "type" ("int"/"float"/"categorical"), "low"/"high"
            (for int/float), optional "log" (bool, for float), and "choices"
            (for categorical).
        building_name: Building identifier for model naming
        column_mapping: Map CSV columns to roles (datetime, target, …)
        device: Compute device for PyTorch-backed models — one of "cuda"
            (NVIDIA GPU), "mps" (Apple Silicon GPU), "cpu", or "auto"/None to
            auto-detect (priority: cuda > mps > cpu). Applied to every trial and
            to the final model training. Ignored by CPU-only models.

    Returns:
        Dict with best_params, best_cv_rmse, all_trial_results, model_id,
        and validation_metrics of the final model.
    """
    import optuna

    # Silence Optuna's default INFO/WARNING chatter
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    try:
        # ------------------------------------------------------------------
        # Validate inputs
        # ------------------------------------------------------------------
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

        if not 1 <= lookback_hours <= 168:
            return create_error_response("lookback_hours must be between 1 and 168")

        if not 1 <= horizon_hours <= 48:
            return create_error_response("horizon_hours must be between 1 and 48")

        if not 0.1 <= validation_split <= 0.3:
            return create_error_response("validation_split must be between 0.1 and 0.3")

        if n_trials < 1:
            return create_error_response("n_trials must be at least 1")

        # Collect soft ML warnings (evaluated before the expensive tuning run)
        tune_ml_warnings = _collect_warnings(
            lambda: _check_horizon_lookback(horizon_hours, lookback_hours),
            lambda: _check_arima_lookback(model_type, lookback_hours),
        )

        # Resolve search space
        effective_space = search_space if search_space is not None else DEFAULT_SEARCH_SPACES.get(model_type, {})
        if not effective_space:
            if model_type == "TimesFM":
                return create_error_response(
                    "TimesFM is not tunable via this interface — the foundation "
                    "model has no meaningful hyperparameter search space. Use "
                    "train_forecast_model with a fixed configuration instead, "
                    "or pass an explicit search_space to override."
                )
            if model_type == "TimesFM+Residual":
                return create_error_response(
                    "TimesFM+Residual is not tunable via this interface — the "
                    "TimesFM backbone has no meaningful hyperparameter search "
                    "space, and the Ridge residual regressor exposes only a "
                    "single scalar (alpha) that is not currently searched. "
                    "Use train_forecast_model with a fixed configuration "
                    "instead, or pass an explicit search_space to override."
                )
            return create_error_response(
                f"No search space defined for model type '{model_type}'. "
                "Provide an explicit search_space or choose XGBoost, LSTM, ARIMA, TFT, TiDE, or TSMixer."
            )

        # ------------------------------------------------------------------
        # Pre-flight data inspection — always run before tuning
        # ------------------------------------------------------------------
        inspection = inspect_data(
            csv_path=csv_path,
            column_mapping=column_mapping,
            frequency=frequency,
        )
        if not inspection.get("success"):
            return create_error_response(
                f"Data inspection failed before tuning: {inspection.get('error')}"
            )
        if inspection.get("blocking_issues"):
            return create_error_response(
                "Data has blocking issues that must be resolved before tuning:\n"
                + "\n".join(f"  - {issue}" for issue in inspection["blocking_issues"])
            )

        # ------------------------------------------------------------------
        # Load data once — shared across all trials
        # ------------------------------------------------------------------
        try:
            loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=column_mapping,
                frequency=frequency,
            )
        except DataLoadError as e:
            return create_error_response(str(e))

        lookback_steps = hours_to_steps(lookback_hours, frequency)
        horizon_steps = hours_to_steps(horizon_hours, frequency)

        _append_warning(tune_ml_warnings, _check_high_step_count(horizon_steps, frequency))

        data_summary = loader.get_data_summary()
        min_samples = lookback_steps + horizon_steps + 100
        if data_summary["total_samples"] < min_samples:
            return create_error_response(
                f"Insufficient data: {data_summary['total_samples']} samples. "
                f"Need at least {min_samples}."
            )

        # Seasonally-stratified split (falls back to sequential if < 4 seasons)
        # plus scaler fitting and full-covariate construction.
        prep = _prepare_training_series(loader, validation_split, model_type=model_type)
        train_loader = prep["train_loader"]
        train_series = prep["train_series"]
        val_series = prep["val_series"]
        train_covariates = prep["train_covariates"]
        train_future_covariates = prep["train_future_covariates"]
        full_covariates = prep["full_covariates"]
        full_future_covariates = prep["full_future_covariates"]
        scaler = prep["scaler"]

        logger.info(
            f"tune_model: {model_type}, {n_trials} trials, "
            f"lookback={lookback_hours}h ({lookback_steps} steps), "
            f"horizon={horizon_hours}h ({horizon_steps} steps)"
        )

        # ------------------------------------------------------------------
        # Optuna objective
        # ------------------------------------------------------------------
        trial_results: list[dict] = []

        def objective(trial) -> float:
            params = sample_params(trial, effective_space)
            # Apply the resolved compute device unless the search space is
            # explicitly tuning the accelerator (rare).
            if model_type in _TORCH_MODEL_NAMES and "accelerator" not in params:
                params["accelerator"] = resolved_device

            try:
                use_past = train_covariates is not None
                use_future = (
                    train_future_covariates is not None
                    and model_type in MIXED_COVARIATE_MODELS
                )
                model = _create_model(
                    model_type,
                    lookback=lookback_steps,
                    horizon=horizon_steps,
                    use_past_covariates=use_past,
                    use_future_covariates=use_future,
                    frequency=frequency,
                    **params,
                )

                # GPU accelerators (mps/cuda) run in float32; cast series to
                # float32 to avoid dtype mismatches (MPS lacks float64 support;
                # CUDA errors on double vs. float mismatch).
                _trial_accelerator = _normalize_device(params.get("accelerator"))
                _trial_use_gpu = (
                    model_type in _TORCH_MODEL_NAMES
                    and _trial_accelerator in ("mps", "cuda")
                )
                _fit_series = (
                    _cast_series_to_float32(train_series) if _trial_use_gpu
                    else train_series
                )
                _fit_covariates = (
                    _cast_series_to_float32(full_covariates) if _trial_use_gpu
                    else full_covariates
                )
                _fit_future_covariates = (
                    _cast_series_to_float32(full_future_covariates) if _trial_use_gpu
                    else full_future_covariates
                )

                # Fit on train split.  A seasonal split yields a list of
                # gap-free chunks; broadcast the covariates to match it.
                fit_kwargs = {}
                _n = len(_fit_series) if isinstance(_fit_series, list) else None
                if train_covariates is not None:
                    fit_kwargs["past_covariates"] = _match_covariates(_fit_covariates, _n) \
                        if _n else _fit_covariates
                if train_future_covariates is not None and use_future:
                    fit_kwargs["future_covariates"] = _match_covariates(_fit_future_covariates, _n) \
                        if _n else _fit_future_covariates
                model.fit(_fit_series, **fit_kwargs)

                # Score on validation split (pooled across chunks by _eval_split)
                metrics = _eval_split(
                    model, val_series, full_covariates, full_future_covariates,
                    lookback_steps, horizon_steps, model_type, scaler, "validation",
                )
                cv_rmse = metrics.get("cv_rmse") or float("inf")

            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                cv_rmse = float("inf")
                params = {}

            trial_results.append({
                "trial": trial.number,
                "params": params,
                "cv_rmse": cv_rmse if cv_rmse != float("inf") else None,
            })
            return cv_rmse

        # ------------------------------------------------------------------
        # Run study (in-memory storage — no SQLite files)
        # ------------------------------------------------------------------
        study = optuna.create_study(direction="minimize", storage=None)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_cv_rmse = study.best_value
        if best_cv_rmse == float("inf"):
            return create_error_response(
                "All tuning trials failed. Check data and model configuration."
            )

        logger.info(f"tune_model best: cv_rmse={best_cv_rmse:.4f}, params={best_params}")

        # If the user tuned 'accelerator' explicitly via search_space, honour the
        # tuned value and drop the tool-level device to avoid a duplicate kwarg.
        _final_device = best_params.pop("accelerator", resolved_device)

        # ------------------------------------------------------------------
        # Train final model with best params (on train split, evaluate on val)
        # ------------------------------------------------------------------
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
            scaler=scaler,
            frequency=frequency,
            accelerator=_final_device,
            **best_params,
        )

        # ------------------------------------------------------------------
        # Save to registry
        # ------------------------------------------------------------------
        registry = ModelRegistry()
        model_id = registry.generate_model_id(building_name, f"{model_type}_tuned")

        config = {
            "lookback_hours": lookback_hours,
            "horizon_hours": horizon_hours,
            "frequency": frequency,
            "validation_split": validation_split,
            "tuned": True,
            "n_trials": n_trials,
            "best_hyperparameters": best_params,
        }

        model_path = _save_trained_model(
            registry, model, model_id, model_type, building_name, config,
            loader, train_loader, data_summary, training_metrics, validation_metrics,
            training_info=training_info,
        )

        # Overfitting check on the final model
        _append_warning(tune_ml_warnings, _check_overfitting(training_metrics, validation_metrics))

        # Always note that tune_model validation metrics are hyperparameter-selection-biased
        tune_ml_warnings.append(
            "Validation metrics are measured on the same split used to select "
            "hyperparameters (Optuna trials). They are optimistic. "
            "Run evaluate_forecast_model on a fully held-out CSV for unbiased metrics."
        )

        response = create_success_response(
            model_id=model_id,
            model_path=model_path,
            model_type=model_type,
            best_params=best_params,
            best_cv_rmse=round(best_cv_rmse, 6),
            n_trials_completed=len(trial_results),
            all_trial_results=trial_results,
            validation_metrics=validation_metrics,
            training_metrics=training_metrics,
            training_info=training_info,
            data_summary=data_summary,
            data_inspection={
                "quality_flags": inspection.get("quality_flags", []),
                "suggestions": inspection.get("suggestions", []),
            },
        )
        if tune_ml_warnings:
            response["ml_warnings"] = tune_ml_warnings
        return response

    except Exception as e:
        logger.exception("tune_model failed")
        return create_error_response(f"Tuning failed: {str(e)}")
