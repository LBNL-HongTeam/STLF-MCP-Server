"""
Tests for Subtask 2 model development: XGBoost, LSTM, ARIMA, TFT, TiDE,
TSMixer, and tune_model.

Covers:
- XGBoost training and evaluation (with and without covariates)
- LSTM training and evaluation
- ARIMA training and evaluation (with and without future covariates)
- TFT training and evaluation (tiny hyperparams for speed)
- TiDE training and evaluation (tiny hyperparams for speed)
- TSMixer training and evaluation (tiny hyperparams for speed)
- tune_model with XGBoost (fast, n_trials=2)
- tune_model with LSTM (fast, tiny n_epochs)
- Error paths: unknown model type, empty search space for non-supported type
- trainer.py unit tests: create_model, covariate gating
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.tools import (
    train_forecast_model,
    evaluate_forecast_model,
    tune_model,
)
from load_forecasting.core.trainer import (
    create_model,
    get_available_models,
    MIXED_COVARIATE_MODELS,
    FUTURE_ONLY_MODELS,
    PAST_COVARIATE_ONLY_MODELS,
)
# Backwards-compatible aliases used by the tests below
_MIXED_COVARIATE_MODELS = MIXED_COVARIATE_MODELS
_FUTURE_ONLY_MODELS = FUTURE_ONLY_MODELS
_PAST_COVARIATE_ONLY_MODELS = PAST_COVARIATE_ONLY_MODELS
from load_forecasting.core.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_hourly_csv(n_points: int = 600, with_covariates: bool = True) -> str:
    """Write a synthetic hourly CSV to a temp file and return the path."""
    import numpy as np

    rng = np.random.default_rng(42)
    t = pd.date_range("2023-01-01", periods=n_points, freq="h")

    load = (
        100
        + 20 * np.sin(2 * np.pi * t.hour / 24)      # daily cycle
        + 5 * np.sin(2 * np.pi * t.dayofweek / 7)   # weekly cycle
        + rng.normal(0, 2, n_points)
    )

    data: dict = {
        "timestamp": t,
        "electricity_kwh": load,
    }
    if with_covariates:
        data["outdoor_temp"] = 15 + 10 * np.sin(2 * np.pi * t.hour / 24) + rng.normal(0, 1, n_points)

    df = pd.DataFrame(data)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


@pytest.fixture
def hourly_csv():
    """Hourly CSV with outdoor_temp covariate."""
    path = _make_hourly_csv(n_points=600, with_covariates=True)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def hourly_csv_no_cov():
    """Hourly CSV without covariates."""
    path = _make_hourly_csv(n_points=600, with_covariates=False)
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_dir(monkeypatch, tmp_path):
    """Isolated model registry backed by a temp directory."""
    monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", str(tmp_path))
    # Clear the in-memory registry cache so the new dir takes effect
    ModelRegistry._registry_cache.clear()
    yield str(tmp_path)
    ModelRegistry._registry_cache.clear()


# ---------------------------------------------------------------------------
# Unit tests: trainer.create_model
# ---------------------------------------------------------------------------

class TestCreateModel:
    def test_available_models_includes_xgboost_and_lstm(self):
        models = get_available_models()
        assert "XGBoost" in models
        assert "LSTM" in models

    def test_available_models_includes_arima_and_tft(self):
        models = get_available_models()
        assert "ARIMA" in models
        assert "TFT" in models

    def test_available_models_includes_tide_and_tsmixer(self):
        models = get_available_models()
        assert "TiDE" in models
        assert "TSMixer" in models

    def test_create_xgboost_returns_correct_type(self):
        from darts.models import XGBModel
        m = create_model("XGBoost", lookback=24, horizon=6)
        assert isinstance(m, XGBModel)

    def test_create_lstm_returns_block_rnn(self):
        from darts.models import BlockRNNModel
        m = create_model("LSTM", lookback=12, horizon=3)
        assert isinstance(m, BlockRNNModel)

    def test_create_arima_returns_correct_type(self):
        from darts.models import ARIMA
        m = create_model("ARIMA", lookback=24, horizon=6)
        assert isinstance(m, ARIMA)
        # Default params
        assert m.order == (1, 1, 0)

    def test_create_arima_model_kwargs_applied(self):
        from darts.models import ARIMA
        m = create_model("ARIMA", lookback=24, horizon=6, p=3, d=0, q=2)
        assert isinstance(m, ARIMA)
        assert m.order == (3, 0, 2)

    def test_create_tft_returns_correct_type(self):
        from darts.models import TFTModel
        m = create_model("TFT", lookback=12, horizon=3, hidden_size=8, n_epochs=2)
        assert isinstance(m, TFTModel)
        assert m.input_chunk_length == 12
        assert m.output_chunk_length == 3

    def test_create_tide_returns_correct_type(self):
        from darts.models import TiDEModel
        m = create_model("TiDE", lookback=12, horizon=3, hidden_size=8, n_epochs=2)
        assert isinstance(m, TiDEModel)
        assert m.input_chunk_length == 12
        assert m.output_chunk_length == 3

    def test_tide_paper_defaults(self):
        """Regression guard: TiDE defaults must match Li et al. (2025), Table A.3."""
        from darts.models import TiDEModel
        m = create_model("TiDE", lookback=24, horizon=6)
        assert isinstance(m, TiDEModel)
        # These are stored on the model instance after construction.
        assert m.hidden_size == 128, "hidden_size default should be 128 (Li et al. 2025, Table A.3)"
        assert m.num_encoder_layers == 1
        assert m.num_decoder_layers == 1
        assert m.decoder_output_dim == 16
        assert m.temporal_width_past == 4
        assert m.temporal_width_future == 4
        assert m.dropout == 0.1
        # Verify overrides propagate.
        m2 = create_model("TiDE", lookback=24, horizon=6, hidden_size=64, dropout=0.2)
        assert m2.hidden_size == 64
        assert m2.dropout == 0.2

    def test_create_tsmixer_returns_correct_type(self):
        from darts.models import TSMixerModel
        m = create_model("TSMixer", lookback=12, horizon=3, hidden_size=8, n_epochs=2)
        assert isinstance(m, TSMixerModel)
        assert m.input_chunk_length == 12
        assert m.output_chunk_length == 3

    def test_tsmixer_paper_defaults(self):
        """Regression guard: TSMixer defaults must match Li et al. (2025), Table A.3."""
        from darts.models import TSMixerModel
        m = create_model("TSMixer", lookback=24, horizon=6)
        assert isinstance(m, TSMixerModel)
        assert m.hidden_size == 64, "hidden_size default should be 64 (Li et al. 2025, Table A.3)"
        assert m.ff_size == 64
        assert m.num_blocks == 2
        assert m.activation == "ReLU"
        assert m.dropout == 0.1
        assert m.norm_type == "LayerNorm"
        # Verify overrides propagate.
        m2 = create_model("TSMixer", lookback=24, horizon=6, hidden_size=128, num_blocks=3)
        assert m2.hidden_size == 128
        assert m2.num_blocks == 3

    def test_xgboost_model_kwargs_applied(self):
        from darts.models import XGBModel
        m = create_model("XGBoost", lookback=12, horizon=3, n_estimators=77, max_depth=4)
        assert isinstance(m, XGBModel)

    def test_xgboost_paper_defaults(self):
        """Regression guard: XGBoost defaults must match Li et al. (2025) Table A.3."""
        m = create_model("XGBoost", lookback=12, horizon=3)
        assert m.kwargs.get("n_estimators") == 40, (
            "n_estimators default should be 40 (Li et al. 2025, Table A.3)"
        )
        assert m.kwargs.get("max_depth") == 6, (
            "max_depth default should be 6 (Li et al. 2025, Table A.3)"
        )
        assert m.kwargs.get("booster") == "gbtree", (
            "booster default should be 'gbtree' (Li et al. 2025, Table A.3)"
        )
        # Verify explicit kwargs still override the defaults
        m2 = create_model("XGBoost", lookback=12, horizon=3, n_estimators=77, max_depth=4)
        assert m2.kwargs.get("n_estimators") == 77
        assert m2.kwargs.get("max_depth") == 4

    def test_lstm_model_kwargs_applied(self):
        from darts.models import BlockRNNModel
        m = create_model("LSTM", lookback=12, horizon=3, hidden_dim=16, n_rnn_layers=1, n_epochs=2)
        assert isinstance(m, BlockRNNModel)

    def test_mixed_covariate_set_contains_xgboost_lstm_and_tft(self):
        assert "XGBoost" in _MIXED_COVARIATE_MODELS
        assert "LSTM" in _MIXED_COVARIATE_MODELS
        assert "TFT" in _MIXED_COVARIATE_MODELS

    def test_mixed_covariate_set_contains_tide_and_tsmixer(self):
        assert "TiDE" in _MIXED_COVARIATE_MODELS
        assert "TSMixer" in _MIXED_COVARIATE_MODELS

    def test_torch_model_set_contains_tide_and_tsmixer(self):
        from load_forecasting.core.trainer import _TORCH_MODEL_NAMES
        assert "TiDE" in _TORCH_MODEL_NAMES
        assert "TSMixer" in _TORCH_MODEL_NAMES

    def test_future_only_set_contains_arima(self):
        assert "ARIMA" in _FUTURE_ONLY_MODELS
        # ARIMA must NOT be in the mixed set (it has no past covariate support)
        assert "ARIMA" not in _MIXED_COVARIATE_MODELS

    def test_lstm_in_past_covariate_only_set(self):
        # LSTM (BlockRNNModel) accepts past_covariates but silently ignores
        # future_covariates, so it belongs in PAST_COVARIATE_ONLY_MODELS.
        assert "LSTM" in _PAST_COVARIATE_ONLY_MODELS

    def test_create_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model("BogusModel")


# ---------------------------------------------------------------------------
# Integration: XGBoost training and evaluation
# ---------------------------------------------------------------------------

class TestXGBoostTrainEvaluate:
    async def test_train_xgboost_basic(self, hourly_csv, temp_model_dir):
        result = train_forecast_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_xgb",
        )

        assert result["success"] is True, result.get("error")
        assert result["model_type"] == "XGBoost"
        assert result["model_id"] is not None
        assert "test_xgb" in result["model_id"]
        assert result["validation_metrics"]["cv_rmse"] is not None
        assert result["validation_metrics"]["cv_rmse"] > 0

    async def test_train_xgboost_with_past_covariates(self, hourly_csv, temp_model_dir):
        result = train_forecast_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
            lookback_hours=24,
            horizon_hours=6,
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "past_covariates": ["outdoor_temp"],
            },
        )

        assert result["success"] is True, result.get("error")
        assert result["validation_metrics"]["cv_rmse"] is not None

    async def test_train_xgboost_no_covariates(self, hourly_csv_no_cov, temp_model_dir):
        result = train_forecast_model(
            csv_path=hourly_csv_no_cov,
            model_type="XGBoost",
            lookback_hours=12,
            horizon_hours=3,
        )

        assert result["success"] is True, result.get("error")

    async def test_evaluate_xgboost(self, hourly_csv, temp_model_dir):
        train_result = train_forecast_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train_result["success"] is True, train_result.get("error")

        eval_result = evaluate_forecast_model(
            model_id=train_result["model_id"],
            csv_path=hourly_csv,
            return_predictions=True,
        )

        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["cv_rmse"] is not None
        assert len(eval_result["predictions"]) > 0

    async def test_xgboost_model_id_in_registry(self, hourly_csv, temp_model_dir):
        result = train_forecast_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
        )
        assert result["success"] is True

        registry = ModelRegistry()
        models, total = registry.list_models(model_type="XGBoost")
        assert total >= 1
        assert any(m["model_id"] == result["model_id"] for m in models)


# ---------------------------------------------------------------------------
# Integration: LSTM training and evaluation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestLSTMTrainEvaluate:
    """
    LSTM tests use tiny hyperparameters (hidden_dim=8, n_epochs=2) to keep
    runtime acceptable in CI.  The column_mapping explicitly sets the model
    kwargs via the model-level API — we cannot pass them through
    train_forecast_model directly, so we call the trainer layer.
    """

    async def test_train_lstm_basic(self, hourly_csv, temp_model_dir):
        """Train LSTM with minimal epochs via the full MCP tool stack."""
        # We inject small n_epochs via a monkey-patch on create_model default
        # by testing through the trainer directly rather than the MCP tool,
        # since the MCP tool doesn't expose model_kwargs.
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, _ = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, _, _ = val_loader.to_darts_series(fit_scalers=False)

        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, _ = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        model, train_metrics, val_metrics, info = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="LSTM",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,
            val_covariates=full_cov,
            scaler=train_loader.target_scaler,
            frequency="h",
            # small kwargs to keep the test fast
            hidden_dim=8,
            n_rnn_layers=1,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None
        assert info["darts_model_class"] == "darts.models.BlockRNNModel"

    async def test_lstm_ignores_future_covariates(self, hourly_csv, temp_model_dir):
        """Verify LSTM fit does not receive future_covariates (no error)."""
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, train_fut = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, _, _ = val_loader.to_darts_series(fit_scalers=False)

        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, full_fut = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        # Pass future covariates — LSTM must silently ignore them
        model, _, val_metrics, _ = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="LSTM",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,
            val_covariates=full_cov,
            train_future_covariates=full_fut,   # should be ignored
            val_future_covariates=full_fut,     # should be ignored
            scaler=train_loader.target_scaler,
            frequency="h",
            hidden_dim=8,
            n_rnn_layers=1,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None


# ---------------------------------------------------------------------------
# Integration: tune_model
# ---------------------------------------------------------------------------

class TestTuneModel:
    async def test_tune_xgboost_runs(self, hourly_csv, temp_model_dir):
        """Tune XGBoost with n_trials=2 — verifies end-to-end flow."""
        result = tune_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
            frequency="h",
            lookback_hours=24,
            horizon_hours=6,
            n_trials=2,
            building_name="tune_test",
        )

        assert result["success"] is True, result.get("error")
        assert result["model_type"] == "XGBoost"
        assert result["model_id"] is not None
        assert "_tuned" in result["model_id"]
        assert result["best_params"] is not None
        assert result["best_cv_rmse"] is not None
        assert result["best_cv_rmse"] > 0
        assert result["n_trials_completed"] == 2
        assert len(result["all_trial_results"]) == 2

    async def test_tune_xgboost_custom_search_space(self, hourly_csv, temp_model_dir):
        """Tune XGBoost with a narrowed custom search space."""
        space = {
            "n_estimators": {"type": "categorical", "choices": [50, 100]},
            "max_depth": {"type": "int", "low": 3, "high": 4},
        }
        result = tune_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
            n_trials=2,
            search_space=space,
        )

        assert result["success"] is True, result.get("error")
        assert "n_estimators" in result["best_params"]
        assert result["best_params"]["n_estimators"] in [50, 100]

    @pytest.mark.slow
    async def test_tune_lstm_runs(self, hourly_csv, temp_model_dir):
        """Tune LSTM with tiny search space and n_trials=2."""
        space = {
            "hidden_dim": {"type": "categorical", "choices": [8, 16]},
            "n_rnn_layers": {"type": "categorical", "choices": [1]},
            "n_epochs": {"type": "categorical", "choices": [2]},
            "dropout": {"type": "categorical", "choices": [0.0]},
        }
        result = tune_model(
            csv_path=hourly_csv,
            model_type="LSTM",
            lookback_hours=12,
            horizon_hours=3,
            n_trials=2,
            search_space=space,
            building_name="lstm_tune",
        )

        assert result["success"] is True, result.get("error")
        assert result["model_type"] == "LSTM"
        assert "_tuned" in result["model_id"]
        assert result["best_cv_rmse"] is not None

    async def test_tune_model_unknown_type_returns_error(self, hourly_csv, temp_model_dir):
        result = tune_model(
            csv_path=hourly_csv,
            model_type="BogusModel",
            n_trials=1,
        )
        assert result["success"] is False
        assert "Unknown model type" in result["error"]

    async def test_tune_model_empty_search_space_returns_error(self, hourly_csv, temp_model_dir):
        """LinearRegression has no built-in search space and no custom one."""
        result = tune_model(
            csv_path=hourly_csv,
            model_type="LinearRegression",
            n_trials=2,
        )
        assert result["success"] is False
        assert "search space" in result["error"].lower()

    async def test_tune_model_result_is_registered(self, hourly_csv, temp_model_dir):
        """The tuned model must appear in the registry under its model_id."""
        result = tune_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
            n_trials=2,
        )
        assert result["success"] is True

        registry = ModelRegistry()
        model, metadata, _ = registry.load_model(result["model_id"])
        assert metadata["model_type"] == "XGBoost"
        assert metadata["config"].get("tuned") is True
        assert metadata["config"]["best_hyperparameters"] == result["best_params"]

    async def test_tune_xgboost_with_covariates(self, hourly_csv, temp_model_dir):
        """Tune XGBoost using past covariates."""
        result = tune_model(
            csv_path=hourly_csv,
            model_type="XGBoost",
            n_trials=2,
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "past_covariates": ["outdoor_temp"],
            },
        )
        assert result["success"] is True, result.get("error")


# ---------------------------------------------------------------------------
# Integration: ARIMA training and evaluation
# ---------------------------------------------------------------------------

class TestARIMATrainEvaluate:
    """
    ARIMA tests use default (p=1, d=1, q=0) to keep runtime acceptable.
    ARIMA ignores lookback_hours and past covariates; these are passed but silently
    skipped by the covariate-gating logic in trainer.py.
    """

    async def test_train_arima_basic(self, hourly_csv_no_cov, temp_model_dir):
        """Train ARIMA without covariates via the MCP tool stack."""
        result = train_forecast_model(
            csv_path=hourly_csv_no_cov,
            model_type="ARIMA",
            lookback_hours=24,
            horizon_hours=6,
            building_name="test_arima",
        )

        assert result["success"] is True, result.get("error")
        assert result["model_type"] == "ARIMA"
        assert result["model_id"] is not None
        assert "test_arima" in result["model_id"]
        assert result["validation_metrics"]["cv_rmse"] is not None
        assert result["validation_metrics"]["cv_rmse"] > 0

    async def test_train_arima_with_future_covariates(self, hourly_csv, temp_model_dir):
        """ARIMA uses future covariates as exogenous variables."""
        result = train_forecast_model(
            csv_path=hourly_csv,
            model_type="ARIMA",
            lookback_hours=24,
            horizon_hours=6,
            column_mapping={
                "datetime": "timestamp",
                "target": "electricity_kwh",
                "future_covariates": ["outdoor_temp"],
            },
        )

        assert result["success"] is True, result.get("error")
        assert result["model_type"] == "ARIMA"
        assert result["validation_metrics"]["cv_rmse"] is not None

    async def test_evaluate_arima(self, hourly_csv_no_cov, temp_model_dir):
        """Round-trip: train ARIMA then evaluate on same data."""
        train_result = train_forecast_model(
            csv_path=hourly_csv_no_cov,
            model_type="ARIMA",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert train_result["success"] is True, train_result.get("error")

        eval_result = evaluate_forecast_model(
            model_id=train_result["model_id"],
            csv_path=hourly_csv_no_cov,
            return_predictions=True,
        )

        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["cv_rmse"] is not None
        assert len(eval_result["predictions"]) > 0

    async def test_arima_in_registry(self, hourly_csv_no_cov, temp_model_dir):
        """Trained ARIMA model must appear in the registry."""
        result = train_forecast_model(
            csv_path=hourly_csv_no_cov,
            model_type="ARIMA",
        )
        assert result["success"] is True

        registry = ModelRegistry()
        models, total = registry.list_models(model_type="ARIMA")
        assert total >= 1
        assert any(m["model_id"] == result["model_id"] for m in models)


# ---------------------------------------------------------------------------
# Integration: TFT training and evaluation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTFTTrainEvaluate:
    """
    TFT tests use tiny hyperparameters (hidden_size=8, n_epochs=2) via the
    trainer layer to keep runtime acceptable.  add_relative_index=True is
    always set inside create_model so future covariates are optional.
    """

    async def test_train_tft_basic(self, hourly_csv, temp_model_dir):
        """Train TFT with minimal hyperparams via the trainer layer."""
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, _ = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, val_cov, _ = val_loader.to_darts_series(fit_scalers=False)

        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, _ = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        model, train_metrics, val_metrics, info = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="TFT",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,
            val_covariates=full_cov,
            scaler=train_loader.target_scaler,
            frequency="h",
            # tiny kwargs to keep the test fast
            hidden_size=8,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None
        assert info["darts_model_class"] == "darts.models.TFTModel"

    async def test_evaluate_tft(self, hourly_csv, temp_model_dir):
        """Round-trip: train TFT via trainer layer (with calendar/lag past covariates
        matching what train_forecast_model generates), save to registry, then evaluate
        via the MCP tool so the same covariate pipeline is applied at both ends."""
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        # Mirror what train_forecast_model does: load via ForecastingDataLoader so
        # that calendar and lag features are added as past_covariates automatically.
        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, _ = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, _, _ = val_loader.to_darts_series(fit_scalers=False)

        # Full-dataset covariates for historical_forecasts window
        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, _ = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        model, _, val_metrics, _ = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="TFT",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,   # calendar + lag features
            val_covariates=full_cov,
            scaler=train_loader.target_scaler,
            frequency="h",
            hidden_size=8,
            lstm_layers=1,
            num_attention_heads=2,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None

        # Save using loader.column_mapping so the eval loader uses the same
        # set of auto-generated columns (same as train_forecast_model does).
        reg = ModelRegistry()
        model_id = reg.generate_model_id("tft_test", "TFT")
        reg.save_model(
            model=model,
            model_id=model_id,
            model_type="TFT",
            building_name="tft_test",
            config={"lookback_hours": lookback, "horizon_hours": horizon, "frequency": "h"},
            column_mapping=loader.column_mapping,  # includes cal_* and lag_* columns
            data_info={},
            training_metrics={},
            validation_metrics=val_metrics,
            scalers={
                "target_scaler": train_loader.target_scaler,
                "covariate_scaler": train_loader.covariate_scaler,
            },
        )

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=hourly_csv,
            return_predictions=True,
        )

        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["cv_rmse"] is not None
        assert len(eval_result["predictions"]) > 0


# ---------------------------------------------------------------------------
# Integration: TiDE training and evaluation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTiDETrainEvaluate:
    """
    TiDE tests use tiny hyperparameters (hidden_size=8, n_epochs=2) via the
    trainer layer to keep runtime acceptable. TiDE is a MixedCovariatesTorchModel
    (MLP encoder-decoder) that supports past and future covariates.
    """

    async def test_train_tide_basic(self, hourly_csv, temp_model_dir):
        """Train TiDE with minimal hyperparams via the trainer layer."""
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, _ = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, val_cov, _ = val_loader.to_darts_series(fit_scalers=False)

        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, _ = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        model, train_metrics, val_metrics, info = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="TiDE",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,
            val_covariates=full_cov,
            scaler=train_loader.target_scaler,
            frequency="h",
            # tiny kwargs to keep the test fast
            hidden_size=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            decoder_output_dim=4,
            temporal_width_past=2,
            temporal_width_future=2,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None
        assert info["darts_model_class"] == "darts.models.TiDEModel"

    async def test_evaluate_tide(self, hourly_csv, temp_model_dir):
        """Round-trip: train TiDE via trainer layer, save via registry, evaluate
        via the MCP tool so the same covariate pipeline is applied at both ends."""
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, _ = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, _, _ = val_loader.to_darts_series(fit_scalers=False)

        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, _ = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        model, _, val_metrics, _ = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="TiDE",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,
            val_covariates=full_cov,
            scaler=train_loader.target_scaler,
            frequency="h",
            hidden_size=8,
            num_encoder_layers=1,
            num_decoder_layers=1,
            decoder_output_dim=4,
            temporal_width_past=2,
            temporal_width_future=2,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None

        reg = ModelRegistry()
        model_id = reg.generate_model_id("tide_test", "TiDE")
        reg.save_model(
            model=model,
            model_id=model_id,
            model_type="TiDE",
            building_name="tide_test",
            config={"lookback_hours": lookback, "horizon_hours": horizon, "frequency": "h"},
            column_mapping=loader.column_mapping,
            data_info={},
            training_metrics={},
            validation_metrics=val_metrics,
            scalers={
                "target_scaler": train_loader.target_scaler,
                "covariate_scaler": train_loader.covariate_scaler,
            },
        )

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=hourly_csv,
            return_predictions=True,
        )

        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["cv_rmse"] is not None
        assert len(eval_result["predictions"]) > 0


# ---------------------------------------------------------------------------
# Integration: TSMixer training and evaluation
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestTSMixerTrainEvaluate:
    """
    TSMixer tests use tiny hyperparameters (hidden_size=8, n_epochs=2) via the
    trainer layer to keep runtime acceptable. TSMixer is a MixedCovariatesTorchModel
    (all-MLP token-mixer) that supports past and future covariates.
    """

    async def test_train_tsmixer_basic(self, hourly_csv, temp_model_dir):
        """Train TSMixer with minimal hyperparams via the trainer layer."""
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, _ = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, val_cov, _ = val_loader.to_darts_series(fit_scalers=False)

        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, _ = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        model, train_metrics, val_metrics, info = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="TSMixer",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,
            val_covariates=full_cov,
            scaler=train_loader.target_scaler,
            frequency="h",
            # tiny kwargs to keep the test fast
            hidden_size=8,
            ff_size=8,
            num_blocks=1,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None
        assert info["darts_model_class"] == "darts.models.TSMixerModel"

    async def test_evaluate_tsmixer(self, hourly_csv, temp_model_dir):
        """Round-trip: train TSMixer via trainer layer, save via registry, evaluate
        via the MCP tool so the same covariate pipeline is applied at both ends."""
        from load_forecasting.core.trainer import train_model as _train_model
        from load_forecasting.core.data_loader import ForecastingDataLoader
        from load_forecasting.core.frequency_utils import hours_to_steps

        loader = ForecastingDataLoader(csv_path=hourly_csv, frequency="h")
        train_loader, val_loader = loader.split_train_val(0.2)

        train_series, train_cov, _ = train_loader.to_darts_series(fit_scalers=True)
        val_loader.target_scaler = train_loader.target_scaler
        val_loader.covariate_scaler = train_loader.covariate_scaler
        val_loader.future_covariate_scaler = train_loader.future_covariate_scaler
        val_series, _, _ = val_loader.to_darts_series(fit_scalers=False)

        loader.target_scaler = train_loader.target_scaler
        loader.covariate_scaler = train_loader.covariate_scaler
        loader.future_covariate_scaler = train_loader.future_covariate_scaler
        _, full_cov, _ = loader.to_darts_series(fit_scalers=False)

        lookback = hours_to_steps(12, "h")
        horizon = hours_to_steps(3, "h")

        model, _, val_metrics, _ = _train_model(
            train_series=train_series,
            val_series=val_series,
            model_type="TSMixer",
            lookback=lookback,
            horizon=horizon,
            train_covariates=full_cov,
            val_covariates=full_cov,
            scaler=train_loader.target_scaler,
            frequency="h",
            hidden_size=8,
            ff_size=8,
            num_blocks=1,
            n_epochs=2,
        )

        assert model is not None
        assert val_metrics["cv_rmse"] is not None

        reg = ModelRegistry()
        model_id = reg.generate_model_id("tsmixer_test", "TSMixer")
        reg.save_model(
            model=model,
            model_id=model_id,
            model_type="TSMixer",
            building_name="tsmixer_test",
            config={"lookback_hours": lookback, "horizon_hours": horizon, "frequency": "h"},
            column_mapping=loader.column_mapping,
            data_info={},
            training_metrics={},
            validation_metrics=val_metrics,
            scalers={
                "target_scaler": train_loader.target_scaler,
                "covariate_scaler": train_loader.covariate_scaler,
            },
        )

        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=hourly_csv,
            return_predictions=True,
        )

        assert eval_result["success"] is True, eval_result.get("error")
        assert eval_result["test_metrics"]["cv_rmse"] is not None
        assert len(eval_result["predictions"]) > 0
