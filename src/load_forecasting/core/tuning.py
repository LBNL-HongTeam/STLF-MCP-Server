"""
Hyperparameter search spaces and Optuna trial sampling for tune_model.

These are domain constants about the models — they belong in core/ alongside
the model definitions, not in the tools layer.
"""

from .trainer import get_available_models

#: Search-space entry format:
#:   {"type": "int"|"float"|"categorical",
#:    "low": ..., "high": ...,       # for int / float
#:    "log": bool,                   # for float only, optional (default False)
#:    "choices": [...]}              # for categorical only
DEFAULT_SEARCH_SPACES: dict[str, dict] = {
    "XGBoost": {
        "n_estimators": {"type": "int", "low": 30, "high": 300},  # low=30 keeps paper default (40) reachable
        "max_depth": {"type": "int", "low": 3, "high": 8},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    },
    "LSTM": {
        "hidden_dim": {"type": "categorical", "choices": [32, 64, 128]},
        "n_rnn_layers": {"type": "int", "low": 1, "high": 3},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "n_epochs": {"type": "categorical", "choices": [20, 50, 100]},
    },
    "LinearRegression": {},  # No tunable hyperparameters
    "ARIMA": {
        "p": {"type": "int", "low": 1, "high": 6},
        "d": {"type": "categorical", "choices": [0, 1]},
        "q": {"type": "int", "low": 0, "high": 3},
    },
    "TFT": {
        "hidden_size": {"type": "categorical", "choices": [16, 32, 64]},
        "lstm_layers": {"type": "int", "low": 1, "high": 2},
        "num_attention_heads": {"type": "categorical", "choices": [2, 4]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "n_epochs": {"type": "categorical", "choices": [20, 50]},
    },
    # TiDE search space brackets the Li et al. (2025), Table A.3 defaults
    # (hidden_size=128, num_encoder_layers=1, num_decoder_layers=1,
    # decoder_output_dim=16, dropout=0.1).
    "TiDE": {
        "hidden_size": {"type": "categorical", "choices": [64, 128, 256]},
        "num_encoder_layers": {"type": "int", "low": 1, "high": 3},
        "num_decoder_layers": {"type": "int", "low": 1, "high": 3},
        "decoder_output_dim": {"type": "categorical", "choices": [8, 16, 32]},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "n_epochs": {"type": "categorical", "choices": [20, 50]},
    },
    # TSMixer search space brackets the Li et al. (2025), Table A.3 defaults
    # (hidden_size=64, ff_size=64, num_blocks=2, dropout=0.1).
    "TSMixer": {
        "hidden_size": {"type": "categorical", "choices": [32, 64, 128]},
        "ff_size": {"type": "categorical", "choices": [32, 64, 128]},
        "num_blocks": {"type": "int", "low": 1, "high": 4},
        "dropout": {"type": "float", "low": 0.0, "high": 0.4},
        "n_epochs": {"type": "categorical", "choices": [20, 50]},
    },
}

# Fill in empty dicts for any model types that have no default search space.
for _m in get_available_models():
    DEFAULT_SEARCH_SPACES.setdefault(_m, {})


def sample_params(trial, search_space: dict) -> dict:
    """
    Sample hyperparameters from a search space using an Optuna trial.

    Args:
        trial: Optuna trial object.
        search_space: Dict mapping param names to space definitions.

    Returns:
        Dict of sampled parameter values.
    """
    params = {}
    for name, spec in search_space.items():
        kind = spec.get("type", "float")
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:  # float
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
            )
    return params
