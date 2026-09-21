"""
Model training wrapper using Darts.

Supports:
- NaiveMean: Predicts mean of training data
- NaiveSeasonal: Repeats pattern from K periods ago
- NaiveMovingAverage: Moving average baseline
- LinearRegression: Regression on lagged features
- XGBoost: Gradient-boosted trees on lagged features
- LSTM: Long short-term memory deep learning model (BlockRNNModel)
- ARIMA: Autoregressive integrated moving average (statsmodels-backed)
- TFT: Temporal Fusion Transformer deep learning model
- TiDE: Time-series Dense Encoder (MLP encoder-decoder) — Li et al. (2025) top performer
- TSMixer: All-MLP token-mixer time-series model — Li et al. (2025) top performer
- TimesFM: Google TimesFM 2.5 foundation model (200M params, HuggingFace weights)
"""

from contextlib import contextmanager, nullcontext
from typing import Optional, Any
import os
import time
import logging

from darts import TimeSeries, concatenate as _darts_concatenate
from darts.models import (
    NaiveMean,
    NaiveSeasonal,
    NaiveMovingAverage,
    RegressionModel,
    XGBModel,
    BlockRNNModel,
    ARIMA,
    TFTModel,
    TiDEModel,
    TSMixerModel,
)
from sklearn.linear_model import LinearRegression

# TimesFM 2.5 is optional — it lives in a specific Darts module and requires
# the huggingface_hub connector to download pretrained weights on first use.
try:
    from darts.models.forecasting.timesfm2p5_model import TimesFM2p5Model
    _HAS_TIMESFM = True
except Exception:  # pragma: no cover - import guard
    TimesFM2p5Model = None  # type: ignore[assignment]
    _HAS_TIMESFM = False

# TimesFM + Ridge residual hybrid — target-only TimesFM plus a covariate-aware
# post-processing regressor.  Provides the covariate handling that pure TimesFM
# lacks.  Import is unconditional (module is local); the constructor still
# raises ImportError if the TimesFM backbone is unavailable.
from .hybrid_timesfm import TimesFMResidualHybrid

from .evaluator import (
    _align_series,
    _maybe_inverse,
    calculate_metrics,
    metrics_from_arrays,
)
from .frequency_utils import get_seasonality_steps

logger = logging.getLogger(__name__)


def _loss_to_float(v):
    """Coerce a Lightning metric tensor/scalar to a finite float or None."""
    if v is None:
        return None
    try:
        f = float(v.item() if hasattr(v, "item") else v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


try:  # module-level so instances remain picklable (Darts pickles the model)
    import pytorch_lightning as _pl

    class LossHistoryCallback(_pl.Callback):
        """Records per-epoch train/val loss into a shared ``history`` dict.

        Darts logs ``train_loss`` and ``val_loss`` into
        ``trainer.callback_metrics`` each epoch; we snapshot them at the end of
        every training epoch so the lists stay index-aligned with ``epochs``.
        ``val_loss`` is only present when a validation series was supplied to
        ``fit``; it is stored as ``None`` otherwise.

        The class is defined at module scope (not inside a factory) so that a
        model holding a reference to this callback can still be pickled by the
        registry's fallback save path.
        """

        def __init__(self, history: dict):
            super().__init__()
            self.history = history

        def on_train_epoch_end(self, trainer, pl_module):  # noqa: D401
            metrics = trainer.callback_metrics or {}
            self.history["epochs"].append(int(trainer.current_epoch))
            self.history["train_loss"].append(_loss_to_float(metrics.get("train_loss")))
            self.history["val_loss"].append(_loss_to_float(metrics.get("val_loss")))

    _HAS_PL = True
except Exception:  # pragma: no cover - torch/lightning missing
    LossHistoryCallback = None  # type: ignore[assignment]
    _HAS_PL = False


def _make_loss_history_callback():
    """Create a ``(callback, history)`` pair for capturing the learning curve.

    Returns ``(None, None)`` when PyTorch Lightning is unavailable (non-Torch
    models never call this).
    """
    if not _HAS_PL:
        return None, None
    history: dict = {"epochs": [], "train_loss": [], "val_loss": []}
    return LossHistoryCallback(history), history


def _strip_loss_history_callback(model) -> None:
    """Remove any LossHistoryCallback from a fitted Torch model's saved kwargs.

    Darts stores the ``pl_trainer_kwargs`` used at fit time on the model object
    so it can reconstruct the Lightning trainer for a subsequent ``.fit()``.
    The callback references live Lightning state, which breaks ``model.save()``
    and pickling.  We strip only our own callback, leaving any user-supplied
    callbacks untouched.  Best-effort: never raises.
    """
    if LossHistoryCallback is None:
        return

    def _clean_callback_list(container: dict) -> None:
        cbs = container.get("callbacks")
        if isinstance(cbs, list):
            filtered = [cb for cb in cbs if not isinstance(cb, LossHistoryCallback)]
            if filtered:
                container["callbacks"] = filtered
            else:
                container.pop("callbacks", None)

    try:
        # 1) Flat trainer_params (Darts stores the resolved Lightning kwargs here).
        tp = getattr(model, "trainer_params", None)
        if isinstance(tp, dict):
            _clean_callback_list(tp)
        # 2) Nested pl_trainer_kwargs inside the stored constructor params.
        for attr in ("_model_params", "model_params"):
            params = getattr(model, attr, None)
            if not isinstance(params, dict):
                continue
            tk = params.get("pl_trainer_kwargs")
            if isinstance(tk, dict):
                _clean_callback_list(tk)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("Could not strip loss-history callback: %s", e)


@contextmanager
def _force_cpu_device():
    """Temporarily set the PyTorch default device to CPU.

    On Apple Silicon, PyTorch 2.x may dispatch ops (e.g. layer_norm) to MPS
    even when pl_trainer_kwargs specifies accelerator='cpu', causing segfaults
    or dtype errors.  Pinning the default device to CPU for the duration of
    Torch model creation and training avoids this without affecting other code.

    This guard is only applied when the target accelerator is 'cpu'.  When MPS
    is the intended target, Lightning manages device placement and this guard
    must not be used — it would counteract MPS dispatch.
    """
    import torch as _torch
    prev = _torch.get_default_device()
    _torch.set_default_device("cpu")
    try:
        yield
    finally:
        _torch.set_default_device(prev)


def _cast_series_to_float32(
    series: Optional["TimeSeries"],
) -> Optional["TimeSeries"]:
    """Cast a Darts TimeSeries to float32, or return None unchanged.

    MPS does not support float64.  Darts infers the model dtype from the
    TimeSeries data (``self.train_sample[0].dtype``), so the source data must
    be float32 before calling ``model.fit()`` on MPS.
    """
    import numpy as np
    if series is None:
        return None
    if isinstance(series, (list, tuple)):
        return [s.astype(np.float32) for s in series]
    return series.astype(np.float32)


def _cast_series_to_model_dtype(
    series: Optional["TimeSeries"],
    model: Any,
) -> Optional["TimeSeries"]:
    """Cast a Darts TimeSeries to match the dtype a Torch model was trained on.

    When a model was trained with float32 data (e.g. on MPS), inference series
    must also be float32.  This avoids Darts' dtype-mismatch warning and the
    downstream MPS float64 error during ``historical_forecasts``.

    For non-Torch models or models with no ``train_sample`` attribute, the
    series is returned unchanged.
    """
    import numpy as np
    if series is None:
        return None
    try:
        train_dtype = model.train_sample[0].dtype  # np.float32 or np.float64
        if train_dtype == np.float32:
            return series.astype(np.float32)
    except Exception:
        pass
    return series


# Accelerator strings understood by this codebase and PyTorch Lightning.
# We use "cuda" (NVIDIA GPU), "mps" (Apple Silicon GPU) and "cpu".  Lightning
# also accepts the alias "gpu"; it is normalised to "cuda" by _normalize_device.
VALID_ACCELERATORS: frozenset = frozenset({"cuda", "mps", "cpu", "auto"})


def _detect_accelerator() -> str:
    """Return the best available accelerator for PyTorch Lightning.

    Detection priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
    Falls back to 'cpu' when no accelerator is available or torch is missing.

    Returns:
        'cuda' when an NVIDIA GPU is available, 'mps' on Apple Silicon with a
        functional MPS backend, otherwise 'cpu'.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _normalize_device(device: Optional[str]) -> str:
    """Resolve a user-requested device string to a canonical accelerator.

    Accepts a case-insensitive device request and returns one of the canonical
    accelerator strings ('cuda', 'mps', 'cpu').  ``None`` or ``'auto'`` triggers
    auto-detection via :func:`_detect_accelerator`.  The alias ``'gpu'`` maps to
    ``'cuda'``.  Unknown values raise ``ValueError`` so misconfiguration surfaces
    immediately rather than silently falling back.

    Args:
        device: Requested device (e.g. 'cuda', 'gpu', 'mps', 'cpu', 'auto', None).

    Returns:
        A canonical accelerator string: 'cuda', 'mps', or 'cpu'.

    Raises:
        ValueError: If ``device`` is not a recognised device string.
    """
    if device is None:
        return _detect_accelerator()
    normalized = str(device).strip().lower()
    if normalized in ("", "auto"):
        return _detect_accelerator()
    if normalized == "gpu":
        normalized = "cuda"
    if normalized not in VALID_ACCELERATORS:
        raise ValueError(
            f"Unknown device '{device}'. Valid options are: "
            "'cuda' (NVIDIA GPU), 'mps' (Apple Silicon GPU), 'cpu', or "
            "'auto' (auto-detect)."
        )
    if normalized == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "device='cuda' requested but no CUDA GPU is available; "
                    "falling back to CPU. Check your NVIDIA driver / CUDA "
                    "PyTorch build (torch.cuda.is_available() is False)."
                )
                return "cpu"
        except Exception:
            logger.warning(
                "device='cuda' requested but torch is unavailable; "
                "falling back to CPU."
            )
            return "cpu"
    elif normalized == "mps":
        try:
            import torch
            if not (torch.backends.mps.is_available()
                    and torch.backends.mps.is_built()):
                logger.warning(
                    "device='mps' requested but MPS is unavailable; "
                    "falling back to CPU."
                )
                return "cpu"
        except Exception:
            logger.warning(
                "device='mps' requested but torch is unavailable; "
                "falling back to CPU."
            )
            return "cpu"
    return normalized


# Frozenset for O(1) local-model membership test
LOCAL_MODEL_NAMES: frozenset = frozenset(
    {"NaiveMean", "NaiveSeasonal", "NaiveMovingAverage"}
)

# ---------------------------------------------------------------------------
# Energy estimation
# ---------------------------------------------------------------------------

# Assumed TDP (Thermal Design Power) in Watts for each model category.
# Matches the paper's hardware profile (Li et al. 2025, Section 3.2):
#   "two Intel Xeon Gold 6146 CPUs (165 W TDP each)"
# CPU-only regression models use a lower fraction of that budget.
# PyTorch models (LSTM, TFT) target the full single-CPU TDP.
_MODEL_WATTS: dict = {
    "NaiveMean": 15,
    "NaiveSeasonal": 15,
    "NaiveMovingAverage": 15,
    "LinearRegression": 65,
    "ARIMA": 65,
    "XGBoost": 100,
    "LSTM": 165,
    "TFT": 165,
    # TiDE and TSMixer are Lightning-backed deep-learning models; Li et al.
    # (2025) run them on the same Xeon Gold 6146 (165 W TDP) hardware profile.
    "TiDE": 165,
    "TSMixer": 165,
    # TimesFM is a 200M-parameter transformer; fine-tuning even a few epochs
    # is comparable to TFT/LSTM in compute cost.
    "TimesFM": 165,
    # TimesFM+Residual = TimesFM backbone + lightweight Ridge post-processor;
    # cost is essentially TimesFM's plus the historical_forecasts pass used
    # for residual training (bounded by O(n/horizon) TimesFM predictions).
    "TimesFM+Residual": 165,
}
_DEFAULT_WATTS = 65  # fallback for unknown model types


def _cpu_brand() -> Optional[str]:
    """Human-readable CPU name; platform.processor() is empty on macOS/arm."""
    import platform as _pl
    import subprocess

    try:
        if _pl.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        elif _pl.system() == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return _pl.processor() or None


def _memory_gb() -> Optional[float]:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def collect_environment(model_type: str, accelerator: Optional[str] = None) -> dict:
    """Snapshot of the machine and libraries a model was trained with.

    Recorded into ``training_info["environment"]`` at training time so the
    training report can show it later, even when the report is rendered on a
    different machine.  Every field is best-effort; failures leave None.
    """
    import platform as _pl
    import importlib.metadata as _im

    def _ver(pkg: str) -> Optional[str]:
        try:
            return _im.version(pkg)
        except Exception:
            return None

    if _pl.system() == "Darwin" and _pl.mac_ver()[0]:
        os_name = f"macOS {_pl.mac_ver()[0]}"
    else:
        os_name = f"{_pl.system()} {_pl.release()}".strip() or None
    env: dict = {
        "hostname": _pl.node() or None,
        "os": os_name,
        "platform": _pl.platform(),
        "machine": _pl.machine(),
        "cpu": _cpu_brand(),
        "cpu_count": os.cpu_count(),
        "memory_gb": _memory_gb(),
        "python": _pl.python_version(),
        "versions": {
            "darts": _ver("darts"),
            "torch": _ver("torch"),
            "pytorch_lightning": _ver("pytorch-lightning") or _ver("lightning"),
            "xgboost": _ver("xgboost"),
            "optuna": _ver("optuna"),
            "numpy": _ver("numpy"),
        },
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "accelerator": {
            "requested": accelerator,
            "effective": None,
            "type": None,
            "name": None,
            "cuda_version": None,
            "device_count": None,
            "memory_gb": None,
            "torch_threads": None,
        },
    }
    acc = env["accelerator"]
    if model_type not in _TORCH_MODEL_NAMES:
        acc["effective"] = "cpu"
        acc["type"] = "cpu"
        acc["name"] = env["cpu"]
        return env
    try:
        import torch

        acc["torch_threads"] = torch.get_num_threads()
        eff = accelerator or _detect_accelerator()
        acc["effective"] = eff
        if eff == "cuda" and torch.cuda.is_available():
            acc["type"] = "cuda"
            acc["name"] = torch.cuda.get_device_name(0)
            acc["cuda_version"] = getattr(torch.version, "cuda", None)
            acc["device_count"] = torch.cuda.device_count()
            try:
                acc["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
            except Exception:
                pass
        elif eff == "mps":
            acc["type"] = "mps"
            acc["name"] = f"{env['cpu']} GPU (Metal)" if env["cpu"] else "Apple Silicon GPU (Metal)"
            acc["memory_gb"] = env["memory_gb"]  # unified memory
        else:
            acc["type"] = "cpu"
            acc["name"] = env["cpu"]
    except Exception as e:  # never let diagnostics break training
        logger.debug("accelerator probe failed: %s", e)
    return env


def _extract_xgb_eval_history(model: Any) -> Optional[dict]:
    """Per-boosting-round eval curve from a fitted Darts XGBModel.

    Darts wraps one ``XGBRegressor`` per output step in a MultiOutputRegressor
    (``multi_models=True``) and forwards ``val_series`` as ``eval_set``; each
    booster then holds ``evals_result()``.  Curves are averaged across the
    per-step boosters.  XGBoost names eval sets ``validation_0, validation_1,
    ...`` in the order given; Darts passes only the validation split, so a
    single set is the validation curve.  Returns None when nothing was
    recorded (e.g. val_series was not passed).
    """
    try:
        inner = getattr(model, "model", None)
        estimators = list(getattr(inner, "estimators_", None) or ([inner] if inner is not None else []))
        per_set: dict[str, list[list[float]]] = {}
        metric_name: Optional[str] = None
        for est in estimators:
            get = getattr(est, "evals_result", None)
            if get is None:
                continue
            ev = get() or {}
            for set_name, metrics in ev.items():
                for m, values in metrics.items():
                    metric_name = metric_name or m
                    if m != metric_name or not values:
                        continue
                    per_set.setdefault(set_name, []).append([float(v) for v in values])
        if not per_set:
            return None

        def _avg(rows: list[list[float]]) -> list[float]:
            n = min(len(r) for r in rows)
            return [round(sum(r[i] for r in rows) / len(rows), 6) for i in range(n)]

        curves = {name: _avg(rows) for name, rows in sorted(per_set.items())}
        names = list(curves)
        if len(names) >= 2:
            train, val = curves[names[0]], curves[names[1]]
        else:
            train, val = None, curves[names[0]]
        n = len(val)
        return {
            "epochs": list(range(1, n + 1)),
            "train_loss": train[:n] if train else [None] * n,
            "val_loss": val,
            "x_label": "iteration",
            "metric": metric_name or "rmse",
            "n_estimators": len(estimators),
        }
    except Exception as e:  # never let diagnostics break training
        logger.debug("XGBoost eval history unavailable: %s", e)
        return None


def _estimate_training_energy_kwh(
    training_time_seconds: float,
    model_type: str,
) -> tuple[float, int]:
    """
    Estimate the energy consumed during model training using a TDP-based proxy.

    This follows the approach of Li et al. (2025) (Table 5) which tracks
    training energy in kWh.  Because we do not depend on eco2ai, we use a
    simple TDP model:

        energy_kwh = watts × time_seconds / 3_600_000

    The ``watts_assumed`` value is returned alongside the estimate so callers
    can note it in their output and users can adjust for their own hardware.

    Args:
        training_time_seconds: Wall-clock training time in seconds.
        model_type: Model type string (e.g. "XGBoost", "LSTM").

    Returns:
        Tuple of (energy_kwh: float, watts_assumed: int).
    """
    watts = _MODEL_WATTS.get(model_type, _DEFAULT_WATTS)
    energy_kwh = watts * training_time_seconds / 3_600_000
    return round(energy_kwh, 8), watts

# Models that accept both past AND future covariates via fit/historical_forecasts.
# TFT is a MixedCovariatesTorchModel; add_relative_index=True lets it work
# without mandatory future covariates while still accepting them when provided.
# NOTE: LSTM (BlockRNNModel) is listed here because it accepts past_covariates
# in fit(), but future_covariates are silently ignored at the call site — see
# train_model() and generate_predictions() for the LSTM-specific exclusion.
MIXED_COVARIATE_MODELS: frozenset = frozenset(
    {"LinearRegression", "XGBoost", "LSTM", "TFT", "TiDE", "TSMixer", "TimesFM+Residual"}
)

# Models that accept only future covariates (not past).
# ARIMA uses exogenous variables (future covariates) only — statsmodels does not
# support past covariate lags in the same way as regression-based models.
FUTURE_ONLY_MODELS: frozenset = frozenset({"ARIMA"})

# Models that accept only past covariates (not future).
# LSTM (BlockRNNModel) fits past_covariates but silently ignores future_covariates
# even though it appears in MIXED_COVARIATE_MODELS for the fit() call.
PAST_COVARIATE_ONLY_MODELS: frozenset = frozenset({"LSTM"})

# PyTorch-backed models that must run with _force_cpu_device() to avoid MPS
# dtype/segfault issues on Apple Silicon (PyTorch 2.x).  TimesFM is a
# PyTorch/Lightning model and follows the same MPS handling as LSTM/TFT
# (float32 casting, CPU guard when accelerator=='cpu').
_TORCH_MODEL_NAMES: frozenset = frozenset(
    {"LSTM", "TFT", "TiDE", "TSMixer", "TimesFM", "TimesFM+Residual"}
)

# Native Darts PyTorch-Lightning models with a standard epoch-based training
# loop and a Darts .fit() that accepts val_series/val_*_covariates kwargs.
# These are the models for which a per-epoch train/val learning curve is
# meaningful.  TimesFM (foundation model, minimal fine-tuning) and
# TimesFM+Residual (custom hybrid fit signature) are intentionally excluded.
_PL_LIGHTNING_MODELS: frozenset = frozenset({"LSTM", "TFT", "TiDE", "TSMixer"})

# Models that are handed the validation split at fit time so a per-epoch /
# per-iteration validation curve is produced.  XGBoost joins the Lightning
# models: Darts' XGBModel.fit accepts val_series and forwards it as
# eval_set, and the booster then records validation RMSE per boosting
# round (no early stopping is enabled, so the fitted model is unchanged).
_VAL_SERIES_MODELS: frozenset = _PL_LIGHTNING_MODELS | frozenset({"XGBoost"})

# Foundation models that support no covariates at all — used to gate the
# tool-layer covariate warnings.  TimesFM 2.5 dropped the XReg pathway from
# the original Google implementation (see darts.models.forecasting.timesfm2p5_model).
# The TimesFM+Residual hybrid is intentionally NOT in this set: it wraps the
# target-only TimesFM backbone with a Ridge regressor that consumes past and
# future covariates externally.
NO_COVARIATE_GLOBAL_MODELS: frozenset = frozenset({"TimesFM"})

# Darts *global* models, i.e. those whose fit() accepts a sequence of
# TimeSeries.  Only these can consume the gap-free per-season chunk lists
# produced by ForecastingDataLoader.to_darts_series_chunks(); local models
# (Naive*, ARIMA) and the TimesFM wrappers must be trained on a single
# contiguous series and therefore fall back to a sequential split.
MULTI_SERIES_MODELS: frozenset = frozenset(
    {"LinearRegression", "XGBoost", "LSTM", "TFT", "TiDE", "TSMixer"}
)


def _as_series_list(series) -> list:
    """Normalise a TimeSeries-or-list into a list."""
    if series is None:
        return []
    return list(series) if isinstance(series, (list, tuple)) else [series]


def _match_covariates(covariates, n: int):
    """Broadcast a covariate series to align 1:1 with ``n`` target series.

    Darts requires ``len(past_covariates) == len(series)`` when fitting on a
    sequence.  The covariate series here always spans the *whole* dataset (so
    that historical_forecasts can read past a split boundary), and Darts slices
    it by timestamp per target series, so the same object is reused for every
    chunk.
    """
    if covariates is None:
        return None
    if isinstance(covariates, (list, tuple)):
        covs = list(covariates)
        if len(covs) == n:
            return covs
        if len(covs) == 1:
            return covs * n
        raise ValueError(
            f"Cannot align {len(covs)} covariate series with {n} target series."
        )
    return [covariates] * n


# Models whose fit/predict happen outside the Darts historical_forecasts
# dispatch pipeline (typically because they compose multiple sub-models).
# Callers must run manual walk-forward evaluation instead.
HYBRID_MODELS: frozenset = frozenset({"TimesFM+Residual"})

# Models that support probabilistic (quantile) forecasting via a Darts
# likelihood.  Torch models (LSTM/TFT/TiDE/TSMixer) accept a
# ``QuantileRegression`` likelihood object; XGBoost accepts ``likelihood=
# "quantile"`` + ``quantiles=[...]``.  Naive baselines, LinearRegression,
# ARIMA, and the TimesFM foundation models produce point forecasts only.
PROBABILISTIC_MODELS: frozenset = frozenset(
    {"XGBoost", "LSTM", "TFT", "TiDE", "TSMixer"}
)

# XGBoost takes the likelihood as a string + a separate quantiles list; the
# torch models take a QuantileRegression likelihood *object*.
_QUANTILE_STRING_MODELS: frozenset = frozenset({"XGBoost"})

# Default quantile levels when probabilistic training is requested without an
# explicit list.  Median plus a symmetric 80% interval (P10/P90) — a common
# operational band for reserve/margin decisions.
DEFAULT_QUANTILES: tuple = (0.1, 0.5, 0.9)


def validate_quantiles(quantiles) -> list[float]:
    """Validate and normalise a list of quantile levels.

    Ensures every value is a float strictly inside (0, 1), de-duplicates,
    sorts ascending, and guarantees the median (0.5) is present so a point
    forecast can always be recovered from the probabilistic output.

    Args:
        quantiles: Iterable of quantile levels, or None for the default set.

    Returns:
        Sorted list of unique quantile floats including 0.5.

    Raises:
        ValueError: If any level is outside (0, 1) or the list is empty.
    """
    if quantiles is None:
        return list(DEFAULT_QUANTILES)
    try:
        qs = [float(q) for q in quantiles]
    except (TypeError, ValueError):
        raise ValueError(
            f"quantiles must be a list of numbers in (0, 1); got {quantiles!r}"
        )
    if not qs:
        raise ValueError("quantiles must contain at least one level")
    for q in qs:
        if not (0.0 < q < 1.0):
            raise ValueError(
                f"Each quantile must be strictly between 0 and 1; got {q}"
            )
    qs.append(0.5)  # ensure a median for point-forecast recovery
    return sorted(set(qs))


def _make_quantile_likelihood(quantiles: list[float]):
    """Build a Darts QuantileRegression likelihood for torch models.

    Imported lazily so the module still loads if the likelihood API moves
    between Darts versions.
    """
    from darts.utils.likelihood_models import QuantileRegression
    return QuantileRegression(quantiles=quantiles)

# TimesFM 2.5 hard architectural limit: output_chunk_length + output_chunk_shift
# must be <= 128 (one output patch).  We currently do not expose output_chunk_shift,
# so the effective cap on horizon steps is 128.
_TIMESFM_MAX_OUTPUT_STEPS = 128
# input_chunk_length + output_chunk_length + output_chunk_shift must be <= 16384.
_TIMESFM_MAX_CONTEXT_STEPS = 16384

# Model class mapping
MODEL_CLASSES = {
    "NaiveMean": NaiveMean,
    "NaiveSeasonal": NaiveSeasonal,
    "NaiveMovingAverage": NaiveMovingAverage,
    "LinearRegression": "RegressionModel",  # Special handling
    "XGBoost": "XGBModel",                  # Special handling
    "LSTM": "BlockRNNModel",                # Special handling
    "ARIMA": "ARIMA",                       # Special handling
    "TFT": "TFTModel",                      # Special handling
    "TiDE": "TiDEModel",                    # Special handling
    "TSMixer": "TSMixerModel",              # Special handling
    "TimesFM": "TimesFM2p5Model",           # Special handling; optional import
    "TimesFM+Residual": "TimesFMResidualHybrid",  # Special handling; local wrapper
}

# Darts class names for metadata
DARTS_CLASS_NAMES = {
    "NaiveMean": "darts.models.NaiveMean",
    "NaiveSeasonal": "darts.models.NaiveSeasonal",
    "NaiveMovingAverage": "darts.models.NaiveMovingAverage",
    "LinearRegression": "darts.models.RegressionModel",
    "XGBoost": "darts.models.XGBModel",
    "LSTM": "darts.models.BlockRNNModel",
    "ARIMA": "darts.models.ARIMA",
    "TFT": "darts.models.TFTModel",
    "TiDE": "darts.models.TiDEModel",
    "TSMixer": "darts.models.TSMixerModel",
    "TimesFM": "darts.models.forecasting.timesfm2p5_model.TimesFM2p5Model",
    "TimesFM+Residual": "load_forecasting.core.hybrid_timesfm.TimesFMResidualHybrid",
}


def get_available_models() -> list[str]:
    """Return list of available model types."""
    return list(MODEL_CLASSES.keys())


def _resolve_accelerator(model_label: str, model_kwargs: dict) -> str:
    """Normalize the requested accelerator and log the GPU choice for a model."""
    accel = _normalize_device(model_kwargs.get("accelerator"))
    if accel == "cuda":
        logger.info("%s: using CUDA accelerator (NVIDIA GPU)", model_label)
    elif accel == "mps":
        logger.info("%s: using MPS accelerator (Apple Silicon GPU)", model_label)
    return accel


def _make_pl_trainer_kwargs(accelerator: str, model_kwargs: dict) -> dict:
    """Standard PyTorch-Lightning trainer kwargs shared by all Torch models."""
    kwargs = {
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "accelerator": accelerator,
        "precision": model_kwargs.get("precision", "32"),
    }
    # Attach the loss-history callback (created in train_model) so every Torch
    # model records per-epoch train/val loss for the learning-curve report.
    callback = model_kwargs.get("_loss_history_callback")
    if callback is not None:
        kwargs["callbacks"] = [callback]
    return kwargs


def _apply_passthroughs(target: dict, model_kwargs: dict, names) -> None:
    """Copy any of ``names`` present in model_kwargs into ``target`` in place."""
    for name in names:
        if name in model_kwargs:
            target[name] = model_kwargs[name]


def _validate_timesfm_limits(lookback: int, horizon: int, label: str) -> None:
    """Enforce TimesFM 2.5 architectural limits, raising ValueError on breach."""
    if not _HAS_TIMESFM:
        raise ImportError(
            "TimesFM2p5Model could not be imported from "
            "darts.models.forecasting.timesfm2p5_model. Ensure darts is "
            "installed with a version that ships the TimesFM 2.5 wrapper, "
            "and that huggingface_hub is available."
        )
    if horizon > _TIMESFM_MAX_OUTPUT_STEPS:
        raise ValueError(
            f"{label} horizon must be <= {_TIMESFM_MAX_OUTPUT_STEPS} steps "
            f"(one output patch). Got horizon={horizon} steps."
        )
    if lookback + horizon > _TIMESFM_MAX_CONTEXT_STEPS:
        raise ValueError(
            f"{label} total context (lookback + horizon = {lookback + horizon} "
            f"steps) exceeds the model's maximum context window of "
            f"{_TIMESFM_MAX_CONTEXT_STEPS} steps."
        )


def _make_timesfm_kwargs(lookback: int, horizon: int, accelerator: str, model_kwargs: dict) -> dict:
    """Build the TimesFM 2.5 constructor kwargs (shared by pure + hybrid)."""
    tfm_kwargs = {
        "input_chunk_length": lookback,
        "output_chunk_length": horizon,
        "n_epochs": model_kwargs.get("n_epochs", 5),
        "enable_finetuning": model_kwargs.get("enable_finetuning", True),
        "pl_trainer_kwargs": model_kwargs.get(
            "pl_trainer_kwargs", _make_pl_trainer_kwargs(accelerator, model_kwargs)
        ),
    }
    _apply_passthroughs(
        tfm_kwargs,
        model_kwargs,
        ("hub_model_name", "hub_model_revision", "local_dir",
         "batch_size", "optimizer_kwargs", "random_state"),
    )
    return tfm_kwargs


def create_model(
    model_type: str,
    lookback: int = 24,
    horizon: int = 6,
    use_past_covariates: bool = False,
    use_future_covariates: bool = False,
    frequency: str = "h",
    # Legacy alias kept for backwards compatibility
    use_covariates: bool = False,
    probabilistic: bool = False,
    quantiles: Optional[list] = None,
    **model_kwargs,
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
        probabilistic: If True, fit a quantile (probabilistic) model so
            prediction intervals can be produced.  Only honoured for models in
            ``PROBABILISTIC_MODELS`` (XGBoost, LSTM, TFT, TiDE, TSMixer); a
            ``ValueError`` is raised for unsupported model types so the caller
            fails fast rather than silently getting a point model.
        quantiles: Quantile levels to fit when ``probabilistic`` is True.
            Defaults to ``DEFAULT_QUANTILES`` (P10/P50/P90).  0.5 is always
            included so a median point forecast is available.
        **model_kwargs: Additional model-specific hyperparameters passed
            through to the underlying model constructor (used by tune_model).

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_type is unknown, or if probabilistic=True for a
            model type that does not support quantile forecasting.
    """
    # Handle legacy alias
    use_past_covariates = use_past_covariates or use_covariates

    if model_type not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MODEL_CLASSES.keys())}"
        )

    # Resolve probabilistic (quantile) configuration once up front.
    resolved_quantiles: Optional[list[float]] = None
    if probabilistic:
        if model_type not in PROBABILISTIC_MODELS:
            raise ValueError(
                f"probabilistic=True is not supported for model type "
                f"'{model_type}'. Supported: {sorted(PROBABILISTIC_MODELS)}."
            )
        resolved_quantiles = validate_quantiles(quantiles)

    def _torch_likelihood_kwargs() -> dict:
        """Return {'likelihood': QuantileRegression} for torch models, else {}."""
        if resolved_quantiles is None:
            return {}
        return {"likelihood": _make_quantile_likelihood(resolved_quantiles)}

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

    elif model_type == "XGBoost":
        lags_future = list(range(0, horizon)) if use_future_covariates else None
        # Defaults match Li et al. (2025) Table A.3:
        #   n_estimators=40, max_depth=6, booster=gbtree.
        # learning_rate, subsample, colsample_bytree are intentionally omitted
        # so the XGBoost library defaults apply (lr=0.3, subsample=1.0,
        # colsample_bytree=1.0), matching the paper which does not specify them.
        # Any of these can still be overridden via model_kwargs.
        # XGBoost takes the likelihood as a string ("quantile") plus a separate
        # quantiles list, unlike the torch models which take a likelihood object.
        _xgb_prob_kwargs: dict = {}
        if resolved_quantiles is not None:
            _xgb_prob_kwargs = {
                "likelihood": "quantile",
                "quantiles": resolved_quantiles,
            }
        model = XGBModel(
            lags=lookback,
            lags_past_covariates=lookback if use_past_covariates else None,
            lags_future_covariates=lags_future,
            output_chunk_length=horizon,
            n_estimators=model_kwargs.get("n_estimators", 40),
            max_depth=model_kwargs.get("max_depth", 6),
            booster=model_kwargs.get("booster", "gbtree"),
            **_xgb_prob_kwargs,
        )

    elif model_type == "LSTM":
        # BlockRNNModel supports past covariates but NOT future covariates.
        _lstm_dropout = model_kwargs.get("dropout", 0.1)
        _lstm_n_rnn_layers = model_kwargs.get("n_rnn_layers", 2)
        _lstm_accelerator = _resolve_accelerator("LSTM", model_kwargs)
        if _lstm_accelerator == "mps" and _lstm_dropout > 0 and _lstm_n_rnn_layers >= 2:
            logger.warning(
                "LSTM on MPS with dropout>0 and n_rnn_layers>=2 has a known "
                "PyTorch bug (github.com/pytorch/pytorch/issues/180744) that "
                "silently applies dropout during inference, producing wrong "
                "predictions. Consider setting dropout=0 or n_rnn_layers=1 "
                "when using MPS."
            )
        model = BlockRNNModel(
            model="LSTM",
            input_chunk_length=lookback,
            output_chunk_length=horizon,
            hidden_dim=model_kwargs.get("hidden_dim", 64),
            n_rnn_layers=_lstm_n_rnn_layers,
            dropout=_lstm_dropout,
            n_epochs=model_kwargs.get("n_epochs", 20),
            pl_trainer_kwargs=_make_pl_trainer_kwargs(_lstm_accelerator, model_kwargs),
            **_torch_likelihood_kwargs(),
        )

    elif model_type == "ARIMA":
        # ARIMA is a TransferableFutureCovariatesLocalForecastingModel.
        # It supports future covariates as exogenous variables (no past covariates).
        # lookback is NOT used: ARIMA's AR order `p` plays that role instead.
        # retrain=False is supported in historical_forecasts via statsmodels apply().
        model = ARIMA(
            p=model_kwargs.get("p", 1),
            d=model_kwargs.get("d", 1),
            q=model_kwargs.get("q", 0),
            seasonal_order=model_kwargs.get("seasonal_order", (0, 0, 0, 0)),
        )

    elif model_type == "TFT":
        # TFTModel is a MixedCovariatesTorchModel.
        # add_relative_index=True lets TFT work without mandatory future covariates
        # by injecting a positional index; user-supplied future covariates are still
        # concatenated on top if provided.
        _tft_accelerator = _resolve_accelerator("TFT", model_kwargs)
        if _tft_accelerator == "mps":
            logger.warning(
                "TFT on MPS has a known open PyTorch issue with MultiheadAttention "
                "+ masking + dropout that may produce NaN losses "
                "(github.com/pytorch/pytorch/issues/151667). Monitor training loss "
                "carefully. Pass accelerator='cpu' to disable MPS if NaNs occur."
            )
        # TFT defaults to its own QuantileRegression likelihood.  When the user
        # does NOT request probabilistic output we pin it to a deterministic
        # point model (loss_fn=MSELoss, likelihood=None) so behaviour is
        # unchanged from before this feature; otherwise we install the
        # requested quantiles.
        _tft_prob_kwargs = _torch_likelihood_kwargs()
        if not _tft_prob_kwargs:
            import torch as _torch_for_tft
            _tft_prob_kwargs = {
                "likelihood": None,
                "loss_fn": _torch_for_tft.nn.MSELoss(),
            }
        model = TFTModel(
            input_chunk_length=lookback,
            output_chunk_length=horizon,
            hidden_size=model_kwargs.get("hidden_size", 16),
            lstm_layers=model_kwargs.get("lstm_layers", 1),
            num_attention_heads=model_kwargs.get("num_attention_heads", 4),
            dropout=model_kwargs.get("dropout", 0.1),
            n_epochs=model_kwargs.get("n_epochs", 20),
            add_relative_index=True,
            pl_trainer_kwargs=_make_pl_trainer_kwargs(_tft_accelerator, model_kwargs),
            **_tft_prob_kwargs,
        )

    elif model_type == "TiDE":
        # TiDEModel is a MixedCovariatesTorchModel (MLP encoder-decoder).
        # Supports past + future covariates and static covariates.
        # Defaults match Li et al. (2025), Table A.3:
        #   hidden_size=128, num_encoder_layers=1, num_decoder_layers=1,
        #   decoder_output_dim=16, temporal_width_past=4,
        #   temporal_width_future=4, dropout=0.1.
        # n_epochs=20 kept for consistency with LSTM/TFT defaults in this
        # codebase; the paper uses up to 50 with cosine-annealing + early
        # stopping (Appendix A) — expose via model_kwargs if desired.
        _tide_accelerator = _resolve_accelerator("TiDE", model_kwargs)
        _tide_kwargs = {
            "input_chunk_length": lookback,
            "output_chunk_length": horizon,
            "hidden_size": model_kwargs.get("hidden_size", 128),
            "num_encoder_layers": model_kwargs.get("num_encoder_layers", 1),
            "num_decoder_layers": model_kwargs.get("num_decoder_layers", 1),
            "decoder_output_dim": model_kwargs.get("decoder_output_dim", 16),
            "temporal_width_past": model_kwargs.get("temporal_width_past", 4),
            "temporal_width_future": model_kwargs.get("temporal_width_future", 4),
            "dropout": model_kwargs.get("dropout", 0.1),
            "n_epochs": model_kwargs.get("n_epochs", 20),
            "pl_trainer_kwargs": _make_pl_trainer_kwargs(_tide_accelerator, model_kwargs),
        }
        # Optional passthroughs for advanced use (paper Appendix A setup,
        # RINorm, batch size, etc.).
        _apply_passthroughs(
            _tide_kwargs, model_kwargs,
            ("temporal_hidden_size_past", "temporal_hidden_size_future",
             "temporal_decoder_hidden", "use_layer_norm", "use_static_covariates",
             "use_reversible_instance_norm", "batch_size", "optimizer_kwargs",
             "lr_scheduler_cls", "lr_scheduler_kwargs", "random_state"),
        )
        _tide_kwargs.update(_torch_likelihood_kwargs())
        model = TiDEModel(**_tide_kwargs)

    elif model_type == "TSMixer":
        # TSMixerModel is a MixedCovariatesTorchModel (all-MLP token-mixer).
        # Supports past + future covariates and static covariates.
        # Defaults match Li et al. (2025), Table A.3:
        #   hidden_size=64, ff_size=64, num_blocks=2, activation=ReLU,
        #   dropout=0.1, norm_type=LayerNorm.
        # n_epochs=20 kept for consistency with LSTM/TFT defaults; paper uses
        # up to 50 with cosine-annealing + early stopping (Appendix A).
        _tsm_accelerator = _resolve_accelerator("TSMixer", model_kwargs)
        _tsm_kwargs = {
            "input_chunk_length": lookback,
            "output_chunk_length": horizon,
            "hidden_size": model_kwargs.get("hidden_size", 64),
            "ff_size": model_kwargs.get("ff_size", 64),
            "num_blocks": model_kwargs.get("num_blocks", 2),
            "activation": model_kwargs.get("activation", "ReLU"),
            "dropout": model_kwargs.get("dropout", 0.1),
            "norm_type": model_kwargs.get("norm_type", "LayerNorm"),
            "n_epochs": model_kwargs.get("n_epochs", 20),
            "pl_trainer_kwargs": _make_pl_trainer_kwargs(_tsm_accelerator, model_kwargs),
        }
        _apply_passthroughs(
            _tsm_kwargs, model_kwargs,
            ("normalize_before", "use_static_covariates",
             "use_reversible_instance_norm", "batch_size", "optimizer_kwargs",
             "lr_scheduler_cls", "lr_scheduler_kwargs", "random_state"),
        )
        _tsm_kwargs.update(_torch_likelihood_kwargs())
        model = TSMixerModel(**_tsm_kwargs)

    elif model_type == "TimesFM":
        # TimesFM 2.5: Google's 200M-param foundation model for time series.
        # No covariate support (XReg was dropped from the port); pretrained
        # weights are auto-downloaded from HuggingFace Hub on first use.
        _validate_timesfm_limits(lookback, horizon, "TimesFM")
        _tfm_accelerator = _resolve_accelerator("TimesFM", model_kwargs)
        logger.warning(
            "TimesFM: pretrained weights (~800MB) will be downloaded from "
            "HuggingFace Hub on first use unless `local_dir` is provided in "
            "model_kwargs. Ensure network access or a pre-cached checkpoint."
        )
        # Light fine-tuning by default (n_epochs=5).  Zero-shot use is available
        # by passing n_epochs=0 or enable_finetuning=False via model_kwargs.
        model = TimesFM2p5Model(
            **_make_timesfm_kwargs(lookback, horizon, _tfm_accelerator, model_kwargs)
        )

    elif model_type == "TimesFM+Residual":
        # TimesFM 2.5 backbone + Ridge residual correction on covariates.
        # The TimesFM sub-model is configured with the same kwargs as pure
        # TimesFM; the hybrid class wraps it plus a StandardScaler + Ridge
        # regressor fit on in-sample residuals.
        _validate_timesfm_limits(lookback, horizon, "TimesFM+Residual")
        _hyb_accelerator = _resolve_accelerator("TimesFM+Residual", model_kwargs)
        logger.warning(
            "TimesFM+Residual: pretrained TimesFM weights (~800MB) will be "
            "downloaded from HuggingFace Hub on first use unless `local_dir` "
            "is provided in model_kwargs."
        )
        _hyb_timesfm_kwargs = _make_timesfm_kwargs(
            lookback, horizon, _hyb_accelerator, model_kwargs
        )
        model = TimesFMResidualHybrid(
            timesfm_kwargs=_hyb_timesfm_kwargs,
            horizon=horizon,
            alpha=model_kwargs.get("alpha", 1.0),
            hdd_base=model_kwargs.get("hdd_base", 18.0),
            cdd_base=model_kwargs.get("cdd_base", 22.0),
            n_knots=model_kwargs.get("n_knots", 3),
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


_NULL_METRICS = {"rmse": None, "mae": None, "mape": None, "cv_rmse": None, "r_squared": None}


def _eval_split(
    model, series, covariates, future_covariates, lookback, horizon, model_type, scaler, label
) -> dict:
    """Score a fitted model on one split, returning metrics (or null metrics on error).

    ``series`` may be a single TimeSeries or a list of gap-free chunks (from a
    seasonal split).  For a list, each chunk is scored independently and the
    actual/predicted pairs are pooled before the metrics are computed, so the
    result is comparable to the single-series case.
    """
    try:
        chunks = _as_series_list(series)
        if len(chunks) == 1:
            pred = generate_predictions(
                model, chunks[0], covariates, lookback, horizon,
                future_covariates=future_covariates, model_type=model_type,
                last_points_only=True,
            )
            if isinstance(pred, list):
                pred = _darts_concatenate(pred, ignore_time_axis=True)
            return calculate_metrics(chunks[0][lookback:], pred, scaler=scaler)

        # Multi-chunk (seasonal) split: score each contiguous chunk on its own
        # real timestamps, then pool the aligned residuals.  Concatenating the
        # chunks first would require ignore_time_axis=True, which shifts the
        # actual and predicted indices independently and destroys alignment.
        import numpy as np
        actual_parts, pred_parts = [], []
        for chunk in chunks:
            # A chunk shorter than one input+output window yields no forecast.
            if len(chunk) <= lookback + horizon:
                logger.debug(
                    "Skipping %s chunk of %d steps (needs > %d).",
                    label, len(chunk), lookback + horizon,
                )
                continue
            pred = generate_predictions(
                model, chunk, covariates, lookback, horizon,
                future_covariates=future_covariates, model_type=model_type,
                last_points_only=True,
            )
            if isinstance(pred, list):
                pred = _darts_concatenate(pred, ignore_time_axis=True)
            a_vals, p_vals = _align_series(
                _maybe_inverse(chunk[lookback:], scaler),
                _maybe_inverse(pred, scaler),
            )
            actual_parts.append(a_vals)
            pred_parts.append(p_vals)

        if not pred_parts:
            logger.warning(
                "No %s chunk was long enough to score (lookback=%d, horizon=%d).",
                label, lookback, horizon,
            )
            return dict(_NULL_METRICS)

        return metrics_from_arrays(
            np.concatenate(actual_parts), np.concatenate(pred_parts)
        )
    except Exception as e:
        logger.warning("Failed to compute %s metrics: %s", label, e)
        return dict(_NULL_METRICS)


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
    **model_kwargs,
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
            Used by ARIMA as exogenous variables and by TFT/LinearRegression/XGBoost
            as future-known inputs. LSTM (BlockRNNModel) does not use them.
        val_future_covariates: Validation future covariates (optional).
        scaler: Scaler for inverse transform when computing metrics
        frequency: Data frequency for seasonality calculation
        **model_kwargs: Additional hyperparameters forwarded to create_model.

    Returns:
        Tuple of (model, training_metrics, validation_metrics, info)
    """
    start_time = time.time()

    # A seasonal split hands us a list of gap-free chunks rather than one
    # contiguous series (see ForecastingDataLoader.to_darts_series_chunks).
    # Darts global models fit on a sequence natively; covariates are broadcast
    # to match its length because the same full-dataset covariate series backs
    # every chunk.
    _multi_series = isinstance(train_series, (list, tuple))
    if _multi_series:
        if model_type not in MULTI_SERIES_MODELS:
            raise ValueError(
                f"{model_type} cannot be trained on multiple series. Pass a single "
                "contiguous TimeSeries (use a sequential split for this model type)."
            )
        train_series = list(train_series)
        logger.info(
            "Training %s on %d contiguous chunks (%d steps total, no gap "
            "interpolation).",
            model_type, len(train_series), sum(len(s) for s in train_series),
        )

    # Create model
    # ARIMA only accepts future covariates; never treat past_covariates as usable for it.
    # TimesFM (and any other NO_COVARIATE_GLOBAL_MODELS) supports neither.
    use_past_covariates = (
        train_covariates is not None
        and model_type not in FUTURE_ONLY_MODELS
        and model_type not in NO_COVARIATE_GLOBAL_MODELS
    )
    use_future_covariates = (
        train_future_covariates is not None
        and model_type in (MIXED_COVARIATE_MODELS | FUTURE_ONLY_MODELS)
        and model_type not in PAST_COVARIATE_ONLY_MODELS
        and model_type not in NO_COVARIATE_GLOBAL_MODELS
    )
    if model_type in NO_COVARIATE_GLOBAL_MODELS and (
        train_covariates is not None or train_future_covariates is not None
    ):
        logger.warning(
            f"{model_type} does not support covariates; provided past/future "
            "covariates will be ignored during training."
        )
    # _force_cpu_device() guards against PyTorch 2.x on Apple Silicon allocating
    # tensors on MPS via the global default device even when Lightning is told to
    # use CPU.  When MPS is the *intended* accelerator, Lightning manages device
    # placement itself and the guard must not be applied — it would fight MPS.
    _requested_accelerator = _normalize_device(model_kwargs.get("accelerator"))
    # Persist the resolved accelerator so create_model uses the same value even
    # if the original request was 'auto'/None or an unavailable device.
    if model_type in _TORCH_MODEL_NAMES:
        model_kwargs["accelerator"] = _requested_accelerator
    _use_cpu_guard = model_type in _TORCH_MODEL_NAMES and _requested_accelerator == "cpu"

    # Attach a per-epoch loss-history callback for Torch models so the
    # evaluation report can render a train/val learning curve.  The callback
    # is threaded to the Lightning trainer via _make_pl_trainer_kwargs, which
    # reads the "_loss_history_callback" key out of model_kwargs.  Non-Torch
    # models leave loss_history as None and emit no training_history.
    loss_history: Optional[dict] = None
    if model_type in _PL_LIGHTNING_MODELS:
        _cb, loss_history = _make_loss_history_callback()
        if _cb is not None:
            model_kwargs["_loss_history_callback"] = _cb

    # Lightning precision="32" initializes Torch model parameters as float32,
    # while CSV-backed Darts TimeSeries are normally float64. Cast every Torch
    # input consistently; otherwise explicit CPU training can fail in a model's
    # first linear layer with ``double != float`` just as CUDA does, while MPS
    # does not support float64 at all.
    if model_type in _TORCH_MODEL_NAMES:
        train_series = _cast_series_to_float32(train_series)
        train_covariates = _cast_series_to_float32(train_covariates)
        train_future_covariates = _cast_series_to_float32(train_future_covariates)
        val_series = _cast_series_to_float32(val_series)
        val_covariates = _cast_series_to_float32(val_covariates)
        val_future_covariates = _cast_series_to_float32(val_future_covariates)
        logger.info(
            "%s mode: cast all TimeSeries to float32",
            _requested_accelerator.upper(),
        )

    with _force_cpu_device() if _use_cpu_guard else nullcontext():
        model = create_model(
            model_type, lookback, horizon,
            use_past_covariates=use_past_covariates,
            use_future_covariates=use_future_covariates,
            frequency=frequency,
            **model_kwargs,
        )
        logger.info(
            f"Created {model_type} model with lookback={lookback}, "
            f"horizon={horizon}, past_covariates={use_past_covariates}, "
            f"future_covariates={use_future_covariates}"
        )

        # Train model
        try:
            if model_type in MIXED_COVARIATE_MODELS:
                # LinearRegression, XGBoost, TFT: past + future covariates.
                # LSTM (BlockRNNModel): past covariates only (future are silently ignored).
                fit_kwargs: dict = {}
                _n = len(train_series) if _multi_series else None
                if train_covariates is not None:
                    fit_kwargs["past_covariates"] = (
                        _match_covariates(train_covariates, _n)
                        if _multi_series else train_covariates
                    )
                if train_future_covariates is not None and model_type not in PAST_COVARIATE_ONLY_MODELS:
                    fit_kwargs["future_covariates"] = (
                        _match_covariates(train_future_covariates, _n)
                        if _multi_series else train_future_covariates
                    )
                # For native Darts Lightning models, pass the validation series
                # so Lightning runs a per-epoch validation loop and logs
                # val_loss — this populates the validation line on the
                # training-curve report.  XGBoost also accepts val_series
                # (forwarded to the booster as eval_set).  LinearRegression
                # ignores it and the TimesFM+Residual hybrid has a custom
                # fit() signature that rejects it -- see _VAL_SERIES_MODELS.
                # Keep only validation chunks long enough to yield at least one
                # training sample; Darts raises if any series in the sequence is
                # shorter than lookback + horizon.
                _val_usable = [
                    s for s in _as_series_list(val_series)
                    if len(s) > (lookback + horizon)
                ]
                if model_type == "XGBoost":
                    # With an eval_set the booster prints one line per round
                    # to STDOUT, which would corrupt the MCP stdio transport.
                    fit_kwargs["verbose"] = False
                if model_type in _VAL_SERIES_MODELS and _val_usable:
                    fit_kwargs["val_series"] = (
                        _val_usable if len(_val_usable) > 1 else _val_usable[0]
                    )
                    _nv = len(_val_usable) if len(_val_usable) > 1 else None
                    if train_covariates is not None:
                        _vc = val_covariates if val_covariates is not None else train_covariates
                        fit_kwargs["val_past_covariates"] = (
                            _match_covariates(_vc, _nv) if _nv else _vc
                        )
                    if train_future_covariates is not None and model_type not in PAST_COVARIATE_ONLY_MODELS:
                        _vf = val_future_covariates if val_future_covariates is not None else train_future_covariates
                        fit_kwargs["val_future_covariates"] = (
                            _match_covariates(_vf, _nv) if _nv else _vf
                        )
                model.fit(train_series, **fit_kwargs)
            elif model_type in FUTURE_ONLY_MODELS:
                # ARIMA: only future covariates are supported (as exogenous variables)
                fit_kwargs = {}
                if train_future_covariates is not None:
                    fit_kwargs["future_covariates"] = train_future_covariates
                model.fit(train_series, **fit_kwargs)
            else:
                model.fit(train_series)
            logger.info("Model training completed")
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
        finally:
            # Detach the loss-history callback from the model's stored trainer
            # kwargs.  Darts persists pl_trainer_kwargs on the model so it can
            # rebuild the Lightning trainer on a later .fit(); leaving our
            # callback there makes the model unpicklable / unsaveable.  The
            # captured history lives in the separate ``loss_history`` dict, so
            # removing the callback here does not lose any data.
            if loss_history is not None:
                _strip_loss_history_callback(model)

    training_time = time.time() - start_time

    # XGBoost: read the per-boosting-round validation curve the booster kept.
    if model_type == "XGBoost" and loss_history is None:
        loss_history = _extract_xgb_eval_history(model)

    # Evaluate on train + validation splits.  last_points_only=True (inside
    # _eval_split) yields a single concatenated TimeSeries for calculate_metrics.
    training_metrics = _eval_split(
        model, train_series, train_covariates, train_future_covariates,
        lookback, horizon, model_type, scaler, "training",
    )
    validation_metrics = _eval_split(
        model, val_series, val_covariates, val_future_covariates,
        lookback, horizon, model_type, scaler, "validation",
    )

    # Training info — note: lookback/horizon here are in *steps*, not hours.
    # At hourly frequency steps == hours; at 15min frequency 24h = 96 steps.
    energy_kwh, watts_assumed = _estimate_training_energy_kwh(training_time, model_type)
    training_info = {
        "training_time_seconds": round(training_time, 2),
        "darts_model_class": DARTS_CLASS_NAMES.get(model_type, "unknown"),
        "lookback_steps": lookback,
        "horizon_steps": horizon,
        "energy_kwh": energy_kwh,
        "watts_assumed": watts_assumed,
        "environment": collect_environment(
            model_type,
            _requested_accelerator if model_type in _TORCH_MODEL_NAMES else None,
        ),
    }
    # Per-epoch learning curve (Torch models only).  Only recorded when the
    # callback captured at least one epoch; absent otherwise so downstream
    # consumers (report payload/JS) can hide the section gracefully.
    if loss_history and loss_history.get("epochs"):
        training_info["training_history"] = {
            "x_label": "epoch",
            "metric": "loss",
            **loss_history,
        }

    return model, training_metrics, validation_metrics, training_info


def generate_predictions(
    model: Any,
    series: TimeSeries,
    covariates: Optional[TimeSeries],
    lookback: int,
    horizon: int,
    future_covariates: Optional[TimeSeries] = None,
    model_type: Optional[str] = None,
     stride: Optional[int] = None,
     start: Optional[int] = None,
     last_points_only: bool = False,
     num_samples: int = 1,
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
            forecast time, e.g. weather forecasts). Supported by
            LinearRegression, XGBoost, and LSTM (BlockRNNModel is a
            MixedCovariatesTorchModel and accepts future_covariates in fit()
            and historical_forecasts()).
        model_type: Model type string for covariate gating. If None, falls
            back to class-name heuristics.
        stride: Step size between forecasts (defaults to horizon for
            non-overlapping windows). Passed through to historical_forecasts.
        start: Index at which to begin rolling forecasts (defaults to
            lookback).  Values smaller than lookback are clamped to lookback.
        last_points_only: If False (default), return the full forecast
            trajectory for every window so that all horizon steps are
            available for peak-day evaluation.  If True, return only the
            last predicted point per window (Darts default).

    Returns:
        Concatenated predictions as TimeSeries
    """
    # Resolve defaults
    effective_stride = stride if stride is not None else horizon
    effective_start = max(lookback, start) if start is not None else lookback

    # Cast series to match the model's trained dtype (e.g. float32 on MPS).
    # Darts reads dtype from the first train sample to determine model precision;
    # inference series must match or Lightning/MPS will reject the batch.
    series = _cast_series_to_model_dtype(series, model)
    covariates = _cast_series_to_model_dtype(covariates, model)
    future_covariates = _cast_series_to_model_dtype(future_covariates, model)

    # Check if this is a LocalForecastingModel (naive baselines)
    model_class_name = model.__class__.__name__
    is_local_model = model_class_name in LOCAL_MODEL_NAMES

    if is_local_model:
        # LocalForecastingModels don't support historical_forecasts
        # with retrain=False. Implement manual walk-forward validation
        logger.info(
            f"Using manual walk-forward validation for {model_class_name}"
        )
        return _manual_walk_forward(
            model, series, lookback, horizon,
            stride=effective_stride, start=effective_start,
        )

    # Determine which covariates this model type actually accepts.
    # Use explicit model_type when provided; fall back to class name.
    effective_type = model_type or model_class_name
    supports_future_covariates = effective_type not in PAST_COVARIATE_ONLY_MODELS

    # Hybrid models (TimesFM+Residual) cannot be dispatched through Darts'
    # historical_forecasts because they are not native Darts models.  Fall
    # back to a covariate-aware manual walk-forward.
    is_hybrid_model = (
        effective_type in HYBRID_MODELS
        or getattr(model, "_is_hybrid", False)
    )
    if is_hybrid_model:
        logger.info(
            f"Using manual walk-forward (covariate-aware) for hybrid model "
            f"{model_class_name}"
        )
        return _manual_walk_forward_with_covariates(
            model, series, lookback, horizon,
            past_covariates=covariates,
            future_covariates=future_covariates,
            stride=effective_stride,
            start=effective_start,
            last_points_only=last_points_only,
        )

    try:
        hf_kwargs: dict = {
            "start": effective_start,
            "forecast_horizon": horizon,
            "stride": effective_stride,
            "retrain": False,
            "verbose": False,
            "last_points_only": last_points_only,
        }
        # Probabilistic models: draw Monte-Carlo samples so callers can slice
        # quantiles from the returned stochastic series.  Point models keep 1.
        if num_samples and num_samples > 1 and getattr(
            model, "supports_probabilistic_prediction", False
        ):
            hf_kwargs["num_samples"] = int(num_samples)
        # ARIMA (FUTURE_ONLY_MODELS) only supports future covariates; skip past.
        if covariates is not None and effective_type not in FUTURE_ONLY_MODELS:
            hf_kwargs["past_covariates"] = covariates
        if future_covariates is not None and supports_future_covariates:
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
    stride: Optional[int] = None,
    start: Optional[int] = None,
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
        stride: Step size between forecasts (defaults to horizon)
        start: Index at which to begin the walk-forward (defaults to lookback)

    Returns:
        Concatenated predictions as TimeSeries
    """
    predictions_list = []
    effective_stride = stride if stride is not None else horizon
    start_idx = max(lookback, start) if start is not None else lookback

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

        # Move forward by stride steps
        start_idx += effective_stride

    # Concatenate all predictions
    if predictions_list:
        return _darts_concatenate(predictions_list, axis=0)
    else:
        # If no predictions, return empty prediction
        return model.predict(n=horizon)


def _manual_walk_forward_with_covariates(
    model: Any,
    series: TimeSeries,
    lookback: int,
    horizon: int,
    past_covariates: Optional[TimeSeries] = None,
    future_covariates: Optional[TimeSeries] = None,
    stride: Optional[int] = None,
    start: Optional[int] = None,
    last_points_only: bool = False,
) -> TimeSeries:
    """
    Walk-forward evaluation for models that Darts historical_forecasts cannot
    dispatch (e.g. TimesFMResidualHybrid).

    Unlike ``_manual_walk_forward`` this variant does NOT refit the model at
    each step — it reuses the already-trained ``model`` and only slides the
    input window forward, calling ``model.predict(n, series=window,
    past_covariates=..., future_covariates=...)`` on each stride.

    Args:
        model: Trained hybrid model exposing a Darts-like ``predict()`` API.
        series: Target series.
        lookback: Minimum lookback window (start position lower bound).
        horizon: Forecast horizon in steps.
        past_covariates: Full past covariate series (spanning both context
            and forecast region).  Optional.
        future_covariates: Full future covariate series (spanning both
            context and forecast region).  Optional.
        stride: Step size between windows.  Defaults to ``horizon``.
        start: Index at which to begin.  Defaults to ``lookback``.
        last_points_only: If True, keep only the last predicted point per
            window; if False, keep the full horizon trajectory.

    Returns:
        Concatenated predictions as a TimeSeries.
    """
    effective_stride = stride if stride is not None else horizon
    start_idx = max(lookback, start) if start is not None else lookback

    predictions_list: list[TimeSeries] = []
    last_points: list[TimeSeries] = []

    while start_idx + horizon <= len(series):
        window = series[:start_idx]
        try:
            pred = model.predict(
                n=horizon,
                series=window,
                past_covariates=past_covariates,
                future_covariates=future_covariates,
            )
        except Exception as e:
            logger.warning(
                "Hybrid walk-forward: predict failed at index %d (%s); "
                "stopping early.", start_idx, e,
            )
            break

        if last_points_only:
            last_points.append(pred[-1])
        else:
            predictions_list.append(pred)
        start_idx += effective_stride

    if last_points_only:
        if last_points:
            return _darts_concatenate(last_points, axis=0)
        # Fallback: empty prediction from the tail
        return model.predict(
            n=horizon,
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

    if predictions_list:
        return _darts_concatenate(predictions_list, axis=0, ignore_time_axis=True)
    return model.predict(
        n=horizon,
        series=series,
        past_covariates=past_covariates,
        future_covariates=future_covariates,
    )


def evaluate_model(
    model: Any,
    test_series: TimeSeries,
    test_covariates: Optional[TimeSeries] = None,
    test_future_covariates: Optional[TimeSeries] = None,
    lookback: int = 24,
    horizon: int = 6,
    scaler=None,
    model_type: Optional[str] = None,
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
        model_type: Model type string for covariate gating.

    Returns:
        Tuple of (metrics_dict, predictions_series)
    """
    predictions = generate_predictions(
        model, test_series, test_covariates, lookback, horizon,
        future_covariates=test_future_covariates,
        model_type=model_type,
    )

    metrics = calculate_metrics(
        test_series[lookback:],
        predictions,
        scaler=scaler,
    )

    return metrics, predictions
