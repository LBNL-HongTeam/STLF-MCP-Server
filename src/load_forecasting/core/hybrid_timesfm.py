"""
TimesFM 2.5 + Ridge residual correction hybrid model.

Motivation
----------
TimesFM 2.5 (as wrapped in ``darts.models.forecasting.timesfm2p5_model``) is a
target-only foundation model — it does not consume past or future covariates.
For building electrical load forecasting, weather (especially temperature) and
calendar effects are strong drivers of demand and cannot be ignored.

This module implements the residual-correction hybrid recommended in the
TimesFM documentation:

    1. Fine-tune TimesFM on the target series alone.
    2. Compute non-leaky in-sample residuals via ``historical_forecasts``
       (retrain=False, stride=horizon).
    3. Fit a Ridge regressor from the covariate design matrix to those
       residuals.
    4. At prediction time: base = TimesFM.predict, correction = Ridge.predict,
       final = base + correction.

Feature engineering
-------------------
Covariate columns are used verbatim, except for an auto-detected
``temperature`` column, for which we add:

    * HDD (heating degree hours) = max(0, base_hdd - T)
    * CDD (cooling degree hours) = max(0, T - base_cdd)
    * A 3-knot natural cubic spline basis (knots at 25/50/75 pct of the
      training temperature distribution).

The engineered features capture the U-shape between temperature and load that a
plain linear model cannot represent.  All other covariates (relative humidity,
solar radiation, cyclic calendar sin/cos, lag features, etc.) are passed
through unchanged — they are already produced by ``ForecastingDataLoader``.

Deterministic first pass
------------------------
This implementation is deterministic (no probabilistic quantile forecasts).
Adding per-quantile residual regressors on top of ``QuantileRegression`` is a
straightforward follow-up if quantile forecasts become desirable.

Save/load layout
----------------
Given a base path ``<path>``, the following sibling files are written:

    <path>                — an empty sentinel file (so registry save/load
                            contract expecting a single file still works;
                            actual content lives in the sibling files below)
    <path>.timesfm.ckpt   — Darts TimesFM2p5Model.save output
    <path>.ridge.joblib   — pickled StandardScaler + Ridge pipeline
    <path>.meta.json      — feature spec (temperature column name, spline
                            knots, HDD/CDD bases, feature order, hyper-params)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from darts import TimeSeries, concatenate
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

try:
    from darts.models.forecasting.timesfm2p5_model import TimesFM2p5Model
    _HAS_TIMESFM = True
except Exception:  # pragma: no cover - import guard
    TimesFM2p5Model = None  # type: ignore[assignment]
    _HAS_TIMESFM = False

logger = logging.getLogger(__name__)


# --- Temperature detection ---------------------------------------------------
# Substring patterns (case-insensitive) used to identify the temperature
# covariate column.  Kept in sync with weather_fetcher renames.
_TEMPERATURE_PATTERNS: tuple[str, ...] = (
    "temperature",
    "temp",
    "t_out",
    "tout",
    "t2m",
)


def _detect_temperature_column(columns: list[str]) -> Optional[str]:
    """Return the first column name matching a temperature pattern, else None.

    Priority: exact matches on "temperature" first, then substring matches.
    Case-insensitive.  Skips columns starting with ``cal_`` (calendar) and
    ``lag_`` (target lags) to avoid false positives.
    """
    lowered = [(c, c.lower()) for c in columns]

    # Exact match first
    for orig, low in lowered:
        if low == "temperature":
            return orig

    # Substring match, ignoring calendar and lag features
    for orig, low in lowered:
        if orig.startswith("cal_") or orig.startswith("lag_"):
            continue
        for pat in _TEMPERATURE_PATTERNS:
            if pat in low:
                return orig
    return None


# --- Feature engineering -----------------------------------------------------

def _spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Truncated cubic spline basis with the given interior knots.

    For 3 interior knots this returns 3 columns: ``(x - knot_i)_+^3``.  Combined
    with the raw temperature column and a constant intercept in the Ridge
    design matrix, this yields a natural-cubic-spline-like basis capable of
    representing the U-shape between temperature and load.
    """
    x = x.astype(np.float64, copy=False)
    diffs = x[:, None] - knots[None, :]
    return np.maximum(diffs, 0.0) ** 3


def _build_temperature_features(
    temp_values: np.ndarray,
    knots: np.ndarray,
    hdd_base: float,
    cdd_base: float,
) -> tuple[np.ndarray, list[str]]:
    """Return (feature_matrix, feature_names) for the temperature column."""
    hdd = np.maximum(hdd_base - temp_values, 0.0)
    cdd = np.maximum(temp_values - cdd_base, 0.0)
    spl = _spline_basis(temp_values, knots)
    feats = np.column_stack([hdd, cdd, spl])
    names = ["temp_hdd", "temp_cdd"] + [f"temp_spline_{i}" for i in range(spl.shape[1])]
    return feats, names


# --- Hybrid model ------------------------------------------------------------

class TimesFMResidualHybrid:
    """TimesFM 2.5 backbone plus Ridge residual correction on covariates.

    Behaves like a Darts global forecasting model for the subset of the API
    used by this codebase: ``fit(series, past_covariates=..., future_covariates=...)``
    and ``predict(n, series=..., past_covariates=..., future_covariates=...)``.

    It is *not* a Darts model — Darts' ``historical_forecasts`` cannot dispatch
    to it.  ``trainer.generate_predictions`` handles rolling-window evaluation
    via a dedicated walk-forward branch.
    """

    # Sentinel that class-name checks in trainer.py can rely on
    _is_hybrid: bool = True

    def __init__(
        self,
        timesfm_kwargs: dict,
        horizon: int,
        alpha: float = 1.0,
        hdd_base: float = 18.0,
        cdd_base: float = 22.0,
        n_knots: int = 3,
    ) -> None:
        if not _HAS_TIMESFM:
            raise ImportError(
                "TimesFM2p5Model could not be imported; TimesFMResidualHybrid "
                "requires the darts TimesFM 2.5 wrapper and huggingface_hub."
            )
        self.timesfm_kwargs = dict(timesfm_kwargs)
        self.horizon = int(horizon)
        self.alpha = float(alpha)
        self.hdd_base = float(hdd_base)
        self.cdd_base = float(cdd_base)
        self.n_knots = int(n_knots)

        # Populated at fit time
        self.timesfm: Optional[Any] = None
        self.ridge: Optional[Ridge] = None
        self.feature_scaler: Optional[StandardScaler] = None
        self.feature_names: list[str] = []
        self.temperature_col: Optional[str] = None
        self.knots: Optional[np.ndarray] = None
        self.past_cov_columns: list[str] = []
        self.future_cov_columns: list[str] = []
        # Track dtype so predict() can align inputs
        self._trained_float32: bool = False

    # ------------------------------------------------------------------
    # Feature-matrix construction
    # ------------------------------------------------------------------

    def _extract_covariate_df(
        self,
        past_covariates: Optional[TimeSeries],
        future_covariates: Optional[TimeSeries],
        time_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Slice past/future covariates to ``time_index`` and return as DataFrame.

        Missing timestamps are filled by forward-then-back fill; a fully-empty
        column will simply be zero after imputation of the empty NaN series.
        """
        frames: list[pd.DataFrame] = []

        for series, expected_cols in (
            (past_covariates, self.past_cov_columns),
            (future_covariates, self.future_cov_columns),
        ):
            if not expected_cols:
                continue
            if series is None:
                # Model was trained with these covariates but none supplied now
                logger.warning(
                    "Hybrid predict: expected covariate columns %s but none "
                    "were provided; filling with zeros.",
                    expected_cols,
                )
                frames.append(
                    pd.DataFrame(
                        0.0, index=time_index, columns=expected_cols
                    )
                )
                continue
            # Darts >=0.30 renamed pd_dataframe to to_dataframe; fall back for older versions.
            if hasattr(series, "to_dataframe"):
                df = series.to_dataframe()
            else:  # pragma: no cover - legacy Darts fallback
                df = series.pd_dataframe()
            # Only keep columns known at fit time (drop unexpected extras)
            keep = [c for c in expected_cols if c in df.columns]
            df = df[keep]
            # Reindex to the target time_index; ffill/bfill any small gaps
            df = df.reindex(time_index).ffill().bfill().fillna(0.0)
            # Add zero columns for any expected column absent from the series
            for c in expected_cols:
                if c not in df.columns:
                    df[c] = 0.0
            df = df[expected_cols]
            frames.append(df)

        if not frames:
            return pd.DataFrame(index=time_index)
        return pd.concat(frames, axis=1)

    def _build_design_matrix(
        self,
        cov_df: pd.DataFrame,
        fit_time: bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """Build the Ridge design matrix from a covariate DataFrame.

        At fit time, chooses temperature knots and stores the final feature
        name order.  At predict time, uses the stored knots and reorders
        columns to match the fitted layout.
        """
        columns = list(cov_df.columns)
        base_matrix = cov_df.to_numpy(dtype=np.float64, copy=False)
        base_names = list(columns)

        temp_feats: Optional[np.ndarray] = None
        temp_names: list[str] = []

        if fit_time:
            self.temperature_col = _detect_temperature_column(columns)
            if self.temperature_col is not None:
                temp_values = cov_df[self.temperature_col].to_numpy(dtype=np.float64)
                # Choose knots as interior quantiles of training temperatures
                qs = np.linspace(0.1, 0.9, self.n_knots)
                self.knots = np.quantile(temp_values, qs)
                temp_feats, temp_names = _build_temperature_features(
                    temp_values, self.knots, self.hdd_base, self.cdd_base
                )
                logger.info(
                    "Hybrid: temperature column detected: %s (knots=%s)",
                    self.temperature_col,
                    np.round(self.knots, 2).tolist(),
                )
            else:
                logger.info(
                    "Hybrid: no temperature column detected; skipping HDD/CDD/spline features."
                )
        else:
            if self.temperature_col is not None and self.knots is not None:
                if self.temperature_col in cov_df.columns:
                    temp_values = cov_df[self.temperature_col].to_numpy(dtype=np.float64)
                    temp_feats, temp_names = _build_temperature_features(
                        temp_values, self.knots, self.hdd_base, self.cdd_base
                    )
                else:
                    # Column missing at predict time — emit zeros for its features
                    n = len(cov_df)
                    temp_feats = np.zeros((n, 2 + self.n_knots), dtype=np.float64)
                    temp_names = ["temp_hdd", "temp_cdd"] + [
                        f"temp_spline_{i}" for i in range(self.n_knots)
                    ]

        if temp_feats is not None:
            X = np.column_stack([base_matrix, temp_feats])
            names = base_names + temp_names
        else:
            X = base_matrix
            names = base_names

        if fit_time:
            self.feature_names = names
        else:
            # Reorder to match training-time layout
            if names != self.feature_names:
                current_lookup = {n: i for i, n in enumerate(names)}
                cols_ordered: list[np.ndarray] = []
                for fname in self.feature_names:
                    if fname in current_lookup:
                        cols_ordered.append(X[:, current_lookup[fname]])
                    else:
                        cols_ordered.append(np.zeros(X.shape[0], dtype=np.float64))
                X = np.column_stack(cols_ordered)

        return X, self.feature_names

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _fit_noop_ridge(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a degenerate Ridge + scaler on ``X``/``y`` (fallback corrections)."""
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(self.feature_scaler.transform(X), y)
        self.feature_names = [f"dummy_{i}" for i in range(X.shape[1])]

    def fit(
        self,
        series: TimeSeries,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
    ) -> "TimesFMResidualHybrid":
        """Fine-tune the TimesFM backbone then fit Ridge on residuals."""
        self._trained_float32 = series.dtype == np.float32
        self.past_cov_columns = (
            list(past_covariates.components) if past_covariates is not None else []
        )
        self.future_cov_columns = (
            list(future_covariates.components) if future_covariates is not None else []
        )

        # 1. TimesFM fit (target only)
        self.timesfm = TimesFM2p5Model(**self.timesfm_kwargs)
        logger.info("Hybrid: fitting TimesFM backbone")
        self.timesfm.fit(series)

        # 2. Non-leaky in-sample forecasts via historical_forecasts
        lookback = int(self.timesfm_kwargs.get("input_chunk_length", 512))
        stride = self.horizon
        if len(series) <= lookback + self.horizon:
            logger.warning(
                "Hybrid: series too short for residual training window "
                "(len=%d, lookback=%d, horizon=%d); Ridge will be fit on an "
                "empty set and act as a no-op correction.",
                len(series), lookback, self.horizon,
            )
            # Fit a trivial zero-correction ridge so predict() has valid
            # (all-zero) coefficients for downstream use.
            n_feat = max(1, len(self.past_cov_columns) + len(self.future_cov_columns))
            self._fit_noop_ridge(np.zeros((2, n_feat)), np.zeros(2))
            return self

        logger.info("Hybrid: computing in-sample residuals (stride=%d)", stride)
        try:
            hist_fc = self.timesfm.historical_forecasts(
                series,
                start=lookback,
                forecast_horizon=self.horizon,
                stride=stride,
                retrain=False,
                last_points_only=True,
                verbose=False,
            )
        except Exception as e:
            logger.warning(
                "Hybrid: historical_forecasts failed (%s); Ridge will act as a "
                "no-op correction.", e,
            )
            hist_fc = None

        if hist_fc is None or len(hist_fc) == 0:
            n_feat = max(1, len(self.past_cov_columns) + len(self.future_cov_columns))
            self._fit_noop_ridge(np.zeros((2, n_feat)), np.zeros(2))
            return self

        # Align residuals with covariates on the shared time index
        actual = series.slice_intersect(hist_fc)
        pred = hist_fc.slice_intersect(actual)
        residuals = actual.values().flatten() - pred.values().flatten()
        time_index = actual.time_index

        cov_df = self._extract_covariate_df(past_covariates, future_covariates, time_index)

        if cov_df.shape[1] == 0:
            # No covariates supplied to a "residual" model — degenerate case;
            # fit a zero-correction ridge on a single feature so predict works.
            logger.warning(
                "Hybrid: no covariate columns supplied; Ridge will act as a "
                "no-op correction."
            )
            self._fit_noop_ridge(np.zeros((len(residuals), 1)), residuals)
            return self

        # 3. Build design matrix (fit-time: choose knots, freeze feature order)
        X, _ = self._build_design_matrix(cov_df, fit_time=True)

        # 4. Standardize features and fit Ridge on residuals
        self.feature_scaler = StandardScaler()
        X_scaled = self.feature_scaler.fit_transform(X)
        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, residuals)
        logger.info(
            "Hybrid: Ridge fit on %d residual rows with %d features (alpha=%.3f)",
            X_scaled.shape[0], X_scaled.shape[1], self.alpha,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(
        self,
        n: int,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
    ) -> TimeSeries:
        """Return TimesFM prediction plus Ridge covariate correction."""
        if self.timesfm is None or self.ridge is None or self.feature_scaler is None:
            raise RuntimeError("TimesFMResidualHybrid.predict called before fit()")

        # 1. Base forecast
        base = self.timesfm.predict(n=n, series=series)

        # 2. Extract covariates aligned to base timestamps
        cov_df = self._extract_covariate_df(
            past_covariates, future_covariates, base.time_index
        )

        # If model was fit with no real covariates (dummy path), skip correction
        no_real_covariates = (
            not self.past_cov_columns
            and not self.future_cov_columns
        )
        if no_real_covariates or cov_df.shape[1] == 0:
            return base

        X, _ = self._build_design_matrix(cov_df, fit_time=False)
        # Pad or trim columns if unexpected shape (safety net; should not fire)
        expected_n = self.feature_scaler.mean_.shape[0]
        if X.shape[1] != expected_n:
            if X.shape[1] < expected_n:
                pad = np.zeros((X.shape[0], expected_n - X.shape[1]))
                X = np.column_stack([X, pad])
            else:
                X = X[:, :expected_n]

        X_scaled = self.feature_scaler.transform(X)
        correction = self.ridge.predict(X_scaled)

        # Cast correction to base dtype so TimeSeries concatenation is clean
        base_arr = base.all_values()
        if base_arr.shape[0] != correction.shape[0]:
            # Length mismatch — trim to shorter
            n_out = min(base_arr.shape[0], correction.shape[0])
            base_arr = base_arr[:n_out]
            correction = correction[:n_out]
            time_index = base.time_index[:n_out]
        else:
            time_index = base.time_index

        final_arr = base_arr + correction.reshape(-1, 1, 1).astype(base_arr.dtype)
        return TimeSeries.from_times_and_values(
            time_index,
            final_arr,
            columns=list(base.components),
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the hybrid to sibling files rooted at ``path``.

        The primary ``path`` is written as a small sentinel file so the
        registry's file-existence check (``model_path.exists()``) passes.  All
        real content lives in ``path + ".timesfm.ckpt"``, ``".ridge.joblib"``,
        and ``".meta.json"``.
        """
        if self.timesfm is None or self.ridge is None or self.feature_scaler is None:
            raise RuntimeError("Cannot save an unfitted TimesFMResidualHybrid")

        p = Path(path)
        timesfm_path = p.with_name(p.name + ".timesfm.ckpt")
        ridge_path = p.with_name(p.name + ".ridge.joblib")
        meta_path = p.with_name(p.name + ".meta.json")

        # TimesFM writes a Lightning checkpoint (plus a .ckpt sibling itself)
        self.timesfm.save(str(timesfm_path))

        # Ridge + StandardScaler via pickle (joblib not a mandatory dep)
        with open(ridge_path, "wb") as f:
            pickle.dump(
                {
                    "ridge": self.ridge,
                    "feature_scaler": self.feature_scaler,
                },
                f,
            )

        # Metadata: everything needed to reconstruct feature engineering
        meta = {
            "timesfm_kwargs": self._json_safe_kwargs(self.timesfm_kwargs),
            "horizon": self.horizon,
            "alpha": self.alpha,
            "hdd_base": self.hdd_base,
            "cdd_base": self.cdd_base,
            "n_knots": self.n_knots,
            "feature_names": self.feature_names,
            "temperature_col": self.temperature_col,
            "knots": self.knots.tolist() if self.knots is not None else None,
            "past_cov_columns": self.past_cov_columns,
            "future_cov_columns": self.future_cov_columns,
            "trained_float32": self._trained_float32,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Sentinel primary file
        with open(p, "w", encoding="utf-8") as f:
            f.write("TimesFMResidualHybrid sentinel — see sibling files\n")

    @staticmethod
    def _json_safe_kwargs(kwargs: dict) -> dict:
        """Drop non-JSON-serializable entries (e.g. pl_trainer_kwargs)."""
        safe: dict = {}
        for k, v in kwargs.items():
            try:
                json.dumps(v)
                safe[k] = v
            except (TypeError, ValueError):
                # Skip non-serializable; reconstructed with defaults on load
                continue
        return safe

    @classmethod
    def load(cls, path: str) -> "TimesFMResidualHybrid":
        if not _HAS_TIMESFM:
            raise ImportError(
                "TimesFM2p5Model wrapper not available; cannot load TimesFMResidualHybrid."
            )
        p = Path(path)
        timesfm_path = p.with_name(p.name + ".timesfm.ckpt")
        ridge_path = p.with_name(p.name + ".ridge.joblib")
        meta_path = p.with_name(p.name + ".meta.json")

        if not meta_path.exists():
            raise FileNotFoundError(f"Hybrid meta file missing: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            timesfm_kwargs=meta.get("timesfm_kwargs", {}),
            horizon=meta["horizon"],
            alpha=meta.get("alpha", 1.0),
            hdd_base=meta.get("hdd_base", 18.0),
            cdd_base=meta.get("cdd_base", 22.0),
            n_knots=meta.get("n_knots", 3),
        )

        instance.timesfm = TimesFM2p5Model.load(str(timesfm_path), weights_only=False)

        with open(ridge_path, "rb") as f:
            payload = pickle.load(f)
        instance.ridge = payload["ridge"]
        instance.feature_scaler = payload["feature_scaler"]

        instance.feature_names = list(meta.get("feature_names", []))
        instance.temperature_col = meta.get("temperature_col")
        knots = meta.get("knots")
        instance.knots = np.asarray(knots, dtype=np.float64) if knots is not None else None
        instance.past_cov_columns = list(meta.get("past_cov_columns", []))
        instance.future_cov_columns = list(meta.get("future_cov_columns", []))
        instance._trained_float32 = bool(meta.get("trained_float32", False))
        return instance

    # ------------------------------------------------------------------
    # Small conveniences so this class integrates with existing code paths
    # ------------------------------------------------------------------

    @property
    def train_sample(self):
        """Delegate to TimesFM so ``_cast_series_to_model_dtype`` works."""
        if self.timesfm is None:
            return None
        return getattr(self.timesfm, "train_sample", None)
