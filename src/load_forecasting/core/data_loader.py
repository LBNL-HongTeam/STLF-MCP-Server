"""
Load CSV data and convert to Darts TimeSeries format.

Handles:
- Column auto-detection from patterns
- Datetime parsing and timezone handling
- Missing value interpolation with gap size enforcement
- Frequency inference and validation
- Target non-negativity and coverage checks
- Calendar feature engineering (always on)
- Lag feature engineering from target (24h, 48h, 168h)
- Future covariates support (columns known at forecast time, e.g. weather forecasts)
- Data validation against spec constraints
"""

from pathlib import Path
from typing import Optional
import logging
import math

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from .spec_loader import get_auto_detect_patterns
from .frequency_utils import FREQ_TO_STEPS_PER_HOUR, hours_to_steps

logger = logging.getLogger(__name__)


# Minimum fraction of non-null target values required (90 % coverage)
_MIN_COVERAGE = 0.90

# Lag offsets (in hours) added as past covariates from the target column
_DEFAULT_LAG_HOURS = [24, 48, 168]

# Calendar feature column names injected automatically
_CALENDAR_COLS = [
    "cal_hour",
    "cal_dow",
    "cal_month",
    "cal_is_weekend",
    "cal_hour_sin",
    "cal_hour_cos",
]


class DataLoadError(Exception):
    """Raised when data loading or validation fails."""

    pass


def _infer_frequency(index: pd.DatetimeIndex) -> Optional[str]:
    """
    Infer the data frequency from a DatetimeIndex.

    Returns one of '15min', '30min', 'h', or None if it cannot be determined.
    Uses the median gap between consecutive timestamps to be robust to a few
    duplicates or outliers left after deduplication.
    """
    if len(index) < 2:
        return None

    diffs = index.to_series().diff().dropna()
    median_gap = diffs.median()

    tolerance = pd.Timedelta(minutes=1)
    if abs(median_gap - pd.Timedelta(minutes=15)) <= tolerance:
        return "15min"
    if abs(median_gap - pd.Timedelta(minutes=30)) <= tolerance:
        return "30min"
    if abs(median_gap - pd.Timedelta(hours=1)) <= tolerance:
        return "h"
    return None


class ForecastingDataLoader:
    """Load and preprocess CSV data for forecasting."""

    def __init__(
        self,
        csv_path: str,
        column_mapping: Optional[dict] = None,
        frequency: str = "h",
        missing_value_strategy: str = "interpolate",
        max_gap: str = "2h",
        add_calendar_features: bool = True,
        lag_hours: Optional[list] = None,
        dataframe: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize data loader.

        Args:
            csv_path: Path to CSV file.
            column_mapping: Map CSV columns to roles (datetime, target,
                past_covariates, future_covariates).  Auto-detected if not
                provided.  ``future_covariates`` must be explicitly listed —
                they are never auto-detected.
            frequency: Expected data frequency ('15min', '30min', 'h').
                Validated against the actual timestamps in the file.
            missing_value_strategy: How to handle missing values
                ('interpolate', 'drop', 'error').
            max_gap: Maximum contiguous gap to interpolate (e.g. '2h').
                Larger gaps are warned about but not filled; you will see NaN
                values remain after interpolation.
            add_calendar_features: If True (default), inject hour-of-day,
                day-of-week, month, is_weekend, and cyclic hour encoding as
                past covariates.
            lag_hours: List of hour offsets for lag features derived from the
                target column.  Defaults to [24, 48, 168].  Pass [] to
                disable lag features.
            dataframe: Pre-loaded DataFrame to use instead of reading
                ``csv_path`` from disk.  When provided, ``csv_path`` is still
                recorded for metadata purposes but the file is not re-read.

        Raises:
            DataLoadError: If the file is not found, cannot be parsed,
                or fails any validation check.
        """
        self.csv_path = Path(csv_path)
        self.frequency = frequency
        self.missing_value_strategy = missing_value_strategy
        self.max_gap = pd.Timedelta(max_gap)
        self.add_calendar_features = add_calendar_features
        self.lag_hours = lag_hours if lag_hours is not None else _DEFAULT_LAG_HOURS

        if dataframe is not None:
            # Caller already has the DataFrame — skip disk I/O entirely.
            self.df = dataframe.copy()
        else:
            # ------------------------------------------------------------------
            # File existence
            # ------------------------------------------------------------------
            if not self.csv_path.exists():
                raise DataLoadError(
                    f"CSV file not found: {csv_path}\n"
                    "Check that the path is correct and the file is accessible."
                )

            # ------------------------------------------------------------------
            # Read CSV
            # ------------------------------------------------------------------
            try:
                self.df = pd.read_csv(csv_path)
            except Exception as e:
                raise DataLoadError(
                    f"Failed to read CSV '{csv_path}': {e}\n"
                    "Ensure the file is a valid, non-corrupted CSV."
                )

        if self.df.empty:
            raise DataLoadError(
                f"CSV file is empty: {csv_path}\n"
                "The file must contain at least one row of data."
            )

        # ------------------------------------------------------------------
        # Column mapping and basic column validation
        # ------------------------------------------------------------------
        self.column_mapping = self._resolve_mapping(column_mapping)
        self._validate_columns()

        # ------------------------------------------------------------------
        # Preprocess (datetime parse, dedup, sort, gap handling)
        # NOTE: coverage is checked BEFORE interpolation so that genuinely
        # sparse data is rejected rather than silently filled.
        # ------------------------------------------------------------------
        self._preprocess_pre_interp()
        self._validate_target_coverage()  # must run before _interpolate_gaps
        self._finish_preprocess()

        # ------------------------------------------------------------------
        # Validate frequency against actual data
        # ------------------------------------------------------------------
        self._validate_frequency()

        # ------------------------------------------------------------------
        # Validate target quality post-interpolation (non-negativity)
        # ------------------------------------------------------------------
        self._validate_target_values()

        # ------------------------------------------------------------------
        # Feature engineering
        # ------------------------------------------------------------------
        if self.add_calendar_features:
            self._add_calendar_features()

        if self.lag_hours:
            self._add_lag_features()

        # ------------------------------------------------------------------
        # Scalers (fit during training, reuse during evaluation)
        # ------------------------------------------------------------------
        self.target_scaler: Optional[Scaler] = None
        self.covariate_scaler: Optional[Scaler] = None
        self.future_covariate_scaler: Optional[Scaler] = None

    # ------------------------------------------------------------------
    # Column resolution
    # ------------------------------------------------------------------

    def _resolve_mapping(self, mapping: Optional[dict]) -> dict:
        """Auto-detect columns if mapping not provided."""
        if mapping:
            # Ensure past_covariates and future_covariates keys always exist
            mapping = dict(mapping)
            if "past_covariates" not in mapping:
                mapping["past_covariates"] = []
            if "future_covariates" not in mapping:
                mapping["future_covariates"] = []
            return mapping

        patterns = get_auto_detect_patterns("train_forecast_model")

        # Fallback patterns if spec is not loaded
        if not patterns:
            patterns = {
                "datetime": ["datetime", "timestamp", "date", "time", "dt"],
                "target": ["kwh", "load", "power", "energy", "electricity", "demand"],
                "past_covariates": ["temp", "temperature", "rh", "humidity", "solar", "wind"],
            }

        resolved: dict = {}

        # --- Datetime column ---
        for col in self.df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns.get("datetime", [])):
                resolved["datetime"] = col
                break

        if "datetime" not in resolved:
            first_col = self.df.columns[0]
            try:
                pd.to_datetime(self.df[first_col].head(10))
                resolved["datetime"] = first_col
                logger.info("Auto-detected datetime column (positional fallback): %s", first_col)
            except Exception:
                pass

        # --- Target column ---
        for col in self.df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns.get("target", [])):
                resolved["target"] = col
                break

        if "target" not in resolved:
            for col in self.df.columns:
                if col != resolved.get("datetime") and pd.api.types.is_numeric_dtype(self.df[col]):
                    resolved["target"] = col
                    logger.info("Auto-detected target column (first numeric): %s", col)
                    break

        # --- Covariate columns ---
        covariate_patterns = patterns.get("past_covariates", [])
        resolved["past_covariates"] = [
            col
            for col in self.df.columns
            if any(p in col.lower() for p in covariate_patterns)
            and col not in [resolved.get("datetime"), resolved.get("target")]
        ]

        # Future covariates are never auto-detected — they must be explicitly
        # listed by the caller because their semantics differ from past covariates.
        resolved["future_covariates"] = []

        logger.info("Resolved column mapping: %s", resolved)
        return resolved

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_columns(self) -> None:
        """Validate that required columns are present."""
        if "datetime" not in self.column_mapping:
            raise DataLoadError(
                "Could not identify a datetime column automatically.\n"
                "Provide column_mapping={'datetime': '<your_column_name>', ...} "
                "or rename the column so it contains one of: "
                "datetime, timestamp, date, time, dt."
            )

        if "target" not in self.column_mapping:
            raise DataLoadError(
                "Could not identify a target (load) column automatically.\n"
                "Provide column_mapping={'target': '<your_column_name>', ...} "
                "or rename the column so it contains one of: "
                "kwh, load, power, energy, electricity, demand."
            )

        dt_col = self.column_mapping["datetime"]
        target_col = self.column_mapping["target"]

        if dt_col not in self.df.columns:
            available = ", ".join(self.df.columns.tolist())
            raise DataLoadError(
                f"Datetime column '{dt_col}' not found in CSV.\n"
                f"Available columns: {available}"
            )

        if target_col not in self.df.columns:
            available = ", ".join(self.df.columns.tolist())
            raise DataLoadError(
                f"Target column '{target_col}' not found in CSV.\n"
                f"Available columns: {available}"
            )

        # Warn and drop any past covariate columns that don't exist
        cov_cols = self.column_mapping.get("past_covariates", [])
        missing_covs = [c for c in cov_cols if c not in self.df.columns]
        if missing_covs:
            logger.warning(
                "Past covariate column(s) not found in CSV and will be skipped: %s",
                missing_covs,
            )
            self.column_mapping["past_covariates"] = [
                c for c in cov_cols if c in self.df.columns
            ]

        # Warn and drop any future covariate columns that don't exist
        fut_cov_cols = self.column_mapping.get("future_covariates", [])
        missing_fut_covs = [c for c in fut_cov_cols if c not in self.df.columns]
        if missing_fut_covs:
            logger.warning(
                "Future covariate column(s) not found in CSV and will be skipped: %s",
                missing_fut_covs,
            )
            self.column_mapping["future_covariates"] = [
                c for c in fut_cov_cols if c in self.df.columns
            ]

    def _validate_frequency(self) -> None:
        """
        Infer the actual data frequency and compare it to the declared one.

        Stores the result as ``self.inferred_frequency`` for callers that need
        it (e.g. inspect_data) to avoid a repeated call to _infer_frequency.

        Logs a warning (not an error) when they differ so that downstream
        Darts operations can still attempt to process the data — Darts will
        raise its own error if the index is truly irregular.
        """
        inferred = _infer_frequency(self.df.index)
        self.inferred_frequency: Optional[str] = inferred

        if inferred is None:
            logger.warning(
                "Could not infer data frequency from timestamps. "
                "Declared frequency '%s' will be used as-is.",
                self.frequency,
            )
            return

        if inferred != self.frequency:
            logger.warning(
                "Declared frequency '%s' does not match inferred frequency '%s'. "
                "The data will be processed at the declared frequency '%s'. "
                "If results look wrong, set frequency='%s' in your call.",
                self.frequency,
                inferred,
                self.frequency,
                inferred,
            )
        else:
            logger.info("Data frequency confirmed: %s", self.frequency)

    def _validate_target_coverage(self) -> None:
        """
        Check that the target column has sufficient non-null coverage.

        This runs BEFORE interpolation so that genuinely sparse data is
        rejected rather than silently patched.
        """
        target_col = self.column_mapping["target"]
        series = self.df[target_col]
        n_total = len(series)
        n_null = int(series.isna().sum())
        coverage = (n_total - n_null) / n_total if n_total > 0 else 0.0

        if coverage < _MIN_COVERAGE:
            raise DataLoadError(
                f"Target column '{target_col}' has only {coverage:.1%} non-null values "
                f"({n_null} missing out of {n_total}). "
                f"At least {_MIN_COVERAGE:.0%} coverage is required.\n"
                "Consider cleaning the data or using a different CSV."
            )

    def _validate_target_values(self) -> None:
        """
        Warn if the target contains negative values after interpolation.

        Building load is physically non-negative; negatives indicate a data
        quality issue but are not treated as a hard error.
        """
        target_col = self.column_mapping["target"]
        series = self.df[target_col].dropna()
        if len(series) > 0 and (series < 0).any():
            n_neg = int((series < 0).sum())
            logger.warning(
                "Target column '%s' contains %d negative value(s). "
                "Building load is expected to be non-negative. "
                "Verify the data source.",
                target_col,
                n_neg,
            )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_pre_interp(self) -> None:
        """Parse datetime, deduplicate, and sort — but do NOT fill missing values yet."""
        dt_col = self.column_mapping["datetime"]

        # Parse datetime
        try:
            self.df[dt_col] = pd.to_datetime(self.df[dt_col], utc=True)
        except Exception:
            try:
                self.df[dt_col] = pd.to_datetime(self.df[dt_col])
                if self.df[dt_col].dt.tz is None:
                    self.df[dt_col] = self.df[dt_col].dt.tz_localize("UTC")
            except Exception as e:
                raise DataLoadError(
                    f"Failed to parse datetime column '{dt_col}': {e}\n"
                    "Ensure the column contains valid date/time strings "
                    "(e.g. '2023-01-01 00:00:00', ISO 8601, or Unix timestamps)."
                )

        # Set index and sort
        self.df = self.df.set_index(dt_col).sort_index()

        # Remove duplicate indices
        n_dups = int(self.df.index.duplicated().sum())
        if n_dups:
            logger.warning(
                "%d duplicate timestamp(s) found; keeping the first occurrence.", n_dups
            )
            self.df = self.df[~self.df.index.duplicated(keep="first")]

    def _finish_preprocess(self) -> None:
        """Apply the chosen missing-value strategy (interpolate / drop / error)."""
        if self.missing_value_strategy == "interpolate":
            self._interpolate_gaps()
        elif self.missing_value_strategy == "drop":
            self.df = self.df.dropna()
        elif self.missing_value_strategy == "error":
            if self.df.isna().any().any():
                raise DataLoadError(
                    "Data contains missing values and missing_value_strategy='error'.\n"
                    "Either clean the data or switch to missing_value_strategy='interpolate'."
                )

    def _interpolate_gaps(self) -> None:
        """Interpolate small gaps; warn (do not fill) on large gaps."""
        time_diff = self.df.index.to_series().diff()
        large_gaps = time_diff > self.max_gap
        n_large = int(large_gaps.sum())

        if n_large:
            gap_locs = self.df.index[large_gaps].tolist()
            logger.warning(
                "%d gap(s) larger than %s found. "
                "Only gaps up to %s will be interpolated. "
                "First large gap starts at: %s",
                n_large,
                self.max_gap,
                self.max_gap,
                gap_locs[0] if gap_locs else "N/A",
            )

        target_col = self.column_mapping["target"]
        if self.df[target_col].isna().any():
            self.df[target_col] = self.df[target_col].interpolate(method="time")

        for cov_col in self.column_mapping.get("past_covariates", []):
            if cov_col in self.df.columns and self.df[cov_col].isna().any():
                self.df[cov_col] = self.df[cov_col].interpolate(method="time")

        for cov_col in self.column_mapping.get("future_covariates", []):
            if cov_col in self.df.columns and self.df[cov_col].isna().any():
                self.df[cov_col] = self.df[cov_col].interpolate(method="time")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _add_calendar_features(self) -> None:
        """
        Inject calendar features derived from the DatetimeIndex.

        Added columns (all prefixed with 'cal_'):
            cal_hour         : 0-23
            cal_dow          : 0 (Mon) - 6 (Sun)
            cal_month        : 1-12
            cal_is_weekend   : 0 or 1
            cal_hour_sin     : sin(2π * hour / 24)
            cal_hour_cos     : cos(2π * hour / 24)

        All new columns are appended to past_covariates in column_mapping.
        """
        idx = self.df.index
        hour = idx.hour.astype(float)

        _two_pi_over_24 = 2 * math.pi / 24  # constant — computed once per call
        self.df["cal_hour"] = hour
        self.df["cal_dow"] = idx.dayofweek.astype(float)
        self.df["cal_month"] = idx.month.astype(float)
        self.df["cal_is_weekend"] = (idx.dayofweek >= 5).astype(float)
        self.df["cal_hour_sin"] = np.sin(hour * _two_pi_over_24)
        self.df["cal_hour_cos"] = np.cos(hour * _two_pi_over_24)

        existing = set(self.column_mapping.get("past_covariates", []))
        new_cols = [c for c in _CALENDAR_COLS if c not in existing]
        self.column_mapping["past_covariates"] = (
            self.column_mapping.get("past_covariates", []) + new_cols
        )
        logger.info("Added calendar features: %s", new_cols)

    def _add_lag_features(self) -> None:
        """
        Create lag features from the target column and add them as past
        covariates.

        For each offset in self.lag_hours, a column named
        'lag_<offset>h' is created.  Rows at the start of the series where
        the lag cannot be computed contain NaN and are forward-filled from
        the first valid value to avoid introducing new gaps.
        """
        target_col = self.column_mapping["target"]
        added: list[str] = []

        for h in self.lag_hours:
            lag_steps = hours_to_steps(h, self.frequency)
            col_name = f"lag_{h}h"
            if col_name in self.df.columns:
                continue  # already present (e.g. explicit mapping)
            self.df[col_name] = self.df[target_col].shift(lag_steps)
            # Forward-fill NaN prefix so Darts gets a fully valid series
            self.df[col_name] = self.df[col_name].bfill()
            added.append(col_name)

        if added:
            existing = set(self.column_mapping.get("past_covariates", []))
            new_cols = [c for c in added if c not in existing]
            self.column_mapping["past_covariates"] = (
                self.column_mapping.get("past_covariates", []) + new_cols
            )
            logger.info("Added lag features: %s", new_cols)

    # ------------------------------------------------------------------
    # Darts conversion
    # ------------------------------------------------------------------

    def to_darts_series(
        self, fit_scalers: bool = True
    ) -> tuple[TimeSeries, Optional[TimeSeries], Optional[TimeSeries]]:
        """
        Convert to Darts TimeSeries objects.

        Args:
            fit_scalers: If True, fit new scalers on this data. If False,
                apply the already-fitted scalers (required for val/test data).

        Returns:
            Tuple of (target_series, past_covariate_series or None,
            future_covariate_series or None).
        """
        target_col = self.column_mapping["target"]
        time_col_name = self.df.index.name or "index"
        df_reset = self.df.reset_index()

        # Target series
        target_series = TimeSeries.from_dataframe(
            df_reset,
            time_col=time_col_name,
            value_cols=[target_col],
            freq=self.frequency,
            fill_missing_dates=True,
            fillna_value=None,
        )

        if fit_scalers:
            self.target_scaler = Scaler()
            target_series = self.target_scaler.fit_transform(target_series)
        elif self.target_scaler:
            target_series = self.target_scaler.transform(target_series)

        # Past covariate series
        cov_cols = [c for c in self.column_mapping.get("past_covariates", []) if c in self.df.columns]
        covariate_series = None

        if cov_cols:
            covariate_series = TimeSeries.from_dataframe(
                df_reset,
                time_col=time_col_name,
                value_cols=cov_cols,
                freq=self.frequency,
                fill_missing_dates=True,
                fillna_value=None,
            )

            if fit_scalers:
                self.covariate_scaler = Scaler()
                covariate_series = self.covariate_scaler.fit_transform(covariate_series)
            elif self.covariate_scaler:
                covariate_series = self.covariate_scaler.transform(covariate_series)

        # Future covariate series
        fut_cov_cols = [
            c for c in self.column_mapping.get("future_covariates", [])
            if c in self.df.columns
        ]
        future_covariate_series = None

        if fut_cov_cols:
            future_covariate_series = TimeSeries.from_dataframe(
                df_reset,
                time_col=time_col_name,
                value_cols=fut_cov_cols,
                freq=self.frequency,
                fill_missing_dates=True,
                fillna_value=None,
            )

            if fit_scalers:
                self.future_covariate_scaler = Scaler()
                future_covariate_series = self.future_covariate_scaler.fit_transform(
                    future_covariate_series
                )
            elif self.future_covariate_scaler:
                future_covariate_series = self.future_covariate_scaler.transform(
                    future_covariate_series
                )

        return target_series, covariate_series, future_covariate_series

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_data_summary(self) -> dict:
        """Return summary statistics for output."""
        target_col = self.column_mapping["target"]
        series = self.df[target_col]

        return {
            "total_samples": len(self.df),
            "start_date": self.df.index.min().isoformat(),
            "end_date": self.df.index.max().isoformat(),
            "target_column": target_col,
            "covariate_columns": self.column_mapping.get("past_covariates", []),
            "future_covariate_columns": self.column_mapping.get("future_covariates", []),
            "frequency_detected": self.frequency,
            "missing_value_count": int(series.isna().sum()),
            "target_mean": float(series.mean()),
            "target_std": float(series.std()),
            "target_min": float(series.min()),
            "target_max": float(series.max()),
        }

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_train_val(
        self, validation_split: float = 0.2
    ) -> tuple["ForecastingDataLoader", "ForecastingDataLoader"]:
        """
        Temporally split data into training and validation sets.

        The last ``validation_split`` fraction of rows (by time) is held out.

        Args:
            validation_split: Fraction for validation (0.1 – 0.3).

        Returns:
            Tuple of (train_loader, val_loader).
        """
        split_idx = int(len(self.df) * (1 - validation_split))
        train_df = self.df.iloc[:split_idx].copy()
        val_df = self.df.iloc[split_idx:].copy()

        train_loader = ForecastingDataLoader.__new__(ForecastingDataLoader)
        train_loader.df = train_df
        train_loader.column_mapping = self.column_mapping.copy()
        train_loader.frequency = self.frequency
        train_loader.csv_path = self.csv_path
        train_loader.target_scaler = None
        train_loader.covariate_scaler = None
        train_loader.future_covariate_scaler = None

        val_loader = ForecastingDataLoader.__new__(ForecastingDataLoader)
        val_loader.df = val_df
        val_loader.column_mapping = self.column_mapping.copy()
        val_loader.frequency = self.frequency
        val_loader.csv_path = self.csv_path
        val_loader.target_scaler = None
        val_loader.covariate_scaler = None
        val_loader.future_covariate_scaler = None

        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freq_to_timedelta(freq: str) -> pd.Timedelta:
        """Convert a pandas frequency string to a Timedelta."""
        return _freq_str_to_timedelta(freq)


# ---------------------------------------------------------------------------
# Module-level helper (used both inside and outside the class)
# ---------------------------------------------------------------------------

# Lookup table built once at import time — not re-created on every call.
_FREQ_TIMEDELTA_MAP: dict = {
    "h": pd.Timedelta(hours=1),
    "H": pd.Timedelta(hours=1),
    "15min": pd.Timedelta(minutes=15),
    "15T": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "30T": pd.Timedelta(minutes=30),
    "D": pd.Timedelta(days=1),
    "d": pd.Timedelta(days=1),
}


def _freq_str_to_timedelta(freq: str) -> pd.Timedelta:
    """Convert a supported frequency string to a pd.Timedelta."""
    if freq in _FREQ_TIMEDELTA_MAP:
        return _FREQ_TIMEDELTA_MAP[freq]
    try:
        return pd.Timedelta(freq)
    except ValueError:
        return pd.Timedelta(hours=1)
