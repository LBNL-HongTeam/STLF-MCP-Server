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
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler

from .spec_loader import get_auto_detect_patterns
from .frequency_utils import  hours_to_steps

logger = logging.getLogger(__name__)


# Minimum fraction of non-null target values required (90 % coverage)
_MIN_COVERAGE = 0.90

# Lag offsets (in hours) added as past covariates from the target column
_DEFAULT_LAG_HOURS = [24, 48, 168]

# Calendar feature column names injected automatically.
# Cyclic sin/cos encodings for hour, day-of-week, and month match the
# feature set described in Li et al. (2025) Table 6.  Raw integer columns
# (cal_hour, cal_dow, cal_month) and cal_is_weekend are retained alongside
# the cyclic encodings because tree models (XGBoost, RandomForest) can split
# on integers directly without needing to invert sin/cos pairs.
_CALENDAR_COLS = [
    "cal_hour",
    "cal_dow",
    "cal_month",
    "cal_is_weekend",
    "cal_hour_sin",
    "cal_hour_cos",
    "cal_dow_sin",    # Li et al. (2025) Table 6: T_week_sin
    "cal_dow_cos",    # Li et al. (2025) Table 6: T_week_cos
    "cal_month_sin",  # Li et al. (2025) Table 6: T_month_sin
    "cal_month_cos",  # Li et al. (2025) Table 6: T_month_cos
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
                day-of-week, month, is_weekend, cyclic hour/dow/month
                sin+cos encodings as past covariates (matching Li et al.
                2025, Table 6).
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

        # Warn and drop any covariate columns that don't exist
        self._prune_missing_columns("past_covariates", "Past")
        self._prune_missing_columns("future_covariates", "Future")

    def _prune_missing_columns(self, mapping_key: str, label: str) -> None:
        """Drop mapped covariate columns absent from the DataFrame, with a warning."""
        cols = self.column_mapping.get(mapping_key, [])
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            logger.warning(
                "%s covariate column(s) not found in CSV and will be skipped: %s",
                label, missing,
            )
            self.column_mapping[mapping_key] = [c for c in cols if c in self.df.columns]

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

        cols = (
            [self.column_mapping["target"]]
            + list(self.column_mapping.get("past_covariates", []))
            + list(self.column_mapping.get("future_covariates", []))
        )
        for col in cols:
            if col in self.df.columns and self.df[col].isna().any():
                self.df[col] = self.df[col].interpolate(method="time")

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
            cal_dow_sin      : sin(2π * dow / 7)   — Li et al. (2025) Table 6: T_week_sin
            cal_dow_cos      : cos(2π * dow / 7)   — Li et al. (2025) Table 6: T_week_cos
            cal_month_sin    : sin(2π * (month-1) / 12) — Li et al. (2025) Table 6: T_month_sin
            cal_month_cos    : cos(2π * (month-1) / 12) — Li et al. (2025) Table 6: T_month_cos

        The month encoding uses (month - 1) so that January (1) maps to 0 and
        December (12) maps to 11, placing them at adjacent points on the unit
        circle and preserving the Dec→Jan periodicity.

        All new columns are appended to past_covariates in column_mapping.
        """
        idx = self.df.index
        hour = idx.hour.astype(float)
        dow = idx.dayofweek.astype(float)
        month0 = (idx.month - 1).astype(float)  # 0..11 for clean Dec→Jan wrap

        _two_pi_over_24 = 2 * math.pi / 24  # computed once per call
        _two_pi_over_7  = 2 * math.pi / 7
        _two_pi_over_12 = 2 * math.pi / 12

        self.df["cal_hour"] = hour
        self.df["cal_dow"] = dow
        self.df["cal_month"] = idx.month.astype(float)
        self.df["cal_is_weekend"] = (idx.dayofweek >= 5).astype(float)
        self.df["cal_hour_sin"] = np.sin(hour   * _two_pi_over_24)
        self.df["cal_hour_cos"] = np.cos(hour   * _two_pi_over_24)
        self.df["cal_dow_sin"]  = np.sin(dow    * _two_pi_over_7)
        self.df["cal_dow_cos"]  = np.cos(dow    * _two_pi_over_7)
        self.df["cal_month_sin"] = np.sin(month0 * _two_pi_over_12)
        self.df["cal_month_cos"] = np.cos(month0 * _two_pi_over_12)

        logger.info(
            "Added calendar features: %s",
            self._extend_past_covariates(_CALENDAR_COLS),
        )

    def _extend_past_covariates(self, cols: list) -> list:
        """Append new (not-yet-present) columns to past_covariates; return them."""
        existing = set(self.column_mapping.get("past_covariates", []))
        new_cols = [c for c in cols if c not in existing]
        self.column_mapping["past_covariates"] = (
            self.column_mapping.get("past_covariates", []) + new_cols
        )
        return new_cols

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
            # Forward-fill NaN prefix so Darts gets a fully valid series.
            # ffill propagates the first valid value forwards (no future leakage).
            # Note: bfill was used previously but introduced temporal leakage for
            # short series by filling early NaN values with future observations.
            self.df[col_name] = self.df[col_name].ffill()
            # Fallback: if the entire column is still NaN (context shorter
            # than the lag offset), fill with the target mean. This is the
            # best "no-information" estimate and prevents NaN from
            # propagating into the feature matrix during inference.
            if self.df[col_name].isna().any():
                target_mean = self.df[target_col].mean()
                self.df[col_name] = self.df[col_name].fillna(target_mean)
            added.append(col_name)

        if added:
            logger.info("Added lag features: %s", self._extend_past_covariates(added))

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

        # When the underlying DataFrame is non-contiguous (e.g. after a seasonal
        # split that interleaves winter/spring/summer/fall chunks), Darts'
        # fill_missing_dates=True inserts NaN rows for the intra-year time gaps.
        # MissingValuesFiller(fill='auto') interpolates those gaps so that
        # sklearn models (LinearRegression, XGBoost) never receive NaN inputs.
        gap_filler = MissingValuesFiller(fill="auto")

        def build(value_cols: list, scaler_attr: str):
            """Build one gap-filled, scaled TimeSeries (or None if no columns)."""
            cols = [c for c in value_cols if c in self.df.columns]
            if not cols:
                return None
            series = TimeSeries.from_dataframe(
                df_reset,
                time_col=time_col_name,
                value_cols=cols,
                freq=self.frequency,
                fill_missing_dates=True,
                fillna_value=None,
            )
            series = gap_filler.transform(series)
            if fit_scalers:
                setattr(self, scaler_attr, Scaler())
                series = getattr(self, scaler_attr).fit_transform(series)
            elif getattr(self, scaler_attr):
                series = getattr(self, scaler_attr).transform(series)
            return series

        target_series = build([target_col], "target_scaler")
        covariate_series = build(
            self.column_mapping.get("past_covariates", []), "covariate_scaler"
        )
        future_covariate_series = build(
            self.column_mapping.get("future_covariates", []), "future_covariate_scaler"
        )
        return target_series, covariate_series, future_covariate_series

    def contiguous_chunks(self) -> list[pd.DataFrame]:
        """Split ``self.df`` at time discontinuities into contiguous blocks.

        A discontinuity is any step in the DatetimeIndex larger than one
        sampling interval.  For a contiguous frame this returns a single-element
        list containing the whole frame.

        This is the basis of :py:meth:`to_darts_series_chunks`, which avoids the
        interpolation that :py:meth:`to_darts_series` would otherwise perform
        across gaps introduced by a seasonal train/validation split.
        """
        if len(self.df) == 0:
            return []
        step = pd.tseries.frequencies.to_offset(self.frequency)
        deltas = self.df.index.to_series().diff()
        # First row has NaT delta; treat as continuation of the first chunk.
        breaks = np.flatnonzero((deltas > step).to_numpy())
        if len(breaks) == 0:
            return [self.df]
        bounds = [0, *breaks.tolist(), len(self.df)]
        return [
            self.df.iloc[a:b].copy()
            for a, b in zip(bounds[:-1], bounds[1:])
            if b > a
        ]

    def to_darts_series_chunks(
        self, fit_scalers: bool = True
    ) -> tuple[list[TimeSeries], Optional[list[TimeSeries]], Optional[list[TimeSeries]]]:
        """Convert to *lists* of gap-free Darts TimeSeries, one per contiguous block.

        Unlike :py:meth:`to_darts_series`, which reindexes to a regular frequency
        and interpolates across any time gap, this method splits at the gaps and
        returns one TimeSeries per contiguous run.  This matters after
        :py:meth:`split_train_val_seasonal`, whose output is non-contiguous by
        construction: passing that frame through ``to_darts_series`` fabricates
        straight-line load across the multi-week holes the split just created
        (~16% of a seasonally-split training year, ~69% of the validation year at
        hourly resolution).  Darts global models accept a sequence of series
        natively, so no interpolation is needed.

        Scalers are fitted with ``global_fit=True`` so that a single set of
        scaling parameters is shared across all chunks.  Per-series fitting (the
        Darts default for sequence input) would scale each season independently
        and destroy the between-season level information.

        Args:
            fit_scalers: If True, fit new scalers across all chunks jointly.
                If False, apply the already-fitted scalers.

        Returns:
            Tuple of (target_series_list, past_covariate_list or None,
            future_covariate_list or None).
        """
        target_col = self.column_mapping["target"]
        time_col_name = self.df.index.name or "index"
        chunks = self.contiguous_chunks()
        if not chunks:
            raise DataLoadError("Cannot build Darts series from an empty DataFrame.")

        gap_filler = MissingValuesFiller(fill="auto")

        def build(value_cols: list, scaler_attr: str):
            cols = [c for c in value_cols if c in self.df.columns]
            if not cols:
                return None
            series_list = []
            for chunk in chunks:
                s = TimeSeries.from_dataframe(
                    chunk.reset_index(),
                    time_col=time_col_name,
                    value_cols=cols,
                    freq=self.frequency,
                    fill_missing_dates=True,
                    fillna_value=None,
                )
                # Chunks are contiguous by construction, so this only patches
                # genuine NaN values inside a block, never split-induced gaps.
                series_list.append(gap_filler.transform(s))
            if fit_scalers:
                setattr(self, scaler_attr, Scaler(global_fit=True))
                series_list = getattr(self, scaler_attr).fit_transform(series_list)
            elif getattr(self, scaler_attr):
                series_list = getattr(self, scaler_attr).transform(series_list)
            return list(series_list)

        target_series = build([target_col], "target_scaler")
        covariate_series = build(
            self.column_mapping.get("past_covariates", []), "covariate_scaler"
        )
        future_covariate_series = build(
            self.column_mapping.get("future_covariates", []), "future_covariate_scaler"
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

    def _clone_with_df(self, df: pd.DataFrame) -> "ForecastingDataLoader":
        """Build a fresh, unscaled loader sharing this loader's config over ``df``."""
        sub = ForecastingDataLoader.__new__(ForecastingDataLoader)
        sub.df = df
        sub.column_mapping = self.column_mapping.copy()
        sub.frequency = self.frequency
        sub.csv_path = self.csv_path
        sub.target_scaler = None
        sub.covariate_scaler = None
        sub.future_covariate_scaler = None
        return sub

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
        return (
            self._clone_with_df(self.df.iloc[:split_idx].copy()),
            self._clone_with_df(self.df.iloc[split_idx:].copy()),
        )

    def split_train_val_seasonal(
        self, validation_split: float = 0.2
    ) -> tuple["ForecastingDataLoader", "ForecastingDataLoader"]:
        """
        Seasonally-stratified train/validation split.

        Implements the approach from Li et al. (2025) "A cross-dimensional
        analysis of data-driven short-term load forecasting methods with
        large-scale smart meter data" (Energy & Buildings, 344):

            "To ensure coverage of different seasons, the development set
            was first divided into four seasonal chunks, each of which was
            further split into 80% for training and 20% for validation."

        The dataset is partitioned into four meteorological seasons by month:

        - Winter : Dec, Jan, Feb
        - Spring : Mar, Apr, May
        - Summer : Jun, Jul, Aug
        - Fall   : Sep, Oct, Nov

        Within each seasonal chunk the **last** ``validation_split`` fraction
        (by time) is held out for validation; the earlier portion trains.
        The four training chunks and four validation chunks are each
        concatenated and re-sorted chronologically before being returned.

        Fallback behaviour
        ------------------
        If fewer than all four seasons are present in the data a
        ``WARNING`` is logged and the method falls back to the standard
        sequential :py:meth:`split_train_val`.  This keeps the API safe
        for short datasets (e.g. single-season smoke-tests).

        Args:
            validation_split: Fraction held out per season (0.1 – 0.3).

        Returns:
            Tuple of (train_loader, val_loader).
        """
        # Map each month to its meteorological season name
        _MONTH_TO_SEASON = {
            12: "winter", 1: "winter", 2: "winter",
            3: "spring",  4: "spring", 5: "spring",
            6: "summer",  7: "summer", 8: "summer",
            9: "fall",   10: "fall",  11: "fall",
        }
        _ALL_SEASONS = {"winter", "spring", "summer", "fall"}

        # Assign a season label to every row using the DatetimeIndex
        seasons = self.df.index.month.map(_MONTH_TO_SEASON)
        present_seasons = set(seasons.unique())
        missing_seasons = _ALL_SEASONS - present_seasons

        if missing_seasons:
            logger.warning(
                "Seasonal split requested but the following season(s) are not "
                "present in the data: %s. "
                "Falling back to sequential split (validation_split=%.2f). "
                "Provide at least 12 months of data to enable seasonal splitting.",
                sorted(missing_seasons),
                validation_split,
            )
            return self.split_train_val(validation_split)

        train_chunks: list[pd.DataFrame] = []
        val_chunks: list[pd.DataFrame] = []

        for season in ("winter", "spring", "summer", "fall"):
            mask = seasons == season
            chunk = self.df.loc[mask]
            n = len(chunk)
            split_idx = int(n * (1 - validation_split))
            # Guard: ensure each chunk has enough rows
            if split_idx == 0 or split_idx == n:
                logger.warning(
                    "Season '%s' has too few rows (%d) for a %.0f%%/%.0f%% split. "
                    "Falling back to sequential split.",
                    season, n,
                    (1 - validation_split) * 100,
                    validation_split * 100,
                )
                return self.split_train_val(validation_split)
            train_chunks.append(chunk.iloc[:split_idx].copy())
            val_chunks.append(chunk.iloc[split_idx:].copy())

        train_df = pd.concat(train_chunks).sort_index()
        val_df = pd.concat(val_chunks).sort_index()

        return self._clone_with_df(train_df), self._clone_with_df(val_df)
