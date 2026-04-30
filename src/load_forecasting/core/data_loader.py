"""
Load CSV data and convert to Darts TimeSeries format.

Handles:
- Column auto-detection from patterns
- Datetime parsing and timezone handling
- Missing value interpolation
- Data validation against spec constraints
"""

from pathlib import Path
from typing import Optional
import logging

import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

from .spec_loader import get_auto_detect_patterns

logger = logging.getLogger(__name__)


class DataLoadError(Exception):
    """Raised when data loading or validation fails."""

    pass


class ForecastingDataLoader:
    """Load and preprocess CSV data for forecasting."""

    def __init__(
        self,
        csv_path: str,
        column_mapping: Optional[dict] = None,
        frequency: str = "h",
        missing_value_strategy: str = "interpolate",
        max_gap: str = "2h",
    ):
        """
        Initialize data loader.

        Args:
            csv_path: Path to CSV file
            column_mapping: Map CSV columns to roles (datetime, target, past_covariates)
            frequency: Expected data frequency (15min, 30min, h)
            missing_value_strategy: How to handle missing values (interpolate, drop, error)
            max_gap: Maximum gap to interpolate (e.g., "2h")

        Raises:
            DataLoadError: If file not found or validation fails
        """
        self.csv_path = Path(csv_path)
        self.frequency = frequency
        self.missing_value_strategy = missing_value_strategy
        self.max_gap = pd.Timedelta(max_gap)

        # Validate file exists
        if not self.csv_path.exists():
            raise DataLoadError(f"CSV file not found: {csv_path}")

        # Load data
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            raise DataLoadError(f"Failed to read CSV: {e}")

        if self.df.empty:
            raise DataLoadError("CSV file is empty")

        # Resolve column mapping
        self.column_mapping = self._resolve_mapping(column_mapping)

        # Validate required columns exist
        self._validate_columns()

        # Preprocess
        self._preprocess()

        # Scalers (fit during training, reuse during evaluation)
        self.target_scaler: Optional[Scaler] = None
        self.covariate_scaler: Optional[Scaler] = None

    def _resolve_mapping(self, mapping: Optional[dict]) -> dict:
        """Auto-detect columns if mapping not provided."""
        if mapping:
            return mapping

        patterns = get_auto_detect_patterns("train_forecast_model")
        resolved = {}

        # Default patterns if spec not loaded
        if not patterns:
            patterns = {
                "datetime": ["datetime", "timestamp", "date", "time", "dt"],
                "target": ["kwh", "load", "power", "energy", "electricity", "demand"],
                "past_covariates": ["temp", "temperature", "rh", "humidity", "solar", "wind"],
            }

        # Detect datetime column
        for col in self.df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns.get("datetime", [])):
                resolved["datetime"] = col
                break

        # If no datetime found, try first column if it looks like dates
        if "datetime" not in resolved:
            first_col = self.df.columns[0]
            try:
                pd.to_datetime(self.df[first_col].head(10))
                resolved["datetime"] = first_col
                logger.info(f"Auto-detected datetime column: {first_col}")
            except Exception:
                pass

        # Detect target column
        for col in self.df.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in patterns.get("target", [])):
                resolved["target"] = col
                break

        # If no target found, use first numeric column that's not datetime
        if "target" not in resolved:
            for col in self.df.columns:
                if col != resolved.get("datetime") and pd.api.types.is_numeric_dtype(self.df[col]):
                    resolved["target"] = col
                    logger.info(f"Auto-detected target column: {col}")
                    break

        # Detect covariates
        covariate_patterns = patterns.get("past_covariates", [])
        resolved["past_covariates"] = [
            col
            for col in self.df.columns
            if any(p in col.lower() for p in covariate_patterns)
            and col not in [resolved.get("datetime"), resolved.get("target")]
        ]

        logger.info(f"Resolved column mapping: {resolved}")
        return resolved

    def _validate_columns(self) -> None:
        """Validate that required columns exist."""
        if "datetime" not in self.column_mapping:
            raise DataLoadError(
                "Could not identify datetime column. "
                "Please provide column_mapping with 'datetime' key."
            )

        if "target" not in self.column_mapping:
            raise DataLoadError(
                "Could not identify target column. "
                "Please provide column_mapping with 'target' key."
            )

        dt_col = self.column_mapping["datetime"]
        target_col = self.column_mapping["target"]

        if dt_col not in self.df.columns:
            raise DataLoadError(f"Datetime column not found in CSV: {dt_col}")

        if target_col not in self.df.columns:
            raise DataLoadError(f"Target column not found in CSV: {target_col}")

        # Check covariates exist
        for cov_col in self.column_mapping.get("past_covariates", []):
            if cov_col not in self.df.columns:
                logger.warning(f"Covariate column not found, skipping: {cov_col}")
                self.column_mapping["past_covariates"].remove(cov_col)

    def _preprocess(self) -> None:
        """Parse datetime, handle missing values, set index."""
        dt_col = self.column_mapping["datetime"]

        # Parse datetime
        try:
            self.df[dt_col] = pd.to_datetime(self.df[dt_col], utc=True)
        except Exception:
            # Try without UTC
            try:
                self.df[dt_col] = pd.to_datetime(self.df[dt_col])
                # Localize to UTC if naive
                if self.df[dt_col].dt.tz is None:
                    self.df[dt_col] = self.df[dt_col].dt.tz_localize("UTC")
            except Exception as e:
                raise DataLoadError(f"Failed to parse datetime column: {e}")

        # Set index and sort
        self.df = self.df.set_index(dt_col).sort_index()

        # Remove duplicate indices
        if self.df.index.duplicated().any():
            logger.warning("Duplicate timestamps found, keeping first occurrence")
            self.df = self.df[~self.df.index.duplicated(keep="first")]

        # Handle missing values
        if self.missing_value_strategy == "interpolate":
            self._interpolate_gaps()
        elif self.missing_value_strategy == "drop":
            self.df = self.df.dropna()
        elif self.missing_value_strategy == "error":
            if self.df.isna().any().any():
                raise DataLoadError("Data contains missing values")

    def _interpolate_gaps(self) -> None:
        """Interpolate small gaps, error on large gaps."""
        # Detect gaps in time index
        time_diff = self.df.index.to_series().diff()
        expected_freq = self._freq_to_timedelta(self.frequency)

    @staticmethod
    def _freq_to_timedelta(freq: str) -> pd.Timedelta:
        """Convert pandas frequency string to Timedelta."""
        freq_map = {
            "h": pd.Timedelta(hours=1),
            "H": pd.Timedelta(hours=1),
            "15min": pd.Timedelta(minutes=15),
            "15T": pd.Timedelta(minutes=15),
            "30min": pd.Timedelta(minutes=30),
            "30T": pd.Timedelta(minutes=30),
            "D": pd.Timedelta(days=1),
            "d": pd.Timedelta(days=1),
        }
        if freq in freq_map:
            return freq_map[freq]
        # Try parsing directly (e.g., "1h", "2h")
        try:
            return pd.Timedelta(freq)
        except ValueError:
            # Default to hourly
            return pd.Timedelta(hours=1)

        # Find gaps larger than max_gap
        large_gaps = time_diff > self.max_gap

        if large_gaps.any():
            gap_locations = self.df.index[large_gaps].tolist()
            # Log warning but don't error - just note the gaps
            logger.warning(
                f"Data contains {len(gap_locations)} gaps larger than {self.max_gap}. "
                f"First gap at: {gap_locations[0] if gap_locations else 'N/A'}"
            )

        # Interpolate missing values
        target_col = self.column_mapping["target"]
        if self.df[target_col].isna().any():
            self.df[target_col] = self.df[target_col].interpolate(method="time")

        # Interpolate covariates
        for cov_col in self.column_mapping.get("past_covariates", []):
            if cov_col in self.df.columns and self.df[cov_col].isna().any():
                self.df[cov_col] = self.df[cov_col].interpolate(method="time")

    def to_darts_series(
        self, fit_scalers: bool = True
    ) -> tuple[TimeSeries, Optional[TimeSeries]]:
        """
        Convert to Darts TimeSeries.

        Args:
            fit_scalers: If True, fit new scalers. If False, use existing.

        Returns:
            Tuple of (target_series, covariate_series or None)
        """
        target_col = self.column_mapping["target"]

        # Create target series
        target_series = TimeSeries.from_dataframe(
            self.df.reset_index(),
            time_col=self.df.index.name or "index",
            value_cols=[target_col],
            freq=self.frequency,
            fill_missing_dates=True,
            fillna_value=None,
        )

        # Scale target
        if fit_scalers:
            self.target_scaler = Scaler()
            target_series = self.target_scaler.fit_transform(target_series)
        elif self.target_scaler:
            target_series = self.target_scaler.transform(target_series)

        # Covariates
        cov_cols = self.column_mapping.get("past_covariates", [])
        covariate_series = None

        if cov_cols:
            # Filter to columns that actually exist
            valid_cov_cols = [c for c in cov_cols if c in self.df.columns]

            if valid_cov_cols:
                covariate_series = TimeSeries.from_dataframe(
                    self.df.reset_index(),
                    time_col=self.df.index.name or "index",
                    value_cols=valid_cov_cols,
                    freq=self.frequency,
                    fill_missing_dates=True,
                    fillna_value=None,
                )

                if fit_scalers:
                    self.covariate_scaler = Scaler()
                    covariate_series = self.covariate_scaler.fit_transform(covariate_series)
                elif self.covariate_scaler:
                    covariate_series = self.covariate_scaler.transform(covariate_series)

        return target_series, covariate_series

    def get_data_summary(self) -> dict:
        """Return summary statistics for output."""
        target_col = self.column_mapping["target"]

        return {
            "total_samples": len(self.df),
            "start_date": self.df.index.min().isoformat(),
            "end_date": self.df.index.max().isoformat(),
            "target_column": target_col,
            "covariate_columns": self.column_mapping.get("past_covariates", []),
            "frequency_detected": self.frequency,
            "missing_value_count": int(self.df[target_col].isna().sum()),
            "target_mean": float(self.df[target_col].mean()),
            "target_std": float(self.df[target_col].std()),
        }

    def split_train_val(
        self, validation_split: float = 0.2
    ) -> tuple["ForecastingDataLoader", "ForecastingDataLoader"]:
        """
        Split data into training and validation sets.

        Uses temporal split (last N% for validation).

        Args:
            validation_split: Fraction for validation (0.1-0.3)

        Returns:
            Tuple of (train_loader, val_loader)
        """
        split_idx = int(len(self.df) * (1 - validation_split))

        train_df = self.df.iloc[:split_idx].copy()
        val_df = self.df.iloc[split_idx:].copy()

        # Create new loaders with pre-processed data
        train_loader = ForecastingDataLoader.__new__(ForecastingDataLoader)
        train_loader.df = train_df
        train_loader.column_mapping = self.column_mapping.copy()
        train_loader.frequency = self.frequency
        train_loader.csv_path = self.csv_path
        train_loader.target_scaler = None
        train_loader.covariate_scaler = None

        val_loader = ForecastingDataLoader.__new__(ForecastingDataLoader)
        val_loader.df = val_df
        val_loader.column_mapping = self.column_mapping.copy()
        val_loader.frequency = self.frequency
        val_loader.csv_path = self.csv_path
        val_loader.target_scaler = None
        val_loader.covariate_scaler = None

        return train_loader, val_loader
