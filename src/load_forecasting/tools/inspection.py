"""
MCP tool: inspect_data — pre-training CSV inspection and quality report.
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd

from ..core.data_loader import (
    ForecastingDataLoader,
    DataLoadError,
    _infer_frequency,
    DEFAULT_DATETIME_PATTERNS,
    DEFAULT_TARGET_PATTERNS,
    DEFAULT_COVARIATE_PATTERNS,
)
from ..core.frequency_utils import hours_to_steps, FREQ_TO_STEPS_PER_HOUR
from ..core.paths import resolve_data_path, not_found_hint
from ._common import (
    create_success_response,
    create_error_response,
)

logger = logging.getLogger(__name__)


def inspect_data(
    csv_path: str,
    column_mapping: Optional[dict] = None,
    frequency: Optional[str] = None,
) -> dict:
    """
    Inspect a CSV file and report everything an agent needs before training.

    Reads the file, auto-detects or validates column roles, infers the data
    frequency, computes per-column statistics, identifies gaps and anomalies,
    and returns actionable feature suggestions.

    Args:
        csv_path: Path to CSV file to inspect.
        column_mapping: Optional explicit column roles (datetime, target,
            past_covariates).  Auto-detected if not provided.
        frequency: Expected data frequency ('15min', '30min', 'h').  If
            omitted, the tool infers it from the timestamps.

    Returns:
        Dict with detected columns, frequency, statistics, gap analysis,
        quality flags, and feature suggestions.
    """
    try:
        path = resolve_data_path(csv_path)
        if not path.exists():
            return create_error_response(
                f"File not found: {csv_path}\n"
                "Check that the path is correct and the file is accessible. "
                + not_found_hint(csv_path)
            )

        # ------------------------------------------------------------------
        # Read raw CSV (no preprocessing yet — we want the raw picture)
        # ------------------------------------------------------------------
        try:
            raw_df = pd.read_csv(path)
        except Exception as e:
            return create_error_response(f"Failed to read CSV: {e}")

        if raw_df.empty:
            return create_error_response("CSV file is empty.")

        n_rows_raw = len(raw_df)
        columns_in_file = raw_df.columns.tolist()

        # ------------------------------------------------------------------
        # Run the data loader (auto-detect columns, parse datetimes, dedup)
        # Use add_calendar_features=False / lag_hours=[] so we inspect only
        # what the user actually has in their file.
        # ------------------------------------------------------------------
        detected_frequency = frequency
        loader_error: Optional[str] = None
        loader: Optional[ForecastingDataLoader] = None

        # Determine frequency to pass to loader
        if detected_frequency is None:
            # Sniff from raw datetime column without full loader initialisation
            detected_frequency = _sniff_frequency(raw_df)

        try:
            # Pass the already-read raw_df to avoid a second disk read.
            loader = ForecastingDataLoader(
                csv_path=str(path),
                column_mapping=column_mapping,
                frequency=detected_frequency or "h",
                add_calendar_features=False,
                lag_hours=[],
                dataframe=raw_df,
            )
        except DataLoadError as e:
            loader_error = str(e)

        # ------------------------------------------------------------------
        # Column detection report
        # ------------------------------------------------------------------
        if loader is not None:
            resolved_mapping = loader.column_mapping
            dt_col = resolved_mapping.get("datetime")
            target_col = resolved_mapping.get("target")
            cov_cols = resolved_mapping.get("past_covariates", [])
            fut_cov_cols = resolved_mapping.get("future_covariates", [])
            # Use the inferred frequency stored by _validate_frequency — no
            # extra _infer_frequency call needed.
            actual_frequency = loader.inferred_frequency or detected_frequency or "unknown"
        else:
            # Loader failed — do best-effort column sniffing from raw df
            resolved_mapping = _sniff_columns(raw_df, column_mapping)
            dt_col = resolved_mapping.get("datetime")
            target_col = resolved_mapping.get("target")
            cov_cols = resolved_mapping.get("past_covariates", [])
            fut_cov_cols = resolved_mapping.get("future_covariates", [])
            actual_frequency = detected_frequency or "unknown"

        mapped_set = {dt_col, target_col, *cov_cols, *fut_cov_cols}
        column_report = {
            "datetime": dt_col,
            "target": target_col,
            "past_covariates": cov_cols,
            "future_covariates": fut_cov_cols,
            "unrecognised": [c for c in columns_in_file if c not in mapped_set],
            "all_columns": columns_in_file,
        }

        # ------------------------------------------------------------------
        # Frequency report
        # ------------------------------------------------------------------
        freq_report = {
            "declared": frequency,
            "inferred": actual_frequency,
            "match": (frequency is None) or (actual_frequency == frequency),
            "supported": actual_frequency in FREQ_TO_STEPS_PER_HOUR,
        }
        if not freq_report["match"]:
            freq_report["recommendation"] = (
                f"Set frequency='{actual_frequency}' to match the data."
            )

        # ------------------------------------------------------------------
        # Row / time range summary
        # ------------------------------------------------------------------
        time_range: dict = {}
        n_rows_clean = n_rows_raw
        if loader is not None:
            n_rows_clean = len(loader.df)
            time_range = {
                "start": loader.df.index.min().isoformat(),
                "end": loader.df.index.max().isoformat(),
                "n_rows_raw": n_rows_raw,
                "n_rows_after_dedup": n_rows_clean,
                "n_duplicates_removed": n_rows_raw - n_rows_clean,
            }
            # Expected row count at detected frequency
            if actual_frequency in FREQ_TO_STEPS_PER_HOUR:
                steps_per_hour = FREQ_TO_STEPS_PER_HOUR[actual_frequency]
                duration_hours = (
                    loader.df.index.max() - loader.df.index.min()
                ).total_seconds() / 3600
                expected_rows = int(duration_hours * steps_per_hour) + 1
                time_range["expected_rows_at_frequency"] = expected_rows
                time_range["coverage_pct"] = round(
                    100.0 * n_rows_clean / max(expected_rows, 1), 2
                )

        # ------------------------------------------------------------------
        # Per-column statistics
        # We use raw_df for missing-value counts (pre-interpolation) so that
        # the report reflects what the user actually has in their file.
        # For numeric stats (mean, std, …) we use the cleaned loader.df when
        # available, falling back to raw_df otherwise.
        # ------------------------------------------------------------------
        column_stats: list[dict] = []
        work_df = loader.df if loader is not None else raw_df

        # Build the list of columns to report stats for
        _stat_cols: list = []
        if target_col:
            _stat_cols.append(target_col)
        _stat_cols.extend(cov_cols)
        _stat_cols.extend(fut_cov_cols)

        for col in _stat_cols:
            if col not in work_df.columns:
                continue
            s = work_df[col]
            # Count nulls from raw data (pre-interpolation) when possible
            raw_col = raw_df[col] if col in raw_df.columns else s
            n_null = int(raw_col.isna().sum())
            n_total_raw = len(raw_col)
            non_null = s.dropna()
            _role = (
                "target" if col == target_col
                else "future_covariate" if col in fut_cov_cols
                else "past_covariate"
            )
            stat: dict = {
                "column": col,
                "role": _role,
                "n_total": n_total_raw,
                "n_missing": n_null,
                "coverage_pct": round(100.0 * (n_total_raw - n_null) / max(n_total_raw, 1), 2),
                "dtype": str(s.dtype),
            }
            if pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
                p5, p95 = np.percentile(non_null, [5, 95])
                stat.update({
                    "mean": round(float(non_null.mean()), 4),
                    "std": round(float(non_null.std()), 4),
                    "min": round(float(non_null.min()), 4),
                    "max": round(float(non_null.max()), 4),
                    "p5": round(float(p5), 4),
                    "p95": round(float(p95), 4),
                    "n_negative": int((non_null < 0).sum()),
                    "n_zero": int((non_null == 0).sum()),
                })
            column_stats.append(stat)

        # ------------------------------------------------------------------
        # Gap analysis (only possible when loader succeeded)
        # ------------------------------------------------------------------
        gap_report: dict = {"analysis_available": loader is not None}
        if loader is not None and actual_frequency in FREQ_TO_STEPS_PER_HOUR:
            expected_td = pd.Timedelta(hours=1) / FREQ_TO_STEPS_PER_HOUR[actual_frequency]
            diffs = loader.df.index.to_series().diff().dropna()
            gaps = diffs[diffs > expected_td * 1.5]  # 50% tolerance
            gap_report["n_gaps"] = len(gaps)
            gap_report["total_missing_steps"] = int((gaps / expected_td - 1).sum())
            if len(gaps) > 0:
                largest = gaps.max()
                gap_report["largest_gap"] = str(largest)
                gap_report["largest_gap_start"] = gaps.idxmax().isoformat()
                gap_list = []
                for ts, dur in gaps.sort_values(ascending=False).head(5).items():
                    gap_list.append({
                        "start": ts.isoformat(),
                        "duration": str(dur),
                        "missing_steps": int(dur / expected_td) - 1,
                    })
                gap_report["top_gaps"] = gap_list

        # ------------------------------------------------------------------
        # Outlier detection (target column only, requires loader success)
        # Two checks:
        #   1. Value outliers: values beyond mean ± 3 * std
        #   2. Step-change spikes: consecutive differences > 3 * std(diffs)
        # Both are warnings only — never blocking.
        # ------------------------------------------------------------------
        outlier_report: dict = {"analysis_available": False}
        if loader is not None and target_col and target_col in loader.df.columns:
            _ts = loader.df[target_col].dropna()
            if len(_ts) >= 10:
                outlier_report["analysis_available"] = True
                _mean = float(_ts.mean())
                _std = float(_ts.std())
                _threshold = 3.0 * _std

                # Value outliers
                _val_outlier_mask = (_ts - _mean).abs() > _threshold
                _n_value_outliers = int(_val_outlier_mask.sum())
                outlier_report["value_outliers"] = {
                    "n_outliers": _n_value_outliers,
                    "threshold_3sigma": round(_mean + _threshold, 4),
                    "threshold_3sigma_low": round(_mean - _threshold, 4),
                    "pct_of_series": round(100.0 * _n_value_outliers / len(_ts), 2),
                }
                if _n_value_outliers > 0:
                    _top_outliers = (
                        _ts[_val_outlier_mask]
                        .abs()
                        .nlargest(5)
                    )
                    outlier_report["value_outliers"]["top_outlier_timestamps"] = [
                        {"timestamp": ts.isoformat(), "value": round(float(v), 4)}
                        for ts, v in zip(_top_outliers.index, _ts[_top_outliers.index])
                    ]

                # Step-change spike detection
                # Guard: if std(diffs) is near-zero (uniform / constant series)
                # the 3σ threshold collapses to ~0 and every step looks like a
                # spike.  Require the threshold to be at least 1% of the series
                # mean before flagging anything.
                _diffs = _ts.diff().dropna().abs()
                _diff_std = float(_diffs.std())
                _diff_threshold = 3.0 * _diff_std
                _min_meaningful_threshold = max(abs(_mean) * 0.01, 1e-6)
                if _diff_threshold >= _min_meaningful_threshold:
                    _spike_mask = _diffs > _diff_threshold
                    _n_spikes = int(_spike_mask.sum())
                else:
                    _spike_mask = pd.Series(False, index=_diffs.index)
                    _n_spikes = 0
                outlier_report["step_change_spikes"] = {
                    "n_spikes": _n_spikes,
                    "diff_threshold_3sigma": round(_diff_threshold, 4),
                    "pct_of_series": round(100.0 * _n_spikes / len(_diffs), 2),
                }
                if _n_spikes > 0:
                    _top_spikes = _diffs[_spike_mask].nlargest(5)
                    outlier_report["step_change_spikes"]["top_spike_timestamps"] = [
                        {
                            "timestamp": ts.isoformat(),
                            "step_change": round(float(v), 4),
                            "value_before": round(float(_ts.get(ts - pd.Timedelta(freq=_ts.index.freq or "h"), default=float("nan"))), 4) if _ts.index.freq else None,
                            "value_at": round(float(_ts.get(ts, float("nan"))), 4),
                        }
                        for ts, v in zip(_top_spikes.index, _top_spikes)
                    ]

        # ------------------------------------------------------------------
        # Seasonality summary (target column only, requires loader success)
        # Reports: peak hour-of-day and peak day-of-week by mean load.
        # Also flags LOW_DIURNAL_RANGE if the signal is weak (ratio < 1.05)
        # which may indicate a non-load or aggregated signal.
        # ------------------------------------------------------------------
        seasonality_report: dict = {"analysis_available": False}
        if loader is not None and target_col and target_col in loader.df.columns:
            _ts_full = loader.df[target_col].dropna()
            if len(_ts_full) >= 48:  # at least 2 days
                seasonality_report["analysis_available"] = True

                # Hour-of-day profile
                _hourly = _ts_full.groupby(_ts_full.index.hour).mean()
                _peak_hour = int(_hourly.idxmax())
                _trough_hour = int(_hourly.idxmin())
                _hourly_ratio = round(float(_hourly.max() / max(_hourly.min(), 1e-9)), 4)
                seasonality_report["hour_of_day"] = {
                    "peak_hour": _peak_hour,
                    "trough_hour": _trough_hour,
                    "peak_mean": round(float(_hourly.max()), 4),
                    "trough_mean": round(float(_hourly.min()), 4),
                    "peak_to_trough_ratio": _hourly_ratio,
                }

                # Day-of-week profile (0=Monday … 6=Sunday)
                _dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                _daily = _ts_full.groupby(_ts_full.index.dayofweek).mean()
                _peak_dow = int(_daily.idxmax())
                _trough_dow = int(_daily.idxmin())
                _daily_ratio = round(float(_daily.max() / max(_daily.min(), 1e-9)), 4)
                seasonality_report["day_of_week"] = {
                    "peak_day": _dow_names[_peak_dow],
                    "trough_day": _dow_names[_trough_dow],
                    "peak_mean": round(float(_daily.max()), 4),
                    "trough_mean": round(float(_daily.min()), 4),
                    "peak_to_trough_ratio": _daily_ratio,
                }

                # Weak diurnal signal flag
                seasonality_report["low_diurnal_variation"] = _hourly_ratio < 1.05

        # ------------------------------------------------------------------
        # Covariate-target correlation
        # Only computed when (a) the loader succeeded, (b) there are past or
        # future covariate columns, AND (c) the target column is present.
        # Uses Pearson r on the processed loader.df (post-interpolation) so
        # that covariate gaps aligned with target gaps don't inflate the score.
        # ------------------------------------------------------------------
        covariate_correlation: list[dict] = []
        _all_cov_cols = list(cov_cols) + list(fut_cov_cols)
        if loader is not None and target_col and _all_cov_cols and target_col in loader.df.columns:
            _target_series = loader.df[target_col]
            for _cov in _all_cov_cols:
                if _cov not in loader.df.columns:
                    continue
                _cov_series = loader.df[_cov]
                _combined = pd.concat([_target_series, _cov_series], axis=1).dropna()
                if len(_combined) < 10:
                    continue
                _pearson_r = float(_combined.iloc[:, 0].corr(_combined.iloc[:, 1]))
                _role = "future_covariate" if _cov in fut_cov_cols else "past_covariate"
                covariate_correlation.append({
                    "column": _cov,
                    "role": _role,
                    "pearson_r": round(_pearson_r, 4),
                    "abs_pearson_r": round(abs(_pearson_r), 4),
                    "n_overlap_rows": len(_combined),
                    "interpretation": (
                        "strong" if abs(_pearson_r) >= 0.6
                        else "moderate" if abs(_pearson_r) >= 0.3
                        else "weak"
                    ),
                })
            # Sort by absolute correlation descending
            covariate_correlation.sort(key=lambda x: x["abs_pearson_r"], reverse=True)

        # ------------------------------------------------------------------
        # Quality flags
        # ------------------------------------------------------------------
        quality_flags: list[str] = []
        if loader_error:
            quality_flags.append(f"LOAD_ERROR: {loader_error}")
        if not freq_report["match"]:
            quality_flags.append(
                f"FREQUENCY_MISMATCH: declared={frequency}, "
                f"inferred={actual_frequency}"
            )
        if not freq_report["supported"]:
            quality_flags.append(
                f"UNSUPPORTED_FREQUENCY: '{actual_frequency}' — "
                "use 15min, 30min, or h"
            )
        if time_range.get("n_duplicates_removed", 0) > 0:
            quality_flags.append(
                f"DUPLICATES: {time_range['n_duplicates_removed']} "
                "duplicate timestamps removed"
            )
        if time_range.get("coverage_pct", 100) < 90:
            quality_flags.append(
                f"LOW_COVERAGE: {time_range.get('coverage_pct')}% row coverage "
                f"(< 90% threshold)"
            )
        if gap_report.get("n_gaps", 0) > 0:
            quality_flags.append(
                f"GAPS: {gap_report['n_gaps']} gap(s) totalling "
                f"{gap_report.get('total_missing_steps', '?')} missing steps"
            )
        # Outlier flags (warnings only)
        if outlier_report.get("analysis_available"):
            _n_val = outlier_report["value_outliers"]["n_outliers"]
            _n_spk = outlier_report["step_change_spikes"]["n_spikes"]
            if _n_val > 0:
                quality_flags.append(
                    f"OUTLIERS: {_n_val} value(s) beyond ±3σ "
                    f"({outlier_report['value_outliers']['pct_of_series']}% of series) — "
                    "may skew model training; review before training"
                )
            if _n_spk > 0:
                quality_flags.append(
                    f"SPIKE: {_n_spk} sudden step-change(s) exceeding 3σ of diffs "
                    f"({outlier_report['step_change_spikes']['pct_of_series']}% of steps) — "
                    "possible sensor fault or data entry error"
                )
        # Seasonality flag
        if seasonality_report.get("analysis_available") and seasonality_report.get("low_diurnal_variation"):
            quality_flags.append(
                "LOW_DIURNAL_VARIATION: Hour-of-day peak/trough ratio < 1.05 — "
                "data shows little diurnal pattern; verify the target column is a "
                "load signal and not a flat or aggregated value"
            )
        for stat in column_stats:
            if stat.get("n_missing", 0) > 0:
                quality_flags.append(
                    f"MISSING_VALUES: column '{stat['column']}' has "
                    f"{stat['n_missing']} null(s) ({100 - stat['coverage_pct']:.1f}%)"
                )
            if stat.get("n_negative", 0) > 0 and stat["role"] == "target":
                quality_flags.append(
                    f"NEGATIVE_TARGET: column '{stat['column']}' has "
                    f"{stat['n_negative']} negative value(s)"
                )

        # ------------------------------------------------------------------
        # Non-midnight start → possible UTC timestamps
        # When data is stored in UTC the first timestamp is typically offset
        # from midnight by the local UTC offset (e.g. 16:00 UTC = 00:00 PST).
        # Flag this so the user knows a timezone conversion may be required
        # before joining with local-time load data.
        # Only apply to multi-day datasets (> 24 rows) to avoid false
        # positives on tiny test files that happen to start mid-day.
        # ------------------------------------------------------------------
        if loader is not None and n_rows_clean > 24:
            first_ts = loader.df.index.min()
            if first_ts.hour != 0 or first_ts.minute != 0:
                quality_flags.append(
                    f"POSSIBLE_UTC: First timestamp is at "
                    f"{first_ts.strftime('%H:%M')}, not midnight. "
                    f"Data may be stored in UTC rather than local time. "
                    f"If merging with local-time load data, convert the "
                    f"timezone before joining to prevent an 8-hour shift "
                    f"in weather features."
                )
                # DST risk: UTC data that spans March or November will produce
                # a duplicate timestamp (fall-back in Nov) and a missing hour
                # (spring-forward in Mar) when converted to a DST-observing
                # local timezone.
                months_covered = set(loader.df.index.month.tolist())
                if months_covered & {3, 11}:
                    quality_flags.append(
                        "DST_MERGE_RISK: Data covers March and/or November. "
                        "Converting UTC timestamps to a DST-observing local "
                        "timezone (e.g. America/Los_Angeles) will produce: "
                        "(1) a duplicate timestamp at 01:00 on the November "
                        "DST fall-back — call drop_duplicates() to keep the "
                        "first occurrence; "
                        "(2) a missing hour at 02:00 on the March DST "
                        "spring-forward — Darts will insert a NaN row for it, "
                        "causing LinearRegression to fail with "
                        "'Input X contains NaN'. Forward-fill that gap before "
                        "calling train_forecast_model."
                    )

        # ------------------------------------------------------------------
        # Feature suggestions
        # ------------------------------------------------------------------
        suggestions: list[str] = []
        if target_col:
            suggestions.append(
                "Calendar features (hour, day-of-week, month, is_weekend, "
                "hour_sin/cos, dow_sin/cos, month_sin/cos) are auto-added by "
                "the data loader — no action needed."
            )
            suggestions.append(
                "Lag features at 24h, 48h, and 168h are auto-added by the "
                "data loader — no action needed."
            )
        if not cov_cols:
            suggestions.append(
                "No weather/covariate columns detected. Adding outdoor "
                "temperature (column name containing 'temp' or 'temperature') "
                "typically improves accuracy."
            )
        if column_report["unrecognised"]:
            suggestions.append(
                f"Columns {column_report['unrecognised']} were not mapped to "
                "any role. If they contain useful signals, pass them explicitly "
                "via column_mapping={'past_covariates': [...]}."
            )
        if time_range.get("coverage_pct", 100) < 95:
            suggestions.append(
                "Coverage is below 95%. Gaps up to 2h are interpolated "
                "automatically; larger gaps will remain as NaN and may hurt "
                "model accuracy."
            )
        # Gap → NaN-at-training-time risk.
        # Darts inserts NaN rows for every missing timestamp when it builds a
        # TimeSeries with fill_missing_dates=True (the default).  Sklearn
        # models like LinearRegression then refuse to train with
        # 'Input X contains NaN'.  This is a concrete failure mode, not just
        # a quality warning, so we surface it explicitly.
        if gap_report.get("n_gaps", 0) > 0:
            suggestions.append(
                "NaN training-failure risk: Darts inserts NaN rows for each "
                "missing timestamp during TimeSeries construction "
                "(fill_missing_dates=True). Models like LinearRegression will "
                "then fail with 'Input X contains NaN'. Fill every gap in the "
                "CSV before calling train_forecast_model — forward-fill (ffill) "
                "is usually the right choice for weather or load data."
            )
        # UTC timezone conversion hint.
        if any(f.startswith("POSSIBLE_UTC") for f in quality_flags):
            suggestions.append(
                "Timezone conversion pattern: "
                "df['date'] = (df['date'].dt.tz_localize('UTC')"
                ".dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)) "
                "converts naive UTC strings to Pacific local time. Adjust the "
                "timezone string for your region."
            )
        # DST merge fix hint.
        if any(f.startswith("DST_MERGE_RISK") for f in quality_flags):
            suggestions.append(
                "DST fix pattern after UTC-to-local conversion: "
                "(1) weather.drop_duplicates(subset='date_local', keep='first') "
                "removes the November fall-back duplicate; "
                "(2) merge with a left join from the load file, then call "
                "ffill() on the weather columns to fill the March spring-forward "
                "gap. See scripts/merge_lents_weather.py for the full example."
            )
        # Outlier suggestions
        if outlier_report.get("analysis_available"):
            _n_val = outlier_report["value_outliers"]["n_outliers"]
            _n_spk = outlier_report["step_change_spikes"]["n_spikes"]
            if _n_val > 0:
                suggestions.append(
                    f"{_n_val} value outlier(s) detected (beyond ±3σ). Consider "
                    "clipping or replacing with interpolated values before training to "
                    "prevent them from dominating the loss function. "
                    "Use: df[col] = df[col].clip(lower=p5, upper=p95)."
                )
            if _n_spk > 0:
                suggestions.append(
                    f"{_n_spk} sudden step-change spike(s) detected. Inspect the "
                    "timestamps listed in outlier_analysis.step_change_spikes and "
                    "replace with ffill() or interpolated values if they are sensor "
                    "faults, not real demand events."
                )
        # Seasonality suggestions
        if seasonality_report.get("analysis_available"):
            _hod = seasonality_report["hour_of_day"]
            _dow = seasonality_report["day_of_week"]
            suggestions.append(
                f"Diurnal pattern: peak load at hour {_hod['peak_hour']:02d}:00 "
                f"(mean={_hod['peak_mean']}), trough at hour {_hod['trough_hour']:02d}:00 "
                f"(mean={_hod['trough_mean']}), ratio={_hod['peak_to_trough_ratio']}x. "
                f"Weekly pattern: highest on {_dow['peak_day']} "
                f"(mean={_dow['peak_mean']}), lowest on {_dow['trough_day']} "
                f"(mean={_dow['trough_mean']}), ratio={_dow['peak_to_trough_ratio']}x."
            )
        # Covariate correlation suggestions
        if covariate_correlation:
            _weak = [c["column"] for c in covariate_correlation if c["interpretation"] == "weak"]
            _strong = [c["column"] for c in covariate_correlation if c["interpretation"] in ("strong", "moderate")]
            if _strong:
                suggestions.append(
                    f"Covariate(s) {_strong} show moderate-to-strong correlation with "
                    "the target (|r| ≥ 0.3) — likely to improve model accuracy."
                )
            if _weak:
                suggestions.append(
                    f"Covariate(s) {_weak} show weak correlation with the target "
                    "(|r| < 0.3) — consider dropping them or engineering interaction "
                    "features before training."
                )
        if loader is not None and n_rows_clean < 500:
            suggestions.append(
                f"Only {n_rows_clean} rows after deduplication. At least ~500 "
                "rows (ideally 3+ months) are recommended for reliable training."
            )
        if target_col and loader is not None:
            # Check if data is long enough for default lookback/horizon
            _freq_for_steps = actual_frequency if actual_frequency in FREQ_TO_STEPS_PER_HOUR else "h"
            min_needed = hours_to_steps(24, _freq_for_steps) + hours_to_steps(6, _freq_for_steps) + 100
            if n_rows_clean < min_needed:
                suggestions.append(
                    f"Data has {n_rows_clean} rows, which may be insufficient "
                    f"for default parameters (lookback=24h, horizon=6h). "
                    f"Minimum recommended: {min_needed} rows."
                )
            else:
                suggestions.append(
                    f"Data length ({n_rows_clean} rows) is sufficient for "
                    "training with default parameters."
                )

        # ------------------------------------------------------------------
        # Readiness verdict
        # ------------------------------------------------------------------
        blocking = [f for f in quality_flags if f.startswith(("LOAD_ERROR", "UNSUPPORTED_FREQUENCY"))]
        ready_to_train = loader_error is None and freq_report["supported"] and bool(target_col)

        return create_success_response(
            file_path=csv_path,
            n_rows_raw=n_rows_raw,
            columns=column_report,
            frequency=freq_report,
            time_range=time_range,
            column_statistics=column_stats,
            gaps=gap_report,
            outlier_analysis=outlier_report,
            seasonality=seasonality_report,
            covariate_correlation=covariate_correlation,
            quality_flags=quality_flags,
            suggestions=suggestions,
            ready_to_train=ready_to_train,
            blocking_issues=blocking,
            loader_error=loader_error,
        )

    except Exception as e:
        logger.exception("inspect_data failed")
        return create_error_response(f"Inspection failed: {str(e)}")


# ---------------------------------------------------------------------------
# Private helpers for inspect_data
# ---------------------------------------------------------------------------

def _sniff_frequency(raw_df: pd.DataFrame) -> Optional[str]:
    """
    Attempt to infer frequency from the first column that parses as datetimes.
    Returns a frequency string or None.

    Skips obviously non-datetime columns (pure numeric dtypes) before trying
    pd.to_datetime, which would otherwise parse all rows before failing.
    """
    for col in raw_df.columns:
        # Fast dtype pre-check: numeric columns are never datetime strings.
        if pd.api.types.is_numeric_dtype(raw_df[col]) and not pd.api.types.is_datetime64_any_dtype(raw_df[col]):
            continue
        try:
            parsed = pd.to_datetime(raw_df[col], utc=True)
            idx = pd.DatetimeIndex(parsed)
            return _infer_frequency(idx)
        except Exception:
            continue
    return None


def _sniff_columns(raw_df: pd.DataFrame, mapping: Optional[dict]) -> dict:
    """
    Best-effort column role detection without running the full loader.
    Used as fallback when ForecastingDataLoader.__init__ fails.
    """
    if mapping:
        result = dict(mapping)
        if "past_covariates" not in result:
            result["past_covariates"] = []
        if "future_covariates" not in result:
            result["future_covariates"] = []
        return result

    dt_patterns = DEFAULT_DATETIME_PATTERNS
    target_patterns = DEFAULT_TARGET_PATTERNS
    cov_patterns = DEFAULT_COVARIATE_PATTERNS

    result: dict = {"past_covariates": [], "future_covariates": []}
    for col in raw_df.columns:
        cl = col.lower()
        if "datetime" not in result and any(p in cl for p in dt_patterns):
            result["datetime"] = col
        elif "target" not in result and any(p in cl for p in target_patterns):
            result["target"] = col
        elif any(p in cl for p in cov_patterns):
            result["past_covariates"].append(col)
    return result
