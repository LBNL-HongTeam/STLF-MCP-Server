"""
Build the JSON payload for the data-inspection HTML report.

Everything the browser draws is computed here, in pandas, from the same
``ForecastingDataLoader`` the training tools use, so what the report shows
is what a model would be trained on:

- the target and every numeric covariate as aligned columnar arrays;
- the train/validation split as date segments (via ``core.splits``), using
  the loader's own ``split_train_val`` / ``split_train_val_seasonal``;
- season spans, gaps in the timestamp index, value/step-change outliers;
- hour-of-day profiles per season, day-of-week and monthly profiles;
- a day x hour heatmap (hourly means);
- per-covariate relationship to the target: Pearson r, a binned-mean curve
  (reveals heating/cooling regimes without fitting a change-point model),
  and a scatter sample coloured by hour of day.

The JSON is intentionally columnar (parallel arrays) to keep a year of
15-minute data with several covariates well under a few megabytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..core.data_loader import ForecastingDataLoader
from ..core.splits import (
    ROLE_TRAINING,
    ROLE_VALIDATION,
    SEASON_ORDER,
    describe_split,
    season_of,
    step_for_frequency,
    summarize_segments,
)
from ._render import to_html_json

SPLIT_STRATEGIES = ("seasonal", "sequential", "none")

_SCATTER_SAMPLE = 3000
_RELATION_BINS = 24
_OUTLIER_SIGMA = 3.0


@dataclass
class DataReportPayload:
    meta: dict
    series: dict
    split: dict
    seasons: list
    gaps: list
    outliers: dict
    profiles: dict
    heatmap: list
    relations: dict
    windows: list = field(default_factory=list)
    inspection: dict = field(default_factory=dict)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def payload_to_json(payload: DataReportPayload) -> str:
    return to_html_json(payload.__dict__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(ts: Any) -> Optional[str]:
    try:
        return pd.Timestamp(ts).isoformat()
    except Exception:
        return None


def _f(v: Any, n: int = 4) -> Optional[float]:
    """Round to n decimals; None for NaN/inf so JSON stays clean."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, n) if math.isfinite(x) else None


def _season_spans(index: pd.DatetimeIndex) -> list[dict]:
    """Contiguous runs of the same meteorological season, for background bands."""
    if len(index) == 0:
        return []
    seasons = [season_of(t) for t in index]
    spans: list[dict] = []
    start = index[0]
    for i in range(1, len(index)):
        if seasons[i] != seasons[i - 1]:
            spans.append({"season": seasons[i - 1], "start": _iso(start), "end": _iso(index[i - 1])})
            start = index[i]
    spans.append({"season": seasons[-1], "start": _iso(start), "end": _iso(index[-1])})
    return spans


def _gaps(index: pd.DatetimeIndex, step: pd.Timedelta) -> list[dict]:
    """Missing-timestamp runs: jumps larger than 1.5 x the nominal step."""
    if len(index) < 2:
        return []
    diffs = index.to_series().diff()
    out: list[dict] = []
    for ts, d in diffs.items():
        if pd.isna(d) or d <= step * 1.5:
            continue
        prev = ts - d
        out.append(
            {
                "start": _iso(prev),
                "end": _iso(ts),
                "n_missing_steps": int(round(d / step)) - 1,
                "hours": _f(d.total_seconds() / 3600, 2),
            }
        )
    return out


def _outliers(target: pd.Series) -> dict:
    """Same rules as inspect_data: |value - mean| > 3 sigma; |step| > 3 sigma(diff)."""
    s = target.dropna()
    result: dict = {"value": [], "spikes": [], "thresholds": {}}
    if len(s) < 10:
        return result
    mean, std = float(s.mean()), float(s.std())
    if std > 0:
        hi, lo = mean + _OUTLIER_SIGMA * std, mean - _OUTLIER_SIGMA * std
        mask = (s > hi) | (s < lo)
        result["value"] = [{"t": _iso(t), "v": _f(v)} for t, v in s[mask].items()]
        result["thresholds"].update({"value_high": _f(hi), "value_low": _f(lo)})
    # Step changes: identical rule to inspect_data -- 3 sigma of the *absolute*
    # first differences, ignored when that threshold is below 1% of the mean
    # (constant-ish series would otherwise flag every step).
    signed = s.diff().dropna()
    abs_diffs = signed.abs()
    threshold = _OUTLIER_SIGMA * float(abs_diffs.std())
    if threshold >= max(abs(mean) * 0.01, 1e-6):
        mask = abs_diffs > threshold
        result["spikes"] = [
            {"t": _iso(t), "v": _f(s.loc[t]), "step": _f(signed.loc[t])}
            for t in abs_diffs[mask].index
        ]
        result["thresholds"]["step_change"] = _f(threshold)
    return result


def _profiles(target: pd.Series) -> dict:
    """Hour-of-day (per season + overall), day-of-week, and monthly profiles."""
    df = pd.DataFrame({"v": target})
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["season"] = [season_of(t) for t in df.index]

    def _agg(frame: pd.DataFrame, keys: list[str]) -> list[dict]:
        g = frame.groupby(keys)["v"]
        stats = pd.DataFrame(
            {
                "mean": g.mean(),
                "p25": g.quantile(0.25),
                "p75": g.quantile(0.75),
                "n": g.count(),
            }
        ).reset_index()
        rows = []
        for rec in stats.to_dict("records"):
            row = {k: (int(rec[k]) if k != "season" else rec[k]) for k in keys}
            row.update({"mean": _f(rec["mean"]), "p25": _f(rec["p25"]), "p75": _f(rec["p75"]), "n": int(rec["n"])})
            rows.append(row)
        return rows

    hour_by_season = _agg(df, ["season", "hour"])
    hour_by_season.sort(key=lambda r: (SEASON_ORDER.index(r["season"]), r["hour"]))
    weekend = df["dow"] >= 5
    return {
        "hour_of_day": _agg(df, ["hour"]),
        "hour_of_day_by_season": hour_by_season,
        "hour_of_day_weekday": _agg(df[~weekend], ["hour"]),
        "hour_of_day_weekend": _agg(df[weekend], ["hour"]) if weekend.any() else [],
        "day_of_week": _agg(df, ["dow"]),
        "month": _agg(df, ["month"]),
    }


def _heatmap(target: pd.Series) -> list[dict]:
    """Hourly means keyed by (date, hour) -- the classic load carpet plot."""
    hourly = target.resample("h").mean().dropna()
    return [
        {"date": t.strftime("%Y-%m-%d"), "hour": int(t.hour), "v": _f(v)}
        for t, v in hourly.items()
    ]


def _relation(target: pd.Series, cov: pd.Series, rng: np.random.Generator) -> dict:
    """How the target responds to one covariate."""
    both = pd.DataFrame({"x": cov, "y": target}).dropna()
    out: dict = {"n": int(len(both)), "pearson_r": None, "binned": [], "scatter": []}
    if len(both) < 3 or both["x"].std() == 0 or both["y"].std() == 0:
        return out
    out["pearson_r"] = _f(both["x"].corr(both["y"]))

    # Binned mean curve: shows shape (e.g. heating/cooling V) without a model fit.
    edges = np.linspace(both["x"].min(), both["x"].max(), _RELATION_BINS + 1)
    bins = pd.cut(both["x"], bins=edges, include_lowest=True)
    g = both.groupby(bins, observed=True)["y"]
    out["binned"] = [
        {"x": _f(iv.mid), "mean": _f(m), "p25": _f(q1), "p75": _f(q3), "n": int(n)}
        for iv, m, q1, q3, n in zip(g.mean().index, g.mean(), g.quantile(0.25), g.quantile(0.75), g.count())
        if n > 0
    ]

    # Scatter sample coloured by hour of day.
    sample = both if len(both) <= _SCATTER_SAMPLE else both.iloc[
        np.sort(rng.choice(len(both), _SCATTER_SAMPLE, replace=False))
    ]
    out["scatter"] = [
        {"x": _f(r.x, 3), "y": _f(r.y, 3), "hour": int(t.hour)}
        for t, r in sample.iterrows()
    ]
    return out


def _peak_windows(target: pd.Series, days_before: int = 3, days_after: int = 4) -> list[dict]:
    """One zoom window per season, centred on that season's peak load.

    Mirrors the 'all-year plus typical winter / summer week' layout used in
    the Li et al. paper figures, but chosen automatically from the data.
    """
    s = target.dropna()
    if s.empty:
        return []
    seasons = pd.Series([season_of(t) for t in s.index], index=s.index)
    out: list[dict] = []
    lo, hi = s.index.min(), s.index.max()
    for season in SEASON_ORDER:
        part = s[seasons == season]
        if part.empty:
            continue
        peak_t = part.idxmax()
        start = max(lo, peak_t - pd.Timedelta(days=days_before))
        end = min(hi, peak_t + pd.Timedelta(days=days_after))
        out.append(
            {
                "label": f"{season.capitalize()} peak week",
                "season": season,
                "peak_time": _iso(peak_t),
                "peak_value": _f(part.max()),
                "start": _iso(start),
                "end": _iso(end),
            }
        )
    return out


def _numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_data_report_payload(
    loader: ForecastingDataLoader,
    *,
    csv_path: str,
    covariates: Optional[list[str]] = None,
    split_strategy: str = "seasonal",
    validation_split: float = 0.2,
    title: Optional[str] = None,
    max_points: int = 60000,
    inspection: Optional[dict] = None,
    seed: int = 0,
) -> DataReportPayload:
    """Compute every block of the data report from a loaded (unsplit) loader.

    Args:
        loader: ``ForecastingDataLoader`` built with
            ``add_calendar_features=False, lag_hours=[]`` so only the user's
            own columns are present.
        csv_path: Path recorded in the report header.
        covariates: Columns to draw as covariate panels.  Default: every
            numeric column other than the target, whether or not the column
            mapping recognises it -- unmapped columns are labelled as such so
            a missed auto-detection is visible.
        split_strategy: "seasonal" (Li et al. 2025, falls back to sequential
            when fewer than four seasons are present), "sequential", or "none".
        validation_split: Fraction held out (per season for "seasonal").
        title: Report title.
        max_points: Overview series longer than this are stride-downsampled
            for the browser; every aggregate is still computed on full data.
        inspection: Optional ``inspect_data`` result to embed (quality flags,
            statistics, suggestions, readiness verdict).
        seed: RNG seed for the scatter sample, for reproducible reports.
    """
    if split_strategy not in SPLIT_STRATEGIES:
        raise ValueError(f"split_strategy must be one of {SPLIT_STRATEGIES}, got {split_strategy!r}")

    df = loader.df
    mapping = loader.column_mapping
    target_col = mapping["target"]
    target = df[target_col].astype(float)
    index = df.index
    frequency = getattr(loader, "inferred_frequency", None) or loader.frequency
    step = step_for_frequency(frequency)

    past = list(mapping.get("past_covariates") or [])
    future = list(mapping.get("future_covariates") or [])
    if covariates is None:
        cov_cols = _numeric_columns(df, exclude={target_col})
    else:
        missing = [c for c in covariates if c not in df.columns]
        if missing:
            raise ValueError(f"covariates not found in CSV: {missing}. Available: {list(df.columns)}")
        cov_cols = list(covariates)

    def _role(col: str) -> str:
        if col in past:
            return "past_covariate"
        if col in future:
            return "future_covariate"
        return "unmapped"

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    seasons_present = {season_of(t) for t in index[:: max(1, len(index) // 500)]} | {season_of(index[-1])}
    effective = split_strategy
    note = None
    segments: list[dict] = []
    if split_strategy == "seasonal":
        if len(seasons_present) < 4:
            effective = "sequential"
            note = (
                f"Seasonal split requested but only {sorted(seasons_present)} present; "
                "the trainer falls back to a sequential split for this file."
            )
            tr, va = loader.split_train_val(validation_split)
        else:
            tr, va = loader.split_train_val_seasonal(validation_split)
        segments = describe_split(tr.df.index, va.df.index, step, label_seasons=(effective == "seasonal"))
    elif split_strategy == "sequential":
        tr, va = loader.split_train_val(validation_split)
        segments = describe_split(tr.df.index, va.df.index, step, label_seasons=False)

    split = {
        "requested": split_strategy,
        "strategy": effective,
        "validation_split": validation_split,
        "segments": segments,
        "summary": summarize_segments(segments) if segments else {},
        "note": note,
    }

    # Per-row role for colouring the target line: 0 training, 1 validation, -1 none.
    role = np.full(len(index), -1, dtype=np.int8)
    if segments:
        for seg in segments:
            lo, hi = pd.Timestamp(seg["start"]), pd.Timestamp(seg["end"])
            mask = (index >= lo) & (index <= hi)
            role[mask] = 0 if seg["role"] == ROLE_TRAINING else 1

    # ------------------------------------------------------------------
    # Series (columnar, optionally stride-downsampled for the overview)
    # ------------------------------------------------------------------
    n = len(index)
    stride = max(1, math.ceil(n / max_points))
    sel = slice(None, None, stride)
    series = {
        "t": [_iso(t) for t in index[sel]],
        "target": [_f(v) for v in target.values[sel]],
        "role": role[sel].tolist(),
        "covariates": {
            c: [_f(v) for v in df[c].astype(float).values[sel]] for c in cov_cols
        },
        "downsampled": stride > 1,
        "stride": stride,
        "n_points": len(index[sel]),
    }

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    relations = {c: _relation(target, df[c].astype(float), rng) for c in cov_cols}

    meta = {
        "title": title or f"Data report — {target_col}",
        "csv_path": csv_path,
        "n_rows": int(n),
        "start": _iso(index.min()),
        "end": _iso(index.max()),
        "days": _f((index.max() - index.min()).total_seconds() / 86400, 1),
        "frequency": frequency,
        "target": target_col,
        "target_stats": {
            "mean": _f(target.mean()), "std": _f(target.std()),
            "min": _f(target.min()), "max": _f(target.max()),
            "n_missing": int(target.isna().sum()),
        },
        "covariates": [
            {
                "name": c,
                "role": _role(c),
                "mean": _f(df[c].mean()), "min": _f(df[c].min()), "max": _f(df[c].max()),
                "n_missing": int(df[c].isna().sum()),
                "pearson_r": relations[c]["pearson_r"],
            }
            for c in cov_cols
        ],
        "unmapped_covariates": [c for c in cov_cols if _role(c) == "unmapped"],
    }

    return DataReportPayload(
        meta=meta,
        series=series,
        split=split,
        seasons=_season_spans(index),
        gaps=_gaps(index, step),
        outliers=_outliers(target),
        profiles=_profiles(target),
        heatmap=_heatmap(target),
        relations=relations,
        windows=_peak_windows(target),
        inspection=inspection or {},
    )
