"""
Describe a train/validation split as date-labelled segments.

``ForecastingDataLoader.split_train_val`` and ``split_train_val_seasonal``
return two loaders whose DataFrames are the training and validation rows.
For a sequential split those are two contiguous blocks; for the seasonal
split (Li et al. 2025) each meteorological season contributes a training
block followed by a validation block, and "winter" (Dec + Jan + Feb) is
non-contiguous within a calendar year.

``describe_split`` collapses the two row indexes into an ordered list of
segments -- ``{"role", "season", "start", "end", "n_rows"}`` -- splitting
wherever the role changes, the season changes, or there is a gap in time.
The result is what ``generate_data_report`` draws and what
``train_forecast_model`` records in ``data_info.split.segments``, so the
picture and the metadata always come from the same computation.
"""

from typing import Optional

import pandas as pd

SEASON_OF_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "fall", 10: "fall", 11: "fall",
}
SEASON_ORDER = ("winter", "spring", "summer", "fall")

ROLE_TRAINING = "training"
ROLE_VALIDATION = "validation"


def season_of(ts: pd.Timestamp) -> str:
    """Meteorological season of a timestamp (DJF / MAM / JJA / SON)."""
    return SEASON_OF_MONTH[int(ts.month)]


def _iso(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).isoformat()


def describe_split(
    train_index: pd.DatetimeIndex,
    val_index: pd.DatetimeIndex,
    step: pd.Timedelta,
    *,
    label_seasons: bool = True,
) -> list[dict]:
    """Return ordered segments describing which rows train and which validate.

    Args:
        train_index: Timestamps of the training rows.
        val_index: Timestamps of the validation rows.
        step: Nominal spacing between rows (e.g. ``pd.Timedelta("1h")``).
            A jump larger than ``1.5 * step`` starts a new segment.
        label_seasons: Attach the meteorological season of each segment and
            break segments at season boundaries.  Set False for a plain
            sequential split where seasons are irrelevant.

    Returns:
        List of ``{"role", "season", "start", "end", "n_rows"}`` dicts sorted
        by ``start``.  ``season`` is None when ``label_seasons`` is False.
    """
    rows: list[tuple[pd.Timestamp, str]] = [
        (pd.Timestamp(t), ROLE_TRAINING) for t in train_index
    ] + [(pd.Timestamp(t), ROLE_VALIDATION) for t in val_index]
    if not rows:
        return []
    rows.sort(key=lambda r: r[0])

    gap_limit = step * 1.5
    segments: list[dict] = []
    seg_start, seg_role = rows[0]
    seg_season = season_of(seg_start) if label_seasons else None
    seg_n = 1
    prev_ts = seg_start

    def _close(end_ts: pd.Timestamp) -> None:
        segments.append(
            {
                "role": seg_role,
                "season": seg_season,
                "start": _iso(seg_start),
                "end": _iso(end_ts),
                "n_rows": seg_n,
            }
        )

    for ts, role in rows[1:]:
        season = season_of(ts) if label_seasons else None
        breaks = (
            role != seg_role
            or season != seg_season
            or (ts - prev_ts) > gap_limit
        )
        if breaks:
            _close(prev_ts)
            seg_start, seg_role, seg_season, seg_n = ts, role, season, 1
        else:
            seg_n += 1
        prev_ts = ts
    _close(prev_ts)
    return segments


def summarize_segments(segments: list[dict]) -> dict:
    """Roll segments up into counts and per-season row totals for metadata."""
    by_role: dict[str, int] = {ROLE_TRAINING: 0, ROLE_VALIDATION: 0}
    by_season: dict[str, dict[str, int]] = {}
    for seg in segments:
        by_role[seg["role"]] = by_role.get(seg["role"], 0) + seg["n_rows"]
        if seg.get("season"):
            bucket = by_season.setdefault(seg["season"], {ROLE_TRAINING: 0, ROLE_VALIDATION: 0})
            bucket[seg["role"]] = bucket.get(seg["role"], 0) + seg["n_rows"]
    total = sum(by_role.values()) or 1
    return {
        "n_segments": len(segments),
        "training_rows": by_role[ROLE_TRAINING],
        "validation_rows": by_role[ROLE_VALIDATION],
        "validation_fraction": round(by_role[ROLE_VALIDATION] / total, 4),
        "by_season": {s: by_season[s] for s in SEASON_ORDER if s in by_season},
    }


def step_for_frequency(frequency: Optional[str]) -> pd.Timedelta:
    """Nominal row spacing for a supported frequency string ('15min', '30min', 'h')."""
    return {
        "15min": pd.Timedelta(minutes=15),
        "30min": pd.Timedelta(minutes=30),
        "h": pd.Timedelta(hours=1),
    }.get(frequency or "h", pd.Timedelta(hours=1))
