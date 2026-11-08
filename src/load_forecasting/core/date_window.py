"""Date-window selection shared by every tool that reads a load CSV.

Tools accept ``start_date`` / ``end_date`` so an agent can restrict a CSV to a
period (e.g. "2023 excluding November and December") without writing a new
file.  This module owns the one interpretation of those strings so training,
evaluation, backtesting and inspection all agree:

* ``start_date`` is inclusive.
* ``end_date`` is inclusive.  A date-only value (``"2023-10-31"``) covers that
  whole day, i.e. every timestamp before ``2023-11-01 00:00``; a value with a
  time component (``"2023-10-31 12:00"``) is inclusive of that exact instant.
* A value without a UTC offset is read in the data's own clock.  The loader
  parses naive CSV timestamps as UTC-labelled wall time, so a naive bound lines
  up with the timestamps exactly as written in the file.  A value *with* an
  offset (``"2023-11-01T00:00-08:00"``) is converted to the data's timezone.

Only contiguous windows are supported here.  Dropping a period from the
*middle* of a series must not be done by deleting rows: the Darts conversion
re-inserts missing timestamps and interpolates across them, which would
fabricate load for the excluded period.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# A time component is present when the string has "HH:MM" after the date.
_TIME_COMPONENT = re.compile(r"\d{1,2}:\d{2}")


class DateWindowError(ValueError):
    """Raised for an unparsable or empty date window."""


def _parse_bound(value: str, name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(str(value).strip())
    except Exception:
        ts = None
    if ts is None or ts is pd.NaT:
        raise DateWindowError(
            f"{name}={value!r} is not a valid date. Use 'YYYY-MM-DD' "
            "(whole day) or 'YYYY-MM-DD HH:MM' (exact time)."
        )
    return ts


def align_to_index_tz(ts: pd.Timestamp, tz) -> pd.Timestamp:
    """Express ``ts`` in timezone ``tz`` (``None`` = tz-naive index).

    A naive ``ts`` is taken to already be in the index's clock and is only
    labelled; an aware ``ts`` is converted.
    """
    if tz is None:
        if ts.tzinfo is None:
            return ts
        return ts.tz_convert("UTC").tz_localize(None)
    if ts.tzinfo is None:
        return ts.tz_localize(tz)
    return ts.tz_convert(tz)


def parse_timestamp(value: str, name: str, tz=None) -> pd.Timestamp:
    """Parse one user-supplied timestamp and align it to the index timezone."""
    return align_to_index_tz(_parse_bound(value, name), tz)


@dataclass(frozen=True)
class DateWindow:
    """A contiguous, inclusive ``[start_date, end_date]`` selection."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None

    @classmethod
    def from_args(
        cls, start_date: Optional[str], end_date: Optional[str]
    ) -> Optional["DateWindow"]:
        """Return a validated window, or ``None`` when neither bound is given."""
        start_date = start_date or None
        end_date = end_date or None
        if start_date is None and end_date is None:
            return None
        window = cls(start_date=start_date, end_date=end_date)
        # Validate eagerly (format + ordering) in a timezone-free frame.
        start, end, end_inclusive = window._bounds(None)
        if start is not None and end is not None:
            empty = start > end if end_inclusive else start >= end
            if empty:
                raise DateWindowError(
                    f"start_date={start_date!r} is after end_date={end_date!r}; "
                    "the window would be empty."
                )
        return window

    def _bounds(self, tz):
        """Return (start, end, end_inclusive) aligned to ``tz``."""
        start = (
            parse_timestamp(self.start_date, "start_date", tz)
            if self.start_date
            else None
        )
        end = None
        end_inclusive = True
        if self.end_date:
            end = parse_timestamp(self.end_date, "end_date", tz)
            if not _TIME_COMPONENT.search(str(self.end_date)):
                # Date-only: cover the whole day -> exclusive next midnight.
                end = end.normalize() + pd.Timedelta(days=1)
                end_inclusive = False
        return start, end, end_inclusive

    def mask(self, index: pd.DatetimeIndex, start_offset: Optional[pd.Timedelta] = None) -> np.ndarray:
        """Boolean mask of ``index`` rows inside the window.

        Args:
            index: DatetimeIndex to select from (tz-aware or naive).
            start_offset: Optional extra history to keep *before* the start
                bound (e.g. so lag features can be computed from real data).
        """
        start, end, end_inclusive = self._bounds(index.tz)
        keep = np.ones(len(index), dtype=bool)
        if start is not None:
            if start_offset is not None:
                start = start - start_offset
            keep &= np.asarray(index >= start)
        if end is not None:
            keep &= np.asarray(index <= end if end_inclusive else index < end)
        return keep

    def as_dict(self) -> dict:
        return {"start_date": self.start_date, "end_date": self.end_date}


def parse_datetime_column(values: pd.Series) -> pd.Series:
    """Parse a raw datetime column the way ForecastingDataLoader does.

    Tz-aware strings are converted to UTC; naive strings are labelled UTC.
    Unparsable entries become NaT instead of raising, so a caller can filter a
    raw frame before the loader runs its own strict validation.
    """
    try:
        return pd.to_datetime(values, utc=True)
    except Exception:
        return pd.to_datetime(values, utc=True, errors="coerce")


def filter_frame(df: pd.DataFrame, datetime_col: str, window: DateWindow) -> pd.DataFrame:
    """Return the rows of a raw (un-indexed) frame whose timestamp is in ``window``."""
    parsed = parse_datetime_column(df[datetime_col])
    idx = pd.DatetimeIndex(parsed)
    return df.loc[window.mask(idx)].copy()


def describe_window(
    window: DateWindow,
    kept_index: pd.DatetimeIndex,
    rows_before: int,
    file_start=None,
    file_end=None,
) -> dict:
    """JSON-safe summary of what a window selected, for tool responses."""
    info = window.as_dict()
    info.update(
        rows_in_window=int(len(kept_index)),
        rows_excluded=int(rows_before - len(kept_index)),
        first_timestamp=kept_index.min().isoformat() if len(kept_index) else None,
        last_timestamp=kept_index.max().isoformat() if len(kept_index) else None,
    )
    if file_start is not None:
        info["file_start"] = pd.Timestamp(file_start).isoformat()
    if file_end is not None:
        info["file_end"] = pd.Timestamp(file_end).isoformat()
    return info
