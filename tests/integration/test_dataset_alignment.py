"""Guard the timestamp alignment of the CSVs used by the paper example.

A weather column that is shifted against the load column it describes is one
of the few data faults that nothing else in the pipeline can catch.  Column
names still match, row counts still match, the merge still succeeds, and
training still converges -- the model simply learns a relationship that is not
there.  It shows up only as a test-time error increase, which is easy to
misread as a genuine generalisation gap.

This fault was present in an earlier revision of
``2023_city_level_observed.csv``: its weather had been converted local->UTC a
second time, moving every reading 7 h (PDT) or 8 h (PST) earlier against the
load.  Training on the correctly aligned 2021 file and testing on that file
inflated TiDE's test MAPE from 3.93% to 9.04% and its CV-RMSE from 5.70% to
12.18%.

The checks below are physical rather than statistical: at Portland's latitude
solar radiation must peak near solar noon and air temperature a couple of
hours after it.  A whole-day shift moves both into the night, which no
plausible measurement error can do.
"""
import os

import pandas as pd
import pytest

AMI_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "examples", "AMI"
)

# Files the Section 3 example trains and evaluates on, plus the weather source
# they are both built from.
PAPER_FILES = [
    "2021_city_level_with_weather.csv",
    "2023_city_level_observed.csv",
]
WEATHER_SRC = "2021-2023_openmeteo_weather.csv"

TARGET = "City (n=41703)"


def _mean_by_hour(df, dt_col, value_col):
    return df.groupby(df[dt_col].dt.hour)[value_col].mean()


def _load(name, dt_col="Datetime"):
    path = os.path.join(AMI_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not present")
    df = pd.read_csv(path)
    df[dt_col] = pd.to_datetime(df[dt_col])
    return df


@pytest.mark.parametrize("name", PAPER_FILES)
def test_solar_radiation_peaks_around_solar_noon(name):
    """Direct radiation must peak in the middle of the day, not before dawn."""
    df = _load(name)
    peak_hour = _mean_by_hour(df, "Datetime", "Direct_Radiation").idxmax()
    assert 10 <= peak_hour <= 16, (
        f"{name}: mean direct radiation peaks at {peak_hour:02d}:00. Expected "
        f"roughly 13:00. The weather columns are shifted against the clock; "
        f"regenerate with figures/build_paper_datasets.py."
    )


@pytest.mark.parametrize("name", PAPER_FILES)
def test_temperature_peaks_in_the_afternoon(name):
    """Air temperature lags solar noon by a few hours."""
    df = _load(name)
    peak_hour = _mean_by_hour(df, "Datetime", "T_out").idxmax()
    assert 12 <= peak_hour <= 19, (
        f"{name}: mean temperature peaks at {peak_hour:02d}:00. Expected "
        f"roughly 15:00. The weather columns are shifted against the clock; "
        f"regenerate with figures/build_paper_datasets.py."
    )


@pytest.mark.parametrize("name", PAPER_FILES)
def test_radiation_is_zero_overnight(name):
    """There is no direct beam radiation in the small hours."""
    df = _load(name)
    by_hour = _mean_by_hour(df, "Datetime", "Direct_Radiation")
    night = by_hour.loc[[0, 1, 2, 3, 23]]
    assert (night < 1.0).all(), (
        f"{name}: non-zero mean direct radiation overnight "
        f"({night.round(1).to_dict()}). The weather columns are shifted."
    )


@pytest.mark.parametrize("name", PAPER_FILES)
def test_load_peaks_in_the_evening(name):
    """Sanity-check the target column itself is not shifted."""
    df = _load(name)
    peak_hour = _mean_by_hour(df, "Datetime", TARGET).idxmax()
    assert 16 <= peak_hour <= 21, (
        f"{name}: mean load peaks at {peak_hour:02d}:00, expected a 17:00-20:00 "
        f"evening peak for a winter-peaking city aggregate."
    )


def test_train_and_test_files_share_a_timestamp_convention():
    """The two paper files must not disagree about what hour a reading is.

    Compares the two years hour by hour.  A whole-day phase difference between
    them is the specific fault this module exists to prevent, and it survives
    every per-file check above if *both* files are shifted by the same amount.
    """
    profiles = {}
    for name in PAPER_FILES:
        df = _load(name)
        profiles[name] = _mean_by_hour(df, "Datetime", "Direct_Radiation").idxmax()

    peaks = list(profiles.values())
    assert max(peaks) - min(peaks) <= 2, (
        f"Radiation peak hour differs between the paper files: {profiles}. "
        f"They must be built from one weather source by "
        f"figures/build_paper_datasets.py."
    )


def test_weather_source_is_local_time():
    """The source the paper files are built from is local, not UTC."""
    df = _load(WEATHER_SRC, dt_col="date")
    t_peak = _mean_by_hour(df, "date", "T_out").idxmax()
    r_peak = _mean_by_hour(df, "date", "Direct_Radiation").idxmax()
    assert 12 <= t_peak <= 19 and 10 <= r_peak <= 16, (
        f"{WEATHER_SRC}: T_out peaks {t_peak:02d}:00, Direct_Radiation "
        f"{r_peak:02d}:00 -- this export is not Portland local time. "
        f"build_paper_datasets.py assumes it is."
    )
