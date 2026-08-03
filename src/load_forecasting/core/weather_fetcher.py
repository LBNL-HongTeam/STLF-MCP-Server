"""
Open-Meteo weather forecast client.

Fetches live hourly or 15-minutely weather forecast data from the Open-Meteo
free API (https://api.open-meteo.com) and returns a tidy pandas DataFrame
aligned to the requested temporal resolution and timezone.

No API key is required for non-commercial use.

Typical use in the MCP tool layer:
    from .weather_fetcher import fetch_weather_forecast_df, VARIABLE_PRESETS

    df = fetch_weather_forecast_df(
        latitude=37.77,
        longitude=-122.41,
        variables=VARIABLE_PRESETS["building_load"],
        forecast_hours=72,
        timezone="America/Los_Angeles",
        resolution="hourly",
    )
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Base URL for the Open-Meteo forecast API (free, no key required).
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

#: Curated variable presets meaningful for building electrical load forecasting.
#: Keys map to the Open-Meteo hourly parameter names accepted by the API.
VARIABLE_PRESETS: dict[str, list[str]] = {
    # Minimal set: temperature and humidity — covers the major drivers.
    "minimal": [
        "temperature_2m",
        "relative_humidity_2m",
    ],
    # Standard set used for most building load models.
    "building_load": [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "shortwave_radiation",
        "wind_speed_10m",
        "cloud_cover",
        "precipitation",
    ],
    # Extended set for research / feature selection.
    "extended": [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "wind_speed_10m",
        "wind_direction_10m",
        "cloud_cover",
        "precipitation",
        "surface_pressure",
        "is_day",
    ],
}

#: Friendly column name aliases applied when the caller sets rename_columns=True.
#: Open-Meteo names → short names expected by ForecastingDataLoader auto-detection.
_RENAME_MAP: dict[str, str] = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "relative_humidity",
    "apparent_temperature": "apparent_temperature",
    "dew_point_2m": "dew_point",
    "shortwave_radiation": "solar_radiation",
    "direct_radiation": "solar_direct",
    "diffuse_radiation": "solar_diffuse",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "cloud_cover": "cloud_cover",
    "precipitation": "precipitation",
    "surface_pressure": "surface_pressure",
    "is_day": "is_day",
}


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------

def fetch_weather_forecast_df(
    latitude: float,
    longitude: float,
    variables: list[str],
    forecast_hours: int = 72,
    past_hours: int = 0,
    timezone: str = "UTC",
    resolution: str = "hourly",
    rename_columns: bool = True,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """
    Fetch weather forecast data from Open-Meteo and return a DataFrame.

    Calls https://api.open-meteo.com/v1/forecast with the given parameters.
    The returned DataFrame has a datetime column (naive, in the requested
    timezone) and one column per requested weather variable.  Rows cover
    ``past_hours`` of archive data followed by ``forecast_hours`` of future
    forecast, giving a continuous time window suitable for passing directly
    to ``generate_forecast`` as future-covariate rows.

    Args:
        latitude: WGS84 latitude (-90 to 90).
        longitude: WGS84 longitude (-180 to 180).
        variables: List of Open-Meteo hourly variable names (e.g.
            ``["temperature_2m", "relative_humidity_2m"]``).  Use
            ``VARIABLE_PRESETS["building_load"]`` for the standard set.
        forecast_hours: Number of future hours to fetch (1–384).
            Open-Meteo supports up to 16 days (384h) for hourly data.
        past_hours: Number of archive hours to include before the current
            time (0–92).  Useful for filling the model's context window
            without needing a separate historical CSV.
        timezone: IANA timezone string (e.g. ``"America/Los_Angeles"``).
            Timestamps in the returned DataFrame are in this timezone but
            stored as naive datetimes (no tzinfo), consistent with how
            ``ForecastingDataLoader`` expects data.
        resolution: ``"hourly"`` (default) or ``"minutely_15"``.  15-minute
            data is only natively available for Central Europe and North
            America; elsewhere it is interpolated from hourly.
        rename_columns: If True (default), rename Open-Meteo variable names
            to shorter aliases that match ``ForecastingDataLoader``
            auto-detection patterns (e.g. ``temperature_2m`` → ``temperature``).
        datetime_col: Name for the datetime column in the output DataFrame.

    Returns:
        DataFrame with ``datetime_col`` and one column per weather variable,
        sorted ascending by time.  The datetime column contains naive
        ``pandas.Timestamp`` values in the requested timezone.

    Raises:
        ImportError: If ``httpx`` is not installed.
        RuntimeError: If the Open-Meteo API returns an error response.
        ValueError: On invalid parameter combinations.
    """
    if httpx is None:
        raise ImportError(
            "httpx is required for weather fetching. "
            "Install it with: pip install httpx"
        )

    if not variables:
        raise ValueError("At least one weather variable must be specified.")
    if not (1 <= forecast_hours <= 384):
        raise ValueError("forecast_hours must be between 1 and 384.")
    if not (0 <= past_hours <= 92):
        raise ValueError("past_hours must be between 0 and 92.")

    if resolution not in ("hourly", "minutely_15"):
        raise ValueError("resolution must be 'hourly' or 'minutely_15'.")

    # Build query params
    params: dict = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "timeformat": "iso8601",
        "forecast_hours": forecast_hours,
    }
    if past_hours > 0:
        params["past_hours"] = past_hours

    if resolution == "minutely_15":
        params["minutely_15"] = ",".join(variables)
    else:
        params["hourly"] = ",".join(variables)

    logger.debug(
        "Fetching Open-Meteo forecast: lat=%.4f lon=%.4f vars=%s "
        "forecast_hours=%d past_hours=%d timezone=%s resolution=%s",
        latitude, longitude, variables, forecast_hours, past_hours,
        timezone, resolution,
    )

    response = httpx.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    if data.get("error"):
        raise RuntimeError(
            f"Open-Meteo API error: {data.get('reason', 'unknown error')}"
        )

    # Parse the response
    resolution_key = "minutely_15" if resolution == "minutely_15" else "hourly"
    payload = data.get(resolution_key)
    if not payload:
        raise RuntimeError(
            f"Open-Meteo response missing '{resolution_key}' data. "
            f"Response keys: {list(data.keys())}"
        )

    time_values = payload.get("time", [])
    if not time_values:
        raise RuntimeError("Open-Meteo returned an empty time array.")

    df = pd.DataFrame({datetime_col: pd.to_datetime(time_values)})

    # Strip timezone info — keep naive datetimes consistent with ForecastingDataLoader
    if df[datetime_col].dt.tz is not None:
        df[datetime_col] = df[datetime_col].dt.tz_localize(None)

    # Populate weather variable columns
    missing_vars: list[str] = []
    for var in variables:
        values = payload.get(var)
        if values is None:
            missing_vars.append(var)
            logger.warning("Variable '%s' missing from Open-Meteo response.", var)
            continue
        col_name = _RENAME_MAP.get(var, var) if rename_columns else var
        df[col_name] = values

    if missing_vars:
        logger.warning(
            "The following requested variables were absent from the "
            "Open-Meteo response: %s", missing_vars
        )

    df = df.sort_values(datetime_col).reset_index(drop=True)

    logger.info(
        "Open-Meteo fetch complete: %d rows from %s to %s (timezone=%s)",
        len(df),
        df[datetime_col].iloc[0],
        df[datetime_col].iloc[-1],
        timezone,
    )

    return df


# ---------------------------------------------------------------------------
# Convenience: validate variable names against Open-Meteo known variables
# ---------------------------------------------------------------------------

#: Complete set of supported hourly variable names (non-exhaustive but covers
#: everything relevant to building load forecasting).
KNOWN_HOURLY_VARIABLES: frozenset[str] = frozenset(
    list(_RENAME_MAP.keys())
    + [
        "precipitation_probability",
        "rain",
        "showers",
        "snowfall",
        "snow_depth",
        "weather_code",
        "pressure_msl",
        "evapotranspiration",
        "et0_fao_evapotranspiration",
        "vapour_pressure_deficit",
        "cape",
        "visibility",
        "wind_gusts_10m",
        "soil_temperature_0cm",
        "soil_temperature_6cm",
        "uv_index",
        "sunshine_duration",
        "wet_bulb_temperature_2m",
        "freezing_level_height",
    ]
)


def validate_variables(variables: list[str]) -> list[str]:
    """
    Return a list of unrecognized variable names.

    An empty list means all variables are valid.  Unrecognized names are
    not blocked — they may be valid Open-Meteo variables not listed here —
    but the caller can surface a warning to the user.
    """
    return [v for v in variables if v not in KNOWN_HOURLY_VARIABLES]
