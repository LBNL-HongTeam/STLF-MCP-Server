"""
MCP tools for data preparation and registry inspection:
  - list_models: list trained models in the registry
  - merge_covariates: join a load CSV with a covariate (e.g. weather) CSV
  - fetch_weather_forecast: fetch live weather from Open-Meteo
"""

from pathlib import Path
from typing import Optional
import logging

import pandas as pd

from ..core.model_registry import ModelRegistry
from ..core.paths import resolve_data_path, not_found_hint
from ..core.weather_fetcher import (
    fetch_weather_forecast_df,
    VARIABLE_PRESETS as _WEATHER_PRESETS,
    validate_variables as _validate_weather_variables,
)
from ._common import (
    create_success_response,
    create_error_response,
    _df_to_records,
)

logger = logging.getLogger(__name__)


def list_models(
    building_name: Optional[str] = None,
    model_type: Optional[str] = None,
    sort_by: str = "created_at",
    limit: int = 20,
) -> dict:
    """
    List trained models in the registry.

    Args:
        building_name: Filter by building
        model_type: Filter by model type
        sort_by: Sort field
        limit: Maximum results

    Returns:
        Dict with models list and total count
    """
    try:
        registry = ModelRegistry()
        models, total_count = registry.list_models(
            building_name=building_name,
            model_type=model_type,
            sort_by=sort_by,
            limit=limit,
        )

        return create_success_response(
            models=models,
            total_count=total_count,
            filters_applied={
                "building_name": building_name,
                "model_type": model_type,
                "sort_by": sort_by,
                "limit": limit,
            },
        )

    except Exception as e:
        logger.exception("List models failed")
        return create_error_response(f"Failed to list models: {str(e)}")


def merge_covariates(
    load_csv_path: str,
    covariate_csv_path: str,
    output_csv_path: str,
    load_datetime_col: Optional[str] = None,
    covariate_datetime_col: Optional[str] = None,
    covariate_columns: Optional[list] = None,
    covariate_timezone: Optional[str] = None,
    load_timezone: Optional[str] = None,
) -> dict:
    """
    Merge a load CSV with a covariate CSV (e.g. weather data), handling
    timezone conversion and DST edge-cases automatically.

    The merge is a LEFT JOIN on the datetime column — all load rows are
    preserved. Any covariate gaps produced by DST spring-forward (missing
    local hour) are forward-filled. DST fall-back duplicates in the
    covariate file (repeated local hour) are resolved by keeping the first
    occurrence.

    Args:
        load_csv_path: Path to the primary load CSV (local-time timestamps).
        covariate_csv_path: Path to the covariate CSV (e.g. weather).
        output_csv_path: Path where the merged CSV will be written.
        load_datetime_col: Datetime column name in the load CSV.
            Auto-detected if not provided.
        covariate_datetime_col: Datetime column name in the covariate CSV.
            Auto-detected if not provided.
        covariate_columns: List of covariate column names to include from
            the covariate CSV. All non-datetime columns are included if not
            provided.
        covariate_timezone: IANA timezone of the covariate file timestamps
            (e.g. "UTC", "America/Los_Angeles"). If provided, timestamps are
            converted to the load file's local time before joining.
            No conversion is applied when None.
        load_timezone: IANA timezone to interpret load timestamps when
            performing the timezone conversion (e.g. "America/Los_Angeles").
            Required when covariate_timezone is set; ignored otherwise.

    Returns:
        Dict with output_csv_path, n_rows, covariate_columns,
        n_missing_filled, start_date, end_date, timezone_conversion,
        and dst_duplicates_dropped.
    """
    try:
        load_path = resolve_data_path(load_csv_path)
        cov_path = resolve_data_path(covariate_csv_path)
        out_path = Path(output_csv_path)

        if not load_path.exists():
            return create_error_response(
                f"Load CSV not found: {load_csv_path}. " + not_found_hint(load_csv_path)
            )
        if not cov_path.exists():
            return create_error_response(
                f"Covariate CSV not found: {covariate_csv_path}. " + not_found_hint(covariate_csv_path)
            )

        # ------------------------------------------------------------------ #
        # 1. Read load file — detect datetime column
        # ------------------------------------------------------------------ #
        load_df = pd.read_csv(load_path)

        if load_datetime_col is None:
            # Auto-detect: first column whose name hints at datetime
            _dt_hints = {"datetime", "date", "timestamp", "time"}
            candidates = [
                c for c in load_df.columns
                if any(h in c.lower() for h in _dt_hints)
            ]
            if not candidates:
                candidates = [load_df.columns[0]]
            load_datetime_col = candidates[0]

        if load_datetime_col not in load_df.columns:
            return create_error_response(
                f"Datetime column '{load_datetime_col}' not found in load CSV. "
                f"Available columns: {list(load_df.columns)}"
            )

        load_df[load_datetime_col] = pd.to_datetime(load_df[load_datetime_col])

        # ------------------------------------------------------------------ #
        # 2. Read covariate file — detect datetime column
        # ------------------------------------------------------------------ #
        cov_df = pd.read_csv(cov_path)

        if covariate_datetime_col is None:
            _dt_hints = {"datetime", "date", "timestamp", "time"}
            candidates = [
                c for c in cov_df.columns
                if any(h in c.lower() for h in _dt_hints)
            ]
            if not candidates:
                candidates = [cov_df.columns[0]]
            covariate_datetime_col = candidates[0]

        if covariate_datetime_col not in cov_df.columns:
            return create_error_response(
                f"Datetime column '{covariate_datetime_col}' not found in covariate CSV. "
                f"Available columns: {list(cov_df.columns)}"
            )

        cov_df[covariate_datetime_col] = pd.to_datetime(cov_df[covariate_datetime_col])

        # ------------------------------------------------------------------ #
        # 3. Select covariate columns
        # ------------------------------------------------------------------ #
        all_cov_cols = [c for c in cov_df.columns if c != covariate_datetime_col]

        if covariate_columns is not None:
            missing_cols = [c for c in covariate_columns if c not in cov_df.columns]
            if missing_cols:
                return create_error_response(
                    f"Requested covariate columns not found in covariate CSV: {missing_cols}. "
                    f"Available: {all_cov_cols}"
                )
            selected_cov_cols = list(covariate_columns)
        else:
            selected_cov_cols = all_cov_cols

        # ------------------------------------------------------------------ #
        # 4. Timezone conversion (optional)
        # ------------------------------------------------------------------ #
        dst_duplicates_dropped = 0
        timezone_conversion = None

        if covariate_timezone is not None:
            if load_timezone is None:
                return create_error_response(
                    "load_timezone is required when covariate_timezone is provided. "
                    "Example: load_timezone='America/Los_Angeles'"
                )

            try:
                # Localize covariate timestamps to their source timezone, then
                # convert to the load file's local timezone, then strip tzinfo
                # so the join key is naive (matching the load file).
                local_col = "_date_local"
                cov_df[local_col] = (
                    cov_df[covariate_datetime_col]
                    .dt.tz_localize(covariate_timezone)
                    .dt.tz_convert(load_timezone)
                    .dt.tz_localize(None)
                )

                # Fall-back: drop DST duplicate local timestamps (keep first)
                before = len(cov_df)
                cov_df = cov_df.drop_duplicates(subset=local_col, keep="first")
                dst_duplicates_dropped = before - len(cov_df)

                cov_join_col = local_col
                timezone_conversion = f"{covariate_timezone} → {load_timezone}"

            except Exception as tz_err:
                return create_error_response(
                    f"Timezone conversion failed: {tz_err}. "
                    f"Ensure covariate_timezone and load_timezone are valid IANA timezone names "
                    f"(e.g. 'UTC', 'America/Los_Angeles')."
                )
        else:
            cov_join_col = covariate_datetime_col

        # ------------------------------------------------------------------ #
        # 5. Left join: load is the anchor, covariate fills in columns
        # ------------------------------------------------------------------ #
        cov_subset = cov_df[[cov_join_col] + selected_cov_cols].copy()

        merged = load_df.merge(
            cov_subset,
            left_on=load_datetime_col,
            right_on=cov_join_col,
            how="left",
        )

        # Drop the covariate join key column if it differs from load datetime
        if cov_join_col != load_datetime_col and cov_join_col in merged.columns:
            merged = merged.drop(columns=[cov_join_col])

        # ------------------------------------------------------------------ #
        # 6. Forward-fill covariate gaps (DST spring-forward missing hour,
        #    or end-of-file weather cutoff)
        # ------------------------------------------------------------------ #
        n_missing_before = int(merged[selected_cov_cols].isna().any(axis=1).sum())

        if n_missing_before > 0:
            logger.info(
                "merge_covariates: forward-filling %d row(s) with missing covariate values",
                n_missing_before,
            )
            merged[selected_cov_cols] = merged[selected_cov_cols].ffill()

        n_still_missing = int(merged[selected_cov_cols].isna().any(axis=1).sum())
        if n_still_missing > 0:
            # Back-fill as last resort (e.g. missing at the very start)
            logger.warning(
                "merge_covariates: %d row(s) still missing after ffill — applying bfill",
                n_still_missing,
            )
            merged[selected_cov_cols] = merged[selected_cov_cols].bfill()

        # ------------------------------------------------------------------ #
        # 7. Write output
        # ------------------------------------------------------------------ #
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False)

        start_date = str(load_df[load_datetime_col].min())
        end_date = str(load_df[load_datetime_col].max())

        logger.info(
            "merge_covariates: wrote %d rows to %s (covariates: %s)",
            len(merged),
            out_path,
            selected_cov_cols,
        )

        return create_success_response(
            output_csv_path=str(out_path),
            n_rows=len(merged),
            covariate_columns=selected_cov_cols,
            n_missing_filled=n_missing_before,
            start_date=start_date,
            end_date=end_date,
            timezone_conversion=timezone_conversion,
            dst_duplicates_dropped=dst_duplicates_dropped,
        )

    except Exception as e:
        logger.exception("merge_covariates failed")
        return create_error_response(f"Covariate merge failed: {str(e)}")


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_hours: int = 72,
    past_hours: int = 0,
    timezone: str = "UTC",
    variables: Optional[list] = None,
    preset: Optional[str] = None,
    resolution: str = "hourly",
    output_csv_path: Optional[str] = None,
    rename_columns: bool = True,
    datetime_col: str = "datetime",
) -> dict:
    """
    Fetch live weather forecast data from Open-Meteo for a given location.

    Calls the Open-Meteo free API (https://api.open-meteo.com/v1/forecast) —
    no API key required for non-commercial use — and returns a weather
    forecast DataFrame covering the requested time window.  The output is
    designed to slot directly into ``generate_forecast`` as the
    future-covariate rows in the context CSV.

    Typical workflow for live inference with weather-aware models:

    1. Call ``fetch_weather_forecast`` to get weather data covering the
       model's context window (past_hours) plus the forecast horizon
       (forecast_hours).
    2. Merge the returned weather data with your recent load readings using
       ``merge_covariates``.
    3. Pass the merged CSV to ``generate_forecast``.

    Args:
        latitude: WGS84 latitude of the building or site (-90 to 90).
        longitude: WGS84 longitude of the building or site (-180 to 180).
        forecast_hours: Number of future hours to fetch (1–384).  For
            typical day-ahead inference use 24–48; for week-ahead use up
            to 168.  Open-Meteo supports up to 16 days (384h).
        past_hours: Number of archive hours before *now* to include (0–92).
            Set this to at least ``lookback_hours`` of your model so the
            merged CSV also covers the context window.
        timezone: IANA timezone string for the returned timestamps (e.g.
            ``"America/Los_Angeles"`` or ``"UTC"``).  Use the same timezone
            as your load data for seamless merging.
        variables: Explicit list of Open-Meteo variable names to fetch (e.g.
            ``["temperature_2m", "relative_humidity_2m", "shortwave_radiation"]``).
            Takes priority over ``preset`` if both are provided.  If neither
            is provided, defaults to the ``"building_load"`` preset.
        preset: Named variable preset — one of ``"minimal"``,
            ``"building_load"`` (default), or ``"extended"``.  Ignored when
            ``variables`` is given.
        resolution: Temporal resolution of the returned data.  ``"hourly"``
            (default) returns one row per hour.  ``"minutely_15"`` returns
            one row per 15 minutes and is natively available in Central
            Europe and North America (elsewhere it is interpolated from
            hourly data by Open-Meteo).
        output_csv_path: Optional path to write the weather data as a CSV.
            The file is written before the tool returns its response dict.
        rename_columns: If True (default), renames Open-Meteo variable names
            to shorter aliases that match ForecastingDataLoader
            auto-detection patterns (e.g. ``temperature_2m`` → ``temperature``,
            ``shortwave_radiation`` → ``solar_radiation``).  Set False to keep
            the original Open-Meteo column names.
        datetime_col: Column name for timestamps in the returned data
            (default: ``"datetime"``).  Use the same name as in your load CSV
            to simplify ``merge_covariates``.

    Returns:
        Dict with:
          - ``success`` (bool)
          - ``latitude``, ``longitude``: echoed from request
          - ``timezone``: echoed from request
          - ``resolution``: ``"hourly"`` or ``"minutely_15"``
          - ``n_rows``: number of rows returned
          - ``forecast_start``, ``forecast_end``: ISO8601 strings
          - ``variables_fetched``: list of column names in the data
          - ``weather_data``: list of dicts (one per row) with ``datetime_col``
            and one key per weather variable — use this for small requests or
            inspection.  For large requests, prefer ``output_csv_path``.
          - ``output_csv_path``: echoed if a file was written

    Notes:
        - Open-Meteo is free for non-commercial use; see open-meteo.com/en/pricing
          for commercial licensing.
        - The returned timestamps are **naive** (no tzinfo) in the requested
          timezone, matching ForecastingDataLoader's expectation.
        - Use ``merge_covariates`` to join the weather data with your load CSV
          before calling ``generate_forecast``.
    """
    try:
        # ------------------------------------------------------------------ #
        # 1. Resolve variable list
        # ------------------------------------------------------------------ #
        if variables:
            resolved_vars = list(variables)
        elif preset:
            if preset not in _WEATHER_PRESETS:
                return create_error_response(
                    f"Unknown preset '{preset}'. "
                    f"Choose one of: {list(_WEATHER_PRESETS.keys())}"
                )
            resolved_vars = _WEATHER_PRESETS[preset]
        else:
            resolved_vars = _WEATHER_PRESETS["building_load"]

        # Warn about unrecognized variable names (don't block — Open-Meteo
        # accepts more variables than we enumerate locally)
        unknown = _validate_weather_variables(resolved_vars)
        warnings: list[str] = []
        if unknown:
            warnings.append(
                f"Unrecognized variable names (may still be valid Open-Meteo "
                f"variables): {unknown}"
            )

        # ------------------------------------------------------------------ #
        # 2. Validate parameters
        # ------------------------------------------------------------------ #
        if not (-90 <= latitude <= 90):
            return create_error_response(
                f"latitude must be between -90 and 90, got {latitude}"
            )
        if not (-180 <= longitude <= 180):
            return create_error_response(
                f"longitude must be between -180 and 180, got {longitude}"
            )
        if not (1 <= forecast_hours <= 384):
            return create_error_response(
                f"forecast_hours must be between 1 and 384, got {forecast_hours}"
            )
        if not (0 <= past_hours <= 92):
            return create_error_response(
                f"past_hours must be between 0 and 92, got {past_hours}"
            )
        if resolution not in ("hourly", "minutely_15"):
            return create_error_response(
                f"resolution must be 'hourly' or 'minutely_15', got '{resolution}'"
            )

        # ------------------------------------------------------------------ #
        # 3. Fetch from Open-Meteo
        # ------------------------------------------------------------------ #
        try:
            df = fetch_weather_forecast_df(
                latitude=latitude,
                longitude=longitude,
                variables=resolved_vars,
                forecast_hours=forecast_hours,
                past_hours=past_hours,
                timezone=timezone,
                resolution=resolution,
                rename_columns=rename_columns,
                datetime_col=datetime_col,
            )
        except ImportError as e:
            return create_error_response(str(e))
        except RuntimeError as e:
            return create_error_response(f"Open-Meteo API error: {e}")
        except Exception as e:
            return create_error_response(
                f"Failed to fetch weather data from Open-Meteo: {e}"
            )

        # ------------------------------------------------------------------ #
        # 4. Write CSV if requested
        # ------------------------------------------------------------------ #
        written_path = None
        if output_csv_path:
            try:
                df.to_csv(output_csv_path, index=False)
                written_path = output_csv_path
            except Exception as e:
                return create_error_response(
                    f"Failed to write weather CSV to '{output_csv_path}': {e}"
                )

        # ------------------------------------------------------------------ #
        # 5. Build response
        # ------------------------------------------------------------------ #
        variables_fetched = [c for c in df.columns if c != datetime_col]
        n_rows = len(df)

        forecast_start = df[datetime_col].iloc[0].isoformat() if n_rows > 0 else None
        forecast_end = df[datetime_col].iloc[-1].isoformat() if n_rows > 0 else None

        # Convert DataFrame to list-of-dicts; coerce Timestamp → ISO string
        weather_records = _df_to_records(df, datetime_col)

        response_data: dict = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "resolution": resolution,
            "n_rows": n_rows,
            "forecast_start": forecast_start,
            "forecast_end": forecast_end,
            "variables_fetched": variables_fetched,
            "weather_data": weather_records,
        }

        if warnings:
            response_data["warnings"] = warnings

        if written_path:
            response_data["output_csv_path"] = written_path

        return create_success_response(**response_data)

    except Exception as e:
        logger.exception("fetch_weather_forecast failed")
        return create_error_response(f"Weather fetch failed: {str(e)}")
