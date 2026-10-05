"""
Tests for the Open-Meteo weather fetcher (core module + MCP tool).

All tests are fully offline: httpx.get is monkeypatched so no real network
calls are made.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Make the package importable without an editable install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.weather_fetcher import (
    VARIABLE_PRESETS,
    fetch_weather_forecast_df,
    validate_variables,
    KNOWN_HOURLY_VARIABLES,
    _RENAME_MAP,
    OPEN_METEO_FORECAST_URL,
)
from load_forecasting.tools import fetch_weather_forecast


# ---------------------------------------------------------------------------
# Helpers — build a fake Open-Meteo JSON response
# ---------------------------------------------------------------------------

def _make_open_meteo_response(
    variables: list[str],
    n_hours: int = 4,
    resolution: str = "hourly",
) -> dict:
    """Build a minimal Open-Meteo-style JSON response dict."""
    import datetime

    base = datetime.datetime(2025, 1, 1, 0, 0)
    if resolution == "minutely_15":
        times = [
            (base + datetime.timedelta(minutes=15 * i)).isoformat()
            for i in range(n_hours)
        ]
        key = "minutely_15"
    else:
        times = [
            (base + datetime.timedelta(hours=i)).isoformat()
            for i in range(n_hours)
        ]
        key = "hourly"

    payload = {"time": times}
    for v in variables:
        payload[v] = [float(i) for i in range(n_hours)]

    return {
        "latitude": 37.77,
        "longitude": -122.41,
        "timezone": "America/Los_Angeles",
        "utc_offset_seconds": -25200,
        key: payload,
        f"{key}_units": {v: "°C" for v in variables},
    }


def _mock_httpx_get(response_dict: dict):
    """Return a mock that makes httpx.get(...).json() return response_dict."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = response_dict
    mock_resp.raise_for_status.return_value = None
    mock_get = MagicMock(return_value=mock_resp)
    return mock_get


# ===========================================================================
# weather_fetcher.py — core module tests
# ===========================================================================

class TestVariablePresets:
    def test_presets_keys_exist(self):
        assert "minimal" in VARIABLE_PRESETS
        assert "building_load" in VARIABLE_PRESETS
        assert "extended" in VARIABLE_PRESETS

    def test_minimal_subset_of_building_load(self):
        for v in VARIABLE_PRESETS["minimal"]:
            assert v in VARIABLE_PRESETS["building_load"]

    def test_building_load_subset_of_extended(self):
        for v in VARIABLE_PRESETS["building_load"]:
            assert v in VARIABLE_PRESETS["extended"]

    def test_presets_non_empty(self):
        for name, vlist in VARIABLE_PRESETS.items():
            assert len(vlist) >= 2, f"Preset '{name}' is too short"


class TestValidateVariables:
    def test_all_known_returns_empty(self):
        assert validate_variables(list(VARIABLE_PRESETS["building_load"])) == []

    def test_unknown_variable_flagged(self):
        unknown = validate_variables(["temperature_2m", "nonexistent_var_xyz"])
        assert "nonexistent_var_xyz" in unknown
        assert "temperature_2m" not in unknown

    def test_empty_list(self):
        assert validate_variables([]) == []


class TestFetchWeatherForecastDf:
    """Tests for the core fetch_weather_forecast_df function."""

    def test_basic_fetch_returns_dataframe(self):
        variables = ["temperature_2m", "relative_humidity_2m"]
        fake = _make_open_meteo_response(variables, n_hours=6)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            df = fetch_weather_forecast_df(
                latitude=37.77,
                longitude=-122.41,
                variables=variables,
                forecast_hours=6,
                past_hours=0,
                timezone="UTC",
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6
        assert "datetime" in df.columns

    def test_rename_columns_applied(self):
        variables = ["temperature_2m", "relative_humidity_2m"]
        fake = _make_open_meteo_response(variables, n_hours=3)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            df = fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
                rename_columns=True,
            )
        # Renamed columns should be present
        assert "temperature" in df.columns
        assert "relative_humidity" in df.columns
        # Original names should NOT be present
        assert "temperature_2m" not in df.columns

    def test_rename_columns_disabled(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=3)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            df = fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
                rename_columns=False,
            )
        assert "temperature_2m" in df.columns
        assert "temperature" not in df.columns

    def test_custom_datetime_col(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=2)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            df = fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
                datetime_col="timestamp",
            )
        assert "timestamp" in df.columns
        assert "datetime" not in df.columns

    def test_minutely_15_resolution(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=8, resolution="minutely_15")
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            df = fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
                resolution="minutely_15",
            )
        assert len(df) == 8

    def test_api_error_raises_runtime_error(self):
        error_response = {"error": True, "reason": "Invalid coordinates"}
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=error_response),
                raise_for_status=MagicMock(return_value=None),
            )
            with pytest.raises(RuntimeError, match="Invalid coordinates"):
                fetch_weather_forecast_df(
                    latitude=999.0, longitude=0.0,
                    variables=["temperature_2m"],
                )

    def test_empty_variables_raises_value_error(self):
        with pytest.raises(ValueError, match="At least one"):
            fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0, variables=[]
            )

    def test_invalid_forecast_hours_raises_value_error(self):
        with pytest.raises(ValueError, match="forecast_hours"):
            fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=["temperature_2m"],
                forecast_hours=999,
            )

    def test_invalid_resolution_raises_value_error(self):
        with pytest.raises(ValueError, match="resolution"):
            fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=["temperature_2m"],
                resolution="daily",
            )

    def test_datetimes_are_naive(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=3)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            df = fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
            )
        # No timezone info — naive timestamps
        assert df["datetime"].dt.tz is None

    def test_output_sorted_ascending(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=5)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            df = fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
            )
        assert df["datetime"].is_monotonic_increasing

    def test_correct_api_url_called(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=2)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            fetch_weather_forecast_df(
                latitude=37.77, longitude=-122.41,
                variables=variables,
            )
        call_args = mock_httpx.get.call_args
        assert call_args[0][0] == OPEN_METEO_FORECAST_URL

    def test_past_hours_included_in_params(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=2)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
                past_hours=24,
            )
        call_params = mock_httpx.get.call_args[1]["params"]
        assert call_params.get("past_hours") == 24

    def test_past_hours_zero_not_included(self):
        variables = ["temperature_2m"]
        fake = _make_open_meteo_response(variables, n_hours=2)
        with patch("load_forecasting.core.weather_fetcher.httpx") as mock_httpx:
            mock_httpx.get.return_value = MagicMock(
                json=MagicMock(return_value=fake),
                raise_for_status=MagicMock(return_value=None),
            )
            fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=variables,
                past_hours=0,
            )
        call_params = mock_httpx.get.call_args[1]["params"]
        assert "past_hours" not in call_params

    def test_import_error_if_httpx_missing(self, monkeypatch):
        # Temporarily hide httpx from the module
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("No module named 'httpx'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        # Also patch the module-level import inside weather_fetcher
        import load_forecasting.core.weather_fetcher as wf
        monkeypatch.setattr(wf, "httpx", None, raising=False)

        # Re-calling fetch_weather_forecast_df should raise ImportError
        # because it does `import httpx` inside the function
        with pytest.raises((ImportError, AttributeError)):
            fetch_weather_forecast_df(
                latitude=0.0, longitude=0.0,
                variables=["temperature_2m"],
            )


# ===========================================================================
# fetch_weather_forecast — MCP tool tests
# ===========================================================================

class TestFetchWeatherForecastTool:
    """Tests for the MCP tool wrapper in data_prep.py."""

    def _fake_fetch_df(self, variables, n_rows=4):
        """Return a small DataFrame as if from Open-Meteo."""
        import datetime
        rows = []
        for i in range(n_rows):
            ts = datetime.datetime(2025, 1, 1) + datetime.timedelta(hours=i)
            row = {"datetime": ts}
            for v in variables:
                row[v] = float(i)
            rows.append(row)
        return pd.DataFrame(rows)

    def test_success_default_preset(self):
        preset_vars = VARIABLE_PRESETS["building_load"]
        df = self._fake_fetch_df(
            [_RENAME_MAP.get(v, v) for v in preset_vars], n_rows=4
        )
        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            return_value=df,
        ):
            result = fetch_weather_forecast(
                latitude=37.77,
                longitude=-122.41,
            )
        assert result["success"] is True
        assert result["n_rows"] == 4
        assert "weather_data" in result
        assert len(result["weather_data"]) == 4

    def test_success_explicit_variables(self):
        vars_requested = ["temperature_2m", "relative_humidity_2m"]
        renamed = ["temperature", "relative_humidity"]
        df = self._fake_fetch_df(renamed, n_rows=3)
        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            return_value=df,
        ):
            result = fetch_weather_forecast(
                latitude=0.0,
                longitude=0.0,
                variables=vars_requested,
            )
        assert result["success"] is True
        assert "temperature" in result["variables_fetched"]

    def test_invalid_preset_returns_error(self):
        result = fetch_weather_forecast(
            latitude=0.0, longitude=0.0, preset="bogus_preset"
        )
        assert result["success"] is False
        assert "preset" in result["error"].lower() or "Unknown" in result["error"]

    def test_out_of_range_latitude_returns_error(self):
        result = fetch_weather_forecast(latitude=999.0, longitude=0.0)
        assert result["success"] is False
        assert "latitude" in result["error"]

    def test_out_of_range_longitude_returns_error(self):
        result = fetch_weather_forecast(latitude=0.0, longitude=999.0)
        assert result["success"] is False
        assert "longitude" in result["error"]

    def test_invalid_forecast_hours_returns_error(self):
        result = fetch_weather_forecast(
            latitude=0.0, longitude=0.0, forecast_hours=500
        )
        assert result["success"] is False
        assert "forecast_hours" in result["error"]

    def test_invalid_past_hours_returns_error(self):
        result = fetch_weather_forecast(
            latitude=0.0, longitude=0.0, past_hours=200
        )
        assert result["success"] is False
        assert "past_hours" in result["error"]

    def test_invalid_resolution_returns_error(self):
        result = fetch_weather_forecast(
            latitude=0.0, longitude=0.0, resolution="daily"
        )
        assert result["success"] is False
        assert "resolution" in result["error"]

    def test_api_runtime_error_returns_error(self):
        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            side_effect=RuntimeError("API rate limit"),
        ):
            result = fetch_weather_forecast(latitude=0.0, longitude=0.0)
        assert result["success"] is False
        assert "API rate limit" in result["error"]

    def test_output_csv_written(self, tmp_path):
        preset_vars = VARIABLE_PRESETS["minimal"]
        df = self._fake_fetch_df(
            [_RENAME_MAP.get(v, v) for v in preset_vars], n_rows=3
        )
        out = str(tmp_path / "weather.csv")
        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            return_value=df,
        ):
            result = fetch_weather_forecast(
                latitude=0.0, longitude=0.0,
                output_csv_path=out,
            )
        assert result["success"] is True
        assert result.get("output_csv_path") == out
        written = pd.read_csv(out)
        assert len(written) == 3

    def test_minutely_15_resolution_passed_through(self):
        df = self._fake_fetch_df(["temperature"], n_rows=4)
        captured_kwargs = {}

        def fake_fetch_df(**kwargs):
            captured_kwargs.update(kwargs)
            return df

        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            side_effect=fake_fetch_df,
        ):
            fetch_weather_forecast(
                latitude=0.0, longitude=0.0,
                resolution="minutely_15",
                variables=["temperature_2m"],
            )
        assert captured_kwargs.get("resolution") == "minutely_15"

    def test_forecast_start_end_in_response(self):
        df = self._fake_fetch_df(["temperature"], n_rows=5)
        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            return_value=df,
        ):
            result = fetch_weather_forecast(latitude=0.0, longitude=0.0)
        assert "forecast_start" in result
        assert "forecast_end" in result
        assert result["forecast_start"] is not None
        assert result["forecast_end"] is not None

    def test_preset_takes_lower_priority_than_variables(self):
        captured_kwargs = {}
        df = self._fake_fetch_df(["temperature"], n_rows=2)

        def fake_fetch_df(**kwargs):
            captured_kwargs.update(kwargs)
            return df

        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            side_effect=fake_fetch_df,
        ):
            fetch_weather_forecast(
                latitude=0.0, longitude=0.0,
                variables=["temperature_2m"],
                preset="extended",  # should be ignored
            )
        assert captured_kwargs["variables"] == ["temperature_2m"]

    def test_metadata_echoed_in_response(self):
        df = self._fake_fetch_df(["temperature"], n_rows=2)
        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            return_value=df,
        ):
            result = fetch_weather_forecast(
                latitude=51.5, longitude=-0.12,
                timezone="Europe/London",
                resolution="hourly",
            )
        assert result["latitude"] == 51.5
        assert result["longitude"] == -0.12
        assert result["timezone"] == "Europe/London"
        assert result["resolution"] == "hourly"

    def test_unknown_variable_warning_surfaced(self):
        df = self._fake_fetch_df(["temperature"], n_rows=2)
        with patch(
            "load_forecasting.tools.data_prep.fetch_weather_forecast_df",
            return_value=df,
        ):
            result = fetch_weather_forecast(
                latitude=0.0, longitude=0.0,
                variables=["temperature_2m", "totally_fake_variable_xyz"],
            )
        assert result["success"] is True
        assert "warnings" in result
        assert any("totally_fake_variable_xyz" in w for w in result["warnings"])
