"""Tests for the pre-flight guardrail helpers in tools/_common.py."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.tools._common import _check_covariate_mismatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _loader(mapping: dict, columns: list):
    """Minimal loader stand-in exposing the two attributes the check reads."""
    return SimpleNamespace(
        column_mapping=mapping,
        df=pd.DataFrame(columns=columns),
    )


def _metadata(past=None, future=None):
    mapping = {"datetime": "Datetime", "target": "load_kwh"}
    if past is not None:
        mapping["past_covariates"] = past
    if future is not None:
        mapping["future_covariates"] = future
    return {"column_mapping": mapping}


# ---------------------------------------------------------------------------
# Matching cases — no error
# ---------------------------------------------------------------------------

class TestCovariateMatch:
    def test_identical_covariates_pass(self):
        meta = _metadata(past=["T_out", "hour_sin", "lag_24h"])
        loader = _loader(
            {"past_covariates": ["T_out", "hour_sin", "lag_24h"]},
            ["load_kwh", "T_out", "hour_sin", "lag_24h"],
        )
        assert _check_covariate_mismatch(loader, meta) is None

    def test_order_does_not_matter(self):
        meta = _metadata(past=["T_out", "RH_out"])
        loader = _loader(
            {"past_covariates": ["RH_out", "T_out"]},
            ["load_kwh", "RH_out", "T_out"],
        )
        assert _check_covariate_mismatch(loader, meta) is None

    def test_model_without_covariates_passes(self):
        loader = _loader({}, ["load_kwh"])
        assert _check_covariate_mismatch(loader, _metadata()) is None

    def test_empty_metadata_passes(self):
        loader = _loader({"past_covariates": ["T_out"]}, ["load_kwh", "T_out"])
        assert _check_covariate_mismatch(loader, {}) is None

    def test_future_covariates_match(self):
        meta = _metadata(past=["T_out"], future=["T_out_forecast"])
        loader = _loader(
            {"past_covariates": ["T_out"], "future_covariates": ["T_out_forecast"]},
            ["load_kwh", "T_out", "T_out_forecast"],
        )
        assert _check_covariate_mismatch(loader, meta) is None


# ---------------------------------------------------------------------------
# Mismatching cases — descriptive error
# ---------------------------------------------------------------------------

class TestCovariateMismatch:
    def test_missing_past_covariates_named_in_error(self):
        """The regression case: model trained with weather, CSV has none."""
        meta = _metadata(past=["T_out", "RH_out", "Direct_Radiation", "lag_24h"])
        loader = _loader({"past_covariates": ["lag_24h"]}, ["load_kwh", "lag_24h"])

        err = _check_covariate_mismatch(loader, meta)

        assert err is not None
        assert "past covariate" in err
        for col in ("T_out", "RH_out", "Direct_Radiation"):
            assert col in err
        assert "lag_24h" not in err.split("missing from the CSV:")[1]

    def test_mapped_but_absent_column_counts_as_missing(self):
        """to_darts_series drops mapped columns absent from the DataFrame."""
        meta = _metadata(past=["T_out"])
        loader = _loader({"past_covariates": ["T_out"]}, ["load_kwh"])

        err = _check_covariate_mismatch(loader, meta)

        assert err is not None
        assert "T_out" in err

    def test_extra_covariate_reported(self):
        meta = _metadata(past=["T_out"])
        loader = _loader(
            {"past_covariates": ["T_out", "RH_out"]},
            ["load_kwh", "T_out", "RH_out"],
        )

        err = _check_covariate_mismatch(loader, meta)

        assert err is not None
        assert "unexpected extra columns" in err
        assert "RH_out" in err

    def test_missing_future_covariates_reported(self):
        meta = _metadata(future=["T_out_forecast"])
        loader = _loader({}, ["load_kwh"])

        err = _check_covariate_mismatch(loader, meta)

        assert err is not None
        assert "future covariate" in err
        assert "T_out_forecast" in err

    def test_error_mentions_remediation(self):
        meta = _metadata(past=["T_out"])
        loader = _loader({}, ["load_kwh"])

        err = _check_covariate_mismatch(loader, meta)

        assert "retrain" in err
        assert "column_mapping" in err
