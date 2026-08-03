"""
Unit tests for the reporting package: payload assembly, JSON encoding,
and self-contained HTML building.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.reporting import (
    build_report_payload,
    build_report_html,
)
from load_forecasting.reporting.payload import (
    payload_to_json,
    _ReportJSONEncoder,
    ReportPayload,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_predictions():
    """Synthetic 96 hourly prediction rows spanning 4 days."""
    ts = pd.date_range("2024-01-01", periods=96, freq="h")
    actual = np.linspace(100, 120, 96) + np.sin(np.linspace(0, 8 * np.pi, 96)) * 5
    predicted = actual + np.random.RandomState(0).normal(0, 1.5, 96)
    return [
        {
            "timestamp": t.isoformat(),
            "actual": float(a),
            "predicted": float(p),
            "residual": float(a - p),
        }
        for t, a, p in zip(ts, actual, predicted)
    ]


@pytest.fixture
def sample_metadata():
    return {
        "model_id": "TestBuilding_LinearRegression_20240101_000000",
        "model_type": "LinearRegression",
        "building_name": "TestBuilding",
        "created_at": "2024-01-01T00:00:00",
        "config": {
            "lookback_hours": 24,
            "horizon_hours": 6,
            "frequency": "h",
            "validation_split": 0.2,
        },
        "column_mapping": {
            "datetime": "timestamp",
            "target": "kwh",
            "past_covariates": ["temp"],
            "future_covariates": [],
        },
        "data_info": {
            "start_date": "2023-01-01T00:00:00",
            "end_date": "2023-12-31T23:00:00",
            "total_samples": 8760,
        },
        "metrics": {
            "validation": {"rmse": 1.2, "mae": 0.9, "mape": 1.0, "cv_rmse": 1.1, "r_squared": 0.98},
        },
    }


@pytest.fixture
def sample_eval_result(sample_predictions):
    return {
        "success": True,
        "error": None,
        "model_id": "TestBuilding_LinearRegression_20240101_000000",
        "model_type": "LinearRegression",
        "test_metrics": {
            "rmse": 1.5,
            "mae": 1.1,
            "mape": 1.2,
            "cv_rmse": 1.4,
            "r_squared": 0.97,
        },
        "comparison_to_validation": {
            "cv_rmse_diff": 0.3,
            "performance_status": "similar",
        },
        "test_summary": {
            "test_samples": 96,
            "start_date": "2024-01-01T00:00:00",
            "end_date": "2024-01-04T23:00:00",
            "column_mapping_source": "from_model_metadata",
        },
        "predictions": sample_predictions,
        "residual_analysis": {
            "mean_residual": -0.05,
            "std_residual": 1.5,
            "autocorrelation_lag1": 0.02,
        },
    }


@pytest.fixture
def sample_input_df():
    idx = pd.date_range("2024-01-01", periods=120, freq="h")
    return pd.DataFrame(
        {
            "kwh": np.linspace(100, 120, 120),
            "temp": np.linspace(50, 60, 120),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------


class TestJSONEncoder:
    def test_encodes_numpy_scalars(self):
        out = json.dumps(
            {"i": np.int64(3), "f": np.float64(1.5), "b": np.bool_(True)},
            cls=_ReportJSONEncoder,
        )
        d = json.loads(out)
        assert d == {"i": 3, "f": 1.5, "b": True}

    def test_encodes_timestamp(self):
        out = json.dumps({"t": pd.Timestamp("2024-01-01")}, cls=_ReportJSONEncoder)
        assert "2024-01-01" in out

    def test_encodes_nan_as_null_via_helper(self):
        # _ReportJSONEncoder maps np.nan -> None, but only via .default;
        # json.dumps still serialises Python float('nan') natively unless
        # allow_nan=False is set.  Our wrapper uses allow_nan=False, so
        # ensure NaN values are scrubbed before they reach the encoder via
        # the _round() helper used inside build_report_payload.
        payload = ReportPayload(
            meta={"x": np.float64(np.nan)},
            metrics={}, residual_analysis={}, series={}, aggregations={},
            input_summary={},
        )
        # numpy NaN goes through .default -> None, so this is safe
        text = payload_to_json(payload)
        assert "null" in text

    def test_escapes_close_script(self):
        payload = ReportPayload(
            meta={"name": "evil </script><script>alert(1)</script>"},
            metrics={}, residual_analysis={}, series={}, aggregations={},
            input_summary={},
        )
        text = payload_to_json(payload)
        assert "</script>" not in text
        assert "<\\/script>" in text


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_minimum_required_fields(self, sample_metadata, sample_eval_result):
        payload = build_report_payload(
            model_id="TestBuilding_LinearRegression_20240101_000000",
            model_metadata=sample_metadata,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=sample_metadata["column_mapping"],
        )
        d = payload.to_dict()
        assert d["meta"]["model_id"] == sample_metadata["model_id"]
        assert d["meta"]["model_type"] == "LinearRegression"
        assert d["metrics"]["test"]["rmse"] == 1.5
        assert d["metrics"]["validation"]["cv_rmse"] == 1.1
        assert d["metrics"]["comparison"]["performance_status"] == "similar"
        assert len(d["series"]["predictions"]) == 96
        # Aggregations precomputed
        assert len(d["aggregations"]["hour_of_day_mae"]) == 24
        assert len(d["aggregations"]["day_of_week_mae"]) == 7
        # No input df → empty preview
        assert d["series"]["input_target"] == []

    def test_with_input_df_populates_preview(
        self, sample_metadata, sample_eval_result, sample_input_df
    ):
        payload = build_report_payload(
            model_id="x",
            model_metadata=sample_metadata,
            eval_result=sample_eval_result,
            input_df=sample_input_df,
            column_mapping=sample_metadata["column_mapping"],
        )
        d = payload.to_dict()
        assert len(d["series"]["input_target"]) == 120
        assert "temp" in d["series"]["covariates"]
        assert d["input_summary"]["target"]["column"] == "kwh"
        assert d["input_summary"]["covariates"]["temp"]["n"] == 120

    def test_title_override(self, sample_metadata, sample_eval_result):
        payload = build_report_payload(
            model_id="x",
            model_metadata=sample_metadata,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=sample_metadata["column_mapping"],
            title="My Custom Report",
        )
        assert payload.meta["title"] == "My Custom Report"

    def test_failed_evaluation_raises(self, sample_metadata):
        with pytest.raises(ValueError, match="failed evaluation"):
            build_report_payload(
                model_id="x",
                model_metadata=sample_metadata,
                eval_result={"success": False, "error": "bad data"},
                input_df=None,
                column_mapping={},
            )

    def test_missing_predictions_raises(self, sample_metadata):
        with pytest.raises(ValueError, match="no predictions"):
            build_report_payload(
                model_id="x",
                model_metadata=sample_metadata,
                eval_result={"success": True, "predictions": []},
                input_df=None,
                column_mapping={},
            )


# ---------------------------------------------------------------------------
# Training curve (learning curve) — surfaced in meta.training_history
# ---------------------------------------------------------------------------


class TestTrainingCurve:
    def test_absent_when_no_training_info(
        self, sample_metadata, sample_eval_result
    ):
        # LinearRegression metadata has no training_info at all.
        payload = build_report_payload(
            model_id="x",
            model_metadata=sample_metadata,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=sample_metadata["column_mapping"],
        )
        assert payload.meta["training_history"] == {}

    def test_absent_when_history_empty(
        self, sample_metadata, sample_eval_result
    ):
        # Non-Torch models carry training_info but no training_history key.
        md = dict(sample_metadata)
        md["training_info"] = {"energy_kwh": 0.01, "training_time_seconds": 3.2}
        payload = build_report_payload(
            model_id="x",
            model_metadata=md,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=md["column_mapping"],
        )
        assert payload.meta["training_history"] == {}

    def test_populated_with_train_and_val_loss(
        self, sample_metadata, sample_eval_result
    ):
        md = dict(sample_metadata)
        md["model_type"] = "TiDE"
        md["training_info"] = {
            "training_history": {
                "epochs": [0, 1, 2],
                "train_loss": [0.9, 0.5, 0.3],
                "val_loss": [1.0, 0.6, 0.45],
            }
        }
        payload = build_report_payload(
            model_id="x",
            model_metadata=md,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=md["column_mapping"],
        )
        th = payload.meta["training_history"]
        assert th["n_epochs"] == 3
        assert th["has_val"] is True
        assert th["points"][0] == {"epoch": 0, "train_loss": 0.9, "val_loss": 1.0}
        assert th["points"][-1]["train_loss"] == 0.3

    def test_populated_without_val_loss(
        self, sample_metadata, sample_eval_result
    ):
        md = dict(sample_metadata)
        md["model_type"] = "TiDE"
        md["training_info"] = {
            "training_history": {
                "epochs": [0, 1],
                "train_loss": [0.9, 0.5],
                "val_loss": [None, None],
            }
        }
        payload = build_report_payload(
            model_id="x",
            model_metadata=md,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=md["column_mapping"],
        )
        th = payload.meta["training_history"]
        assert th["has_val"] is False
        assert all(p["val_loss"] is None for p in th["points"])

    def test_html_contains_training_curve_section(
        self, sample_metadata, sample_eval_result
    ):
        md = dict(sample_metadata)
        md["model_type"] = "TiDE"
        md["training_info"] = {
            "training_history": {
                "epochs": [0, 1, 2],
                "train_loss": [0.9, 0.5, 0.3],
                "val_loss": [1.0, 0.6, 0.45],
            }
        }
        payload = build_report_payload(
            model_id="x",
            model_metadata=md,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=md["column_mapping"],
        )
        html = build_report_html(payload)
        assert 'id="training-curve"' in html
        assert 'id="chart-training-curve"' in html
        # The training_history data is embedded in the payload JSON.
        assert "training_history" in html


# ---------------------------------------------------------------------------
# HTML builder — verifies self-containment
# ---------------------------------------------------------------------------


class TestBuildHTML:
    def test_html_is_self_contained(
        self, sample_metadata, sample_eval_result, sample_input_df
    ):
        payload = build_report_payload(
            model_id="x",
            model_metadata=sample_metadata,
            eval_result=sample_eval_result,
            input_df=sample_input_df,
            column_mapping=sample_metadata["column_mapping"],
        )
        html = build_report_html(payload)

        # Must be a full HTML doc
        assert html.startswith("<!doctype html>") or html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

        # Must NOT contain any external script references
        assert 'script src=' not in html.lower()
        assert 'href="http' not in html.lower()
        assert 'src="http' not in html.lower()
        assert "cdn." not in html.lower()

        # Embedded libraries present (look for their banner comments)
        assert "d3js.org" in html, "d3 library not embedded"
        assert "observablehq/plot" in html, "Observable Plot not embedded"

        # Data marker present and parseable
        marker = "window.__REPORT_DATA__"
        assert marker in html

        # File size should be substantial (libs alone are ~480 KB)
        assert len(html) > 400_000

    def test_writes_to_disk_and_reopens(
        self, tmp_path, sample_metadata, sample_eval_result
    ):
        payload = build_report_payload(
            model_id="x",
            model_metadata=sample_metadata,
            eval_result=sample_eval_result,
            input_df=None,
            column_mapping=sample_metadata["column_mapping"],
        )
        html = build_report_html(payload)
        out = tmp_path / "report.html"
        out.write_text(html, encoding="utf-8")
        assert out.exists()
        round_trip = out.read_text(encoding="utf-8")
        assert "window.__REPORT_DATA__" in round_trip
