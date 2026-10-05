"""
Integration tests for the batch multi-series MCP tools:

  * ``batch_train_forecast_models`` — wide-CSV expansion + job-list modes
  * ``batch_generate_forecast`` — fleet-scale forward forecasting
  * fault tolerance (one bad series does not abort the batch)
  * probabilistic bands flow through the batch path

Models are written to a temporary directory via LOAD_FORECASTING_MODEL_DIR.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.tools import (
    batch_train_forecast_models,
    batch_generate_forecast,
)


def _make_wide_csv(n_hours: int = 700) -> str:
    """Wide CSV with three feeder columns + a shared datetime and temperature."""
    t = np.arange(n_hours)
    rng = np.random.default_rng(3)
    data = {"timestamp": pd.date_range("2023-01-01", periods=n_hours, freq="h")}
    for name, base in [("feeder_A", 50), ("feeder_B", 80), ("feeder_C", 30)]:
        data[name] = (
            base
            + 20 * np.sin(2 * np.pi * t / 24)
            + 4 * np.sin(2 * np.pi * t / 168)
            + rng.normal(0, 2, n_hours)
        )
    df = pd.DataFrame(data)
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


@pytest.fixture
def wide_csv():
    path = _make_wide_csv()
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def temp_model_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


class TestBatchTrain:
    def test_wide_csv_trains_one_model_per_column(self, wide_csv, temp_model_dir):
        res = batch_train_forecast_models(
            csv_path=wide_csv,
            target_columns=["feeder_A", "feeder_B", "feeder_C"],
            datetime_col="timestamp",
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
        )
        assert res["success"]
        assert res["summary"]["total"] == 3
        assert res["summary"]["succeeded"] == 3
        assert res["summary"]["failed"] == 0
        ids = {r["series_id"]: r["model_id"] for r in res["results"]}
        assert set(ids) == {"feeder_A", "feeder_B", "feeder_C"}

    def test_job_list_mode(self, wide_csv, temp_model_dir):
        jobs = [
            {
                "csv_path": wide_csv,
                "series_id": "A",
                "column_mapping": {"datetime": "timestamp", "target": "feeder_A"},
            },
            {
                "csv_path": wide_csv,
                "series_id": "B",
                "column_mapping": {"datetime": "timestamp", "target": "feeder_B"},
            },
        ]
        res = batch_train_forecast_models(
            jobs=jobs, model_type="LinearRegression",
            lookback_hours=24, horizon_hours=6,
        )
        assert res["success"]
        assert res["summary"]["succeeded"] == 2

    def test_rejects_both_modes(self, wide_csv, temp_model_dir):
        res = batch_train_forecast_models(
            jobs=[{"csv_path": wide_csv}],
            csv_path=wide_csv,
            target_columns=["feeder_A"],
        )
        assert not res["success"]

    def test_missing_column_reported(self, wide_csv, temp_model_dir):
        res = batch_train_forecast_models(
            csv_path=wide_csv,
            target_columns=["does_not_exist"],
            datetime_col="timestamp",
        )
        assert not res["success"]
        assert "not found" in res["error"].lower()

    def test_fault_tolerant_one_bad_job(self, wide_csv, temp_model_dir):
        jobs = [
            {
                "csv_path": wide_csv, "series_id": "good",
                "column_mapping": {"datetime": "timestamp", "target": "feeder_A"},
            },
            {"csv_path": "/nonexistent/path.csv", "series_id": "bad"},
        ]
        res = batch_train_forecast_models(
            jobs=jobs, model_type="LinearRegression",
            lookback_hours=24, horizon_hours=6,
        )
        assert res["success"]  # batch itself ran
        assert res["summary"]["succeeded"] == 1
        assert res["summary"]["failed"] == 1
        assert "bad" in res["summary"]["failed_series"]


class TestBatchGenerate:
    def _train_fleet(self, wide_csv):
        return batch_train_forecast_models(
            csv_path=wide_csv,
            target_columns=["feeder_A", "feeder_B"],
            datetime_col="timestamp",
            model_type="LinearRegression",
            lookback_hours=24,
            horizon_hours=6,
        )

    def test_batch_forecast_fleet(self, wide_csv, temp_model_dir):
        trained = self._train_fleet(wide_csv)
        ids = {r["series_id"]: r["model_id"] for r in trained["results"]}

        df = pd.read_csv(wide_csv)
        ctx_paths = {}
        for f in ["feeder_A", "feeder_B"]:
            p = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
            df[["timestamp", f]].tail(200).to_csv(p.name, index=False)
            p.close()
            ctx_paths[f] = p.name

        try:
            jobs = [
                {"model_id": ids["feeder_A"], "csv_path": ctx_paths["feeder_A"], "series_id": "feeder_A"},
                {"model_id": ids["feeder_B"], "csv_path": ctx_paths["feeder_B"], "series_id": "feeder_B"},
                {"model_id": "missing_model", "csv_path": ctx_paths["feeder_A"], "series_id": "bad"},
            ]
            res = batch_generate_forecast(jobs=jobs)
            assert res["success"]
            assert res["summary"]["succeeded"] == 2
            assert res["summary"]["failed"] == 1
            good = [r for r in res["results"] if r["series_id"] == "feeder_A"][0]
            assert len(good["predictions"]) == 6
        finally:
            for p in ctx_paths.values():
                Path(p).unlink(missing_ok=True)

    def test_empty_jobs_rejected(self, temp_model_dir):
        res = batch_generate_forecast(jobs=[])
        assert not res["success"]


class TestBatchProbabilistic:
    def test_probabilistic_flows_through_batch(self, wide_csv, temp_model_dir):
        trained = batch_train_forecast_models(
            csv_path=wide_csv,
            target_columns=["feeder_A"],
            datetime_col="timestamp",
            model_type="XGBoost",
            lookback_hours=24,
            horizon_hours=6,
            probabilistic=True,
            quantiles=[0.1, 0.9],
        )
        assert trained["success"]
        model_id = trained["results"][0]["model_id"]

        df = pd.read_csv(wide_csv)
        p = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df[["timestamp", "feeder_A"]].tail(200).to_csv(p.name, index=False)
        p.close()
        try:
            res = batch_generate_forecast(
                jobs=[{"model_id": model_id, "csv_path": p.name, "series_id": "feeder_A"}],
                num_samples=100,
            )
            assert res["success"]
            entry = res["results"][0]
            assert entry["success"]
            assert entry["probabilistic"] is True
            first_row = entry["predictions"][0]
            assert "q0.1" in first_row and "q0.9" in first_row
            assert first_row["q0.1"] <= first_row["predicted_load"] <= first_row["q0.9"]
        finally:
            Path(p.name).unlink(missing_ok=True)
