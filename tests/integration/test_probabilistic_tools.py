"""
Integration tests: probabilistic forecasting through the MCP tools
(train -> forecast -> evaluate) with real model artefacts in a temp dir.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.tools import (
    train_forecast_model,
    generate_forecast,
    evaluate_forecast_model,
)


def _make_csv(n_hours: int = 700) -> str:
    t = np.arange(n_hours)
    load = (
        100
        + 20 * np.sin(2 * np.pi * t / 24)
        + 5 * np.sin(2 * np.pi * t / 168)
        + np.random.default_rng(11).normal(0, 3, n_hours)
    )
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=n_hours, freq="h"),
            "electricity_kwh": load,
        }
    )
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.to_csv(f.name, index=False)
    f.close()
    return f.name


@pytest.fixture
def csv_path():
    p = _make_csv()
    yield p
    Path(p).unlink(missing_ok=True)


@pytest.fixture
def temp_model_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", tmpdir)
        yield tmpdir


def test_probabilistic_rejected_for_point_model(csv_path, temp_model_dir):
    res = train_forecast_model(
        csv_path=csv_path, model_type="LinearRegression",
        lookback_hours=24, horizon_hours=6, probabilistic=True,
    )
    assert not res["success"]
    assert "probabilistic" in res["error"].lower()


def test_train_forecast_evaluate_probabilistic(csv_path, temp_model_dir):
    # Train quantile XGBoost
    train = train_forecast_model(
        csv_path=csv_path, model_type="XGBoost",
        lookback_hours=24, horizon_hours=6,
        probabilistic=True, quantiles=[0.1, 0.9],
        building_name="prob",
    )
    assert train["success"], train.get("error")
    model_id = train["model_id"]

    # Forecast → quantile bands present, ordered
    df = pd.read_csv(csv_path)
    ctx = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    df.tail(200).to_csv(ctx.name, index=False)
    ctx.close()
    try:
        fc = generate_forecast(model_id=model_id, csv_path=ctx.name, num_samples=100)
        assert fc["success"], fc.get("error")
        assert fc["probabilistic"] is True
        assert fc["quantiles"] == [0.1, 0.5, 0.9]
        row = fc["predictions"][0]
        assert row["q0.1"] <= row["predicted_load"] <= row["q0.9"]

        # Evaluate → probabilistic_metrics block present with sane values
        test = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        df.tail(300).to_csv(test.name, index=False)
        test.close()
        try:
            ev = evaluate_forecast_model(
                model_id=model_id, csv_path=test.name,
                return_predictions=False, num_samples=100,
            )
            assert ev["success"], ev.get("error")
            pm = ev["probabilistic_metrics"]
            assert pm["pinball_loss"] is not None
            assert 0.0 <= pm["coverage"] <= 1.0
            assert pm["nominal_coverage"] == pytest.approx(0.8)
            assert pm["mean_interval_width"] > 0
        finally:
            Path(test.name).unlink(missing_ok=True)
    finally:
        Path(ctx.name).unlink(missing_ok=True)
