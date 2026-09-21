"""Tests for the standalone training report and the capture that feeds it:
XGBoost validation curves, tuning-study persistence, effective split
strategy and device provenance, and the report payload / tool."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.model_registry import ModelRegistry, ModelNotFoundError
from load_forecasting.reporting import build_training_report_payload, build_training_report_html
from load_forecasting.reporting.training_payload import _curve, _flags, _environment
from load_forecasting.core.trainer import collect_environment
from load_forecasting.tools import generate_training_report, train_forecast_model, tune_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    d = tmp_path / "models"
    monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", str(d))
    return d


@pytest.fixture
def load_csv(tmp_path):
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-01", periods=24 * 30, freq="h")
    h = idx.hour.values
    df = pd.DataFrame({
        "ts": idx,
        "load_kwh": 100 + 30 * np.sin((h - 6) / 24 * 2 * np.pi) + rng.normal(0, 3, len(idx)),
        "T_out": 10 + 8 * np.sin(h / 24 * 2 * np.pi),
    })
    p = tmp_path / "load.csv"
    df.to_csv(p, index=False)
    return p


def _meta(model_id, model_type, *, history=None, train=None, val=None, tuning=None, split=None, extra_cfg=None):
    """Minimal metadata.json content in the shape the registry writes."""
    return {
        "model_id": model_id,
        "model_type": model_type,
        "darts_class": f"darts.models.{model_type}",
        "building_name": "b",
        "created_at": "2026-09-21T10:00:00+00:00",
        "config": {"lookback_hours": 24, "horizon_hours": 6, "frequency": "h", "validation_split": 0.2, **(extra_cfg or {})},
        "column_mapping": {"datetime": "ts", "target": "load_kwh", "past_covariates": ["T_out", "cal_hour", "lag_24h"], "future_covariates": []},
        "data_info": {"csv_path": "/x/load.csv", "target_column": "load_kwh", "frequency_detected": "h", "total_samples": 720,
                      "start_date": "2023-01-01T00:00:00+00:00", "end_date": "2023-01-30T23:00:00+00:00", "split": split or {}},
        "metrics": {"training": train or {"rmse": 1, "mae": 1, "mape": 1, "cv_rmse": 1.0, "r_squared": 0.99},
                    "validation": val or {"rmse": 1, "mae": 1, "mape": 1, "cv_rmse": 1.2, "r_squared": 0.98}},
        "training_info": {"training_time_seconds": 3.2, "energy_kwh": 1e-5, "watts_assumed": 15,
                          **({"training_history": history} if history else {}),
                          **({"tuning": tuning} if tuning else {})},
    }


# ---------------------------------------------------------------------------
# Curve diagnostics (pure)
# ---------------------------------------------------------------------------

class TestCurveDiagnostics:
    def test_over_trained(self):
        c = _curve({"epochs": [0, 1, 2, 3, 4], "train_loss": [1, .5, .3, .2, .1], "val_loss": [1, .5, .4, .5, .6]})
        assert c["best"] == {"x": 2, "val": 0.4}
        assert any("over-trained" in d for d in c["diagnostics"])
        assert c["gap_ratio"] == 6.0

    def test_still_improving(self):
        c = _curve({"epochs": [0, 1, 2, 3], "train_loss": [1, .6, .4, .3], "val_loss": [1, .7, .5, .4]})
        assert c["best"]["x"] == 3
        assert any("still falling" in d for d in c["diagnostics"])

    def test_converged_no_diagnostic(self):
        c = _curve({"epochs": [0, 1, 2, 3], "train_loss": [1, .5, .4, .4], "val_loss": [1, .5, .41, .41]})
        assert c["diagnostics"] == []

    def test_val_only_curve(self):
        c = _curve({"epochs": [1, 2, 3], "train_loss": [None, None, None], "val_loss": [.3, .2, .25], "x_label": "iteration", "metric": "rmse"})
        assert c["has_train"] is False and c["has_val"] is True
        assert c["x_label"] == "iteration" and c["metric"] == "rmse"
        assert c["gap_ratio"] is None

    def test_empty(self):
        assert _curve(None) is None and _curve({"epochs": []}) is None

    def test_flags(self):
        m = {"training": {"cv_rmse": 1.0}, "validation": {"cv_rmse": 2.5}}
        f = _flags("XGBoost", m, None)
        assert any(x.startswith("OVERFIT") for x in f) and any(x.startswith("NO_CURVE") for x in f)
        assert any("one shot" in x for x in _flags("LinearRegression", {"training": {}, "validation": {}}, None))


# ---------------------------------------------------------------------------
# Payload + HTML from synthetic metadata
# ---------------------------------------------------------------------------

class TestPayload:
    def test_blocks_and_comparison(self):
        hist = {"epochs": [0, 1, 2], "train_loss": [1, .5, .3], "val_loss": [1, .6, .5]}
        tuning = {"n_trials": 3, "n_trials_completed": 3, "best_trial": 1, "best_cv_rmse": 2.0,
                  "best_params": {"max_depth": 3}, "search_space": {"max_depth": {"type": "int", "low": 2, "high": 6}},
                  "trials": [{"trial": 0, "params": {"max_depth": 5}, "cv_rmse": 3.0, "duration_s": 1, "metrics": {}},
                             {"trial": 1, "params": {"max_depth": 3}, "cv_rmse": 2.0, "duration_s": 1, "metrics": {}, "train_loss": [1, .5]},
                             {"trial": 2, "params": {}, "cv_rmse": None, "duration_s": 0.1, "metrics": {}}]}
        split = {"strategy": "sequential", "segments": [{"role": "training", "season": None, "start": "2023-01-01T00:00:00", "end": "2023-01-24T23:00:00", "n_rows": 576},
                                                        {"role": "validation", "season": None, "start": "2023-01-25T00:00:00", "end": "2023-01-30T23:00:00", "n_rows": 144}],
                 "segment_summary": {"n_segments": 2, "training_rows": 576, "validation_rows": 144}}
        models = [
            ("a_lstm", _meta("a_lstm", "LSTM", history=hist, val={"cv_rmse": 5.0, "mape": 4, "r_squared": .9}, split=split, extra_cfg={"device": "mps", "model_kwargs": {"n_epochs": 3}})),
            ("b_lr", _meta("b_lr", "LinearRegression", val={"cv_rmse": 3.0, "mape": 2.5, "r_squared": .95})),
            ("c_xgb", _meta("c_xgb", "XGBoost", tuning=tuning, train={"cv_rmse": 0.5}, val={"cv_rmse": 2.0, "mape": 2, "r_squared": .97}, extra_cfg={"tuned": True, "best_hyperparameters": {"max_depth": 3}})),
        ]
        p = build_training_report_payload(models, title="T")
        assert p.meta["n_models"] == 3 and p.meta["title"] == "T"
        a, b, c = p.models
        assert a["curve"]["n_points"] == 3 and a["config"]["device"] == "mps"
        assert a["data"]["past_covariates"] == ["T_out"] and a["data"]["generated_covariates"] == ["cal_hour", "lag_24h"]
        assert a["split"]["segments"][1]["role"] == "validation"
        assert b["curve"] is None and any(f.startswith("NO_CURVE") for f in b["flags"])
        assert c["tuning"]["best_trial"] == 1 and c["tuning"]["n_failed"] == 1
        assert c["tuning"]["numeric_params"] == ["max_depth"] and c["tuning"]["has_trial_curves"] is True
        assert c["tuning"]["best_so_far"] == [{"trial": 0, "best": 3.0}, {"trial": 1, "best": 2.0}, {"trial": 2, "best": 2.0}]
        assert c["config"]["model_kwargs"] == {"max_depth": 3} and c["config"]["tuned"] is True
        assert any(f.startswith("OVERFIT") for f in c["flags"])
        # ranked by validation cv_rmse: c (2.0) < b (3.0) < a (5.0)
        assert [r["model_id"] for r in p.comparison["rows"]] == ["c_xgb", "b_lr", "a_lstm"]
        assert p.comparison["best_model_id"] == "c_xgb"
        assert p.comparison["n_with_curves"] == 1 and p.comparison["n_tuned"] == 1

    def test_html_self_contained(self):
        p = build_training_report_payload([("m", _meta("m", "LSTM", history={"epochs": [0, 1], "train_loss": [1, .5], "val_loss": [1, .6]}))])
        html = build_training_report_html(p)
        assert "window.__REPORT_DATA__" in html and "renderCurves" in html
        assert 'src="http' not in html and "<link" not in html


# ---------------------------------------------------------------------------
# Capture in the real training paths
# ---------------------------------------------------------------------------

def test_xgboost_records_validation_curve_without_touching_stdout(load_csv, model_dir, capsys):
    r = train_forecast_model(csv_path=str(load_csv), model_type="XGBoost", lookback_hours=24, horizon_hours=6,
                             building_name="b", model_kwargs={"n_estimators": 12})
    assert r["success"] is True, r.get("error")
    assert capsys.readouterr().out == ""          # eval_set printing would corrupt MCP stdio
    h = r["training_info"]["training_history"]
    assert h["x_label"] == "iteration" and h["metric"] == "rmse"
    assert len(h["epochs"]) == 12 and all(v is not None for v in h["val_loss"])
    assert h["train_loss"][0] is None             # only the validation set is passed as eval_set
    meta = json.loads((model_dir / r["model_id"] / "metadata.json").read_text())
    assert meta["training_info"]["training_history"]["val_loss"] == h["val_loss"]
    assert meta["config"]["device"] is None       # non-Torch


def test_effective_split_strategy_is_recorded(load_csv, model_dir):
    # 30 days -> seasonal split must fall back; metadata must say "sequential", not "seasonal_chunked".
    r = train_forecast_model(csv_path=str(load_csv), model_type="XGBoost", lookback_hours=24, horizon_hours=6,
                             building_name="b", model_kwargs={"n_estimators": 5})
    split = r["data_summary"]["split"]
    assert split["strategy"] == "sequential"
    assert [s["role"] for s in split["segments"]] == ["training", "validation"]
    assert all(s["season"] is None for s in split["segments"])


def test_tune_model_persists_study(load_csv, model_dir):
    r = tune_model(csv_path=str(load_csv), model_type="XGBoost", n_trials=3, building_name="b",
                   search_space={"n_estimators": {"type": "int", "low": 5, "high": 15}, "max_depth": {"type": "int", "low": 2, "high": 4}})
    assert r["success"] is True, r.get("error")
    meta = json.loads((model_dir / r["model_id"] / "metadata.json").read_text())
    t = meta["training_info"]["tuning"]
    assert t["n_trials_completed"] == 3 and t["best_trial"] in {0, 1, 2}
    assert set(t["search_space"]) == {"n_estimators", "max_depth"}
    for trial in t["trials"]:
        assert {"trial", "params", "cv_rmse", "metrics", "duration_s"} <= set(trial)
        assert trial["duration_s"] >= 0
    assert meta["data_info"]["split"]["segments"]                     # provenance recorded for tuned models too
    assert meta["training_info"]["training_history"]["x_label"] == "iteration"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

def test_load_metadata(model_dir, load_csv):
    r = train_forecast_model(csv_path=str(load_csv), model_type="NaiveMean", building_name="b")
    reg = ModelRegistry()
    meta = reg.load_metadata(r["model_id"])
    assert meta["model_id"] == r["model_id"]
    with pytest.raises(ModelNotFoundError):
        reg.load_metadata("nope")


def test_generate_training_report_tool(load_csv, model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("LOAD_FORECASTING_OUTPUT_DIR", str(tmp_path / "out"))
    a = train_forecast_model(csv_path=str(load_csv), model_type="XGBoost", lookback_hours=24, horizon_hours=6, building_name="b", model_kwargs={"n_estimators": 8})
    b = train_forecast_model(csv_path=str(load_csv), model_type="LinearRegression", lookback_hours=24, horizon_hours=6, building_name="b")
    out = tmp_path / "r" / "train.html"
    r = generate_training_report(model_ids=[a["model_id"], b["model_id"]], output_html_path=str(out), title="T")
    assert r["success"] is True, r.get("error")
    assert Path(r["output_html_path"]).exists() and r["n_models"] == 2
    xgb, lr = r["models"]
    assert xgb["has_curve"] is True and xgb["x_label"] == "iteration" and xgb["best"]["x"] >= 1
    assert lr["has_curve"] is False and any(f.startswith("NO_CURVE") for f in lr["flags"])
    assert r["comparison"]["best_model_id"] in {a["model_id"], b["model_id"]}
    assert "learning curves" in r["sections"]

    # single model + default output path
    single = generate_training_report(model_id=a["model_id"])
    assert single["success"] is True and single["output_html_path"].endswith(f"{a['model_id']}_training_report.html")
    assert single["output_html_path"].startswith(str((tmp_path / "out").resolve()))

    # errors
    assert generate_training_report()["success"] is False
    assert "not found" in generate_training_report(model_id="nope")["error"].lower()
    assert generate_training_report(model_id=a["model_id"], output_html_path=str(tmp_path / "x.txt"))["success"] is False


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

class TestEnvironment:
    def test_collect_environment_shape(self):
        env = collect_environment("XGBoost")
        assert {"hostname", "os", "platform", "machine", "cpu", "cpu_count", "memory_gb", "python", "versions", "accelerator"} <= set(env)
        assert env["python"] and env["versions"]["darts"] and env["versions"]["xgboost"]
        assert env["accelerator"]["type"] == "cpu" and env["accelerator"]["effective"] == "cpu"
        json.dumps(env)  # must be JSON-serialisable for metadata.json

    def test_collect_environment_torch_cpu(self):
        env = collect_environment("LSTM", "cpu")
        acc = env["accelerator"]
        assert acc["requested"] == "cpu" and acc["type"] == "cpu" and acc["torch_threads"] >= 1
        assert env["versions"]["torch"]

    def test_recorded_at_training_time(self, load_csv, model_dir):
        r = train_forecast_model(csv_path=str(load_csv), model_type="XGBoost", lookback_hours=24, horizon_hours=6,
                                 building_name="b", model_kwargs={"n_estimators": 3})
        env = r["training_info"]["environment"]
        assert env["cpu"] and env["os"] and env["accelerator"]["type"] == "cpu"
        meta = json.loads((model_dir / r["model_id"] / "metadata.json").read_text())
        assert meta["training_info"]["environment"]["python"] == env["python"]

    def test_payload_block_and_missing(self):
        env = collect_environment("LSTM", "cpu")
        block = _environment(env)
        assert block["summary"].startswith("CPU · ")
        assert block["versions"]["torch"] == env["versions"]["torch"]
        assert _environment(None) is None            # models trained before capture existed
        p = build_training_report_payload([("old", _meta("old", "LSTM"))])
        assert p.models[0]["environment"] is None

    def test_tool_summary_includes_environment(self, load_csv, model_dir, tmp_path):
        r = train_forecast_model(csv_path=str(load_csv), model_type="NaiveMean", building_name="b")
        rep = generate_training_report(model_id=r["model_id"], output_html_path=str(tmp_path / "e.html"))
        e = rep["models"][0]["environment"]
        assert e and e["cpu"] and e["accelerator_type"] == "cpu" and e["darts"]
        assert "Environment" in Path(rep["output_html_path"]).read_text()
