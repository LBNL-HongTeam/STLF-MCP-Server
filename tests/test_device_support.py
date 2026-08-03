"""
Tests for compute-device handling (CUDA / MPS / CPU).

Covers:
- _detect_accelerator priority: cuda > mps > cpu
- _normalize_device: aliases, auto/None, case-insensitivity, validation, and
  graceful fallback to CPU when a requested accelerator is unavailable
- Tool-level `device` parameter validation for train_forecast_model / tune_model
"""

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core import trainer
from load_forecasting.core.trainer import (
    _detect_accelerator,
    _normalize_device,
    VALID_ACCELERATORS,
)
from load_forecasting.tools import (
    train_forecast_model,
    tune_model,
)


# ---------------------------------------------------------------------------
# Fake torch backends for deterministic detection tests
# ---------------------------------------------------------------------------

def _fake_torch(cuda: bool, mps: bool):
    """Build a mock ``torch`` module with configurable cuda/mps availability."""
    m = mock.MagicMock()
    m.cuda.is_available.return_value = cuda
    m.backends.mps.is_available.return_value = mps
    m.backends.mps.is_built.return_value = mps
    return m


def _patch_torch(monkeypatch, cuda: bool, mps: bool):
    """Patch ``import torch`` inside trainer to return our fake module."""
    fake = _fake_torch(cuda, mps)
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return fake
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    return fake


# ---------------------------------------------------------------------------
# _detect_accelerator
# ---------------------------------------------------------------------------

def test_detect_prefers_cuda_over_mps(monkeypatch):
    _patch_torch(monkeypatch, cuda=True, mps=True)
    assert _detect_accelerator() == "cuda"


def test_detect_cuda_when_only_cuda(monkeypatch):
    _patch_torch(monkeypatch, cuda=True, mps=False)
    assert _detect_accelerator() == "cuda"


def test_detect_mps_when_no_cuda(monkeypatch):
    _patch_torch(monkeypatch, cuda=False, mps=True)
    assert _detect_accelerator() == "mps"


def test_detect_cpu_when_nothing(monkeypatch):
    _patch_torch(monkeypatch, cuda=False, mps=False)
    assert _detect_accelerator() == "cpu"


def test_detect_cpu_when_torch_missing(monkeypatch):
    real_import = __import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_torch)
    assert _detect_accelerator() == "cpu"


# ---------------------------------------------------------------------------
# _normalize_device
# ---------------------------------------------------------------------------

def test_normalize_none_autodetects(monkeypatch):
    _patch_torch(monkeypatch, cuda=True, mps=False)
    assert _normalize_device(None) == "cuda"


def test_normalize_auto_autodetects(monkeypatch):
    _patch_torch(monkeypatch, cuda=False, mps=True)
    assert _normalize_device("auto") == "mps"


def test_normalize_empty_string_autodetects(monkeypatch):
    _patch_torch(monkeypatch, cuda=False, mps=False)
    assert _normalize_device("") == "cpu"


def test_normalize_gpu_alias_maps_to_cuda(monkeypatch):
    _patch_torch(monkeypatch, cuda=True, mps=False)
    assert _normalize_device("gpu") == "cuda"


def test_normalize_is_case_insensitive(monkeypatch):
    _patch_torch(monkeypatch, cuda=True, mps=False)
    assert _normalize_device("CUDA") == "cuda"


def test_normalize_cpu_always_ok(monkeypatch):
    # cpu never requires torch availability
    _patch_torch(monkeypatch, cuda=False, mps=False)
    assert _normalize_device("cpu") == "cpu"


def test_normalize_cuda_unavailable_falls_back_to_cpu(monkeypatch):
    _patch_torch(monkeypatch, cuda=False, mps=True)
    assert _normalize_device("cuda") == "cpu"


def test_normalize_mps_unavailable_falls_back_to_cpu(monkeypatch):
    _patch_torch(monkeypatch, cuda=True, mps=False)
    assert _normalize_device("mps") == "cpu"


def test_normalize_cuda_available_returns_cuda(monkeypatch):
    _patch_torch(monkeypatch, cuda=True, mps=False)
    assert _normalize_device("cuda") == "cuda"


def test_normalize_rejects_unknown_device():
    with pytest.raises(ValueError, match="Unknown device"):
        _normalize_device("tpu")


def test_valid_accelerators_contents():
    assert {"cuda", "mps", "cpu", "auto"} <= VALID_ACCELERATORS


# ---------------------------------------------------------------------------
# Tool-level validation
# ---------------------------------------------------------------------------

def test_train_tool_rejects_bad_device(tmp_path):
    # Minimal CSV; validation should fail before any heavy work.
    csv = tmp_path / "d.csv"
    csv.write_text("datetime,load\n2023-01-01 00:00:00,1.0\n")
    resp = train_forecast_model(csv_path=str(csv), model_type="LinearRegression",
                                device="tpu")
    assert resp["success"] is False
    assert "Unknown device" in resp["error"]


def test_tune_tool_rejects_bad_device(tmp_path):
    csv = tmp_path / "d.csv"
    csv.write_text("datetime,load\n2023-01-01 00:00:00,1.0\n")
    resp = tune_model(csv_path=str(csv), model_type="XGBoost", device="banana")
    assert resp["success"] is False
    assert "Unknown device" in resp["error"]


def test_train_tool_accepts_cpu_device(monkeypatch, tmp_path):
    """A valid device string must pass validation (LinearRegression ignores it)."""
    import numpy as np
    import pandas as pd

    n = 400
    t = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "datetime": t,
        "load": 10 + np.sin(np.arange(n) * 2 * np.pi / 24),
    })
    csv = tmp_path / "load.csv"
    df.to_csv(csv, index=False)

    resp = train_forecast_model(
        csv_path=str(csv),
        model_type="LinearRegression",
        lookback_hours=24,
        horizon_hours=6,
        device="cpu",
    )
    assert resp["success"] is True, resp.get("error")
