"""
Build the JSON payload for the standalone training report.

Everything comes from ``metadata.json`` of one or more trained models -- no
test CSV, no re-evaluation, no model unpickling.  Per model:

- identity, data provenance, configuration (incl. model_kwargs / tuned params);
- the learning curve recorded at training time (per epoch for Lightning
  models, per boosting round for XGBoost), with best-point and gap
  diagnostics;
- training vs validation metrics and the same overfitting rule the training
  tool warns with;
- the train/validation split segments (when recorded);
- compute (time, estimated energy, device);
- the Optuna study when the model came from tune_model (trials, best-so-far,
  per-parameter scatter, per-trial train-loss curves when captured).

With several models the payload adds a comparison table ranked by
validation CV-RMSE so the page can overlay curves and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Optional

from ._render import to_html_json

METRIC_KEYS = ("rmse", "mae", "mape", "cv_rmse", "r_squared")
OVERFIT_RATIO = 2.0          # same threshold as tools._common._check_overfitting
VAL_RISE_TOLERANCE = 0.05    # final val > best val * (1 + tol) -> over-trained
STILL_IMPROVING_DROP = 0.01  # last step still lowering val by > 1 % -> under-trained


@dataclass
class TrainingReportPayload:
    meta: dict
    models: list
    comparison: dict
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def payload_to_json(payload: TrainingReportPayload) -> str:
    return to_html_json(payload.__dict__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f(v: Any, n: int = 6) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return round(x, n) if math.isfinite(x) else None


def _metrics(block: Optional[dict]) -> dict:
    block = block or {}
    out = {k: _f(block.get(k), 4) for k in METRIC_KEYS}
    for extra in ("peak_mape", "peak_timing_error_hours", "pinball_loss", "coverage"):
        if extra in block:
            out[extra] = _f(block.get(extra), 4)
    return out


def _curve(history: Optional[dict]) -> Optional[dict]:
    """Normalise a recorded training_history into points + diagnostics."""
    if not history:
        return None
    xs = history.get("epochs") or []
    if not xs:
        return None
    train = history.get("train_loss") or []
    val = history.get("val_loss") or []
    points = []
    for i, x in enumerate(xs):
        t = _f(train[i]) if i < len(train) else None
        v = _f(val[i]) if i < len(val) else None
        points.append({"x": int(x), "train": t, "val": v})

    has_train = any(p["train"] is not None for p in points)
    has_val = any(p["val"] is not None for p in points)
    out: dict = {
        "x_label": history.get("x_label") or "epoch",
        "metric": history.get("metric") or "loss",
        "points": points,
        "n_points": len(points),
        "has_train": has_train,
        "has_val": has_val,
        "best": None,
        "final": {"train": points[-1]["train"], "val": points[-1]["val"]},
        "gap_ratio": None,
        "diagnostics": [],
    }
    if has_val:
        valid = [p for p in points if p["val"] is not None]
        best = min(valid, key=lambda p: p["val"])
        out["best"] = {"x": best["x"], "val": best["val"]}
        final_val = valid[-1]["val"]
        if best["val"] and final_val is not None and best is not valid[-1]:
            if final_val > best["val"] * (1 + VAL_RISE_TOLERANCE):
                out["diagnostics"].append(
                    f"Validation {out['metric']} rose {100 * (final_val / best['val'] - 1):.0f}% after "
                    f"{out['x_label']} {best['x']} (best {best['val']:.4g}, final {final_val:.4g}). "
                    f"The model is over-trained; fewer {out['x_label']}s or early stopping would help."
                )
        if len(valid) >= 3 and best is valid[-1]:
            prev = valid[-2]["val"]
            if prev and final_val is not None and (prev - final_val) / prev > STILL_IMPROVING_DROP:
                out["diagnostics"].append(
                    f"Validation {out['metric']} was still falling at the last {out['x_label']} "
                    f"({prev:.4g} -> {final_val:.4g}); more {out['x_label']}s may improve the model."
                )
        if has_train and points[-1]["train"] and final_val is not None and points[-1]["train"] > 0:
            out["gap_ratio"] = _f(final_val / points[-1]["train"], 3)
    return out


def _flags(model_type: str, metrics: dict, curve: Optional[dict]) -> list[str]:
    flags: list[str] = []
    tr, va = metrics["training"].get("cv_rmse"), metrics["validation"].get("cv_rmse")
    if tr and va and tr > 0 and va / tr > OVERFIT_RATIO:
        flags.append(
            f"OVERFIT: validation CV-RMSE ({va:.2f}%) is {va / tr:.1f}x training CV-RMSE ({tr:.2f}%)."
        )
    if curve is None:
        flags.append(
            f"NO_CURVE: {model_type} records no per-epoch history "
            + ("(fits in one shot)." if model_type in ("LinearRegression", "ARIMA", "NaiveMean", "NaiveSeasonal", "NaiveMovingAverage")
               else "(not captured for this model type or trained before curves were recorded).")
        )
    else:
        flags.extend(f"CURVE: {d}" for d in curve["diagnostics"])
    return flags


def _tuning(info: Optional[dict]) -> Optional[dict]:
    if not info or not info.get("trials"):
        return None
    trials = []
    best_so_far = []
    running = None
    for t in info["trials"]:
        cv = _f(t.get("cv_rmse"), 4)
        if cv is not None:
            running = cv if running is None else min(running, cv)
        trials.append(
            {
                "trial": int(t.get("trial", len(trials))),
                "cv_rmse": cv,
                "duration_s": _f(t.get("duration_s"), 2),
                "params": t.get("params") or {},
                "metrics": _metrics(t.get("metrics")),
                "train_loss": [_f(v) for v in t["train_loss"]] if t.get("train_loss") else None,
                "failed": cv is None,
            }
        )
        best_so_far.append({"trial": trials[-1]["trial"], "best": running})
    # Numeric parameters that actually varied -> scatter panels
    numeric = []
    for name in {k for t in trials for k in t["params"]}:
        vals = [t["params"].get(name) for t in trials if isinstance(t["params"].get(name), (int, float)) and not isinstance(t["params"].get(name), bool)]
        if len(set(vals)) > 1:
            numeric.append(name)
    return {
        "n_trials": info.get("n_trials"),
        "n_completed": info.get("n_trials_completed", len(trials)),
        "n_failed": sum(1 for t in trials if t["failed"]),
        "best_trial": info.get("best_trial"),
        "best_cv_rmse": _f(info.get("best_cv_rmse"), 4),
        "best_params": info.get("best_params") or {},
        "search_space": info.get("search_space") or {},
        "trials": trials,
        "best_so_far": best_so_far,
        "numeric_params": sorted(numeric),
        "has_trial_curves": any(t["train_loss"] for t in trials),
    }


def _environment(env: Optional[dict]) -> Optional[dict]:
    """Machine / library snapshot recorded at training time (None for older models)."""
    if not env:
        return None
    acc = env.get("accelerator") or {}
    ver = env.get("versions") or {}
    parts = [p for p in (acc.get("type") and acc["type"].upper(), acc.get("name")) if p]
    return {
        "hostname": env.get("hostname"),
        "os": env.get("os"),
        "platform": env.get("platform"),
        "machine": env.get("machine"),
        "cpu": env.get("cpu"),
        "cpu_count": env.get("cpu_count"),
        "memory_gb": env.get("memory_gb"),
        "python": env.get("python"),
        "versions": {k: ver.get(k) for k in ("darts", "torch", "pytorch_lightning", "xgboost", "optuna", "numpy")},
        "omp_num_threads": env.get("omp_num_threads"),
        "accelerator": {
            "requested": acc.get("requested"),
            "effective": acc.get("effective"),
            "type": acc.get("type"),
            "name": acc.get("name"),
            "cuda_version": acc.get("cuda_version"),
            "device_count": acc.get("device_count"),
            "memory_gb": acc.get("memory_gb"),
            "torch_threads": acc.get("torch_threads"),
        },
        "summary": " · ".join(parts) if parts else None,
    }


def _model_block(model_id: str, meta: dict) -> dict:
    cfg = meta.get("config") or {}
    info = meta.get("training_info") or {}
    data = meta.get("data_info") or {}
    metrics = {
        "training": _metrics((meta.get("metrics") or {}).get("training")),
        "validation": _metrics((meta.get("metrics") or {}).get("validation")),
    }
    curve = _curve(info.get("training_history"))
    kwargs = cfg.get("model_kwargs") or {}
    if cfg.get("tuned"):
        kwargs = {**(cfg.get("best_hyperparameters") or {}), **kwargs}
    split = data.get("split") or {}
    csv_path = data.get("csv_path")
    cm = meta.get("column_mapping") or {}
    past_all = list(cm.get("past_covariates") or [])
    generated = [c for c in past_all if str(c).startswith(("cal_", "lag_"))]
    return {
        "model_id": model_id,
        "model_type": meta.get("model_type"),
        "darts_class": meta.get("darts_class"),
        "building_name": meta.get("building_name"),
        "created_at": meta.get("created_at"),
        "data": {
            "csv_path": csv_path,
            "csv_name": Path(csv_path).name if csv_path else None,
            "target": data.get("target_column"),
            "frequency": data.get("frequency_detected") or cfg.get("frequency"),
            "n_rows": data.get("total_samples"),
            "start": data.get("start_date"),
            "end": data.get("end_date"),
            "training_samples": data.get("training_samples"),
            "validation_samples": data.get("validation_samples"),
            "past_covariates": [c for c in past_all if c not in generated],
            "generated_covariates": generated,
            "future_covariates": list(cm.get("future_covariates") or []),
        },
        "config": {
            "lookback_hours": cfg.get("lookback_hours"),
            "horizon_hours": cfg.get("horizon_hours"),
            "validation_split": cfg.get("validation_split"),
            "probabilistic": bool(cfg.get("probabilistic")),
            "quantiles": cfg.get("quantiles"),
            "tuned": bool(cfg.get("tuned")),
            "model_kwargs": kwargs,
            "device": cfg.get("device") or kwargs.get("accelerator") or kwargs.get("device"),
        },
        "split": {
            "strategy": split.get("strategy"),
            "segments": split.get("segments") or [],
            "summary": split.get("segment_summary") or {},
            "train_steps": split.get("train_steps"),
            "validation_steps": split.get("validation_steps"),
        },
        "environment": _environment(info.get("environment")),
        "training": {
            "time_s": _f(info.get("training_time_seconds"), 2),
            "energy_kwh": _f(info.get("energy_kwh"), 8),
            "watts_assumed": info.get("watts_assumed"),
            "n_points": curve["n_points"] if curve else None,
            "x_label": curve["x_label"] if curve else None,
        },
        "curve": curve,
        "metrics": metrics,
        "flags": _flags(meta.get("model_type") or "", metrics, curve),
        "tuning": _tuning(info.get("tuning")),
    }


def build_training_report_payload(
    models: list[tuple[str, dict]],
    *,
    title: Optional[str] = None,
) -> TrainingReportPayload:
    """``models`` is a list of ``(model_id, metadata_dict)`` in the order requested."""
    blocks = [_model_block(mid, meta) for mid, meta in models]

    def _rank_key(b: dict):
        v = b["metrics"]["validation"].get("cv_rmse")
        return (v is None, v if v is not None else 0.0)

    ranked = sorted(blocks, key=_rank_key)
    rows = [
        {
            "rank": i + 1,
            "model_id": b["model_id"],
            "model_type": b["model_type"],
            "val_cv_rmse": b["metrics"]["validation"].get("cv_rmse"),
            "val_mape": b["metrics"]["validation"].get("mape"),
            "val_r_squared": b["metrics"]["validation"].get("r_squared"),
            "train_cv_rmse": b["metrics"]["training"].get("cv_rmse"),
            "time_s": b["training"]["time_s"],
            "energy_kwh": b["training"]["energy_kwh"],
            "n_points": b["training"]["n_points"],
            "best_x": (b["curve"] or {}).get("best", {}) and b["curve"]["best"]["x"] if b["curve"] and b["curve"].get("best") else None,
            "tuned": b["config"]["tuned"],
            "flags": len(b["flags"]),
        }
        for i, b in enumerate(ranked)
    ]
    comparison = {
        "basis": "validation cv_rmse (lower is better)",
        "rows": rows,
        "best_model_id": rows[0]["model_id"] if rows and rows[0]["val_cv_rmse"] is not None else None,
        "n_with_curves": sum(1 for b in blocks if b["curve"]),
        "n_tuned": sum(1 for b in blocks if b["tuning"]),
    }
    meta = {
        "title": title or (f"Training report — {blocks[0]['model_id']}" if len(blocks) == 1 else f"Training report — {len(blocks)} models"),
        "n_models": len(blocks),
        "model_ids": [b["model_id"] for b in blocks],
    }
    return TrainingReportPayload(meta=meta, models=blocks, comparison=comparison)
