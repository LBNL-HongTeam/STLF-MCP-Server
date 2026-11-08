#!/usr/bin/env python3
"""Reproduce the Section 3 example of the STLF-MCP SoftwareX paper.

Day-ahead TiDE forecast for the city-level AMI series:
  - train on 2021, evaluate on 2023 (fully observed hours only,
    1 Jan 00:00 to 17 Nov 2023 00:00, i.e. 7,681 hours)
  - 48 h lookback, 24 h horizon, hourly frequency
  - measured Open-Meteo T_out, RH_out and Direct_Radiation as future
    covariates (perfect weather foresight)
  - peak-day metrics on 22-25 Feb and 14-17 Aug 2023 (Li et al. 2025)

The script calls the same tool functions the MCP server exposes, with a
fixed random seed and CPU training so that repeated runs give identical
numbers on the same machine and software versions.

Usage (from the repository root):
    uv run python scripts/paper/reproduce_section3.py [--seed 42] [--out outputs/paper]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Same OpenMP guard as main.py; must run before torch / xgboost are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
AMI = REPO / "data" / "examples" / "AMI"

TARGET = "City (n=41703)"
WEATHER = ["T_out", "RH_out", "Direct_Radiation"]
PEAK_DATES = [
    "2023-02-22", "2023-02-23", "2023-02-24", "2023-02-25",
    "2023-08-14", "2023-08-15", "2023-08-16", "2023-08-17",
]
EVAL_START = "2023-01-01 00:00"
EVAL_END = "2023-11-17 00:00"  # last fully observed hour (inclusive)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPO / "outputs" / "paper"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-type", default="TiDE",
                        help="TiDE (paper example) or a baseline such as NaiveSeasonal or XGBoost")
    parser.add_argument("--test-temp-noise", type=float, default=0.0,
                        help="std (degC) of Gaussian noise added to the 2023 T_out to emulate "
                             "temperature forecast error; training data stay measured")
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="RNG seed for --test-temp-noise, kept fixed across model seeds")
    args = parser.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Keep models from this script out of the default registry.
    os.environ["LOAD_FORECASTING_MODEL_DIR"] = str(out / "models")

    sys.path.insert(0, str(REPO / "src"))
    from load_forecasting.tools import (  # noqa: E402
        evaluate_forecast_model,
        merge_covariates,
        train_forecast_model,
    )

    merged = {}
    for year in (2021, 2023):
        path = out / f"city_{year}_with_weather.csv"
        res = merge_covariates(
            load_csv_path=str(AMI / f"{year}_city_level.csv"),
            covariate_csv_path=str(AMI / "2021-2023_openmeteo_weather.csv"),
            output_csv_path=str(path),
            covariate_columns=WEATHER,
            covariate_timezone="UTC",
            load_timezone="America/Los_Angeles",
        )
        if not res.get("success"):
            print(json.dumps(res, indent=2, default=str))
            return 1
        merged[year] = path

    if args.test_temp_noise > 0:
        import numpy as np
        import pandas as pd

        df = pd.read_csv(merged[2023])
        rng = np.random.default_rng(args.noise_seed)
        df["T_out"] = df["T_out"] + rng.normal(0.0, args.test_temp_noise, len(df))
        noisy = out / f"city_2023_with_weather_tnoise{args.test_temp_noise:g}_rng{args.noise_seed}.csv"
        df.to_csv(noisy, index=False)
        merged[2023] = noisy

    mapping = {
        "datetime": "Datetime",
        "target": TARGET,
        "future_covariates": WEATHER,
    }

    train = train_forecast_model(
        csv_path=str(merged[2021]),
        model_type=args.model_type,
        lookback_hours=48,
        horizon_hours=24,
        frequency="h",
        building_name="paper_section3",
        column_mapping=mapping,
        device=args.device,
        model_kwargs=None if args.model_type.startswith("Naive") else {"random_state": args.seed},
    )
    if not train.get("success"):
        print(json.dumps(train, indent=2, default=str))
        return 1

    ev = evaluate_forecast_model(
        model_id=train["model_id"],
        csv_path=str(merged[2023]),
        column_mapping=mapping,
        return_predictions=False,
        peak_dates=PEAK_DATES,
        start_date=EVAL_START,
        end_date=EVAL_END,
    )
    if not ev.get("success"):
        print(json.dumps(ev, indent=2, default=str))
        return 1

    summary = {
        "model_type": args.model_type,
        "seed": args.seed,
        "device": args.device,
        "model_id": train["model_id"],
        "split": train.get("data_summary", {}).get("split"),
        "validation_metrics": train.get("validation_metrics") or train.get("metrics", {}).get("validation"),
        "training_time_seconds": train.get("training_info", {}).get("training_time_seconds"),
        "test_metrics": ev.get("test_metrics"),
        "test_summary": ev.get("test_summary"),
        "peak_metrics": ev.get("peak_metrics"),
        "evaluation_window": {"start": EVAL_START, "end": EVAL_END},
        "test_temp_noise_std_degC": args.test_temp_noise,
        "noise_seed": args.noise_seed if args.test_temp_noise > 0 else None,
        "environment": train.get("training_info", {}).get("environment"),
    }
    tag = "" if args.model_type == "TiDE" else f"{args.model_type}_"
    if args.test_temp_noise > 0:
        tag += f"tnoise{args.test_temp_noise:g}_"
    result_path = out / f"section3_{tag}seed{args.seed}.json"
    result_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("model_id", "test_metrics", "peak_metrics")}, indent=2, default=str))
    print(f"\nFull results written to {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
