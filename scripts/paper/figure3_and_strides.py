#!/usr/bin/env python3
"""Figure 3 and the stride check of the STLF-MCP SoftwareX paper.

Uses a model trained by reproduce_section3.py (the seed is chosen there):
  - Figure 3: day-ahead forecasts over 14-17 Aug 2023, RMSE at each forecast
    step h = 1..24, and the highest-demand day with measured/predicted peaks
  - stride check: pooled backtest MAPE for strides of 1, 6, 12 and 24 h

Usage (from the repository root):
    uv run python scripts/paper/reproduce_section3.py --seed 10 --out outputs/paper/sweep
    uv run python scripts/paper/figure3_and_strides.py \
        --results outputs/paper/sweep/section3_seed10.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

REPO = Path(__file__).resolve().parents[2]
EVAL_START = "2023-01-01 00:00"
EVAL_END = "2023-11-17 00:00"
# First forecast the backtest can score: 168 h of lag history plus the 48 h lookback.
BACKTEST_START = "2023-01-10"
STRIDES = [1, 6, 12, 24]
HORIZON = 24

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
PRED = "#2a78d6"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True, help="section3_seed<N>.json from reproduce_section3.py")
    p.add_argument("--out", default=None, help="output directory (default: next to --results)")
    p.add_argument("--skip-strides", action="store_true")
    args = p.parse_args()

    res_path = Path(args.results).resolve()
    run = json.loads(res_path.read_text())
    run_dir = res_path.parent
    out = Path(args.out).resolve() if args.out else run_dir / f"figure3_seed{run['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    os.environ["LOAD_FORECASTING_MODEL_DIR"] = str(run_dir / "models")
    csv_2023 = run_dir / "city_2023_with_weather.csv"

    sys.path.insert(0, str(REPO / "src"))
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402
    from load_forecasting.tools import backtest_model, evaluate_forecast_model  # noqa: E402

    ev = evaluate_forecast_model(
        model_id=run["model_id"], csv_path=str(csv_2023), return_predictions=True,
        start_date=EVAL_START, end_date=EVAL_END,
    )
    if not ev.get("success"):
        print(json.dumps(ev, indent=2, default=str))
        return 1
    df = pd.DataFrame(ev["predictions"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    # evaluate_forecast_model issues one forecast every HORIZON steps from the
    # first predicted timestamp, so the forecast step is the offset in the window.
    df["h"] = (np.arange(len(df)) % HORIZON) + 1
    origin_hour = df.index[0].hour
    hstep = (
        df.assign(se=df["residual"] ** 2)
        .groupby("h")["se"].mean().pow(0.5).rename("rmse_kwh").reset_index()
    )
    hstep.to_csv(out / "hstep_rmse.csv", index=False)
    df.to_csv(out / "evaluation_predictions.csv")

    daily_max = df["actual"].resample("D").max()
    peak_day = daily_max.idxmax().normalize()

    summary = {
        "seed": run["seed"],
        "model_id": run["model_id"],
        "test_metrics": ev["test_metrics"],
        "window_first_hour": int(origin_hour),
        "hstep_rmse_kwh": {int(r.h): round(float(r.rmse_kwh), 1) for r in hstep.itertuples()},
        "highest_demand_day": str(peak_day.date()),
    }

    _plot(df, hstep, peak_day, out / "figure3.png", out / "figure3.pdf")

    if not args.skip_strides:
        strides = {}
        for s in STRIDES:
            bt = backtest_model(
                model_id=run["model_id"], csv_path=str(csv_2023), stride_hours=s,
                backtest_start_date=BACKTEST_START, end_date=EVAL_END, return_predictions=False,
            )
            if not bt.get("success"):
                print(json.dumps(bt, indent=2, default=str))
                return 1
            strides[s] = {**bt["backtest_metrics"], "n_windows": bt["backtest_summary"]["n_windows"]}
            print(f"stride {s:>2} h: MAPE {bt['backtest_metrics']['mape']:.2f}%  windows {strides[s]['n_windows']}")
        summary["backtest_start"] = BACKTEST_START
        summary["strides"] = strides

    else:
        prev = out / "figure3_summary.json"
        if prev.exists():  # keep stride results from an earlier full run
            old = json.loads(prev.read_text())
            for k in ("backtest_start", "strides"):
                if k in old:
                    summary[k] = old[k]

    (out / "figure3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "strides"}, indent=2, default=str))
    print(f"\nWritten to {out}")
    return 0


def _plot(df, hstep, peak_day, png, pdf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK_2,
        "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"width_ratios": [2.2, 1.2, 1.2]})

    def kmwh(ax):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}"))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # (a) four consecutive day-ahead forecasts
    ax = axes[0]
    a = df.loc["2023-08-14":"2023-08-17"]
    starts = a.index[a["h"] == 1]
    for i, t0 in enumerate(starts):
        w = a.loc[t0:t0 + pd.Timedelta(hours=HORIZON - 1)]
        ax.plot(w.index, w["predicted"], color=PRED, lw=2, label="Predicted" if i == 0 else None, zorder=3)
        ax.axvline(t0, color=INK_2, lw=0.8, ls="--")
    ax.plot(a.index, a["actual"], color=INK, lw=1.5, label="Measured", zorder=2)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("Load (MWh)")
    ax.set_title("(a) Day-ahead forecasts, 14–17 Aug 2023", loc="left", fontsize=9.5)
    ax.set_ylim(30_000, 115_000)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], frameon=False, loc="upper right", ncol=2)
    kmwh(ax)

    # (b) RMSE at each forecast step
    ax = axes[1]
    ax.bar(hstep["h"], hstep["rmse_kwh"], width=0.7, color=PRED)
    ax.set_xlabel("Forecast step h (hours ahead)")
    ax.set_ylabel("RMSE (MWh)")
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.set_title("(b) RMSE by forecast step", loc="left", fontsize=9.5)
    kmwh(ax)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1000))

    # (c) highest-demand day
    ax = axes[2]
    d = df.loc[str(peak_day.date())]
    ax.plot(d.index, d["actual"], color=INK, lw=1.5, label="Measured")
    ax.plot(d.index, d["predicted"], color=PRED, lw=2, label="Predicted")
    ta, tp = d["actual"].idxmax(), d["predicted"].idxmax()
    ax.axvline(ta, color=INK, lw=0.8, ls="--")
    ax.axvline(tp, color=PRED, lw=0.8, ls="--")
    ax.plot([ta], [d["actual"].max()], marker="v", ms=8, color=INK, mec="white", mew=1.5, ls="none")
    ax.plot([tp], [d["predicted"].max()], marker="v", ms=8, color=PRED, mec="white", mew=1.5, ls="none")
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 6, 12, 18]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_title(f"(c) Highest-demand day, {peak_day:%d %b %Y}", loc="left", fontsize=9.5)
    ax.legend(frameon=False, loc="upper left")
    kmwh(ax)

    fig.tight_layout()
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)


if __name__ == "__main__":
    raise SystemExit(main())
