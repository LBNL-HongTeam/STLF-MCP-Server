"""Tests for the data-inspection report: split segments, payload, tool, and
the split provenance recorded at training time."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from load_forecasting.core.data_loader import ForecastingDataLoader
from load_forecasting.core.paths import ENV_OUTPUT_DIR
from load_forecasting.core.splits import (
    describe_split,
    season_of,
    step_for_frequency,
    summarize_segments,
)
from load_forecasting.reporting import build_data_report_payload, build_data_report_html
from load_forecasting.tools import generate_data_report, inspect_data, train_forecast_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _year_csv(path: Path, freq: str = "h", covariates: bool = True, gap: bool = False) -> Path:
    """A full calendar year with a diurnal + seasonal load signal."""
    idx = pd.date_range("2023-01-01", "2023-12-31 23:00", freq=freq)
    hours = idx.hour.values + idx.minute.values / 60
    doy = idx.dayofyear.values
    temp = 10 + 12 * np.sin((doy - 100) / 365 * 2 * np.pi) + 4 * np.sin(hours / 24 * 2 * np.pi)
    load = 1000 + 300 * np.sin((hours - 6) / 24 * 2 * np.pi) + 8 * np.abs(temp - 18) ** 1.3
    df = pd.DataFrame({"Datetime": idx, "load_kwh": load.round(2)})
    if covariates:
        df["T_out"] = temp.round(2)
        df["RH_out"] = (60 + 20 * np.cos(hours / 24 * 2 * np.pi)).round(1)
        df["site_index"] = (np.arange(len(idx)) % 7).astype(float)  # matches no pattern
    if gap:
        df = df.drop(df.index[1000:1030])  # 30-step hole
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def year_csv(tmp_path):
    return _year_csv(tmp_path / "year.csv")


@pytest.fixture
def year_loader(year_csv):
    return ForecastingDataLoader(str(year_csv), frequency="h", add_calendar_features=False, lag_hours=[])


# ---------------------------------------------------------------------------
# core/splits.py
# ---------------------------------------------------------------------------

class TestDescribeSplit:
    def test_sequential_two_segments(self, year_loader):
        tr, va = year_loader.split_train_val(0.2)
        segs = describe_split(tr.df.index, va.df.index, step_for_frequency("h"), label_seasons=False)
        assert [s["role"] for s in segs] == ["training", "validation"]
        assert all(s["season"] is None for s in segs)
        assert segs[0]["n_rows"] == len(tr.df) and segs[1]["n_rows"] == len(va.df)
        assert segs[0]["end"] < segs[1]["start"]

    def test_seasonal_segments_match_loader_and_show_winter_quirk(self, year_loader):
        tr, va = year_loader.split_train_val_seasonal(0.2)
        segs = describe_split(tr.df.index, va.df.index, step_for_frequency("h"))
        # 4 seasons x (train, val) + the Jan-Feb winter training block = 9
        assert len(segs) == 9
        assert sum(s["n_rows"] for s in segs if s["role"] == "training") == len(tr.df)
        assert sum(s["n_rows"] for s in segs if s["role"] == "validation") == len(va.df)
        # Jan-Feb is training-only; winter validation falls in December.
        first = segs[0]
        assert first["season"] == "winter" and first["role"] == "training"
        assert first["start"].startswith("2023-01-01") and first["end"].startswith("2023-02-28")
        winter_val = [s for s in segs if s["season"] == "winter" and s["role"] == "validation"]
        assert len(winter_val) == 1 and winter_val[0]["start"].startswith("2023-12")
        # Chronological and season-consistent
        assert [s["start"] for s in segs] == sorted(s["start"] for s in segs)
        for s in segs:
            assert season_of(pd.Timestamp(s["start"])) == s["season"]

    def test_gap_breaks_segment(self):
        idx = pd.date_range("2023-06-01", periods=48, freq="h")
        train = idx[:20].append(idx[30:40])   # hole between 20 and 30
        val = idx[40:]
        segs = describe_split(train, val, pd.Timedelta("1h"), label_seasons=False)
        assert [(s["role"], s["n_rows"]) for s in segs] == [("training", 20), ("training", 10), ("validation", 8)]

    def test_summary(self, year_loader):
        tr, va = year_loader.split_train_val_seasonal(0.2)
        segs = describe_split(tr.df.index, va.df.index, step_for_frequency("h"))
        summ = summarize_segments(segs)
        assert summ["n_segments"] == 9
        assert set(summ["by_season"]) == {"winter", "spring", "summer", "fall"}
        assert abs(summ["validation_fraction"] - 0.2) < 0.01
        assert summ["training_rows"] + summ["validation_rows"] == 8760

    def test_empty(self):
        assert describe_split(pd.DatetimeIndex([]), pd.DatetimeIndex([]), pd.Timedelta("1h")) == []
        assert summarize_segments([])["n_segments"] == 0


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

class TestPayload:
    def test_default_covariates_include_unmapped_numeric_columns(self, year_loader):
        p = build_data_report_payload(year_loader, csv_path="x.csv")
        names = {c["name"]: c["role"] for c in p.meta["covariates"]}
        # T_out / RH_out match the auto-detect patterns; site_index does not,
        # but is still drawn and tagged so a missed mapping is visible.
        assert names["T_out"] == "past_covariate"
        assert names["RH_out"] == "past_covariate"
        assert names["site_index"] == "unmapped"
        assert p.meta["unmapped_covariates"] == ["site_index"]
        assert set(p.series["covariates"]) == {"T_out", "RH_out", "site_index"}
        assert len(p.series["t"]) == len(p.series["target"]) == len(p.series["role"]) == 8760

    def test_explicit_covariate_subset_and_missing(self, year_loader):
        p = build_data_report_payload(year_loader, csv_path="x.csv", covariates=["T_out"])
        assert [c["name"] for c in p.meta["covariates"]] == ["T_out"]
        with pytest.raises(ValueError, match="covariates not found"):
            build_data_report_payload(year_loader, csv_path="x.csv", covariates=["nope"])

    @pytest.mark.parametrize("strategy,expect_segments,expect_roles", [
        ("seasonal", 9, {0, 1}),
        ("sequential", 2, {0, 1}),
        ("none", 0, {-1}),
    ])
    def test_split_strategies(self, year_loader, strategy, expect_segments, expect_roles):
        p = build_data_report_payload(year_loader, csv_path="x.csv", split_strategy=strategy)
        assert p.split["strategy"] == strategy
        assert len(p.split["segments"]) == expect_segments
        assert set(p.series["role"]) == expect_roles

    def test_seasonal_falls_back_with_note_on_short_data(self, tmp_path):
        idx = pd.date_range("2023-01-01", periods=24 * 60, freq="h")  # Jan-Feb only
        pd.DataFrame({"ts": idx, "load": np.random.default_rng(0).normal(100, 10, len(idx))}).to_csv(tmp_path / "s.csv", index=False)
        ldr = ForecastingDataLoader(str(tmp_path / "s.csv"), frequency="h", add_calendar_features=False, lag_hours=[])
        p = build_data_report_payload(ldr, csv_path="s.csv", split_strategy="seasonal")
        assert p.split["requested"] == "seasonal" and p.split["strategy"] == "sequential"
        assert "fall" in p.split["note"].lower() or "falls back" in p.split["note"]

    def test_invalid_strategy(self, year_loader):
        with pytest.raises(ValueError, match="split_strategy"):
            build_data_report_payload(year_loader, csv_path="x.csv", split_strategy="random")

    def test_downsampling_keeps_aggregates_full(self, year_loader):
        p = build_data_report_payload(year_loader, csv_path="x.csv", max_points=1000)
        assert p.series["downsampled"] is True and p.series["stride"] == 9
        assert p.series["n_points"] == len(p.series["t"]) <= 1000
        assert len(p.heatmap) == 8760          # aggregates ignore max_points
        assert p.meta["n_rows"] == 8760

    def test_seasons_gaps_windows(self, tmp_path):
        csv = _year_csv(tmp_path / "g.csv", gap=True)
        ldr = ForecastingDataLoader(str(csv), frequency="h", add_calendar_features=False, lag_hours=[])
        p = build_data_report_payload(ldr, csv_path="g.csv")
        assert [s["season"] for s in p.seasons] == ["winter", "spring", "summer", "fall", "winter"]
        assert len(p.gaps) == 1 and p.gaps[0]["n_missing_steps"] == 30
        assert [w["season"] for w in p.windows] == ["winter", "spring", "summer", "fall"]
        for w in p.windows:
            assert w["start"] <= w["peak_time"] <= w["end"]

    def test_outliers_match_inspect_data(self, year_csv, year_loader):
        p = build_data_report_payload(year_loader, csv_path="x.csv")
        ins = inspect_data(str(year_csv))["outlier_analysis"]
        assert len(p.outliers["value"]) == ins["value_outliers"]["n_outliers"]
        assert len(p.outliers["spikes"]) == ins["step_change_spikes"]["n_spikes"]

    def test_profiles_and_relations_shape(self, year_loader):
        p = build_data_report_payload(year_loader, csv_path="x.csv")
        assert len(p.profiles["hour_of_day"]) == 24
        assert len(p.profiles["hour_of_day_by_season"]) == 96
        assert len(p.profiles["day_of_week"]) == 7 and len(p.profiles["month"]) == 12
        rel = p.relations["T_out"]
        assert rel["pearson_r"] is not None
        assert 1 <= len(rel["binned"]) <= 24
        assert len(rel["scatter"]) == 3000 and {"x", "y", "hour"} <= set(rel["scatter"][0])

    def test_html_is_self_contained(self, year_loader):
        p = build_data_report_payload(year_loader, csv_path="x.csv", title="T")
        html = build_data_report_html(p)
        assert "<title>T</title>" in html
        assert "window.__REPORT_DATA__" in html
        assert "renderOverview" in html and "Plot.cell" in html
        assert "http://" not in html.split("<script")[0]  # no external assets in head
        assert 'src="http' not in html and "<link" not in html


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TestGenerateDataReportTool:
    def test_writes_report_and_summarises(self, year_csv, tmp_path):
        out = tmp_path / "r" / "report.html"
        r = generate_data_report(str(year_csv), str(out), title="Year")
        assert r["success"] is True, r.get("error")
        assert Path(r["output_html_path"]) == out.resolve() and out.stat().st_size == r["file_size_bytes"]
        assert r["n_rows"] == 8760 and r["frequency"] == "h" and r["target"] == "load_kwh"
        assert r["split"]["strategy"] == "seasonal" and len(r["split"]["segments"]) == 9
        assert r["unmapped_covariates"] == ["site_index"]
        assert r["ready_to_train"] is True
        assert "split" in " ".join(r["sections"])

    def test_default_output_path_uses_env_dir(self, year_csv, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_OUTPUT_DIR, str(tmp_path / "out"))
        r = generate_data_report(str(year_csv))
        assert r["success"] is True
        assert r["output_html_path"] == str((tmp_path / "out" / "reports" / "year_data_report.html").resolve())
        assert Path(r["output_html_path"]).exists()

    def test_bad_inputs(self, year_csv, tmp_path):
        assert generate_data_report(str(year_csv), str(tmp_path / "x.txt"))["success"] is False
        assert "split_strategy" in generate_data_report(str(year_csv), str(tmp_path / "x.html"), split_strategy="random")["error"]
        assert "validation_split" in generate_data_report(str(year_csv), str(tmp_path / "x.html"), validation_split=0.9)["error"]
        assert generate_data_report(str(tmp_path / "missing.csv"), str(tmp_path / "x.html"))["success"] is False
        r = generate_data_report(str(year_csv), str(tmp_path / "x.html"), covariates=["nope"])
        assert r["success"] is False and "nope" in r["error"]

    def test_fifteen_minute_sequential(self, tmp_path):
        csv = _year_csv(tmp_path / "q.csv", freq="15min", covariates=False)
        r = generate_data_report(str(csv), str(tmp_path / "q.html"), split_strategy="sequential")
        assert r["success"] is True and r["frequency"] == "15min"
        assert r["covariates"] == [] and len(r["split"]["segments"]) == 2


# ---------------------------------------------------------------------------
# Training records the same segments
# ---------------------------------------------------------------------------

def test_training_metadata_records_split_segments(year_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("LOAD_FORECASTING_MODEL_DIR", str(tmp_path / "models"))
    r = train_forecast_model(csv_path=str(year_csv), model_type="NaiveMean", building_name="seg")
    assert r["success"] is True, r.get("error")
    split = r["data_summary"]["split"]
    assert split["strategy"] == "sequential"        # NaiveMean is not a multi-series model
    assert [s["role"] for s in split["segments"]] == ["training", "validation"]
    assert split["segment_summary"]["n_segments"] == 2
    meta = json.loads((tmp_path / "models" / r["model_id"] / "metadata.json").read_text())
    assert meta["data_info"]["split"]["segments"] == split["segments"]
