"""
MCP tools for HTML report/dashboard generation:
  - generate_evaluation_report
  - generate_backtest_report
  - generate_inference_dashboard
"""

from pathlib import Path
from typing import Optional
import logging

from ..core.data_loader import ForecastingDataLoader, DataLoadError
from ..core.evaluator import calculate_residual_analysis
from ..core.model_registry import (
    ModelRegistry,
    ModelNotFoundError,
    ModelCorruptedError,
)
from ..core.frequency_utils import FREQ_TO_STEPS_PER_HOUR
from ..core.weather_fetcher import (
    fetch_weather_forecast_df,
    VARIABLE_PRESETS as _WEATHER_PRESETS,
)
from ._common import (
    create_success_response,
    create_error_response,
    _ModelLoadError,
    _load_model_context,
    _peak_metrics_safe,
    _merge_peak_headline,
    _validate_html_output_path,
    _df_to_records,
    _parse_raw_csv_for_inference,
)
from .evaluation import evaluate_forecast_model
from .backtest import _run_backtest_core
from .inference import generate_forecast

logger = logging.getLogger(__name__)


def generate_evaluation_report(
    model_id: str,
    csv_path: str,
    output_html_path: str,
    column_mapping: Optional[dict] = None,
    include_residual_analysis: bool = True,
    peak_dates: Optional[list] = None,
    title: Optional[str] = None,
) -> dict:
    """
    Generate a self-contained HTML evaluation report for a trained model.

    Runs evaluate_forecast_model internally to obtain test metrics and
    per-step predictions, then renders an interactive HTML page that bundles
    every JS dependency and the full dataset inline. The resulting file can
    be opened directly in any modern browser (file://) — no web server, no
    network access required.

    Args:
        model_id: ID of a trained model (from train_forecast_model).
        csv_path: Path to CSV with test data (same column layout as training).
        output_html_path: Path where the .html report will be written.
        column_mapping: Optional column mapping override.  Defaults to the
            mapping stored in the model metadata.
        include_residual_analysis: If True, compute mean/std residual and
            lag-1 autocorrelation and include them in the report.
        peak_dates: Optional list of date strings ("YYYY-MM-DD") identifying
            peak demand days for PMAPE / PTE evaluation (Li et al. 2025,
            Table 5). Example:
            ["2023-02-22", "2023-02-23", "2023-02-24", "2023-02-25"]
            When provided, a Peak Metrics section is added to the HTML
            report and ``peak_metrics`` is added to the tool response.
        title: Optional human-friendly title for the report header.

    Returns:
        Dict with output_html_path, file_size_bytes, n_points, model_id,
        model_type, and test_metrics.  Includes ``peak_metrics`` when
        ``peak_dates`` is provided.
    """
    try:
        # Validate output path
        if not output_html_path:
            return create_error_response("output_html_path is required.")
        path_err = _validate_html_output_path(output_html_path)
        if path_err:
            return create_error_response(path_err)
        out_path = Path(output_html_path)

        # ------------------------------------------------------------------
        # Run evaluation to get predictions + metrics (single source of truth)
        # ------------------------------------------------------------------
        eval_result = evaluate_forecast_model(
            model_id=model_id,
            csv_path=csv_path,
            column_mapping=column_mapping,
            return_predictions=True,
            output_csv_path=None,
            include_residual_analysis=include_residual_analysis,
            peak_dates=peak_dates,
        )
        if not eval_result.get("success"):
            return create_error_response(
                f"Evaluation step failed: {eval_result.get('error')}"
            )

        # ------------------------------------------------------------------
        # Load model metadata for the header section
        # ------------------------------------------------------------------
        registry = ModelRegistry()
        try:
            _, metadata, _ = registry.load_model(model_id)
        except (ModelNotFoundError, ModelCorruptedError) as e:
            # Already evaluated successfully above, so this is unusual — but
            # fall back to a metadata stub rather than failing the report.
            logger.warning(f"Could not reload metadata for {model_id}: {e}")
            metadata = {
                "model_type": eval_result.get("model_type"),
                "config": {},
            }

        resolved_mapping = column_mapping or metadata.get("column_mapping") or {}

        # ------------------------------------------------------------------
        # Load raw input frame for the "input data preview" section.
        # Best-effort: a failure here only disables the preview, it does not
        # fail the whole report (the predictions section is still useful).
        # ------------------------------------------------------------------
        input_df = None
        try:
            frequency = (metadata.get("config") or {}).get("frequency", "h")
            preview_loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=resolved_mapping or None,
                frequency=frequency,
                add_calendar_features=False,
                lag_hours=[],
            )
            input_df = preview_loader.df
            # Make sure resolved_mapping reflects what the loader detected
            resolved_mapping = preview_loader.column_mapping
        except Exception as e:
            logger.warning(f"Skipping input-data preview: {e}")

        # ------------------------------------------------------------------
        # Build payload + HTML
        # ------------------------------------------------------------------
        from ..reporting import build_report_payload, build_report_html

        payload = build_report_payload(
            model_id=model_id,
            model_metadata=metadata,
            eval_result=eval_result,
            input_df=input_df,
            column_mapping=resolved_mapping,
            title=title,
        )

        html = build_report_html(payload)

        out_path.write_text(html, encoding="utf-8")
        file_size = out_path.stat().st_size

        response = create_success_response(
            output_html_path=str(out_path),
            file_size_bytes=int(file_size),
            n_points=len(eval_result.get("predictions") or []),
            model_id=model_id,
            model_type=eval_result.get("model_type"),
            test_metrics=eval_result.get("test_metrics"),
            comparison_to_validation=eval_result.get("comparison_to_validation"),
        )
        if eval_result.get("peak_metrics"):
            response["peak_metrics"] = eval_result["peak_metrics"]
        return response

    except Exception as e:
        logger.exception("generate_evaluation_report failed")
        return create_error_response(f"Report generation failed: {str(e)}")


def generate_backtest_report(
    model_id: str,
    csv_path: str,
    output_html_path: str,
    column_mapping: Optional[dict] = None,
    stride_hours: Optional[int] = None,
    start_fraction: float = 0.2,
    include_residual_analysis: bool = True,
    peak_dates: Optional[list] = None,
    title: Optional[str] = None,
    num_samples: int = 200,
) -> dict:
    """
    Generate a self-contained HTML backtest report for a trained model.

    Runs a rolling-window backtest internally, computes h-step-ahead error
    metrics (RMSE/MAE/MAPE per forecast step), and renders an interactive HTML
    page with:
      - A forecast playback slider that steps through every re-forecast window,
        showing the model's predicted trajectory overlaid on actual load.
      - An h-step-ahead RMSE bar chart showing how accuracy degrades with horizon.
      - For quantile-trained models: a shaded prediction band on the playback
        chart plus coverage and interval width broken out per horizon step,
        which pooled interval metrics cannot reveal.
      - Overall backtest metrics (RMSE, MAE, MAPE, CV-RMSE, R²).
      - Hour-of-day and day-of-week MAE error profiles.
      - Optional residual analysis (mean residual, std, lag-1 autocorrelation).

    The output HTML is fully self-contained (d3 v7 + Observable Plot v0.6 are
    vendored inline) and can be opened directly in any modern browser via
    file://.

    Args:
        model_id: ID of a trained model (from train_forecast_model).
        csv_path: Path to CSV with historical data.
        output_html_path: Path where the .html report will be written.
        column_mapping: Optional column role mapping override.  Defaults to the
            mapping stored in the model metadata.
        stride_hours: Step size between re-forecast windows in hours.  Defaults
            to the model's training horizon (non-overlapping windows).
        start_fraction: Fraction of the series to skip before starting the
            rolling forecast.  Must be between 0.1 and 0.5.  Default: 0.2.
        include_residual_analysis: If True, compute and include mean/std
            residual and lag-1 autocorrelation statistics.
        peak_dates: Optional list of date strings ("YYYY-MM-DD") identifying
            peak demand days for PMAPE / PTE evaluation (Li et al. 2025).
        title: Optional human-friendly title for the report header.
        num_samples: Monte-Carlo sample count for quantile-trained models.
            Adds a Prediction intervals section with pooled interval metrics
            and a coverage-by-horizon-step chart, plus a shaded band on the
            playback chart.  Ignored for point models (default 200).

    Returns:
        Dict with output_html_path, file_size_bytes, n_windows, n_points,
        model_id, model_type, backtest_metrics, and comparison_to_validation.
    """
    try:
        if not output_html_path:
            return create_error_response("output_html_path is required.")
        path_err = _validate_html_output_path(output_html_path)
        if path_err:
            return create_error_response(path_err)
        out_path = Path(output_html_path)

        try:
            core = _run_backtest_core(
                model_id=model_id,
                csv_path=csv_path,
                column_mapping=column_mapping,
                stride_hours=stride_hours,
                start_fraction=start_fraction,
                include_residual_analysis=include_residual_analysis,
                peak_dates=peak_dates,
                reconstruct_windows=True,
                num_samples=num_samples,
            )
        except _ModelLoadError as e:
            return e.response

        metadata = core["metadata"]
        model_type = core["model_type"]
        target_series = core["target_series"]
        start_idx = core["start_idx"]
        target_scaler = core["target_scaler"]
        resolved_mapping = core["resolved_mapping"]
        frequency = core["frequency"]
        backtest_metrics = core["backtest_metrics"]
        comparison = core["comparison"]
        backtest_summary = core["backtest_summary"]
        n_windows = core["n_windows"]
        predictions_flat = core["predictions_flat"]
        min_len = len(predictions_flat)

        # Optional residual analysis
        residual_analysis: dict = {}
        if include_residual_analysis:
            try:
                residual_analysis = calculate_residual_analysis(
                    target_series[start_idx:],
                    core["predictions_concat"],
                    scaler=target_scaler,
                )
            except Exception as e:
                logger.warning("Residual analysis failed: %s", e)

        peak_metrics = _peak_metrics_safe(
            target_series[start_idx:], core["predictions_concat"], peak_dates, target_scaler
        )
        # Surface the peak headline alongside the standard metrics (both in the
        # embedded payload and the tool response); full detail stays separate.
        _merge_peak_headline(backtest_metrics, peak_metrics)

        # Load raw input frame for the input-preview section (best-effort)
        input_df = None
        try:
            preview_loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=resolved_mapping or None,
                frequency=frequency,
                add_calendar_features=False,
                lag_hours=[],
            )
            input_df = preview_loader.df
        except Exception as e:
            logger.warning("Skipping input-data preview in backtest report: %s", e)

        # Build payload + HTML
        from ..reporting import build_backtest_payload, build_backtest_report_html

        payload = build_backtest_payload(
            model_id=model_id,
            model_metadata=metadata,
            backtest_result={
                "backtest_metrics": backtest_metrics,
                "comparison_to_validation": comparison,
                "backtest_summary": backtest_summary,
                "predictions": predictions_flat,
                "residual_analysis": residual_analysis,
                "peak_metrics": peak_metrics,
            },
            windows_raw=core["windows_inv"],
            actual_series_inv=core["actual_inv"],
            horizon_metrics=core["horizon_metrics"],
            input_df=input_df,
            column_mapping=resolved_mapping,
            title=title,
            probabilistic=core.get("probabilistic"),
        )

        html = build_backtest_report_html(payload)
        out_path.write_text(html, encoding="utf-8")

        response = create_success_response(
            output_html_path=str(out_path),
            file_size_bytes=int(out_path.stat().st_size),
            n_windows=n_windows,
            n_points=min_len,
            model_id=model_id,
            model_type=model_type,
            backtest_metrics=backtest_metrics,
            comparison_to_validation=comparison,
            backtest_summary=backtest_summary,
        )
        if peak_metrics:
            response["peak_metrics"] = peak_metrics
        if core["ml_warnings"]:
            response["ml_warnings"] = core["ml_warnings"]
        return response

    except Exception as e:
        logger.exception("generate_backtest_report failed")
        return create_error_response(f"Backtest report generation failed: {str(e)}")


def generate_inference_dashboard(
    model_id: str,
    csv_path: str,
    output_html_path: str,
    latitude: float,
    longitude: float,
    timezone: str = "UTC",
    column_mapping: Optional[dict] = None,
    horizon_hours: Optional[int] = None,
    weather_variables: Optional[list] = None,
    weather_preset: str = "building_load",
    title: Optional[str] = None,
) -> dict:
    """
    Generate a self-contained interactive HTML inference dashboard.

    Runs a one-shot forward inference with a trained model, fetches live
    weather data from the Open-Meteo free API for the given location, and
    produces a single HTML file that:

    * Embeds the model predictions (load forecast) as a time series chart.
    * Shows a dashed grey context trace (the recent actuals fed to the model).
    * Fetches live weather from Open-Meteo **in the browser** via the
      "Refresh Weather" button — no server is required after the file is
      generated.
    * Displays temperature, solar radiation, and wind as overlay charts,
      updated on every refresh.
    * Provides an auto-refresh toggle (polls every 5 minutes).

    The dashboard is fully self-contained: all JS (d3 v7, Observable Plot v0.6)
    and CSS are inlined — open it with ``file://`` in any modern browser.

    Workflow:
        1. Supply a context CSV covering at least ``lookback_hours`` of recent
           load readings (target column non-NaN).  For future-covariate models
           (e.g. weather-aware), also include ``horizon_hours`` of future
           covariate rows with NaN target.  Alternatively let the tool fetch
           weather automatically by providing ``latitude`` / ``longitude``.
        2. The tool runs ``generate_forecast`` internally to produce the
           load predictions.
        3. It also fetches weather from Open-Meteo server-side to pre-populate
           the baked-in weather snapshot (shown immediately on file open).
        4. The browser JS re-fetches live weather whenever the user clicks
           "Refresh Weather".

    Args:
        model_id: ID of a trained model (from ``train_forecast_model``).
        csv_path: Path to a recent-context CSV.  Must have at least
            ``lookback_hours`` of non-NaN target rows.  For weather-aware
            models, include future covariate columns for the horizon period.
        output_html_path: Destination path for the generated HTML file
            (e.g. ``"reports/inference_dashboard.html"``).
        latitude: WGS84 latitude of the building (-90 to 90).  Used for
            server-side weather fetch (baked into HTML) and browser-side
            live refresh.
        longitude: WGS84 longitude of the building (-180 to 180).
        timezone: IANA timezone string (e.g. ``"America/Los_Angeles"`` or
            ``"UTC"``).  Applied to both the baked-in weather fetch and the
            live browser refresh.
        column_mapping: Column role mapping.  If not provided, the mapping
            stored in the model metadata from training is used.
        horizon_hours: Forecast horizon override in hours (1–96).  If not
            provided, the model's training horizon is used.
        weather_variables: Explicit list of Open-Meteo variable names to
            fetch (e.g. ``["temperature_2m", "relative_humidity_2m"]``).
            Overrides ``weather_preset`` if provided.
        weather_preset: Named weather preset — ``"minimal"``,
            ``"building_load"`` (default), or ``"extended"``.
        title: Optional dashboard title.  Defaults to
            ``"Load Forecast — <model_id>"``.

    Returns:
        Dict with:
          - ``success`` (bool)
          - ``output_html_path``: path to the generated HTML file
          - ``file_size_bytes``: size of the HTML file
          - ``model_id``, ``model_type``
          - ``forecast_start``, ``forecast_end``: ISO8601 strings
          - ``n_steps``: number of forecast steps
          - ``context_rows``: number of context rows fed to the model
          - ``weather_fetched``: True if server-side weather fetch succeeded
          - ``weather_variables``: list of weather variables in the snapshot
          - ``ml_warnings``: list of soft-warning strings (may be empty)
    """
    try:
        # 1. Load model + config context from registry
        try:
            ctx = _load_model_context(model_id, column_mapping)
        except _ModelLoadError as e:
            return e.response
        model = ctx["model"]
        metadata = ctx["metadata"]
        scalers = ctx["scalers"]
        model_type = ctx["model_type"]
        frequency = ctx["frequency"]
        lookback_hours_cfg = ctx["lookback_hours"]
        horizon_hours_cfg = ctx["horizon_hours"]

        resolved_mapping = ctx["column_mapping"] or {}
        if not resolved_mapping:
            return create_error_response(
                "No column_mapping provided and none stored in model metadata."
            )

        # Effective horizon
        effective_horizon_hours = horizon_hours if horizon_hours is not None else horizon_hours_cfg

        # ------------------------------------------------------------------ #
        # 2. Validate coordinates
        # ------------------------------------------------------------------ #
        if not (-90 <= latitude <= 90):
            return create_error_response(f"latitude must be between -90 and 90, got {latitude}")
        if not (-180 <= longitude <= 180):
            return create_error_response(f"longitude must be between -180 and 180, got {longitude}")

        # ------------------------------------------------------------------ #
        # 3. Parse raw CSV → extract context + (optionally) horizon rows
        # ------------------------------------------------------------------ #
        try:
            raw_df = _parse_raw_csv_for_inference(csv_path, resolved_mapping)
        except DataLoadError as e:
            return create_error_response(f"Failed to read context CSV: {e}")

        target_col   = resolved_mapping.get("target")
        datetime_col = resolved_mapping.get("datetime")
        if not target_col or target_col not in raw_df.columns:
            return create_error_response(
                f"Target column '{target_col}' not found in CSV. "
                "Check column_mapping or CSV column names."
            )

        last_valid_pos = raw_df[target_col].last_valid_index()
        if last_valid_pos is None:
            return create_error_response(
                f"Target column '{target_col}' has no valid (non-NaN) values."
            )
        context_df = raw_df.iloc[: last_valid_pos + 1].copy()

        # Build context series for the chart (actual load values)
        context_series: list[dict] = []
        try:
            dt_vals = (
                context_df[datetime_col].tolist()
                if datetime_col and datetime_col in context_df.columns
                else context_df.index.tolist()
            )
            load_vals = context_df[target_col].tolist()
            steps_per_hour = FREQ_TO_STEPS_PER_HOUR.get(frequency, 1)
            # Show at most 7 days of context in the chart to keep it readable
            max_ctx_rows = 7 * 24 * steps_per_hour
            for dt_v, lv in zip(dt_vals[-max_ctx_rows:], load_vals[-max_ctx_rows:]):
                iso = dt_v.isoformat() if hasattr(dt_v, "isoformat") else str(dt_v)
                context_series.append({
                    "datetime":    iso,
                    "actual_load": float(lv) if lv is not None and not (isinstance(lv, float) and (lv != lv)) else None,
                })
        except Exception as e:
            logger.warning("Could not build context series for dashboard: %s", e)

        # ------------------------------------------------------------------ #
        # 4. Server-side weather fetch (baked into HTML snapshot)
        # ------------------------------------------------------------------ #
        weather_records: list[dict] = []
        weather_fetched = False
        effective_weather_vars: list[str] = []

        try:
            if weather_variables:
                wvars = list(weather_variables)
            elif weather_preset in _WEATHER_PRESETS:
                wvars = _WEATHER_PRESETS[weather_preset]
            else:
                wvars = _WEATHER_PRESETS["building_load"]

            effective_weather_vars = wvars

            # past_hours covers the context window; forecast_hours covers horizon
            past_h     = min(lookback_hours_cfg, 92)
            forecast_h = min(max(effective_horizon_hours, 24), 384)

            weather_df = fetch_weather_forecast_df(
                latitude=latitude,
                longitude=longitude,
                variables=wvars,
                forecast_hours=forecast_h,
                past_hours=past_h,
                timezone=timezone,
                resolution="hourly",
                rename_columns=True,
            )

            weather_records = _df_to_records(weather_df, "datetime")

            weather_fetched = True
            logger.info(
                "generate_inference_dashboard: fetched %d weather rows "
                "(lat=%.4f lon=%.4f tz=%s)",
                len(weather_records), latitude, longitude, timezone,
            )

        except ImportError:
            logger.warning("httpx not installed — skipping server-side weather fetch")
        except Exception as e:
            logger.warning("Server-side weather fetch failed (non-fatal): %s", e)

        # ------------------------------------------------------------------ #
        # 5. Run inference (reuse generate_forecast logic)
        # ------------------------------------------------------------------ #
        forecast_result = generate_forecast(
            model_id=model_id,
            csv_path=csv_path,
            column_mapping=resolved_mapping,
            horizon_hours=horizon_hours,
        )

        if not forecast_result.get("success"):
            return create_error_response(
                f"Inference failed: {forecast_result.get('error', 'unknown error')}"
            )

        forecast_steps: list[dict] = forecast_result.get("predictions", [])
        forecast_start = forecast_result.get("forecast_start")
        forecast_end   = forecast_result.get("forecast_end")
        ml_warnings    = forecast_result.get("ml_warnings", [])

        # ------------------------------------------------------------------ #
        # 6. Build + write HTML
        # ------------------------------------------------------------------ #
        from ..reporting import build_inference_dashboard_html

        effective_title = title or f"Load Forecast — {model_id}"

        html = build_inference_dashboard_html(
            model_id=model_id,
            model_metadata=metadata,
            forecast=forecast_steps,
            context_series=context_series,
            weather_forecast=weather_records,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
            title=effective_title,
            quantiles=forecast_result.get("quantiles"),
            num_samples=forecast_result.get("num_samples"),
        )

        out_path = Path(output_html_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        file_size = out_path.stat().st_size

        logger.info(
            "generate_inference_dashboard: wrote %d bytes to %s (%d forecast steps)",
            file_size, out_path, len(forecast_steps),
        )

        response = create_success_response(
            output_html_path=str(out_path),
            file_size_bytes=int(file_size),
            model_id=model_id,
            model_type=model_type,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            n_steps=len(forecast_steps),
            context_rows=len(context_df),
            weather_fetched=weather_fetched,
            weather_variables=effective_weather_vars,
            latitude=latitude,
            longitude=longitude,
            timezone=timezone,
        )
        if ml_warnings:
            response["ml_warnings"] = ml_warnings
        return response

    except Exception as e:
        logger.exception("generate_inference_dashboard failed")
        return create_error_response(f"Inference dashboard generation failed: {str(e)}")


def generate_data_report(
    csv_path: str,
    output_html_path: Optional[str] = None,
    column_mapping: Optional[dict] = None,
    frequency: Optional[str] = None,
    split_strategy: str = "seasonal",
    validation_split: float = 0.2,
    covariates: Optional[list] = None,
    title: Optional[str] = None,
    max_points: int = 60000,
) -> dict:
    """
    Generate a self-contained HTML report that visualises a load CSV before
    training: the target and every covariate over time, the train/validation
    split the trainer would use, load profiles, a day x hour heatmap,
    covariate-vs-load relationships, and data-quality findings.

    Use this when the user wants to *see* the data (rather than the numeric
    inspect_data summary), to check a seasonal split, to compare covariates,
    or to judge whether flagged outliers are real events. The page bundles
    every dependency inline and opens from file:// with no network.

    Args:
        csv_path: Path to the CSV (absolute, or relative to the data roots).
        output_html_path: Where to write the .html report. Optional: when
            omitted the report is written as <csv stem>_data_report.html
            under LOAD_FORECASTING_OUTPUT_DIR (if set) or <repo>/outputs/reports,
            and the path is returned. Give an absolute path to control it.
        column_mapping: Optional explicit roles (datetime, target,
            past_covariates, future_covariates). Auto-detected if omitted;
            numeric columns the mapping does not recognise are still plotted
            and labelled "unmapped".
        frequency: Data frequency ('15min', '30min', 'h'); inferred if omitted.
        split_strategy: How to draw the split: "seasonal" (per meteorological
            season, as train_forecast_model uses for deep models; falls back
            to sequential when fewer than four seasons are present),
            "sequential" (last fraction by time), or "none".
        validation_split: Fraction held out (per season for "seasonal").
        covariates: Subset of columns to draw as covariate panels. Default:
            every numeric column except the target.
        title: Report title.
        max_points: Overview series longer than this are stride-downsampled
            for the browser; all statistics still use the full data.

    Returns:
        Dict with output_html_path, file_size_bytes, n_rows, frequency,
        target, covariates (with roles), split (strategy + segments),
        counts of gaps/outliers, and the inspect_data readiness verdict.
    """
    try:
        from ..reporting import (
            build_data_report_payload,
            build_data_report_html,
            SPLIT_STRATEGIES,
        )
        from .inspection import inspect_data

        if split_strategy not in SPLIT_STRATEGIES:
            return create_error_response(
                f"split_strategy must be one of {list(SPLIT_STRATEGIES)}, got {split_strategy!r}"
            )
        if not 0.05 <= float(validation_split) <= 0.5:
            return create_error_response("validation_split must be between 0.05 and 0.5")
        if not output_html_path:
            from ..core.paths import default_output_dir, resolve_data_path
            stem = resolve_data_path(csv_path).stem or "data"
            output_html_path = str(default_output_dir() / "reports" / f"{stem}_data_report.html")
        err = _validate_html_output_path(output_html_path)
        if err:
            return create_error_response(err)
        out_path = Path(output_html_path)

        # Numeric profile first: it also resolves the frequency and mapping.
        inspection = inspect_data(csv_path, column_mapping=column_mapping, frequency=frequency)
        if not inspection.get("success"):
            return create_error_response(f"inspect_data failed: {inspection.get('error')}")
        resolved_frequency = frequency or (inspection.get("frequency") or {}).get("inferred") or "h"

        try:
            loader = ForecastingDataLoader(
                csv_path=csv_path,
                column_mapping=column_mapping,
                frequency=resolved_frequency,
                add_calendar_features=False,
                lag_hours=[],
            )
        except DataLoadError as e:
            return create_error_response(str(e))

        payload = build_data_report_payload(
            loader,
            csv_path=str(loader.csv_path),
            covariates=covariates,
            split_strategy=split_strategy,
            validation_split=float(validation_split),
            title=title,
            max_points=int(max_points),
            inspection=inspection,
        )
        html = build_data_report_html(payload)
        out_path.write_text(html, encoding="utf-8")

        return create_success_response(
            output_html_path=str(out_path.resolve()),
            file_size_bytes=int(out_path.stat().st_size),
            csv_path=payload.meta["csv_path"],
            n_rows=payload.meta["n_rows"],
            frequency=payload.meta["frequency"],
            time_range={"start": payload.meta["start"], "end": payload.meta["end"], "days": payload.meta["days"]},
            target=payload.meta["target"],
            covariates=payload.meta["covariates"],
            unmapped_covariates=payload.meta["unmapped_covariates"],
            split={
                "requested": payload.split["requested"],
                "strategy": payload.split["strategy"],
                "validation_split": payload.split["validation_split"],
                "summary": payload.split["summary"],
                "segments": payload.split["segments"],
                "note": payload.split["note"],
            },
            peak_windows=payload.windows,
            n_gaps=len(payload.gaps),
            n_value_outliers=len(payload.outliers.get("value", [])),
            n_step_changes=len(payload.outliers.get("spikes", [])),
            quality_flags=inspection.get("quality_flags", []),
            ready_to_train=inspection.get("ready_to_train"),
            blocking_issues=inspection.get("blocking_issues", []),
            sections=[
                "summary", "time series (with window buttons)", "train/validation split",
                "covariates", "load profiles", "day x hour heatmap",
                "covariate relationships", "data quality",
            ],
        )
    except ValueError as e:
        return create_error_response(str(e))
    except Exception as e:
        logger.exception("generate_data_report failed")
        return create_error_response(f"Data report generation failed: {e}")
