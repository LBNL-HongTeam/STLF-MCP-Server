"""
MCP tools: batch multi-series forecasting.

The single-series tools (``train_forecast_model`` / ``generate_forecast``)
handle one target series per call.  A utility with many feeders, substations,
or buildings needs to fan out across dozens or hundreds of series.  These
batch tools do exactly that — they loop over a list of jobs (or the target
columns of one wide CSV), calling the underlying single-series tool once per
series and collecting the results.

Design notes:
  * **Fault tolerant.** A failure on one series is captured in that series'
    result and does NOT abort the batch.  The batch always returns a summary
    with ``succeeded`` / ``failed`` counts so an operator can retry only the
    failures.
  * **Thin wrappers.** All the heavy lifting (data loading, scaling, model
    dispatch, probabilistic handling) stays in the single-series tools; these
    functions only orchestrate.  That keeps behaviour identical to a manual
    per-series call and avoids duplicating the ML guardrails.
  * **Wide-CSV convenience.** ``batch_train_forecast_models`` can expand one
    CSV with several load columns into one training job per column, sharing a
    common datetime (and optional covariate) mapping — the typical shape of an
    AMI export where each column is a meter/feeder.
"""

from typing import Optional
import logging

import pandas as pd

from ..core.data_loader import DataLoadError
from ..core.paths import resolve_data_path
from ._common import create_success_response, create_error_response
from .train import train_forecast_model
from .inference import generate_forecast

logger = logging.getLogger(__name__)


def _summarize(results: list) -> dict:
    """Build the succeeded/failed counts summary block from per-series results."""
    succeeded = sum(1 for r in results if r.get("success"))
    failed = len(results) - succeeded
    return {
        "total": len(results),
        "succeeded": succeeded,
        "failed": failed,
        "failed_series": [
            r.get("series_id") for r in results if not r.get("success")
        ],
    }


def _expand_wide_csv_jobs(
    csv_path: str,
    target_columns: list,
    datetime_col: Optional[str],
    shared_past_covariates: Optional[list],
    shared_future_covariates: Optional[list],
) -> list:
    """Turn one wide CSV + a list of target columns into per-series job dicts.

    Each job points at the same ``csv_path`` but carries a distinct
    ``column_mapping`` selecting one target column (plus the shared datetime
    and covariate columns).  ``series_id`` defaults to the target column name.

    Raises:
        DataLoadError: If the CSV cannot be read or a requested column is
            missing.
    """
    try:
        header = pd.read_csv(resolve_data_path(csv_path), nrows=1)
    except Exception as e:
        raise DataLoadError(f"Failed to read CSV '{csv_path}': {e}")

    available = set(header.columns)
    missing = [c for c in target_columns if c not in available]
    if missing:
        raise DataLoadError(
            f"target_columns not found in CSV: {missing}. "
            f"Available columns: {sorted(available)}"
        )
    if datetime_col and datetime_col not in available:
        raise DataLoadError(
            f"datetime column '{datetime_col}' not found in CSV."
        )

    jobs = []
    for col in target_columns:
        mapping = {"target": col}
        if datetime_col:
            mapping["datetime"] = datetime_col
        if shared_past_covariates:
            mapping["past_covariates"] = list(shared_past_covariates)
        if shared_future_covariates:
            mapping["future_covariates"] = list(shared_future_covariates)
        jobs.append(
            {
                "csv_path": csv_path,
                "series_id": col,
                "building_name": col,
                "column_mapping": mapping,
            }
        )
    return jobs


def batch_train_forecast_models(
    jobs: Optional[list] = None,
    csv_path: Optional[str] = None,
    target_columns: Optional[list] = None,
    datetime_col: Optional[str] = None,
    shared_past_covariates: Optional[list] = None,
    shared_future_covariates: Optional[list] = None,
    model_type: str = "LinearRegression",
    lookback_hours: int = 24,
    horizon_hours: int = 6,
    frequency: str = "h",
    validation_split: float = 0.2,
    probabilistic: bool = False,
    quantiles: Optional[list] = None,
    device: Optional[str] = None,
    continue_on_error: bool = True,
) -> dict:
    """Train one forecasting model per series across many series.

    Two ways to specify the series:

    1. **Job list** — pass ``jobs`` as a list of dicts, one per series.  Each
       job supports the keys: ``csv_path`` (required), ``series_id`` (label),
       ``building_name``, ``column_mapping``, and any per-job override of
       ``model_type``, ``lookback_hours``, ``horizon_hours``, ``frequency``,
       ``validation_split``, ``probabilistic``, ``quantiles``.  Missing keys
       fall back to the batch-level defaults (the other arguments below).

    2. **Wide CSV** — pass ``csv_path`` + ``target_columns`` to train one model
       per target column of a single CSV (typical AMI export where each column
       is a meter/feeder).  ``datetime_col`` and the ``shared_*_covariates``
       lists are applied to every column.

    The batch is fault tolerant: by default (``continue_on_error=True``) a
    failure on one series is recorded and the batch proceeds.  Set it False to
    stop at the first failure.

    Args:
        jobs: Explicit per-series job dicts (mode 1).  Mutually exclusive with
            ``target_columns``.
        csv_path: Path to a wide CSV (mode 2).
        target_columns: Target column names to expand into jobs (mode 2).
        datetime_col: Shared datetime column for the wide CSV (mode 2).
        shared_past_covariates: Past-covariate columns shared by all wide-CSV
            series (mode 2).
        shared_future_covariates: Future-covariate columns shared by all
            wide-CSV series (mode 2).
        model_type: Default model type for every job.
        lookback_hours: Default lookback window (hours).
        horizon_hours: Default forecast horizon (hours).
        frequency: Default data frequency (15min/30min/h).
        validation_split: Default validation fraction.
        probabilistic: Default probabilistic flag (quantile forecasting).
        quantiles: Default quantile levels when probabilistic.
        device: Default compute device for torch models.
        continue_on_error: If True, keep going after a failed series.

    Returns:
        Dict with ``summary`` (total/succeeded/failed) and ``results`` — a list
        of per-series dicts each carrying ``series_id``, ``success``, and
        either ``model_id``/metrics or an ``error`` string.
    """
    # ------------------------------------------------------------------
    # Resolve the job list from whichever mode the caller used.
    # ------------------------------------------------------------------
    if jobs and target_columns:
        return create_error_response(
            "Provide either `jobs` (job-list mode) or `csv_path`+`target_columns` "
            "(wide-CSV mode), not both."
        )

    if not jobs:
        if not (csv_path and target_columns):
            return create_error_response(
                "No series specified. Provide `jobs`, or `csv_path` + "
                "`target_columns`."
            )
        try:
            jobs = _expand_wide_csv_jobs(
                csv_path=csv_path,
                target_columns=target_columns,
                datetime_col=datetime_col,
                shared_past_covariates=shared_past_covariates,
                shared_future_covariates=shared_future_covariates,
            )
        except DataLoadError as e:
            return create_error_response(str(e))

    if not isinstance(jobs, list) or not jobs:
        return create_error_response("`jobs` must be a non-empty list of dicts.")

    # ------------------------------------------------------------------
    # Run each job through the single-series trainer.
    # ------------------------------------------------------------------
    results = []
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            results.append(
                {"series_id": f"job_{i}", "success": False,
                 "error": "job entry must be a dict"}
            )
            if not continue_on_error:
                break
            continue

        series_id = job.get("series_id") or job.get("building_name") or f"job_{i}"
        job_csv = job.get("csv_path", csv_path)
        if not job_csv:
            results.append(
                {"series_id": series_id, "success": False,
                 "error": "job is missing required `csv_path`"}
            )
            if not continue_on_error:
                break
            continue

        try:
            res = train_forecast_model(
                csv_path=job_csv,
                model_type=job.get("model_type", model_type),
                lookback_hours=job.get("lookback_hours", lookback_hours),
                horizon_hours=job.get("horizon_hours", horizon_hours),
                frequency=job.get("frequency", frequency),
                validation_split=job.get("validation_split", validation_split),
                building_name=job.get("building_name", series_id),
                model_name=job.get("model_name"),
                column_mapping=job.get("column_mapping"),
                probabilistic=job.get("probabilistic", probabilistic),
                quantiles=job.get("quantiles", quantiles),
                device=job.get("device", device),
            )
        except Exception as e:  # defensive — single-series tool already guards
            logger.exception("batch_train job %s crashed", series_id)
            res = create_error_response(f"Unhandled error: {e}")

        entry = {
            "series_id": series_id,
            "success": res.get("success", False),
        }
        if res.get("success"):
            entry.update(
                {
                    "model_id": res.get("model_id"),
                    "model_type": res.get("model_type"),
                    "validation_metrics": res.get("validation_metrics"),
                }
            )
        else:
            entry["error"] = res.get("error")
        results.append(entry)

        if not entry["success"] and not continue_on_error:
            break

    return create_success_response(
        summary=_summarize(results),
        results=results,
    )


def batch_generate_forecast(
    jobs: list,
    horizon_hours: Optional[int] = None,
    num_samples: int = 200,
    output_dir: Optional[str] = None,
    continue_on_error: bool = True,
) -> dict:
    """Generate forward forecasts for many trained models in one call.

    Each job is a dict with:
        * ``model_id`` (required) — a trained model to run.
        * ``csv_path`` (required) — recent-context CSV for that model.
        * ``series_id`` (optional) — label for the result (defaults to
          ``model_id``).
        * ``column_mapping`` (optional) — override the model's stored mapping.
        * ``horizon_hours`` (optional) — per-job horizon override.
        * ``output_csv_path`` (optional) — write that series' predictions.

    Args:
        jobs: List of per-series forecast job dicts (see above).
        horizon_hours: Default horizon override applied to every job that does
            not set its own.
        num_samples: Monte-Carlo samples for probabilistic models (default 200).
        output_dir: If given, each series' predictions are written to
            ``<output_dir>/<series_id>.csv`` unless the job sets its own
            ``output_csv_path``.
        continue_on_error: If True, keep going after a failed series.

    Returns:
        Dict with ``summary`` and ``results`` — one entry per series carrying
        ``series_id``, ``success``, and either the forecast payload (predictions,
        forecast_start/end, probabilistic band info) or an ``error`` string.
    """
    if not isinstance(jobs, list) or not jobs:
        return create_error_response("`jobs` must be a non-empty list of dicts.")

    import os

    results = []
    for i, job in enumerate(jobs):
        if not isinstance(job, dict):
            results.append(
                {"series_id": f"job_{i}", "success": False,
                 "error": "job entry must be a dict"}
            )
            if not continue_on_error:
                break
            continue

        model_id = job.get("model_id")
        job_csv = job.get("csv_path")
        series_id = job.get("series_id") or model_id or f"job_{i}"

        if not model_id or not job_csv:
            results.append(
                {"series_id": series_id, "success": False,
                 "error": "job requires both `model_id` and `csv_path`"}
            )
            if not continue_on_error:
                break
            continue

        out_csv = job.get("output_csv_path")
        if out_csv is None and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_csv = os.path.join(output_dir, f"{series_id}.csv")

        try:
            res = generate_forecast(
                model_id=model_id,
                csv_path=job_csv,
                column_mapping=job.get("column_mapping"),
                horizon_hours=job.get("horizon_hours", horizon_hours),
                output_csv_path=out_csv,
                num_samples=num_samples,
            )
        except Exception as e:
            logger.exception("batch_generate_forecast job %s crashed", series_id)
            res = create_error_response(f"Unhandled error: {e}")

        entry = {"series_id": series_id, "success": res.get("success", False)}
        if res.get("success"):
            entry.update(
                {
                    "model_id": res.get("model_id"),
                    "forecast_start": res.get("forecast_start"),
                    "forecast_end": res.get("forecast_end"),
                    "predictions": res.get("predictions"),
                    "probabilistic": res.get("probabilistic", False),
                    "quantiles": res.get("quantiles"),
                    "output_csv_path": res.get("output_csv_path"),
                }
            )
        else:
            entry["error"] = res.get("error")
        results.append(entry)

        if not entry["success"] and not continue_on_error:
            break

    return create_success_response(
        summary=_summarize(results),
        results=results,
    )
