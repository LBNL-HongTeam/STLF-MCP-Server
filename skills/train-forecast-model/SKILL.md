---
name: train-forecast-model
description: Use when the user wants to build, train, tune, evaluate, backtest, or benchmark an electrical load forecasting model on this STLF-MCP-Server project. Triggers on requests mentioning load forecasting, kWh/kW prediction, training a model on AMI/meter/building data, tuning hyperparameters with Optuna, backtesting, peak-day metrics, merging weather with load, or comparing algorithms (LinearRegression, XGBoost, LSTM, ARIMA, TFT, TiDE, TSMixer, TimesFM). Enforces the inspect -> (merge) -> train -> evaluate -> backtest -> report workflow using the STLF-MCP-Server MCP tools.
---

# train-forecast-model

Enforces a best-practice, end-to-end workflow for building electrical load forecasting models via the STLF-MCP-Server MCP tools. Do NOT skip steps. Do NOT invent shortcuts. Every mandatory gate must be executed and its result surfaced to the user before proceeding.

The MCP tools do the heavy lifting — this skill's job is to sequence them correctly, catch silent-failure modes, and confirm assumptions with the user before spending compute.

---

## Section 0 — MCP connection is mandatory (hard-stop, non-negotiable)

Every step of this workflow depends on the STLF-MCP-Server MCP tools. If the connection to the MCP server is lost — or any tool call fails because the server is unreachable, disconnected, timed out, or otherwise unavailable — at ANY point in the process:

- **STOP immediately.** Do not continue to the next step.
- **Do NOT attempt to do the work manually** (e.g., reading/parsing CSVs yourself, hand-computing metrics, training models with local Python/scripts, merging covariates by hand, or generating reports without the tools). The whole point of this skill is that the MCP tools own the logic; substituting manual work produces results that silently diverge from the framework's behaviour and defeats the workflow's guarantees.
- **Do NOT fabricate or estimate** any tool output (model IDs, metrics, inspection results, merge summaries, etc.).
- **Report the failure to the user verbatim** — state which tool call failed, at which step, and the raw error/connection message.
- **Wait for the user** to restore the MCP server connection and explicitly instruct you to resume. Only then continue from the failed step.

This rule overrides every other instruction in this skill. There is no fallback path around a lost MCP connection.

---

## Section 1 — Preconditions

### 1.0 — Timezone-first gate (hard-stop, ask immediately, never guess)

**The moment the user provides a CSV path, the FIRST thing you confirm is the timezone of its timestamps — before any other precondition and before any tool call (including `inspect_data`).**

- **Never guess, assume, or infer the timezone.** Do NOT derive it from the filename, the column values, a `+00:00`/`Z`/offset suffix, the building's presumed location, the user's locale, or any other context. A CSV almost never carries authoritative timezone metadata; naive timestamps are inherently ambiguous, and an inference that "looks right" is the #1 source of silent bad merges and misaligned weather covariates.
- If the user has not **explicitly stated** the timezone, **STOP and ask** before doing anything else. Ask for every CSV provided (see §2.5.4 for the multi-file table).
- Ask two things per file: (1) the IANA timezone (e.g. `America/Los_Angeles`, `UTC`), and (2) whether the timestamps are naive (local-time strings) or tz-aware (carry an explicit offset).
- Only after the user answers do you proceed to the rest of §1 and to §2 / §2.5.

This gate overrides the ordering of the numbered list below.

### 1.1 — Remaining preconditions

Before any tool call, confirm with the user:

1. **Timezone.** (Already handled by the §1.0 gate — must be explicitly answered by the user, never guessed.) What timezone are the load timestamps in? Are they naive (local time strings) or tz-aware (with `+00:00` etc.)? Timezone confusion is the #1 source of silent bad merges.
2. **CSV path(s).** How many input files? Where are they? If more than one, jump to §2.5 (multi-CSV merge workflow); the single-file path in §2 applies only when exactly one CSV was provided.
   - If the user has **no data yet**, or asks what is available, call `list_datasets` (no arguments) and present the bundled sample datasets with their absolute `path` values. Never conclude that no data exists without calling it. The §1.0 timezone gate still applies to a bundled file — ask, do not infer.
   - Pass `csv_path` as the absolute `path` returned by `list_datasets` whenever possible. Relative paths are resolved against the server's data roots, not the client's working directory.
3. **Target semantics.** Is the target column energy (kWh per interval) or power (kW instantaneous)? This affects interpretation of metrics and peak-day analysis.
4. **Data cadence.** 15-min, 30-min, or hourly? Must match one of `{"15min", "30min", "h"}`. Anything else is unsupported and will hard-block.
5. **Model choice.** Do NOT silently default to `LinearRegression`. Show the decision matrix in §3 and let the user pick, or ask for their constraints (speed vs accuracy, covariate availability, deployment target).
6. **Covariate availability.** Does the user have weather data? Occupancy? Holidays? Weather covariates are the strongest predictor of building load; if the user chose a covariate-supporting model but has no weather data, §5a will force a confirmation before training.

---

## Section 2 — Mandatory `inspect_data` (single-file case only)

Skip this section if two or more CSVs were provided — go directly to §2.5.

**Precondition:** the §1.0 timezone-first gate must already be satisfied — the user must have explicitly stated the file's timezone (naive vs tz-aware). If they have not, STOP and ask before running `inspect_data`. Never guess it from the data.

Run `inspect_data` on the CSV before any training call, even though it is auto-invoked inside `train_forecast_model`. If the user wants to *see* the data, or a quality flag needs judgement (are those outliers real events or bad data?), also call `generate_data_report` and give the user the returned HTML path; it draws the same split the trainer will use, so a seasonal-split question is answered by its split timeline. Rationale:

- Catches blocking issues (`LOAD_ERROR`, `UNSUPPORTED_FREQUENCY`) before wasting a training run.
- Confirms the auto-detected column mapping (datetime, target, past covariates).
- Surfaces coverage/gap/outlier/DST quality flags.
- Reports `time_range.start` / `time_range.end` needed for the length and seasonal-split checks below.

Show the user: `quality_flags`, `suggestions`, `ready_to_train`, `blocking_issues`, detected `columns`, and inferred `frequency`. **Refuse to proceed if `ready_to_train == False`.**

### 2a — Data-length check

Compute `(time_range.end - time_range.start)` in days from the `inspect_data` response.

- **If < 365 days**, warn the user verbatim (or paraphrased):

  > "Training data covers only {N} days (< 1 full year). Load-forecasting models learn strong seasonal patterns (summer/winter cooling vs heating, DST transitions, holidays). With less than a full year of data, the model will not have seen a complete seasonal cycle and generalization to unseen seasons will be poor. Recommendation: gather ≥ 365 days if possible, or restrict evaluation to the same season(s) present in training."

  Require explicit user confirmation to proceed.

- **If ≥ 365 days**, no warning; proceed to 2b.

### 2b — Seasonal-split note (data span ≥ 365 days)

`ForecastingDataLoader.split_train_val_seasonal` partitions the data into four meteorological seasons (DJF, MAM, JJA, SON) and takes the first 80% of each season chronologically for training, the last 20% for validation. It falls back silently to sequential split if any season is absent or too short.

Action items:

1. Set `validation_split=0.2` (default). The tool constrains this to `[0.1, 0.3]`.
2. **After training**, verify from `data_summary`:
   - `training_samples + validation_samples ≈ total_samples`
   - `training_samples / total_samples ≈ 0.8` (within a few percent)
   - If the ratio is off, the loader silently fell back to sequential split — warn the user that stratification did not occur, likely because a season is under-represented despite the ≥365-day span (large gaps, or start/end mid-season).

**Do not** attempt random-within-season sampling — the framework does not support it and doing so requires a code change (would introduce temporal leakage anyway).

Now proceed to §3 (model selection).

---

## Section 2.5 — Multi-CSV merge workflow (N ≥ 2 input files)

`merge_covariates` accepts exactly one primary + one covariate CSV per call. For N ≥ 3 files, orchestrate N-1 sequential pairwise merges. Do NOT try to feed multiple CSVs directly to `train_forecast_model` — it only accepts one.

**Do not manually pre-generate `cal_hour`, `cal_dow`, `cal_month`, `cal_is_weekend`, `cal_hour_sin/cos`, `cal_dow_sin/cos`, `cal_month_sin/cos`, `lag_24h`, `lag_48h`, or `lag_168h`.** `ForecastingDataLoader` injects all of these automatically as past covariates during `train_forecast_model` / `tune_model` / `evaluate_forecast_model`. Duplicating them wastes columns and, for lag features, risks temporal-leakage bugs.

### 2.5.0 — Enumerate and classify every input file

Ask the user to identify each CSV's role. Exactly one must be the primary. Refuse to proceed if the primary is ambiguous or missing.

| Role | Description | Example |
|---|---|---|
| **Primary (load)** | Contains the target column. Exactly one. | building AMI meter data |
| **Historical covariate** | Time-aligned features covering the training period. | historical weather, historical occupancy |
| **Forecast covariate** | Features known at forecast time; used as `future_covariates`. | Open-Meteo weather forecast |
| **Static / calendar** | Time-indexed flag columns. | `is_holiday` calendar |

### 2.5.1 — Pre-merge inspection of EVERY file

Run `inspect_data` on each CSV independently. For each file, record: `frequency.inferred`, `time_range.start`, `time_range.end`, `time_range.n_duplicates_removed`, `columns.all_columns`, `gaps.n_gaps`, `quality_flags`. Present the per-file summary to the user.

### 2.5.2 — Column-name collision check across ALL files

Cross-reference every non-datetime column name across all N files. If any column name appears in ≥2 files (e.g., two files both have `temperature`), hard-stop. `pandas.merge` will silently suffix them `_x` / `_y` and the auto-detect patterns will pick the wrong one.

Resolution: ask the user to either (a) rename the column in one file before proceeding, or (b) pass an explicit `covariate_columns=[...]` subset on the merge call that excludes the colliding name from one side.

### 2.5.3 — Determine common frequency

- Default target cadence: the **finest** cadence among all files (e.g., load 15-min + weather 1-h → target = 15-min; hourly weather ffills across 15-min rows).
- If any covariate is **coarser** than the load file: warn — coarse covariate values will be forward-filled across finer load rows.
- If any covariate is **finer** than the load file: **hard-stop**. `merge_covariates` LEFT-JOINs on exact timestamp match; sub-load-cadence rows are silently dropped. The user must aggregate that file to the load cadence outside MCP before merging.

### 2.5.4 — Timezone confirmation for every file

Ask the user for the timezone of every file independently — **never guess or infer it** from the filename, column values, offset suffixes, or the data source's presumed location (see §1.0). If the timezone of any file has not been explicitly stated by the user, STOP and ask before merging. Build a per-file table. Any file NOT in the load timezone will require `covariate_timezone=<file-tz>` + `load_timezone=<load-tz>` on its merge call.

Decision matrix per-file:

| Load tz | Covariate tz | Action |
|---|---|---|
| Same, both naive | Same, both naive | Merge without tz params. |
| Different (any combination) | | Pass `covariate_timezone=<cov-tz>`, `load_timezone=<load-tz>`. Tool converts + handles DST fall-back dedup. |
| Either is tz-aware in the CSV | | Hard-stop. `merge_covariates` cannot merge tz-aware timestamps (raises "Already tz-aware"). User must strip tz info first. |
| Load naive, Open-Meteo weather CSV | | Standard: `covariate_timezone="UTC"`, `load_timezone=<user's local tz>`. |

### 2.5.5 — Global overlap check across ALL files

Compute:

- `overlap_start = max(file.start for file in all_files)`
- `overlap_end = min(file.end for file in all_files)`
- `overlap_fraction = (overlap_end - overlap_start) / (load.end - load.start)`

Thresholds:

- `overlap_fraction < 0.95` → warn. Significant portions of the load timeline will be ffill/bfill-imputed from stale covariate values.
- Any covariate `start > load.start` → early load rows will be bfilled with a constant future value; consider truncating load to `overlap_start`.
- Any covariate `end < load.end` → late load rows ffilled indefinitely with the last covariate value; consider truncating load to `overlap_end`.

### 2.5.6 — Pre-flight duplicate check

If any file's `time_range.n_duplicates_removed > 0`, hard-stop. `merge_covariates` does NOT dedup the load side; load-side duplicates cause row-multiplication in the LEFT JOIN. `inspect_data`'s dedup is a report, not a fix. User must clean externally.

### 2.5.7 — Merge order and naming strategy

Sequential pairwise merges. Recommended order:

1. Start with the **primary load CSV** as the current base.
2. Merge covariates from **longest coverage → shortest coverage**. Rationale: keeps ffill artifacts from the most-truncated covariate isolated at the end where diagnostics are easiest to read.
3. Merge same-timezone-as-load files first; tz-converted files last (per-merge tz diagnostics stay clean).

Intermediate file naming (do NOT overwrite originals):

```
load.csv + weather.csv                           -> load__weather.csv
load__weather.csv + occupancy.csv                -> load__weather__occupancy.csv
load__weather__occupancy.csv + holidays.csv      -> merged_final.csv
```

Write intermediates to a temp directory or alongside the final output.

### 2.5.8 — Per-merge verification (repeat for every pairwise call)

After each `merge_covariates` call:

| Check | Threshold | Action |
|---|---|---|
| `n_missing_filled / n_rows` | > 5 % | Warn, continue, flag in cumulative summary. |
| `n_missing_filled / n_rows` | > 20 % | **Hard-stop.** Do not proceed to next merge. Report which covariate caused it. Fix coverage and re-run. |
| `dst_duplicates_dropped` | > 4 per year of data span | Warn: unusually high; tz spec may be wrong. |
| `dst_duplicates_dropped` == 0 AND tz conversion was requested AND data spans a DST transition | | Warn: expected ≥1. TZ setup may not be doing what the user thinks. |
| `n_rows` returned | ≠ current base row count | **Hard-stop.** Base had duplicate timestamps (should have been caught by 2.5.6). |
| `covariate_columns` returned | doesn't match requested subset | Warn: auto-detection picked the wrong columns. |

Always pass **explicit** `load_datetime_col`, `covariate_datetime_col`, and `covariate_columns` on every merge call. Never rely on substring auto-detection in a multi-CSV workflow — it picks columns like `end_time` / `update_time` before the real datetime column.

### 2.5.9 — Cumulative post-merge report

After all N-1 merges succeed, present a consolidated table to the user:

| Covariate file | Columns added | n_missing_filled | % filled | dst_duplicates_dropped | Tz conversion |
|---|---|---|---|---|---|
| weather.csv | temperature, humidity | 12 | 0.1 % | 1 | UTC → America/LA |
| occupancy.csv | occupancy_pct | 340 | 3.2 % | 0 | — |
| holidays.csv | is_holiday | 0 | 0.0 % | 0 | — |

### 2.5.10 — Final `inspect_data` on the fully merged file

Non-negotiable. Run `inspect_data` on `merged_final.csv`. Verify:

- `ready_to_train == True`
- All expected covariate columns present under `columns.past_covariates` or `columns.unrecognised`
- No `_x` / `_y` suffixed columns (collision-free merge confirmed)
- Apply the §2a data-length check and §2b seasonal-split awareness on the merged span

Now proceed to §3.

### Known limitations of `merge_covariates`

- Pairwise-only tool — for N ≥ 3 files, orchestrate N-1 sequential calls.
- No cross-file collision detection — same column name in two files silently gets `_x`/`_y` suffixes.
- No sub-load-cadence aggregation — finer-cadence covariate files must be aggregated externally.
- No resampling of any kind — coarse covariates get ffill'd across finer load rows silently.
- No sort — unsorted inputs produce garbage ffill.
- No load-side dedup — load duplicates cause row-multiplication.
- Coverage report is a single `n_missing_filled` scalar — conflates "1 row missing" with "3 months missing".
- Substring auto-detection of datetime column picks the first-matching column in file order.
- Cannot merge tz-aware timestamps — fails with a generic error.
- `covariate_columns=None` includes every non-datetime column, including junk.

---

## Section 3 — Model decision matrix

If the user has not stated a specific model, present this matrix and ask them to choose. Do NOT default silently.

| Model | past_cov | future_cov | Speed | Notes |
|---|:---:|:---:|:---:|---|
| NaiveMean / NaiveSeasonal / NaiveMovingAverage | – | – | Slow (refit every stride) | Sanity baseline only. Use for benchmarking, not deployment. |
| LinearRegression | ✅ | ✅ | Fast | Strong baseline. Always run this first for comparison. |
| XGBoost | ✅ | ✅ | Fast | Nonlinear baseline. Usually best cost/accuracy tradeoff. |
| LSTM (BlockRNNModel) | ✅ | ❌ silently ignored | Slow (GPU strongly preferred — see §3.5) | Past-cov only despite mixed-cov API. |
| ARIMA | ❌ | ✅ (as exog) | Medium | **Ignores `lookback_hours`** — use `model_kwargs={"p": ...}` or tune p/d/q. |
| TFT | ✅ | ✅ | Slow (GPU strongly preferred — see §3.5) | Attention-based. Needs data and tuning to shine. |
| TiDE | ✅ | ✅ | Slow (GPU strongly preferred — see §3.5) | Modern MLP; per Li et al. 2025 Table A.3 defaults. |
| TSMixer | ✅ | ✅ | Slow (GPU strongly preferred — see §3.5) | Modern MLP; per Li et al. 2025 Table A.3 defaults. |
| TimesFM | ❌ | ❌ | Medium (800 MB one-time download; GPU strongly preferred — see §3.5) | Zero-shot foundation model. Silently ignores all covariates. Hard cap: horizon ≤ 128 steps, lookback+horizon ≤ 16384 steps. |
| TimesFM+Residual | ✅ | ✅ | Very slow (per-stride TimesFM forward pass; GPU strongly preferred — see §3.5) | Hybrid: TimesFM target + Ridge residual on covariates. Not tunable via Optuna interface. |

**Deep-learning models (LSTM, TFT, TiDE, TSMixer, TimesFM, TimesFM+Residual) require the §3.5 GPU pre-flight before any train/tune call.** CPU training for these models is only permitted after an explicit user-consent gate (§3.5.3). Any GPU-related training error must be triaged via §3.5.4 — silent CPU retry is forbidden.

### Hard constraints

- Minimum training rows: `lookback_steps + horizon_steps + 100`. Enforced by the trainer.
- `lookback_hours`: 1–168.
- `horizon_hours`: 1–96 for `train_forecast_model`, `generate_forecast`, `backtest_model.stride_hours`; 1–**48** for `tune_model`.
- `validation_split`: 0.1–0.3.
- `frequency`: one of `{"15min", "30min", "h"}`.
- TimesFM: horizon ≤ 128 steps, lookback+horizon ≤ 16384 steps.

### Long-horizon warning

If the user requests `horizon_hours >= 96` on `train_forecast_model`, `tune_model`, `generate_forecast`, or `backtest_model`:

> "Horizons ≥ 96 h (4 days) degrade substantially for short-term load forecasting. Weather-forecast uncertainty compounds beyond ~72 h, calendar/lag features lose signal, and validation CV-RMSE typically 2–3× a 24-h horizon. Consider running an additional shorter-horizon model (e.g. 24 h) alongside for comparison, and expect elevated peak-timing error."

Require explicit user confirmation before proceeding.

### MPS (Apple Silicon) gotchas for Torch models

See §3.5.5 for the current enforcement. Summary of the underlying bugs (do NOT use `accelerator="cpu"` as a silent workaround — CPU requires the §3.5.3 consent gate):

- LSTM on MPS with `dropout > 0` AND `n_rnn_layers >= 2` — pytorch#180744 causes silently wrong results. §3.5.5 refuses to submit; user must set `dropout=0` or `n_rnn_layers=1`.
- TFT on MPS — pytorch#151667 (`MultiheadAttention` + masking + dropout) can produce NaN losses. §3.5.5 requires monitoring; NaN loss is treated as a GPU error (§3.5.4).
- Default precision is `"32"`. No `float64` on MPS.

---

## Section 3.5 — GPU-preferred training for deep-learning models

**Applies to:** `{LSTM, TFT, TiDE, TSMixer, TimesFM, TimesFM+Residual}` — every Torch-backed model in the framework (`_TORCH_MODEL_NAMES` in `src/load_forecasting/core/trainer.py:236`).

Rationale: the internal helper `_detect_accelerator()` (`trainer.py:121`) silently falls back to `"cpu"` when no GPU backend is present. This skill forbids that silent behaviour. GPU (CUDA or Apple Silicon MPS) is the required default; CPU is only permitted after the §3.5.3 consent gate, and GPU errors must never be silently retried on CPU.

### 3.5.1 — Mandatory GPU pre-flight probe

Before any `train_forecast_model` or `tune_model` call for a DL model, run:

```
python -c "import torch; print('cuda=', torch.cuda.is_available(), 'mps=', torch.backends.mps.is_available() and torch.backends.mps.is_built())"
```

Interpret the result:

- **CUDA available** → proceed with `model_kwargs={"accelerator": "gpu"}`.
- **MPS available (Apple Silicon)** → proceed with `model_kwargs={"accelerator": "mps"}`. Apply §3.5.5 safeguards.
- **Neither available** → go to §3.5.3 (CPU-consent gate). Do NOT silently proceed.

### 3.5.2 — Explicit `accelerator` kwarg required

Every DL-model train/tune call MUST pass `model_kwargs={"accelerator": "gpu"|"mps"|"cpu"}` explicitly. Never omit and rely on `_detect_accelerator()`'s silent fallback.

### 3.5.3 — CPU-consent gate (only path to CPU training for DL models)

When the §3.5.1 probe reports no CUDA and no MPS, STOP and show the user verbatim (or paraphrased):

> "Deep-learning model {model_type} was requested, but no GPU accelerator (CUDA or Apple Silicon MPS) was detected on this host. Training on CPU is technically possible but will be substantially slower: LSTM/TFT/TiDE/TSMixer typically take 10–50× longer on CPU than on GPU, and a run that finishes in minutes on GPU can take many hours on CPU. TimesFM and TimesFM+Residual are even more affected. Do you want to (a) proceed anyway on CPU (long wait), (b) switch to a non-DL model (LinearRegression, XGBoost, ARIMA), or (c) abort and re-run on a GPU-equipped host?"

Require an explicit choice:

- (a) → pass `model_kwargs={"accelerator": "cpu"}` and continue.
- (b) → return to §3 and pick a non-DL model.
- (c) → abort.

Never proceed on CPU without capturing this consent.

### 3.5.4 — GPU-error handling (no silent CPU retry)

If a DL training call fails with any error message referencing GPU / CUDA / MPS / device / driver (examples: OOM, kernel not implemented, MPS operator missing, device not found, driver version mismatch, NaN loss on MPS treated as a GPU error per §3.5.5), STOP immediately. Do NOT silently retry on CPU. Surface the raw error and ask:

> "Training on {gpu_backend} failed with the above error. Options: (a) retry on CPU (expect much longer training time — see §3.5.3 for magnitude); (b) adjust the failing hyperparameter (e.g. reduce `batch_size`, `hidden_dim`, `n_epochs`) and retry on GPU; (c) switch to a non-DL model; (d) abort. Which do you want?"

Route the user's choice: (a) follows §3.5.3 consent semantics; (b) re-issues on GPU with adjusted kwargs; (c) returns to §3; (d) stops.

### 3.5.5 — MPS-specific safeguards

- **LSTM on MPS with `dropout > 0` AND `n_rnn_layers >= 2`** → refuse to submit. Ask user to set `dropout=0` or `n_rnn_layers=1`; only then submit with `accelerator="mps"`. The pytorch#180744 bug produces silently wrong results, so this is non-negotiable — CPU fallback still requires §3.5.3 consent.
- **TFT on MPS** → submit with `accelerator="mps"`, but watch training logs / `training_metrics` for NaN loss (pytorch#151667). If NaN is observed, treat as a GPU error per §3.5.4.
- **Precision** default `"32"`. `float64` is unavailable on MPS.

### 3.5.6 — Post-run verification for DL models

After every DL train/tune call, inspect `training_info` (and logs if available) to confirm training actually ran on the requested accelerator. If the framework silently fell back to CPU without going through §3.5.3, treat the run as invalid, discard the model, and re-issue the call with the accelerator kwarg fixed.

---

## Section 3.6 — Hyperparameter decision gate (mandatory before any train/tune call)

**Applies to every tunable model:** `{XGBoost, LSTM, ARIMA, TFT, TiDE, TSMixer, TimesFM, TimesFM+Residual}`. (LinearRegression has no tunable hyperparameters — skip this gate for it. NaiveMean/NaiveSeasonal/NaiveMovingAverage have none either.)

Never silently accept the framework's default hyperparameters. Deep-learning and gradient-boosted accuracy is highly sensitive to them — especially `n_epochs` (under/over-fit) and capacity knobs (`hidden_size`/`hidden_dim`, `max_depth`). Before invoking `train_forecast_model`, surface the chosen model's hyperparameters, their defaults, and a one-line explanation of each, then let the user decide.

**Ordering:** run this gate *after* the §3.5 GPU gate (for DL models) so any user overrides are merged into the **same** `model_kwargs` dict that already carries `accelerator` — never create a second dict or clobber the accelerator key. For XGBoost/ARIMA (no GPU gate), just build a fresh `model_kwargs` from the overrides.

### 3.6.1 — Per-model hyperparameter reference

Show ONLY the table for the model the user actually selected. All Torch models (LSTM/TFT/TiDE/TSMixer/TimesFM/TimesFM+Residual) additionally accept advanced passthroughs (`batch_size`, `optimizer_kwargs`, `lr_scheduler_cls`, `lr_scheduler_kwargs`, `random_state`, `use_reversible_instance_norm`, `precision`) — mention these exist but only enumerate on request.

**XGBoost** (defaults per Li et al. 2025 Table A.3; lr/subsample/colsample fall back to XGBoost library defaults)

| HP | Default | What it does |
|---|---|---|
| `n_estimators` | 40 | Number of boosting trees; more = higher capacity, slower, more overfit risk. |
| `max_depth` | 6 | Max depth per tree; deeper captures more interactions but overfits sooner. |
| `learning_rate` | 0.3 (lib) | Shrinkage per tree; lower needs more trees but generalizes better. |
| `subsample` | 1.0 (lib) | Row fraction sampled per tree; <1 adds regularization. |
| `colsample_bytree` | 1.0 (lib) | Feature fraction sampled per tree; <1 adds regularization. |
| `booster` | gbtree | Base learner type (`gbtree` / `gblinear` / `dart`). |

**ARIMA** (ignores `lookback_hours` — order `p` plays that role)

| HP | Default | What it does |
|---|---|---|
| `p` | 1 | Autoregressive order; how many past values feed the linear model. |
| `d` | 1 | Differencing order to remove trend / achieve stationarity. |
| `q` | 0 | Moving-average order; how many past forecast errors are modeled. |
| `seasonal_order` | (0,0,0,0) | Seasonal `(P,D,Q,m)`; set `m` to the seasonal period (e.g. 24) to model daily cycles. |

**LSTM (BlockRNNModel)** — past-covariate only

| HP | Default | What it does |
|---|---|---|
| `n_epochs` | 20 | Full passes over training data; too few underfits, too many overfits. |
| `hidden_dim` | 64 | Size of the LSTM hidden state; larger = more capacity, slower, more overfit risk. |
| `n_rnn_layers` | 2 | Stacked LSTM layers; more layers capture deeper temporal structure but train slower. |
| `dropout` | 0.1 | Fraction of units randomly zeroed for regularization (only active with ≥2 layers; see MPS caveat §3.5.5). |

**TFT (TFTModel)**

| HP | Default | What it does |
|---|---|---|
| `n_epochs` | 20 | Training passes over the data. |
| `hidden_size` | 16 | Width of the model's internal representation; main capacity knob. |
| `lstm_layers` | 1 | LSTM encoder/decoder layers inside the TFT. |
| `num_attention_heads` | 4 | Parallel attention heads; more heads model more feature interactions. |
| `dropout` | 0.1 | Regularization strength (see MPS NaN caveat §3.5.5). |

**TiDE (TiDEModel)** — defaults per Li et al. 2025 Table A.3

| HP | Default | What it does |
|---|---|---|
| `n_epochs` | 20 | Training passes (paper uses up to 50 with early stopping). |
| `hidden_size` | 128 | Width of encoder/decoder MLP blocks; main capacity knob. |
| `num_encoder_layers` | 1 | Depth of the encoder MLP stack. |
| `num_decoder_layers` | 1 | Depth of the decoder MLP stack. |
| `decoder_output_dim` | 16 | Per-step decoder output width before final projection. |
| `temporal_width_past` | 4 | Dim of the learned projection for past covariates. |
| `temporal_width_future` | 4 | Dim of the learned projection for future covariates. |
| `dropout` | 0.1 | Regularization strength. |

**TSMixer (TSMixerModel)** — defaults per Li et al. 2025 Table A.3

| HP | Default | What it does |
|---|---|---|
| `n_epochs` | 20 | Training passes (paper uses up to 50 with early stopping). |
| `hidden_size` | 64 | Feature-mixing hidden width; main capacity knob. |
| `ff_size` | 64 | Feed-forward width inside each mixer block. |
| `num_blocks` | 2 | Stacked time/feature mixer blocks; more = deeper, slower. |
| `activation` | ReLU | Nonlinearity used in the mixer blocks. |
| `dropout` | 0.1 | Regularization strength. |
| `norm_type` | LayerNorm | Normalization layer type inside blocks. |

**TimesFM / TimesFM+Residual** (foundation model; ~800 MB weights auto-downloaded on first use)

| HP | Default | What it does |
|---|---|---|
| `n_epochs` | 5 | Light fine-tuning passes on your data; set `0` for pure zero-shot. |
| `enable_finetuning` | true (n_epochs>0) | Whether to fine-tune the pretrained backbone at all; `False` = zero-shot. |
| `local_dir` | none | Path to pre-cached weights to skip the HuggingFace download. |
| *(TimesFM+Residual)* Ridge residual | internal | Ridge regressor on covariates corrects the target-only TimesFM output; not exposed as a simple scalar. |

### 3.6.2 — Decision prompt (surface verbatim, parameterized per model)

> "About to train {model_type}. Its key hyperparameters, defaults, and a one-line explanation of each are shown above. Results are sensitive to these — especially `n_epochs` (default {N}), which can under- or over-fit. Do you want to (a) train with these defaults now, (b) override specific hyperparameters — tell me which and their values and I'll pass them via `model_kwargs`, or (c) run Optuna hyperparameter tuning via `tune_model` instead (best metrics, more compute — §5e)?"

### 3.6.3 — Routing

- **(a) defaults** → proceed to §5d unchanged (DL models keep the `model_kwargs` from §3.5 carrying only `accelerator`).
- **(b) override** → merge the user's chosen values into the existing `model_kwargs` dict (preserving `accelerator` for DL models), then proceed to §5d. Echo the final `model_kwargs` back to the user before submitting.
- **(c) tune** → go to §5e (`tune_model`). Remind the user: `horizon_hours` cap is **48** there (vs 96 for train), and TimesFM / TimesFM+Residual are rejected unless an explicit `search_space` is supplied. The built-in Optuna search spaces are: XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`), LSTM (`hidden_dim`, `n_rnn_layers`, `dropout`, `n_epochs`), ARIMA (`p`, `d`, `q`), TFT (`hidden_size`, `lstm_layers`, `num_attention_heads`, `dropout`, `n_epochs`), TiDE (`hidden_size`, `num_encoder_layers`, `num_decoder_layers`, `decoder_output_dim`, `dropout`, `n_epochs`), TSMixer (`hidden_size`, `ff_size`, `num_blocks`, `dropout`, `n_epochs`).

---

## Section 4 — Deprecated

(Section number preserved for historical reasons; content moved into §5.)

---

## Section 5 — Train vs Tune (user-directed)

### 5a — Covariate confirmation gate (mandatory before any train/tune call)

Trigger: user chose a model in the covariate-supporting set:
`{LinearRegression, XGBoost, LSTM, ARIMA, TFT, TiDE, TSMixer, TimesFM+Residual}`.

Inspect the merged file's (or single file's) auto-detected `columns.past_covariates` and any explicit `column_mapping.future_covariates`. If **no weather-derived column** is present (patterns: `temp`, `temperature`, `rh`, `humidity`, `solar`, `wind`):

> "You selected {model_type}, which is designed to exploit exogenous covariates. Weather is by far the strongest predictor of building load. No weather columns were detected. Training a covariate-capable model on target-only data typically produces marginal or no gains over LinearRegression and wastes compute (especially for LSTM/TFT/TiDE/TSMixer). Strongly recommended: provide weather data via `merge_covariates` (historical) or `fetch_weather_forecast` + merge (for inference). Do you want to (a) proceed anyway, (b) pause and add weather data, or (c) fall back to a naive baseline?"

Special cases:

- **Weather is only in `past_covariates`** and the model supports future covariates: warn that at inference time the model will need weather as a future covariate. Recommend re-inspecting with explicit `column_mapping.future_covariates=[...]`.
- **LSTM chosen with future weather available**: warn that LSTM silently ignores future covariates; if forecast weather is the primary reason for the choice, LSTM is wrong.
- **Pure TimesFM chosen with weather available**: TimesFM ignores all covariates. Recommend `TimesFM+Residual` or a different model.

### 5b — Long-horizon re-confirmation

Re-apply the §3 long-horizon warning at invocation time if `horizon_hours >= 96`. Do not proceed on remembered consent from an earlier turn if the user has changed the value.

### 5c — Holiday features (optional but strongly recommended)

The framework auto-injects hour/day/month/weekend features but **does not** auto-generate holiday flags. Holiday-driven load reductions (e.g., a commercial building on Christmas Day) will look like model error unless supplied.

If the user cares about holiday accuracy:

1. Ask for their locale (country / state) to identify the correct holiday calendar.
2. Preprocessing holidays into a CSV is outside the MCP tool surface — offer two options:
   - User prepares an `is_holiday` (0/1) CSV covering both train and inference date ranges, then flows through the multi-CSV merge in §2.5.
   - Skip holiday features and accept larger errors on those days. If `peak_dates` in evaluation includes holidays, expect elevated PMAPE.
3. If added, pass explicitly in `column_mapping.future_covariates=["is_holiday", ...]` (holiday calendars are known in advance). Auto-detection never assigns future covariates; `inspect_data` lists such columns under `columns.future_covariate_candidates` and says so in `suggestions`, but you must pass them explicitly.
4. For LSTM (ignores future covariates) or pure TimesFM (ignores all covariates), holiday flags will be silently ignored.

### 5d — Train path (`train_forecast_model`)

Single fit with sensible defaults. Use when the user is exploring, iterating, or accepting default hyperparameters.

**Before invoking, complete the §3.6 hyperparameter decision gate** for any tunable model (`{XGBoost, LSTM, ARIMA, TFT, TiDE, TSMixer, TimesFM, TimesFM+Residual}`): show the per-model HP table with explanations and capture the user's choice (defaults / overrides / tune). Any user-chosen overrides are merged into the same `model_kwargs` dict that (for DL models) already carries `accelerator` — do not clobber it.

Required + recommended arguments:

- `csv_path` — the (possibly merged) input CSV.
- `model_type` — user's choice from §3.
- `frequency` — from §2 / §2.5 inspection.
- `lookback_hours` (default 24, range 1–168).
- `horizon_hours` (default 6, range 1–96).
- `validation_split` (default 0.2).
- `building_name` — for a readable `model_id`. Always pass this.
- `column_mapping` — pass explicitly if the merged file has future covariates (weather forecast, holiday) that must be treated as such. Remember: `future_covariates` are NEVER auto-detected.
- `augment_weather_noise=True` when future covariates include forecast weather — matches Li et al. 2025 §3.1.2. Default `weather_noise_std=1.0` (σ in raw units).
- **For DL models** (`{LSTM, TFT, TiDE, TSMixer, TimesFM, TimesFM+Residual}`): complete the §3.5.1 GPU pre-flight probe first, then pass `model_kwargs={"accelerator": "gpu"|"mps"|"cpu"}` explicitly. `"cpu"` is only permitted after capturing consent per §3.5.3. Never rely on `_detect_accelerator()`'s silent fallback.

Read back and surface to the user:

- `model_id`, `model_path`
- `training_metrics`, `validation_metrics` (all five: rmse, mae, mape, cv_rmse, r_squared)
- `data_summary` — verify §2b seasonal-split ratio landed
- `training_info.training_time_seconds`, `training_info.energy_kwh`
- `data_inspection.quality_flags` / `suggestions`
- Any `ml_warnings` — all of them, verbatim
- **For DL models**: verify from `training_info` / logs that training actually ran on the requested accelerator (§3.5.6). If it silently fell back to CPU without §3.5.3 consent, discard the model and re-issue with the accelerator kwarg fixed. If the call errored on a GPU-related message, follow §3.5.4 — do NOT silently retry on CPU.

### 5e — Tune path (`tune_model`)

Optuna hyperparameter search followed by a final fit. Use when the user wants best possible metrics and can spend the compute. This is where §3.6 option (c) routes.

Tunable model types: **XGBoost, LSTM, ARIMA, TFT, TiDE, TSMixer**. LinearRegression has an empty search space (nothing to tune). TimesFM and TimesFM+Residual are rejected unless the caller supplies an explicit `search_space`.

Constraints:

- `horizon_hours` cap here is **48**, not 96 (this is a real difference vs `train_forecast_model`).
- `n_trials` default 20 — recommend the user pick explicitly based on compute budget.
- **For DL models** (`{LSTM, TFT, TiDE, TSMixer}` — tunable subset): complete the §3.5.1 GPU pre-flight probe first, then pass `model_kwargs={"accelerator": "gpu"|"mps"|"cpu"}` explicitly. `"cpu"` is only permitted after capturing consent per §3.5.3. GPU errors during any trial must be handled per §3.5.4 — never silently retry on CPU. Note: `n_trials × per-trial-time` compounds any CPU-vs-GPU speed gap; the §3.5.3 warning is especially critical for tuning.

Mandatory warning to surface every time:

> "Metrics from `tune_model` are optimistic — they measure the best-selected hyperparameters on the same validation split used to select them. The saved model is marked `config.tuned=True` and `evaluate_forecast_model` will warn if you evaluate on the same CSV. You MUST run `evaluate_forecast_model` on an independent holdout CSV before trusting these metrics."

Read back same fields as 5d (including the §3.5.6 accelerator-verification step for DL models) plus: `best_params`, `best_cv_rmse`, `n_trials_completed`, `all_trial_results` (offer to summarize if long).

---

## Section 6 — Mandatory `evaluate_forecast_model` on independent holdout

Do NOT declare training successful without an out-of-sample evaluation.

Requirements:

- `csv_path` MUST be an independent holdout, not the training CSV and (for tuned models) not the tuning CSV.
- Frequency mismatch between the model and the holdout CSV is a hard block from the tool — verify frequency alignment before calling.
- Enable `include_residual_analysis=True`.
- If the user has known peak / high-load days, pass `peak_dates=["YYYY-MM-DD", ...]` for PMAPE + PTE (Li et al. 2025 Table 5).

Interpret `comparison_to_validation.performance_status`:

- `similar` → healthy generalization.
- `degraded` → overfitting or distribution shift; investigate.
- `improved` → suspicious; check for data leakage or trivially-easy holdout.
- `unknown` → validation metrics missing from the model metadata.

Surface all `ml_warnings` (short test window, interpolated NaNs, training-data overlap, tuned-model warning if applicable).

---

## Section 7 — Rolling `backtest_model` for realistic performance

Complements §6: catches degradation over time and horizon-specific weaknesses that a single evaluate misses.

- `stride_hours` defaults to `horizon` (non-overlapping windows). Reduce for finer temporal coverage at proportionally higher cost.
- `start_fraction=0.2` skips the first 20 % (warm-up).
- Enable `include_residual_analysis=True` and `peak_dates=[...]` when relevant.

**Compute-cost warnings** — surface proactively:

- Local baselines (NaiveMean, NaiveSeasonal, NaiveMovingAverage) refit at **every stride step** (O(n) refits). Slow on large datasets.
- `TimesFM+Residual` runs one full TimesFM forward pass per stride. Significantly slower than pure TimesFM. Plan compute budget accordingly.

---

## Section 8 — Reports

Prefer `generate_backtest_report` over `generate_evaluation_report` for stakeholder-facing output — the backtest report includes h-step-ahead RMSE curves and richer diagnostics.

Both require `output_html_path` ending in `.html` or `.htm`.

---

## Section 9 — Post-run checklist

Before declaring the task complete, verify every item:

- [ ] The MCP server connection held for the entire workflow. If it was ever lost, work STOPPED immediately, nothing was done manually, and the user was asked to restore the connection before resuming (§0).
- [ ] Timezone was explicitly confirmed by the user (naive vs tz-aware, IANA zone) for every CSV before any tool call — never guessed or inferred (§1.0). For multi-CSV, the per-file tz table was built (§2.5.4).
- [ ] For any DL model (LSTM/TFT/TiDE/TSMixer/TimesFM/TimesFM+Residual): §3.5.1 GPU pre-flight probe was executed and its result recorded.
- [ ] For any DL model: `model_kwargs={"accelerator": ...}` was passed explicitly (`"gpu"`, `"mps"`, or `"cpu"` post-consent).
- [ ] If CPU was used for a DL model: §3.5.3 consent was captured from the user verbatim (choice `(a)`).
- [ ] If a GPU-related error occurred during a DL training call: §3.5.4 flow was followed — no silent CPU retry.
- [ ] Post-run §3.5.6 verification confirms the DL model actually trained on the requested accelerator (or the run was discarded and re-issued).
- [ ] For any tunable model (XGBoost/ARIMA/LSTM/TFT/TiDE/TSMixer/TimesFM/TimesFM+Residual): §3.6 hyperparameter gate was surfaced (per-model HPs + explanations shown) and the user chose defaults / overrides / tuning.
- [ ] `inspect_data` was run and returned `ready_to_train == True` (single-file: §2; multi-CSV: §2.5.10 on the final merged file).
- [ ] If N ≥ 2 CSVs merged: every pairwise merge passed the 5 % warn / 20 % hard-stop check.
- [ ] If N ≥ 2 CSVs merged: cumulative merge summary table shown to the user.
- [ ] If N ≥ 2 CSVs merged: final merged file has no `_x` / `_y` suffixed columns.
- [ ] Training data span ≥ 365 days confirmed, OR user acknowledged the short-data warning.
- [ ] If horizon_hours ≥ 96: user confirmed the accuracy-degradation warning.
- [ ] If covariate-supporting model chosen: weather covariates present and confirmed, OR user explicitly opted out.
- [ ] Post-training seasonal-stratification landed (`training_samples / total_samples ≈ 0.8`), OR user warned of silent fallback.
- [ ] `model_id` recorded and communicated to user.
- [ ] Train vs validation CV-RMSE ratio inspected (>2× triggers a built-in overfitting warning).
- [ ] Independent-holdout `evaluate_forecast_model` run (§6).
- [ ] If model was tuned: holdout CSV is independent of the tuning CSV.
- [ ] Rolling `backtest_model` metrics reviewed (§7).
- [ ] All `ml_warnings` from every tool call surfaced to the user verbatim.

---

## Section 10 — Common pitfalls (quick reference)

- **Lost MCP connection = hard stop.** If the MCP server disconnects or a tool call fails because the server is unreachable at any point, STOP immediately, do NOT do the work manually or fabricate outputs, report the failure verbatim, and wait for the user to restore the connection (§0).
- **Never guess the timezone.** CSVs rarely carry authoritative tz metadata, so ask the user the moment a CSV is provided — before `inspect_data` or any other tool call — and never infer it from filenames, values, offsets, or location (§1.0).
- **Hours vs steps.** Tool API is in hours; internal Darts is in steps. `hours_to_steps(hours, freq)`: 24 h at 15 min = 96 steps.
- **`future_covariates` are NEVER auto-detected.** Must pass explicit `column_mapping.future_covariates=[...]` for models that consume them.
- **LSTM silently ignores future covariates** despite being in the mixed-cov API.
- **ARIMA ignores `lookback_hours`.** Use `model_kwargs={"p": ...}` or tune p/d/q.
- **TimesFM silently drops all covariates.** A warning is logged, not raised. Use TimesFM+Residual for covariate handling.
- **Holidays are not auto-generated.** Only weekend/hour/dow/month features are injected.
- **`merge_covariates` is pairwise.** For N ≥ 3 inputs, orchestrate N-1 sequential calls.
- **Same-named columns across covariate files silently collide** with `_x` / `_y` suffixes — catch pre-merge.
- **`merge_covariates` does not resample, sort, or dedup the load side.** Callers must ensure aligned cadence and clean load timestamps.
- **`horizon_hours` bounds differ** — `train_forecast_model` allows 1–96; `tune_model` allows 1–48.
- **Registry cache is per-process.** `ModelRegistry._registry_cache` is stale if another process modifies `registry.json`.
- **PyTorch model loading uses `weights_only=False`** — required for Lightning checkpoints under PyTorch ≥ 2.6.
- **MPS gotchas on Apple Silicon** — LSTM dropout+multi-layer bug; TFT MultiheadAttention NaN. See §3.5.5.
- **DL models are GPU-preferred; CPU requires explicit consent.** `_detect_accelerator()` silently returns `"cpu"` when no GPU is present — never let this happen without going through §3.5.3. Always pass `model_kwargs={"accelerator": ...}` explicitly on DL model train/tune calls.
- **GPU errors are terminal until re-triaged.** Any CUDA/MPS error during a DL training call must NOT be silently retried on CPU — follow §3.5.4.
- **`tune_model` metrics are optimistic** — always requires an independent holdout for `evaluate_forecast_model`.
- **Never silently accept default hyperparameters for tunable models.** Surface the §3.6 gate — show each HP with a one-line explanation (defaults / override `n_epochs` etc. via `model_kwargs` / `tune_model`). DL and boosted results are hyperparameter-sensitive.

---

## Section 11 — Tool cheat sheet

| Tool | Purpose | Required args | Notable optional args | Key return keys |
|---|---|---|---|---|
| `list_datasets` | Enumerate available CSV/Parquet files with absolute paths | — | `directory`, `recursive`, `include_stats`, `limit` | `datasets[].path`, `rows`, `columns`, `frequency`, `start`/`end` |
| `generate_data_report` | Visual EDA: series + covariates with the train/validation split (seasonal or sequential), season bands, gaps/outliers, load profiles, day×hour heatmap, covariate relationships | `csv_path` | `output_html_path` (optional), `split_strategy`, `validation_split`, `covariates`, `column_mapping` | `output_html_path`, `split.segments`, `unmapped_covariates`, `ready_to_train` |
| `inspect_data` | EDA + validation pre-flight | `csv_path` | `column_mapping`, `frequency` | `ready_to_train`, `blocking_issues`, `quality_flags`, `columns`, `time_range`, `frequency` |
| `merge_covariates` | Pairwise LEFT-JOIN load + covariate CSV | `load_csv_path`, `covariate_csv_path`, `output_csv_path` | `load_datetime_col`, `covariate_datetime_col`, `covariate_columns`, `covariate_timezone`, `load_timezone` | `n_rows`, `covariate_columns`, `n_missing_filled`, `dst_duplicates_dropped`, `timezone_conversion` |
| `fetch_weather_forecast` | Pull Open-Meteo weather | `latitude`, `longitude` | `forecast_hours`, `past_hours`, `timezone`, `variables`/`preset`, `resolution` | weather rows + optional CSV write |
| `train_forecast_model` | Single-fit training | `csv_path` | `model_type`, `lookback_hours`, `horizon_hours`, `frequency`, `validation_split`, `building_name`, `column_mapping`, `augment_weather_noise` | `model_id`, `model_path`, `training_metrics`, `validation_metrics`, `data_summary`, `training_info`, `ml_warnings` |
| `tune_model` | Optuna HP search + final fit | `csv_path`, `model_type` | `n_trials`, `search_space`, `lookback_hours`, `horizon_hours` (cap 48) | above + `best_params`, `best_cv_rmse`, `all_trial_results` |
| `evaluate_forecast_model` | Holdout evaluation | `model_id`, `csv_path` | `column_mapping`, `include_residual_analysis`, `peak_dates`, `output_csv_path` | `test_metrics`, `comparison_to_validation`, `predictions`, `peak_metrics`, `ml_warnings` |
| `backtest_model` | Rolling-window backtest | `model_id`, `csv_path` | `stride_hours`, `start_fraction`, `include_residual_analysis`, `peak_dates` | `backtest_metrics`, `backtest_summary`, `predictions`, `peak_metrics` |
| `generate_evaluation_report` | HTML report from evaluate | `model_id`, `csv_path`, `output_html_path` | `column_mapping`, `title` | `output_html_path`, `test_metrics` |
| `generate_backtest_report` | HTML report from backtest (richer) | `model_id`, `csv_path`, `output_html_path` | `stride_hours`, `start_fraction`, `peak_dates`, `title` | `output_html_path`, `backtest_metrics`, h-step curves |
| `generate_inference_dashboard` | HTML forecast dashboard | `model_id`, `csv_path`, `output_html_path`, `latitude`, `longitude` | `column_mapping`, `horizon_hours` | dashboard HTML |
| `generate_forecast` | One-shot forward inference | `model_id`, `csv_path` | `column_mapping`, `horizon_hours`, `output_csv_path` | `predictions`, `forecast_start/end`, `context_summary` |
| `list_models` | Query the model registry | — | `building_name`, `model_type`, `sort_by`, `limit` | `models[]` incl. `csv_path`, `target_column`, `frequency` (what each model was trained on), `total_count` |
| `get_algorithm_specifications` | Return all YAML specs | — | — | `specs`, `count` |
