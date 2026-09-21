# Multi-CSV merge protocol (train-forecast-model, Section 2.5)

Load this file when the user supplies two or more input CSVs (load + weather, load + holidays, ...). It is the full ten-step protocol; the summary in SKILL.md §2.5 is not a substitute.

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

