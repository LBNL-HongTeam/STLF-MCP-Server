# STLF-MCP-Server

A Model Context Protocol (MCP) server that provides **16 tools** for
**short-term load forecasting (STLF)** of building and grid electrical demand.
This server enables AI assistants and other MCP clients to inspect data, merge
weather, train, tune, evaluate, backtest, and forecast electrical loads — and
generate interactive HTML reports — through a standardized interface.

Built on [Darts](https://github.com/unit8co/darts) for the modelling and
[FastMCP](https://github.com/jlowin/fastmcp) for the server. Methodology aligns
with Li et al. (2025), *Energy & Buildings* 344.

> **Version**: 0.1.0
> **Python**: 3.11–3.12 (3.10 is not supported — numpy ≥ 2.3 requires 3.11;
> 3.13 is not supported — numba/llvmlite incompatibility)

## 📑 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Using the MCP Server](#using-the-mcp-server)
    - [Claude Desktop](#claude-desktop)
    - [VS Code](#vs-code)
    - [Cursor](#cursor)
  - [Development Setup](#development-setup)
  - [Streamable HTTP Transport](#streamable-http-transport)
- [Available Tools](#available-tools)
- [Usage Examples](#usage-examples)
- [Architecture](#architecture)
- [Agent Skills](#agent-skills)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

STLF-MCP-Server makes time-series load forecasting accessible to AI assistants
and automation tools through the Model Context Protocol. It wraps a full
forecasting workflow — from raw CSV inspection to trained models, evaluation,
and forward forecasts — as MCP tools any client can call.

**Key Features:**

- 📊 **12 model types**: NaiveMean, NaiveSeasonal, NaiveMovingAverage,
  LinearRegression, XGBoost, LSTM, ARIMA, TFT, TiDE, TSMixer, TimesFM, and
  TimesFM+Residual (TimesFM 2.5 backbone with a Ridge residual regressor).
- 🎯 **Probabilistic forecasting**: optional quantile (P10/P50/P90) prediction
  intervals for XGBoost, LSTM, TFT, TiDE, and TSMixer, with pinball loss /
  coverage / interval-width metrics.
- 🌦️ **Weather integration**: live Open-Meteo forecast fetch plus a DST-safe
  merge into your load CSV.
- 🛠️ **Automatic feature engineering**: cyclic calendar features (hour,
  day-of-week, month) and 24/48/168h lag features are auto-injected.
- 📈 **Rigorous evaluation**: RMSE, MAE, MAPE, CV-RMSE, R², plus peak-day
  PMAPE/PTE metrics.
- 🔁 **Rolling-window backtesting** with h-step-ahead error degradation.
- 🔍 **Hyperparameter tuning** via Optuna.
- 🗂️ **Batch / multi-series** fan-out over many feeders or meters (fault
  tolerant).
- 🖥️ **Self-contained HTML reports**: evaluation, backtest, and live inference
  dashboards (all JS/CSS inlined; open with `file://`, no server required).
- 💾 **Model registry** for persistent storage and versioning.

## Installation

### Using the MCP Server

**Prerequisites (all clients):**

- Python 3.11–3.12 on your PATH
- The repository cloned locally and its dependencies installed
  (see [Development Setup](#development-setup))
- `git` on your PATH
- **macOS only**: the OpenMP runtime, `brew install libomp`. XGBoost links
  against it and the server imports XGBoost at startup, so without it the
  server fails to launch.

Choose the appropriate setup for your AI assistant or IDE.

#### Claude Desktop

1. **Install dependencies** (one-time setup):

   ```bash
   git clone https://github.com/LBNL-HongTeam/STLF-MCP-Server.git
   cd STLF-MCP-Server
   uv sync
   ```

2. **Locate the Claude Desktop config file** for your OS:

   - **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

   Create the file if it does not exist, then add:

   ```json
   {
     "mcpServers": {
       "load_forecasting": {
         "command": "uv",
         "args": [
           "run",
           "--directory", "/path/to/STLF-MCP-Server",
           "python", "main.py"
         ]
       }
     }
   }
   ```

   **Important**: Replace `/path/to/STLF-MCP-Server` with the absolute path to
   your cloned repo (Windows users: use double backslashes in JSON, e.g.
   `C:\\Users\\yourname\\code\\STLF-MCP-Server`).

3. **Restart Claude Desktop**. The `load_forecasting` server should appear in
   the MCP servers panel.

4. **Verify**: in a new chat, ask *"List the load forecasting MCP tools you have
   access to."* You should see tools like `train_forecast_model`,
   `evaluate_forecast_model`, and `inspect_data`. If not, check
   [Troubleshooting](#troubleshooting).

#### VS Code

VS Code 1.102+ ships native MCP support. Config goes in `.vscode/mcp.json` at
the workspace root (or in user settings under `"mcp"`).

1. **Install dependencies** (same as Claude Desktop step 1 above).

2. **Create `.vscode/mcp.json`** in your project:

   ```json
   {
     "servers": {
       "load_forecasting": {
         "command": "uv",
         "args": [
           "run",
           "--directory", "${workspaceFolder}",
           "python", "main.py"
         ]
       }
     }
   }
   ```

3. **Reload VS Code** (`Ctrl/Cmd+Shift+P` → *Developer: Reload Window*). Open the
   Chat view and confirm the `load_forecasting` MCP server shows as *Running*.

4. **Verify**: ask the chat *"What load forecasting tools are available?"* — you
   should see the tool list.

#### Cursor

1. **Install dependencies** (same as Claude Desktop step 1 above).

2. **Locate the Cursor MCP config file** for your OS:

   - **macOS/Linux**: `~/.cursor/mcp.json`
   - **Windows**: `%USERPROFILE%\.cursor\mcp.json`

   Create the file if it does not exist, then add:

   ```json
   {
     "mcpServers": {
       "load_forecasting": {
         "command": "uv",
         "args": [
           "run",
           "--directory", "/path/to/STLF-MCP-Server",
           "python", "main.py"
         ]
       }
     }
   }
   ```

   **Important**: Replace `/path/to/STLF-MCP-Server` with the absolute path to
   your cloned repo (Windows users: use double backslashes in JSON).

3. **Restart Cursor**. Open *Settings → MCP* and confirm the `load_forecasting`
   server is listed as connected.

4. **Verify**: ask Cursor chat *"What load forecasting tools are available?"* —
   you should see the tool list.

### Development Setup

For contributors who want to modify or extend the MCP server.
[`uv`](https://github.com/astral-sh/uv) is the package manager; the lockfile is
`uv.lock`.

**Prerequisites:**

- Python 3.11–3.12
- [uv package manager](https://github.com/astral-sh/uv)
- **macOS only**: `brew install libomp` (OpenMP runtime required by XGBoost)

```bash
# Clone and install
git clone https://github.com/LBNL-HongTeam/STLF-MCP-Server.git
cd STLF-MCP-Server
uv sync

# Or with pip (development dependencies)
pip install -e ".[dev]"

# Run the server for testing (STDIO mode)
python main.py
```

PyTorch and PyTorch Lightning are pulled in automatically and are required for
the LSTM, TFT, TiDE, TSMixer, and TimesFM models.

### Streamable HTTP Transport

By default the server runs over **stdio**, which is what every MCP client config
in this README uses. The server can also run over **streamable HTTP** — useful
for the AlphaBuilding-Agents integration or connecting clients that expect an
HTTP MCP endpoint.

```bash
python main.py --http        # port 8003 (default)
python main.py --http 8080   # custom port
# or
MCP_TRANSPORT=http python main.py
```

The server listens at `http://<host>:8003/mcp`. Connect an MCP client by
replacing the stdio command stanza with an HTTP one:

```json
{
  "mcpServers": {
    "load_forecasting": {
      "type": "http",
      "url": "http://localhost:8003/mcp"
    }
  }
}
```

> **Note**: HTTP mode binds `0.0.0.0` with **no authentication or TLS**. It is
> intended for trusted internal networks, not public exposure.

## Available Tools

The server provides **16 tools** organized into **6 categories**. Full parameter
and return-shape documentation lives in each tool's function signature (FastMCP
derives the JSON schema from it) and in the YAML specs under
`src/load_forecasting/specs/`.

### 🗂️ Data Preparation (4 tools)

- `list_datasets` - Enumerate CSV/Parquet files available to the server with
  absolute paths, row counts, columns, date range and inferred frequency.
  Scans `LOAD_FORECASTING_DATA_DIR` or the bundled `data/examples` by default
- `inspect_data` - Profile a CSV before training: detect column roles, infer
  frequency, compute per-column stats, flag gaps/anomalies, suggest features
- `merge_covariates` - Left-join a covariate CSV (e.g. weather) into a load CSV
  with automatic timezone conversion and DST handling
- `fetch_weather_forecast` - Fetch live weather from the Open-Meteo API,
  formatted to slot into `generate_forecast` as future covariates

### 🎓 Training & Tuning (2 tools)

- `train_forecast_model` - Train a model on historical load data (all 12 model
  types; optional probabilistic quantile fitting)
- `tune_model` - Hyperparameter-tune a model with Optuna and register the best
  result

### 📏 Evaluation & Backtesting (2 tools)

- `evaluate_forecast_model` - Evaluate a trained model on test data
  (RMSE/MAE/MAPE/CV-RMSE/R², optional peak-day and probabilistic metrics)
- `backtest_model` - Rolling-window backtest with configurable stride and start
  position across many historical windows

### 🔮 Forecasting (3 tools)

- `generate_forecast` - Generate a forward forecast from recent context data;
  emits quantile bands for probabilistic models
- `batch_train_forecast_models` - Train one model per series across many series
  (job list or wide CSV of meter columns); fault tolerant
- `batch_generate_forecast` - Generate forward forecasts for many trained models
  in one call

### 📊 Reporting (3 tools)

- `generate_evaluation_report` - Self-contained interactive HTML evaluation
  report
- `generate_backtest_report` - Self-contained HTML backtest report with a
  forecast-playback slider and h-step error charts
- `generate_inference_dashboard` - Live inference dashboard that refreshes
  weather in the browser

### 🔎 Registry & Discovery (2 tools)

- `list_models` - List trained models in the registry, with filtering and
  sorting
- `get_algorithm_specifications` - Return all YAML algorithm specs for
  agent-side model selection

## Usage Examples

### Basic Workflow

0. **Find the data** (optional — skip if you already have a path). Bundled
   sample datasets ship with the repo under `data/examples/`:

   ```json
   { "tool": "list_datasets", "arguments": {} }
   ```

   Each record's `path` is absolute and can be passed straight to the tools
   below. Included samples: a 3-month 15-minute building load with outdoor
   temperature (`sample_building_load.csv`) and hourly 2021/2023 AMI
   aggregates at city, substation and feeder level with matching Open-Meteo
   weather (`AMI/`).

1. **Inspect the data** before training:

   ```json
   {
     "tool": "inspect_data",
     "arguments": {
       "csv_path": "data/building_33_hourly.csv",
       "frequency": "h"
     }
   }
   ```

2. **Train a model**:

   ```json
   {
     "tool": "train_forecast_model",
     "arguments": {
       "csv_path": "data/building_33_hourly.csv",
       "model_type": "XGBoost",
       "lookback_hours": 48,
       "horizon_hours": 24,
       "building_name": "Building_33"
     }
   }
   ```

3. **Evaluate the trained model**:

   ```json
   {
     "tool": "evaluate_forecast_model",
     "arguments": {
       "model_id": "Building_33_XGBoost_20250124_143022",
       "csv_path": "data/building_33_test.csv"
     }
   }
   ```

4. **Generate a forward forecast**:

   ```json
   {
     "tool": "generate_forecast",
     "arguments": {
       "model_id": "Building_33_XGBoost_20250124_143022",
       "csv_path": "data/building_33_recent.csv",
       "horizon_hours": 24
     }
   }
   ```

### Advanced Features

**Probabilistic (interval) forecasting** — train with quantiles, then forecast
with prediction bands:

```json
{
  "tool": "train_forecast_model",
  "arguments": {
    "csv_path": "data/feeder.csv",
    "model_type": "XGBoost",
    "lookback_hours": 48,
    "horizon_hours": 24,
    "probabilistic": true,
    "quantiles": [0.1, 0.5, 0.9]
  }
}
```

```json
{
  "tool": "generate_forecast",
  "arguments": {
    "model_id": "feeder_XGBoost_20250124_143022",
    "csv_path": "data/feeder_recent.csv",
    "num_samples": 200
  }
}
```

**Weather-aware forecasting** — fetch weather, merge it, then forecast:

```json
{
  "tool": "fetch_weather_forecast",
  "arguments": {
    "latitude": 37.87,
    "longitude": -122.27,
    "forecast_hours": 48,
    "past_hours": 48,
    "timezone": "America/Los_Angeles",
    "output_csv_path": "data/weather.csv"
  }
}
```

```json
{
  "tool": "merge_covariates",
  "arguments": {
    "load_csv_path": "data/feeder_recent.csv",
    "covariate_csv_path": "data/weather.csv",
    "output_csv_path": "data/feeder_with_weather.csv"
  }
}
```

**Batch multi-series training** — one model per meter column of a wide AMI
export:

```json
{
  "tool": "batch_train_forecast_models",
  "arguments": {
    "csv_path": "data/ami_wide.csv",
    "target_columns": ["feeder_A", "feeder_B", "feeder_C"],
    "datetime_col": "timestamp",
    "model_type": "XGBoost",
    "lookback_hours": 48,
    "horizon_hours": 24
  }
}
```

**Generate an interactive HTML report**:

```json
{
  "tool": "generate_evaluation_report",
  "arguments": {
    "model_id": "Building_33_XGBoost_20250124_143022",
    "csv_path": "data/building_33_test.csv",
    "output_html_path": "reports/building_33_eval.html"
  }
}
```

### Using with MCP Inspector

Test tools interactively (requires Node.js 18+):

```bash
npx @modelcontextprotocol/inspector uv run python main.py
```

The Inspector opens a browser UI where you can list tools and invoke them with
JSON arguments — useful for sanity-checking the install before wiring up a
client.

## Architecture

The server follows a layered architecture:

```
┌─────────────────────────┐
│   MCP Protocol Layer    │  FastMCP server handling client communications
├─────────────────────────┤
│      Tools Layer        │  16 tools organized into 6 categories
├─────────────────────────┤
│       Core Layer        │  Data loader, trainer, evaluator, model registry,
│                         │  tuning (Optuna), weather fetcher, spec loader
├─────────────────────────┤
│    Darts Integration    │  Time-series models + covariate handling
└─────────────────────────┘
```

**Project Structure:**

```
STLF-MCP-Server/
├── main.py                       # Entry point (stdio / HTTP transport)
├── src/load_forecasting/
│   ├── server.py                 # FastMCP server + tool registration
│   ├── core/                     # data_loader, trainer, evaluator,
│   │                             #   model_registry, tuning, weather_fetcher,
│   │                             #   spec_loader, frequency_utils
│   ├── tools/                    # MCP tool implementations by workflow stage
│   ├── specs/                    # YAML algorithm specs (agent discovery)
│   └── reporting/                # Self-contained HTML report generation
├── skills/                       # Optional AI-agent workflow instructions
├── models/                       # Trained models + registry.json (runtime)
├── reports/                      # Generated HTML reports (runtime)
└── tests/                        # Unit + integration tests
```

See `AGENTS.md` for architecture internals and non-obvious gotchas (covariate
handling, hours-vs-steps conversion, MPS/OpenMP caveats, spec/registry
resolution).

## Agent Skills

The top-level `skills/` directory contains optional instruction bundles for AI
coding agents. Each skill lives in its own directory and is defined by a
`SKILL.md` file with YAML front matter (`name` and `description`) followed by
the workflow instructions.

`skills/train-forecast-model/SKILL.md` guides an agent through the recommended
inspect, merge, train, evaluate, backtest, and report sequence using this
server's MCP tools. It does not implement forecasting logic and is not required
to run the server; the Python tools remain the authoritative implementation.

Agent skills are separate from `src/load_forecasting/specs/`. The `skills/`
content tells an agent how to orchestrate tools, while the packaged YAML specs
describe individual algorithms for runtime agent discovery. To use a skill,
configure your AI client to load or import its directory according to that
client's skill-discovery mechanism.

## Configuration

The server uses sensible defaults. Configuration can be customized via
environment variables (copy `.env.example` → `.env` for local overrides):

| Variable | Default | Description |
|----------|---------|-------------|
| `LOAD_FORECASTING_MODEL_DIR` | `./models` | Model storage directory. |
| `LOAD_FORECASTING_DATA_DIR` | *(unset)* | Directory `list_datasets` scans by default, and an extra root for relative `csv_path` values. Falls back to the bundled `data/examples`. |
| `MCP_TRANSPORT` | `stdio` | Transport mode (`stdio` or `http`). |
| `MCP_HTTP_PORT` | `8003` | HTTP port. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` (set by `main.py`) | Tolerate duplicate OpenMP runtime (xgboost + torch on macOS); must be set pre-import. |
| `OMP_NUM_THREADS` | `1` (set by `main.py`) | Cap OpenMP thread pool. |

**Path resolution.** MCP hosts launch the server from arbitrary working
directories (Claude Desktop uses `/`), so a relative `csv_path` is resolved
against, in order: the process working directory, `LOAD_FORECASTING_DATA_DIR`,
the repository root, and `data/examples`. `data/examples/AMI/2021_city_level.csv`
and `AMI/2021_city_level.csv` both work from any client. Output paths
(`output_csv_path`, `output_html_path`) are *not* resolved this way — pass
them absolute. The server also advertises the dataset directory and these
rules in its MCP `instructions`, which most hosts place in the model's context.

## Troubleshooting

**Common Issues:**

1. **"Module not found"**: Run `uv sync` (or `pip install -e ".[dev]"`) to
   install dependencies.
2. **Python 3.10 or 3.13 errors on install**: Python 3.10 is not supported
   because the pinned numpy requires ≥ 3.11; Python 3.13 is not supported due
   to a numba/llvmlite incompatibility. Use Python 3.11–3.12.
3. **`XGBoost Library (libxgboost.dylib) could not be loaded` on macOS**: the
   OpenMP runtime is missing. Run `brew install libomp` and relaunch.
4. **Process crashes (SIGSEGV) on macOS after mixing XGBoost and Torch models**:
   an OpenMP duplicate-runtime conflict. `main.py` sets `KMP_DUPLICATE_LIB_OK=TRUE`
   and `OMP_NUM_THREADS=1` automatically, and `tests/conftest.py` applies the
   same pre-import guard for pytest. If you launch Python through another entry
   point, export both variables yourself before importing torch/xgboost.
5. **LSTM/TFT wrong results or NaN losses on Apple Silicon (MPS)**: known
   PyTorch MPS bugs. Workarounds — for LSTM set `dropout=0` or `n_rnn_layers=1`;
   for TFT pass `accelerator="cpu"` via `model_kwargs`.
6. **TimesFM first run is slow / downloads data**: TimesFM 2.5 downloads ~800MB
   of pretrained weights from HuggingFace Hub on first use.
7. **"Data frequency mismatch" during evaluation**: the CSV frequency differs
   from the model's training frequency. Provide a matching CSV or retrain.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run the test suite:

   ```bash
   pytest
   ```

   Default `pytest` excludes `@pytest.mark.slow` and `@pytest.mark.acceptance`.
   Torch training, year-long Naive walk-forward, and stride=1 backtests:

   ```bash
   pytest -m slow
   ```

   Real AMI pipeline, paper replication, and the AMI scale/model matrix:

   ```bash
   pytest -m acceptance
   ```

   Real TimesFM inference is also opt-in so routine tests never trigger the
   pretrained-weight download. Point `STLF_TIMESFM_LOCAL_DIR` at a cached
   checkpoint before running its smoke test:

   ```bash
   STLF_TIMESFM_LOCAL_DIR=/path/to/checkpoint \
     pytest -m acceptance tests/integration/test_timesfm_acceptance.py
   ```

5. Submit a pull request
