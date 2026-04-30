# LoadForecasting-MCP

MCP server for building load forecasting with time-series models.

## Overview

This MCP server provides tools for training, evaluating, and managing time-series forecasting models for building electrical loads. It uses [Darts](https://github.com/unit8co/darts) for forecasting and integrates with the AlphaBuilding platform via the Model Context Protocol (MCP).

## Features

- **Train forecasting models** on historical building load data
- **Evaluate models** on test data with comprehensive metrics
- **Model registry** for persistent storage and versioning
- **Auto-detection** of column mappings from CSV files
- **Multiple model types**: Naive baselines, Linear Regression (Phase 1), XGBoost, LSTM (Phase 4)

## Installation

```bash
# From the LoadForecasting-MCP directory
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Usage

### STDIO Mode (Claude Desktop)

```bash
python main.py
```

Add to Claude Desktop config (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "load_forecasting": {
      "command": "python",
      "args": ["/path/to/LoadForecasting-MCP/main.py"]
    }
  }
}
```

### HTTP Mode (AlphaBuilding-Agents)

```bash
python main.py --http
# or
MCP_TRANSPORT=http python main.py
```

Server runs on port 8003 by default.

## Tools

### `train_forecast_model`

Train a time-series forecasting model on historical data.

**Inputs:**
- `csv_path` (required): Path to CSV with datetime and load data
- `model_type`: NaiveMean, NaiveSeasonal, NaiveMovingAverage, LinearRegression
- `lookback_hours`: Hours of history for input (default: 24)
- `horizon_hours`: Hours ahead to forecast (default: 6)
- `frequency`: Data frequency - 15min, 30min, h (default: h)
- `validation_split`: Fraction for validation (default: 0.2)
- `building_name`: Building identifier
- `column_mapping`: Map CSV columns to roles

**Outputs:**
- `model_id`: Unique identifier for the trained model
- `training_metrics`: RMSE, MAE, MAPE on training data
- `validation_metrics`: RMSE, MAE, MAPE, CV-RMSE, R² on validation data
- `data_summary`: Statistics about input data

### `evaluate_forecast_model`

Evaluate a trained model on new/test data.

**Inputs:**
- `model_id` (required): ID from training
- `csv_path` (required): Path to test data CSV
- `return_predictions`: Include predictions in output (default: true)
- `output_csv_path`: Save predictions to CSV
- `include_residual_analysis`: Add residual statistics

**Outputs:**
- `test_metrics`: Performance on test data
- `comparison_to_validation`: Compare to validation metrics
- `predictions`: Timestamped predictions with residuals

### `list_models`

List trained models in the registry.

**Inputs:**
- `building_name`: Filter by building
- `model_type`: Filter by model type
- `sort_by`: created_at, validation_cv_rmse, model_type
- `limit`: Maximum results (default: 20)

### `get_algorithm_specifications`

Return all YAML specs for AI agent discovery.

## Data Format

### Input CSV

The tool expects a CSV with:
- A datetime column (auto-detected: timestamp, datetime, date, time)
- A target column for load (auto-detected: kwh, load, power, energy, electricity)
- Optional covariate columns (auto-detected: temp, humidity, solar, wind)

Example:
```csv
timestamp,electricity_kwh,outdoor_temp
2023-01-01 00:00:00,150.5,45.2
2023-01-01 01:00:00,142.3,44.8
...
```

### Column Mapping

If auto-detection fails, provide explicit mapping:

```python
{
    "datetime": "timestamp",
    "target": "electricity_kwh",
    "past_covariates": ["outdoor_temp"]
}
```

## Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| CV-RMSE | Coefficient of Variation of RMSE | < 15% excellent, 15-25% acceptable |
| R² | Coefficient of determination | > 0.8 |
| MAPE | Mean Absolute Percentage Error | < 10% |

## Model Registry

Trained models are stored in `./models/` (configurable via `LOAD_FORECASTING_MODEL_DIR`).

```
models/
├── registry.json                     # Index of all models
└── Building_33_LinearRegression_.../
    ├── model.pkl                     # Trained Darts model
    ├── metadata.json                 # Config, metrics, data info
    └── scalers/
        ├── target_scaler.pkl
        └── covariate_scaler.pkl
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOAD_FORECASTING_MODEL_DIR` | `./models` | Model storage directory |
| `MCP_TRANSPORT` | `stdio` | Transport mode (stdio or http) |
| `MCP_HTTP_PORT` | `8003` | HTTP port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Development

```bash
# Run tests
pytest

# Run specific test
pytest tests/test_data_loader.py -v
```

## Roadmap

| Phase | Features |
|-------|----------|
| 1 | train_forecast_model, get_algorithm_specifications |
| 2 | evaluate_forecast_model, list_models |
| 3 | generate_forecast (predict future) |
| 4 | XGBoost, ARIMA, LSTM, TFT models |
| 5 | compare_models (AutoML-lite) |

## License

MIT
