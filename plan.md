## **Technical Subtasks**

Items in **bold** are MVP. Items in regular text are nice-to-have, picked up if time permits. Deliverables are tagged by type: `[mcp_tool]`, `[model]`, `[test]`, `[docker]`, `[doc]`.

### **Current State**

| Area | Status |
| :---- | :---- |
| Models | Naive baselines and Linear Regression working |
| Training | train\_forecast\_model shipped (past covariates only) |
| Evaluation | evaluate\_forecast\_model shipped |
| Data pipeline | Preprocessing hardened: frequency validation, 90% coverage check, calendar features, lag features (24h/48h/168h), clearer errors |
| Inference | Missing. No way to forecast forward in time |
| Hyperparameter tuning | Missing |
| Tests | 146 tests (35 data loader + 9 future-cov unit + 5 registry + 30 inspect_data + 13 AMI integration + 10 tools integration + 44 multi-resolution integration) |
| Deployment | Local only. No Dockerfile |
| Docs | README and design notes exist, but no runnable end-to-end examples |

### **Subtask 1\. Data pipeline**

1. ~~**Strengthen preprocessing in the data loader: feature standardization, input validation, basic feature engineering (calendar features, lag features), and clear error messages.**~~ ✓ Done — frequency inference/validation, 90% coverage check, negative-value warning, calendar features (6 columns), lag features (24h/48h/168h), actionable error messages, bug fix in evaluate_forecast_model.
  2. ~~**Add future covariates support to `train_forecast_model` and `evaluate_forecast_model`.**~~ ✓ Done — explicit future_covariates key in column_mapping; lags_future_covariates wired into LinearRegression; future_covariate_scaler fitted, persisted, and reloaded on evaluate; 9 unit tests + 3 integration tests with real weather data.
  3. ~~**Build a new `inspect_data` MCP tool that takes a CSV and reports detected columns, frequency, gaps, basic statistics, and feature suggestions for the agent.**~~ ✓ Done — column detection, frequency inference/validation, time range, per-column stats (mean/std/min/max/p5/p95/n_missing/n_negative), gap analysis with top-5 list, typed quality_flags, actionable suggestions, ready_to_train flag; 30 tests.  
  4. ~~**Validate the pipeline against the provided sample data at all three spatial resolutions.**~~ ✓ Done — 13 integration tests in tests/integration/test_ami_data.py covering inspect_data + train + evaluate on city (district), GLENDOVEER substation, and GLENDOVEER-13599 feeder circuits using 2021 AMI data merged with Open-Meteo weather; future covariates tested end-to-end with real temperature data.

**Deliverables**

* ~~`[mcp_tool]` `train_forecast_model` and `evaluate_forecast_model` extended with future covariates and richer preprocessing, validated on feeder, substation, and district level sample data.~~ ✓ Shipped.
* ~~`[mcp_tool]` New `inspect_data` MCP tool for assessing a CSV before training.~~ ✓ Shipped.

### **Subtask 2\. Model development**

1. **Add XGBoost as a model type in `train_forecast_model`.**  
2. **Add LSTM as the first deep learning model type in `train_forecast_model`.**  
3. **Explore hyperparameter tuning options (e.g., Optuna, local grid search frameworks, Darts utilities) and build a new `tune_model` MCP tool that takes a model type, training data, and a search space, returns the best configuration, and registers the tuned model. Designed for deep learning models like LSTM and to generalize to other models (e.g., TFT, NHiTS).**  
4. Add ARIMA as a model type.  
5. Add TFT as a model type.

**Deliverables**

* `[model]` XGBoost and LSTM model types added to `train_forecast_model`.  
* `[mcp_tool]` New `tune_model` MCP tool for hyperparameter optimization, validated with LSTM.

### **Subtask 3\. Inference**

1. **Decide the inference tool's input, output, what to do when the recent data is shorter than the model needs, and how future covariates flow through.**  
2. **Build a new `generate_forecast` MCP tool that takes a model ID, recent context, and (optionally) future covariates, and returns predictions over a requested horizon.**  
3. **Build a new `backtest_model` MCP tool that runs rolling-window backtests on historical data and returns metrics, sharing the forecasting code with `generate_forecast`.**  
4. Probabilistic forecasts with optional uncertainty bands, plus quantile-aware metrics in the evaluator.  
5. `compare_models` MCP tool that takes a CSV and a list of model types, trains each, and returns a ranked leaderboard.

**Deliverables**

* `[mcp_tool]` New `generate_forecast` MCP tool for forward forecasting from a trained model.  
* `[mcp_tool]` New `backtest_model` MCP tool for rolling-window evaluation on historical data.

### **Subtask 4\. Testing, packaging, and docs**

1. **Write unit tests for the main code (training, evaluation, frequency utilities, spec loader) and for all MCP tools, covering both normal and error cases.**  
2. **Create a docker image for starting and running the MCP server.**  
3. **Confirm MCP clients (e.g., Claude Code, Goose AI, Codex) can connect to the local Docker instance and call the tools.**  
4. **Update the README, write a one-page architecture summary, and add one runnable quickstart example.**  
5. Add one end-to-end test that trains a model, lists it, and evaluates it on a sample CSV.  
6. Set up GitHub Actions to run the tests automatically when code is pushed.  
7. Live cloud deployment: compare GCP, AWS, and LBNL Lawrencium; deploy one instance with public HTTPS; validate from multiple clients.  
8. Contributor guide and more examples (model comparison, probabilistic forecasting, weather-aware forecasting).

**Deliverables**

* `[test]` Unit tests covering the core modules and every MCP tool.  
* `[docker]` `Dockerfile` and `docker-compose.yml` in the repo; server runnable via `docker compose up` and confirmed from at least two MCP clients.  
* `[doc]` Updated README, one-page architecture summary, and one runnable quickstart example.
