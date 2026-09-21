# Per-model hyperparameter reference (train-forecast-model, Section 3.6.1)

Load this file at the §3.6 gate and show ONLY the table for the model the user selected. Defaults follow Li et al. (2025) where the paper specifies them; otherwise the library default applies.

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

