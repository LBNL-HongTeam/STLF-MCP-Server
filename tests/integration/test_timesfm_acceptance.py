"""Opt-in smoke test for the real TimesFM 2.5 Darts backbone.

Set ``STLF_TIMESFM_LOCAL_DIR`` to a downloaded TimesFM checkpoint and run
``pytest -m acceptance tests/integration/test_timesfm_acceptance.py``. Normal
test runs remain offline and do not download the approximately 800 MB model.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from darts import TimeSeries

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from load_forecasting.core import trainer


pytestmark = pytest.mark.acceptance


def test_real_timesfm_zero_shot_predicts_finite_horizon():
    local_dir = os.environ.get("STLF_TIMESFM_LOCAL_DIR")
    if not local_dir:
        pytest.skip("Set STLF_TIMESFM_LOCAL_DIR to run the real TimesFM smoke test")
    if not Path(local_dir).is_dir():
        pytest.fail(f"STLF_TIMESFM_LOCAL_DIR is not a directory: {local_dir}")
    if not trainer._HAS_TIMESFM:
        pytest.fail("The installed Darts build does not provide TimesFM2p5Model")

    periods = 240
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    values = 100 + 15 * np.sin(2 * np.pi * np.arange(periods) / 24)
    series = TimeSeries.from_times_and_values(index, values)
    model = trainer.create_model(
        "TimesFM",
        lookback=168,
        horizon=24,
        accelerator="cpu",
        enable_finetuning=False,
        n_epochs=0,
        local_dir=local_dir,
    )

    forecast = model.predict(24, series=series)

    assert len(forecast) == 24
    assert np.isfinite(forecast.values()).all()
