"""Test-process safeguards that must be applied before ML libraries import."""

import os


# trainer.py imports both XGBoost and Torch. On macOS they can resolve separate
# libomp runtimes, which may segfault as soon as a Torch model starts parallel
# tensor work. Match the server entry-point guard for direct pytest execution.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
