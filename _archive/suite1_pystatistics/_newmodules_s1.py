"""Shared helpers for new-module validation tests (suite 1).

These helpers are imported explicitly by each test_*.py under newmodules so
that pytest's auto-discovery of conftest.py (which provides california_data /
r_results for the original suite) is not disturbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NEW_FIX = Path(__file__).parent / "fixtures" / "newmodules"


def load_dataset(name: str) -> pd.DataFrame:
    path = NEW_FIX / name
    if not path.exists():
        import pytest
        pytest.skip(f"Missing dataset {path}. Run generate_newmodules_data.py.")
    return pd.read_csv(path)


def load_r_results() -> dict:
    path = NEW_FIX / "r_results.json"
    if not path.exists():
        import pytest
        pytest.skip(f"Missing {path}. Run run_r_newmodules.R.")
    with open(path) as f:
        return json.load(f)


# Runtime-parity policy
# --------------------
# We record the Python wall clock alongside the R wall clock. Failure modes we
# are trying to catch: Python solutions that are ORDERS OF MAGNITUDE slower
# than R (the 1000x slower Cox PH incident). We do NOT require Python to beat
# R; we require that it is "in the same league".
#
# Policy:
#   - For fits that R completes in >= 0.05s: Python must be within MAX_RATIO of
#     R runtime. Tasks below 0.05s are noise-dominated (wall clock resolution
#     and Python import overhead dwarf the actual compute) so we skip the
#     timing check — both are "instant".
MIN_R_TIME_FOR_RATIO = 0.05
MAX_RATIO = 20.0


def assert_runtime_parity(py_elapsed: float, r_elapsed: float, label: str) -> None:
    """Assert Python runtime is not egregiously slower than R."""
    if r_elapsed < MIN_R_TIME_FOR_RATIO:
        return
    ratio = py_elapsed / r_elapsed
    assert ratio <= MAX_RATIO, (
        f"[{label}] Python {py_elapsed:.3f}s vs R {r_elapsed:.3f}s "
        f"(ratio {ratio:.1f}x > {MAX_RATIO}x limit). "
        f"Python is egregiously slower than R."
    )


def to_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)
