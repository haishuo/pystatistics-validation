"""Helpers for new pystatsbio module validation (suite 2).

See suite1_pystatistics/conftest_newmodules.py for the rationale behind the
runtime-parity policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

NEW_FIX = Path(__file__).parent / "fixtures" / "newmodules"


def load_json(name: str) -> dict:
    path = NEW_FIX / name
    if not path.exists():
        pytest.skip(f"Missing {path}. Run generate_newmodules_data.py.")
    with open(path) as f:
        return json.load(f)


def load_csv(name: str) -> pd.DataFrame:
    path = NEW_FIX / name
    if not path.exists():
        pytest.skip(f"Missing {path}. Run generate_newmodules_data.py.")
    return pd.read_csv(path)


def load_r_results() -> dict:
    path = NEW_FIX / "r_results.json"
    if not path.exists():
        pytest.skip(f"Missing {path}. Run run_r_newmodules.R.")
    with open(path) as f:
        return json.load(f)


MIN_R_TIME_FOR_RATIO = 0.05
MAX_RATIO = 20.0


def assert_runtime_parity(py_elapsed: float, r_elapsed: float, label: str) -> None:
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
