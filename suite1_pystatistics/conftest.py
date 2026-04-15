"""Suite 1 shared fixtures: California Housing data and R results."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def california_data():
    """Load the prepared California Housing dataset."""
    csv_path = FIXTURES_DIR / "california_prepared.csv"
    if not csv_path.exists():
        pytest.skip("Run generate_data.py first")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def r_results():
    """Load all R reference results."""
    r_path = FIXTURES_DIR / "r_results.json"
    if not r_path.exists():
        pytest.skip("Run run_r_validation.R first")
    with open(r_path) as f:
        return json.load(f)
