"""Suite 2 shared fixtures: NHANES data and synthetic DR/PK data."""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="session")
def nhanes_data():
    """Load the prepared NHANES dataset."""
    csv_path = DATA_DIR / "nhanes_biomarker.csv"
    if not csv_path.exists():
        pytest.skip("Run download_datasets.py first")
    return pd.read_csv(csv_path)


@pytest.fixture(scope="session")
def dr_data():
    """Load synthetic dose-response data."""
    npz_path = FIXTURES_DIR / "dose_response_data.npz"
    if not npz_path.exists():
        pytest.skip("Run generate_data.py first")
    return np.load(npz_path, allow_pickle=True)


@pytest.fixture(scope="session")
def pk_data():
    """Load synthetic PK data."""
    npz_path = FIXTURES_DIR / "pk_data.npz"
    if not npz_path.exists():
        pytest.skip("Run generate_data.py first")
    return np.load(npz_path, allow_pickle=True)


@pytest.fixture(scope="session")
def r_results():
    """Load all R reference results."""
    r_path = FIXTURES_DIR / "r_results.json"
    if not r_path.exists():
        pytest.skip("Run run_r_validation.R first")
    with open(r_path) as f:
        return json.load(f)
