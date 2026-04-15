"""Shared test infrastructure for PyStatistics/PyStatsBio Linux validation."""

import pytest
import numpy as np


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


GPU_AVAILABLE = _gpu_available()


@pytest.fixture(scope="session")
def gpu_available():
    return GPU_AVAILABLE


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU tests when no CUDA GPU present."""
    if not GPU_AVAILABLE:
        skip_gpu = pytest.mark.skip(reason="No CUDA GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Tolerance helpers — match the tiers documented in pystatistics README
# ---------------------------------------------------------------------------

def assert_close(actual, expected, rtol=1e-10, atol=1e-12, label=""):
    """Assert numerical closeness with informative error message."""
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol,
        err_msg=f"Mismatch in {label}" if label else "",
    )


def cpu_r_tol(is_ill_conditioned=False):
    """Tolerances for CPU vs R comparison."""
    if is_ill_conditioned:
        return dict(rtol=1e-4, atol=1e-6)
    return dict(rtol=1e-10, atol=1e-12)


def gpu_cpu_tol():
    """Tolerances for GPU vs CPU comparison (FP32 vs FP64)."""
    return dict(rtol=1e-3, atol=1e-5)
