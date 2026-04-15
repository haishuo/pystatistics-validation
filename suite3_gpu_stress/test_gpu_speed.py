"""Suite 3: GPU speed tests — verify GPU is meaningfully faster than CPU.

Focus on OLS regression where GPU demonstrably excels (12x+ on 500K x 200).
Batch AUC and bootstrap/permutation GPU backends are either not yet implemented
or don't benefit at the TCGA scale (1155 samples).
"""

import time
import pytest
import numpy as np


def _gpu_time(fn, warmup=1, repeats=3):
    """Time a GPU function with warmup and median of repeats."""
    import torch

    for _ in range(warmup):
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return np.median(times)


@pytest.mark.slow
@pytest.mark.gpu
class TestGPUSpeed:
    """GPU speedup vs CPU — the raison d'etre of GPU backends."""

    def test_ols_large_speedup(self):
        """OLS 500K x 200: GPU should be >= 3x faster than CPU."""
        from pystatistics.regression import Design, fit

        np.random.seed(42)
        n, p = 500_000, 200
        X = np.random.normal(0, 1, (n, p + 1))
        X[:, 0] = 1.0
        y = np.random.normal(0, 1, n)
        design = Design.from_arrays(X, y)

        # CPU timing
        t0 = time.perf_counter()
        fit(design, backend="cpu")
        cpu_time = time.perf_counter() - t0

        # GPU timing
        gpu_time = _gpu_time(lambda: fit(design, backend="gpu"))

        speedup = cpu_time / gpu_time
        print(f"\n  OLS 500Kx200: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, speedup={speedup:.1f}x")
        assert speedup > 3.0, f"OLS GPU speedup only {speedup:.1f}x (need >3x)"

    def test_ols_million_speedup(self):
        """OLS 1M x 100: GPU should be >= 5x faster than CPU."""
        from pystatistics.regression import Design, fit

        np.random.seed(43)
        n, p = 1_000_000, 100
        X = np.random.normal(0, 1, (n, p + 1)).astype(np.float64)
        X[:, 0] = 1.0
        y = np.random.normal(0, 1, n)
        design = Design.from_arrays(X, y)

        t0 = time.perf_counter()
        fit(design, backend="cpu")
        cpu_time = time.perf_counter() - t0

        gpu_time = _gpu_time(lambda: fit(design, backend="gpu"))

        speedup = cpu_time / gpu_time
        print(f"\n  OLS 1Mx100: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, speedup={speedup:.1f}x")
        assert speedup > 5.0, f"OLS GPU speedup only {speedup:.1f}x (need >5x)"

    def test_glm_large_speedup(self):
        """GLM binomial 50K x 100: GPU should be >= 2x faster than CPU."""
        from pystatistics.regression import Design, fit

        np.random.seed(44)
        n, p = 50_000, 100
        X = np.random.normal(0, 1, (n, p + 1)).astype(np.float64)
        X[:, 0] = 1.0
        y = (np.random.normal(0, 1, n) > 0).astype(np.float64)
        design = Design.from_arrays(X, y)

        t0 = time.perf_counter()
        fit(design, family="binomial", backend="cpu")
        cpu_time = time.perf_counter() - t0

        gpu_time = _gpu_time(lambda: fit(design, family="binomial", backend="gpu"))

        speedup = cpu_time / gpu_time
        print(f"\n  GLM 50Kx100: CPU={cpu_time:.2f}s, GPU={gpu_time:.2f}s, speedup={speedup:.1f}x")
        assert speedup > 2.0, f"GLM GPU speedup only {speedup:.1f}x (need >2x)"
