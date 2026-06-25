"""Suite 3: GPU stability tests — determinism, no NaN, memory cleanup."""

import gc
import pytest
import numpy as np
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def tcga_full():
    path = FIXTURES_DIR / "tcga_full.npz"
    if not path.exists():
        pytest.skip("Run generate_stress_data.py first")
    return np.load(path, allow_pickle=True)


@pytest.fixture(scope="module")
def tcga_top500():
    path = FIXTURES_DIR / "tcga_top500.npz"
    if not path.exists():
        pytest.skip("Run generate_stress_data.py first")
    return np.load(path, allow_pickle=True)


@pytest.mark.gpu
class TestDeterminism:
    """GPU should produce deterministic output for the same input."""

    def test_batch_auc_deterministic(self, tcga_full):
        from pystatsbio.diagnostic import batch_auc

        expression = tcga_full["expression"][:, :100].astype(np.float64)
        labels = tcga_full["labels"]

        results = []
        for _ in range(3):
            r = batch_auc(labels, expression, backend="gpu")
            results.append(r.auc.copy())

        for i in range(1, 3):
            max_diff = np.max(np.abs(results[i] - results[0]))
            assert max_diff < 1e-6, f"Run {i} differs from run 0 by {max_diff:.2e}"

    def test_ols_deterministic(self):
        from pystatistics.regression import Design, fit

        np.random.seed(42)
        X = np.random.normal(0, 1, (10000, 51))
        X[:, 0] = 1.0
        y = np.random.normal(0, 1, 10000)
        design = Design.from_arrays(X, y)

        results = []
        for _ in range(3):
            r = fit(design, backend="gpu")
            results.append(r.coefficients.copy())

        for i in range(1, 3):
            max_diff = np.max(np.abs(results[i] - results[0]))
            assert max_diff < 1e-6, f"OLS run {i} differs from run 0 by {max_diff:.2e}"

    def test_glm_deterministic(self, tcga_top500):
        from pystatistics.regression import Design, fit

        expression = tcga_top500["expression"][:, :50]
        labels = tcga_top500["labels"]

        X = np.column_stack([np.ones(len(labels)), expression])
        y = labels.astype(np.float64)
        design = Design.from_arrays(X, y)

        results = []
        for _ in range(3):
            r = fit(design, family="binomial", backend="gpu")
            results.append(r.coefficients.copy())

        for i in range(1, 3):
            corr = np.corrcoef(results[i], results[0])[0, 1]
            assert corr > 0.99999, f"GLM run {i} correlation: {corr:.8f}"


@pytest.mark.gpu
class TestNoSilentFailures:
    """GPU must not quietly produce NaN or Inf."""

    def test_batch_auc_no_nan(self, tcga_full):
        from pystatsbio.diagnostic import batch_auc

        expression = tcga_full["expression"][:, :100].astype(np.float64)
        labels = tcga_full["labels"]

        result = batch_auc(labels, expression, backend="gpu")
        assert np.all(np.isfinite(result.auc)), (
            f"NaN/Inf in batch AUC: {np.sum(~np.isfinite(result.auc))} bad values"
        )

    def test_ols_no_nan(self):
        from pystatistics.regression import Design, fit

        np.random.seed(42)
        X = np.random.normal(0, 1, (100_000, 51))
        X[:, 0] = 1.0
        y = np.random.normal(0, 1, 100_000)
        design = Design.from_arrays(X, y)

        result = fit(design, backend="gpu")
        assert np.all(np.isfinite(result.coefficients)), "NaN/Inf in OLS coefficients"

    def test_glm_no_nan(self, tcga_top500):
        from pystatistics.regression import Design, fit

        expression = tcga_top500["expression"][:, :50]
        labels = tcga_top500["labels"]

        X = np.column_stack([np.ones(len(labels)), expression])
        y = labels.astype(np.float64)
        design = Design.from_arrays(X, y)

        result = fit(design, family="binomial", backend="gpu")
        assert np.all(np.isfinite(result.coefficients)), "NaN/Inf in GLM coefficients"


@pytest.mark.gpu
class TestMemoryCleanup:
    """GPU memory should be released after computation."""

    def test_memory_released(self):
        import torch
        from pystatistics.regression import Design, fit

        np.random.seed(42)
        X = np.random.normal(0, 1, (200_000, 101))
        X[:, 0] = 1.0
        y = np.random.normal(0, 1, 200_000)
        design = Design.from_arrays(X, y)

        result = fit(design, backend="gpu")
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        allocated = torch.cuda.memory_allocated(0) / 1e6  # MB
        assert allocated < 100, (
            f"GPU memory not released: {allocated:.0f} MB still allocated"
        )


@pytest.mark.slow
@pytest.mark.gpu
class TestFullPipeline:
    """End-to-end: screen genes -> select top -> fit GLM -> verify."""

    def test_full_pipeline(self, tcga_full, tcga_top500):
        from pystatsbio.diagnostic import batch_auc
        from pystatistics.regression import Design, fit

        expression = tcga_full["expression"][:, :200].astype(np.float64)
        labels = tcga_full["labels"]

        # Step 1: Screen 200 genes via batch AUC on GPU
        auc_result = batch_auc(labels, expression, backend="gpu")
        assert auc_result.n_markers == 200

        # Step 2: Select top 20 genes
        auc_dir = np.maximum(auc_result.auc, 1 - auc_result.auc)
        top20_idx = np.argsort(auc_dir)[-20:]
        top20_expr = expression[:, top20_idx]

        # Step 3: Fit GLM on GPU
        X = np.column_stack([np.ones(len(labels)), top20_expr])
        y = labels.astype(np.float64)
        design = Design.from_arrays(X, y)
        glm_result = fit(design, family="binomial", backend="gpu")
        assert np.all(np.isfinite(glm_result.coefficients))

        print(f"\n  Pipeline complete:")
        print(f"    Screened {auc_result.n_markers} genes")
        print(f"    Top AUC: {auc_dir[top20_idx[-1]]:.4f}")
        print(f"    GLM deviance: {glm_result.deviance:.2f}")
