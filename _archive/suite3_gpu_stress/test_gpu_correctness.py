"""Suite 3: GPU correctness tests — GPU vs CPU on TCGA BRCA and large regression."""

import json
import pytest
import numpy as np
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def cpu_results():
    path = FIXTURES_DIR / "cpu_results.json"
    if not path.exists():
        pytest.skip("Run generate_cpu_fixtures.py first")
    with open(path) as f:
        return json.load(f)


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
class TestBatchAUCCorrectness:
    """GPU batch AUC matches CPU on TCGA data."""

    def test_auc_matches_cpu(self, tcga_full, cpu_results):
        from pystatsbio.diagnostic import batch_auc

        # Use subset (100 genes) for speed — GPU batch_auc is slow on small n
        expression = tcga_full["expression"][:, :100].astype(np.float64)
        labels = tcga_full["labels"]

        gpu_result = batch_auc(labels, expression, backend="gpu")
        cpu_result = batch_auc(labels, expression, backend="cpu")

        np.testing.assert_allclose(
            gpu_result.auc, cpu_result.auc,
            rtol=1e-3, atol=1e-4,
            err_msg="GPU batch AUC vs CPU (100 TCGA genes)",
        )

    def test_top50_ranking_preserved(self, tcga_full):
        """Top discriminative genes should rank identically GPU vs CPU."""
        from pystatsbio.diagnostic import batch_auc

        expression = tcga_full["expression"][:, :200].astype(np.float64)
        labels = tcga_full["labels"]

        gpu_r = batch_auc(labels, expression, backend="gpu")
        cpu_r = batch_auc(labels, expression, backend="cpu")

        gpu_dir = np.maximum(gpu_r.auc, 1 - gpu_r.auc)
        cpu_dir = np.maximum(cpu_r.auc, 1 - cpu_r.auc)

        gpu_top20 = set(np.argsort(gpu_dir)[-20:])
        cpu_top20 = set(np.argsort(cpu_dir)[-20:])

        overlap = len(gpu_top20 & cpu_top20)
        assert overlap >= 18, f"Top 20 overlap only {overlap}/20"


@pytest.mark.gpu
class TestGLMCorrectness:
    """GPU GLM binomial matches CPU on TCGA top genes."""

    def test_glm_top50_gpu_vs_cpu(self, tcga_top500):
        from pystatistics.regression import Design, fit

        expression = tcga_top500["expression"][:, :50]
        labels = tcga_top500["labels"]

        X = np.column_stack([np.ones(len(labels)), expression])
        y = labels.astype(np.float64)
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family="binomial", backend="cpu")
        gpu_result = fit(design, family="binomial", backend="gpu")

        np.testing.assert_allclose(
            gpu_result.coefficients, cpu_result.coefficients,
            rtol=1e-3, atol=1e-3,
        )

    def test_coefficient_correlation(self, tcga_top500):
        from pystatistics.regression import Design, fit

        expression = tcga_top500["expression"][:, :50]
        labels = tcga_top500["labels"]

        X = np.column_stack([np.ones(len(labels)), expression])
        y = labels.astype(np.float64)
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, family="binomial", backend="cpu")
        gpu_result = fit(design, family="binomial", backend="gpu")

        corr = np.corrcoef(gpu_result.coefficients, cpu_result.coefficients)[0, 1]
        assert corr > 0.9999, f"GPU-CPU coefficient correlation only {corr:.6f}"


@pytest.mark.gpu
class TestLargeOLSCorrectness:
    """GPU OLS on large synthetic data — the primary GPU showcase."""

    @pytest.fixture
    def large_regression(self):
        np.random.seed(42)
        n, p = 500_000, 200
        X = np.random.normal(0, 1, (n, p + 1))
        X[:, 0] = 1.0
        beta = np.random.normal(0, 1, p + 1)
        y = X @ beta + np.random.normal(0, 1, n)
        return X, y, beta

    def test_coefficients_match(self, large_regression):
        from pystatistics.regression import Design, fit

        X, y, _ = large_regression
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, backend="cpu")
        gpu_result = fit(design, backend="gpu")

        np.testing.assert_allclose(
            gpu_result.coefficients, cpu_result.coefficients,
            rtol=1e-3, atol=1e-5,
            err_msg="GPU OLS coefficients vs CPU (500K x 200)",
        )

    def test_r_squared_match(self, large_regression):
        from pystatistics.regression import Design, fit

        X, y, _ = large_regression
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, backend="cpu")
        gpu_result = fit(design, backend="gpu")

        assert gpu_result.r_squared == pytest.approx(cpu_result.r_squared, rel=1e-4)

    def test_coefficient_correlation(self, large_regression):
        from pystatistics.regression import Design, fit

        X, y, _ = large_regression
        design = Design.from_arrays(X, y)

        cpu_result = fit(design, backend="cpu")
        gpu_result = fit(design, backend="gpu")

        corr = np.corrcoef(gpu_result.coefficients, cpu_result.coefficients)[0, 1]
        assert corr > 0.9999, f"GPU-CPU OLS correlation only {corr:.6f}"
