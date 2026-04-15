"""Suite 1: Descriptive statistics tests against R on California Housing."""

import pytest
import numpy as np

from pystatistics.descriptive import describe, cor, quantile


NUMERIC_COLS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


class TestDescribe:
    """Column-wise summary statistics vs R."""

    def _data(self, california_data):
        return california_data[NUMERIC_COLS].values

    def test_means(self, california_data, r_results):
        result = describe(self._data(california_data), backend="cpu")
        np.testing.assert_allclose(
            result.mean, r_results["descriptive"]["means"],
            rtol=1e-10, atol=1e-12,
        )

    def test_sds(self, california_data, r_results):
        result = describe(self._data(california_data), backend="cpu")
        np.testing.assert_allclose(
            result.sd, r_results["descriptive"]["sds"],
            rtol=1e-10, atol=1e-12,
        )

    def test_skewness(self, california_data, r_results):
        result = describe(self._data(california_data), backend="cpu")
        np.testing.assert_allclose(
            result.skewness, r_results["descriptive"]["skewness"],
            rtol=1e-4, atol=1e-6,
            err_msg="Skewness (small formula differences between R moments and pystatistics)",
        )

    def test_kurtosis(self, california_data, r_results):
        """R moments::kurtosis returns regular kurtosis (excess+3), pystatistics returns excess."""
        result = describe(self._data(california_data), backend="cpu")
        r_kurtosis = np.array(r_results["descriptive"]["kurtosis"])
        # R moments::kurtosis = excess + 3, so subtract 3 to get excess
        np.testing.assert_allclose(
            result.kurtosis, r_kurtosis - 3.0,
            rtol=5e-3, atol=1e-2,
            err_msg="Kurtosis (R moments returns non-excess, pystatistics returns excess; "
                     "small formula differences expected for highly skewed distributions)",
        )


class TestQuantiles:
    """Quantiles (type 7) vs R."""

    def test_quartiles(self, california_data, r_results):
        data = california_data[NUMERIC_COLS].values
        result = quantile(data, probs=[0.25, 0.50, 0.75], type=7, backend="cpu")
        r_ref = r_results["descriptive"]

        np.testing.assert_allclose(
            result.quantiles[0], r_ref["quantiles_25"], rtol=1e-10,
        )
        np.testing.assert_allclose(
            result.quantiles[1], r_ref["quantiles_50"], rtol=1e-10,
        )
        np.testing.assert_allclose(
            result.quantiles[2], r_ref["quantiles_75"], rtol=1e-10,
        )


class TestCorrelation:
    """Correlation matrices vs R."""

    def test_pearson(self, california_data, r_results):
        data = california_data[NUMERIC_COLS].values
        result = cor(data, method="pearson", backend="cpu")
        r_cor = np.array(r_results["descriptive"]["cor_pearson"])
        np.testing.assert_allclose(
            result.correlation_matrix, r_cor, rtol=1e-10, atol=1e-12,
        )

    def test_spearman(self, california_data, r_results):
        data = california_data[NUMERIC_COLS].values
        result = cor(data, method="spearman", backend="cpu")
        r_cor = np.array(r_results["descriptive"]["cor_spearman"])
        np.testing.assert_allclose(
            result.correlation_matrix, r_cor, rtol=1e-10, atol=1e-12,
        )
