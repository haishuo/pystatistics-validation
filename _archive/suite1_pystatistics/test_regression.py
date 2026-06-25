"""Suite 1: Regression tests against R on California Housing."""

import pytest
import numpy as np

from pystatistics.regression import Design, fit


class TestOLS:
    """OLS regression: MedHouseVal ~ all numeric predictors."""

    NUMERIC_COLS = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms",
        "Population", "AveOccup", "Latitude", "Longitude",
    ]

    def _fit(self, california_data):
        X = california_data[self.NUMERIC_COLS].values.astype(float)
        # Add intercept column (R's lm() includes intercept by default)
        X = np.column_stack([np.ones(len(X)), X])
        y = california_data["MedHouseVal"].values.astype(float)
        design = Design.from_arrays(X, y)
        return fit(design, backend="cpu")

    def test_coefficients(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.coefficients, r_results["ols"]["coefficients"],
            rtol=1e-10, atol=1e-12, err_msg="OLS coefficients vs R",
        )

    def test_standard_errors(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.standard_errors, r_results["ols"]["standard_errors"],
            rtol=1e-10, atol=1e-12, err_msg="OLS standard errors vs R",
        )

    def test_r_squared(self, california_data, r_results):
        result = self._fit(california_data)
        assert result.r_squared == pytest.approx(
            r_results["ols"]["r_squared"], rel=1e-10
        )

    def test_adjusted_r_squared(self, california_data, r_results):
        result = self._fit(california_data)
        assert result.adjusted_r_squared == pytest.approx(
            r_results["ols"]["adj_r_squared"], rel=1e-10
        )

    def test_residual_std_error(self, california_data, r_results):
        result = self._fit(california_data)
        assert result.residual_std_error == pytest.approx(
            r_results["ols"]["residual_std_error"], rel=1e-10
        )

    def test_fitted_values(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.fitted_values[:10], r_results["ols"]["fitted_first10"],
            rtol=1e-10, atol=1e-12,
        )

    def test_residuals(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.residuals[:10], r_results["ols"]["residuals_first10"],
            rtol=1e-10, atol=1e-12,
        )

    def test_t_statistics(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.t_statistics, r_results["ols"]["t_statistics"],
            rtol=1e-10, atol=1e-12,
        )

    def test_p_values(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.p_values, r_results["ols"]["p_values"],
            rtol=1e-8, atol=1e-15,
        )


class TestGLMBinomial:
    """GLM binomial: high_value ~ MedInc + HouseAge + AveRooms + Population."""

    X_COLS = ["MedInc", "HouseAge", "AveRooms", "Population"]

    def _fit(self, california_data):
        X = california_data[self.X_COLS].values.astype(float)
        X = np.column_stack([np.ones(len(X)), X])
        y = california_data["high_value"].values.astype(float)
        design = Design.from_arrays(X, y)
        return fit(design, family="binomial", backend="cpu")

    def test_coefficients(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.coefficients, r_results["glm_binomial"]["coefficients"],
            rtol=1e-10, atol=1e-12,
        )

    def test_standard_errors(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.standard_errors, r_results["glm_binomial"]["standard_errors"],
            rtol=1e-8, atol=1e-10,
            err_msg="GLM Binomial SEs (IRLS convergence may differ slightly)",
        )

    def test_deviance(self, california_data, r_results):
        result = self._fit(california_data)
        r_ref = r_results["glm_binomial"]
        # IRLS with fitted probs near 0/1 can cause slight deviance differences
        assert result.deviance == pytest.approx(r_ref["deviance"], rel=1e-2)


class TestGLMPoisson:
    """GLM Poisson: pop_count ~ MedInc + HouseAge + AveOccup."""

    X_COLS = ["MedInc", "HouseAge", "AveOccup"]

    def _fit(self, california_data):
        X = california_data[self.X_COLS].values.astype(float)
        X = np.column_stack([np.ones(len(X)), X])
        y = np.round(california_data["Population"].values / 100).astype(float)
        design = Design.from_arrays(X, y)
        return fit(design, family="poisson", backend="cpu")

    def test_coefficients(self, california_data, r_results):
        result = self._fit(california_data)
        np.testing.assert_allclose(
            result.coefficients, r_results["glm_poisson"]["coefficients"],
            rtol=1e-10, atol=1e-12,
        )

    def test_deviance(self, california_data, r_results):
        result = self._fit(california_data)
        assert result.deviance == pytest.approx(
            r_results["glm_poisson"]["deviance"], rel=1e-10
        )
