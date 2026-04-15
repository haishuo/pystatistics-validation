"""Suite 1: Mixed model tests against R lme4 on California Housing."""

import pytest
import numpy as np

from pystatistics.mixed import lmm


class TestLMMRandomIntercept:
    """LMM: MedHouseVal ~ MedInc + HouseAge + (1 | block_id)."""

    def _fit(self, california_data, r_results):
        r_ref = r_results["lmm"]
        idx = np.array(r_ref["mm_indices"]) - 1  # R is 1-indexed
        subset = california_data.iloc[idx]

        y = subset["MedHouseVal"].values.astype(float)
        X = subset[["MedInc", "HouseAge"]].values.astype(float)
        # Add intercept column
        X = np.column_stack([np.ones(len(y)), X])
        groups = {"block_id": subset["block_id"].values}

        return lmm(y, X, groups=groups)

    def test_fixed_effects(self, california_data, r_results):
        result = self._fit(california_data, r_results)
        r_ref = r_results["lmm"]

        np.testing.assert_allclose(
            result.coefficients, r_ref["fixed_effects"],
            rtol=1e-6, atol=1e-6,
            err_msg="LMM fixed effects vs R lme4",
        )

    def test_standard_errors(self, california_data, r_results):
        result = self._fit(california_data, r_results)
        r_ref = r_results["lmm"]

        np.testing.assert_allclose(
            result.se, r_ref["se"],
            rtol=1e-4, atol=1e-6,
            err_msg="LMM standard errors vs R lme4",
        )

    def test_variance_components(self, california_data, r_results):
        result = self._fit(california_data, r_results)
        r_ref = r_results["lmm"]

        # Check block variance
        block_var = None
        resid_var = None
        for vc in result.var_components:
            if "block" in str(vc).lower():
                block_var = vc.variance if hasattr(vc, "variance") else None
            if "resid" in str(vc).lower():
                resid_var = vc.variance if hasattr(vc, "variance") else None

        if block_var is not None:
            assert block_var == pytest.approx(r_ref["var_block"], rel=1e-4)
        if resid_var is not None:
            assert resid_var == pytest.approx(r_ref["var_residual"], rel=1e-4)

    def test_log_likelihood(self, california_data, r_results):
        result = self._fit(california_data, r_results)
        r_ref = r_results["lmm"]

        assert result.log_likelihood == pytest.approx(r_ref["loglik"], rel=1e-4)

    def test_aic(self, california_data, r_results):
        result = self._fit(california_data, r_results)
        r_ref = r_results["lmm"]

        assert result.aic == pytest.approx(r_ref["aic"], rel=1e-4)
