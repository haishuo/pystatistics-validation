"""Suite 1: Monte Carlo tests against R boot on California Housing."""

import pytest
import numpy as np

from pystatistics.montecarlo import boot, boot_ci


class TestBootstrap:
    """Bootstrap of MedInc mean vs R boot package."""

    def _mean_stat(self, data, indices):
        return np.array([np.mean(data[indices])])

    def test_t0(self, california_data, r_results):
        """t0 (observed statistic) must match R exactly — it's deterministic."""
        r_ref = r_results["bootstrap"]
        data = np.array(r_ref["data"])
        result = boot(data, self._mean_stat, R=2000, seed=46)

        assert result.t0[0] == pytest.approx(r_ref["t0"], rel=1e-10)

    def test_se(self, california_data, r_results):
        """Bootstrap SE — stochastic, so use wide tolerance."""
        r_ref = r_results["bootstrap"]
        data = np.array(r_ref["data"])
        result = boot(data, self._mean_stat, R=2000, seed=46)

        # SE should be in the right ballpark (within 20%)
        assert result.se[0] == pytest.approx(r_ref["se"], rel=0.2)

    def test_percentile_ci(self, california_data, r_results):
        """Percentile CI — stochastic, wide tolerance."""
        r_ref = r_results["bootstrap"]
        data = np.array(r_ref["data"])
        result = boot(data, self._mean_stat, R=2000, seed=46)
        ci_result = boot_ci(result, type="perc")

        ci = ci_result.ci["perc"]
        # CI bounds should be within 10% of R
        assert ci[0, 0] == pytest.approx(r_ref["ci_perc_lower"], rel=0.1)
        assert ci[0, 1] == pytest.approx(r_ref["ci_perc_upper"], rel=0.1)

    def test_bias(self, california_data, r_results):
        """Bootstrap bias estimate — stochastic."""
        r_ref = r_results["bootstrap"]
        data = np.array(r_ref["data"])
        result = boot(data, self._mean_stat, R=2000, seed=46)

        # Bias should be very small for the mean
        assert abs(result.bias[0]) < 0.1
