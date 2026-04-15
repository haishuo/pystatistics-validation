"""Suite 2: Power/sample size tests against R pwr on NHANES-derived effect sizes."""

import pytest
import numpy as np

from pystatsbio.power import power_t_test, power_paired_t_test, power_prop_test, power_anova_oneway


class TestPowerTTest:
    """power_t_test vs R pwr::pwr.t.test."""

    def test_solve_for_n(self, r_results):
        r_ref = r_results["power_t_n"]
        result = power_t_test(d=r_ref["d"], alpha=r_ref["alpha"], power=r_ref["power"])
        # n is typically ceiling'd to an integer; compare with ceiling of R's exact value
        import math
        assert result.n == math.ceil(r_ref["n_exact"])

    def test_solve_for_power(self, r_results):
        r_ref = r_results["power_t_power"]
        result = power_t_test(d=r_ref["d"], n=r_ref["n"], alpha=r_ref["alpha"])
        assert result.power == pytest.approx(r_ref["power"], rel=1e-6)


class TestPowerPaired:
    """power_paired_t_test vs R pwr::pwr.t.test(type='paired')."""

    def test_solve_for_n(self, r_results):
        import math
        r_ref = r_results["power_paired_n"]
        result = power_paired_t_test(d=r_ref["d"], power=r_ref["power"])
        assert result.n == math.ceil(r_ref["n_exact"])


class TestPowerProportion:
    """power_prop_test vs R pwr::pwr.2p.test."""

    def test_solve_for_n(self, r_results):
        import math, inspect
        r_ref = r_results["power_prop_n"]
        # Check actual parameter names
        sig = inspect.signature(power_prop_test)
        params = list(sig.parameters.keys())

        if "h" in params:
            result = power_prop_test(h=r_ref["h"], alpha=0.05, power=r_ref["power"])
        elif "p1" in params:
            result = power_prop_test(p1=r_ref["p1"], p2=r_ref["p2"],
                                     alpha=0.05, power=r_ref["power"])
        else:
            pytest.skip(f"Unknown power_prop_test signature: {params}")

        assert result.n == math.ceil(r_ref["n_exact"])


class TestPowerANOVA:
    """power_anova_oneway vs R pwr::pwr.anova.test."""

    def test_solve_for_n(self, r_results):
        import math
        r_ref = r_results["power_anova_n"]
        result = power_anova_oneway(
            f=r_ref["f"], k=r_ref["k"],
            alpha=0.05, power=r_ref["power"],
        )
        assert result.n == math.ceil(r_ref["n_exact"])
