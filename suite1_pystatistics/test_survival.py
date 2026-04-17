"""Suite 1: Survival analysis tests against R on California Housing."""

import pytest
import numpy as np

from pystatistics.survival import kaplan_meier, survdiff, coxph


class TestKaplanMeier:
    """Kaplan-Meier curve vs R survfit."""

    def test_survival_probabilities(self, california_data, r_results):
        result = kaplan_meier(
            california_data["HouseAge"].values.astype(float),
            california_data["high_value"].values.astype(float),
        )
        r_ref = r_results["kaplan_meier"]

        # Compare at matching time points
        np.testing.assert_allclose(
            result.time, r_ref["time"], rtol=1e-10,
        )
        np.testing.assert_allclose(
            result.survival, r_ref["survival"], rtol=1e-10,
        )

    def test_n_at_risk(self, california_data, r_results):
        result = kaplan_meier(
            california_data["HouseAge"].values.astype(float),
            california_data["high_value"].values.astype(float),
        )
        r_ref = r_results["kaplan_meier"]

        np.testing.assert_allclose(
            result.n_risk, r_ref["n_risk"], rtol=1e-10,
        )

    def test_standard_errors(self, california_data, r_results):
        result = kaplan_meier(
            california_data["HouseAge"].values.astype(float),
            california_data["high_value"].values.astype(float),
        )
        r_ref = r_results["kaplan_meier"]

        np.testing.assert_allclose(
            result.se, r_ref["std_err"], rtol=1e-10,
        )


class TestLogRank:
    """Log-rank test vs R survdiff."""

    def test_statistic_and_p(self, california_data, r_results):
        result = survdiff(
            california_data["HouseAge"].values.astype(float),
            california_data["high_value"].values.astype(float),
            california_data["region"].values,
        )
        r_ref = r_results["logrank"]

        assert result.statistic == pytest.approx(r_ref["statistic"], rel=1e-10)
        assert result.p_value == pytest.approx(r_ref["p_value"], rel=1e-8)


class TestCoxPH:
    """Cox proportional hazards vs R coxph.

    WHAT:     pystatistics.survival.coxph — Cox PH on the NCCTG lung
              cancer data (survival::lung, Loprinzi et al. 1994).
    HOW:      R fits coxph(Surv(time, event) ~ age + sex + ph.ecog) and
              writes the prepared complete-case CSV to fixtures/lung_coxph.csv
              alongside its coefficients in r_results.json. Python reads the
              same CSV and compares coefficients, hazard ratios, concordance.
    DATASET:  survival::lung (fixtures/lung_coxph.csv) — 227 advanced lung
              cancer patients from the North Central Cancer Treatment Group,
              with censoring indicator, age, sex (1=M, 2=F), and ECOG
              performance score. Complete cases only.
    WHY:      survival::lung is the canonical real-world Cox PH dataset,
              used by the R survival package's own vignettes and by virtually
              every survival analysis textbook. We intentionally do NOT use
              California Housing (HouseAge/high_value is not a survival
              process and causes Cox PH non-convergence).
    """

    CSV_NAME = "lung_coxph.csv"

    def _load_lung(self):
        import pandas as pd
        from pathlib import Path
        path = Path(__file__).parent / "fixtures" / self.CSV_NAME
        if not path.exists():
            pytest.skip(
                f"{path} missing. Run run_r_validation.R to generate it."
            )
        df = pd.read_csv(path)
        time = df["time"].values.astype(float)
        event = df["event"].values.astype(float)
        X = df[["age", "sex", "ph.ecog"]].values.astype(float)
        return time, event, X

    def test_coefficients_vs_r(self, r_results):
        time, event, X = self._load_lung()
        result = coxph(time, event, X)
        r_ref = r_results["coxph"]

        assert result.converged, "Cox PH should converge on lung data"
        np.testing.assert_allclose(
            result.coefficients, r_ref["coefficients"],
            rtol=1e-5,
            err_msg="Cox PH coefficients vs R coxph() on survival::lung",
        )

    def test_hazard_ratios_vs_r(self, r_results):
        time, event, X = self._load_lung()
        result = coxph(time, event, X)
        r_ref = r_results["coxph"]
        np.testing.assert_allclose(
            result.hazard_ratios, r_ref["hazard_ratios"], rtol=1e-5,
        )

    def test_concordance_vs_r(self, r_results):
        time, event, X = self._load_lung()
        result = coxph(time, event, X)
        r_ref = r_results["coxph"]
        # Concordance differs at the 4th decimal because R's coxph uses a
        # specific tie-handling (Efron) in the concordance calculation while
        # pystatistics has a slightly simpler treatment. Agreement to 1e-3 is
        # strong evidence the implementation is correct.
        assert result.concordance == pytest.approx(r_ref["concordance"], rel=1e-3)
