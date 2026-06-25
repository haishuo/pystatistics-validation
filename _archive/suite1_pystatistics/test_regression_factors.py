"""Suite 1: categorical predictors & interaction terms vs R.

Validates pystatistics' structured term spec (C(name, ref=...) + interaction
tuples) against R's model.matrix / lm / glm / survival::coxph on the same
data. Coefficient arrays are compared positionally — pystatistics emits
columns in R's model.matrix order (first factor varies fastest), so the
arrays line up element-for-element. R's coefficient names are recorded in the
fixtures for documentation; see r_results.json.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pystatistics import DataSource
from pystatistics.regression import Design, fit, C
from pystatistics.survival import coxph

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def california_ds(california_data):
    return DataSource.from_dataframe(california_data)


@pytest.fixture(scope="session")
def lung_ds():
    csv = FIXTURES_DIR / "lung_coxph.csv"
    if not csv.exists():
        pytest.skip("Run run_r_validation.R first to create lung_coxph.csv")
    return pd.read_csv(csv)


class TestOLSNumericByCategorical:
    """MedHouseVal ~ MedInc * region (numeric, factor, numeric:factor)."""

    TERMS = ["MedInc", C("region"), ("MedInc", C("region"))]

    def _fit(self, ds):
        return fit(Design.from_datasource(ds, y="MedHouseVal", terms=self.TERMS))

    def test_coefficients(self, california_ds, r_results):
        res = self._fit(california_ds)
        np.testing.assert_allclose(
            res.coefficients, r_results["ols_factor_interaction"]["coefficients"],
            rtol=1e-9, atol=1e-11,
        )

    def test_standard_errors(self, california_ds, r_results):
        res = self._fit(california_ds)
        np.testing.assert_allclose(
            res.standard_errors,
            r_results["ols_factor_interaction"]["standard_errors"],
            rtol=1e-9, atol=1e-11,
        )

    def test_labels_name_each_level(self, california_ds):
        res = self._fit(california_ds)
        assert list(res.coef) == [
            "(Intercept)", "MedInc", "region[North]", "region[South]",
            "MedInc:region[North]", "MedInc:region[South]",
        ]


class TestOLSCategoricalByCategorical:
    """MedHouseVal ~ region * pop_quartile (factor:factor, R column order)."""

    TERMS = [C("region"), C("pop_quartile"), (C("region"), C("pop_quartile"))]

    def _fit(self, ds):
        return fit(Design.from_datasource(ds, y="MedHouseVal", terms=self.TERMS))

    def test_coefficients(self, california_ds, r_results):
        res = self._fit(california_ds)
        np.testing.assert_allclose(
            res.coefficients, r_results["ols_cat_cat"]["coefficients"],
            rtol=1e-9, atol=1e-11,
        )

    def test_standard_errors(self, california_ds, r_results):
        res = self._fit(california_ds)
        np.testing.assert_allclose(
            res.standard_errors, r_results["ols_cat_cat"]["standard_errors"],
            rtol=1e-9, atol=1e-11,
        )

    def test_interaction_column_order_matches_r(self, california_ds):
        res = self._fit(california_ds)
        # First factor (region) varies fastest, matching R's model.matrix.
        assert list(res.coef)[-6:] == [
            "region[North]:pop_quartile[Q2]", "region[South]:pop_quartile[Q2]",
            "region[North]:pop_quartile[Q3]", "region[South]:pop_quartile[Q3]",
            "region[North]:pop_quartile[Q4]", "region[South]:pop_quartile[Q4]",
        ]


class TestOLSReferenceLevel:
    """Selectable baseline: region releveled to 'South'."""

    TERMS = ["MedInc", C("region", ref="South")]

    def _fit(self, ds):
        return fit(Design.from_datasource(ds, y="MedHouseVal", terms=self.TERMS))

    def test_coefficients(self, california_ds, r_results):
        res = self._fit(california_ds)
        np.testing.assert_allclose(
            res.coefficients, r_results["ols_relevel"]["coefficients"],
            rtol=1e-9, atol=1e-11,
        )

    def test_baseline_is_south(self, california_ds):
        res = self._fit(california_ds)
        # South is dropped; Central and North remain.
        assert list(res.coef) == [
            "(Intercept)", "MedInc", "region[Central]", "region[North]",
        ]


class TestGLMBinomialFactor:
    """high_value ~ MedInc + region (factor predictor in a GLM)."""

    TERMS = ["MedInc", C("region")]

    def _fit(self, ds):
        return fit(
            Design.from_datasource(ds, y="high_value", terms=self.TERMS),
            family="binomial",
        )

    def test_coefficients(self, california_ds, r_results):
        res = self._fit(california_ds)
        np.testing.assert_allclose(
            res.coefficients, r_results["glm_factor"]["coefficients"],
            rtol=1e-8, atol=1e-10,
        )

    def test_standard_errors(self, california_ds, r_results):
        res = self._fit(california_ds)
        np.testing.assert_allclose(
            res.standard_errors, r_results["glm_factor"]["standard_errors"],
            rtol=1e-7, atol=1e-9,
        )


class TestCoxFactor:
    """Surv(time, event) ~ age + factor(sex) + factor(ph.ecog), no intercept."""

    TERMS = ["age", C("sex"), C("ph.ecog")]

    def _fit(self, lung_ds):
        ds = DataSource.from_dataframe(lung_ds)
        return coxph(
            lung_ds["time"].to_numpy(float),
            lung_ds["event"].to_numpy(float),
            ds,
            terms=self.TERMS,
        )

    def test_coefficients(self, lung_ds, r_results):
        res = self._fit(lung_ds)
        np.testing.assert_allclose(
            res.coefficients, r_results["coxph_factor"]["coefficients"],
            rtol=1e-6, atol=1e-8,
        )

    def test_no_intercept_and_labels(self, lung_ds):
        res = self._fit(lung_ds)
        assert list(res.coef) == [
            "age", "sex[2.0]", "ph.ecog[1.0]", "ph.ecog[2.0]", "ph.ecog[3.0]",
        ]
