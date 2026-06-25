"""Validation: Gamma GLM and Negative Binomial GLM vs R on real datasets.

Gamma GLM
  WHAT:     pystatistics.regression.fit with GammaFamily(link='log').
  HOW:      R fits glm(Ozone ~ Solar.R + Temp + Wind, family=Gamma(link='log'))
            on datasets::airquality (complete cases) and writes the prepared
            CSV. Python reads the same CSV and fits the same model.
  DATASET:  datasets::airquality (fixtures/newmodules/airquality.csv) —
            daily NYC air quality, May-September 1973, 111 complete cases.
  WHY:      Ozone is positive, continuous, right-skewed — the textbook Gamma
            regression target. Real data beats a simulated stand-in.

Negative Binomial GLM
  WHAT:     pystatistics.regression.fit with family='negative.binomial'.
  HOW:      R fits MASS::glm.nb(Days ~ Eth + Sex + Age + Lrn) on MASS::quine
            and writes the model-matrix CSV so Python sees the exact same
            numeric design.
  DATASET:  MASS::quine (fixtures/newmodules/quine.csv) — 146 Australian
            schoolchildren, days absent by ethnicity/sex/age/learner status.
  WHY:      quine is THE canonical negative binomial dataset (McCullagh &
            Nelder, Generalized Linear Models). Real overdispersed counts.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from pystatistics.regression import Design, GammaFamily, fit

from _newmodules_s1 import (
    assert_runtime_parity,
    load_dataset,
    load_r_results,
    to_array,
)


class TestGammaGLM:
    def test_matches_r_and_runtime(self):
        df = load_dataset("airquality.csv")
        r_all = load_r_results()
        r_ref = r_all["results"]["gamma_glm"]
        r_time = r_all["timing"]["gamma_glm"]

        X = np.column_stack([
            np.ones(len(df)),
            df["Solar.R"].values.astype(float),
            df["Temp"].values.astype(float),
            df["Wind"].values.astype(float),
        ])
        y = df["Ozone"].values.astype(float)
        design = Design.from_arrays(X, y)

        t0 = time.perf_counter()
        result = fit(design, family=GammaFamily(link="log"), backend="cpu")
        py_time = time.perf_counter() - t0

        np.testing.assert_allclose(
            result.coefficients, to_array(r_ref["coefficients"]),
            rtol=1e-6, atol=1e-8,
            err_msg="Gamma GLM coefficients vs R on airquality",
        )
        # SEs: pystatistics uses a slightly different dispersion estimator
        # from R (MLE vs Pearson/df).
        np.testing.assert_allclose(
            result.standard_errors, to_array(r_ref["standard_errors"]),
            rtol=5e-2, atol=1e-3,
            err_msg="Gamma GLM SEs vs R on airquality",
        )
        assert result.deviance == pytest.approx(r_ref["deviance"], rel=1e-3)
        assert_runtime_parity(py_time, r_time, "GammaGLM")


class TestNegativeBinomialGLM:
    def test_matches_r_and_runtime(self):
        df = load_dataset("quine.csv")
        r_all = load_r_results()
        r_ref = r_all["results"]["negbin_glm"]
        r_time = r_all["timing"]["negbin_glm"]

        # R wrote the model-matrix columns (minus the all-ones intercept)
        # alongside Days. Reconstruct X with intercept in front.
        predictors = [c for c in df.columns if c != "Days"]
        X = np.column_stack(
            [np.ones(len(df))] +
            [df[c].values.astype(float) for c in predictors]
        )
        y = df["Days"].values.astype(float)
        design = Design.from_arrays(X, y)

        t0 = time.perf_counter()
        result = fit(design, family="negative.binomial", backend="cpu")
        py_time = time.perf_counter() - t0

        np.testing.assert_allclose(
            result.coefficients, to_array(r_ref["coefficients"]),
            rtol=5e-2, atol=5e-3,
            err_msg="NegBin coefficients vs R on quine (theta profile-estimated)",
        )
        # NOTE: pystatistics does not expose the auto-estimated theta on the
        # solution object. Coefficient agreement is the practical validation:
        # the NB mean structure is identified only up to theta, so matching
        # coefficients on real overdispersed data is strong evidence.
        assert_runtime_parity(py_time, r_time, "NegBinGLM")
