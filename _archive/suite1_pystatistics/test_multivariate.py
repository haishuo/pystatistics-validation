"""Validation: PCA and ML factor analysis vs R on real datasets.

PCA
  WHAT:     pystatistics.multivariate.pca(X, center=True, scale=True).
  HOW:      R runs prcomp(USArrests, center=TRUE, scale.=TRUE) and saves
            USArrests.csv. Python reads the same CSV.
  DATASET:  datasets::USArrests — 50 US states × 4 crime rates per 100,000
            (1973). The dataset the prcomp() help page uses.
  WHY:      Canonical real PCA dataset with an obvious correlation
            structure (violent crimes cluster).

Factor analysis
  WHAT:     pystatistics.multivariate.factor_analysis(X, n_factors=2).
  HOW:      R runs factanal(mtcars, factors=2, rotation='varimax') and
            saves mtcars.csv. Python reads the same CSV.
  DATASET:  datasets::mtcars — 32 cars × 11 numeric variables. USArrests
            has only 4 vars, not enough for a 2-factor fit.
  WHY:      mtcars is canonical. Two factors typically capture engine
            size/power vs. fuel economy/transmission.
"""

from __future__ import annotations

import time

import numpy as np

from pystatistics.multivariate import pca, factor_analysis

from _newmodules_s1 import (
    assert_runtime_parity,
    load_dataset,
    load_r_results,
    to_array,
)


class TestPCA:
    def test_matches_r_and_runtime(self):
        df = load_dataset("USArrests.csv")
        X = df.select_dtypes(include=[np.number]).values.astype(float)
        r_all = load_r_results()
        r_ref = r_all["results"]["pca"]
        r_time = r_all["timing"]["pca"]

        t0 = time.perf_counter()
        result = pca(X, center=True, scale=True)
        py_time = time.perf_counter() - t0

        np.testing.assert_allclose(
            result.sdev, to_array(r_ref["sdev"]),
            rtol=1e-10, atol=1e-12, err_msg="PCA sdev vs R on USArrests",
        )
        py_rot = np.asarray(result.rotation)
        r_rot = np.asarray(r_ref["rotation"])
        np.testing.assert_allclose(
            np.abs(py_rot), np.abs(r_rot),
            rtol=1e-8, atol=1e-10,
            err_msg="PCA |rotation| vs R on USArrests",
        )
        assert_runtime_parity(py_time, r_time, "pca")


class TestFactorAnalysis:
    def test_matches_r_and_runtime(self):
        df = load_dataset("mtcars.csv")
        X = df.select_dtypes(include=[np.number]).values.astype(float)
        r_all = load_r_results()
        r_ref = r_all["results"]["factor_analysis"]
        r_time = r_all["timing"]["factor_analysis"]

        t0 = time.perf_counter()
        result = factor_analysis(X, n_factors=2, rotation="varimax")
        py_time = time.perf_counter() - t0

        r_uniq = to_array(r_ref["uniquenesses"])
        np.testing.assert_allclose(
            np.sort(result.uniquenesses), np.sort(r_uniq),
            rtol=5e-2, atol=5e-2, err_msg="Uniquenesses vs R on mtcars",
        )
        py_h2 = np.sum(np.asarray(result.loadings) ** 2, axis=1)
        r_h2 = np.sum(np.asarray(r_ref["loadings"]) ** 2, axis=1)
        np.testing.assert_allclose(
            np.sort(py_h2), np.sort(r_h2),
            rtol=5e-2, atol=5e-2, err_msg="Communalities vs R on mtcars",
        )
        assert_runtime_parity(py_time, r_time, "factor_analysis")
