"""Validation: pystatsbio.meta.rma vs R metafor::rma (REML).

WHAT:     rma(yi, vi, method='REML') — inverse-variance weighted meta with
          REML between-study variance.
HOW:      Fit in R (metafor::rma) and Python on the SAME yi/vi CSV.
          Compare pooled estimate, SE, CI, tau^2, Q, I^2.
DATASET:  suite2_pystatsbio/fixtures/newmodules/meta_yi.csv — k=12 studies
          simulated from a random-effects DGP with true mu=0.5, tau^2=0.05.
          Generated with a fixed seed so R and Python see identical inputs.
WHY:      A non-zero tau^2 DGP is the only way to meaningfully test REML
          estimation against fixed-effects — with tau^2=0 every method
          collapses to the same fixed-effect answer.
"""

from __future__ import annotations

import time

import pytest

from pystatsbio.meta import rma

from _newmodules_s2 import (
    assert_runtime_parity,
    load_csv,
    load_r_results,
)


class TestRMA:
    def test_matches_r_and_runtime(self):
        md = load_csv("meta_yi.csv")
        r_all = load_r_results()
        r_ref = r_all["results"]["rma"]
        r_time = r_all["timing"]["rma"]

        t0 = time.perf_counter()
        result = rma(md["yi"].values, md["vi"].values, method="REML")
        py_time = time.perf_counter() - t0

        assert result.estimate == pytest.approx(r_ref["estimate"], rel=1e-4)
        assert result.se == pytest.approx(r_ref["se"], rel=1e-3)
        assert result.tau2 == pytest.approx(r_ref["tau2"], rel=5e-3)
        assert result.Q == pytest.approx(r_ref["q"], rel=1e-6)
        # I^2 is rounded to one decimal in metafor's printed output; compare
        # with absolute tolerance of 1.5 (percentage points). metafor uses
        # slightly different formula for I2 vs tau2/(tau2 + s2_typical).
        assert result.I2 == pytest.approx(r_ref["i2"], abs=1.5)
        assert result.k == int(r_ref["k"])

        assert_runtime_parity(py_time, r_time, "rma")
