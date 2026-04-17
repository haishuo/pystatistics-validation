"""Validation: proportional odds (polr) vs R on a real dataset.

WHAT:     pystatistics.ordinal.polr(y, X, method='logistic').
HOW:      R fits MASS::polr(Sat ~ Infl + Type + Cont, data=housing),
          expanded by Freq so each respondent is one row, and writes the
          numeric design CSV. Python reads the same CSV and fits the same
          model.
DATASET:  MASS::housing (fixtures/newmodules/housing.csv) — Copenhagen
          housing satisfaction survey (Madsen 1976). Sat ordered Low/
          Medium/High; predictors Infl/Type/Cont. 1,681 expanded rows.
WHY:      housing is the dataset the MASS::polr help page itself uses.
          Real, published survey data with a genuine ordered outcome.
"""

from __future__ import annotations

import time

import numpy as np

from pystatistics.ordinal import polr

from _newmodules_s1 import (
    assert_runtime_parity,
    load_dataset,
    load_r_results,
    to_array,
)


class TestPolr:
    def test_matches_r_and_runtime(self):
        df = load_dataset("housing.csv")
        r_all = load_r_results()
        r_ref = r_all["results"]["polr"]
        r_time = r_all["timing"]["polr"]

        # R wrote Sat as integer 1..3; polr expects 0..K-1.
        y = (df["Sat"].values - 1).astype(int)
        X = df.drop(columns=["Sat"]).values.astype(float)

        t0 = time.perf_counter()
        result = polr(y, X, method="logistic")
        py_time = time.perf_counter() - t0

        np.testing.assert_allclose(
            result.coefficients, to_array(r_ref["coefficients"]),
            rtol=1e-4, atol=1e-5,
            err_msg="polr slopes vs R on MASS::housing",
        )
        np.testing.assert_allclose(
            result.threshold_values, to_array(r_ref["thresholds"]),
            rtol=1e-4, atol=1e-5,
            err_msg="polr thresholds vs R on MASS::housing",
        )
        np.testing.assert_allclose(
            result.standard_errors, to_array(r_ref["standard_errors"]),
            rtol=1e-3, atol=1e-5,
            err_msg="polr slope SEs vs R on MASS::housing",
        )
        assert_runtime_parity(py_time, r_time, "polr")
