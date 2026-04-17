"""Validation: multinomial logit (multinom) vs R on a real dataset.

WHAT:     pystatistics.multinomial.multinom(y, X) — softmax regression with
          the LAST class as reference.
HOW:      R fits nnet::multinom(type ~ RI + Na + Al, data=fgl) — which uses
          the FIRST class as reference — and writes the numeric design to
          CSV. Python reads the same CSV. The two reference-class
          conventions differ; we reparameterize the Python output into
          R's parameterization before comparing (see body).
DATASET:  MASS::fgl (fixtures/newmodules/fgl.csv) — 214 glass fragments
          from forensic science, 6 glass types (WinF, WinNF, Veh, Con,
          Tabl, Head), 9 chemical composition predictors. We use 3
          predictors (RI, Na, Al) to keep the fit small and stable.
WHY:      fgl is the standard multinom example in MASS/VR7. Real forensic
          data with a genuine unordered multiclass outcome.
"""

from __future__ import annotations

import time

import numpy as np

from pystatistics.multinomial import multinom

from _newmodules_s1 import (
    assert_runtime_parity,
    load_dataset,
    load_r_results,
    to_array,
)


class TestMultinom:
    def test_matches_r_and_runtime(self):
        df = load_dataset("fgl.csv")
        r_all = load_r_results()
        r_ref = r_all["results"]["multinom"]
        r_time = r_all["timing"]["multinom"]

        y = df["type"].values.astype(int)    # 0..5 after R's subtraction
        predictors = [c for c in df.columns if c != "type"]
        X = np.column_stack(
            [np.ones(len(df))] +
            [df[c].values.astype(float) for c in predictors]
        )

        t0 = time.perf_counter()
        # fgl's 6-class fit needs more iterations than the default 200.
        result = multinom(y, X, max_iter=1000)
        py_time = time.perf_counter() - t0

        # pystatistics: (J-1) × p coefs for classes 0..J-2 vs ref = J-1.
        # R (nnet):     (J-1) × p coefs for classes 1..J-1 vs ref = 0.
        coefs_py = np.asarray(result.coefficient_matrix)   # shape (5, 4)
        ref_row = coefs_py[0]                               # class 0 vs J-1
        py_as_R = np.vstack(
            [coefs_py[k] - ref_row for k in range(1, coefs_py.shape[0])]
            + [-ref_row]
        )

        r_coefs = np.asarray(r_ref["coefficients"])        # shape (5, 4)
        np.testing.assert_allclose(
            py_as_R, r_coefs, rtol=5e-2, atol=5e-2,
            err_msg="multinom coefs vs nnet::multinom on MASS::fgl "
                    "(reparameterized to R's first-class-as-reference)",
        )
        assert_runtime_parity(py_time, r_time, "multinom")
