"""Validation: pystatsbio.gee.gee vs R geepack::geeglm on real data.

WHAT:     gee(y, X, cluster_id, family='gaussian', corr_structure='exchangeable')
HOW:      R fits geepack::geeglm(Weight ~ Time + Cu, id=Pig,
          corstr='exchangeable') on dietox and writes the numeric frame
          (Pig, Weight, Time, Cu) to gee_long.csv. Python reads the same
          CSV. Compares coefficients and robust (sandwich) SEs.
DATASET:  geepack::dietox (fixtures/newmodules/gee_long.csv) — 72 pigs
          weighed weekly for ~12 weeks while receiving different dietary
          copper levels. 861 observations total.
WHY:      dietox is the canonical geepack example, used in the geepack
          vignette and in the GEE chapter of Fitzmaurice/Laird/Ware.
          Real longitudinal data where the within-pig correlation is
          non-trivial — exactly where GEE's robust SE is needed.
"""

from __future__ import annotations

import time

import numpy as np

from pystatsbio.gee import gee

from _newmodules_s2 import (
    assert_runtime_parity,
    load_csv,
    load_r_results,
    to_array,
)


class TestGEE:
    def test_matches_r_and_runtime(self):
        gd = load_csv("gee_long.csv")
        gd = gd.sort_values(["Pig", "Time"]).reset_index(drop=True)
        y = gd["Weight"].values.astype(float)
        X = np.column_stack([
            np.ones(len(gd)),
            gd["Time"].values.astype(float),
            gd["Cu"].values.astype(float),
        ])
        cluster = gd["Pig"].values.astype(int)

        r_all = load_r_results()
        r_ref = r_all["results"]["gee"]
        r_time = r_all["timing"]["gee"]

        t0 = time.perf_counter()
        result = gee(
            y, X, cluster,
            family="gaussian", corr_structure="exchangeable",
        )
        py_time = time.perf_counter() - t0

        np.testing.assert_allclose(
            result.coefficients, to_array(r_ref["coefficients"]),
            rtol=1e-3, atol=1e-3,
            err_msg="GEE coefs vs geepack::geeglm on dietox",
        )
        np.testing.assert_allclose(
            result.robust_se, to_array(r_ref["robust_se"]),
            rtol=5e-2, atol=1e-3,
            err_msg="GEE robust SEs vs geepack on dietox",
        )
        assert result.n_clusters == int(r_ref["n_clusters"])
        assert_runtime_parity(py_time, r_time, "gee")
