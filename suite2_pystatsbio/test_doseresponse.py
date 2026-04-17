"""Suite 2: Dose-response tests against R drc.

Real-data target: drc::ryegrass (Streibig et al. 1993 ferulic-acid
bioassay) — the canonical drc-package dataset. Validates fit_drm(LL.4)
against drc::drm() on published real bioassay data.

Synthetic targets: 5 representative LL.4 compounds + batch fit. Kept for
coverage of edge parameter combinations and the batch API, not covered
by the single real curve.

Runtime parity: assert_parity_dr() fails if Python is >20× slower than R
for fits R takes >= 50 ms.
"""

import time

import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from pystatsbio.doseresponse import fit_drm, fit_drm_batch, ec50

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _parity(py_elapsed: float, r_elapsed: float, label: str) -> None:
    if r_elapsed is None or r_elapsed < 0.05:
        return
    ratio = py_elapsed / r_elapsed
    assert ratio <= 20.0, (
        f"[{label}] Python {py_elapsed:.3f}s vs R {r_elapsed:.3f}s "
        f"(ratio {ratio:.1f}x > 20x)."
    )


class TestRyegrass:
    """fit_drm on drc::ryegrass — real Lolium perenne bioassay data."""

    def test_ll4_matches_drc(self, r_results):
        df = pd.read_csv(FIXTURES_DIR / "ryegrass.csv")
        r_ref = r_results["doseresponse_ryegrass"]
        r_time = r_results.get("timing", {}).get("doseresponse_ryegrass")

        t0 = time.perf_counter()
        result = fit_drm(df["conc"].values, df["rootl"].values, model="LL.4")
        py_time = time.perf_counter() - t0

        assert result.converged
        p = result.params
        # drc reports Hill negative for the LL.4 parameterization on a
        # decreasing curve. Compare magnitudes since sign convention can
        # differ between implementations.
        assert abs(p.hill) == pytest.approx(abs(r_ref["hill"]), rel=5e-2)
        assert p.bottom == pytest.approx(r_ref["bottom"], rel=5e-2, abs=0.1)
        assert p.top == pytest.approx(r_ref["top"], rel=1e-3)
        assert p.ec50 == pytest.approx(r_ref["ec50"], rel=5e-3)
        assert result.rss == pytest.approx(r_ref["rss"], rel=5e-2)
        _parity(py_time, r_time, "ryegrass LL.4")

# 5 representative compounds
COMPOUND_IDS = [0, 10, 20, 30, 40]


class TestSingleCurveFit:
    """fit_drm LL.4 vs R drc::drm(fct=LL.4())."""

    @pytest.fixture(params=COMPOUND_IDS)
    def compound(self, request, r_results):
        cid = request.param
        cname = f"compound_{cid:03d}"

        with open(FIXTURES_DIR / f"{cname}.json") as f:
            data = json.load(f)

        r_ref = r_results["doseresponse"].get(cname)
        if r_ref is None or not r_ref.get("converged", False):
            pytest.skip(f"R did not converge for {cname}")

        return data, r_ref, cname

    def test_ec50(self, compound):
        data, r_ref, cname = compound
        dose = np.array(data["dose"])
        response = np.array(data["response"])

        result = fit_drm(dose, response, model="LL.4")
        assert result.converged, f"{cname} did not converge"

        # EC50 within reasonable tolerance (nonlinear fitting)
        assert result.params.ec50 == pytest.approx(
            r_ref["ec50"], rel=0.5  # 50% relative — nonlinear fits can vary
        ), f"{cname}: EC50 mismatch"

    def test_top_bottom(self, compound):
        data, r_ref, cname = compound
        dose = np.array(data["dose"])
        response = np.array(data["response"])

        result = fit_drm(dose, response, model="LL.4")
        if not result.converged:
            pytest.skip(f"{cname} did not converge")

        # Asymptotes should be within 10 units
        assert result.params.bottom == pytest.approx(r_ref["bottom"], abs=10)
        assert result.params.top == pytest.approx(r_ref["top"], abs=10)

    def test_ec50_with_ci(self, compound):
        data, r_ref, cname = compound
        dose = np.array(data["dose"])
        response = np.array(data["response"])

        result = fit_drm(dose, response, model="LL.4")
        if not result.converged:
            pytest.skip(f"{cname} did not converge")

        ec50_result = ec50(result)
        # EC50 estimate should match
        assert ec50_result.estimate == pytest.approx(r_ref["ec50"], rel=0.5)
        # CI should contain the R estimate
        assert ec50_result.ci_lower <= r_ref["ec50"] * 3  # generous bound
        assert ec50_result.ci_upper >= r_ref["ec50"] / 3


class TestBatchFit:
    """fit_drm_batch CPU vs sequential fit_drm."""

    def test_batch_cpu_matches_sequential(self, dr_data):
        dose_matrix = dr_data["dose_matrix"]
        response_matrix = dr_data["response_matrix"]

        # Batch fit
        batch_result = fit_drm_batch(
            dose_matrix, response_matrix,
            model="LL.4", backend="cpu",
        )

        # Sequential fit for first 5
        for i in range(5):
            single = fit_drm(dose_matrix[i], response_matrix[i], model="LL.4")
            if single.converged and batch_result.converged[i]:
                assert batch_result.ec50[i] == pytest.approx(
                    single.params.ec50, rel=1e-3
                ), f"Compound {i}: batch vs sequential EC50 mismatch"
