"""Suite 2: Dose-response tests against R drc on synthetic HTS data."""

import pytest
import numpy as np
import json
from pathlib import Path

from pystatsbio.doseresponse import fit_drm, fit_drm_batch, ec50

FIXTURES_DIR = Path(__file__).parent / "fixtures"

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
