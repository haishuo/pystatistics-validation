"""Suite 2: Pharmacokinetics NCA tests against R PKNCA."""

import pytest
import numpy as np
import json
from pathlib import Path

from pystatsbio.pk import nca

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_pk_subjects():
    """Load PK subject data."""
    path = FIXTURES_DIR / "pk_subjects.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


class TestNCAOral:
    """NCA for oral (extravascular) subjects vs R PKNCA."""

    @pytest.fixture(params=range(5))  # 5 oral subjects
    def oral_subject(self, request, r_results):
        subjects = _load_pk_subjects()
        if not subjects:
            pytest.skip("No PK data")

        idx = request.param
        if idx >= len(subjects):
            pytest.skip(f"Subject {idx} not available")

        subj = subjects[idx]
        sname = f"subject_{idx + 1}"
        r_ref = r_results.get("pk", {}).get(sname)
        if r_ref is None or "error" in r_ref:
            pytest.skip(f"R failed for {sname}")

        return subj, r_ref, sname

    def test_auc_last(self, oral_subject):
        subj, r_ref, sname = oral_subject
        if subj["route"] != "ev":
            pytest.skip("Not oral")

        result = nca(
            np.array(subj["time"]),
            np.array(subj["concentration"]),
            dose=subj["dose"],
            route="ev",
        )

        if r_ref["auc_last"] is not None:
            assert result.auc_last == pytest.approx(
                r_ref["auc_last"], rel=1e-6
            ), f"{sname}: AUC_last mismatch"

    def test_cmax(self, oral_subject):
        subj, r_ref, sname = oral_subject
        if subj["route"] != "ev":
            pytest.skip("Not oral")

        result = nca(
            np.array(subj["time"]),
            np.array(subj["concentration"]),
            dose=subj["dose"],
            route="ev",
        )

        if r_ref["cmax"] is not None:
            assert result.cmax == pytest.approx(r_ref["cmax"], rel=1e-10)

    def test_tmax(self, oral_subject):
        subj, r_ref, sname = oral_subject
        if subj["route"] != "ev":
            pytest.skip("Not oral")

        result = nca(
            np.array(subj["time"]),
            np.array(subj["concentration"]),
            dose=subj["dose"],
            route="ev",
        )

        if r_ref["tmax"] is not None:
            assert result.tmax == pytest.approx(r_ref["tmax"], abs=1e-10)

    def test_half_life(self, oral_subject):
        subj, r_ref, sname = oral_subject
        if subj["route"] != "ev":
            pytest.skip("Not oral")

        result = nca(
            np.array(subj["time"]),
            np.array(subj["concentration"]),
            dose=subj["dose"],
            route="ev",
        )

        if r_ref["half_life"] is not None and result.half_life is not None:
            # Terminal phase estimation can differ due to point selection algorithm
            assert result.half_life == pytest.approx(
                r_ref["half_life"], rel=1e-2
            ), f"{sname}: half-life mismatch"


class TestNCAIV:
    """NCA for IV bolus subject vs R PKNCA."""

    def test_iv_auc(self, r_results):
        subjects = _load_pk_subjects()
        # Last subject is IV
        iv_subj = [s for s in subjects if s["route"] == "iv"]
        if not iv_subj:
            pytest.skip("No IV subject")

        subj = iv_subj[0]
        sname = f"subject_{len(subjects)}"
        r_ref = r_results.get("pk", {}).get(sname)
        if r_ref is None or "error" in r_ref:
            pytest.skip(f"R failed for {sname}")

        result = nca(
            np.array(subj["time"]),
            np.array(subj["concentration"]),
            dose=subj["dose"],
            route="iv",
        )

        if r_ref["auc_last"] is not None:
            assert result.auc_last == pytest.approx(r_ref["auc_last"], rel=1e-6)

        if r_ref["auc_inf"] is not None and result.auc_inf is not None:
            assert result.auc_inf == pytest.approx(r_ref["auc_inf"], rel=1e-6)
