"""Validation: pystatsbio.pk.nca vs R PKNCA::pk.nca on Theoph.

WHAT:     pystatsbio.pk.nca(time, conc, dose, route='ev') — non-
          compartmental PK analysis.
HOW:      R fits PKNCA::pk.nca on each of the 12 Theoph subjects and
          writes Theoph.csv. Python reads the same rows and computes NCA
          parameters per subject.
DATASET:  datasets::Theoph (fixtures/Theoph.csv) — 12 subjects, oral
          theophylline concentrations sampled over 24 hours after a
          single dose. Boeckmann, Sheiner & Beal (1994).
WHY:      Theoph is THE canonical R PK dataset — used by the PKNCA
          vignette and the nlme package. Real clinical PK data beats
          the synthetic one-compartment simulation we use elsewhere.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pystatsbio.pk import nca

FIX_DIR = Path(__file__).parent / "fixtures"


def _parity(py_elapsed: float, r_elapsed: float, label: str) -> None:
    if r_elapsed is None or r_elapsed < 0.05:
        return
    ratio = py_elapsed / r_elapsed
    assert ratio <= 20.0, (
        f"[{label}] Python {py_elapsed:.3f}s vs R {r_elapsed:.3f}s "
        f"(ratio {ratio:.1f}x > 20x)."
    )


def _load_theoph() -> pd.DataFrame:
    path = FIX_DIR / "Theoph.csv"
    if not path.exists():
        pytest.skip(f"{path} missing. Run run_r_validation.R first.")
    return pd.read_csv(path)


class TestTheophNCA:
    """Per-subject NCA on real theophylline PK data."""

    def test_all_subjects_match_pknca(self, r_results):
        df = _load_theoph()
        ref = r_results.get("theoph")
        if ref is None:
            pytest.skip("r_results has no 'theoph' block; rerun R script.")

        r_time = r_results.get("timing", {}).get("theoph")

        subjects = ref["subjects"]
        # Compute Python NCA for all 12 subjects; aggregate elapsed time
        # to compare to R's total NCA time (R does them in one pk.nca call).
        py_start = time.perf_counter()
        py_params: dict[str, dict] = {}
        for sid in subjects:
            sub = df[df["Subject"].astype(str) == sid]
            t = sub["Time"].values.astype(float)
            c = sub["conc"].values.astype(float)
            dose = float(sub["Dose"].iloc[0])
            r = nca(t, c, dose=dose, route="ev")
            py_params[sid] = {
                "auc_last":  r.auc_last,
                "cmax":      r.cmax,
                "tmax":      r.tmax,
                "half_life": r.half_life,
            }
        py_elapsed = time.perf_counter() - py_start

        # Compare each subject.
        for sid in subjects:
            r_sub = ref["per_subject"][sid]
            p = py_params[sid]
            # AUC_last — trapezoid result, should match tightly.
            assert p["auc_last"] == pytest.approx(r_sub["auc_last"], rel=5e-3), (
                f"Subject {sid}: AUC_last py={p['auc_last']:.3f} "
                f"vs R={r_sub['auc_last']:.3f}"
            )
            # Cmax / Tmax — exact matches (they are observed values).
            assert p["cmax"] == pytest.approx(r_sub["cmax"], rel=1e-10)
            assert p["tmax"] == pytest.approx(r_sub["tmax"], rel=1e-10)
            # Half-life: depends on which points PKNCA/pystatsbio pick for
            # lambda_z regression. Allow 10% relative difference.
            if r_sub["half_life"] is not None and p["half_life"] is not None:
                assert p["half_life"] == pytest.approx(
                    r_sub["half_life"], rel=0.10,
                ), f"Subject {sid}: t1/2 mismatch"

        _parity(py_elapsed, r_time, "theoph_nca")
