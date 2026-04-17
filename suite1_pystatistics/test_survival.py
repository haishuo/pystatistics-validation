"""Suite 1: Survival analysis tests against R on survival::lung.

All three survival tests (KM, log-rank, Cox PH) now fit on the NCCTG
advanced lung cancer dataset (Loprinzi et al. 1994) — the canonical real-
world survival dataset. KM fits the overall curve; log-rank compares
survival by sex; Cox PH uses age + sex + ph.ecog as covariates.

California Housing (HouseAge as "time", high_value as "event") is NOT a
survival process — the original version used it as a proxy and Cox PH
failed to converge. The lung dataset is the R survival package's own
canonical teaching example.

Runtime parity: every test records Python wall-clock and asserts Python
is no more than 20× slower than R for fits R completes in >= 50 ms.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pystatistics.survival import kaplan_meier, survdiff, coxph

FIX_DIR = Path(__file__).parent / "fixtures"


def _runtime_parity(py_elapsed: float, r_elapsed: float, label: str) -> None:
    """Fail if Python is >20× slower than R (skip if R < 50 ms)."""
    if r_elapsed < 0.05:
        return
    ratio = py_elapsed / r_elapsed
    assert ratio <= 20.0, (
        f"[{label}] Python {py_elapsed:.3f}s vs R {r_elapsed:.3f}s "
        f"(ratio {ratio:.1f}x > 20x limit)."
    )


def _load_lung_km():
    """Load the lung KM dataset written by R (time, event, sex)."""
    path = FIX_DIR / "lung_km.csv"
    if not path.exists():
        pytest.skip(f"{path} missing. Run run_r_validation.R first.")
    df = pd.read_csv(path)
    return (df["time"].values.astype(float),
            df["event"].values.astype(float),
            df["sex"].values.astype(int))


def _load_lung_cox():
    """Load the lung Cox PH dataset written by R (time, event, X)."""
    path = FIX_DIR / "lung_coxph.csv"
    if not path.exists():
        pytest.skip(f"{path} missing. Run run_r_validation.R first.")
    df = pd.read_csv(path)
    time_ = df["time"].values.astype(float)
    event = df["event"].values.astype(float)
    X = df[["age", "sex", "ph.ecog"]].values.astype(float)
    return time_, event, X


class TestKaplanMeier:
    """KM overall curve on survival::lung."""

    def test_survival_probabilities(self, r_results):
        time_, event, _ = _load_lung_km()
        t0 = time.perf_counter()
        result = kaplan_meier(time_, event)
        py_time = time.perf_counter() - t0

        r_ref = r_results["kaplan_meier"]
        np.testing.assert_allclose(result.time, r_ref["time"], rtol=1e-10)
        np.testing.assert_allclose(result.survival, r_ref["survival"], rtol=1e-10)
        _runtime_parity(py_time, r_results["timing"]["kaplan_meier"], "kaplan_meier")

    def test_n_at_risk(self, r_results):
        time_, event, _ = _load_lung_km()
        result = kaplan_meier(time_, event)
        np.testing.assert_allclose(
            result.n_risk, r_results["kaplan_meier"]["n_risk"], rtol=1e-10,
        )

    def test_standard_errors(self, r_results):
        time_, event, _ = _load_lung_km()
        result = kaplan_meier(time_, event)
        np.testing.assert_allclose(
            result.se, r_results["kaplan_meier"]["std_err"], rtol=1e-10,
        )


class TestLogRank:
    """Log-rank survival by sex on survival::lung."""

    def test_statistic_and_p(self, r_results):
        time_, event, sex = _load_lung_km()
        t0 = time.perf_counter()
        result = survdiff(time_, event, sex)
        py_time = time.perf_counter() - t0

        r_ref = r_results["logrank"]
        assert result.statistic == pytest.approx(r_ref["statistic"], rel=1e-8)
        assert result.p_value == pytest.approx(r_ref["p_value"], rel=1e-6)
        _runtime_parity(py_time, r_results["timing"]["logrank"], "logrank")


class TestCoxPH:
    """Cox PH on survival::lung (age + sex + ph.ecog)."""

    def test_coefficients_vs_r(self, r_results):
        time_, event, X = _load_lung_cox()
        t0 = time.perf_counter()
        result = coxph(time_, event, X)
        py_time = time.perf_counter() - t0

        r_ref = r_results["coxph"]
        assert result.converged
        np.testing.assert_allclose(
            result.coefficients, r_ref["coefficients"], rtol=1e-5,
            err_msg="Cox PH coefficients vs R on survival::lung",
        )
        _runtime_parity(py_time, r_results["timing"]["coxph"], "coxph")

    def test_hazard_ratios_vs_r(self, r_results):
        time_, event, X = _load_lung_cox()
        result = coxph(time_, event, X)
        np.testing.assert_allclose(
            result.hazard_ratios, r_results["coxph"]["hazard_ratios"], rtol=1e-5,
        )

    def test_concordance_vs_r(self, r_results):
        time_, event, X = _load_lung_cox()
        result = coxph(time_, event, X)
        # pystatistics uses a simpler tie-handling for concordance than R's
        # Efron; agreement to 1e-3 is strong.
        assert result.concordance == pytest.approx(
            r_results["coxph"]["concordance"], rel=1e-3,
        )
