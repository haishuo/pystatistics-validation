"""R10 adversarial hard-case grid + R15 default-invocation validation for mvnmle.

Two RIGOR rules the pre-red-team v3.18.0 report never exercised:

  * R15 — validate the naive DEFAULT call ``mlest(X)`` (backend=cpu, method=direct,
    force=False), not just the tuned/expert path. The footgun a user hits day one
    (a near-degenerate covariance) must FAIL LOUD, never silently return a
    separated garbage fit.
  * R10 — the correctness grid must reach the adversarial regime AND match R's own
    failures/warnings: near-collinear, exactly-collinear, constant-column,
    heavy-missingness designs. "Matches R on easy data" is not the claim; "matches
    R's numbers where it succeeds and R's refusals where it fails" is.

For each problem we run the DEFAULT ``mlest(X)`` and R's ``mvnmle::mlest`` on the
SAME NaN-coded matrix, and record each side's outcome (converged+loglik, or the
exception/error class). The verdict is the agreement of outcomes, not just of
numbers on the cases that succeed.

Usage:
    python hard_cases.py [tag]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path[:0] = [str(_HERE.parent / "_shared")]

from run_r_mvnmle import run_r_mvnmle  # noqa: E402

import pystatistics as _ps  # noqa: E402
_OUTDIR = _REPO / "artifacts" / "mvnmle" / f"v{_ps.__version__}" / "runs"
_SEED = 20260707


# --------------------------------------------------------------------------- #
# Adversarial problem constructors (deterministic).
# --------------------------------------------------------------------------- #
def _mcar(X: np.ndarray, rate: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    M = rng.random(X.shape) < rate
    M[M.all(axis=1), 0] = False  # no fully-missing row
    X = X.astype(np.float64).copy()
    X[M] = np.nan
    return X


def _base_mvn(n: int, p: int, seed: int, rho: float = 0.4) -> np.ndarray:
    """Well-conditioned AR(1)-correlated MVN draw (no missingness yet)."""
    idx = np.arange(p)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p)) @ np.linalg.cholesky(Sigma).T


def problems() -> list[tuple[str, str, np.ndarray]]:
    """Return (name, expectation, X) triples. ``expectation`` is prose for the
    report, not asserted mechanically — the R-vs-py outcome match is the test."""
    out = []
    n = 4000

    # R15 — the naive default on clean data: must converge and match R.
    X = _base_mvn(n, 6, _SEED)
    out.append(("default_clean", "both converge; loglik agrees",
                _mcar(X, 0.15, _SEED + 1)))

    # Near-collinear: two columns correlated ~0.99999 (below exact). Ill-cond
    # but full-rank; sits at the documented refusal boundary.
    X = _base_mvn(n, 5, _SEED + 2)
    X = np.column_stack([X, X[:, 0] + 1e-4 * np.random.default_rng(3).standard_normal(n)])
    out.append(("near_collinear_0.99999", "at/over refusal boundary; expect refuse",
                _mcar(X, 0.10, _SEED + 3)))

    # Exactly collinear: a duplicated column (corr = 1) → rank-deficient, no
    # interior MLE. Must fail loud (SingularMatrixError); R's Cholesky must also fail.
    X = _base_mvn(n, 5, _SEED + 4)
    X = np.column_stack([X, X[:, 2]])  # exact duplicate
    out.append(("exact_collinear", "rank-deficient; both must refuse/error",
                _mcar(X, 0.10, _SEED + 5)))

    # Constant (zero-variance) column: degenerate marginal.
    X = _base_mvn(n, 5, _SEED + 6)
    X = np.column_stack([X, np.full(n, 2.5)])
    out.append(("constant_column", "zero-variance col; degenerate",
                _mcar(X, 0.10, _SEED + 7)))

    # Heavy missingness: 60% MCAR — many patterns, thin per-pattern support.
    X = _base_mvn(n, 8, _SEED + 8)
    out.append(("heavy_missing_60pct", "both converge; loglik agrees",
                _mcar(X, 0.60, _SEED + 9)))

    return out


# --------------------------------------------------------------------------- #
# Run one problem through the DEFAULT call + R.
# --------------------------------------------------------------------------- #
def run_default(X: np.ndarray) -> dict:
    """The naive default invocation: ``mlest(X)`` with everything default."""
    from pystatistics.mvnmle import mlest
    try:
        sol = mlest(X)  # R15: no backend, no force, no tuning
        return {"py_outcome": "converged" if sol.converged else "not_converged",
                "py_loglik": float(sol.loglik),
                "py_backend": sol.backend_name,
                "py_warnings": "; ".join(sol.warnings) if sol.warnings else ""}
    except Exception as exc:  # noqa: BLE001 — fail-loud is the correct R15 outcome
        return {"py_outcome": f"refused:{type(exc).__name__}",
                "py_error": str(exc)[:200], "py_loglik": None}


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else "hard_cases"
    records = []
    for name, expectation, X in problems():
        n, p = X.shape
        rec = {"case": name, "expectation": expectation,
               "n": int(n), "p": int(p),
               "missing_frac": float(np.isnan(X).mean())}
        rec.update(run_default(X))
        # R reference on the identical matrix.
        r = run_r_mvnmle(X, survey=name, n_patterns=0,
                         missing_frac=rec["missing_frac"], reps=1, warmup=0,
                         timeout_s=600.0)
        rec["r_loglik"] = r.get("loglik")
        rec["r_error"] = r.get("error")
        rec["r_outcome"] = "error" if r.get("error") else "ok"
        if rec.get("py_loglik") is not None and rec["r_loglik"] is not None:
            rec["loglik_abs_diff"] = abs(rec["py_loglik"] - rec["r_loglik"])
        records.append(rec)
        print(f"{name:26s} py={rec['py_outcome']:24s} r={rec['r_outcome']:6s} "
              f"py_ll={rec.get('py_loglik')} r_ll={rec.get('r_loglik')} "
              f"d={rec.get('loglik_abs_diff')}")

    import pystatistics
    payload = {"study": "hard_cases", "kind": "r10-r15-adversarial-default",
               "pystatistics_version": pystatistics.__version__,
               "records": records}
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    out = _OUTDIR / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}  ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
