"""Validate the discrete-time convergence diagnostics (.converged / .n_iter), 4.3.3.

pystatistics 4.3.3 closes finding C5: DiscreteTimeSolution now exposes .converged
(bool) and .n_iter (int), matching CoxSolution / GLMSolution (Amendment A3). This
is net-new validatable surface — a convergence FLAG is only useful if it tells the
truth — so it is validated here against R glm(binomial)'s OWN convergence behaviour
on the identical person-period design (RIGOR R9/G3: a convergence claim is a
hypothesis to test against the reference, not to assume).

For each case the driver fits the public discrete_time (CPU, fp64), reads the new
.converged / .n_iter, reconstructs the exact person-period design (guarded by the
person_period_n check), and fits R glm with diagnostics (converged / n_iter /
warnings / max|coef|). The verdict compares pystatistics' flag to R's:

  flag-match            py.converged == R.converged
  py-more-conservative  py.converged=False where R reports True (a defensible
                        difference: on a degenerate / separated design where the
                        MLE is at +/-inf, R declares 'converged' with a warning at
                        a large finite iterate while pystatistics honestly reports
                        non-convergence at the iteration cap) -- recorded, not
                        treated as a defect, but flagged for the writeup
  FLAG-WRONG            py.converged=True where R diverged AND py's coefficients are
                        still running away (a false-positive convergence claim) --
                        STOP and surface (this is the R9/G3 failure mode)

Cases span well-posed (lung coarse bins; flchain yearly), the intervals=None
default, the separation-prone extreme, a perfectly-separated covariate, and an
all-censored degenerate design (where the changelog says no IRLS runs).

    python -m drivers.survival.generate_convergence_diag --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.survival import datasets
from drivers.survival._continuous_synth import continuous_survival
from drivers.survival._person_period import build_person_period
from drivers.survival.run_r_survival import run_r_glmdiag

_REPO = Path(__file__).resolve().parent.parent.parent


def _cases() -> list[tuple[str, tuple, Any]]:
    """(label, (time, event, X, names), intervals_or_None)."""
    t, e, X, nm = datasets.load_lung_cox()
    b5 = datasets.discrete_interval_bounds(t, e, n_bins=5)
    cases: list[tuple[str, tuple, Any]] = [
        ("lung_coarse_5bin", (t, e, X, nm), b5),
        ("lung_default", (t, e, X, nm), None),
        ("separation_extreme", continuous_survival(60, seed=7, censoring="none"), None),
    ]
    # flchain yearly (well-posed at scale)
    ft, fe, fX, fnm = datasets.load_flchain()
    fb = datasets.flchain_interval_bounds(ft, fe, 365.25)
    cases.append(("flchain_yearly", (ft, fe, fX, fnm), fb))

    # perfectly-separated covariate: a 0/1 predictor that perfectly splits late events
    rng = np.random.default_rng(1)
    n = 80
    tt = np.sort(rng.uniform(1, 100, n)) + np.arange(n) * 1e-3
    ev = np.ones(n)
    xperf = (tt > tt[n // 2]).astype(float)
    cases.append(("separated_covariate", (tt, ev, xperf.reshape(-1, 1), ["x"]),
                  datasets.discrete_interval_bounds(tt, ev, 5)))

    # all-censored degenerate design (no events at all)
    tc, ec, Xc, nmc = datasets.load_lung_cox()
    cases.append(("all_censored", (tc, np.zeros_like(ec), Xc, nmc),
                  datasets.discrete_interval_bounds(tc, ec, 5)))
    return cases


def generate(host: str) -> Path:
    from pystatistics.survival import discrete_time

    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []

    for label, (time_, event, X, names), intervals in _cases():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = discrete_time(time_, event, X, names=names, intervals=intervals,
                                backend="cpu")
        py_conv = bool(sol.converged)
        py_iter = int(sol.n_iter)
        n_int = int(sol.n_intervals)
        pp_n = int(sol.person_period_n)

        # Reconstruct the exact design R must fit (default uses unique event times).
        bounds = (np.unique(time_[event == 1]) if intervals is None
                  else np.asarray(intervals, float))
        X_pp, y_pp, n_int_rec = build_person_period(time_, event, X, bounds)
        if len(y_pp) != pp_n:
            raise RuntimeError(f"{label}: person-period mismatch {len(y_pp)} vs {pp_n}")

        r = run_r_glmdiag(X_pp, y_pp, n_intervals=n_int_rec, covariate_names=names)
        r_conv = bool(r["converged"])
        r_iter = int(r["n_iter"])

        if py_conv == r_conv:
            verdict = "flag-match"
        elif (not py_conv) and r_conv:
            verdict = "py-more-conservative"
        else:
            # py says converged, R diverged: a possible false-positive. The R9/G3
            # test -- is it really wrong? Only if R failed to converge AND R's
            # coefficients blew up far past py's (py claims a stable point R denies).
            verdict = "FLAG-WRONG"

        row = {
            "case": label, "n": len(time_), "n_events": int(event.sum()),
            "n_intervals": n_int, "person_period_n": pp_n,
            "py_converged": py_conv, "py_n_iter": py_iter,
            "r_converged": r_conv, "r_n_iter": r_iter,
            "r_warnings": (r["warnings"][0][:60] if r["warnings"] else "none"),
            "max_abs_coef_r": r["max_abs_coef"], "verdict": verdict,
        }
        rows.append(row)
        records.append({"engine": "convergence_diag:cpu", "dataset": label,
                        "n": len(time_), "p": len(names), **row})
        if verdict == "FLAG-WRONG":
            flagged.append(row)
        print(f"  {label:20} py(conv={py_conv},it={py_iter:>2}) "
              f"R(conv={r_conv},it={r_iter:>2},warn={'Y' if r['warnings'] else 'n'}) "
              f"maxcoef={r['max_abs_coef']:.3g} -> {verdict}", flush=True)

    config = {"study": "convergence_diag_c5",
              "reference": "R glm(binomial) convergence behaviour on the identical design",
              "feature": "DiscreteTimeSolution.converged / .n_iter (new in 4.3.3, C5)"}
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "survival" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"convergence_diag_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    cols = ["case", "n", "n_events", "n_intervals", "person_period_n",
            "py_converged", "py_n_iter", "r_converged", "r_n_iter", "r_warnings",
            "max_abs_coef_r", "verdict"]
    with (out_dir / f"{stem}_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {run_path}")
    from collections import Counter
    print("verdict tally:", dict(Counter(r["verdict"] for r in rows)))
    if flagged:
        print("\n" + "!" * 72)
        print(f"!!! {len(flagged)} FLAG-WRONG case(s) -- a false-positive convergence "
              "claim. STOP, surface to user (R9/G3).")
        print("!" * 72)
    return run_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    args = ap.parse_args()
    generate(args.host)


if __name__ == "__main__":
    main()
