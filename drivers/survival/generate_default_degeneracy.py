"""RIGOR R15 — validate the DEFAULT ``discrete_time(intervals=None)`` invocation.

The frozen survival validation covered only coarse, expert-chosen bins (5-bin /
calendar). R15 demands the call a NAIVE user actually makes: the default, which on
continuous data makes one interval per unique event time. The fear (recorded as a
limitation in the v4.2.4 report) was that this "separates perfectly" and silently
returns garbage with huge coefficients — which would be a first-class correctness
defect (must fail loud or warn, A6/R15).

This driver determines the behaviour EMPIRICALLY and validates it against the
reference, mirroring the regression R10 behaviour-match study. For each continuous
design it:

  1. calls the public ``discrete_time(intervals=None)`` (CPU, fp64) and records
     whether it RAISED, WARNED, or RETURNED — and if it returned, the covariate
     coef/SE/z/p, the per-interval baseline hazard, and the GLM deviance/AIC;
  2. reconstructs the EXACT every-unique-event-time person-period design (guarded
     by the ``person_period_n`` check) and fits R ``glm(binomial)`` on it, capturing
     R's OWN behaviour — convergence flag, iteration count, glm warnings (e.g.
     "fitted probabilities numerically 0 or 1 occurred"), and max |coefficient|;
  3. fits ``regression.fit(binomial)`` on the same reconstruction to expose the
     UNDERLYING GLM's convergence flag / iteration count (the public
     ``DiscreteTimeSolution`` does not surface them — a transparency gap recorded
     as a finding), and the full coefficient vector for the max-|coef| comparison;
  4. scores the default against R: covariate coef/SE agreement, deviance/AIC
     agreement (the R8 cross-module check on the 4.3.2 binomial-deviance clamp,
     evaluated here where fitted probabilities can reach ~1), and a behaviour
     classification.

Designs span the realistic continuous regime (lung; censored synthetic seeds) and
the separation-prone extreme (no censoring, strictly-increasing unique event times
so a tail interval holds a single at-risk subject who has the event).

A "default-silent-wrong" verdict (the default returns but disagrees with R on the
covariate optimum, or returns a non-finite/garbage fit R refused) is the
correctness defect R15 forbids — the driver prints a loud banner; STOP and surface
to the user before anything else.

    python -m drivers.survival.generate_default_degeneracy --host powerhouse
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
from pystatsval.serialize import build_run

# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
# Drivers hardcode their artifact dir, so an ordinary run would otherwise
# silently destroy the artifacts a report was blessed against.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from drivers.survival import datasets
from drivers.survival._continuous_synth import continuous_survival
from drivers.survival._person_period import build_person_period
from drivers.survival.run_r_survival import run_r_glmdiag

from artifact_root import artifact_root  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent

# Covariate agreement at or below this (relative) means the default reached R's
# SAME covariate optimum — sound inference, not separated garbage. fp64 round-off
# on a well-posed design is ~1e-12; 1e-6 is a generous behaviour-match boundary.
_COEF_TOL = 1e-6
# Deviance/AIC agreement boundary for the R8 binomial-clamp cross-check.
_DEV_TOL = 1e-6


def _rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300)))


def _cases() -> list[tuple[str, tuple]]:
    """(label, (time, event, X, names)) for the R15 sweep."""
    t, e, X, nm = datasets.load_lung_cox()
    cases = [("lung_default", (t, e, X, nm))]
    for seed in (1, 2, 3):
        cases.append((f"continuous_seed{seed}",
                      continuous_survival(120, seed=seed, censoring="realistic")))
    cases.append(("separation_extreme",
                  continuous_survival(60, seed=7, censoring="none")))
    return cases


def generate(host: str) -> Path:
    from pystatistics.survival import discrete_time
    from pystatistics.regression import Design, fit
    from pystatistics.core.exceptions import PyStatisticsError

    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []

    for label, (time_, event, X, names) in _cases():
        n_cov = len(names)
        n_ev = int(event.sum())

        # 1. The public DEFAULT call (intervals=None).
        pystat_behavior = ""
        raised = None
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            try:
                sol = discrete_time(time_, event, X, names=names)  # default
                py_cov = np.asarray(sol.coefficients, float)
                py_se = np.asarray(sol.standard_errors, float)
                bh = np.asarray(sol.baseline_hazard, float)
                py_dev = float(sol.glm_deviance)
                py_aic = float(sol.glm_aic)
                n_int = int(sol.n_intervals)
                pp_n = int(sol.person_period_n)
                py_warns = [str(w.message) for w in wl] + list(sol.warnings or ())
            except (PyStatisticsError, Exception) as exc:  # noqa: BLE001
                raised = f"{type(exc).__name__}: {exc}"

        if raised is not None:
            # Failing loud is an ACCEPTABLE R15 outcome (A6) — record and continue.
            pystat_behavior = f"raised {raised.split(':',1)[0]}"
            row = {"case": label, "n": len(time_), "n_events": n_ev,
                   "n_intervals": None, "person_period_n": None,
                   "pystat_behavior": pystat_behavior, "r_behavior": "",
                   "match": "fail-loud (A6-ok)",
                   "cov_coef_max_rel_vs_r": None, "cov_se_max_rel_vs_r": None,
                   "baseline_hazard_max": None, "max_abs_glm_coef": None,
                   "deviance_rel_vs_r": None, "aic_rel_vs_r": None,
                   "detail": raised[:120]}
            rows.append(row)
            records.append({"engine": "default_r15:cpu", "dataset": label, **row})
            print(f"  {label:20} pystat RAISED -> {raised[:80]}", flush=True)
            continue

        # 2. Reconstruct the exact default (every-unique-event-time) design.
        bounds = np.unique(time_[event == 1])
        X_pp, y_pp, n_int_rec = build_person_period(time_, event, X, bounds)
        if len(y_pp) != pp_n:
            raise RuntimeError(
                f"{label}: person-period reconstruction mismatch "
                f"({len(y_pp)} vs library {pp_n}) — update _person_period.py")

        # 3. Underlying GLM convergence (DiscreteTimeSolution hides it) + full coef.
        g = fit(Design.from_arrays(X_pp.astype(float), y_pp.astype(float)),
                family="binomial", backend="cpu")
        glm_coef = np.asarray(g.coefficients, float)
        glm_conv = bool(getattr(g, "converged", True))
        glm_iter = int(getattr(g, "n_iter", 0) or 0)
        py_max_abs_coef = float(np.abs(glm_coef).max())

        # 4. R reference behaviour on the identical design.
        r = run_r_glmdiag(X_pp, y_pp, n_intervals=n_int_rec, covariate_names=names)
        coef_rel = _rel(py_cov, r["coefficients"])
        se_rel = _rel(py_se, r["standard_errors"])
        dev_rel = abs(py_dev - r["deviance"]) / max(abs(r["deviance"]), 1e-300)
        aic_rel = abs(py_aic - r["aic"]) / max(abs(r["aic"]), 1e-300)

        r_behavior = (f"converged={r['converged']} ({r['n_iter']} it), "
                      f"{r['n_warnings']} warning(s), max|coef|={r['max_abs_coef']:.3g}")
        # Behaviour classification.
        sound = (coef_rel <= _COEF_TOL and np.isfinite(py_cov).all())
        if not sound:
            match = "DEFAULT-SILENT-WRONG"
        else:
            match = "behaviour-match"
        pystat_behavior = (f"returns; GLM converged={glm_conv} ({glm_iter} it); "
                           f"covariate inference sound")
        warn_note = (f"; pystat warnings={py_warns}" if py_warns else "")
        detail = (f"R warns={r['warnings'] or 'none'}; "
                  f"py_max|coef|={py_max_abs_coef:.3g}{warn_note}")

        row = {
            "case": label, "n": len(time_), "n_events": n_ev,
            "n_intervals": n_int, "person_period_n": pp_n,
            "pystat_behavior": pystat_behavior, "r_behavior": r_behavior,
            "match": match,
            "cov_coef_max_rel_vs_r": coef_rel, "cov_se_max_rel_vs_r": se_rel,
            "baseline_hazard_max": float(bh.max()),
            "max_abs_glm_coef": py_max_abs_coef,
            "deviance_rel_vs_r": dev_rel, "aic_rel_vs_r": aic_rel,
            "detail": detail[:200],
        }
        rows.append(row)
        records.append({"engine": "default_r15:cpu", "dataset": label,
                        "n": len(time_), "p": n_cov, **row})
        if match != "behaviour-match":
            flagged.append(row)
        print(f"  {label:20} nint={n_int:>4} pp={pp_n:>6} coef_rel={coef_rel:.2e} "
              f"se_rel={se_rel:.2e} dev_rel={dev_rel:.2e} bh_max={bh.max():.4f} "
              f"R[{r_behavior}] -> {match}", flush=True)

    config = {"study": "default_degeneracy_r15",
              "coef_tol": _COEF_TOL, "dev_tol": _DEV_TOL,
              "reference": "R glm(binomial) on the every-unique-event-time design",
              "cases": [c[0] for c in _cases()]}
    run = build_run(env=env, config=config, records=records)
    out_dir = artifact_root(_REPO) / "survival" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"default_degeneracy_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    cols = ["case", "n", "n_events", "n_intervals", "person_period_n",
            "pystat_behavior", "r_behavior", "match", "cov_coef_max_rel_vs_r",
            "cov_se_max_rel_vs_r", "baseline_hazard_max", "max_abs_glm_coef",
            "deviance_rel_vs_r", "aic_rel_vs_r", "detail"]
    with (out_dir / f"{stem}_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {run_path}")
    from collections import Counter
    print("match tally:", dict(Counter(r["match"] for r in rows)))
    if flagged:
        print("\n" + "!" * 72)
        print(f"!!! {len(flagged)} DEFAULT-SILENT-WRONG case(s) — STOP, surface to user "
              "(R15/A6 violation: the default returned but disagrees with R).")
        print("!" * 72)
    return run_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    args = ap.parse_args()
    generate(args.host)


if __name__ == "__main__":
    main()
