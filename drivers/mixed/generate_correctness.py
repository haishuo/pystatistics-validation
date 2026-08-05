"""Generate the LMM correctness-vs-R artifacts (CPU reference path).

One job: for each canonical lme4 design fit BOTH pystatistics ``mixed.lmm`` and R
``lme4``/``lmerTest`` on the identical numbers, reduce every estimate
(fixed effects, SEs, Satterthwaite df, p-values, variance components, BLUPs,
logLik/REML criterion, AIC/BIC) to scalar agreement metrics split by the TWO
honest tolerance tiers, and freeze a ``validation-run/v1`` artifact plus a flat
summary CSV under ``artifacts/mixed/v<ver>/runs/``.

The two-tier contract (priority 1, stated honestly):
  - TIGHT tier  — fixed-effect coefficients + the log-likelihood/REML criterion.
    The profiled (RE)ML estimator pins these tightly; agreement is ~machine-eps.
  - OPTIMIZER tier — variance components, their derived SEs, Satterthwaite df and
    p-values, BLUPs. lme4 (bobyqa/Nelder-Mead) and pystatistics (scipy L-BFGS-B)
    stop the variance-parameter search at a finite optimizer tolerance, so these
    necessarily agree to a looser, optimizer-bound level — a contract to STATE,
    not a defect to hide.

R10 hard cases live in ``generate_hardcases.py``; this module covers the main
grid (random intercept, random slopes+correlation, crossed, nested) under both
REML and ML, plus the R15 default invocation.

PyPI-only (``require_pypi``). Run from the dedicated 4.4.1 PyPI venv with
``MVNMLE_DATA_DIR`` pointing at the curated HDF5 store.

    python -m drivers.mixed.generate_correctness --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import socket
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

from drivers.mixed import datasets
from drivers.mixed.run_pystatistics import run_lmm_record
from drivers.mixed.run_r_mixed import run_r_lmm_record

_REPO = Path(__file__).resolve().parent.parent.parent

# Correctness grid: (dataset_key, reml). All five at REML (the default); a ML
# (reml=False) pass on the two designs with non-trivial fixed effects / a clean
# variance estimate, since ML is what a user runs for LRTs on fixed effects.
_GRID = [
    ("sleepstudy", True), ("penicillin", True), ("dyestuff", True),
    ("dyestuff2", True), ("pastes", True),
    ("sleepstudy", False), ("dyestuff", False),
]

# A variance component is "at the boundary" (compared by ABSOLUTE tolerance, not
# relative) when both engines put it at/near zero — a relative metric there is
# meaningless (R reports exactly 0; pystatistics ~1e-16).
_BOUNDARY_ABS = 1e-6


def _max_rel(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _max_abs(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _pair_varcomps(py_vc, r_vc):
    """Pair variance components by (group, name); return aligned (py, r) lists of
    {variance, std_dev, corr}. Both engines emit the Residual row."""
    rd = {(d["group"], d["name"]): d for d in r_vc}
    pairs = []
    for d in py_vc:
        k = (d["group"], d["name"])
        if k in rd:
            pairs.append((d, rd[k], k))
    return pairs


def _varcomp_agreement(py_vc, r_vc) -> dict[str, Any]:
    """Split variance-component agreement into away-from-boundary (relative) and
    boundary (absolute), plus correlation agreement (the loosest sub-tier)."""
    pairs = _pair_varcomps(py_vc, r_vc)
    rel, boundary_abs, corr_abs = [], [], []
    for pd_, rd_, _k in pairs:
        if abs(rd_["variance"]) <= _BOUNDARY_ABS:
            boundary_abs.append(abs(pd_["variance"] - rd_["variance"]))
        else:
            rel.append(abs(pd_["variance"] - rd_["variance"]) / abs(rd_["variance"]))
            rel.append(abs(pd_["std_dev"] - rd_["std_dev"]) / abs(rd_["std_dev"]))
        if pd_["corr"] is not None and rd_["corr"] is not None:
            corr_abs.append(abs(pd_["corr"] - rd_["corr"]))
    return {
        "varcomp_max_rel": (max(rel) if rel else 0.0),
        "varcomp_boundary_max_abs": (max(boundary_abs) if boundary_abs else None),
        "corr_max_abs": (max(corr_abs) if corr_abs else None),
        "n_varcomps": len(pairs),
    }


def _blup_agreement(py_blups, r_blups) -> float | None:
    """Max relative disagreement of the conditional modes across all grouping
    factors. BLUPs are an optimizer-tier quantity (they depend on the converged
    variance components). None if no BLUPs are exposed."""
    worst = None
    for grp, pmat in py_blups.items():
        if grp not in r_blups:
            continue
        a = np.asarray(pmat, float)
        b = np.asarray(r_blups[grp], float)
        if a.shape != b.shape or a.size == 0:
            continue
        # BLUPs centre near zero; use a scaled-absolute metric (abs / RMS) so a
        # near-zero mode does not blow up a pure relative ratio.
        rms = float(np.sqrt(np.mean(b ** 2))) or 1.0
        m = float(np.max(np.abs(a - b)) / rms)
        worst = m if worst is None else max(worst, m)
    return worst


def _agreement(sut: dict[str, Any], ref: dict[str, Any], ds) -> dict[str, Any]:
    vc = _varcomp_agreement(sut["var_components"], ref["var_components"])
    row: dict[str, Any] = {
        "dataset": ds.key,
        "criterion": "REML" if ds.reml else "ML",
        "n": sut.get("n"), "p": sut.get("p"),
        "n_groups": ";".join(f"{k}={len(np.unique(v))}" for k, v in ds.groups.items()),
        # --- TIGHT tier ---
        "coef_max_rel": _max_rel(sut["coefficients"], ref["coefficients"]),
        "loglik_rel": _max_rel([sut["log_likelihood"]], [ref["log_likelihood"]]),
        "reml_crit_rel": _max_rel([sut["reml_criterion"]], [ref["reml_criterion"]]),
        "aic_rel": _max_rel([sut["aic"]], [ref["aic"]]),
        "bic_rel": _max_rel([sut["bic"]], [ref["bic"]]),
        # --- OPTIMIZER tier ---
        "se_max_rel": _max_rel(sut["standard_errors"], ref["standard_errors"]),
        "tval_max_rel": _max_rel(sut["t_values"], ref["t_values"]),
        "df_satt_max_rel": (_max_rel(sut["df_satterthwaite"], ref["df_satterthwaite"])
                            if sut.get("df_satterthwaite") else None),
        "pval_max_abs": (_max_abs(sut["p_values"], ref["p_values"])
                         if sut.get("p_values") else None),
        "varcomp_max_rel": vc["varcomp_max_rel"],
        "varcomp_boundary_max_abs": vc["varcomp_boundary_max_abs"],
        "corr_max_abs": vc["corr_max_abs"],
        "blup_max_scaled": _blup_agreement(sut["blups"], ref["blups"]),
        # --- behaviour (R10) ---
        "py_singular": sut.get("is_singular"),
        "r_singular": ref.get("is_singular"),
        "r_diagnostics": " | ".join(ref.get("r_warnings", []) or []),
        "py_converged": sut.get("converged"),
    }
    return row


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for key, reml in _GRID:
        ds = datasets.LMM_LOADERS[key]()
        if not reml:
            ds = dataclasses.replace(ds, reml=False)
        sut = run_lmm_record(ds, repeats=repeats, warmup=1)
        if sut.get("error"):
            raise RuntimeError(f"pystatistics LMM failed for {key} "
                               f"({'REML' if reml else 'ML'}): {sut['error']}")
        ref, _raw = run_r_lmm_record(ds, reps=repeats)
        records.extend([sut, ref])
        row = _agreement(sut, ref, ds)
        rows.append(row)
        print(f"  {key:11s} {row['criterion']:4s}  "
              f"coef={row['coef_max_rel']:.1e} ll={row['loglik_rel']:.1e} "
              f"se={row['se_max_rel']:.1e} vc={row['varcomp_max_rel']:.1e} "
              f"df={(_fmt(row['df_satt_max_rel']))} blup={_fmt(row['blup_max_scaled'])} "
              f"| sing py/r={row['py_singular']}/{row['r_singular']}")

    config = {
        "study": "correctness_vs_r",
        "grid": [{"dataset": k, "criterion": "REML" if r else "ML"} for k, r in _GRID],
        "backend": "cpu",
        "repeats": repeats,
        "reference": "R lme4::lmer (estimates) + lmerTest::lmer (Satterthwaite df/p)",
        "tolerance_tiers": {
            "tight": "fixed-effect coefficients, log-likelihood, REML criterion, "
                     "AIC/BIC — profiled-(RE)ML pins these to ~machine precision",
            "optimizer": "variance components, SEs, Satterthwaite df, p-values, "
                         "BLUPs — bounded by the variance-parameter optimizer "
                         "tolerance (lme4 bobyqa vs scipy L-BFGS-B)",
        },
        "boundary_abs_tol": _BOUNDARY_ABS,
    }
    run = build_run(env=env, config=config, records=records)

    out_dir = _REPO / "artifacts" / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"correctness_cpu_{host}.json", run)
    _write_summary_csv(out_dir / f"correctness_{host}_summary.csv", rows)
    print(f"\nwrote {run_path}")
    return run_path


def _fmt(v) -> str:
    return "----" if v is None else f"{v:.1e}"


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
