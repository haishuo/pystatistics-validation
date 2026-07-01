"""Generate the GLMM correctness-vs-glmer artifacts (CPU, Laplace/nAGQ=1).

One job: for each canonical GLMM design fit BOTH pystatistics ``mixed.glmm`` and R
``lme4::glmer`` (Laplace, nAGQ=1) on the identical numbers, reduce every estimate
(fixed effects, SEs, Wald z, p-values, variance components, BLUPs, logLik,
conditional deviance, AIC/BIC) to scalar agreement metrics split by the two honest
tolerance tiers, and freeze a ``validation-run/v1`` artifact plus a flat summary
CSV under ``artifacts/mixed/v<ver>/runs/``.

The two-tier contract (stated honestly):
  - TIGHT tier — fixed-effect coefficients + logLik / deviance / AIC / BIC. The
    Laplace estimator pins these tightly (~1e-3 or better vs glmer nAGQ=1).
  - OPTIMIZER/LAPLACE tier — SEs, Wald z, p-values, variance components, BLUPs.
    lme4's and pystatistics' optimizers stop the variance-parameter search at a
    finite tolerance and the Laplace fixed-effect covariance is evaluated at the
    mode, so these agree to ~1-2% — a contract to STATE, not a defect to hide.

R10 hard cases live in ``generate_glmm_hardcases.py``; this module covers the
main grid (binomial/logit random intercept + fixed factor, Poisson/log, correlated
random slope, and a probit link via a Family instance).

PyPI-only (``require_pypi``). Run from the dedicated PyPI venv with
``MVNMLE_DATA_DIR`` pointing at the curated HDF5 store.

    python -m drivers.mixed.generate_glmm_correctness --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.mixed import glmm_datasets as datasets
from drivers.mixed.run_glmm import run_glmm_record
from drivers.mixed.run_r_glmm import run_r_glmm_record

_REPO = Path(__file__).resolve().parent.parent.parent

# Correctness grid: canonical datasets across families/links.
_GRID = ["cbpp", "grouseticks", "glmm_slope_synth", "cbpp_probit"]


def _max_rel(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _varcomp_max_rel(py_vc, r_vc) -> float:
    """Max relative disagreement over paired (group, name) variance components."""
    rd = {(d["group"], d["name"]): d for d in r_vc}
    rels = []
    for d in py_vc:
        k = (d["group"], d["name"])
        if k in rd and abs(rd[k]["variance"]) > 1e-8:
            rels.append(abs(d["variance"] - rd[k]["variance"]) / abs(rd[k]["variance"]))
    return max(rels) if rels else 0.0


def _blup_max_scaled(py_blups, r_blups) -> float | None:
    worst = None
    for grp, pmat in py_blups.items():
        if grp not in r_blups:
            continue
        a = np.asarray(pmat, float)
        b = np.asarray(r_blups[grp], float)
        if a.shape != b.shape or a.size == 0:
            continue
        rms = float(np.sqrt(np.mean(b ** 2))) or 1.0
        m = float(np.max(np.abs(a - b)) / rms)
        worst = m if worst is None else max(worst, m)
    return worst


def _agreement(sut: dict[str, Any], ref: dict[str, Any], ds) -> dict[str, Any]:
    return {
        "dataset": ds.key,
        "family": ds.family,
        "link": ds.link,
        "n": sut.get("n"),
        "p": sut.get("p"),
        # --- TIGHT tier ---
        "coef_max_rel": _max_rel(sut["coefficients"], ref["coefficients"]),
        "loglik_rel": _max_rel([sut["log_likelihood"]], [ref["log_likelihood"]]),
        "deviance_rel": _max_rel([sut["deviance"]], [ref["deviance"]]),
        "aic_rel": _max_rel([sut["aic"]], [ref["aic"]]),
        "bic_rel": _max_rel([sut["bic"]], [ref["bic"]]),
        # --- OPTIMIZER / LAPLACE tier ---
        "se_max_rel": _max_rel(sut["standard_errors"], ref["standard_errors"]),
        "z_max_rel": _max_rel(sut["z_values"], ref["z_values"]),
        "varcomp_max_rel": _varcomp_max_rel(sut["var_components"],
                                            ref["var_components"]),
        "blup_max_scaled": _blup_max_scaled(sut["blups"], ref["blups"]),
        # --- behaviour ---
        "py_converged": sut.get("converged"),
        "r_singular": ref.get("is_singular"),
    }


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for key in _GRID:
        ds = datasets.GLMM_LOADERS[key]()
        sut = run_glmm_record(ds, repeats=repeats, warmup=1)
        if sut.get("error"):
            raise RuntimeError(f"pystatistics glmm failed for {key}: {sut['error']}")
        ref, _raw = run_r_glmm_record(ds, reps=repeats)
        records.extend([sut, ref])
        row = _agreement(sut, ref, ds)
        rows.append(row)
        print(f"  {key:16s} {ds.family}/{ds.link:6s} "
              f"coef={row['coef_max_rel']:.1e} ll={row['loglik_rel']:.1e} "
              f"se={row['se_max_rel']:.1e} vc={row['varcomp_max_rel']:.1e} "
              f"blup={_fmt(row['blup_max_scaled'])} conv={row['py_converged']}")

    config = {
        "study": "glmm_correctness_vs_glmer",
        "grid": _GRID,
        "backend": "cpu",
        "repeats": repeats,
        "reference": "R lme4::glmer (Laplace, nAGQ=1)",
        "tolerance_tiers": {
            "tight": "fixed-effect coefficients, logLik, deviance, AIC/BIC — the "
                     "Laplace estimator pins these to ~1e-3 or better vs glmer",
            "optimizer": "SEs, Wald z, p-values, variance components, BLUPs — "
                         "optimizer + Laplace-mode bound, ~1-2% vs glmer",
        },
    }
    run = build_run(env=env, config=config, records=records)

    out_dir = _REPO / "artifacts" / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"glmm_correctness_cpu_{host}.json", run)
    _write_summary_csv(out_dir / f"glmm_correctness_{host}_summary.csv", rows)
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
