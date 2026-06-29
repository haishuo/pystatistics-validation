"""Adversarial no-silent-wrong proof for the fp32 GPU GLM relaxation (RIGOR R12).

4.2.3/4.2.4 relaxed the plain (unpenalized) float32 log-link GLM from fail-loud to
"converges across the validated stability grid". But that grid is well-conditioned.
R12 (the complement of R9) demands a dedicated stress test that the accept/refuse
gate is a TRUE CLASSIFIER in both directions:

  * never ACCEPT a wrong fit   (no silent-wrong band — the A6 violation R12 forbids)
  * never REFUSE a correct fit (no false negative — the R9 violation)

This driver constructs log-link Poisson/Gamma designs that STRADDLE the float32
precision floor along three independent stress axes, and for each one decides the
gate's verdict against the float64 reference:

  - **conditioning**  near-collinear predictors (squares badly in the float32
                      XᵀWX), severity swept by the collinearity gap.
  - **large_coef**    coefficients scaled up so the IRLS weights span many orders
                      of magnitude (the log-link weight is the mean itself).
  - **low_weight**    a block of rows driven to mean≈0 (IRLS weight≈0) — the
                      discrete-time baseline-hazard pathology, where most of the
                      Fisher information sits in a few heavily-weighted rows.

For every design:
  * the float64 reference is the CPU fit (and, on CUDA, gpu_fp64 — exact);
  * the plain float32 GPU fit is attempted. If ACCEPTED, it is compared to the
    float64 reference — it must have reached the SAME optimum (deviance to the
    float32 tier). If REFUSED (fail-loud NumericalError), the fit is FORCED and
    compared to float64 to confirm the forced fit is genuinely wrong (a true
    positive, not an R9 false negative).

Classification per design:
  safe                    accepted AND deviance matches float64 (the right optimum)
  refused-true-positive   refused AND the forced float32 fit is genuinely wrong /
                          overflows (the gate correctly caught an fp32-infeasible fit)
  refused-force-recovers  refused BUT force=True yields a correct fit — the
                          documented override working; a conservative gate, never a
                          silently-wrong number
  SILENT-WRONG            accepted BUT deviance is materially worse (A6 violation)

A SILENT-WRONG verdict is a correctness regression behind a speed win — the driver
prints a loud banner and the row is preserved for the report; surface it to the
user before doing anything else.

    python -m drivers.regression.generate_fp32_boundary --host powerhouse --device mps
    python -m drivers.regression.generate_fp32_boundary --host forge --device cuda
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

_REPO = Path(__file__).resolve().parent.parent.parent

# A deviance gap larger than this (relative) means the float32 fit reached a
# DIFFERENT (worse) optimum than float64 — i.e. it is wrong, not merely
# non-identifiable in its coefficients. The float32 tier is ~1e-4..1e-6; 1e-3 is a
# generous boundary that still cleanly separates "same optimum" from "wrong".
_DEV_TOL = 1e-3


def _fam(family: str):
    from pystatistics.regression import GammaFamily
    return GammaFamily(link="log") if family == "gamma" else family


def _design(mechanism: str, severity: float, family: str, *, seed: int):
    """Build a log-link GLM design stressed along one axis at a given severity."""
    rng = np.random.default_rng(seed)
    n, p = 4000, 6

    if mechanism == "conditioning":
        # near-collinear predictor pair; smaller `severity` (eps) → worse cond.
        base = rng.standard_normal((n, p - 1))
        collinear = base[:, 0:1] + severity * rng.standard_normal((n, 1))
        Xp = np.column_stack([base, collinear])
        beta = rng.standard_normal(p + 1) * 0.4 / np.sqrt(p)
    elif mechanism == "large_coef":
        Xp = rng.standard_normal((n, p))
        beta = rng.standard_normal(p + 1) * (severity / np.sqrt(p))   # severity = coef scale
    elif mechanism == "low_weight":
        Xp = rng.standard_normal((n, p))
        beta = rng.standard_normal(p + 1) * 0.4 / np.sqrt(p)
    else:
        raise ValueError(mechanism)

    X = np.column_stack([np.ones(n), Xp])
    eta = X @ beta
    if mechanism == "low_weight":
        # Drive a block of rows to mean≈0 (weight≈0): a large negative offset on
        # the linear predictor for `severity` fraction of the rows.
        k = int(severity * n)
        eta = eta.copy()
        eta[:k] -= 18.0
    eta = np.clip(eta, -30.0, 30.0)
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    else:  # gamma (log link)
        y = rng.gamma(shape=2.0, scale=np.exp(eta) / 2.0) + 1e-9
    return X, y


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# Severity sweeps per mechanism (deliberately pushed past the float32 floor).
_SWEEPS = {
    "conditioning": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],   # eps (smaller = worse)
    "large_coef":   [1.0, 3.0, 6.0, 10.0, 16.0, 24.0],      # coefficient scale
    "low_weight":   [0.1, 0.5, 0.8, 0.9, 0.95, 0.99],       # fraction near-zero weight
}


def generate(host: str, device: str) -> Path:
    from pystatistics.regression import Design, fit
    from pystatistics.core.exceptions import NumericalError, PyStatisticsError

    env = env_manifest(device=device, host=host)
    require_pypi(env)
    has_fp64 = (device == "cuda")

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    silent_wrong: list[dict[str, Any]] = []
    seed = 0

    for family in ("poisson", "gamma"):
        for mechanism, sweep in _SWEEPS.items():
            for sev in sweep:
                seed += 1
                X, y = _design(mechanism, sev, family, seed=seed)
                d = Design.from_arrays(X, y)
                npcond = float(np.linalg.cond(X))

                # float64 reference (CPU; always available, fp64).
                cpu = fit(d, family=_fam(family), backend="cpu")
                ref_dev = float(cpu.deviance)
                ref_coef = np.asarray(cpu.coefficients, float)

                # plain float32 GPU fit — the relaxed path under test.
                plain_status = ""
                acc_coef_rel = acc_dev_rel = None
                forced_coef_rel = forced_dev_rel = None
                classification = ""
                try:
                    g = fit(d, family=_fam(family), backend="gpu")
                    plain_status = "converged" if bool(getattr(g, "converged", True)) else "not_converged"
                    acc_coef_rel = _rel(g.coefficients, ref_coef)
                    acc_dev_rel = abs(float(g.deviance) - ref_dev) / max(abs(ref_dev), 1e-300)
                    if acc_dev_rel <= _DEV_TOL:
                        classification = "safe"
                    else:
                        classification = "SILENT-WRONG"
                except (NumericalError, PyStatisticsError) as exc:
                    plain_status = f"refused:{type(exc).__name__}"
                    # Force the float32 fit and check whether refusal was warranted.
                    try:
                        gf = fit(d, family=_fam(family), backend="gpu", force=True)
                        forced_coef_rel = _rel(gf.coefficients, ref_coef)
                        forced_dev_rel = abs(float(gf.deviance) - ref_dev) / max(abs(ref_dev), 1e-300)
                        # A non-finite forced deviance means the forced fit blew up
                        # (overflow) — the refusal was warranted, not conservative.
                        if (not np.isfinite(forced_dev_rel)) or forced_dev_rel > _DEV_TOL:
                            classification = "refused-true-positive"
                        else:
                            classification = "refused-force-recovers"
                    except Exception:  # noqa: BLE001 - forced fit also blew up → refusal warranted
                        classification = "refused-true-positive"
                        forced_dev_rel = float("inf")

                # On CUDA, gpu_fp64 is an exact cross-check (numerically exact path).
                fp64_dev_rel = None
                if has_fp64:
                    try:
                        g64 = fit(d, family=_fam(family), backend="gpu_fp64")
                        fp64_dev_rel = abs(float(g64.deviance) - ref_dev) / max(abs(ref_dev), 1e-300)
                    except Exception:  # noqa: BLE001
                        fp64_dev_rel = None

                row = {
                    "family": family, "mechanism": mechanism, "severity": sev,
                    "np_cond": npcond, "plain_status": plain_status,
                    "accepted_coef_rel_vs_fp64": acc_coef_rel,
                    "accepted_dev_rel_vs_fp64": acc_dev_rel,
                    "forced_coef_rel_vs_fp64": forced_coef_rel,
                    "forced_dev_rel_vs_fp64": forced_dev_rel,
                    "fp64_dev_rel_vs_cpu": fp64_dev_rel,
                    "classification": classification,
                }
                rows.append(row)
                records.append({"engine": f"fp32boundary:{device}", "dataset": mechanism,
                                "n": X.shape[0], "p": X.shape[1], **row})
                if classification == "SILENT-WRONG":
                    silent_wrong.append(row)
                print(f"  {family:7} {mechanism:12} sev={sev:<7g} cond={npcond:.1e} "
                      f"{plain_status:24} -> {classification}"
                      + (f"  accDevRel={acc_dev_rel:.2e}" if acc_dev_rel is not None else "")
                      + (f"  forcedDevRel={forced_dev_rel:.2e}" if forced_dev_rel is not None else ""),
                      flush=True)

    config = {"study": "fp32_boundary_r12", "device": device,
              "dev_tol": _DEV_TOL, "sweeps": _SWEEPS,
              "mechanisms": list(_SWEEPS.keys()), "families": ["poisson", "gamma"]}
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fp32_boundary_{device}_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    cols = ["family", "mechanism", "severity", "np_cond", "plain_status",
            "accepted_coef_rel_vs_fp64", "accepted_dev_rel_vs_fp64",
            "forced_coef_rel_vs_fp64", "forced_dev_rel_vs_fp64",
            "fp64_dev_rel_vs_cpu", "classification"]
    with (out_dir / f"{stem}_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {run_path}")

    # Summary by classification — the headline of the no-silent-wrong proof.
    from collections import Counter
    tally = Counter(r["classification"] for r in rows)
    print("classification tally:", dict(tally))
    if silent_wrong:
        print("\n" + "!" * 72)
        print(f"!!! {len(silent_wrong)} SILENT-WRONG fit(s) detected on {device} — "
              "STOP and surface to the user (A6/R12 violation).")
        for r in silent_wrong:
            print(f"!!!   {r['family']} {r['mechanism']} sev={r['severity']} "
                  f"dev_rel={r['accepted_dev_rel_vs_fp64']:.3e}")
        print("!" * 72)
    return run_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--device", default="mps", choices=["mps", "cuda"])
    args = ap.parse_args()
    generate(args.host, args.device)


if __name__ == "__main__":
    main()
