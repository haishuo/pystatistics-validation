"""Inference (SE/CI) no-silent-wrong evidence — the R12 proof extended to the
standard errors, the output the inference user actually relies on (RIGOR R12 / R9).

R12 proved the fp32 GPU accept/refuse gate never accepts a fit at the WRONG
optimum (deviance to the fp32 tier). But the product's differentiator is
INFERENCE, and on an ill-conditioned design the coefficients become
non-identifiable even at the right optimum — coef_rel can reach ~1.0 while the
deviance is exact. The decisive question is therefore about the STANDARD ERRORS:
on such an accepted fit, do the reported fp32 SEs blow up to reflect the
instability (so a large SE warns the user, exactly as in R) — or do they
understate it (a coefficient that is wrong AND looks precise: an
inference-silent-wrong)?

This study sweeps near-collinear (conditioning) designs for OLS and the two
log-link GLM families, and for every ACCEPTED fp32 GPU fit compares its SEs to
the fp64 CPU fit and to R:

  - se_rel_gpu_vs_cpu   max relative SE gap, GPU(fp32) vs CPU(fp64) — must stay at
                        the documented Tier-2 contract (rtol <= 1e-2).
  - se_rel_gpu_vs_r     GPU vs R's SEs (the ground-truth "large SE = don't trust"
                        warning); the gamma rows carry the documented ~1e-2
                        dispersion-estimator gap on top.
  - cpu/gpu/r se_max     the SE on the least-identified coefficient — shown growing
                        with conditioning (the blow-up that warns the user).

Verdict per accepted fit:
  inference-warned       SEs match CPU (<=1e-2) AND no coefficient's SE is
                         understated (gpu_se >= 0.5*cpu_se) — the instability
                         surfaces in the SE, like R.
  INFERENCE-SILENT-WRONG a coefficient is off but its SE is understated — STOP.

Mathematically the two cannot decouple: the SE is sqrt(diag of the same
(XᵀWX)⁻¹) that drives the coefficient instability, so a non-identifiable
coefficient necessarily carries a large SE. This study is the empirical
confirmation, in fp32, on both MPS and CUDA.

    python -m drivers.regression.generate_inference_se --host powerhouse --device mps
    python -m drivers.regression.generate_inference_se --host forge --device cuda
"""

from __future__ import annotations

import argparse
import csv
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

from drivers.regression.run_r_regression import run_r_regression_record

_REPO = Path(__file__).resolve().parent.parent.parent

# model_key -> (pystatistics family arg builder tag, R family tag)
_MODELS = (
    ("ols",         None,        "lm"),
    ("glm_poisson", "poisson",   "poisson"),
    ("glm_gamma",   "gamma",     "Gamma"),
)
# Conditioning severities (collinearity gap): smaller gap -> higher condition.
_SEVERITIES = (1e-1, 1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)


def _fam(family: str | None):
    from pystatistics.regression import Gamma
    if family == "gamma":
        return Gamma(link="log")
    return family


def _design(family: str | None, gap: float, *, seed: int):
    """Near-collinear design: columns x1 and x2=x1+gap·noise plus two independent
    predictors. Smaller gap -> the x1/x2 pair is less identifiable (large SEs)."""
    rng = np.random.default_rng(seed)
    n = 4000
    x1 = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    xb = rng.standard_normal(n)
    xc = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x1 + gap * noise, xb, xc])
    beta = np.array([0.5, 0.3, -0.2, 0.15, 0.1])
    eta = np.clip(X @ beta, -8.0, 8.0)
    if family is None:                       # OLS
        y = eta + 0.3 * rng.standard_normal(n)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    else:                                    # gamma (log link)
        y = rng.gamma(shape=2.0, scale=np.exp(eta) / 2.0) + 1e-9
    return X, y


def _eq_cond(X: np.ndarray) -> float:
    g = X.T @ X
    d = np.sqrt(np.clip(np.diag(g), 0, None))
    return float(np.linalg.cond(g / np.outer(d, d)) ** 0.5)


def _se_rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300)))


def generate(host: str, device: str) -> Path:
    from pystatistics.regression import Design, fit
    from pystatistics.core.exceptions import NumericalError, PyStatisticsError

    env = env_manifest(device=device, host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    flagged: list[dict[str, Any]] = []
    seed = 0
    for model_key, family, r_family in _MODELS:
        for gap in _SEVERITIES:
            seed += 1
            X, y = _design(family, gap, seed=seed)
            eqc = _eq_cond(X)
            d = Design.from_arrays(X, y)
            cpu = fit(d, family=_fam(family), backend="cpu")
            cse = np.asarray(cpu.standard_errors, float)
            cco = np.asarray(cpu.coefficients, float)
            # R reference SEs (ground-truth instability warning).
            try:
                rrec, _ = run_r_regression_record(X, y, r_family=r_family,
                                                  dataset="synthetic", model_key=model_key)
                rse = np.asarray(rrec["standard_errors"], float)
            except Exception:  # noqa: BLE001
                rse = np.full_like(cse, np.nan)

            row: dict[str, Any] = {
                "model_key": model_key, "family": (family or "gaussian"),
                "eq_cond": eqc,
                "cpu_se_max": float(np.nanmax(cse)),
                "r_se_max": float(np.nanmax(rse)),
            }
            try:
                g = fit(d, family=_fam(family), backend="gpu")
                gse = np.asarray(g.standard_errors, float)
                gco = np.asarray(g.coefficients, float)
                coef_rel = float(np.max(np.abs(gco - cco) / np.maximum(np.abs(cco), 1e-9)))
                se_rel_cpu = _se_rel(gse, cse)
                se_rel_r = _se_rel(gse, rse)
                # Understatement test on the least-identified coefficients.
                understates = bool(np.any(gse < 0.5 * cse))
                verdict = ("inference-warned"
                           if (se_rel_cpu <= 1e-2 and not understates)
                           else "INFERENCE-SILENT-WRONG")
                row.update(gpu_status="accepted", coef_rel_gpu_vs_cpu=coef_rel,
                           se_rel_gpu_vs_cpu=se_rel_cpu, se_rel_gpu_vs_r=se_rel_r,
                           gpu_se_max=float(np.nanmax(gse)),
                           inference_verdict=verdict)
                if verdict != "inference-warned":
                    flagged.append(row)
            except (NumericalError, PyStatisticsError):
                row.update(gpu_status="refused", coef_rel_gpu_vs_cpu=None,
                           se_rel_gpu_vs_cpu=None, se_rel_gpu_vs_r=None,
                           gpu_se_max=None, inference_verdict="refused (fail-loud)")
            rows.append(row)
            records.append({"engine": f"inference_se:{device}", "dataset": model_key,
                            "n": X.shape[0], "p": X.shape[1], **row})
            print(f"  {model_key:12} eq_cond={eqc:.2e} {row['gpu_status']:9} "
                  + (f"coef_rel={row['coef_rel_gpu_vs_cpu']:.2e} "
                     f"se_rel(cpu)={row['se_rel_gpu_vs_cpu']:.2e} "
                     f"SE cpu/gpu/r={row['cpu_se_max']:.3g}/{row['gpu_se_max']:.3g}/{row['r_se_max']:.3g} "
                     f"-> {row['inference_verdict']}"
                     if row['gpu_status'] == "accepted" else ""), flush=True)

    config = {"study": "inference_se_r12", "device": device,
              "models": [m for m, _, _ in _MODELS], "severities": list(_SEVERITIES),
              "reference": "R lm()/glm()"}
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"inference_se_{device}_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    cols = ["model_key", "family", "eq_cond", "gpu_status", "coef_rel_gpu_vs_cpu",
            "se_rel_gpu_vs_cpu", "se_rel_gpu_vs_r", "cpu_se_max", "gpu_se_max",
            "r_se_max", "inference_verdict"]
    with (out_dir / f"{stem}_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {run_path}")
    from collections import Counter
    print("verdict tally:", dict(Counter(r["inference_verdict"] for r in rows)))
    if flagged:
        print("\n" + "!" * 72)
        print(f"!!! {len(flagged)} INFERENCE-SILENT-WRONG fit(s) on {device} — STOP, surface to user.")
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
