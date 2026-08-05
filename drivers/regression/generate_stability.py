"""Generate the GPU-GLM stability study: where plain GLM fails, ridge / fp64 hold.

One job: for log-link GLM families at large (n, p) — the regime where a plain
float32 GPU fit is ill-conditioned — record, on one device, the outcome of three
strategies so the report can tell the optimized story honestly:

- ``plain``    : ``fit(family=..., backend='gpu')`` — may FAIL LOUD (NumericalError)
                 in float32 at large n (recorded as ``status='raised'``).
- ``ridge``    : ``ridge(..., backend='gpu')`` — the penalty conditions XᵀWX, so the
                 float32 fit converges; agreement is measured against the CPU ridge.
- ``fp64``     : ``fit(..., backend='gpu_fp64')`` — double precision (CUDA only;
                 ``status='unsupported'`` on MPS); agreement vs the CPU fit.

This is the evidence behind "ridge is the fast, stable GPU GLM at enormous scale,
and gpu_fp64 is the exact (CUDA) correctness path".

    python -m drivers.regression.generate_stability --host powerhouse --device mps
    python -m drivers.regression.generate_stability --host forge --device cuda
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

_REPO = Path(__file__).resolve().parent.parent.parent
_GRID = ((100_000, 64), (200_000, 128), (500_000, 128))
_FAMILIES = (("glm_poisson", "poisson"), ("glm_gamma", "gamma"))
_LAM = 100.0


def _synth(n, p, family, *, seed):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.standard_normal((n, p))])
    eta = X @ (rng.standard_normal(p + 1) * 0.4 / np.sqrt(p))
    if family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    else:
        y = rng.gamma(shape=2.0, scale=np.exp(eta) / 2.0) + 1e-3
    return X, y


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def generate(host: str, device: str) -> Path:
    from pystatistics.regression import Design, fit, ridge, Gamma
    from pystatistics.core.exceptions import NumericalError

    env = env_manifest(device=device, host=host)
    require_pypi(env)
    backend = "gpu"

    def _fam(f):
        return Gamma(link="log") if f == "gamma" else f

    rows: list[dict[str, Any]] = []
    seed = 0
    for model_key, family in _FAMILIES:
        for n, p in _GRID:
            seed += 1
            X, y = _synth(n, p, family, seed=seed)
            d = Design.from_arrays(X, y)
            cpu = fit(d, family=_fam(family), backend="cpu")
            cpu_ridge = ridge(X, y, l2=_LAM, family=_fam(family), backend="cpu")
            row: dict[str, Any] = {"model_key": model_key, "n": n, "p": p, "lam": _LAM}

            # plain GPU GLM — may fail loud
            try:
                g = fit(d, family=_fam(family), backend=backend)
                row["plain_status"] = "converged"
                row["plain_rel_vs_cpu"] = _rel(g.coefficients, cpu.coefficients)
            except NumericalError:
                row["plain_status"] = "raised"
                row["plain_rel_vs_cpu"] = None

            # ridge GPU GLM — should converge (penalty conditions the solve)
            try:
                rg = ridge(X, y, l2=_LAM, family=_fam(family), backend=backend)
                row["ridge_status"] = "converged" if rg.converged else "not_converged"
                row["ridge_rel_vs_cpu_ridge"] = _rel(rg.coefficients, cpu_ridge.coefficients)
            except NumericalError:
                row["ridge_status"] = "raised"
                row["ridge_rel_vs_cpu_ridge"] = None

            # gpu_fp64 — CUDA only
            try:
                g64 = fit(d, family=_fam(family), backend="gpu_fp64")
                row["fp64_status"] = "converged"
                row["fp64_rel_vs_cpu"] = _rel(g64.coefficients, cpu.coefficients)
            except RuntimeError:
                row["fp64_status"] = "unsupported"
                row["fp64_rel_vs_cpu"] = None

            rows.append(row)
            print(f"  {model_key:12} n={n:>7} p={p:>3}: plain={row['plain_status']:9} "
                  f"ridge={row['ridge_status']:9}({row['ridge_rel_vs_cpu_ridge']}) "
                  f"fp64={row['fp64_status']}", flush=True)

    records = [{"engine": f"pystatistics:{backend}", "dataset": r["model_key"],
                "n": r["n"], "p": r["p"], **r} for r in rows]
    run = build_run(env=env, config={"study": "gpu_glm_stability", "lam": _LAM,
                                      "grid": [list(g) for g in _GRID]},
                    records=records)
    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    stem = f"stability_{device}_{host}"
    path = write_run(out_dir / f"{stem}.json", run)
    cols = ["model_key", "n", "p", "lam", "plain_status", "plain_rel_vs_cpu",
            "ridge_status", "ridge_rel_vs_cpu_ridge", "fp64_status", "fp64_rel_vs_cpu"]
    with (out_dir / f"{stem}_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--device", default="mps", choices=["mps", "cuda"])
    args = ap.parse_args()
    generate(args.host, args.device)


if __name__ == "__main__":
    main()
