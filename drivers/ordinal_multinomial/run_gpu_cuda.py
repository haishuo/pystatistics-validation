"""GPU (CUDA) validation for polr + multinom — run ON Forge, emits JSON to stdout.

Self-contained (no driver imports, so it runs against a bare PyPI install on a CUDA
host). Redirect stdout to artifacts/ordinal_multinomial/v<ver>/runs/gpu_cuda.json.
Three studies:

  cf1        — the CF-1 fix verification: across a cond(X) sweep, the fp32 GPU path's
               standard errors vs the CPU fp64 result (SE_relerr) and a NEGATIVE-
               variance flag; plus the gpu_fp64-vs-cpu pivot (R11, isolates precision
               from hardware). The fix computes the vcov Hessian in fp64, so there
               must be NO negative variances and SE_relerr must stay at the fp32
               tier (<~2e-2) — where 4.6.8 returned negative variances / 100% error.
  twotier    — GPU_FP32 two-tier correctness on a well-conditioned design: fp32 GPU
               coefficients / SEs / fitted probs vs CPU fp64 (rtol 1e-4, atol 1e-5).
  perf       — CPU vs gpu_fp32 vs gpu_fp64 fit wall time across n (the GPU corollary
               to Guarantee 3: the GPU must beat the CPU in its large-n regime).

Run:  PYTHONPATH=~/scratch_om/lib python run_gpu_cuda.py [cuda|mps] > gpu_cuda.json
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np


# ---- self-contained designs -------------------------------------------------
def _conditioned_X(n, p, cond, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, p))
    U, _, Vt = np.linalg.svd(A, full_matrices=False)
    X = (U * np.geomspace(1.0, cond, p)) @ Vt
    X = (X - X.mean(0)) / X.std(0)
    U2, _, Vt2 = np.linalg.svd(X, full_matrices=False)
    return (U2 * np.geomspace(1.0, cond, p)) @ Vt2


def _ord_design(n, p, cond, seed):
    X = _conditioned_X(n, p, cond, seed)
    eta = X @ np.linspace(0.8, -0.8, p)
    eta = eta / (eta.std() + 1e-12)
    u = np.random.default_rng(seed + 1).logistic(size=n)
    lat = eta + u
    y = np.searchsorted(np.quantile(lat, [0.25, 0.5, 0.75]), lat)
    return y.astype(int), X, 4


def _mn_design(n, p, cond, seed, K=3):
    Xf = _conditioned_X(n, p, cond, seed)
    X = np.column_stack([np.ones(n), Xf])
    B = np.linspace(0.8, -0.8, (K - 1) * p).reshape(K - 1, p)
    cols = [Xf @ B[j] for j in range(K - 1)]
    cols = [c / (c.std() + 1e-12) for c in cols]
    eta = np.column_stack(cols + [np.zeros(n)])
    P = np.exp(eta - eta.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    u = np.random.default_rng(seed + 1).uniform(size=n)
    y = (u[:, None] > np.cumsum(P, 1)).sum(1).astype(int)
    return y, X, K


def _relerr(a, b):
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if a.shape != b.shape:
        return float("nan")
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-12))


def _bt(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best * 1e3


# ---- studies ----------------------------------------------------------------
def cf1(has_fp64):
    from pystatistics.ordinal import polr
    from pystatistics.multinomial import multinom
    from pystatistics.core.exceptions import ConvergenceError, NotPositiveDefiniteError
    rows = {"polr": [], "multinom": []}
    for cond in [1e1, 1e2, 1e3, 1e4]:
        seed = int(round(np.log10(cond)))
        # polr
        y, X, K = _ord_design(4000, 6, cond, seed)
        rec = {"cond": cond, "xtx_cond": float(np.linalg.cond(X.T @ X))}
        try:
            cpu = polr(y, X, backend="cpu")
            g = polr(y, X, backend="gpu")
            rec["fp32_se_relerr"] = _relerr(g.standard_errors, cpu.standard_errors)
            rec["fp32_neg_var"] = bool(np.any(np.asarray(g.standard_errors) <= 0)
                                       or np.any(~np.isfinite(g.standard_errors)))
            rec["fp32_converged"] = bool(g.converged)
            if has_fp64:
                g64 = polr(y, X, backend="gpu_fp64")
                rec["fp64_se_relerr"] = _relerr(g64.standard_errors, cpu.standard_errors)
        except (ConvergenceError, NotPositiveDefiniteError) as e:
            rec["error"] = f"{type(e).__name__}"
        rows["polr"].append(rec)
        # multinom
        y, X, K = _mn_design(4000, 6, cond, seed)
        rec = {"cond": cond, "xtx_cond": float(np.linalg.cond(X.T @ X))}
        try:
            cpu = multinom(y, X, backend="cpu", max_iter=2000)
            g = multinom(y, X, backend="gpu", max_iter=2000)
            rec["fp32_se_relerr"] = _relerr(g.standard_errors, cpu.standard_errors)
            rec["fp32_neg_var"] = bool(np.any(np.asarray(g.standard_errors) <= 0)
                                       or np.any(~np.isfinite(g.standard_errors)))
            rec["fp32_converged"] = bool(g.converged)
            if has_fp64:
                g64 = multinom(y, X, backend="gpu_fp64", max_iter=2000)
                rec["fp64_se_relerr"] = _relerr(g64.standard_errors, cpu.standard_errors)
        except (ConvergenceError, NotPositiveDefiniteError) as e:
            rec["error"] = f"{type(e).__name__}"
        rows["multinom"].append(rec)
    return rows


def twotier():
    """GPU_FP32 two-tier: fp32 GPU vs CPU fp64 on a well-conditioned design."""
    from pystatistics.ordinal import polr
    from pystatistics.multinomial import multinom
    out = []
    yo, Xo, _ = _ord_design(5000, 4, 5.0, 1)
    cpu, g = polr(yo, Xo, backend="cpu"), polr(yo, Xo, backend="gpu")
    out.append({"surface": "polr", "n": 5000,
                "coef_relerr": _relerr(g.coefficients, cpu.coefficients),
                "se_relerr": _relerr(g.standard_errors, cpu.standard_errors)})
    ym, Xm, _ = _mn_design(5000, 4, 5.0, 1)
    cpu, g = multinom(ym, Xm, backend="cpu", max_iter=2000), multinom(ym, Xm, backend="gpu", max_iter=2000)
    out.append({"surface": "multinom", "n": 5000,
                "coef_relerr": _relerr(g.coefficient_matrix, cpu.coefficient_matrix),
                "se_relerr": _relerr(g.standard_errors, cpu.standard_errors),
                "probs_relerr": _relerr(g.fitted_probs, cpu.fitted_probs)})
    return out


def perf(has_fp64):
    from pystatistics.ordinal import polr
    from pystatistics.multinomial import multinom
    out = {"polr": [], "multinom": []}
    for n in [2000, 20000, 100000]:
        yo, Xo, _ = _ord_design(n, 4, 5.0, 1)
        row = {"n": n, "cpu_ms": _bt(lambda: polr(yo, Xo, backend="cpu")),
               "gpu_fp32_ms": _bt(lambda: polr(yo, Xo, backend="gpu"))}
        if has_fp64:
            row["gpu_fp64_ms"] = _bt(lambda: polr(yo, Xo, backend="gpu_fp64"))
        row["speedup_fp32"] = row["cpu_ms"] / row["gpu_fp32_ms"]
        out["polr"].append(row)
        ym, Xm, _ = _mn_design(n, 4, 5.0, 1)
        row = {"n": n, "cpu_ms": _bt(lambda: multinom(ym, Xm, backend="cpu", max_iter=2000)),
               "gpu_fp32_ms": _bt(lambda: multinom(ym, Xm, backend="gpu", max_iter=2000))}
        if has_fp64:
            row["gpu_fp64_ms"] = _bt(lambda: multinom(ym, Xm, backend="gpu_fp64", max_iter=2000))
        row["speedup_fp32"] = row["cpu_ms"] / row["gpu_fp32_ms"]
        out["multinom"].append(row)
    return out


def main():
    import torch
    dev = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available()
                                                  else "mps")
    has_fp64 = (dev == "cuda")
    import pystatistics
    result = {"device": dev, "has_fp64": has_fp64,
              "pystatistics_version": pystatistics.__version__,
              "torch_version": torch.__version__,
              "gpu_name": (torch.cuda.get_device_name(0) if dev == "cuda" else dev),
              "cf1": cf1(has_fp64), "twotier": twotier(), "perf": perf(has_fp64)}
    print(json.dumps(result, indent=2, default=lambda o: (
        float(o) if isinstance(o, np.floating) else int(o)
        if isinstance(o, np.integer) else bool(o) if isinstance(o, np.bool_)
        else str(o))))


if __name__ == "__main__":
    main()
