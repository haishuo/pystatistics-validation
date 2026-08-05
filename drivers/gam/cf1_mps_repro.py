"""CF-1 reproduction on Apple MPS for the gam fp32 GPU path (R13).

The GPU-feasibility investigation (handoffs/gam-gpu-investigation.md, study B)
proved on CUDA that forming ``X'WX`` in fp32 corrupts the EDF trace in the
small-lambda regime the smoothing-parameter optimizer probes by design. R13
forbids carrying a CUDA result to another device as fact, so we re-prove the
band on the hardware this validation session actually has: Apple MPS.

Two layers of evidence:

1. **Kernel layer** (mirrors investigation study B): fp32 ``X'WX`` on MPS,
   promoted to host fp64, EDF trace vs the fp64 reference on a near-rank-
   deficient basis across a lambda sweep. Shows the gram-in-fp32 corruption
   in isolation.
2. **User-facing layer**: the public ``gam(..., backend='gpu')`` vs
   ``backend='cpu'`` on a hard high-k fit, comparing selected lambda / EDF /
   fitted values. Shows the corruption reaches a user who opts into the fp32
   GPU backend, silently (no gate, no warning).

Writes artifacts/gam/v4.5.7/runs/cf1_mps.json. Deterministic (seeded).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from pystatsval.device import env_manifest, require_pypi

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_root import artifact_root  # noqa: E402

ARTIFACT = artifact_root(Path(__file__).resolve().parents[2]) / "gam/v4.5.7/runs/cf1_mps.json"


# ---- kernel-layer helpers (from investigation study B) ------------------

def near_rank_deficient(n: int, p: int = 40, decay: float = 0.55, seed: int = 1):
    """Geometric singular-value decay — the spectrum a high-k spline basis
    on clustered x actually has (many near-null directions)."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    Vt, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return (U * (decay ** np.arange(p))) @ Vt


def second_diff_penalty(p: int):
    D = np.zeros((p - 2, p))
    for i in range(p - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D.T @ D  # rank p-2, null space dim 2 (const + linear)


def edf_from_gram(XtWX, S, lam):
    A = XtWX + lam * S
    return float(np.trace(np.linalg.solve(A, XtWX)))


def kernel_band(dev: str) -> list[dict]:
    p = 40
    S = second_diff_penalty(p)
    rows: list[dict] = []
    for n in (5000, 100000):
        X = near_rank_deficient(n, p)
        XtWX64 = X.T @ X
        Xt32 = torch.as_tensor(X, dtype=torch.float32, device=dev)
        XtWX32 = (Xt32.T @ Xt32).cpu().double().numpy()
        for lam in (1e-2, 1e-4, 1e-6, 1e-8):
            A64 = XtWX64 + lam * S
            cond = float(np.linalg.cond(A64))
            e64 = edf_from_gram(XtWX64, S, lam)
            e32 = edf_from_gram(XtWX32, S, lam)
            rows.append({
                "n": n, "lam": lam, "cond_A": cond,
                "edf_fp64": e64, "edf_fp32_mps": e32,
                "abs_err": abs(e32 - e64),
            })
    return rows


# ---- user-facing layer: public gam() backend='gpu' vs 'cpu' -------------

def user_facing() -> dict:
    """A hard fit whose lambda search visits small lambda: high k, wiggly
    signal, moderate noise. Compare the two public backends head to head."""
    from pystatistics.gam import gam, s

    # A genuinely wiggly, low-noise signal so GCV wants a small lambda /
    # high EDF fit — i.e. the fit lands *inside* the corrupted small-lambda
    # band, not away from it. (seed=7 chosen deterministically; a near-flat
    # over-smoothed signal would sit away from the band and hide the defect.)
    rng = np.random.default_rng(7)
    n, k, freq, noise = 1000, 40, 8, 0.03
    x = np.sort(rng.uniform(0, 1, n))
    f = np.sin(2 * np.pi * freq * x)
    y = f + rng.normal(0, noise, n)

    out = {"n": n, "k": k, "bs": "cr", "family": "gaussian",
           "method": "GCV", "seed": 7, "signal": "sin(2*pi*8*x)+N(0,0.03)"}
    for backend in ("cpu", "gpu"):
        try:
            sol = gam(y, smooths=[s("x", k=k, bs="cr")],
                      smooth_data={"x": x}, family="gaussian",
                      method="GCV", backend=backend)
            out[backend] = {
                "backend_name": sol.params.backend_name
                if hasattr(sol.params, "backend_name") else None,
                "total_edf": float(sol.total_edf),
                "edf": [float(v) for v in np.atleast_1d(sol.edf)],
                "gcv": float(sol.gcv),
                "deviance": float(sol.deviance),
                "scale": float(sol.scale),
                "converged": bool(sol.converged),
                "fitted_head": [float(v) for v in sol.fitted_values[:5]],
                "_fitted_full": np.asarray(sol.fitted_values, dtype=float),
            }
        except Exception as e:  # noqa: BLE001 — record whatever the path does
            out[backend] = {"error": f"{type(e).__name__}: {e}"}
    # divergence summary (over the full fitted vector, not just the head)
    if "total_edf" in out.get("cpu", {}) and "total_edf" in out.get("gpu", {}):
        cpu_fit = out["cpu"].pop("_fitted_full")
        gpu_fit = out["gpu"].pop("_fitted_full")
        out["divergence"] = {
            "d_total_edf": out["gpu"]["total_edf"] - out["cpu"]["total_edf"],
            "d_gcv_rel": abs(out["gpu"]["gcv"] - out["cpu"]["gcv"])
            / max(abs(out["cpu"]["gcv"]), 1e-30),
            "fitted_max_abs": float(np.max(np.abs(gpu_fit - cpu_fit))),
            "fitted_rmse": float(np.sqrt(np.mean((gpu_fit - cpu_fit) ** 2))),
        }
    else:
        for b in ("cpu", "gpu"):
            out.get(b, {}).pop("_fitted_full", None)
    return out


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS not available — this repro is MPS-specific (R13).")
    env = env_manifest(device="mps")
    require_pypi(env)

    result = {
        "study": "CF-1 fp32 gram -> EDF band + user-facing gam(backend='gpu')",
        "env": env,
        "kernel_band_mps": kernel_band("mps"),
        "user_facing": user_facing(),
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2))
    print(f"wrote {ARTIFACT}")

    print("\n== kernel layer: fp32 X'WX -> EDF on MPS ==")
    print(f"{'n':>8} {'lam':>8} {'cond(A)':>11} {'EDF64':>9} {'EDF32mps':>10} {'err':>9}")
    for r in result["kernel_band_mps"]:
        print(f"{r['n']:>8} {r['lam']:>8.0e} {r['cond_A']:>11.2e} "
              f"{r['edf_fp64']:>9.4f} {r['edf_fp32_mps']:>10.4f} {r['abs_err']:>9.3f}")

    print("\n== user-facing: public gam() backend='gpu' vs 'cpu' ==")
    print(json.dumps(result["user_facing"], indent=2))


if __name__ == "__main__":
    main()
