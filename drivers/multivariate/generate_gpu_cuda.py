"""Self-contained CUDA PCA study for Forge (priority 4 + CF-1 clearance).

Runs on Forge's NVIDIA GPU only. DELIBERATELY self-contained: imports only
``pystatistics`` + ``numpy`` + ``torch`` (no ``pystatsval`` / sibling drivers), so
it can be scp'd to a throwaway Forge venv that inherits gpumice's CUDA torch and
overlays ``pystatistics==4.4.0`` from PyPI. Emits one JSON artifact (env + the
three studies) to the path given as argv[1]; the JSON is pulled back to the Mac
and frozen under ``artifacts/multivariate/v4.4.0/runs/`` for rendering.

Three studies:

  1. **CF-1 gate boundary (R12/R13)** -- the centerpiece. The ``solver='gram'``
     fp32 path forms X'X in single precision (the exact CF-1 defect class). Sweep
     designs whose cond(X) straddles the fp32 Gram gate (~1e3). For every design:
     does the gate ACCEPT or REFUSE? Every ACCEPTED fp32 fit must be CORRECT vs
     the fp64 CPU reference (subspace angle < 1 deg, sdev to the fp32 tier) -- if
     any accepted fit is wrong, that is a silent-wrong band (R16). Every REFUSED
     design must fail loud (NumericalError) with force=True the documented
     override. Repeated for the randomized path's gate (re-proven on CUDA, R13).

  2. **R11 precision/hardware isolation** -- gpu_fp64 vs cpu_fp64 (same precision)
     isolates the hardware effect; gpu (fp32) gives the bundled number. gpu_fp64
     must be numerically exact vs CPU (~1e-13). On a consumer part fp64 is ~1/64
     fp32, so gpu_fp64 may LOSE to CPU on SVD -- reported honestly.

  3. **CUDA randomized value + correctness** -- the randomized path on CUDA:
     subspace vs CPU fp64, and speedup vs CPU.

    python generate_gpu_cuda.py <out_json> [--host forge]
"""

from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np


def _now_sync(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _time_call(fn, torch, repeats=5, warmup=1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = _now_sync(torch)
        fn()
        ts.append(_now_sync(torch) - t0)
    return float(np.median(ts))


def _spectrum_design(n, p, spectrum, seed):
    """X = (U * s) @ V.T with orthonormal U,V and the given singular values."""
    rng = np.random.default_rng(seed)
    r = len(spectrum)
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V, _ = np.linalg.qr(rng.standard_normal((p, r)))
    return ((U * np.asarray(spectrum)) @ V.T).astype(np.float64)


def _subspace_angle_deg(Va, Vb):
    Qa, _ = np.linalg.qr(np.asarray(Va, float))
    Qb, _ = np.linalg.qr(np.asarray(Vb, float))
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(sv.min(), -1.0, 1.0))))


# ── Study 1: CF-1 gate boundary ───────────────────────────────────────────────

def study_cf1_gate(pca, NumericalError):
    """For each method gate (gram, randomized), sweep cond and classify."""
    n, p, k = 4000, 60, 8
    # cond(X) straddling the fp32 gates: gram refuses cond(X) > ~1e3; randomized
    # gates on the retained-subspace conditioning. We make the TOP-k span the
    # condition number (the components PCA actually returns), then a small tail.
    conds = [10, 100, 300, 700, 1000, 1500, 3000, 1e4, 1e5]
    results = {"gram": [], "randomized": []}
    for method in ("gram", "randomized"):
        for cond in conds:
            # top-k spectrum spans [1, 1/cond]; tail an order below the smallest.
            top = np.geomspace(1.0, 1.0 / cond, k)
            tail = np.full(p - k, top[-1] / 10.0)
            X = _spectrum_design(n, p, np.concatenate([top, tail]), seed=int(cond) + 11)
            cpu = pca(X, backend="cpu", n_components=k)  # fp64 reference
            row = {"method": method, "cond_X": float(cond)}
            try:
                sol = pca(X, backend="gpu", solver=method, n_components=k, seed=0)
                ang = _subspace_angle_deg(cpu.rotation, sol.rotation)
                sdev_rel = float(np.max(np.abs(np.asarray(sol.sdev) - np.asarray(cpu.sdev))
                                        / np.maximum(np.abs(np.asarray(cpu.sdev)), 1e-300)))
                row.update(outcome="accepted", subspace_angle_deg=ang,
                           sdev_max_rel=sdev_rel,
                           accepted_correct=bool(ang < 1.0 and sdev_rel < 1e-2))
            except NumericalError as exc:
                # refused: confirm force=True bypasses (and is then unreliable).
                forced_ok = False
                try:
                    pca(X, backend="gpu", solver=method, n_components=k, force=True, seed=0)
                    forced_ok = True
                except Exception:  # noqa: BLE001
                    forced_ok = False
                row.update(outcome="refused", error=type(exc).__name__,
                           force_bypasses=forced_ok)
            except Exception as exc:  # noqa: BLE001
                row.update(outcome="error", error=f"{type(exc).__name__}: {exc}")
            results[method].append(row)
    # silent-wrong check: any accepted-but-wrong fit anywhere?
    silent_wrong = [r for rows in results.values() for r in rows
                    if r.get("outcome") == "accepted" and not r.get("accepted_correct", True)]
    return {"sweep": results, "n": n, "p": p, "k": k,
            "silent_wrong_count": len(silent_wrong),
            "silent_wrong_rows": silent_wrong}


# ── Study 2: R11 precision/hardware isolation ─────────────────────────────────

def study_r11_isolation(pca, torch):
    shapes = [(5000, 100, 10), (50000, 100, 10), (200000, 200, 10)]
    rows = []
    for n, p, k in shapes:
        # well-conditioned gapped design (top-k separated, then floor)
        s = np.full(min(n, p), 0.5); s[:k] = np.linspace(10.0, 6.0, k)
        X = _spectrum_design(n, p, s, seed=n + p)
        cpu = pca(X, backend="cpu", n_components=k)
        t_cpu = _time_call(lambda: pca(X, backend="cpu", n_components=k), torch)
        # gpu_fp64 (same precision as CPU -> isolates hardware) using svd solver
        gpu64 = pca(X, backend="gpu_fp64", solver="svd", n_components=k)
        t_g64 = _time_call(lambda: pca(X, backend="gpu_fp64", solver="svd",
                                       n_components=k), torch)
        # gpu fp32 (bundled precision+hardware) randomized -- the speed path
        t_g32 = _time_call(lambda: pca(X, backend="gpu", solver="randomized",
                                       n_components=k, seed=0), torch)
        exact_rel = float(np.max(np.abs(np.asarray(gpu64.sdev) - np.asarray(cpu.sdev))
                                 / np.maximum(np.abs(np.asarray(cpu.sdev)), 1e-300)))
        rows.append({
            "n": n, "p": p, "k": k,
            "wall_cpu_fp64_s": t_cpu, "wall_gpu_fp64_s": t_g64, "wall_gpu_fp32_s": t_g32,
            "hardware_only_speedup_fp64": t_cpu / t_g64 if t_g64 else None,
            "bundled_speedup_fp32": t_cpu / t_g32 if t_g32 else None,
            "gpu_fp64_vs_cpu_sdev_rel": exact_rel,
        })
    return {"shapes": rows}


# ── Study 3: CUDA randomized value + correctness ──────────────────────────────

def study_randomized(pca, torch):
    shapes = [(5000, 100, 10), (100000, 100, 10), (500000, 200, 10), (2000, 5000, 10)]
    rows = []
    for n, p, k in shapes:
        s = np.full(min(n, p), 0.5); s[:k] = np.linspace(10.0, 6.0, k)
        X = _spectrum_design(n, p, s, seed=n + 2 * p)
        cpu = pca(X, backend="cpu", n_components=k)
        mps = pca(X, backend="gpu", solver="randomized", n_components=k, seed=0)
        ang = _subspace_angle_deg(cpu.rotation, mps.rotation)
        sdev_rel = float(np.max(np.abs(np.asarray(mps.sdev) - np.asarray(cpu.sdev))
                                / np.maximum(np.abs(np.asarray(cpu.sdev)), 1e-300)))
        t_cpu = _time_call(lambda: pca(X, backend="cpu", n_components=k), torch, repeats=3)
        t_gpu = _time_call(lambda: pca(X, backend="gpu", solver="randomized",
                                       n_components=k, seed=0), torch, repeats=3)
        rows.append({
            "n": n, "p": p, "k": k,
            "subspace_max_angle_deg": ang, "sdev_max_rel": sdev_rel,
            "wall_cpu_s": t_cpu, "wall_gpu_s": t_gpu,
            "gpu_speedup_vs_cpu": t_cpu / t_gpu if t_gpu else None,
        })
    return {"shapes": rows}


def main():
    out_json = sys.argv[1]
    host = sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--host" else "forge"
    import torch
    import pystatistics
    from pystatistics.multivariate import pca
    from pystatistics.core.exceptions import NumericalError

    env = {
        "host": host,
        "pystatistics_version": pystatistics.__version__,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if not env["cuda_available"]:
        raise SystemExit("CUDA not available on this host -- this study is CUDA-only.")
    print(f"env: {env['device_name']} torch {env['torch_version']} "
          f"cu{env['cuda_version']} pystatistics {env['pystatistics_version']}")

    print("study 1: CF-1 gate boundary ...")
    cf1 = study_cf1_gate(pca, NumericalError)
    print(f"  silent_wrong_count = {cf1['silent_wrong_count']}")
    for m, rows in cf1["sweep"].items():
        for r in rows:
            print(f"   {m:10s} cond={r['cond_X']:>8.0f} -> {r['outcome']:8s} "
                  + (f"ang={r.get('subspace_angle_deg', float('nan')):.4f}deg "
                     f"sdev_rel={r.get('sdev_max_rel', float('nan')):.2e} "
                     f"ok={r.get('accepted_correct')}"
                     if r["outcome"] == "accepted"
                     else f"force_bypasses={r.get('force_bypasses')}"))
    print("study 2: R11 isolation ...")
    r11 = study_r11_isolation(pca, torch)
    for r in r11["shapes"]:
        print(f"   {r['n']}x{r['p']}: gpu_fp64 hw-only={r['hardware_only_speedup_fp64']:.2f}x "
              f"fp32 bundled={r['bundled_speedup_fp32']:.2f}x  "
              f"gpu_fp64_exact_rel={r['gpu_fp64_vs_cpu_sdev_rel']:.2e}")
    print("study 3: CUDA randomized ...")
    rnd = study_randomized(pca, torch)
    for r in rnd["shapes"]:
        print(f"   {r['n']}x{r['p']}: subspace={r['subspace_max_angle_deg']:.4f}deg "
              f"speedup={r['gpu_speedup_vs_cpu']:.2f}x")

    out = {"schema": "multivariate-cuda-study/v1", "env": env,
           "cf1_gate": cf1, "r11_isolation": r11, "randomized": rnd}
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
