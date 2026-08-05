"""GPU tier (Guarantee-3 corollary) — device-agnostic; runs on MPS and CUDA.

montecarlo is the corpus's cleanest embarrassingly-parallel GPU candidate: B
independent resamples reduced to a mean / mean-difference. This tier holds the
GPU path (the 4.6.7 declared-``gpu_statistic`` opt-in) to the full bar:

  1. SPEED — GPU must beat the CPU in the intended large-(n·B) regime (a GPU that
     ties/loses has no reason to exist). Reported across (n, B).
  2. fp32 ACCURACY — MPS runs the reduction in float32 (no gpu_fp64 for montecarlo,
     the honest subset). Isolate fp32 accumulation drift over large n on SHARED
     resample indices: torch-fp32 mean vs numpy-fp64 mean. Must stay within the
     GPU_FP32 tier; a drift that grows past it at large n is a finding.
  3. GPU RNG — reproducible with a seed (same seed -> identical), and the
     permutation kernel (random-key argsort) produces unbiased permutations
     (no resample-correlation artifact).
  4. FAIL-LOUD — the opt-in never silently substitutes (re-confirmed on-device).

Emits artifacts/montecarlo/v<ver>/runs/gpu_<device>.json.

Run (MPS):   DATASETS_ROOT=Dev/datasets python drivers/montecarlo/run_gpu.py
Run (CUDA):  same, on Forge.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcdata  # noqa: E402
from mccompare import expect_raises  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run  # noqa: E402
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from pystatistics.montecarlo import boot, permutation_test  # noqa: E402
from pystatistics.core.exceptions import ValidationError  # noqa: E402

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/montecarlo/v{ver}/runs/gpu_{dev}.json")


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda", torch
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch
    return None, torch


def _best(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


def run_speed(dev) -> list[dict]:
    recs = []
    rng = np.random.default_rng(0)
    mean = mcdata.STAT["mean"]
    # warm up the device (first kernel pays init)
    _ = boot(rng.normal(0, 1, 100), mean, n_resamples=100, seed=0,
             backend="gpu", gpu_statistic="mean")
    for n, B in [(1000, 20000), (5000, 50000), (20000, 100000)]:
        data = rng.normal(5, 2, n)
        tg = _best(lambda: boot(data, mean, n_resamples=B, seed=1,
                                backend="gpu", gpu_statistic="mean"))
        tc = _best(lambda: boot(data, mean, n_resamples=B, seed=1, backend="cpu"))
        recs.append({"group": "speed", "op": "boot_mean", "n": n, "B": B,
                     "gpu_s": tg, "cpu_s": tc, "speedup": float(tc / tg),
                     "pass": bool(tc / tg > 1.0)})
    for n, B in [(500, 50000), (2000, 100000), (5000, 200000)]:
        x = rng.normal(0, 1, n); y = rng.normal(0.2, 1, n)
        tg = _best(lambda: permutation_test(x, y, mcdata.mean_diff, n_resamples=B,
                   seed=1, backend="gpu", gpu_statistic="mean_diff"))
        tc = _best(lambda: permutation_test(x, y, mcdata.mean_diff, n_resamples=B,
                   seed=1, backend="cpu"))
        recs.append({"group": "speed", "op": "perm_meandiff", "n": n, "B": B,
                     "gpu_s": tg, "cpu_s": tc, "speedup": float(tc / tg),
                     "pass": bool(tc / tg > 1.0)})
    return recs


def run_fp32_drift(dev, torch) -> list[dict]:
    """Isolate fp32 accumulation: torch-fp32 mean vs numpy-fp64 mean on the SAME
    bootstrap resample indices, across n. This is exactly the reduction the GPU
    backend performs (float32 on MPS)."""
    from pystatistics.core.compute import tolerances as tol
    recs = []
    device = torch.device(dev)
    use_dtype = torch.float32 if dev == "mps" else torch.float64
    rng = np.random.default_rng(3)
    for n in [100, 1000, 10000, 100000]:
        data = rng.normal(50.0, 10.0, n)   # large magnitude -> stress fp32 sum
        idx = rng.integers(0, n, size=(2000, n))
        # fp64 reference (numpy)
        m64 = data[idx].mean(axis=1)
        # device reduction at the backend's dtype
        dt = torch.from_numpy(data).to(device=device, dtype=use_dtype)
        it = torch.from_numpy(idx).to(device=device)
        md = dt[it].mean(dim=1).cpu().numpy().astype(np.float64)
        rel = float(np.max(np.abs(md - m64) / (np.abs(m64) + 1e-12)))
        recs.append({"group": "fp32_drift", "n": n, "dtype": str(use_dtype),
                     "max_rel_vs_fp64": rel,
                     # MPS fp32 tier ~1e-5; CUDA fp64 exact ~1e-13
                     "tier": 1e-5 if dev == "mps" else 1e-12,
                     "pass": bool(rel < (2e-5 if dev == "mps" else 1e-12))})
    return recs


def run_rng_quality(dev) -> list[dict]:
    """GPU RNG: reproducible + unbiased permutations (no correlation artifact)."""
    recs = []
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1, 500)
    r1 = boot(data, mcdata.STAT["mean"], n_resamples=3000, seed=42,
              backend="gpu", gpu_statistic="mean")
    r2 = boot(data, mcdata.STAT["mean"], n_resamples=3000, seed=42,
              backend="gpu", gpu_statistic="mean")
    recs.append({"group": "rng", "check": "boot_reproducible",
                 "identical": bool(np.array_equal(r1.t, r2.t)),
                 "pass": bool(np.array_equal(r1.t, r2.t))})
    # permutation uniformity: mean of the GPU permutation distribution should be
    # ~0 under the null-symmetric mean-difference (unbiased shuffling)
    x = rng.normal(0, 1, 400); y = rng.normal(0, 1, 400)   # same dist -> null
    p = permutation_test(x, y, mcdata.mean_diff, n_resamples=100000, seed=7,
                         backend="gpu", gpu_statistic="mean_diff")
    perm_mean = float(np.mean(p.perm_stats))
    perm_se = float(np.std(p.perm_stats) / np.sqrt(len(p.perm_stats)))
    recs.append({"group": "rng", "check": "perm_unbiased_null",
                 "perm_dist_mean": perm_mean, "perm_dist_se": perm_se,
                 "note": "permutation distribution centered at 0 under the null "
                         "(argsort-key shuffle is unbiased)",
                 "pass": bool(abs(perm_mean) < 4 * perm_se)})
    return recs


def run_fail_loud(dev) -> list[dict]:
    """The 4.6.7 opt-in never silently substitutes, re-confirmed on-device."""
    data = np.random.default_rng(0).gamma(2, 3, 300)
    checks = {
        "missing_flag_raises": expect_raises(
            lambda: boot(data, mcdata.STAT["mean"], n_resamples=99, seed=1,
                         backend="gpu"), ValidationError),
        "declared_wrong_raises": expect_raises(
            lambda: boot(data, mcdata.STAT["median"], n_resamples=99, seed=1,
                         backend="gpu", gpu_statistic="mean"), ValidationError),
    }
    return [{"group": "fail_loud", **{k: v["pass"] for k, v in checks.items()},
             "pass": all(v["pass"] for v in checks.values())}]


def main() -> None:
    warnings.filterwarnings("ignore")
    dev, torch = _device()
    if dev is None:
        print("no GPU device available — skipping GPU tier")
        return
    env = env_manifest(device=dev)
    require_pypi(env)

    speed = run_speed(dev)
    drift = run_fp32_drift(dev, torch)
    rngq = run_rng_quality(dev)
    fail = run_fail_loud(dev)

    run = build_run(
        env=env,
        config={"suite": f"montecarlo-gpu-{dev}",
                "reference": "CPU fp64 (same estimator); GPU-vs-CPU speed",
                "tolerance_contract": "GPU beats CPU at large n*B; fp32 reduction "
                "within the GPU_FP32 tier (MPS) / exact (CUDA fp64); RNG "
                "reproducible + unbiased; opt-in never silently substitutes. "
                "montecarlo exposes no gpu_fp64 (honest subset)."},
        records=[{"key": "speed", "checks": speed},
                 {"key": "fp32_drift", "checks": drift},
                 {"key": "rng_quality", "checks": rngq},
                 {"key": "fail_loud", "checks": fail}],
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"], dev=dev))
    write_run(out, run)
    print(f"wrote {out}")
    for r in speed:
        print(f"  speed {r['op']:14s} n={r['n']:6d} B={r['B']:7d}: "
              f"{r['speedup']:.2f}x  ({'WIN' if r['pass'] else 'LOSS'})")
    for r in drift:
        print(f"  fp32_drift n={r['n']:6d}: max_rel={r['max_rel_vs_fp64']:.2e} "
              f"tier={r['tier']:.0e}  {'ok' if r['pass'] else 'EXCEEDS'}")
    print(f"  rng: {[r['pass'] for r in rngq]}  fail_loud: {fail[0]['pass']}")


if __name__ == "__main__":
    main()
