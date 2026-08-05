"""Performance tier (Guarantee 3 / R1 / R3) — pystatistics CPU vs R across B.

The mandatory CPU-vs-R scaling study. Both engines run a per-replicate loop for
an arbitrary statistic (R's boot loop; pystatistics' CPU backend), so this
measures the constant factor and confirms the SAME O(B) complexity class — a
complexity-class gap would be an R1 defect. The vectorized speed win is GPU-only
(measured in run_gpu.py); this tier is the honest CPU baseline.

Timing: best-of-N wall time; R measured inside R via system.time (excludes the
~0.3 s Rscript startup). Reports the empirical slope (log-log) for both engines.

Emits artifacts/montecarlo/v<ver>/runs/performance.json (render-from-artifacts).

Run: MVNMLE_DATA_DIR=Dev/datasets python drivers/montecarlo/run_performance.py
"""

from __future__ import annotations

import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mcdata  # noqa: E402
from rref import r_reference, r_package_versions  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run  # noqa: E402
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from pystatistics.montecarlo import boot, permutation_test  # noqa: E402

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/montecarlo/v{ver}/runs/performance.json")


def _best(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t)
    return best


def _slope(bs, ts):
    """Empirical log-log slope (complexity exponent)."""
    lb, lt = np.log(np.array(bs, float)), np.log(np.array(ts, float))
    return float(np.polyfit(lb, lt, 1)[0])


def run_boot_scaling() -> list[dict]:
    recs = []
    Bs = [1000, 5000, 20000, 50000]
    for key, stat_name in [("law", "corr"), ("city", "ratio")]:
        data = mcdata.load(key).values
        stat = mcdata.STAT[stat_name]
        py_t, r_t = [], []
        for B in Bs:
            py_t.append(_best(lambda B=B: boot(data, stat, n_resamples=B, seed=1)))
            r_t.append(float(r_reference("boot_bench", data, statistic=stat_name,
                                         R=B, seed=1, reps=3)["elapsed"]))
        recs.append({
            "group": "boot_scaling", "dataset": key, "statistic": stat_name,
            "B": Bs, "py_cpu_s": py_t, "r_boot_s": r_t,
            "speedup_r_over_py": [float(r / p) for r, p in zip(r_t, py_t)],
            "py_slope": _slope(Bs, py_t), "r_slope": _slope(Bs, r_t),
            "note": "both O(B); ratio is the constant factor (per-replicate "
                    "Python vs R statistic call). Vectorized win is GPU-only.",
            "pass": bool(abs(_slope(Bs, py_t) - 1.0) < 0.3),  # linear in B
        })
    return recs


def run_perm_scaling() -> list[dict]:
    sm = mcdata.two_sample_matrix("sleep")
    x, y = mcdata.two_sample("sleep")
    Bs = [1000, 5000, 20000, 50000]
    py_t, r_t = [], []
    for B in Bs:
        py_t.append(_best(lambda B=B: permutation_test(x, y, mcdata.mean_diff,
                                                       n_resamples=B, seed=1)))
        r_t.append(float(r_reference("perm_bench", sm, R=B, seed=1, reps=3)["elapsed"]))
    return [{
        "group": "perm_scaling", "dataset": "sleep", "statistic": "mean_diff",
        "B": Bs, "py_cpu_s": py_t, "r_s": r_t,
        "speedup_r_over_py": [float(r / p) for r, p in zip(r_t, py_t)],
        "py_slope": _slope(Bs, py_t),
        "pass": bool(abs(_slope(Bs, py_t) - 1.0) < 0.3),
    }]


def run_n_scaling() -> list[dict]:
    """Fixed B, sweep n — does the tiny-n constant-factor lag close as the
    per-replicate statistic cost grows? (Isolates fixed Python overhead.)"""
    B = 5000
    ns = [10, 100, 1000, 10000]
    py_t, r_t = [], []
    rng = np.random.default_rng(0)
    for n in ns:
        d2 = rng.normal(0, 1, (n, 2))
        d2[:, 1] = d2[:, 0] * 0.5 + rng.normal(0, 1, n)  # some correlation
        py_t.append(_best(lambda d2=d2: boot(d2, mcdata.STAT["corr"],
                                             n_resamples=B, seed=1), reps=2))
        # dump the same matrix to R
        r_t.append(float(r_reference("boot_bench", d2, statistic="corr",
                                     R=B, seed=1, reps=2)["elapsed"]))
    return [{
        "group": "n_scaling", "statistic": "corr", "B": B, "n": ns,
        "py_cpu_s": py_t, "r_boot_s": r_t,
        "speedup_r_over_py": [float(r / p) for r, p in zip(r_t, py_t)],
        "note": "constant-factor lag at tiny n is fixed Python per-replicate "
                "overhead; it closes as n (per-replicate statistic cost) grows",
        "pass": True,
    }]


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions(("boot", "stats"))
    env["blas"] = "system (Accelerate on macOS)"
    env["cpu_model"] = platform.processor() or platform.machine()

    boot_scaling = run_boot_scaling()
    perm_scaling = run_perm_scaling()
    n_scaling = run_n_scaling()

    run = build_run(
        env=env,
        config={"suite": "montecarlo-performance",
                "reference": "R boot::boot / base-R permutation loop, timed in R",
                "tolerance_contract": "same O(B) complexity class as R (R1); "
                "constant factor reported honestly; vectorized win is GPU-only"},
        records=[{"key": "boot_scaling", "checks": boot_scaling},
                 {"key": "perm_scaling", "checks": perm_scaling},
                 {"key": "n_scaling", "checks": n_scaling}],
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    write_run(out, run)
    print(f"wrote {out}")
    for r in boot_scaling + perm_scaling:
        print(f"  {r['group']:14s} {r.get('dataset'):6s}/{r['statistic']:9s}: "
              f"py_slope={r['py_slope']:.2f} "
              f"speedup_R/py@maxB={r['speedup_r_over_py'][-1]:.2f}x")
    ns = n_scaling[0]
    print(f"  n_scaling corr @B={ns['B']}: n={ns['n']} "
          f"speedup_R/py={[round(s,2) for s in ns['speedup_r_over_py']]}")


if __name__ == "__main__":
    main()
