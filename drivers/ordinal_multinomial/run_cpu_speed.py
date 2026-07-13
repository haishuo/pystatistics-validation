"""G3 performance — CPU pystatistics vs R across problem sizes (R1/R2/R3).

The mandatory CPU-vs-R scaling study (Guarantee 3). Both engines fit the SAME
generated design; the fit is timed best-of-reps. We sweep:
  - polr: n at fixed (p, K) — MASS::polr (optim BFGS) vs polr (L-BFGS-B + Newton
    polish + a finite-difference observed-information Hessian for the vcov R also
    computes with Hess=TRUE).
  - multinom: n at fixed (p, K), and K (number of categories) at fixed (n, p) —
    nnet::multinom (BFGS) vs multinom (L-BFGS-B).

We report py_ms, r_ms, ratio, and the empirical py/R log-log slopes so a
complexity-class gap (R1) would show as a diverging slope, not just a constant
factor. A documented slower-for-a-reason gap (R2) is named, not buried.

Emits artifacts/ordinal_multinomial/v<ver>/runs/cpu_speed.json.
Run: python drivers/ordinal_multinomial/run_cpu_speed.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omdata  # noqa: E402
from omref import (r_time_polr, r_time_multinom, r_package_versions)  # noqa: E402
from omcompare import to_native  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402
from pystatistics.ordinal import polr  # noqa: E402
from pystatistics.multinomial import multinom  # noqa: E402

_ART = (Path(__file__).resolve().parents[2]
        / "artifacts/ordinal_multinomial/v{ver}/runs/cpu_speed.json")


def _py_best(fn, reps: int = 5) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _slopes(sizes, py_ms, r_ms) -> tuple[float, float]:
    """log-log slope of ms vs size for each engine (empirical complexity exponent)."""
    lx = np.log(np.asarray(sizes, float))
    def slope(ms):
        ly = np.log(np.asarray(ms, float))
        return float(np.polyfit(lx, ly, 1)[0])
    return slope(py_ms), slope(r_ms)


def sweep_polr() -> list[dict]:
    ns = [500, 2000, 8000, 20000]
    py_ms, r_ms, rows = [], [], []
    for n in ns:
        des = omdata.synth_ordinal(n=n)                  # K=4, p=4, cont+factor
        tp = _py_best(lambda: polr(des.y, des.X, link="logit"))
        tr = r_time_polr(des.y, des.X, "logistic")
        py_ms.append(tp * 1e3); r_ms.append(tr * 1e3)
        rows.append({"func": "polr", "n": n, "p": des.X.shape[1],
                     "K": des.n_levels, "py_ms": tp * 1e3, "r_ms": tr * 1e3,
                     "ratio_py_over_r": tp / max(tr, 1e-9)})
    ps, rs = _slopes(ns, py_ms, r_ms)
    for row in rows:
        row["py_slope"], row["r_slope"] = ps, rs
        row["pass"] = row["ratio_py_over_r"] <= 1.5   # at/below R (G3), tolerance
    return rows


def sweep_multinom_n() -> list[dict]:
    ns = [500, 2000, 8000, 20000]
    py_ms, r_ms, rows = [], [], []
    for n in ns:
        des = omdata.synth_multinom(n=n, K=3, p=3)
        tp = _py_best(lambda: multinom(des.y, des.X, max_iter=2000))
        tr = r_time_multinom(des.y, des.X[:, 1:], des.r_levels)
        py_ms.append(tp * 1e3); r_ms.append(tr * 1e3)
        rows.append({"func": "multinom(n)", "n": n, "p": des.X.shape[1],
                     "K": des.n_classes, "py_ms": tp * 1e3, "r_ms": tr * 1e3,
                     "ratio_py_over_r": tp / max(tr, 1e-9)})
    ps, rs = _slopes(ns, py_ms, r_ms)
    for row in rows:
        row["py_slope"], row["r_slope"] = ps, rs
        row["pass"] = row["ratio_py_over_r"] <= 1.5
    return rows


def sweep_multinom_k() -> list[dict]:
    Ks = [3, 5, 8, 12]
    py_ms, r_ms, rows = [], [], []
    for K in Ks:
        des = omdata.synth_multinom(n=4000, K=K, p=4)
        tp = _py_best(lambda: multinom(des.y, des.X, max_iter=2000))
        tr = r_time_multinom(des.y, des.X[:, 1:], des.r_levels)
        py_ms.append(tp * 1e3); r_ms.append(tr * 1e3)
        rows.append({"func": "multinom(K)", "n": 4000, "p": des.X.shape[1],
                     "K": K, "py_ms": tp * 1e3, "r_ms": tr * 1e3,
                     "ratio_py_over_r": tp / max(tr, 1e-9)})
    ps, rs = _slopes(Ks, py_ms, r_ms)
    for row in rows:
        row["py_slope"], row["r_slope"] = ps, rs
        row["pass"] = row["ratio_py_over_r"] <= 1.5
    return rows


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions()

    polr_rows = sweep_polr()
    mn_n = sweep_multinom_n()
    mn_k = sweep_multinom_k()

    run = build_run(
        env=env,
        config={"suite": "ordinal-multinomial-cpu-speed",
                "reference": "R MASS::polr (optim BFGS) / nnet::multinom (BFGS)",
                "note": "best-of-5 fit wall time; both engines fit identical data"},
        records=to_native([{"key": "polr_scaling", "checks": polr_rows},
                           {"key": "multinom_n_scaling", "checks": mn_n},
                           {"key": "multinom_k_scaling", "checks": mn_k}]),
    )
    out = Path(str(_ART).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    for name, grp in [("polr", polr_rows), ("multinom(n)", mn_n),
                      ("multinom(K)", mn_k)]:
        ratios = ", ".join(f"{r['ratio_py_over_r']:.2f}" for r in grp)
        print(f"  {name:12s} py/R ratios: {ratios}")


if __name__ == "__main__":
    main()
