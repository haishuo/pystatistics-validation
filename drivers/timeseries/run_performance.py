"""Performance / scaling runner (Guarantee 3, R1/R3) — CPU pystatistics vs R
across series lengths. Confirms no complexity-class gap: the Kalman/loess/FFT
recursions are O(n) (or O(n log n)), and pystatistics does not lag R with n.

Times the fit only (data generated once per n, fed identically to both engines).
Emits artifacts/timeseries/v<ver>/runs/performance.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tscompare import to_native  # noqa: E402
from rref import r_package_versions  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402
from pystatistics.timeseries import acf, arima, ets, stl  # noqa: E402

_HERE = Path(__file__).resolve().parent
_PERF_R = _HERE / "perf_reference.R"
_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/timeseries/v{ver}/runs/performance.json")


def _ar1(n: int, phi: float = 0.5, seed: int = 7) -> np.ndarray:
    """Deterministic stationary AR(1) of length n (fits cleanly at every n)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    y = np.empty(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y + 10.0


def _r_time(func: str, y: np.ndarray, *, period: int = 1, reps: int = 5,
            **params) -> float:
    job = {"func": func, "data": [float(v) for v in y], "period": period,
           "reps": reps, "params": params}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as jf:
        json.dump(job, jf)
        path = jf.name
    try:
        proc = subprocess.run(["Rscript", str(_PERF_R), path],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"R perf '{func}' failed: {proc.stderr[:200]}")
        return float(json.loads(proc.stdout)["elapsed"])
    finally:
        Path(path).unlink(missing_ok=True)


def _py_time(fn, reps: int = 5) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _slope(ns, ts):
    """Empirical complexity exponent from a log-log least-squares fit."""
    ns, ts = np.asarray(ns, float), np.asarray(ts, float)
    ok = ts > 0
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(ns[ok]), np.log(ts[ok]), 1)[0])


def scale(func: str, ns, py_call, *, period=1, reps=5, **rp) -> dict:
    rows = []
    for n in ns:
        y = _ar1(n)
        pt = _py_time(lambda: py_call(y), reps=reps)
        rt = _r_time(func, y, period=period, reps=reps, **rp)
        rows.append({"n": n, "py_s": pt, "r_s": rt,
                     "ratio_py_over_r": (pt / rt) if rt > 0 else None})
    return {"func": func, "rows": rows,
            "py_slope": _slope([r["n"] for r in rows], [r["py_s"] for r in rows]),
            "r_slope": _slope([r["n"] for r in rows], [r["r_s"] for r in rows])}


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions(("forecast", "tseries", "stats"))
    env["r_blas"] = subprocess.run(
        ["Rscript", "-e", "cat(La_library())"],
        capture_output=True, text=True).stdout.strip() or "unknown"

    studies = []
    # acf: O(n) grid up to 20000
    studies.append(scale(
        "acf", [200, 500, 1000, 2000, 5000, 10000, 20000],
        lambda y: acf(y, max_lag=20), max_lag=20, reps=7))
    # arima(1,0,0) ML: Kalman O(n)
    studies.append(scale(
        "arima", [200, 500, 1000, 2000, 5000],
        lambda y: arima(y, order=(1, 0, 0), method="CSS-ML"),
        order=[1, 0, 0], method="CSS-ML", reps=3))
    # ets(ANN): O(n) state-space recursion
    studies.append(scale(
        "ets", [200, 500, 1000, 2000, 5000],
        lambda y: ets(y, model="ANN"), model="ANN", reps=3))
    # stl (needs a seasonal series): use period=12, O(n)
    def _stl_py(y):
        return stl(y, period=12)
    stl_ns = [240, 480, 960, 2400, 4800]
    rows = []
    for n in stl_ns:
        y = _ar1(n) + 5 * np.sin(2 * np.pi * np.arange(n) / 12)
        pt = _py_time(lambda: _stl_py(y), reps=3)
        rt = _r_time("stl", y, period=12, reps=3)
        rows.append({"n": n, "py_s": pt, "r_s": rt,
                     "ratio_py_over_r": (pt / rt) if rt > 0 else None})
    studies.append({"func": "stl", "rows": rows,
                    "py_slope": _slope([r["n"] for r in rows], [r["py_s"] for r in rows]),
                    "r_slope": _slope([r["n"] for r in rows], [r["r_s"] for r in rows])})

    run = build_run(
        env=env,
        config={"suite": "timeseries-performance",
                "reference": "R stats/forecast/tseries on the same hardware",
                "tolerance_contract": "no complexity-class gap; CPU py must not "
                "lag R with n (Guarantee 3 / R1)"},
        records=to_native([{"key": "scaling", "studies": studies}]),
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    for s in studies:
        big = s["rows"][-1]
        ratio = big["ratio_py_over_r"]
        rs = f"{ratio:.2f}x" if ratio is not None else "n/a"
        print(f"  {s['func']:6s} py_slope {s['py_slope']:.2f} r_slope "
              f"{s['r_slope']:.2f} | n={big['n']}: py {big['py_s']*1e3:.1f}ms "
              f"R {big['r_s']*1e3:.1f}ms ratio {rs}")


if __name__ == "__main__":
    main()
