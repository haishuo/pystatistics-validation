"""arima_batch failure-contract + correctness verification (CPU).

Emits artifacts/timeseries/v<ver>/runs/batch_contract.json.

The 4.6.5 fix made both backends honor ONE loud, consistent failure contract:
- all series fail        -> raise ConvergenceError (no usable non-stationary batch)
- some series fail       -> failed rows' estimates are NaN + a loud warning naming
                            the count; healthy rows are correct (never a plain
                            non-stationary number presented as a fit)
- all series good        -> normal result, no warning
- single-series Whittle  -> still raises (no partial-failure case)

arima_batch has no direct R equivalent (R has no batched ARMA), so batch
CORRECTNESS is anchored to the single-series `arima(method='Whittle')` path
(itself validated vs R / exact Kalman in mle.json): the CPU batch is a loop, so
its per-row estimates must equal the single-series fit to machine precision.

The GPU half of the contract (backend='gpu'/'gpu_fp64' must behave identically)
is verified on Forge CUDA in gpu_cuda_forge.json.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402
from tscompare import to_native, expect_raises, arr_cmp, within  # noqa: E402

from pystatistics.timeseries import arima_batch, arima  # noqa: E402
from pystatistics.core.exceptions import ConvergenceError  # noqa: E402

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/timeseries/v{ver}/runs/batch_contract.json")


def _arma(n: int, seed: int, ar: float, ma: float) -> np.ndarray:
    """Deterministic ARMA(1,1) realization (seed recorded in the artifact)."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 50)
    y = np.zeros(n + 50)
    for t in range(1, n + 50):
        y[t] = ar * y[t - 1] + e[t] + ma * e[t - 1]
    return y[50:]


def _batch_call(Y):
    """Run a CPU batch capturing whether a warning fired and its text."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = arima_batch(Y, order=(1, 0, 1), method="Whittle", backend="cpu")
        msgs = [str(x.message) for x in w]
    return b, msgs


def run_contract() -> list[dict]:
    checks: list[dict] = []
    n, ar, ma = 1500, 0.6, 0.4

    # 1. all-good: no warning, all converged, no NaN, estimates recover the DGP
    Yg = np.stack([_arma(n, 5000 + i, ar, ma) for i in range(16)])
    b, msgs = _batch_call(Yg)
    arr = np.asarray(b.ar).ravel()
    checks.append(to_native({
        "key": "all_good", "desc": "healthy batch: no warning, all converged, no NaN",
        "converged": int(np.sum(b.converged)), "k": len(b.converged),
        "n_nan": int(np.isnan(arr).sum()), "warned": len(msgs),
        "ar_mean": float(np.nanmean(arr)), "ma_mean": float(np.nanmean(b.ma)),
        "pass": bool(np.all(b.converged) and np.isnan(arr).sum() == 0
                     and len(msgs) == 0 and abs(np.nanmean(arr) - ar) < 0.05),
    }))

    # 2. partial-fail: failed rows NaN + loud warning; healthy rows intact; and
    #    crucially NO finite AR > 1 returned as a plain number (the old GPU bug).
    Yp = np.stack([_arma(n, 6000 + i, ar, ma) for i in range(12)]
                  + [_arma(n, 7000 + i, 0.99, 0.3) for i in range(4)])
    b, msgs = _batch_call(Yp)
    arr = np.asarray(b.ar).ravel()
    finite = arr[np.isfinite(arr)]
    checks.append(to_native({
        "key": "partial_fail",
        "desc": "mixed batch: failed rows -> NaN + warning; no non-stationary number returned",
        "converged": int(np.sum(b.converged)), "k": 16,
        "n_nan": int(np.isnan(arr).sum()), "warned": len(msgs),
        "warning": msgs[0] if msgs else None,
        "max_finite_ar": float(finite.max()) if finite.size else float("nan"),
        "pass": bool(np.isnan(arr).sum() == 4 and len(msgs) == 1
                     and (finite.size == 0 or finite.max() < 1.0)),
    }))

    # 3. all-fail: raises (no usable batch of non-stationary fits)
    Yf = np.stack([_arma(n, 8000 + i, 0.99, 0.3) for i in range(2)])
    r = expect_raises(lambda: arima_batch(Yf, order=(1, 0, 1), method="Whittle",
                                          backend="cpu"), ConvergenceError)
    r.update({"key": "all_fail", "desc": "all series non-stationary -> raise ConvergenceError"})
    checks.append(to_native(r))

    # 4. single-series Whittle still raises (no partial case there)
    r = expect_raises(lambda: arima(_arma(n, 9999, 0.99, 0.3), order=(1, 0, 1),
                                    method="Whittle"), ConvergenceError)
    r.update({"key": "single_series_raises",
              "desc": "single-series Whittle non-stationary -> raise"})
    checks.append(to_native(r))

    # 5. batch correctness anchor: at MATCHED tol the CPU batch IS the single-
    #    series loop, so per-row estimates agree exactly (0.0). (At default tols
    #    they differ by ~1e-6 = the batch's looser default grad tol 1e-5 vs
    #    arima's 1e-8 — a convergence-tolerance difference, not a discrepancy.)
    tol = 1e-8
    bg = arima_batch(Yg, order=(1, 0, 1), method="Whittle", backend="cpu", tol=tol)
    single_ar, single_ma = [], []
    for i in range(16):
        s = arima(Yg[i], order=(1, 0, 1), method="Whittle", tol=tol)
        single_ar.append(float(s.ar[0])); single_ma.append(float(s.ma[0]))
    ar_cmp = arr_cmp(np.asarray(bg.ar).ravel(), single_ar)
    ma_cmp = arr_cmp(np.asarray(bg.ma).ravel(), single_ma)
    checks.append(to_native({
        "key": "batch_vs_single_series",
        "desc": "CPU batch per-row estimates == single-series arima(Whittle) loop "
                "at matched tol (exact)",
        "tol": tol, "ar": ar_cmp, "ma": ma_cmp,
        "pass": bool(within(ar_cmp, abs_tol=1e-10) and within(ma_cmp, abs_tol=1e-10)),
    }))

    return checks


def main() -> None:
    warnings.filterwarnings("default")
    env = env_manifest(device="cpu")
    require_pypi(env)
    checks = run_contract()
    run = build_run(
        env=env,
        config={"suite": "timeseries-batch-contract",
                "reference": "single-series arima(Whittle) + A6 fail-loud contract",
                "tolerance_contract": "all-fail raises; partial -> NaN+warn; "
                "healthy unchanged; CPU batch == single-series to 1e-8; GPU parity "
                "on Forge (gpu_cuda_forge.json)"},
        records=to_native([{"key": "batch_contract", "checks": checks}]),
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    write_run(out, run)
    print(f"wrote {out}")
    for c in checks:
        print(f"  {c['key']:<24} pass={c['pass']}")


if __name__ == "__main__":
    main()
