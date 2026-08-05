"""Fidelity runner (Guarantee 2 / A6) — fail loud, never silently substitute.

Every unsupported spec, invalid order, non-convergent fit, or non-finite input
must raise (not return a silently-wrong result or a quietly-different model).
NA handling is a deliberate fail-loud scope limit: pystatistics refuses non-finite
input rather than applying an na.action convention the way R does.

Emits artifacts/timeseries/v<ver>/runs/fidelity.json.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsdata import load_series, missing_series  # noqa: E402
from tscompare import expect_raises, to_native  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run  # noqa: E402
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402
from pystatistics.core.exceptions import ValidationError, ConvergenceError  # noqa: E402
from pystatistics.timeseries import (  # noqa: E402
    ets, arima, adf_test, kpss_test, stl, ndiffs, acf, pacf, arima_batch)

from artifact_root import artifact_root  # noqa: E402

_ARTIFACT = (artifact_root(Path(__file__).resolve().parents[2])
             / "timeseries/v{ver}/runs/fidelity.json")


def run_fidelity() -> list[dict]:
    y = load_series("nile").y
    na = missing_series().y
    neg = np.array([1., -2, 3, 4, 5, 6, 7, 8, -1, 2, 3, 4] * 3)
    checks = []

    def rec(key, desc, fn, exc, na_note=None):
        r = expect_raises(fn, exc)
        r.update({"key": key, "desc": desc})
        if na_note:
            r["note"] = na_note
        checks.append(r)

    # unsupported / invalid specs
    rec("ets_bad_model", "ets invalid model string 'XNN'",
        lambda: ets(y, model="XNN"), ValidationError)
    rec("ets_mult_nonpositive", "ets multiplicative error on non-positive data",
        lambda: ets(neg, model="MNN"), ValidationError)
    rec("arima_neg_order", "arima negative order component",
        lambda: arima(y, order=(-1, 0, 0)), ValidationError)
    rec("arima_nonconv", "arima over-parameterized / non-convergent fails loud",
        lambda: arima(y[:5], order=(3, 0, 3)), (ConvergenceError, ValidationError))
    rec("adf_bad_regression", "adf_test invalid regression",
        lambda: adf_test(y, regression="xx"), ValidationError)
    rec("kpss_bad_regression", "kpss_test invalid regression",
        lambda: kpss_test(y, regression="xx"), ValidationError)
    rec("stl_period_lt2", "stl period < 2",
        lambda: stl(y, period=1), ValidationError)
    rec("stl_too_short", "stl series shorter than two periods",
        lambda: stl(y[:10], period=12), ValidationError)
    rec("ndiffs_bad_test", "ndiffs invalid test",
        lambda: ndiffs(y, test="xx"), ValidationError)
    # NA handling — fail loud (documented divergence from R's na.action/Kalman)
    rec("acf_na", "acf on series with NA -> fail loud (R uses na.action)",
        lambda: acf(na, max_lag=5), ValidationError,
        na_note="R stats::acf handles NA via na.action=na.pass; pystatistics "
                "refuses non-finite input — a deliberate fail-loud scope limit")
    rec("arima_na", "arima on series with NA -> fail loud (R uses Kalman)",
        lambda: arima(na, order=(1, 0, 0)), ValidationError,
        na_note="R stats::arima skips NA via the Kalman filter; pystatistics "
                "refuses non-finite input — impute or drop NAs first")
    # arima_batch: non-stationary Whittle fit fails loud, not a silent wrong ARMA
    rec("batch_nonstationary", "arima_batch non-stationary Whittle -> fail loud",
        lambda: arima_batch(np.stack([load_series("sim_nearunitroot").y] * 2),
                            order=(1, 0, 1)), ConvergenceError)
    return checks


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    checks = run_fidelity()
    run = build_run(
        env=env,
        config={"suite": "timeseries-fidelity",
                "reference": "A6 fail-loud / no silent substitution",
                "tolerance_contract": "every unsupported spec / invalid order / "
                "non-convergence / non-finite input must raise"},
        records=to_native([{"key": "fidelity", "checks": checks}]),
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    npass = sum(c["pass"] for c in checks)
    print(f"  fidelity {npass}/{len(checks)} fail-loud as expected")
    for c in checks:
        if not c["pass"]:
            print(f"    MISS {c['key']}: expected {c['expected']} got {c['got']}")


if __name__ == "__main__":
    main()
