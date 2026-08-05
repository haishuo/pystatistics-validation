"""Correctness runner — the DETERMINISTIC tier (diagnostics + stationarity +
decomposition). These quantities are defined by their algorithm, not by an
optimizer, so agreement with R is expected TIGHT (near machine precision or the
defined convention).

Emits artifacts/timeseries/v<ver>/runs/deterministic.json (render-from-artifacts).

Run:  MVNMLE_DATA_DIR=Dev/datasets python drivers/timeseries/run_deterministic.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsdata import (load_series, CANONICAL, short_series, missing_series)  # noqa: E402
from rref import r_reference, r_package_versions  # noqa: E402
from tscompare import arr_cmp, scalar_cmp, within  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run  # noqa: E402
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from pystatistics.timeseries import (  # noqa: E402
    acf, pacf, diff, ndiffs, adf_test, kpss_test, decompose, stl)

try:
    from statsmodels.tsa.stattools import adfuller
    _HAVE_SM = True
except Exception:
    _HAVE_SM = False

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/timeseries/v{ver}/runs/deterministic.json")


# --------------------------------------------------------------------------
# Diagnostics: acf / pacf / diff / ndiffs
# --------------------------------------------------------------------------
def run_diagnostics() -> list[dict]:
    recs = []
    for key in CANONICAL:
        s = load_series(key)
        L = min(24, len(s.y) // 4)
        pa = acf(s.y, max_lag=L)
        ra = r_reference("acf", s.y, max_lag=L, demean=True)
        pp = pacf(s.y, max_lag=L)
        rp = r_reference("pacf", s.y, max_lag=L)
        ca, cp = arr_cmp(pa.acf, ra["acf"]), arr_cmp(pp.acf, rp["pacf"])
        recs.append({"group": "acf_pacf", "series": key, "max_lag": L,
                     "acf": ca, "pacf": cp,
                     "pass": within(ca, abs_tol=1e-10) and within(cp, abs_tol=1e-9)})
    # diff: a few lag/difference combos on one series
    y = load_series("air_passengers").y
    for lag, d in [(1, 1), (1, 2), (12, 1)]:
        pd_ = diff(y, lag=lag, differences=d)
        rd = r_reference("diff", y, lag=lag, differences=d)
        c = arr_cmp(pd_, rd["diff"])
        recs.append({"group": "diff", "lag": lag, "differences": d,
                     "cmp": c, "pass": within(c, abs_tol=1e-12)})
    # ndiffs: both tests, several series (matched test between engines)
    for key in ["air_passengers", "co2", "nile", "sim_arma11"]:
        y = load_series(key).y
        for test in ["kpss", "adf"]:
            pv = ndiffs(y, test=test)
            rv = r_reference("ndiffs", y, test=test, alpha=0.05, max_d=2)["ndiffs"]
            rec = {"group": "ndiffs", "series": key, "test": test,
                   "py": int(pv), "r": int(rv), "pass": int(pv) == int(rv)}
            if int(pv) != int(rv) and test == "adf":
                # forecast::ndiffs uses its own internal ADF; a one-difference
                # gap arises only when the ADF p-value straddles alpha.
                rec["note"] = ("boundary: ADF p at d0 = "
                               f"{adf_test(y).p_value:.4f} (~alpha=0.05); "
                               "default test='kpss' matches R everywhere")
                rec["boundary_documented"] = True
            recs.append(rec)
    return recs


# --------------------------------------------------------------------------
# Stationarity: adf (stat vs tseries exact; p vs MacKinnon/statsmodels) + kpss
# --------------------------------------------------------------------------
def run_stationarity() -> list[dict]:
    recs = []
    for key in ["sim_arma11", "sim_nearunitroot", "nile", "lynx", "co2"]:
        y = load_series(key).y
        a = adf_test(y)  # default now regression='ct' (matches tseries)
        r = r_reference("adf", y, k=a.n_lags)
        stat = scalar_cmp(a.statistic, r["statistic"])
        rec = {"group": "adf", "series": key, "regression": "ct",
               "n_lags": int(a.n_lags), "stat_vs_tseries": stat}
        if _HAVE_SM:
            smp = adfuller(y, maxlag=a.n_lags, regression="ct", autolag=None)[1]
            rec["p_vs_statsmodels"] = scalar_cmp(a.p_value, smp)
            rec["pass"] = (within(stat, abs_tol=1e-4)
                           and within(rec["p_vs_statsmodels"], abs_tol=1e-3))
        else:
            rec["pass"] = within(stat, abs_tol=1e-4)
        recs.append(rec)
    for key in ["sim_arma11", "nile", "co2"]:
        y = load_series(key).y
        for null in ["Level", "Trend"]:
            k = kpss_test(y, regression=("c" if null == "Level" else "ct"))
            r = r_reference("kpss", y, null=null)
            stat = scalar_cmp(k.statistic, r["statistic"])
            recs.append({"group": "kpss", "series": key, "null": null,
                         "lag_py": int(k.n_lags), "lag_r": int(r["lag"]),
                         "stat": stat, "p_py": k.p_value, "p_r": r["p_value"],
                         "pass": within(stat, abs_tol=1e-4)
                                 and int(k.n_lags) == int(r["lag"])})
    return recs


# --------------------------------------------------------------------------
# Decomposition: classical decompose + STL (both deterministic -> tight)
# --------------------------------------------------------------------------
def run_decomposition() -> list[dict]:
    recs = []
    for key, typ in [("air_passengers", "additive"),
                     ("air_passengers", "multiplicative"),
                     ("co2", "additive")]:
        s = load_series(key)
        p = decompose(s.y, period=s.period, kind=typ)
        r = r_reference("decompose", s.y, period=s.period, type=typ)
        cs = arr_cmp(p.seasonal, r["seasonal"])
        ct = arr_cmp(p.trend, r["trend"])
        cr = arr_cmp(p.residual, r["random"])
        recs.append({"group": "decompose", "series": key, "type": typ,
                     "seasonal": cs, "trend": ct, "random": cr,
                     "pass": within(cs, abs_tol=1e-9)
                             and within(ct, abs_tol=1e-9)})
    # STL — feed R pystatistics' EXACT windows/degrees/jumps (from .info) so both
    # engines run identical parameters; agreement is then to machine precision.
    def _stl_case(key, period, s_window=None, robust=False, n_inner=2,
                  n_outer=0, label=""):
        y = load_series(key).y
        p = stl(y, period=period, seasonal_window=s_window, robust=robust,
                n_inner=n_inner, n_outer=n_outer)
        w, d, j = p.info["windows"], p.info["degrees"], p.info["jumps"]
        r = r_reference("stl", y, period=period,
                        s_window=(None if s_window is None else s_window),
                        s_degree=d["seasonal"], t_window=w["trend"],
                        t_degree=d["trend"], l_window=w["lowpass"],
                        l_degree=d["lowpass"], s_jump=j["seasonal"],
                        t_jump=j["trend"], l_jump=j["lowpass"],
                        n_inner=p.info["n_inner"], n_outer=p.info["n_outer"],
                        robust=robust)
        cs, ct, cr = (arr_cmp(p.seasonal, r["seasonal"]),
                      arr_cmp(p.trend, r["trend"]),
                      arr_cmp(p.residual, r["remainder"]))
        return {"group": "stl", "series": key, "config": label,
                "windows": w, "seasonal": cs, "trend": ct, "remainder": cr,
                "pass": within(cs, abs_tol=1e-8) and within(ct, abs_tol=1e-8)}

    recs.append(_stl_case("co2", 12, label="default/periodic"))
    recs.append(_stl_case("air_passengers", 12, label="default/periodic"))
    recs.append(_stl_case("co2", 12, s_window=7, label="s.window=7"))
    recs.append(_stl_case("co2", 12, s_window=35, label="s.window=35"))
    recs.append(_stl_case("co2", 12, robust=True, n_outer=15,
                          label="robust (inner=1,outer=15)"))
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions(
        ("forecast", "tseries", "urca", "stats"))
    import statsmodels
    env["statsmodels"] = statsmodels.__version__

    diagnostics = run_diagnostics()
    stationarity = run_stationarity()
    decomposition = run_decomposition()

    run = build_run(
        env=env,
        config={"suite": "timeseries-deterministic",
                "reference": "R stats/forecast/tseries + statsmodels (ADF p)",
                "tolerance_contract": "TIGHT — deterministic quantities to "
                "machine precision or defined convention"},
        records=[{"key": "diagnostics", "checks": diagnostics},
                 {"key": "stationarity", "checks": stationarity},
                 {"key": "decomposition", "checks": decomposition}],
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    for grp in (diagnostics, stationarity, decomposition):
        npass = sum(bool(r.get("pass")) for r in grp)
        print(f"  {grp[0]['group']:12s}... {npass}/{len(grp)} pass")


if __name__ == "__main__":
    main()
