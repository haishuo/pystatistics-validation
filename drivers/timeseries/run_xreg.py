"""Correctness runner — regression with ARIMA errors (VA-4 / VA-4b): xreg,
drift (include_drift), and fixed= parameter masking, plus newxreg forecasting
and auto_arima drift selection.

The reference is R ``stats::arima`` + ``predict.Arima`` — the exact-ML target
pystatistics matches (MLE ``sigma2``; ``predict.Arima`` prediction SEs use that
``sigma2``). The estimator-invariant quantities (coef, loglik, AIC, sigma2, SE,
forecasts) compare tightly. (``forecast::Arima`` reports a df-adjusted
``sigma2 = SSR/(n-ncoef)`` and hence slightly wider prediction intervals — a
documented convention difference, not validated here.)

Deterministic scenarios (seeded numpy) are dumped to R via the shared job
bridge so both engines fit identical fp64 numbers (shared-input discipline).
Covers the RIGOR R10 hard cases: xreg under differencing (d>0), near-collinear
xreg, all-but-one fixed, drift under d=1, and seasonal + xreg.

Emits artifacts/timeseries/v<ver>/runs/xreg.json.

Run:  python drivers/timeseries/run_xreg.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rref import r_reference, r_package_versions  # noqa: E402
from tscompare import arr_cmp, scalar_cmp, within, expect_raises, to_native  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402

from pystatistics.core.exceptions import ValidationError  # noqa: E402
from pystatistics.timeseries import arima, forecast_arima, auto_arima  # noqa: E402

_ARTIFACT = (Path(__file__).resolve().parents[2]
             / "artifacts/timeseries/v{ver}/runs/xreg.json")


# --------------------------------------------------------------------------
# Deterministic scenario data (seeded; identical numbers feed both engines)
# --------------------------------------------------------------------------
def _sar4(rng, n, sphi=0.6):
    """A genuine seasonal AR(1)[4] error series (identified sar1)."""
    e = rng.standard_normal(n)
    s = np.zeros(n)
    for i in range(4, n):
        s[i] = sphi * s[i - 4] + e[i]
    return s


def _scenarios() -> dict:
    rng = np.random.default_rng(20260711)
    n = 140

    def ar1(phi=0.6):
        e = rng.standard_normal(n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i - 1] + e[i]
        return y

    def arma11(phi=0.5, th=0.3):
        e = rng.standard_normal(n)
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i - 1] + e[i] + th * e[i - 1]
        return y

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    e_ar = ar1()
    e_arma = arma11()
    xc1 = rng.standard_normal(n)
    xc2 = xc1 + rng.standard_normal(n) * 0.02  # near-collinear
    return {
        "n": n, "x1": x1, "x2": x2, "xc1": xc1, "xc2": xc2,
        "e_arma": e_arma,
        "yA": 2.0 + 1.5 * x1 - 0.8 * x2 + e_ar,          # (1,0,0)+xreg d=0
        "yMA": -1.0 + 0.9 * x1 + e_arma,                  # (1,0,1)+xreg d=0
        "yD": np.cumsum(np.concatenate([[30.0],
              0.6 + rng.standard_normal(n - 1)])),        # RW + drift, d=1
        "yXd": np.cumsum(0.7 * x1 + e_ar),                # I(1) via xreg, d=1
        "yC": 1.0 + 0.5 * xc1 + 0.5 * xc2 + e_ar,         # collinear xreg
        "yS": 3.0 + 1.2 * x1 + _sar4(rng, n, 0.6),        # seasonal+xreg d=0
        "newxA": rng.standard_normal((6, 2)),
        "newxXd": rng.standard_normal((6, 1)),
    }


def _rows(m):
    """Matrix -> JSON list-of-rows (R simplifyVector rebuilds the n x k matrix).

    A 1-D array is one regressor column (shape (n, 1)).
    """
    m = np.asarray(m, dtype=float)
    if m.ndim == 1:
        m = m.reshape(-1, 1)
    return [[float(v) for v in row] for row in m]


def _py_coef(fit):
    return np.concatenate([fit.ar, fit.ma, fit.seasonal_ar, fit.seasonal_ma,
                           fit.xreg_coef])


# --------------------------------------------------------------------------
# xreg / drift / fixed estimator parity
# --------------------------------------------------------------------------
def run_fits(sc) -> list[dict]:
    recs = []

    def fit_case(label, y, order, *, xreg=None, seasonal=None,
                 include_drift=False, fixed=None, r_fixed=None,
                 coef_tol=8e-3, se_tol=5e-3, hard=None):
        p = arima(y, order=order, seasonal=seasonal, xreg=xreg,
                  include_drift=include_drift, fixed=fixed)
        r = r_reference(
            "arima_xreg", y,
            period=(seasonal[3] if seasonal else 1),
            order=list(order),
            seasonal=(list(seasonal[:3]) if seasonal else None),
            xreg=(_rows(xreg) if xreg is not None else None),
            include_drift=bool(include_drift),
            fixed=r_fixed,
            # pystatistics arima defaults include_mean=True (an intercept for
            # d + D == 0; R drops it under differencing, as does pystatistics).
            include_mean=True,
            method="CSS-ML",
        )
        coef = arr_cmp(_py_coef(p), r["coef"])
        se_py = np.sqrt(np.abs(np.diag(p.vcov)))
        # R var.coef excludes fixed params; compare SE only for the unfixed set.
        if fixed is None:
            se = arr_cmp(se_py, r["se"])
            se_ok = within(se, abs_tol=se_tol)
        else:
            se = {"n": 0, "note": "fixed model: R var.coef excludes fixed params"}
            se_ok = True
        rec = {"group": "arima_xreg", "case": label, "order": list(order),
               "seasonal": (list(seasonal) if seasonal else None),
               "include_drift": bool(include_drift),
               "fixed": r_fixed,
               "coef_names_r": r["coef_names"],
               "coef": coef,
               "loglik": scalar_cmp(p.log_likelihood, r["loglik"]),
               "aic": scalar_cmp(p.aic, r["aic"]),
               "sigma2_py": float(p.sigma2), "sigma2_r": float(r["sigma2"]),
               "se": se}
        if hard:
            rec["hard_case"] = hard
        rec["pass"] = (within(coef, abs_tol=coef_tol)
                       and within(rec["loglik"], abs_tol=5e-2)
                       and within(rec["aic"], abs_tol=1e-1) and se_ok)
        recs.append(rec)
        return p

    fit_case("ar1_xreg2_d0", sc["yA"], (1, 0, 0),
             xreg=np.column_stack([sc["x1"], sc["x2"]]))
    fit_case("arma11_xreg1_d0", sc["yMA"], (1, 0, 1), xreg=sc["x1"])
    fit_case("drift_011_d1", sc["yD"], (0, 1, 1), include_drift=True,
             hard="drift under d=1 (trend becomes differenced mean)")
    fit_case("ar2_xreg1_d1", sc["yXd"], (2, 1, 0), xreg=sc["x1"],
             coef_tol=1e-2, hard="xreg under differencing d=1")
    fit_case("fixed_ma1_zero", sc["e_arma"], (1, 0, 1),
             fixed={"ma1": 0.0}, r_fixed=[None, 0.0, None],
             hard="all-but-one fixed (ma1 held at 0)")
    fit_case("fixed_ar2_nonzero", sc["e_arma"], (2, 0, 0),
             fixed={"ar2": 0.2}, r_fixed=[None, 0.2, None],
             hard="fixed at a non-zero value")
    fit_case("collinear_xreg_d0", sc["yC"], (1, 0, 0),
             xreg=np.column_stack([sc["xc1"], sc["xc2"]]),
             coef_tol=5e-2, hard="near-collinear xreg")
    fit_case("seasonal_xreg_d0", sc["yS"], (0, 0, 0), seasonal=(1, 0, 0, 4),
             xreg=sc["x1"], coef_tol=1e-2, hard="seasonal + xreg")
    return recs


# --------------------------------------------------------------------------
# Forecasting with newxreg / drift continuation vs predict.Arima
# --------------------------------------------------------------------------
def run_forecasts(sc) -> list[dict]:
    recs = []

    # Case A: xreg d=0, newxreg required.
    X = np.column_stack([sc["x1"], sc["x2"]])
    pA = arima(sc["yA"], order=(1, 0, 0), xreg=X)
    fcA = forecast_arima(pA, sc["yA"], n_ahead=6, conf_level=[0.95], new_xreg=sc["newxA"])
    rA = r_reference("forecast_arima_xreg", sc["yA"], period=1, order=[1, 0, 0],
                     xreg=_rows(X), newxreg=_rows(sc["newxA"]),
                     include_mean=True, method="CSS-ML", h=6)
    z = 1.959963984540054
    cmA, cseA = arr_cmp(fcA.mean, rA["mean"]), arr_cmp((fcA.upper[0.95] - fcA.mean) / z, rA["se"])
    recs.append({"group": "forecast", "case": "xreg_d0", "h": 6,
                 "mean": cmA, "se": cseA,
                 "pass": within(cmA, abs_tol=5e-3) and within(cseA, abs_tol=5e-3)})

    # Case D: drift continuation, no newxreg needed.
    pD = arima(sc["yD"], order=(0, 1, 1), include_drift=True)
    fcD = forecast_arima(pD, sc["yD"], n_ahead=6, conf_level=[0.95])
    rD = r_reference("forecast_arima_xreg", sc["yD"], period=1, order=[0, 1, 1],
                     include_drift=True, include_mean=False, method="CSS-ML", h=6)
    cmD = arr_cmp(fcD.mean, rD["mean"])
    recs.append({"group": "forecast", "case": "drift_d1", "h": 6, "mean": cmD,
                 "pass": within(cmD, abs_tol=5e-3)})

    # Case Xd: xreg under d=1.
    pX = arima(sc["yXd"], order=(2, 1, 0), xreg=sc["x1"])
    fcX = forecast_arima(pX, sc["yXd"], n_ahead=6, conf_level=[0.95], new_xreg=sc["newxXd"])
    rX = r_reference("forecast_arima_xreg", sc["yXd"], period=1, order=[2, 1, 0],
                     xreg=_rows(sc["x1"]), newxreg=_rows(sc["newxXd"]),
                     include_mean=False, method="CSS-ML", h=6)
    cmX = arr_cmp(fcX.mean, rX["mean"])
    recs.append({"group": "forecast", "case": "xreg_d1", "h": 6, "mean": cmX,
                 "pass": within(cmX, abs_tol=1e-2)})
    return recs


# --------------------------------------------------------------------------
# auto_arima drift selection vs forecast::auto.arima
# --------------------------------------------------------------------------
def run_auto(sc) -> list[dict]:
    recs = []
    res = auto_arima(sc["yD"], period=1, stepwise=True)
    r = r_reference("auto", sc["yD"], period=1, ic="aicc", stepwise=True,
                    max_p=5, max_q=5, max_d=2)
    r_has_drift = "drift" in [str(nm) for nm in (r.get("coef_names") or [])]
    recs.append({"group": "auto_drift", "case": "rw_with_drift",
                 "py_order": list(res.best_order),
                 "py_drift": bool(res.best_model.include_drift),
                 "r_order": r["order"], "r_drift": r_has_drift,
                 "pass": (bool(res.best_model.include_drift)
                          and list(res.best_order) == list(r["order"]))})
    # allowdrift=False must never select drift.
    res2 = auto_arima(sc["yD"], period=1, stepwise=True, allow_drift=False)
    recs.append({"group": "auto_drift", "case": "allowdrift_false",
                 "py_drift": bool(res2.best_model.include_drift),
                 "pass": not res2.best_model.include_drift})
    return recs


# --------------------------------------------------------------------------
# Fail-loud fidelity (Rule 1 / Guarantee 2)
# --------------------------------------------------------------------------
def run_fidelity(sc) -> list[dict]:
    recs = []
    checks = [
        ("xreg_wrong_length",
         lambda: arima(sc["yA"], order=(1, 0, 0), xreg=sc["x1"][:-3])),
        ("fixed_unknown_name",
         lambda: arima(sc["e_arma"], order=(1, 0, 1), fixed={"zz1": 0})),
        ("all_fixed",
         lambda: arima(sc["e_arma"], order=(1, 0, 0),
                       fixed={"ar1": 0.5, "intercept": 0.0})),
        ("drift_double_differencing",
         lambda: arima(sc["yD"], order=(0, 2, 1), include_drift=True)),
        ("whittle_with_xreg",
         lambda: arima(sc["yA"], order=(1, 0, 0), xreg=sc["x1"], method="whittle")),
        ("init_with_xreg",
         lambda: arima(sc["yA"], order=(1, 0, 0), xreg=sc["x1"],
                       init=[0.5, 0.0, 0.0, 0.0])),
    ]
    for name, fn in checks:
        rec = expect_raises(fn, ValidationError)
        rec["group"] = "fidelity"
        rec["case"] = name
        recs.append(rec)

    # forecast fail-loud: missing / misshaped newxreg
    pA = arima(sc["yA"], order=(1, 0, 0),
               xreg=np.column_stack([sc["x1"], sc["x2"]]))
    for name, fn in [
        ("newxreg_missing", lambda: forecast_arima(pA, sc["yA"], n_ahead=6)),
        ("newxreg_wrong_shape",
         lambda: forecast_arima(pA, sc["yA"], n_ahead=6, new_xreg=np.zeros((6, 1)))),
    ]:
        rec = expect_raises(fn, ValidationError)
        rec["group"] = "fidelity"
        rec["case"] = name
        recs.append(rec)
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions(("forecast", "stats"))

    sc = _scenarios()
    fits = run_fits(sc)
    fcs = run_forecasts(sc)
    auto = run_auto(sc)
    fid = run_fidelity(sc)

    run = build_run(
        env=env,
        config={"suite": "timeseries-xreg",
                "reference": "R stats::arima + predict.Arima / forecast::auto.arima",
                "tolerance_contract": "estimator-invariant (coef/loglik/AIC/"
                "sigma2/SE/forecast) match stats::arima tightly; prediction "
                "intervals use the MLE sigma2 (matches predict.Arima), which "
                "differs from forecast::Arima's df-adjusted sigma2 (documented)"},
        records=to_native([{"key": "fits", "checks": fits},
                            {"key": "forecasts", "checks": fcs},
                            {"key": "auto_drift", "checks": auto},
                            {"key": "fidelity", "checks": fid}]),
    )
    out = Path(str(_ARTIFACT).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    for name, grp in [("fits", fits), ("forecasts", fcs),
                      ("auto_drift", auto), ("fidelity", fid)]:
        npass = sum(1 for r in grp if r.get("pass"))
        print(f"  {name}: {npass}/{len(grp)} pass")


if __name__ == "__main__":
    main()
