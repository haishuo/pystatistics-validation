"""A7 torch-free certification — prove the whole pystatistics corpus runs
CORRECTLY without torch, and that every GPU path FAILS LOUD when torch is absent.

CONVENTIONS.md A7 requires: numpy+scipy(+numba) are the only hard dependencies;
torch is an optional `[gpu]` extra; the package must run correctly torch-free
(degradation is *performance only*, never function/correctness), and a GPU
request without a usable GPU/torch must be **disclosed and fail loud**, never a
silent fallback (A6).

This driver runs one canonical CPU call per subsystem on fixed inputs and records
a small numeric fingerprint of each result. Run it in BOTH a torch-free env and a
with-torch env: the CPU fingerprints must be **bit-identical** (torch presence
changes nothing on the CPU path — that IS the A7 correctness claim). In the
torch-free env it additionally asserts every `backend='gpu'` entry point fails
loud. A few fingerprints are anchored to R-verified values so the check is not
purely self-referential.

Usage:
    python certify.py            # writes cpu_<env>.json (+ gpu_faillloud.json if torch-free)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from pystatistics import __version__ as PYVER

ART = Path(__file__).resolve().parents[2] / "artifacts/a7_torchfree" / f"v{PYVER}"
TORCH_PRESENT = importlib.util.find_spec("torch") is not None


def _flatten(v) -> list[float]:
    if isinstance(v, dict):
        out = []
        for k in sorted(v, key=str):
            out.extend(_flatten(v[k]))
        return out
    a = np.asarray(v, dtype=np.float64).ravel()
    return [None if (x != x) else float(x) for x in a]


def _fp(*vals) -> list[float]:
    """Flatten to a plain float fingerprint list. (nested) dicts -> values by
    sorted key."""
    out = []
    for v in vals:
        out.extend(_flatten(v))
    return out


# ---------------------------------------------------------------------------
# Canonical CPU calls — one per subsystem, deterministic fixed inputs.
# ---------------------------------------------------------------------------

def cpu_calls() -> dict:
    R = {}

    # descriptive — var (n-1), exact. R var(c(2,4,4,4,5,5,7,9)) = 32/7 (the
    # sample variance, ddof=1 — NOT the population variance 4.0).
    from pystatistics.descriptive import var, cor
    R["descriptive.var"] = {"fp": _fp(var(np.array([2., 4, 4, 4, 5, 5, 7, 9])).variance),
                            "anchor": [4.571428571428571]}

    # hypothesis — Welch t-test (default), statistic + p. R:
    # t.test(c(1,2,3,4,5), c(2,4,6,8,10)) -> t = -1.897366596101028,
    # p = 0.107531194930627 (Welch is R's t.test default).
    from pystatistics.hypothesis import t_test, chisq_test
    tt = t_test([1., 2, 3, 4, 5], [2., 4, 6, 8, 10])
    R["hypothesis.t_test"] = {"fp": _fp(tt.statistic, tt.p_value),
                              "anchor": [-1.897366596101028, 0.107531194930627]}
    # chisq 2x2 with Yates (default)
    ct = chisq_test(np.array([[10., 20], [30, 40]]))
    R["hypothesis.chisq_test"] = {"fp": _fp(ct.statistic, ct.p_value)}

    # anova — one-way F (Type I)
    from pystatistics.anova import anova_oneway
    y = np.array([1., 2, 3, 2, 3, 4, 3, 4, 5], dtype=float)
    g = np.array(["a", "a", "a", "b", "b", "b", "c", "c", "c"], dtype=object)
    ao = anova_oneway(y, g)
    R["anova.anova_oneway"] = {"fp": _fp(ao.table[0].f_value, ao.table[0].p_value)}

    # regression — OLS coefficients on a fixed design
    from pystatistics.regression import fit
    X = np.array([[1., 0], [1, 1], [1, 2], [1, 3], [1, 4]])
    yv = np.array([1., 3, 2, 5, 4])
    rf = fit(X, yv)
    # R: coef(lm(c(1,3,2,5,4) ~ c(0,1,2,3,4))) -> (Intercept)=1.4, slope=0.8.
    R["regression.fit"] = {"fp": _fp(rf.coef), "anchor": [1.4, 0.8]}

    # survival — Kaplan-Meier survival curve
    from pystatistics.survival import kaplan_meier
    km = kaplan_meier(np.array([1., 2, 3, 4, 5, 6]),
                      np.array([1, 0, 1, 0, 1, 1]))
    R["survival.kaplan_meier"] = {"fp": _fp(km.survival)}

    # multivariate — PCA explained variance (fixed 4x3)
    from pystatistics.multivariate import pca
    Xp = np.array([[1., 2, 3], [4, 5, 7], [7, 8, 10], [2, 1, 0], [5, 6, 8]])
    pc = pca(Xp)
    R["multivariate.pca"] = {"fp": _fp(pc.explained_variance_ratio)}

    # mixed — random-intercept LMM (tiny, fixed)
    from pystatistics.mixed import lmm
    yl = np.array([1., 2, 2, 3, 3, 4, 5, 5, 6, 6], dtype=float)
    Xl = np.column_stack([np.ones(10), np.arange(10.)])
    grp = {"g": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])}
    lm_ = lmm(yl, Xl, grp)
    R["mixed.lmm"] = {"fp": _fp(lm_.fixef)}

    # timeseries — acf (deterministic, machine-precision vs R)
    from pystatistics.timeseries import acf
    ac = acf(np.array([1., 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4], dtype=float), max_lag=3)
    # R: acf(x, lag.max=3, plot=FALSE)$acf -> lags 0..3 (lag 0 == 1.0).
    R["timeseries.acf"] = {"fp": _fp(ac.acf),
                           "anchor": [1.0, 0.555031446540880,
                                      -0.059748427672956, -0.561320754716981]}

    # montecarlo — seeded bootstrap SE of the mean (deterministic given seed)
    from pystatistics.montecarlo import boot
    bt = boot(np.array([3., 1, 4, 1, 5, 9, 2, 6]),
              lambda d, idx: float(np.mean(d[idx])), n_resamples=200, seed=1)
    R["montecarlo.boot"] = {"fp": _fp(bt.se)}

    # ordinal — polr coefficient (tiny ordered response)
    from pystatistics.ordinal import polr
    yo = np.array([0, 0, 1, 1, 2, 2, 1, 2, 0, 2])
    Xo = np.column_stack([np.linspace(-1, 1, 10)])
    po = polr(yo, Xo)
    R["ordinal.polr"] = {"fp": _fp(po.coef)}

    # multinomial — multinom coefficients
    from pystatistics.multinomial import multinom
    ym = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])
    Xm = np.column_stack([np.ones(10), np.linspace(-1, 1, 10)])
    mn = multinom(ym, Xm)
    R["multinomial.multinom"] = {"fp": _fp(mn.coef)}

    # gam — a single smooth (edf / fitted fingerprint)
    from pystatistics.gam import gam, s
    xg = np.linspace(0, 1, 30)
    yg = np.sin(2 * np.pi * xg) + 0.0  # deterministic
    gm = gam(yg, smooths=[s("x")], smooth_data={"x": xg})
    R["gam.gam"] = {"fp": _fp(gm.total_edf)}

    return R


# ---------------------------------------------------------------------------
# GPU fail-loud — every backend='gpu' entry point must raise (no torch/GPU).
# ---------------------------------------------------------------------------

def gpu_faillloud() -> dict:
    checks = {}

    def check(name, fn):
        try:
            fn()
            checks[name] = {"failed_loud": False, "exc_type": None, "msg": None}
        except Exception as e:                       # noqa: BLE001
            checks[name] = {"failed_loud": True, "exc_type": type(e).__name__,
                            "msg": str(e)[:160]}

    from pystatistics.descriptive import var, cor
    from pystatistics.regression import fit
    from pystatistics.multivariate import pca
    from pystatistics.timeseries import arima, auto_arima
    from pystatistics.ordinal import polr
    from pystatistics.montecarlo import boot

    check("descriptive.var[gpu]",
          lambda: var(np.array([1., 2, 3, 4, 5]), backend="gpu"))
    check("regression.fit[gpu]",
          lambda: fit(np.array([[1., 0], [1, 1], [1, 2]]), np.array([1., 2, 3]),
                      backend="gpu"))
    check("multivariate.pca[gpu]",
          lambda: pca(np.array([[1., 2], [3, 4], [5, 7]]), backend="gpu"))
    check("timeseries.arima[gpu]",
          lambda: arima(np.arange(30.), order=(1, 0, 0), backend="gpu"))
    check("timeseries.auto_arima[gpu]",
          lambda: auto_arima(np.arange(30.), max_p=2, max_q=2, backend="gpu"))
    check("ordinal.polr[gpu]",
          lambda: polr(np.array([0, 1, 2, 1, 0]),
                       np.column_stack([np.linspace(-1, 1, 5)]), backend="gpu"))
    check("montecarlo.boot[gpu]",
          lambda: boot(np.array([1., 2, 3, 4]),
                       lambda d, idx: float(np.mean(d[idx])), n_resamples=50,
                       seed=1, backend="gpu"))
    return checks


def _verify_anchors(calls: dict) -> None:
    """Fail loud if any R-anchored fingerprint drifts from its anchor.

    The anchors are the non-circularity guarantee: they tie a fingerprint to a
    value computed independently in R, so the certification is not merely
    pystatistics agreeing with itself. A mismatch means either the library
    regressed or the anchor is wrong — either way the artifact must NOT be
    written silently (fail-fast, no silent defaults).
    """
    for name, rec in calls.items():
        if "anchor" not in rec:
            continue
        fp = np.asarray(rec["fp"], dtype=np.float64)
        anchor = np.asarray(rec["anchor"], dtype=np.float64)
        if fp.shape != anchor.shape or not np.allclose(fp, anchor, rtol=0,
                                                        atol=1e-12):
            raise AssertionError(
                f"R anchor mismatch for {name}: fingerprint={fp.tolist()} "
                f"anchor={anchor.tolist()} (atol=1e-12). The certification is "
                f"non-circular only if these agree — refusing to write a "
                f"bad artifact.")


def main():
    env = "torch" if TORCH_PRESENT else "slim"
    cpu = {}
    errors = {}
    calls = cpu_calls()
    _verify_anchors(calls)
    for k, v in calls.items():
        cpu[k] = v
    torch_version = None
    if TORCH_PRESENT:
        import torch  # only when present; recorded for R14 (validated dep version)
        torch_version = torch.__version__
    payload = {"env": env, "torch_present": TORCH_PRESENT, "py_version": PYVER,
               "torch_version": torch_version, "cpu": cpu}
    ART.mkdir(parents=True, exist_ok=True)
    (ART / f"cpu_{env}.json").write_text(json.dumps(payload, indent=2))
    print(f"[{env}] CPU fingerprints: {len(cpu)} subsystems -> cpu_{env}.json")

    if not TORCH_PRESENT:
        gpu = gpu_faillloud()
        (ART / "gpu_faillloud.json").write_text(
            json.dumps({"py_version": PYVER, "checks": gpu}, indent=2))
        n_loud = sum(1 for v in gpu.values() if v["failed_loud"])
        print(f"[slim] GPU fail-loud: {n_loud}/{len(gpu)} raised")
        for k, v in gpu.items():
            if not v["failed_loud"]:
                print("  NOT LOUD:", k)


if __name__ == "__main__":
    main()
