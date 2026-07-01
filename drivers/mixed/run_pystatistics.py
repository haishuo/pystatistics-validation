"""Run and time a pystatistics LMM fit -> a canonical validation record.

One job: fit ``pystatistics.mixed.lmm`` on a :class:`MixedDataset` under the shared
device-aware timing protocol, and return one ``validation-run/v1`` record carrying
every estimate a coefficient-/variance-component-/BLUP-for-the-same comparison
against R needs, plus the fit-level summary.

Consumes the PyPI pystatistics library; never modifies it. Timing and the record
envelope come from the shared harness ``pystatsval``. ``mixed`` is CPU-only (no
GPU backend exists — see the report's priority-4 section), so the device is always
``cpu`` and the precision fp64.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.measure import measure
from pystatsval.record import make_record


def _as_list(v: Any) -> list[float] | None:
    if v is None:
        return None
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


def _var_components(sol) -> list[dict[str, Any]]:
    """Flatten LMMSolution.var_components + the residual into a comparable list.

    Each entry: {group, name, variance, std_dev, corr}. The residual variance
    (lme4's 'Residual' row) is appended as group='Residual' so the renderer
    pairs it with R's residual variance.
    """
    out: list[dict[str, Any]] = []
    for vc in sol.var_components:
        out.append({
            "group": str(vc.group), "name": str(vc.name),
            "variance": float(vc.variance), "std_dev": float(vc.std_dev),
            "corr": (None if vc.corr is None else float(vc.corr)),
        })
    out.append({
        "group": "Residual", "name": "", "corr": None,
        "variance": float(sol.params.residual_variance),
        "std_dev": float(sol.params.residual_std),
    })
    return out


def _blups(sol) -> dict[str, list[list[float]]]:
    """Conditional modes (BLUPs) per grouping factor as a (n_groups x q) matrix."""
    out: dict[str, list[list[float]]] = {}
    for grp, arr in sol.ranef.items():
        a = np.asarray(arr, dtype=float)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        out[grp] = a.tolist()
    return out


def run_lmm_record(
    ds, *, repeats: int = 7, warmup: int = 1,
    compute_satterthwaite: bool = True,
) -> dict[str, Any]:
    """Fit + time ``lmm(...)`` on a :class:`MixedDataset` (CPU, fp64).

    On failure a record with timing nulled and an ``error`` field is returned
    rather than raising — a sweep records the failure (e.g. a fail-loud refusal)
    and continues.
    """
    from pystatistics.mixed import lmm

    n, p = ds.n, ds.p
    engine = "pystatistics:cpu"

    def _call():
        return lmm(ds.y, ds.X, groups=ds.groups,
                   random_effects=ds.random_effects,
                   random_data=ds.random_data, reml=ds.reml,
                   compute_satterthwaite=compute_satterthwaite)

    try:
        wall, sol = measure(_call, device="cpu", repeats=repeats, warmup=warmup)
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        return make_record(
            engine=engine, dataset=ds.key, n=n, p=p, wall=None,
            backend_name="lmm_cpu",
            extra={"analysis": "lmm", "reml": ds.reml,
                   "error": f"{type(exc).__name__}: {exc}"})

    # Capture any Python warning the library emits (R10: lme4 emits a
    # 'boundary (singular) fit' MESSAGE on degenerate fits; record whether
    # pystatistics emits anything comparable — direct evidence for finding F1).
    with _warnings.catch_warnings(record=True) as wlist:
        _warnings.simplefilter("always")
        _call()
    py_warnings = [str(w.message) for w in wlist]

    summary: dict[str, Any] = {
        "coefficients": _as_list(sol.coefficients),
        "standard_errors": _as_list(sol.standard_errors),
        "t_values": _as_list(sol.t_values),
        "df_satterthwaite": (_as_list(sol.df_satterthwaite)
                             if compute_satterthwaite else None),
        "p_values": (_as_list(sol.p_values) if compute_satterthwaite else None),
        "coef_names": list(ds.fixed_names),
        "var_components": _var_components(sol),
        "blups": _blups(sol),
        "log_likelihood": float(sol.log_likelihood),
        "aic": float(sol.aic),
        "bic": float(sol.bic),
        "reml_criterion": float(-2.0 * sol.log_likelihood),
    }
    # As of 4.5.0 the library exposes LMMSolution.is_singular (finding F1 fix);
    # record it as the authoritative flag, keep the driver-side detector as a
    # cross-check, and keep any emitted warning (the boundary RuntimeWarning).
    lib_is_singular = getattr(sol, "is_singular", None)
    extra: dict[str, Any] = {
        "analysis": "lmm", "reml": ds.reml,
        "satterthwaite": compute_satterthwaite,
        "converged": bool(sol.converged),
        "is_singular": (bool(lib_is_singular) if lib_is_singular is not None
                        else _is_singular(sol)),
        "lib_is_singular": (None if lib_is_singular is None else bool(lib_is_singular)),
        "driver_detected_singular": _is_singular(sol),
        "py_warnings": py_warnings,
    }
    return make_record(
        engine=engine, dataset=ds.key, n=n, p=p, wall=wall,
        backend_name="lmm_cpu", precision="fp64",
        loglik=float(sol.log_likelihood), n_iter=int(sol.n_iter),
        converged=bool(sol.converged), summary=summary, extra=extra)


def _is_singular(sol) -> bool:
    """True if the fitted RE covariance is at the boundary -- a variance collapsed
    to ~0 OR an intercept/slope correlation driven to +-1 (a rank-deficient 2x2
    block). This is a VALIDATION-side detector on the outputs; the LIBRARY itself
    exposes no singular flag and emits no warning (finding F1) -- the contrast
    between this (the fit IS singular) and ``py_warnings`` (empty) is the
    evidence. Mirrors lme4's isSingular() semantics."""
    eps = 1e-6
    for vc in sol.var_components:
        if float(vc.variance) <= eps:
            return True
        if vc.corr is not None and abs(float(vc.corr)) >= 1.0 - 1e-4:
            return True
    return False


def strip_arrays(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the bulky per-fit arrays for timing/scaling records (the renderer
    reads the scalar summary, never the BLUP/coef arrays)."""
    drop = ("blups", "var_components", "coefficients", "standard_errors",
            "t_values", "df_satterthwaite", "p_values")
    return [{k: v for k, v in r.items() if k not in drop} for r in records]
