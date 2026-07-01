"""Run and time a pystatistics GLMM fit -> a canonical validation record.

One job: fit ``pystatistics.mixed.glmm`` on a :class:`GLMMDataset` under the
shared device-aware timing protocol, and return one ``validation-run/v1`` record
carrying every estimate a coefficient-/SE-/variance-component-/BLUP-for-the-same
comparison against ``lme4::glmer`` (Laplace, nAGQ=1) needs, plus the fit-level
summary.

Consumes the PyPI pystatistics library; never modifies it. ``mixed`` is CPU-only
(no GPU backend exists for the general GLMM — see the report), so the device is
always ``cpu`` and the precision fp64.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Any

import numpy as np

from pystatsval.measure import measure
from pystatsval.record import make_record


def _as_list(v: Any) -> list[float] | None:
    if v is None:
        return None
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


def family_arg(ds):
    """Resolve the GLMMDataset's (family, link) into the ``family`` argument
    glmm() expects: a plain string when the link is the family default, else a
    Family instance carrying the non-default link (e.g. Binomial('probit'))."""
    from pystatistics.regression.families import (
        Binomial, Poisson, Gaussian,
    )
    fam, link = ds.family, ds.link
    defaults = {"binomial": "logit", "poisson": "log",
                "gaussian": "identity", "gamma": "inverse"}
    cls = {"binomial": Binomial, "poisson": Poisson, "gaussian": Gaussian}
    # Default link, or a family with no non-default-link constructor here (e.g.
    # gamma) → pass the string and let glmm() resolve (and, for a free-dispersion
    # family, fail loud).
    if defaults.get(fam) == link or fam not in cls:
        return fam
    return cls[fam](link=link)


def _var_components(sol) -> list[dict[str, Any]]:
    """Flatten GLMMSolution.var_components into a comparable list. GLMM has no
    residual-variance row (dispersion is fixed at 1)."""
    out: list[dict[str, Any]] = []
    for vc in sol.var_components:
        out.append({
            "group": str(vc.group), "name": str(vc.name),
            "variance": float(vc.variance), "std_dev": float(vc.std_dev),
            "corr": (None if vc.corr is None else float(vc.corr)),
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


def run_glmm_record(ds, *, repeats: int = 7, warmup: int = 1) -> dict[str, Any]:
    """Fit + time ``glmm(...)`` on a :class:`GLMMDataset` (CPU, fp64).

    On failure a record with timing nulled and an ``error`` field is returned
    rather than raising — a sweep records the failure (e.g. a fail-loud refusal)
    and continues.
    """
    from pystatistics.mixed import glmm

    n, p = ds.n, ds.p
    engine = "pystatistics:cpu"
    fam = family_arg(ds)

    def _call():
        return glmm(ds.y, ds.X, groups=ds.groups,
                    family=fam,
                    random_effects=ds.random_effects,
                    random_data=ds.random_data)

    try:
        wall, sol = measure(_call, device="cpu", repeats=repeats, warmup=warmup)
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        return make_record(
            engine=engine, dataset=ds.key, n=n, p=p, wall=None,
            backend_name="glmm_cpu",
            extra={"analysis": "glmm", "family": ds.family, "link": ds.link,
                   "error": f"{type(exc).__name__}: {exc}"})

    # Capture any Python warning the library emits (R10: does glmm flag boundary
    # / non-convergence the way glmer does?).
    with _warnings.catch_warnings(record=True) as wlist:
        _warnings.simplefilter("always")
        _call()
    py_warnings = [str(w.message) for w in wlist]

    summary: dict[str, Any] = {
        "coefficients": _as_list(sol.coefficients),
        "standard_errors": _as_list(sol.standard_errors),
        "z_values": _as_list(sol.z_values),
        "p_values": _as_list(sol.p_values),
        "coef_names": list(ds.fixed_names),
        "var_components": _var_components(sol),
        "blups": _blups(sol),
        "log_likelihood": float(sol.log_likelihood),
        "deviance": float(sol.deviance),
        "aic": float(sol.aic),
        "bic": float(sol.bic),
    }
    extra: dict[str, Any] = {
        "analysis": "glmm", "family": ds.family, "link": ds.link,
        "converged": bool(sol.converged),
        "py_warnings": py_warnings,
    }
    return make_record(
        engine=engine, dataset=ds.key, n=n, p=p, wall=wall,
        backend_name="glmm_cpu", precision="fp64",
        loglik=float(sol.log_likelihood), n_iter=int(sol.n_iter),
        converged=bool(sol.converged), summary=summary, extra=extra)
