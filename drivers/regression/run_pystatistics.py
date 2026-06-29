"""Run and time a pystatistics regression fit → a canonical validation record.

One job: fit ``pystatistics.regression.fit`` on a numeric design for a given family
and backend, under the shared device-aware timing protocol, and return one
``validation-run/v1`` record carrying the full estimate vectors (so the study
generator can compare coefficient-for-coefficient against R) plus the model-level
fit summary.

Consumes the pystatistics library; never modifies it. Timing/memory and the record
envelope come from the shared harness ``pystatsval``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.measure import measure
from pystatsval.record import make_record

# Families that take the OLS LinearSolution path (family=None), not IRLS.
_OLS_FAMILIES = {"gaussian", "ols", "lm", None}


def _as_list(v: Any) -> list[float] | None:
    if v is None:
        return None
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


def run_regression_record(
    X: NDArray[np.floating], y: NDArray[np.floating], *,
    family: str | None,
    backend: str,
    dataset: str,
    model_key: str,
    link: str | None = None,
    repeats: int = 3,
    warmup: int = 0,
) -> dict[str, Any]:
    """Fit + time ``fit(design, family=..., backend=...)`` on design ``X``.

    ``X`` includes a leading intercept column (column 0 = ones), matching what
    ``Design.from_arrays`` expects. ``family`` is a pystatistics family string
    (``'binomial'``/``'poisson'``/``'gamma'``/``'negative.binomial'``) or one of
    ``gaussian``/``None`` for OLS. ``link`` pins a non-default link where the
    reference uses one (Gamma is fit with ``log`` to match the canonical target;
    the pystatistics ``'gamma'`` string would otherwise default to ``inverse``).
    ``backend`` is ``'cpu'`` or ``'gpu'``.

    On failure a record with timing nulled and an ``error`` field is returned rather
    than raising — a sweep records the failure and continues.
    """
    from pystatistics.regression import Design, GammaFamily, fit

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    device = "gpu" if backend == "gpu" else "cpu"
    engine = f"pystatistics:{backend}"
    is_ols = family in _OLS_FAMILIES
    # Build the family argument, pinning the link where the reference needs one.
    if is_ols:
        fam_arg = None
    elif family == "gamma":
        fam_arg = GammaFamily(link=link or "log")
    else:
        fam_arg = family

    design = Design.from_arrays(X, y)

    def _call():
        return fit(design, family=fam_arg, backend=backend)

    try:
        wall, sol = measure(_call, device=device, repeats=repeats, warmup=warmup)
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        return make_record(
            engine=engine, dataset=dataset, n=n, p=p, wall=None,
            backend_name=f"{model_key}_{backend}",
            extra={"model_key": model_key, "error": f"{type(exc).__name__}: {exc}"})

    precision = "fp64" if backend == "cpu" else "fp32"
    summary: dict[str, Any] = {
        "coefficients": _as_list(sol.coefficients),
        "standard_errors": _as_list(sol.standard_errors),
        "p_values": _as_list(sol.p_values),
        "coef_names": list(getattr(sol, "coef", []) or []),
    }
    if is_ols:
        summary.update(
            test_statistic=_as_list(sol.t_values),
            stat_name="t value",
            r_squared=float(sol.r_squared),
            adjusted_r_squared=float(sol.adjusted_r_squared),
            residual_std_error=float(sol.residual_std_error),
            fitted_first10=_as_list(sol.fitted_values[:10]),
            residuals_first10=_as_list(sol.residuals[:10]),
        )
    else:
        summary.update(
            test_statistic=_as_list(sol.z_values),
            stat_name="z value",
            deviance=float(sol.deviance),
            null_deviance=float(sol.null_deviance),
            aic=float(sol.aic),
            bic=float(sol.bic),
            dispersion=float(sol.dispersion),
        )
        # Negative binomial exposes the auto-estimated dispersion theta (4.3.1+).
        info = getattr(sol, "info", None) or {}
        if info.get("theta") is not None:
            summary["theta"] = float(info["theta"])
            summary["theta_estimated"] = bool(info.get("theta_estimated", True))

    peak = wall.get("peak_mem_bytes")
    extra: dict[str, Any] = {
        "model_key": model_key,
        "family": "gaussian" if is_ols else family,
        "peak_mem_mb": (round(peak / 1e6, 1) if peak else None),
    }
    if not is_ols:
        extra["n_iter"] = int(getattr(sol, "n_iter", 0) or 0)

    return make_record(
        engine=engine, dataset=dataset, n=n, p=p, wall=wall,
        backend_name=f"{model_key}_{backend}", precision=precision,
        converged=(None if is_ols else bool(getattr(sol, "converged", True))),
        n_iter=(None if is_ols else int(getattr(sol, "n_iter", 0) or 0)),
        summary=summary, extra=extra)
