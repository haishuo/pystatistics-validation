"""Run and time a pystatistics grm_lmm fit -> a canonical validation record.

One job: fit ``pystatistics.mixed.grm_lmm`` on a :class:`GRMDataset` for a given
backend under the shared device-aware timing protocol, and return one
``validation-run/v1`` record with the estimates (β, SEs, variance components,
heritability, genetic-value BLUPs, logLik) + fit summary, comparable against the
rrBLUP reference and the CPU fp64 optimum.

Consumes the PyPI library; never modifies it. On the GPU paths the CF-1 fp32
conditioning gate may REFUSE (``NumericalError``) — that is a fail-loud result, not
an error to hide, so it is recorded (with ``refused=True``) and the sweep continues.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.measure import measure
from pystatsval.record import make_record


def _as_list(v: Any) -> list[float] | None:
    if v is None:
        return None
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


def run_grm_record(
    ds, *, backend: str, device: str = "cpu", force: bool = False,
    repeats: int = 5, warmup: int = 1,
) -> dict[str, Any]:
    """Fit + time ``grm_lmm(y, X, W, backend=...)`` on a :class:`GRMDataset`."""
    from pystatistics.mixed import grm_lmm
    from pystatistics.core.exceptions import NumericalError

    n, p, M = ds.n, ds.p, ds.M
    engine = f"pystatistics:{backend}"

    def _call():
        return grm_lmm(ds.y, ds.X, ds.W, backend=backend, reml=ds.reml, force=force)

    try:
        wall, sol = measure(_call, device=device, repeats=repeats, warmup=warmup)
    except NumericalError as exc:  # CF-1 gate refusal — a fail-loud result
        return make_record(
            engine=engine, dataset=ds.key, n=n, p=p, wall=None,
            backend_name=f"grm_{backend}",
            extra={"analysis": "grm", "backend": backend, "refused": True,
                   "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001
        return make_record(
            engine=engine, dataset=ds.key, n=n, p=p, wall=None,
            backend_name=f"grm_{backend}",
            extra={"analysis": "grm", "backend": backend, "refused": False,
                   "error": f"{type(exc).__name__}: {exc}"})

    precision = "fp64" if backend in ("cpu", "gpu_fp64") else "fp32"
    summary: dict[str, Any] = {
        "coefficients": _as_list(sol.coefficients),
        "standard_errors": _as_list(sol.standard_errors),
        "coef_names": list(ds.fixed_names),
        "var_genetic": float(sol.var_genetic),
        "var_residual": float(sol.var_residual),
        "heritability": float(sol.heritability),
        "variance_ratio": float(sol.variance_ratio),
        "genetic_values": _as_list(sol.genetic_values),
        "log_likelihood": float(sol.log_likelihood),
        "aic": float(sol.aic),
        "bic": float(sol.bic),
    }
    extra: dict[str, Any] = {
        "analysis": "grm", "backend": backend, "refused": False,
        "M": M, "true_h2": ds.true_h2,
        "converged": bool(getattr(sol, "converged", True)),
    }
    return make_record(
        engine=engine, dataset=ds.key, n=n, p=p, wall=wall,
        backend_name=f"grm_{backend}", precision=precision,
        loglik=float(sol.log_likelihood),
        converged=bool(getattr(sol, "converged", True)),
        summary=summary, extra=extra)
