"""Run and time the discrete-time GPU path → canonical validation records.

One job: benchmark the ONE GPU-capable survival path — ``discrete_time``, which
forwards ``backend=`` to ``pystatistics.regression.fit(family='binomial')`` over
the person-period expansion — under the shared device-aware timing protocol.

Two granularities of measurement, deliberately separated:

- :func:`fit_pp_glm_record` times ``regression.fit(family='binomial', backend=)``
  on a PRE-BUILT person-period design. This isolates the GLM FIT — exactly the
  operation the GPU accelerates — from the (device-independent, pure-Python)
  person-period expansion. It is the device-pivot speedup measurement. That this
  reproduces the public ``discrete_time`` fit is verified in the correctness
  study (the covariate coefficients must match to round-off).
- :func:`discrete_time_e2e_record` times the public ``discrete_time(backend=)``
  END TO END on the original survival data — the "does the real GPU path break"
  measurement, expansion included.

Both record failures (a loud ``NumericalError`` from the float32 acceptance gate,
or a ``solve_failed``) as a record with ``error`` set rather than raising, so a
sweep documents the boundary and continues. :func:`forced_fp32_accuracy` applies
the RIGOR R9 protocol: when the gate rejects a float32 fit, force it and report
its true accuracy vs the fp64 reference, so a rejection is recorded as
genuinely-wrong (correct fail-loud) rather than assumed-broken.

Consumes the pystatistics library; never modifies it.
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


def _precision(backend: str) -> str:
    return {"cpu": "fp64", "gpu": "fp32", "gpu_fp64": "fp64"}[backend]


def _device(backend: str) -> str:
    return "cpu" if backend == "cpu" else "gpu"


def fit_pp_glm_record(
    X_pp: NDArray, y_pp: NDArray, *, n_cov: int, backend: str,
    dataset: str, repeats: int = 5, warmup: int = 1,
) -> dict[str, Any]:
    """Fit + time ``regression.fit(binomial, backend=)`` on a person-period design.

    ``X_pp`` is ``[interval dummies | covariates]`` (no intercept). Records the
    full wall/peak-memory, the covariate coefficient tail (the last ``n_cov``
    entries — the discrete-time hazard-ratio covariates), and convergence. The
    ``engine`` is ``pystatistics:cpu`` / ``pystatistics:gpu`` so the renderer's
    ``device_pivot`` pairs CPU against the accelerator; the exact backend
    (``gpu`` fp32 vs ``gpu_fp64``) lives in ``precision`` / ``backend_name``.
    """
    from pystatistics.regression import Design, fit

    X_pp = np.asarray(X_pp, dtype=np.float64)
    y_pp = np.asarray(y_pp, dtype=np.float64).ravel()
    n, p = X_pp.shape
    engine = "pystatistics:cpu" if backend == "cpu" else "pystatistics:gpu"
    design = Design.from_arrays(X_pp, y_pp)

    def _call():
        return fit(design, family="binomial", backend=backend)

    try:
        wall, sol = measure(_call, device=_device(backend), repeats=repeats,
                            warmup=warmup)
    except Exception as exc:  # noqa: BLE001 - record the loud failure, continue
        return make_record(
            engine=engine, dataset=dataset, n=n, p=p, wall=None,
            backend_name=f"discrete_glm_{backend}", precision=_precision(backend),
            extra={"procedure": "discrete_time_fit", "backend": backend,
                   "n_cov": n_cov, "error": f"{type(exc).__name__}: {exc}"})

    coef = np.asarray(sol.coefficients, float).ravel()
    peak = wall.get("peak_mem_bytes")
    return make_record(
        engine=engine, dataset=dataset, n=n, p=p, wall=wall,
        backend_name=f"discrete_glm_{backend}", precision=_precision(backend),
        converged=bool(getattr(sol, "converged", True)),
        n_iter=int(getattr(sol, "n_iter", 0) or 0),
        summary={"covariate_coefficients": _as_list(coef[-n_cov:]),
                 "all_coefficients": _as_list(coef)},
        extra={"procedure": "discrete_time_fit", "backend": backend,
               "n_cov": n_cov,
               "peak_mem_mb": (round(peak / 1e6, 1) if peak else None)})


def forced_fp32_accuracy(X_pp: NDArray, y_pp: NDArray, cpu_coef: NDArray, *,
                         backend: str = "gpu") -> dict[str, Any]:
    """RIGOR R9: force the (possibly rejected) float32 fit and score it vs fp64.

    Returns ``{forced_max_abs, forced_max_rel}`` over the full coefficient vector,
    so a gate rejection can be recorded as genuinely-inaccurate (correct
    fail-loud) rather than presumed-broken. Returns ``None`` values if even the
    forced fit raises (a true solve breakdown).
    """
    from pystatistics.regression import Design, fit

    cpu_coef = np.asarray(cpu_coef, float).ravel()
    design = Design.from_arrays(np.asarray(X_pp, float), np.asarray(y_pp, float).ravel())
    try:
        sol = fit(design, family="binomial", backend=backend, force=True)
    except Exception as exc:  # noqa: BLE001
        return {"forced_max_abs": None, "forced_max_rel": None,
                "forced_error": f"{type(exc).__name__}: {exc}"}
    g = np.asarray(sol.coefficients, float).ravel()
    absd = np.abs(g - cpu_coef)
    rel = absd / np.maximum(np.abs(cpu_coef), 1e-12)
    return {"forced_max_abs": float(absd.max()),
            "forced_max_rel": float(rel.max()), "forced_error": None}


def discrete_time_e2e_record(
    time: NDArray, event: NDArray, X: NDArray, names: list[str],
    interval_bounds: NDArray, *, backend: str, dataset: str,
    repeats: int = 3, warmup: int = 1,
) -> dict[str, Any]:
    """Fit + time the PUBLIC ``discrete_time(backend=)`` end to end on real data.

    Expansion + fit together — the "does the real GPU path break on a real,
    non-synthetic dataset" measurement. Records the covariate coefficients,
    ``person_period_n``, and on a loud failure the ``error`` (so the boundary is
    documented rather than crashing the sweep).
    """
    from pystatistics.survival import discrete_time

    engine = "pystatistics:cpu" if backend == "cpu" else "pystatistics:gpu"

    def _call():
        return discrete_time(time, event, X, names=names,
                             intervals=interval_bounds, backend=backend)

    try:
        wall, sol = measure(_call, device=_device(backend), repeats=repeats,
                            warmup=warmup)
    except Exception as exc:  # noqa: BLE001
        return make_record(
            engine=engine, dataset=dataset, wall=None,
            backend_name=f"discrete_time_{backend}", precision=_precision(backend),
            extra={"procedure": "discrete_time_e2e", "backend": backend,
                   "error": f"{type(exc).__name__}: {exc}"})

    peak = wall.get("peak_mem_bytes")
    return make_record(
        engine=engine, dataset=dataset, n=int(sol.person_period_n),
        p=len(sol.coefficients), wall=wall,
        backend_name=f"discrete_time_{backend}", precision=_precision(backend),
        summary={"covariate_coefficients": _as_list(sol.coefficients)},
        extra={"procedure": "discrete_time_e2e", "backend": backend,
               "n_intervals": int(sol.n_intervals),
               "person_period_n": int(sol.person_period_n),
               "peak_mem_mb": (round(peak / 1e6, 1) if peak else None)})
