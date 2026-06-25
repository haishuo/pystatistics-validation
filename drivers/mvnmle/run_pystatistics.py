"""Run and time the pystatistics MVN MLE on a curated problem.

One job: fit ``pystatistics.mvnmle.mlest`` on a given matrix with a given backend
(``'cpu'`` = inverse-Cholesky FP64, ``'gpu'`` = forward-Cholesky FP32) under the
shared timing protocol, and return a uniform benchmark record in the canonical
validation-run schema.

This module *consumes* the pystatistics library; it never modifies it. The timing,
record envelope, and covariance summary come from the shared harness ``pystatsval``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.estimates import summarize_covariance
from pystatsval.record import make_record
from pystatsval.timing import time_call


def run_pystatistics(X: NDArray[np.floating],
                     *,
                     backend: str,
                     survey: str,
                     n_patterns: int,
                     missing_frac: float,
                     reps: int = 5,
                     warmup: int = 1,
                     max_iter: int | None = None,
                     tol: float | None = None) -> dict[str, Any]:
    """Fit + time mlest on ``X`` for ``backend`` ('cpu' or 'gpu').

    On any failure (no GPU, optimiser error) a record with the covariance summary
    nulled and an ``error`` field is returned rather than raising — the sweep
    records the failure and moves on.
    """
    from pystatistics.mvnmle import mlest

    n, p = X.shape
    # CPU path is FP64; GPU casts internally. Pass FP64 in so the input is
    # identical across engines and only the backend differs.
    data = np.asarray(X, dtype=np.float64)

    def call():
        return mlest(data, backend=backend, algorithm="direct",
                     max_iter=max_iter, tol=tol)

    engine = f"pystatistics:{backend}"
    common = {"n_patterns": int(n_patterns), "missing_frac": float(missing_frac)}
    try:
        wall, result = time_call(call, warmup=warmup, reps=reps)
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        return make_record(
            engine=engine, dataset=survey, n=n, p=p, wall=None,
            summary=summarize_covariance(None),
            extra={**common, "error": f"{type(exc).__name__}: {exc}"})

    info = dict(result.info) if result.info else {}
    mu = (np.asarray(result.muhat, float).tolist()
          if result.muhat is not None else None)
    return make_record(
        engine=engine, dataset=survey, n=n, p=p,
        loglik=result.loglik, n_iter=result.n_iter, converged=result.converged,
        wall=wall, backend_name=result.backend_name,
        precision=info.get("precision"),
        parameterization=info.get("parameterization"),
        summary=summarize_covariance(result.sigmahat),
        extra={
            **common,
            "mu": mu,
            "n_function_evals": info.get("n_function_evals"),
            "n_gradient_evals": info.get("n_gradient_evals"),
            "internal_total_s": (result.timing or {}).get("total_seconds"),
            "max_iter": max_iter,
            "tol": tol,
        },
    )
