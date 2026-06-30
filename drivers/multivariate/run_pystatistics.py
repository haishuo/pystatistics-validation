"""Run and time a pystatistics PCA / factor-analysis fit -> a canonical record.

One job: fit ``pystatistics.multivariate.pca`` / ``factor_analysis`` on a numeric
matrix for a given backend, under the shared device-aware timing protocol, and
return one ``validation-run/v1`` record carrying the full estimate arrays (so the
study generator can compare eigenvalue-for-eigenvalue / loading-for-loading
against R) plus the analysis-level summary.

Consumes the pystatistics library; never modifies it. Timing/memory and the
record envelope come from the shared harness ``pystatsval``.
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


# Bulky per-fit arrays that scale with n or p — kept in correctness records (small
# data) but stripped from timing/scaling records before freezing (a 1e5 x k scores
# matrix would bloat the artifact by hundreds of MB; the renderer reads the scalar
# summary CSV, never these).
_BULKY_RECORD_KEYS = ("scores", "rotation", "loadings")


def strip_arrays(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records with the bulky n-/p-scaled arrays removed (timing studies)."""
    return [{k: v for k, v in r.items() if k not in _BULKY_RECORD_KEYS}
            for r in records]


def _as_matrix(v: Any) -> list[list[float]] | None:
    if v is None:
        return None
    return np.asarray(v, dtype=float).tolist()


def run_pca_record(
    X: NDArray[np.floating], *,
    backend: str,
    dataset: str,
    center: bool = True,
    scale: bool = False,
    n_components: int | None = None,
    solver: str | None = None,
    seed: int = 0,
    device: str = "cpu",
    repeats: int = 7,
    warmup: int = 1,
) -> dict[str, Any]:
    """Fit + time ``pca(X, backend=..., center=, scale=, ...)`` on matrix ``X``.

    ``device`` selects the timing protocol ('cpu'/'mps'/'cuda'); ``backend`` is
    the pystatistics backend string ('cpu'/'gpu'/'gpu_fp64'). For the GPU paths
    the matrix is handed in as a numpy array (the gateway streams it to device).

    On failure a record with timing nulled and an ``error`` field is returned
    rather than raising — a sweep records the failure (e.g. a fail-loud refusal)
    and continues.
    """
    from pystatistics.multivariate import pca

    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    engine = f"pystatistics:{backend}"
    kwargs: dict[str, Any] = dict(
        center=center, scale=scale, n_components=n_components, backend=backend)
    if solver is not None:
        kwargs["solver"] = solver
    # seed/oversample/n_iter only affect the randomized GPU path; harmless on CPU.
    kwargs["seed"] = seed

    def _call():
        return pca(X, **kwargs)

    try:
        wall, sol = measure(_call, device=device, repeats=repeats, warmup=warmup)
    except Exception as exc:  # noqa: BLE001 - record and continue the sweep
        return make_record(
            engine=engine, dataset=dataset, n=n, p=p, wall=None,
            backend_name=f"pca_{backend}",
            extra={"analysis": "pca", "center": center, "scale": scale,
                   "error": f"{type(exc).__name__}: {exc}"})

    sdev = np.asarray(sol.sdev, dtype=float).ravel()
    var = sdev ** 2
    summary: dict[str, Any] = {
        "sdev": _as_list(sdev),
        "explained_variance": _as_list(var),
        "explained_variance_ratio": _as_list(var / var.sum()),
        "rotation": _as_matrix(sol.rotation),
        "scores": _as_matrix(sol.x),
        "center": _as_list(sol.center),
        "scale": _as_list(sol.scale) if sol.scale is not None else None,
        "n_components": int(sdev.size),
    }
    precision = "fp64" if backend == "cpu" else ("fp64" if backend == "gpu_fp64" else "fp32")
    peak = wall.get("peak_mem_bytes") if wall else None
    extra: dict[str, Any] = {
        "analysis": "pca", "center": center, "scale": scale, "solver": solver,
        "backend_resolved": str(getattr(sol._result, "backend_name", "")),
        "peak_mem_mb": (round(peak / 1e6, 1) if peak else None),
    }
    return make_record(
        engine=engine, dataset=dataset, n=n, p=p, wall=wall,
        backend_name=f"pca_{backend}", precision=precision,
        summary=summary, extra=extra)


def run_fa_record(
    X: NDArray[np.floating], *,
    n_factors: int,
    dataset: str,
    rotation: str = "varimax",
    repeats: int = 5,
    warmup: int = 1,
) -> dict[str, Any]:
    """Fit + time ``factor_analysis(X, n_factors=...)`` (CPU-only ML FA)."""
    from pystatistics.multivariate import factor_analysis

    X = np.asarray(X, dtype=np.float64)
    n, p = X.shape
    engine = "pystatistics:cpu"

    def _call():
        return factor_analysis(X, n_factors=n_factors, rotation=rotation)

    try:
        wall, sol = measure(_call, device="cpu", repeats=repeats, warmup=warmup)
    except Exception as exc:  # noqa: BLE001
        return make_record(
            engine=engine, dataset=dataset, n=n, p=p, wall=None,
            backend_name="factanal_cpu",
            extra={"analysis": "factor_analysis", "n_factors": n_factors,
                   "error": f"{type(exc).__name__}: {exc}"})

    summary: dict[str, Any] = {
        "loadings": _as_matrix(sol.loadings),
        "uniquenesses": _as_list(sol.uniquenesses),
        "communalities": _as_list(sol.communalities),
        "objective": float(sol.objective),
        "chi_sq": (float(sol.chi_sq) if sol.chi_sq is not None else None),
        "p_value": (float(sol.p_value) if sol.p_value is not None else None),
        "dof": int(sol.dof),
    }
    peak = wall.get("peak_mem_bytes") if wall else None
    extra: dict[str, Any] = {
        "analysis": "factor_analysis", "n_factors": n_factors,
        "rotation_method": rotation,
        "converged": bool(getattr(sol, "converged", True)),
        "n_iter": int(getattr(sol, "n_iter", 0) or 0),
        "peak_mem_mb": (round(peak / 1e6, 1) if peak else None),
    }
    return make_record(
        engine=engine, dataset=dataset, n=n, p=p, wall=wall,
        backend_name="factanal_cpu", precision="fp64",
        converged=bool(getattr(sol, "converged", True)),
        n_iter=int(getattr(sol, "n_iter", 0) or 0),
        summary=summary, extra=extra)
