"""Numeric comparison helpers for the ordinal/multinomial validation.

One job: turn two engines' estimates into compact, JSON-serialisable agreement
summaries (max abs / max rel difference), plus the pass predicate and the
native-type coercion the frozen artifact needs. Mirrors the timeseries
``tscompare`` helpers so the report tables read the same across subsystems.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def to_native(obj: Any) -> Any:
    """Recursively coerce numpy scalars/arrays to JSON-native Python types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_native(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _flat(a: Any) -> NDArray:
    return np.asarray(a, dtype=float).ravel()


def arr_cmp(a: Any, b: Any) -> dict[str, float]:
    """max_abs / max_rel difference between two equal-length numeric containers."""
    x, y = _flat(a), _flat(b)
    if x.shape != y.shape:
        return {"n": int(min(x.size, y.size)), "max_abs": float("nan"),
                "max_rel": float("nan"), "shape_mismatch": True}
    if x.size == 0:
        return {"n": 0, "max_abs": 0.0, "max_rel": 0.0}
    d = np.abs(x - y)
    denom = np.maximum(np.abs(y), 1e-12)
    return {"n": int(x.size), "max_abs": float(d.max()),
            "max_rel": float((d / denom).max())}


def scalar_cmp(a: float, b: float) -> dict[str, float]:
    a, b = float(a), float(b)
    return {"py": a, "r": b, "abs": abs(a - b),
            "rel": abs(a - b) / max(abs(b), 1e-12)}


def within(cmp: dict[str, float], *, abs_tol: float | None = None,
           rel_tol: float | None = None) -> bool:
    """Pass if the abs and/or rel difference is within the given tolerance(s)."""
    ok = True
    ma = cmp.get("max_abs", cmp.get("abs"))
    mr = cmp.get("max_rel", cmp.get("rel"))
    if cmp.get("shape_mismatch"):
        return False
    if abs_tol is not None and ma is not None:
        ok = ok and (ma <= abs_tol)
    if rel_tol is not None and mr is not None:
        ok = ok and (mr <= rel_tol)
    return bool(ok)


def align_multinom_coef(py_mat: NDArray, r_mat: NDArray) -> tuple[NDArray, NDArray]:
    """Return (py, r) coefficient matrices as (K-1, 1+p) float arrays.

    pystatistics ``coefficient_matrix`` rows are the non-reference classes in code
    order 0..K-2; nnet's rows (once the baseline is releveled to the last code) are
    the same non-baseline levels in that same order — so the rows align directly.
    Both put the intercept in column 0 followed by the covariates.
    """
    py = np.atleast_2d(np.asarray(py_mat, float))
    r = np.atleast_2d(np.asarray(r_mat, float))
    return py, r
