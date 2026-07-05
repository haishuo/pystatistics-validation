"""Comparison helpers for timeseries validation records.

Small, dependency-free helpers to turn a pystatistics-vs-R comparison into a
serializable record with max absolute / relative differences, NaN-aware. Kept
separate so every runner builds records the same way (one definition of "agree").
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays/bools to JSON-native types."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    return obj


def _pair(a: Any, b: Any) -> tuple[NDArray, NDArray]:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def arr_cmp(a: Any, b: Any) -> dict[str, float]:
    """Max abs/rel difference between two arrays (NaN cells dropped pairwise)."""
    a, b = _pair(a, b)
    if a.size == 0:
        return {"n": 0, "max_abs": float("nan"), "max_rel": float("nan")}
    absd = np.abs(a - b)
    denom = np.maximum(np.abs(b), 1e-12)
    return {"n": int(a.size), "max_abs": float(absd.max()),
            "max_rel": float((absd / denom).max())}


def scalar_cmp(a: float, b: float) -> dict[str, float]:
    """Abs/rel difference between two scalars."""
    a, b = float(a), float(b)
    absd = abs(a - b)
    return {"py": a, "r": b, "abs": absd, "rel": absd / max(abs(b), 1e-12)}


def within(cmp: dict[str, float], *, abs_tol: float | None = None,
           rel_tol: float | None = None) -> bool:
    """True if a comparison meets an absolute and/or relative tolerance."""
    ok = True
    if abs_tol is not None:
        v = cmp.get("max_abs", cmp.get("abs"))
        ok = ok and (v is not None and not np.isnan(v) and v <= abs_tol)
    if rel_tol is not None:
        v = cmp.get("max_rel", cmp.get("rel"))
        ok = ok and (v is not None and not np.isnan(v) and v <= rel_tol)
    return bool(ok)


def expect_raises(fn, exc_types) -> dict[str, Any]:
    """Run fn expecting it to raise one of exc_types; record the outcome."""
    if not isinstance(exc_types, tuple):
        exc_types = (exc_types,)
    names = "/".join(t.__name__ for t in exc_types)
    try:
        fn()
        return {"raised": False, "expected": names, "got": None, "pass": False}
    except exc_types as e:
        return {"raised": True, "expected": names,
                "got": type(e).__name__, "pass": True}
    except Exception as e:  # wrong exception type — a fail-loud miss
        return {"raised": True, "expected": names,
                "got": type(e).__name__, "pass": False}
