"""Shared runtime-parity helper for Suite 1 core tests.

Policy (see README Notes): if R takes >= 50 ms for the fit, Python must
complete within 20× that time. Shorter fits skip the check because wall-
clock noise (import overhead, scheduler jitter) dominates the signal.
"""

from __future__ import annotations

MIN_R_TIME = 0.05
MAX_RATIO = 20.0


def assert_parity(py_elapsed: float, r_elapsed: float, label: str) -> None:
    if r_elapsed is None or r_elapsed < MIN_R_TIME:
        return
    ratio = py_elapsed / r_elapsed
    assert ratio <= MAX_RATIO, (
        f"[{label}] Python {py_elapsed:.3f}s vs R {r_elapsed:.3f}s "
        f"(ratio {ratio:.1f}x > {MAX_RATIO}x limit)."
    )
