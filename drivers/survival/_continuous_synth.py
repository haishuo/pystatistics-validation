"""Deterministic continuous-time survival data for the R15 default-degeneracy study.

One job: emit ``(time, event, X, names)`` from a proportional-hazards-style DGP
whose event times are GENUINELY CONTINUOUS (distinct, no ties), so the default
``discrete_time(intervals=None)`` makes one interval per unique event time — the
exact regime RIGOR R15 says must be validated (the call a naive user makes).

Two regimes, selected by ``censoring``:

- ``"realistic"`` — exponential survival with independent exponential censoring
  (~30-50% censored). The everyday continuous dataset; large risk sets keep each
  per-interval baseline hazard well below 1.
- ``"none"`` — every subject is an event with strictly increasing unique times.
  The adversarial extreme: the final intervals have a single at-risk subject who
  has the event, so the last baseline-hazard dummy separates (hazard -> 1). This
  is where the "default separates perfectly" fear is sharpest.

Deterministic (seeded); no randomness outside data construction.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def continuous_survival(
    n: int, *, seed: int, censoring: str = "realistic",
    beta: tuple[float, ...] = (0.4, -0.3),
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Return ``(time, event, X, names)`` with distinct continuous event times.

    ``beta`` are the true log-time coefficients (a positive entry lengthens
    survival, i.e. LOWERS the discrete-time hazard — the hazard coefficient the
    discrete-time model recovers has the opposite sign). ``X`` has no intercept.
    """
    rng = np.random.default_rng(seed)
    p = len(beta)
    X = rng.standard_normal((n, p))
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    lin = X @ np.asarray(beta, float)
    base = rng.exponential(scale=50.0, size=n) * np.exp(lin)

    if censoring == "none":
        # Every subject is an event; force strictly-increasing unique times so the
        # tail intervals hold a single at-risk subject (separation-prone extreme).
        order = np.argsort(base)
        time_ = np.empty(n, dtype=np.float64)
        time_[order] = 1.0 + np.arange(n, dtype=np.float64)
        event = np.ones(n, dtype=np.float64)
        names = [f"x{j + 1}" for j in range(p)]
        return time_, event, X, names

    if censoring != "realistic":
        raise ValueError(f"censoring must be 'realistic' or 'none', got {censoring!r}")

    cens = rng.exponential(scale=80.0, size=n)
    time_ = np.minimum(base, cens)
    event = (base <= cens).astype(np.float64)
    # Jitter to guarantee uniqueness of the event times (genuine continuity).
    time_ = time_ + rng.uniform(0.0, 1e-6, size=n)
    names = [f"x{j + 1}" for j in range(p)]
    return time_, event, X, names
