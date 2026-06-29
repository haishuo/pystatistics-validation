"""Person-period boundary designs that stress the float32 GPU GLM solve (R13).

One job: build deterministic ``[interval one-hot | covariates]`` binomial designs
that STRADDLE the float32 precision floor along the axes that are adversarial for
DISCRETE-TIME specifically — so the R12 no-silent-wrong gate can be re-proven on
discrete-time's OWN regime rather than inherited from the regression GLM (RIGOR
R13: a forwarded guarantee must be re-proven on the new input regime).

The discrete-time-specific stressors, and why they are not the regression grid:

  - **low_hazard**     a heavy low-weight interval BLOCK: half the baseline
                       interval dummies carry a deep negative log-hazard (tiny
                       per-interval probability -> IRLS weights w=p(1-p) near
                       zero), while the other half stay at a moderate baseline so
                       the design still carries enough events to IDENTIFY the
                       model — mirroring real flchain (low per-interval hazard but
                       thousands of total events). A uniformly-deep baseline is
                       not discrete-time's regime; it is an empty dataset (zero
                       events), where the fp64 reference itself is undefined — the
                       driver's reference-degeneracy guard excludes any such
                       design rather than judging fp32 against a meaningless
                       optimum. severity = the depth of the low block (more
                       negative = tinier weights in that block).
  - **near_separation** a growing fraction of the intervals are sparse (few rows,
                       all events), so those baseline dummies near-separate
                       (fitted hazard -> 1, coef -> large). severity = fraction of
                       intervals made sparse-separating.
  - **large_coef**     covariate effects scaled up so the linear predictor — and
                       thus the IRLS weights — span many orders of magnitude.
                       severity = covariate coefficient scale.
  - **conditioning**   a near-collinear covariate pair on top of the interval
                       block (squares badly in the float32 XᵀWX). severity = the
                       collinearity gap (smaller = worse conditioning).

Deterministic: fixed seed; the design is built once, outside any timed region.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Severity sweeps per mechanism (deliberately pushed past the float32 floor).
SWEEPS: dict[str, list[float]] = {
    "low_hazard":     [4.0, 6.0, 8.0, 10.0, 12.0, 14.0],   # baseline log-hazard depth
    "near_separation": [0.1, 0.3, 0.5, 0.7, 0.85, 0.95],   # fraction sparse-separating
    "large_coef":     [1.0, 3.0, 6.0, 10.0, 16.0, 24.0],   # covariate coef scale
    "conditioning":   [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],  # collinearity gap
}

_N_PP = 8000          # person-period rows
_N_INTERVALS = 40     # baseline-hazard interval dummies
_N_COV = 4            # discrete-time covariates


def make_pp_boundary(
    mechanism: str, severity: float, *, seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int, int]:
    """Return ``(X_pp, y_pp, n_intervals, n_cov)`` for one stressed PP design.

    ``X_pp`` is ``(n, n_intervals + n_cov)``: a one-hot interval block (the
    baseline hazard, no intercept) followed by ``n_cov`` covariates. ``y_pp`` is
    Bernoulli from the true logistic model.
    """
    rng = np.random.default_rng(seed)
    n, m, k = _N_PP, _N_INTERVALS, _N_COV

    interval_idx = rng.integers(0, m, size=n)
    X_int = np.zeros((n, m), dtype=np.float64)
    X_int[np.arange(n), interval_idx] = 1.0

    if mechanism == "conditioning":
        # near-collinear covariate pair + two independent predictors.
        x1 = rng.standard_normal(n)
        x2 = x1 + severity * rng.standard_normal(n)
        Xc = np.column_stack([x1, x2, rng.standard_normal(n), rng.standard_normal(n)])
    else:
        Xc = rng.standard_normal((n, k))
    Xc = (Xc - Xc.mean(axis=0)) / Xc.std(axis=0)

    # Baseline log-hazards. low_hazard builds a HEAVY LOW-WEIGHT BLOCK: half the
    # intervals are driven deeply negative (tiny weights at `severity` depth) while
    # the other half stay moderate (~ -2) so the design still carries events and
    # the fp64 optimum is well-defined. The other mechanisms use a single moderate
    # low baseline (~ -4, expit ≈ 0.018).
    if mechanism == "low_hazard":
        base = -2.0 + rng.uniform(-0.5, 0.5, size=m)
        low_block = rng.permutation(m)[: m // 2]
        base[low_block] = -severity + rng.uniform(-0.5, 0.5, size=low_block.size)
    else:
        base = -4.0 + rng.uniform(-0.5, 0.5, size=m)

    # Covariate effects: large_coef scales them up; others keep them modest.
    coef_scale = severity if mechanism == "large_coef" else 0.4
    beta_cov = rng.normal(0.0, coef_scale / np.sqrt(k), size=k)

    eta = base[interval_idx] + Xc @ beta_cov
    eta = np.clip(eta, -30.0, 30.0)
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < prob).astype(np.float64)

    if mechanism == "near_separation":
        # Force a fraction of intervals to be sparse-and-all-events: pick those
        # intervals, keep only a few rows in each, set their outcome to 1 so the
        # baseline dummy near-separates (fitted hazard -> 1).
        n_sparse = max(1, int(severity * m))
        sparse_intervals = rng.choice(m, size=n_sparse, replace=False)
        for iv in sparse_intervals:
            rows = np.where(interval_idx == iv)[0]
            if rows.size > 3:
                drop = rows[3:]                 # keep at most 3 rows in this interval
                # zero out their interval dummy and reassign to interval 0 with y=0,
                # so the sparse interval has few rows, all events.
                X_int[drop, iv] = 0.0
                X_int[drop, 0] = 1.0
            keep = rows[:3]
            y[keep] = 1.0

    X_pp = np.column_stack([X_int, Xc])
    return X_pp, y, m, k


def eq_cond(X: NDArray) -> float:
    """Correlation-scaled condition number (sqrt) of the design — the float32
    XᵀWX conditioning proxy, comparable across designs of different scale."""
    g = np.asarray(X, float).T @ np.asarray(X, float)
    d = np.sqrt(np.clip(np.diag(g), 1e-300, None))
    return float(np.linalg.cond(g / np.outer(d, d)) ** 0.5)
