"""Synthesize a person-period-shaped binomial design at a controlled size.

One job: build a deterministic ``[interval one-hot | covariates]`` design with a
binary response, matching the SHAPE of a discrete-time person-period expansion
(many low-hazard baseline-interval dummies plus a few covariates), so the GPU
scaling study can sweep person-period row counts past what flchain reaches while
holding the column count fixed. This is the synthetic scale-up grid; the real
claim is carried by flchain (see ``datasets.load_flchain``).

Why not just reuse the regression scaling DGP: discrete-time designs have a
specific, harder structure — a block of one-hot interval dummies whose
coefficients are the (small, negative) baseline log-hazards, giving tiny IRLS
weights. That structure, not n alone, is what stresses a float32 GPU solve, so a
faithful scale-up design must reproduce it rather than use dense Gaussian columns.

Deterministic: fixed seed, no randomness in any timed region (the design is built
once, before timing).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def make_pp_design(
    n_pp: int, n_intervals: int, n_cov: int, *,
    seed: int, baseline_logit: float = -4.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(X_pp, y_pp)`` for a synthetic person-period binomial design.

    ``X_pp`` is ``(n_pp, n_intervals + n_cov)``: a one-hot interval block (each
    row assigned to one interval, so the block is the baseline hazard) followed
    by ``n_cov`` standardized Gaussian covariates. The interval baseline
    log-hazards decline around ``baseline_logit`` (so per-interval event
    probabilities are small, ~expit(-4) ≈ 0.018, reproducing the tiny-weight
    structure of a real discrete-time fit). ``y_pp`` is Bernoulli from the true
    logistic model. No intercept column (the interval dummies span it).
    """
    if n_pp < n_intervals + n_cov:
        raise ValueError(f"n_pp={n_pp} too small for {n_intervals}+{n_cov} columns")
    rng = np.random.default_rng(seed)

    interval_idx = rng.integers(0, n_intervals, size=n_pp)
    X_interval = np.zeros((n_pp, n_intervals), dtype=np.float64)
    X_interval[np.arange(n_pp), interval_idx] = 1.0

    Xc = rng.standard_normal((n_pp, n_cov))
    Xc = (Xc - Xc.mean(axis=0)) / Xc.std(axis=0)

    # True parameters: baseline log-hazards spread mildly around baseline_logit;
    # modest covariate effects so the linear predictor stays O(1).
    base = baseline_logit + rng.uniform(-0.6, 0.6, size=n_intervals)
    beta_cov = rng.normal(0.0, 0.3, size=n_cov)

    eta = base[interval_idx] + Xc @ beta_cov
    prob = 1.0 / (1.0 + np.exp(-eta))
    y_pp = (rng.random(n_pp) < prob).astype(np.float64)

    X_pp = np.column_stack([X_interval, Xc])
    return X_pp, y_pp
