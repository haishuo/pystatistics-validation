"""
Data-generating processes with KNOWN truth, plus missingness mechanisms.

The validity study (Study B) needs data whose estimands are exact, so bias and
CI coverage can be measured against a real target rather than another estimate.
This module provides:

  * ``make_complete`` — a linear model ``y = beta0 + X beta + eps`` with X drawn
    from an AR(1)-correlated Gaussian. The returned matrix is ``D = [y | X]``
    (response in column 0), and ``beta`` (intercept first) is the exact estimand
    of the analysis model ``lm(y ~ X)``.
  * ``impose_mcar`` — missing completely at random: each eligible cell dropped
    independently with probability ``rate``.
  * ``impose_mar`` — missing at random: missingness in the target columns depends
    on the *observed* values of a fully-observed driver column (the regime that
    actually tests Rubin's-rules validity).

All functions are deterministic given ``seed`` (CLAUDE.md Rule 6) and fail loud
on degenerate requests (Rule 1): a column left with no observed values, or an
all-missing row, cannot be imputed, so we refuse to produce one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CompleteData:
    """A complete draw plus the exact estimand of ``lm(y ~ X)``."""

    matrix: NDArray[np.float64]   # (n, p+1); column 0 = y, columns 1.. = X
    beta: NDArray[np.float64]     # (p+1,) intercept first — the truth
    n: int
    p: int                        # number of predictors (matrix has p+1 columns)


@dataclass(frozen=True)
class MixedData:
    """A complete mixed-type draw: numeric columns + categorical code columns.

    Used for the categorical/mixed-type benchmarks (scaling + R fidelity). No
    known estimand — those studies measure speed and distributional agreement,
    not coverage (coverage stays on the numeric DGP).
    """

    matrix: NDArray[np.float64]       # (n, p); categorical columns hold int codes
    column_kinds: tuple[str, ...]     # per column: numeric/binary/categorical/ordered
    n: int
    p: int


# Default categorical mix: one of each method-exercising kind.
DEFAULT_CAT_SPECS = (("binary", 2), ("ordered", 4), ("categorical", 4))


def make_mixed(
    n: int,
    *,
    seed: int,
    n_numeric: int = 4,
    cat_specs: tuple[tuple[str, int], ...] = DEFAULT_CAT_SPECS,
    rho: float = 0.5,
) -> MixedData:
    """Draw a correlated mixed-type matrix: numeric columns + categorical codes.

    A latent AR(1)-correlated Gaussian drives every column, so the conditional
    models are informative. Numeric columns are the latent values; each
    categorical column discretizes its latent into ``K`` quantile bins (codes
    ``0..K-1``). The ``kind`` (binary/categorical/ordered) selects which model
    MICE uses (logreg/polyreg/polr); the codes are identical either way.
    """
    if n_numeric < 0 or (n_numeric + len(cat_specs)) < 2:
        raise ValueError("need at least 2 columns total")
    total = n_numeric + len(cat_specs)
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(ar1_cov(total, rho))
    Z = rng.standard_normal((n, total)) @ L.T

    cols = [Z[:, j] for j in range(n_numeric)]
    kinds = ["numeric"] * n_numeric
    for i, (kind, k) in enumerate(cat_specs):
        if k < 2:
            raise ValueError(f"categorical spec {kind!r} needs >= 2 levels, got {k}")
        z = Z[:, n_numeric + i]
        edges = np.quantile(z, np.linspace(0, 1, k + 1)[1:-1])
        cols.append(np.digitize(z, edges).astype(np.float64))  # codes 0..k-1
        kinds.append(kind)

    return MixedData(
        matrix=np.column_stack(cols), column_kinds=tuple(kinds), n=n, p=total
    )


def default_beta(p: int) -> NDArray[np.float64]:
    """A fixed, well-separated coefficient vector for ``p`` predictors.

    Deterministic (no RNG): intercept 0.5, then slopes alternating in sign with
    decreasing magnitude, so different coefficients are distinguishable in the
    coverage table and none is exactly zero.
    """
    if p < 1:
        raise ValueError(f"need at least 1 predictor, got {p}")
    slopes = np.array([(-1.0) ** j * (1.0 - 0.6 * j / p) for j in range(p)])
    return np.concatenate([[0.5], slopes])


def ar1_cov(p: int, rho: float) -> NDArray[np.float64]:
    """AR(1) covariance ``Sigma[i,j] = rho^|i-j|`` for ``p`` predictors."""
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (-1, 1), got {rho}")
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def make_complete(
    n: int,
    p: int,
    *,
    seed: int,
    beta: NDArray[np.float64] | None = None,
    rho: float = 0.5,
    noise_sd: float = 1.0,
) -> CompleteData:
    """Draw complete data from ``y = beta0 + X beta + N(0, noise_sd^2)``.

    X has rows iid ``N(0, Sigma)`` with ``Sigma`` AR(1) (correlation ``rho``), so
    the per-column imputation regressions are informative. Returns the stacked
    matrix ``[y | X]`` and the known coefficient vector.
    """
    if n < 2:
        raise ValueError(f"need n >= 2, got {n}")
    b = default_beta(p) if beta is None else np.asarray(beta, dtype=np.float64)
    if b.shape != (p + 1,):
        raise ValueError(f"beta must have shape ({p + 1},), got {b.shape}")
    if noise_sd <= 0:
        raise ValueError(f"noise_sd must be positive, got {noise_sd}")

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(ar1_cov(p, rho))
    X = rng.standard_normal((n, p)) @ L.T
    y = b[0] + X @ b[1:] + rng.standard_normal(n) * noise_sd
    matrix = np.column_stack([y, X])
    return CompleteData(matrix=matrix, beta=b, n=n, p=p)


# --------------------------------------------------------------- missingness
def _eligible_cols(matrix: NDArray, cols: tuple[int, ...] | None) -> tuple[int, ...]:
    p1 = matrix.shape[1]
    if cols is None:
        return tuple(range(p1))
    bad = [c for c in cols if not (0 <= c < p1)]
    if bad:
        raise ValueError(f"column(s) {bad} out of range [0, {p1})")
    return tuple(cols)


def _finalize(matrix: NDArray, mask: NDArray) -> NDArray[np.float64]:
    """Apply a missingness mask, guarding the two states MICE cannot accept.

    Repairs all-missing rows by reviving one cell, then asserts every masked
    column keeps >= 2 observed values. Mutates a copy, returns it with NaNs.
    """
    mask = mask.copy()
    # No row may be fully missing (it carries no information for any chain).
    full_row = mask.all(axis=1)
    if full_row.any():
        # Revive a deterministic cell (column 0) for each offending row.
        mask[full_row, 0] = False
    # Every column that has any missing must keep enough observed rows to fit.
    n = matrix.shape[0]
    observed_per_col = n - mask.sum(axis=0)
    starved = np.where((mask.any(axis=0)) & (observed_per_col < 2))[0]
    if starved.size:
        raise ValueError(
            f"column(s) {starved.tolist()} would be left with < 2 observed "
            f"values; lower the missing rate or exclude them."
        )
    out = matrix.astype(np.float64, copy=True)
    out[mask] = np.nan
    return out


def impose_mcar(
    matrix: NDArray[np.float64],
    *,
    rate: float,
    seed: int,
    cols: tuple[int, ...] | None = None,
) -> NDArray[np.float64]:
    """Missing completely at random: drop each eligible cell w.p. ``rate``."""
    if not (0.0 < rate < 1.0):
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    rng = np.random.default_rng(seed)
    eligible = _eligible_cols(matrix, cols)
    mask = np.zeros(matrix.shape, dtype=bool)
    draw = rng.random((matrix.shape[0], len(eligible))) < rate
    mask[:, list(eligible)] = draw
    return _finalize(matrix, mask)


def impose_mar(
    matrix: NDArray[np.float64],
    *,
    rate: float,
    seed: int,
    driver_col: int,
    target_cols: tuple[int, ...],
    strength: float = 1.5,
) -> NDArray[np.float64]:
    """Missing at random: missingness in ``target_cols`` depends on the observed
    ``driver_col``.

    The driver column is never made missing. For each target column the missing
    probability is ``sigmoid(a + strength * z(driver))`` where ``z`` is the
    standardized driver and ``a`` is solved (deterministic bisection) so the mean
    missing probability equals ``rate``. This makes the missingness depend only
    on observed data — the defining MAR condition — so Rubin's rules should stay
    valid, which is exactly what Study B checks.
    """
    if not (0.0 < rate < 1.0):
        raise ValueError(f"rate must be in (0, 1), got {rate}")
    p1 = matrix.shape[1]
    if not (0 <= driver_col < p1):
        raise ValueError(f"driver_col {driver_col} out of range [0, {p1})")
    if driver_col in target_cols:
        raise ValueError("driver_col must not be among target_cols")
    targets = _eligible_cols(matrix, target_cols)

    driver = matrix[:, driver_col]
    z = (driver - driver.mean()) / (driver.std() + 1e-12)
    eta = strength * z
    a = _solve_intercept(eta, rate)
    prob = _sigmoid(a + eta)

    rng = np.random.default_rng(seed)
    mask = np.zeros(matrix.shape, dtype=bool)
    for c in targets:
        mask[:, c] = rng.random(matrix.shape[0]) < prob
    return _finalize(matrix, mask)


def _sigmoid(x: NDArray) -> NDArray:
    return 1.0 / (1.0 + np.exp(-x))


def _solve_intercept(eta: NDArray, target_rate: float) -> float:
    """Find ``a`` with ``mean(sigmoid(a + eta)) == target_rate`` by bisection.

    ``mean(sigmoid(a + eta))`` is monotone increasing in ``a`` from 0 to 1, so a
    bracketed bisection converges; deterministic, no RNG.
    """
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _sigmoid(mid + eta).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
