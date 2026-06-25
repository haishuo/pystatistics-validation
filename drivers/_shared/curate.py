"""Pure curation logic for turning raw survey arrays into MVN MLE problems.

One job: given a raw value matrix, a typed missingness mask, and per-column
metadata, decide which columns make a well-posed multivariate-normal problem and
produce the NaN-coded matrix the estimator consumes.

This module deliberately contains NO file I/O — every function operates on plain
NumPy arrays so the selection/masking rules can be unit-tested without the
(gitignored, multi-hundred-MB) survey files present. File reading lives in
``survey_io``.

Missingness legend used throughout (from the survey HDF5 schema)::

    0 = observed, 1 = dont_know, 2 = refused, 3 = not_asked, 4 = other_missing

Code 3 (``not_asked``) is *structural* missingness — the question was not part of
that respondent's wave/module — and is treated separately from genuine item
nonresponse (codes 1, 2, 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

NOT_ASKED_CODE = 3
OBSERVED_CODE = 0
DEFAULT_MISSING_CODES = (1, 2, 3, 4)  # everything that is not OBSERVED_CODE


@dataclass(frozen=True)
class ColumnStats:
    """Per-column coverage statistics derived from a missingness mask.

    All arrays are 1-D with length ``n_cols``.
    """

    n_rows: int
    n_cols: int
    observed_frac: NDArray[np.float64]        # P(code == 0)
    asked_frac: NDArray[np.float64]           # P(code != not_asked)
    observed_among_asked: NDArray[np.float64]  # P(observed | asked)


def column_stats(missing_state: NDArray[np.integer], *,
                 not_asked_code: int = NOT_ASKED_CODE) -> ColumnStats:
    """Compute per-column coverage from a typed missingness mask.

    Parameters
    ----------
    missing_state
        Integer array of shape (n_rows, n_cols) using the legend above.
    not_asked_code
        Code denoting structural (not-asked) missingness.

    Raises
    ------
    ValueError
        If ``missing_state`` is not a 2-D integer array.
    """
    arr = np.asarray(missing_state)
    if arr.ndim != 2:
        raise ValueError(f"missing_state must be 2-D, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"missing_state must be integer, got dtype {arr.dtype}")

    n_rows, n_cols = arr.shape
    if n_rows == 0:
        raise ValueError("missing_state has zero rows")

    observed = (arr == OBSERVED_CODE).sum(axis=0)
    asked = (arr != not_asked_code).sum(axis=0)
    observed_frac = observed / n_rows
    asked_frac = asked / n_rows
    with np.errstate(invalid="ignore", divide="ignore"):
        among = np.where(asked > 0, observed / asked, 0.0)
    return ColumnStats(
        n_rows=n_rows,
        n_cols=n_cols,
        observed_frac=observed_frac.astype(np.float64),
        asked_frac=asked_frac.astype(np.float64),
        observed_among_asked=among.astype(np.float64),
    )


def rank_eligible_columns(stats: ColumnStats,
                          data_types: NDArray,
                          *,
                          allowed_types: tuple[str, ...] = ("continuous", "ordinal"),
                          min_asked_frac: float = 0.30,
                          min_observed_among_asked: float = 0.50,
                          observed_frac_band: tuple[float, float] = (0.0, 1.0),
                          ) -> NDArray[np.intp]:
    """Return *all* eligible column indices in descending quality order.

    A column is *eligible* if its declared type is in ``allowed_types``, it
    clears the coverage thresholds, and its overall ``observed_frac`` falls
    within ``observed_frac_band`` (inclusive). The band is the key knob for a
    *missingness* benchmark: an upper bound below 1.0 excludes near-complete
    administrative columns (weights, IDs, year) whose ~0% missingness would
    collapse the problem to a handful of patterns, while a lower bound keeps
    columns estimable. Eligible columns are ranked by ``observed_among_asked``
    (desc), then ``asked_frac`` (desc), then column index (asc) — fully
    reproducible, no randomness.

    Returns indices in **rank order** (best first), *not* sorted ascending, so
    callers can take or filter a prefix and still know the quality ordering.
    """
    lo, hi = observed_frac_band
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"observed_frac_band must satisfy 0<=lo<=hi<=1, got {observed_frac_band}")
    types = np.asarray(data_types).astype(str)
    if types.shape[0] != stats.n_cols:
        raise ValueError(
            f"data_types length {types.shape[0]} != n_cols {stats.n_cols}")

    type_ok = np.isin(types, np.asarray(allowed_types, dtype=str))
    coverage_ok = (
        (stats.observed_among_asked >= min_observed_among_asked)
        & (stats.asked_frac >= min_asked_frac)
    )
    band_ok = (stats.observed_frac >= lo) & (stats.observed_frac <= hi)
    eligible = np.flatnonzero(type_ok & coverage_ok & band_ok)

    # Lexsort is ascending on its last key; negate the descending keys.
    order = np.lexsort((
        eligible,                                       # tie-break: index asc
        -stats.asked_frac[eligible],                    # then asked desc
        -stats.observed_among_asked[eligible],          # primary: obs|asked desc
    ))
    return eligible[order]


def select_columns(stats: ColumnStats,
                   data_types: NDArray,
                   p: int,
                   *,
                   allowed_types: tuple[str, ...] = ("continuous", "ordinal"),
                   min_asked_frac: float = 0.30,
                   min_observed_among_asked: float = 0.50,
                   observed_frac_band: tuple[float, float] = (0.0, 1.0),
                   ) -> NDArray[np.intp]:
    """Pick the ``p`` best columns for an MVN problem, deterministically.

    Thin wrapper over :func:`rank_eligible_columns`: take the top ``p`` and
    return them **sorted ascending** (h5py fancy indexing requires increasing
    order). Collinearity is *not* screened here — see :func:`prune_collinear`
    and the ``max_abs_corr`` path in :func:`survey_io.build_mvn_problem`.

    Raises
    ------
    ValueError
        If ``p`` is not positive, the band is malformed, the metadata length is
        inconsistent, or fewer than ``p`` columns are eligible (fail loud rather
        than silently returning a smaller problem).
    """
    if p <= 0:
        raise ValueError(f"p must be positive, got {p}")
    ranked = rank_eligible_columns(
        stats, data_types,
        allowed_types=allowed_types,
        min_asked_frac=min_asked_frac,
        min_observed_among_asked=min_observed_among_asked,
        observed_frac_band=observed_frac_band,
    )
    if ranked.size < p:
        raise ValueError(
            f"only {ranked.size} eligible columns "
            f"(type in {allowed_types}, asked>={min_asked_frac}, "
            f"observed|asked>={min_observed_among_asked}, "
            f"observed_frac in {observed_frac_band}); cannot build p={p}")
    return np.sort(ranked[:p])


def prune_collinear(matrix: NDArray[np.floating],
                    max_abs_corr: float) -> NDArray[np.intp]:
    """Greedily drop near-collinear columns, keeping the earlier (higher-ranked)
    one of each offending pair.

    ``matrix`` is a NaN-coded value matrix whose columns are already in the
    desired priority order (best first). A column is dropped if its
    pairwise-complete Pearson correlation with any already-kept column exceeds
    ``max_abs_corr`` in absolute value. Near-collinear columns make the MLE
    covariance ill-conditioned (and, at correlation 1, rank-deficient, so the
    MLE does not exist); screening them keeps the problem well-posed.

    Returns the kept column positions (into ``matrix``) in ascending order.
    """
    if not (0.0 < max_abs_corr <= 1.0):
        raise ValueError(f"max_abs_corr must be in (0, 1], got {max_abs_corr}")
    k = matrix.shape[1]
    corr = np.ma.corrcoef(np.ma.masked_invalid(matrix), rowvar=False).filled(0.0)
    corr = np.atleast_2d(corr)
    kept: list[int] = []
    for j in range(k):
        if all(abs(corr[j, i]) <= max_abs_corr for i in kept):
            kept.append(j)
    return np.asarray(kept, dtype=np.intp)


def to_nan_matrix(values: NDArray[np.floating],
                  missing_state: NDArray[np.integer],
                  *,
                  missing_codes: tuple[int, ...] = DEFAULT_MISSING_CODES,
                  ) -> NDArray[np.float32]:
    """Return a float32 copy of ``values`` with missing cells set to NaN.

    Any cell whose mask code is in ``missing_codes`` becomes NaN; cells coded
    ``0`` (observed) keep their value.

    Raises
    ------
    ValueError
        If the two arrays' shapes disagree.
    """
    vals = np.asarray(values)
    mask = np.asarray(missing_state)
    if vals.shape != mask.shape:
        raise ValueError(
            f"values shape {vals.shape} != missing_state shape {mask.shape}")
    out = vals.astype(np.float32, copy=True)
    is_missing = np.isin(mask, np.asarray(missing_codes))
    out[is_missing] = np.nan
    return out


def standardize_columns(matrix: NDArray[np.floating]) -> NDArray[np.float32]:
    """Z-score each column using its observed (non-NaN) entries.

    Survey items arrive on wildly different raw scales (codes, counts, weights),
    which ill-conditions the covariance and slows — or breaks — optimisation for
    *both* parameterizations. Standardising each column to mean 0 / sd 1 over its
    observed values removes that confound so the benchmark compares the
    parameterizations, not the data's arbitrary units. NaN positions are
    preserved (a standardized matrix still carries the same missingness mask).

    Raises
    ------
    ValueError
        If any column has fewer than 2 observed values or zero observed variance
        (degenerate for a multivariate-normal model — fail loud).
    """
    mat = np.asarray(matrix, dtype=np.float64)
    out = np.empty_like(mat, dtype=np.float32)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        obs = col[~np.isnan(col)]
        if obs.size < 2:
            raise ValueError(f"column {j} has < 2 observed values; cannot standardize")
        sd = obs.std(ddof=1)
        if not np.isfinite(sd) or sd == 0.0:
            raise ValueError(f"column {j} has zero/invalid observed variance")
        out[:, j] = ((col - obs.mean()) / sd).astype(np.float32)
    return out


def drop_empty_rows(matrix: NDArray[np.floating], *,
                    min_observed: int = 1) -> NDArray[np.float32]:
    """Drop rows with fewer than ``min_observed`` non-NaN entries.

    Respondents who answered too few of the selected items contribute nothing
    estimable and only add degenerate missingness patterns.

    Raises
    ------
    ValueError
        If ``min_observed`` is negative or every row is dropped.
    """
    if min_observed < 0:
        raise ValueError(f"min_observed must be >= 0, got {min_observed}")
    mat = np.asarray(matrix, dtype=np.float32)
    keep = (~np.isnan(mat)).sum(axis=1) >= min_observed
    if not keep.any():
        raise ValueError(
            f"all rows dropped at min_observed={min_observed}; "
            "problem is empty")
    return mat[keep]
