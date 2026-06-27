"""Reconstruct the discrete-time person-period design, mirroring pystatistics.

One job: rebuild the EXACT person-period (long-format) design matrix and response
that ``pystatistics.survival.discrete_time`` constructs internally, so the R
``glm(binomial)`` reference fits the identical design. This is a faithful copy of
the expansion in ``pystatistics/survival/_discrete.py`` — if the library's
expansion ever changes, the ``person_period_n`` consistency guard in
``generate_correctness`` fails loud rather than comparing mismatched designs.

The expansion is deterministic. The returned design is ``[interval_dummies |
covariates]`` with NO intercept (the interval dummies serve as the baseline
hazard), exactly the matrix pystatistics hands to its binomial GLM.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_person_period(
    time: NDArray, event: NDArray, X: NDArray, interval_bounds: NDArray,
) -> tuple[NDArray, NDArray, int]:
    """Return ``(X_pp, y_pp, n_intervals)`` mirroring ``discrete_time_fit``.

    ``interval_bounds`` are ascending interval start times (as
    ``discrete_time(intervals=...)`` receives them). ``X`` has no intercept.
    ``X_pp`` is ``[interval one-hot | covariates]``; ``y_pp`` is the event
    indicator per person-period row.
    """
    interval_bounds = np.sort(np.asarray(interval_bounds, dtype=np.float64))
    n, p = X.shape
    n_intervals = len(interval_bounds)

    rows_y: list[float] = []
    rows_X_cov: list[NDArray] = []
    rows_interval_idx: list[int] = []

    # This loop is a line-for-line mirror of _discrete.py's expansion.
    for i in range(n):
        for j, t_bound in enumerate(interval_bounds):
            at_risk = True if j == 0 else time[i] >= t_bound
            if not at_risk:
                break

            if event[i] == 1 and time[i] == t_bound:
                y_ij = 1.0
            elif (event[i] == 1 and j < n_intervals - 1
                  and time[i] < interval_bounds[j + 1] and time[i] >= t_bound):
                y_ij = 1.0
            elif event[i] == 1 and j == n_intervals - 1 and time[i] >= t_bound:
                y_ij = 1.0
            else:
                y_ij = 0.0

            rows_y.append(y_ij)
            rows_X_cov.append(X[i])
            rows_interval_idx.append(j)

            if y_ij == 1.0:
                break
            if event[i] == 0 and j < n_intervals - 1 and time[i] < interval_bounds[j + 1]:
                break
            if event[i] == 0 and j == n_intervals - 1:
                break

    y_pp = np.array(rows_y, dtype=np.float64)
    X_cov_pp = np.array(rows_X_cov, dtype=np.float64)
    interval_idx_pp = np.array(rows_interval_idx, dtype=np.intp)
    person_period_n = len(y_pp)

    X_interval = np.zeros((person_period_n, n_intervals), dtype=np.float64)
    for k in range(person_period_n):
        X_interval[k, interval_idx_pp[k]] = 1.0

    X_pp = np.column_stack([X_interval, X_cov_pp])
    return X_pp, y_pp, n_intervals
