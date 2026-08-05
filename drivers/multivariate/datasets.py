"""Curate the canonical multivariate (PCA / factor-analysis) validation datasets.

One job: produce the exact, deterministic numeric matrices that BOTH pystatistics
and R analyse, so an eigenvalue-for-eigenvalue / loading-for-loading comparison is
meaningful. No fitting, no timing, no R invocation for the analysis itself — just
the data.

Source of truth is the centralized HDF5 store (R17): ``iris.h5`` and
``usarrests.h5`` live in the ``pystatistics`` namespace of the central store,
reached via ``DATASETS_ROOT``. The store
writes ``/data/values`` as float32 (schema 1.0.0); we promote to float64 on load
so the CPU reference path runs in double precision. Because the loaded float64
array is the single source handed to BOTH engines (Python fits it directly; R
reads the identical values via the CSV the driver dumps), agreement is to fp64
round-off of the SAME numbers — not contaminated by an input mismatch.

Two real datasets, each the textbook target for its method:

- **iris** (``datasets::iris``, 150x4) — the canonical PCA teaching set. Strongly
  correlated petal/sepal measurements; ``prcomp(iris[,1:4])`` is the worked
  example. Also the factor-analysis correctness target (4 features, 1 factor).
- **USArrests** (``datasets::USArrests``, 50x4) — R's own ``prcomp`` help-page
  example. Murder/Assault/Rape rates plus UrbanPop, on wildly different scales
  (Assault counts in the hundreds dwarf the rates), so covariance-PCA and
  correlation-PCA (``scale=TRUE``) diverge — the textbook scaling case (R10/R15).

Contract: every loader returns ``(X, names, spec)`` where ``X`` is float64
``(n, p)`` with NO centring/scaling applied (the analysis does that), ``names``
are the p column names, and ``spec`` documents the analysis + R reference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from store_io import store_h5_path  # noqa: E402


@dataclass(frozen=True)
class MVSpec:
    """What analysis is run on a curated design (documentation + R bridge)."""

    key: str                       # stable id, e.g. "iris", "usarrests"
    dataset: str                   # dataset id (matches the .h5 stem)
    r_reference: str               # canonical R call, e.g. "stats::prcomp"
    why: str                       # why this dataset for this method
    n_factors: int | None = None   # factor-analysis factor count (None = PCA-only)
    names: list[str] = field(default_factory=list)


def _store_h5(stem: str) -> Path:
    """Locate ``<stem>.h5`` in the central store, else fail loud (R5/R17).

    Store resolution lives in ``drivers/_shared/store_io.py`` — the single place
    that knows where the store is. Do not reintroduce a local search path here.
    """
    return store_h5_path(stem)


def _load_values(stem: str) -> tuple[NDArray[np.float64], list[str]]:
    """Read ``/data/values`` (float32) from the store and promote to float64.

    Fails loud if any cell is missing (NaN): the PCA/FA reference datasets are
    fully observed by construction, and a NaN would silently corrupt the
    covariance — exactly the kind of quiet-wrong this validation exists to catch.
    """
    import h5py

    path = _store_h5(stem)
    with h5py.File(path, "r") as f:
        values = np.asarray(f["data/values"][:], dtype=np.float64)
        raw = [s.decode() if isinstance(s, bytes) else str(s)
               for s in f["columns/raw_names"][:]]
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"{stem}.h5: {n_bad} non-finite cell(s); the multivariate reference "
            f"datasets must be fully observed.")
    return values, raw


def load_iris() -> tuple[NDArray[np.float64], list[str], MVSpec]:
    """iris 150x4 (Sepal/Petal Length+Width). PCA + 1-factor FA target."""
    X, names = _load_values("iris")
    spec = MVSpec(
        key="iris", dataset="iris",
        r_reference="stats::prcomp / stats::factanal",
        why="Canonical PCA teaching set: 150 flowers x 4 correlated measurements. "
            "prcomp(iris[,1:4]) is the worked example; 1-factor ML FA also fits.",
        n_factors=1, names=names)
    return X, names, spec


def load_usarrests() -> tuple[NDArray[np.float64], list[str], MVSpec]:
    """USArrests 50x4. R's prcomp help-page example; the scaling case (R10/R15)."""
    X, names = _load_values("usarrests")
    spec = MVSpec(
        key="usarrests", dataset="usarrests",
        r_reference="stats::prcomp",
        why="R's prcomp help-page example. Mixed scales (Assault counts dwarf the "
            "per-100k rates) so covariance- and correlation-PCA diverge -- the "
            "textbook scale=TRUE case.",
        names=names)
    return X, names, spec


# Registry: stable key -> loader. Drives the PCA correctness sweep. Both datasets
# carry a PCA claim; iris additionally carries the factor-analysis claim.
PCA_LOADERS = {
    "iris": load_iris,
    "usarrests": load_usarrests,
}

FA_LOADERS = {
    "iris": load_iris,
}
