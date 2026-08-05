"""Read survey HDF5 files and assemble MVN MLE benchmark problems.

One job: file I/O for the survey HDF5 layout used by this project's ``data/``
files, plus the thin composition that turns a file path into a curated
NaN-coded matrix (delegating all selection and masking rules to :mod:`curate`).

HDF5 layout (per file)::

    /data/values            (n, p) float32   — raw numeric values
    /data/missing_state     (n, p) uint8     — typed missingness (see curate)
    /columns/data_types     (p,)   object    — continuous/ordinal/binary/nominal
    /columns/display_names  (p,)   object
    /metadata               group, attrs: dataset_name, source, ...

See ``PROVENANCE.md`` in the dataset store (``$DATASETS_ROOT``, e.g.
``Dev/datasets/PROVENANCE.md``) for the origin of these files.

The values matrix can be multi-GB (e.g. GSS is ~2 GB), so column statistics are
computed from the much smaller uint8 mask in chunks, and only the *selected*
columns are ever read out of ``/data/values``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from curate import (
    ColumnStats,
    drop_empty_rows,
    select_columns,
    rank_eligible_columns,
    prune_collinear,
    to_nan_matrix,
    DEFAULT_MISSING_CODES,
    NOT_ASKED_CODE,
    OBSERVED_CODE,
)

_VALUES = "/data/values"
_MISSING = "/data/missing_state"
_DTYPES = "/columns/data_types"
_NAMES = "/columns/display_names"


@dataclass(frozen=True)
class MVNProblem:
    """A curated multivariate-normal benchmark problem drawn from one survey."""

    dataset_name: str
    source: str
    X: NDArray[np.float32]              # (n_kept, p) with NaN for missing
    column_indices: NDArray[np.intp]    # indices into the original file columns
    column_names: list[str]
    n_rows_original: int
    overall_missing_frac: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape


def _decode(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def _require_datasets(f: h5py.File, path: Path) -> None:
    for key in (_VALUES, _MISSING, _DTYPES):
        if key not in f:
            raise KeyError(f"{path.name}: missing required dataset '{key}'")


def survey_column_stats(path: str | Path, *,
                        chunk: int = 1024) -> tuple[ColumnStats, NDArray, list[str]]:
    """Compute per-column coverage stats and return them with type/name metadata.

    Reads only the uint8 mask (column-chunked) — never the full value matrix.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"survey file not found: {path}")
    with h5py.File(path, "r") as f:
        _require_datasets(f, path)
        ms = f[_MISSING]
        n, p = ms.shape
        observed = np.zeros(p, dtype=np.int64)
        asked = np.zeros(p, dtype=np.int64)
        for s in range(0, p, chunk):
            blk = ms[:, s:s + chunk]
            observed[s:s + chunk] = (blk == OBSERVED_CODE).sum(axis=0)
            asked[s:s + chunk] = (blk != NOT_ASKED_CODE).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            among = np.where(asked > 0, observed / asked, 0.0)
        stats = ColumnStats(
            n_rows=int(n), n_cols=int(p),
            observed_frac=(observed / n).astype(np.float64),
            asked_frac=(asked / n).astype(np.float64),
            observed_among_asked=among.astype(np.float64),
        )
        data_types = np.asarray(_decode(f[_DTYPES][:]))
        names = _decode(f[_NAMES][:]) if _NAMES in f else [""] * p
    return stats, data_types, names


def build_mvn_problem(path: str | Path,
                      p: int,
                      *,
                      allowed_types: tuple[str, ...] = ("continuous", "ordinal"),
                      min_asked_frac: float = 0.30,
                      min_observed_among_asked: float = 0.50,
                      observed_frac_band: tuple[float, float] = (0.50, 0.95),
                      missing_codes: tuple[int, ...] = DEFAULT_MISSING_CODES,
                      min_row_observed: int = 2,
                      max_abs_corr: float | None = 0.99) -> MVNProblem:
    """Assemble a curated ``p``-variable MVN problem from a survey HDF5 file.

    Steps: compute column stats from the mask, deterministically rank eligible
    columns (see :func:`curate.rank_eligible_columns`), read their values + mask,
    NaN-code the missing cells, optionally screen out near-collinear columns, keep
    the best ``p``, and drop rows with fewer than ``min_row_observed`` answered
    items.

    ``max_abs_corr`` (default ``0.99``) screens near-collinear columns: the best
    ``p`` columns are chosen so that no pair has |pairwise-complete correlation|
    above this value. This keeps the MLE well-posed --- at correlation 1 the data
    covariance is rank-deficient and the MLE does not exist, and even near 1 it is
    severely ill-conditioned. Set to ``None`` to disable screening (e.g. to
    reproduce a deliberately ill-posed problem). Screening draws replacement
    columns from the ranked pool so the result still has exactly ``p`` columns.

    The input contract (file exists, schema present, enough eligible/independent
    columns) is validated here and in :mod:`curate`; violations raise rather than
    silently producing a smaller or degenerate problem.
    """
    path = Path(path)
    stats, data_types, names = survey_column_stats(path)

    if max_abs_corr is None:
        idx = select_columns(
            stats, data_types, p,
            allowed_types=allowed_types,
            min_asked_frac=min_asked_frac,
            min_observed_among_asked=min_observed_among_asked,
            observed_frac_band=observed_frac_band,
        )
        with h5py.File(path, "r") as f:
            values = f[_VALUES][:, list(idx)]
            mask = f[_MISSING][:, list(idx)]
            meta = f["/metadata"].attrs if "/metadata" in f else {}
            dataset_name = str(meta.get("dataset_name", path.stem))
            source = str(meta.get("source", "unknown"))
        X_full = to_nan_matrix(values, mask, missing_codes=missing_codes)
    else:
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
                f"only {ranked.size} eligible columns; cannot build p={p}")
        # Read the whole ranked pool (h5py needs increasing column order), then
        # reorder columns back to descending-quality rank order for screening.
        pool_sorted = np.sort(ranked)
        with h5py.File(path, "r") as f:
            values = f[_VALUES][:, list(pool_sorted)]
            mask = f[_MISSING][:, list(pool_sorted)]
            meta = f["/metadata"].attrs if "/metadata" in f else {}
            dataset_name = str(meta.get("dataset_name", path.stem))
            source = str(meta.get("source", "unknown"))
        pos = {int(c): i for i, c in enumerate(pool_sorted)}
        rank_pos = [pos[int(c)] for c in ranked]            # sorted -> rank order
        X_pool = to_nan_matrix(values, mask, missing_codes=missing_codes)[:, rank_pos]
        kept = prune_collinear(X_pool, max_abs_corr)         # positions, rank order
        if kept.size < p:
            raise ValueError(
                f"only {kept.size} columns remain after collinearity screening "
                f"(|corr|<={max_abs_corr}); cannot build p={p}. The survey lacks "
                f"{p} sufficiently independent items in-band.")
        sel = kept[:p]                                        # top-p independent
        sel_file_cols = ranked[sel]
        order = np.argsort(sel_file_cols)                    # ascending file order
        idx = sel_file_cols[order]
        X_full = X_pool[:, sel][:, order]

    n_original = X_full.shape[0]
    X = drop_empty_rows(X_full, min_observed=min_row_observed)
    missing_frac = float(np.isnan(X).mean())

    return MVNProblem(
        dataset_name=dataset_name,
        source=source,
        X=X,
        column_indices=np.asarray(idx),
        column_names=[names[i] for i in idx],
        n_rows_original=n_original,
        overall_missing_frac=missing_frac,
    )
