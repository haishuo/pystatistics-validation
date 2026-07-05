"""montecarlo validation data layer — canonical resampling datasets + statistics.

One job: load the reference tables from the central HDF5 store
(``MVNMLE_DATA_DIR``) and expose them as fp64 arrays, together with the
statistic functions that MUST mirror the R reference registry in
``r_reference.R`` exactly (so both engines compute the identical statistic on
identical numbers, R17). Also provides the seeded known-DGP generator for the
coverage study (its master seed is recorded in the artifact).

Statistics use boot's ``stype='i'`` convention: ``fn(data, indices) -> (k,)``.
No CSVs live in the repo; the store is the single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import h5py

_STORE_FALLBACKS = [
    Path("/Volumes/Archive/Documents/Dropbox/Dev/datasets"),
    Path("/mnt/data/pystatistics-datasets"),
]

_WHY = {
    "law": "Efron & Tibshirani law-school data (n=15) — the canonical BCa "
           "correlation example; percentile != BCa here (skewed r-hat).",
    "city": "Cochran city population data (n=10) — the canonical ratio "
            "estimator; a biased, skewed statistic where BCa correction bites.",
    "sleep": "R sleep data (n=20, two groups of 10) — two-sample permutation "
             "on the difference in means; small enough for exact enumeration.",
}


@dataclass(frozen=True)
class Dataset:
    key: str
    why: str
    values: NDArray[np.float64]   # (n, p)
    columns: list[str]


def _store_dir() -> Path:
    root = os.environ.get("MVNMLE_DATA_DIR")
    if root and Path(root).is_dir():
        return Path(root)
    for p in _STORE_FALLBACKS:
        if p.is_dir():
            return p
    raise FileNotFoundError(
        "montecarlo store not found; set MVNMLE_DATA_DIR (Mac: Dev/datasets, "
        "Forge: /mnt/data/pystatistics-datasets).")


def load(key: str) -> Dataset:
    """Load a stored dataset as an fp64 (n, p) matrix with column names."""
    path = _store_dir() / f"{key}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"{key}.h5 not found in {path.parent}")
    with h5py.File(path, "r") as f:
        vals = np.asarray(f["data/values"][:], dtype=np.float64)
        cols = [c.decode() for c in f["columns/raw_names"][:]]
    return Dataset(key=key, why=_WHY.get(key, ""), values=vals, columns=cols)


# ---------------------------------------------------------------------------
# Statistic functions — MUST match r_reference.R::STAT exactly (stype='i').
# ---------------------------------------------------------------------------

def stat_corr(data: NDArray, idx: NDArray) -> NDArray:
    x = data[idx]
    return np.array([np.corrcoef(x[:, 0], x[:, 1])[0, 1]])


def stat_ratio(data: NDArray, idx: NDArray) -> NDArray:
    x = data[idx]
    return np.array([x[:, 1].sum() / x[:, 0].sum()])   # sum(x)/sum(u)


def _col0(data: NDArray, idx: NDArray) -> NDArray:
    """Column 0 of a stored (n,p) matrix, or the flat array if 1-D."""
    return data[idx, 0] if data.ndim == 2 else data[idx]


def stat_mean(data: NDArray, idx: NDArray) -> NDArray:
    return np.array([_col0(data, idx).mean()])


def stat_median(data: NDArray, idx: NDArray) -> NDArray:
    return np.array([np.median(_col0(data, idx))])


def stat_variance(data: NDArray, idx: NDArray) -> NDArray:
    return np.array([np.var(_col0(data, idx), ddof=1)])


STAT = {
    "corr": stat_corr, "ratio": stat_ratio, "mean": stat_mean,
    "median": stat_median, "variance": stat_variance,
}


def two_sample(key: str = "sleep") -> tuple[NDArray, NDArray]:
    """Return (x, y) 1-D groups from a two-column [value, group] dataset."""
    d = load(key)
    val, grp = d.values[:, 0], d.values[:, 1]
    codes = np.unique(grp)
    assert len(codes) == 2, f"{key} must have exactly 2 groups"
    return val[grp == codes[0]], val[grp == codes[1]]


def two_sample_matrix(key: str = "sleep") -> NDArray:
    """Return the [value, group] matrix as fed to the R perm_exact enumerator."""
    return load(key).values


def mean_diff(x: NDArray, y: NDArray) -> float:
    return float(np.mean(x) - np.mean(y))


# ---------------------------------------------------------------------------
# Coverage-study DGP: a seeded generator with a KNOWN sampling distribution.
# For a mean statistic on N(mu, sigma) data, the nominal-95% CI should cover mu
# ~95% of the time. The master seed is recorded in the artifact (R6/R17).
# ---------------------------------------------------------------------------

COVERAGE_DGP = {
    "family": "lognormal",   # skewed -> where percentile != BCa coverage differ
    "meanlog": 0.0, "sdlog": 0.5,
    "n": 40,
    "true_mean": float(np.exp(0.0 + 0.5 * 0.5**2)),  # E[X] of lognormal
}


def coverage_samples(n_rep: int, master_seed: int) -> list[NDArray]:
    """Draw ``n_rep`` independent samples from the coverage DGP (seeded)."""
    rng = np.random.default_rng(master_seed)
    n = COVERAGE_DGP["n"]
    return [np.exp(rng.normal(COVERAGE_DGP["meanlog"], COVERAGE_DGP["sdlog"], n))
            for _ in range(n_rep)]
