"""timeseries validation data layer — canonical series + derived hard cases.

One job: load the univariate reference series from the central HDF5 store
(``DATASETS_ROOT``) and expose them, plus a handful of deterministically
derived R10 hard cases (short, missing-value, pure-noise, near-non-invertible),
each with its seasonal period. Every case carries the values as a plain fp64
array so the SAME numbers feed pystatistics and R (the R reference reads them
back from a dumped file — the float32-store shared-input discipline, R17).

No CSVs live in the repo; the store is the single source of truth.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import h5py

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from store_io import DEFAULT_NAMESPACE, store_root  # noqa: E402

# Seasonal period of each stored series (a fixed known property of the named
# series; monthly series are m=12, annual series m=1).
_PERIOD = {
    "air_passengers": 12,
    "lynx": 1,
    "sunspot_year": 1,
    "nile": 1,
    "co2": 12,
    "sim_arma11": 1,
    "sim_nearunitroot": 1,
}


@dataclass(frozen=True)
class Series:
    """One univariate validation series."""

    key: str
    why: str
    y: NDArray[np.float64]
    period: int
    seed: int | None = None
    dgp: str | None = None


def _store_dir() -> Path:
    """Directory holding this project's curated datasets.

    Store resolution lives in ``drivers/_shared/store_io.py`` — the single place
    that knows where the store is. Do not reintroduce a local search path here.
    """
    return store_root() / DEFAULT_NAMESPACE


def load_series(key: str) -> Series:
    """Load a stored series as fp64, tagged with its seasonal period."""
    path = _store_dir() / f"{key}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"{key}.h5 not found in {path.parent}")
    with h5py.File(path, "r") as f:
        y = np.asarray(f["data/values"][:, 0], dtype=np.float64)
    return Series(key=key, why=_WHY[key], y=y, period=_PERIOD[key])


_WHY = {
    "air_passengers": "monthly airline totals, m=12 — multiplicative seasonal, "
                      "the seasonal-ARIMA / multiplicative-ETS benchmark",
    "lynx": "annual lynx trappings — long, strongly cyclic (~10 yr)",
    "sunspot_year": "annual sunspot numbers — long, cyclic (~11 yr)",
    "nile": "annual Nile flow — contains the ~1899 level shift",
    "co2": "monthly Mauna Loa CO2, m=12 — strong trend+season, the STL benchmark",
    "sim_arma11": "seeded ARMA(1,1) ar=0.6 ma=0.4 — exact-recovery known DGP",
    "sim_nearunitroot": "seeded AR(1) ar=0.95 — near-unit-root hard case",
}


CANONICAL = list(_PERIOD)


# --------------------------------------------------------------------------
# Deterministically derived R10 hard cases. Generated here (seeded) and dumped
# for R so both engines fit identical numbers.
# --------------------------------------------------------------------------

def short_series(n: int = 20) -> Series:
    """First n points of AirPassengers — the short-series regime."""
    base = load_series("air_passengers")
    return Series("short_air", f"first {n} obs of AirPassengers — short series",
                  base.y[:n].copy(), period=12)


def missing_series() -> Series:
    """AirPassengers with a handful of interior NaNs — NA-handling vs R."""
    base = load_series("air_passengers")
    y = base.y.copy()
    for i in (17, 53, 90):
        y[i] = np.nan
    return Series("na_air", "AirPassengers with 3 interior NA — NA handling",
                  y, period=12)


def pure_noise(n: int = 300, seed: int = 20260703) -> Series:
    """Seeded iid Gaussian white noise — auto_arima should pick (0,0,0)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    return Series("pure_noise", "iid Gaussian noise — auto_arima -> (0,0,0)?",
                  y, period=1, seed=seed, dgp="iid N(0,1)")


def near_noninvertible(n: int = 300, seed: int = 20260703) -> Series:
    """MA(1) with theta near -1 (near-non-invertible) — the ML boundary case."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n + 1)
    theta = 0.95
    y = e[1:] + theta * e[:-1]
    return Series("near_noninv", f"MA(1) theta={theta} — near-non-invertible",
                  y, period=1, seed=seed, dgp=f"MA(1) theta={theta}")
