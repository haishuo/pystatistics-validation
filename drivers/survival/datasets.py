"""Curate the survival validation datasets (survival::lung; survival::flchain).

One job: produce the exact, deterministic arrays that both pystatistics and R
fit, so a quantity-for-quantity comparison is meaningful. No fitting, no timing —
just the data.

The NCCTG advanced lung cancer dataset (Loprinzi et al. 1994) is R's own
canonical teaching example for survival analysis — KM fits the overall curve,
log-rank compares survival by sex, and Cox PH regresses on age + sex + ph.ecog.
It is a real survival process (unlike a housing-age proxy), so Cox PH converges
and the comparison against R is meaningful.

The CSVs are emitted once from R (``_r/prep_lung.R``) with the documented
complete-case NA drop and committed under ``data/``, so Python reads the EXACT
rows R fit. Two designs, each complete-cased over only the columns it uses:

- ``lung_km.csv``    — time, event, sex  (n=228, 165 events): KM + log-rank.
- ``lung_coxph.csv`` — time, event, age, sex, ph.ecog  (n=227, 164 events):
  Cox PH + discrete-time.

Contract: ``load_lung_km()`` returns ``(time, event, sex)``; ``load_lung_cox()``
returns ``(time, event, X, names)`` where X has NO intercept (Cox/discrete have
no intercept term).

The **flchain** dataset (Dispenzieri et al. 2012; ``survival::flchain``) is the
large real cohort for the discrete-time / GPU study: n=7874 subjects, 2169
deaths, follow-up to 5215 days. Read from the curated HDF5 (schema 1.0.0 — see
``Dev/datasets/SCHEMA.md``) via ``MVNMLE_DATA_DIR`` (Mac: ``Dev/datasets``;
Forge: ``/mnt/data/pystatistics-datasets``). Expanded to person-period, it yields
~83k rows yearly → ~952k monthly → ~1.9M biweekly, the scale at which the
discrete-time GPU claim is made.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"
_R_PREP = _HERE / "_r" / "prep_lung.R"

COX_COVARIATES = ["age", "sex", "ph.ecog"]

# flchain covariates fed to the discrete-time person-period logistic model. The
# only column carrying missingness (``creat``, 1350 NA cells) is intentionally
# dropped so all 7874 subjects are kept (the keep-all-rows choice, maximizing
# person-period scale). Binary indicators (sex, mgus) are left as 0/1; every
# other covariate is z-scored at load (see ``load_flchain``) to keep the linear
# predictor O(1) and the float32 GPU solve well-conditioned.
FLCHAIN_COVARIATES = ["age", "sex", "kappa", "lambda", "flc_grp", "mgus", "sampleyr"]
_FLCHAIN_BINARY = {"sex", "mgus"}
_FLCHAIN_DROPPED = {"creat"}


def _ensure_r_prepped() -> None:
    """Emit lung_km.csv + lung_coxph.csv from R if not already present.

    The CSVs are committed; this only runs on a fresh checkout / regeneration.
    """
    km = _DATA / "lung_km.csv"
    cox = _DATA / "lung_coxph.csv"
    if km.is_file() and cox.is_file():
        return
    if not _R_PREP.is_file():
        raise FileNotFoundError(f"R prep script missing: {_R_PREP}")
    proc = subprocess.run(["Rscript", str(_R_PREP), str(_DATA)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"R lung prep failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")


def load_lung_km() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """KM / log-rank design: ``(time, event, sex)`` on survival::lung (n=228)."""
    _ensure_r_prepped()
    df = pd.read_csv(_DATA / "lung_km.csv")
    return (df["time"].to_numpy(float),
            df["event"].to_numpy(float),
            df["sex"].to_numpy(np.intp))


def load_lung_cox() -> tuple[NDArray[np.float64], NDArray[np.float64],
                             NDArray[np.float64], list[str]]:
    """Cox / discrete-time design: ``(time, event, X, names)`` (n=227).

    ``X`` is the (n, 3) covariate matrix age + sex + ph.ecog with NO intercept
    column — Cox and discrete-time models have no intercept.
    """
    _ensure_r_prepped()
    df = pd.read_csv(_DATA / "lung_coxph.csv")
    time_ = df["time"].to_numpy(float)
    event = df["event"].to_numpy(float)
    X = df[COX_COVARIATES].to_numpy(float)
    return time_, event, X, list(COX_COVARIATES)


def _flchain_h5_path() -> Path:
    """Locate ``flchain.h5`` via ``MVNMLE_DATA_DIR`` (Mac/Forge), else fail loud."""
    root = os.environ.get("MVNMLE_DATA_DIR")
    candidates = []
    if root:
        candidates.append(Path(root) / "flchain.h5")
    candidates += [
        Path("/Volumes/Archive/Documents/Dropbox/Dev/datasets/flchain.h5"),
        Path("/mnt/data/pystatistics-datasets/flchain.h5"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "flchain.h5 not found. Set MVNMLE_DATA_DIR to the curated dataset dir "
        f"(tried: {[str(c) for c in candidates]}).")


def load_flchain(*, standardize: bool = True,
                 ) -> tuple[NDArray[np.float64], NDArray[np.float64],
                            NDArray[np.float64], list[str]]:
    """Load survival::flchain from the curated HDF5 → ``(time, event, X, names)``.

    Reads the schema-1.0.0 layout (``/data/values`` float32, ``/columns/raw_names``,
    ``/columns/semantic_types``, ``/columns/data_types``): the time/event columns
    are located by ``semantic_type`` (``time_to_event`` / ``event_indicator``), not
    by hard-coded position. All 7874 subjects are kept; ``creat`` (the only column
    with missingness) is dropped. ``X`` has the seven :data:`FLCHAIN_COVARIATES`
    in that order, with NO intercept (the discrete-time interval dummies are the
    baseline). When ``standardize`` is set (the default), every non-binary
    covariate is z-scored — this does not change the model, only conditioning, and
    is applied to the identical design handed to both pystatistics and R.
    """
    import h5py

    path = _flchain_h5_path()
    with h5py.File(path, "r") as f:
        values = np.asarray(f["/data/values"][:], dtype=np.float64)
        raw = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["/columns/raw_names"][:]]
        sem = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["/columns/semantic_types"][:]]

    def _col(semantic: str) -> int:
        idx = [i for i, s in enumerate(sem) if s == semantic]
        if len(idx) != 1:
            raise ValueError(
                f"expected exactly one '{semantic}' column in flchain.h5, "
                f"found {len(idx)} (semantic_types={sem})")
        return idx[0]

    time_ = values[:, _col("time_to_event")].astype(np.float64)
    event = values[:, _col("event_indicator")].astype(np.float64)

    name_to_idx = {nm: i for i, nm in enumerate(raw)}
    missing = [c for c in FLCHAIN_COVARIATES if c not in name_to_idx]
    if missing:
        raise ValueError(f"flchain.h5 is missing covariates {missing} (have {raw})")

    cols = [values[:, name_to_idx[c]].astype(np.float64) for c in FLCHAIN_COVARIATES]
    X = np.column_stack(cols)
    if not np.isfinite(X).all():
        raise ValueError("flchain covariate matrix has non-finite cells after "
                         "dropping creat — unexpected missingness")

    if standardize:
        for j, c in enumerate(FLCHAIN_COVARIATES):
            if c in _FLCHAIN_BINARY:
                continue
            col = X[:, j]
            sd = col.std()
            if sd > 0:
                X[:, j] = (col - col.mean()) / sd

    return time_, event, X, list(FLCHAIN_COVARIATES)


def flchain_interval_bounds(time: NDArray, event: NDArray, bin_days: float,
                            ) -> NDArray[np.float64]:
    """Regular ``bin_days``-wide interval start times for the discrete-time model.

    Unlike :func:`discrete_interval_bounds` (quantile bins for the tiny lung
    design), flchain uses genuine calendar bins: the discrete-time grid is
    ``[t0, t0+w, t0+2w, ...]`` from the first event time ``t0`` up to the last
    event time, with the final open-ended interval catching the remainder. The
    bin width sets the person-period scale (yearly→monthly→biweekly sweep). The
    returned bounds match ``discrete_time(intervals=...)`` semantics (ascending
    interval starts).
    """
    event_times = time[event == 1]
    t0 = float(event_times.min())
    t_max = float(event_times.max())
    bounds = np.arange(t0, t_max, float(bin_days), dtype=np.float64)
    if bounds.size == 0:
        bounds = np.array([t0], dtype=np.float64)
    return bounds


def discrete_interval_bounds(time: NDArray, event: NDArray, n_bins: int = 5) -> NDArray:
    """Coarse, well-posed interval boundaries for the discrete-time model.

    Discrete-time survival is designed for genuinely binned time. Using every
    unique event time as its own interval (the ``intervals=None`` default) on
    continuous data produces hundreds of single-event intervals that separate
    perfectly — the logistic baseline blows up and neither pystatistics nor R
    converges cleanly. We instead bin event times into ``n_bins`` quantile
    intervals, a standard and numerically sound discretization, so the
    person-period logistic fit is identified and the R head-to-head is clean.

    Returns the ascending interval start boundaries (the first is the minimum
    event time), matching ``discrete_time(intervals=...)`` semantics.
    """
    event_times = np.unique(time[event == 1])
    if len(event_times) <= n_bins:
        return event_times
    # Quantile cut points over the event-time distribution; deduplicate.
    qs = np.linspace(0.0, 1.0, n_bins + 1)[:-1]   # n_bins lower edges
    bounds = np.unique(np.quantile(event_times, qs))
    return np.asarray(bounds, dtype=np.float64)
