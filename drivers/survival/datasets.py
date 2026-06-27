"""Curate the canonical survival validation datasets (survival::lung).

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
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"
_R_PREP = _HERE / "_r" / "prep_lung.R"

COX_COVARIATES = ["age", "sex", "ph.ecog"]


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
