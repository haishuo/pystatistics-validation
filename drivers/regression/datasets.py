"""Curate the canonical regression validation datasets.

One job: produce the exact, deterministic numeric designs that both pystatistics
and R fit, so a coefficient-for-coefficient comparison is meaningful. No fitting,
no timing, no R invocation for the model itself — just the data.

Three real datasets, each chosen because it is the textbook target for its family
(rationale preserved from the archived suite-1 references):

- **California Housing** (`MedHouseVal` + derived columns) — OLS, binomial, Poisson.
  ~20k California census blocks, 8 numeric predictors. Large enough that the OLS
  normal-equations/QR path and the binomial/Poisson IRLS loops are exercised on a
  non-trivial design.
- **airquality** (`datasets::airquality`) — Gamma(log). Daily NYC ozone, 111
  complete cases. Ozone is positive, continuous, right-skewed: the textbook Gamma
  regression target.
- **quine** (`MASS::quine`) — negative binomial. 146 Australian schoolchildren,
  days absent. THE canonical NB dataset (McCullagh & Nelder).

The California derivations are deterministic functions of the raw data (percentile
and median cutpoints — no randomness), so R and Python land on identical rows. The
airquality/quine CSVs are emitted once from R (so Python sees R's exact model
matrix, including factor coding) and committed under ``data/`` for reproducibility.

Contract: every loader returns ``(X, y, spec)`` where ``X`` includes a leading
intercept column, ``y`` is the response, and ``spec`` documents the model.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
_DATA = _HERE / "data"
_REPO = _HERE.parent.parent  # pystatistics-validation/
_RAW_CALIFORNIA = _REPO / "data" / "california_housing.csv"
_R_PREP = _HERE / "_r" / "prep_datasets.R"


@dataclass(frozen=True)
class ModelSpec:
    """What model is being fit on a curated design (documentation + R bridge)."""

    key: str                       # stable id, e.g. "ols", "glm_binomial"
    family: str                    # pystatistics family ('gaussian' uses OLS path)
    r_family: str                  # how R names it: 'lm', 'binomial', ...
    link: str                      # 'identity', 'log', 'logit', ...
    formula: str                   # human-readable model formula
    dataset: str                   # dataset id
    why: str                       # why this dataset for this family
    coef_names: list[str] = field(default_factory=list)


# ── California Housing ────────────────────────────────────────────────────────

_CAL_NUMERIC = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def _california_prepared() -> pd.DataFrame:
    """Deterministically derive the prepared California frame from the raw CSV.

    Mirrors the archived ``generate_data.py`` derivations exactly (median /
    percentile cutpoints), so the prepared frame is reproducible and matches the
    R reference rows. No randomness anywhere.
    """
    if not _RAW_CALIFORNIA.is_file():
        raise FileNotFoundError(
            f"raw California Housing CSV not found at {_RAW_CALIFORNIA}; "
            "it ships in the validation repo's data/ directory.")
    df = pd.read_csv(_RAW_CALIFORNIA).copy()

    # high_value: 1 if MedHouseVal above its median (binomial response).
    df["high_value"] = (df["MedHouseVal"] > df["MedHouseVal"].median()).astype(int)
    # pop_count: Population/100 rounded (Poisson count response).
    df["pop_count"] = np.round(df["Population"] / 100.0).astype(int)
    return df


def load_california_ols() -> tuple[NDArray[np.float64], NDArray[np.float64], ModelSpec]:
    """OLS: ``MedHouseVal ~ all 8 numeric predictors`` (intercept included)."""
    df = _california_prepared()
    X = np.column_stack([np.ones(len(df))] +
                        [df[c].to_numpy(float) for c in _CAL_NUMERIC])
    y = df["MedHouseVal"].to_numpy(float)
    spec = ModelSpec(
        key="ols", family="gaussian", r_family="lm", link="identity",
        formula="MedHouseVal ~ " + " + ".join(_CAL_NUMERIC),
        dataset="california_housing",
        why="Large real OLS design (20k blocks, 8 predictors); the foundational "
            "normal-equations/QR path.",
        coef_names=["(Intercept)", *_CAL_NUMERIC],
    )
    return X, y, spec


def load_california_binomial() -> tuple[NDArray[np.float64], NDArray[np.float64], ModelSpec]:
    """Binomial GLM: ``high_value ~ MedInc + HouseAge + AveRooms + Population``."""
    df = _california_prepared()
    cols = ["MedInc", "HouseAge", "AveRooms", "Population"]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    y = df["high_value"].to_numpy(float)
    spec = ModelSpec(
        key="glm_binomial", family="binomial", r_family="binomial", link="logit",
        formula="high_value ~ " + " + ".join(cols),
        dataset="california_housing",
        why="Logistic regression on a real binary outcome at scale; exercises the "
            "binomial IRLS loop with fitted probabilities spanning (0,1).",
        coef_names=["(Intercept)", *cols],
    )
    return X, y, spec


def load_california_poisson() -> tuple[NDArray[np.float64], NDArray[np.float64], ModelSpec]:
    """Poisson GLM: ``pop_count ~ MedInc + HouseAge + AveOccup`` (log link)."""
    df = _california_prepared()
    cols = ["MedInc", "HouseAge", "AveOccup"]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    y = df["pop_count"].to_numpy(float)
    spec = ModelSpec(
        key="glm_poisson", family="poisson", r_family="poisson", link="log",
        formula="pop_count ~ " + " + ".join(cols),
        dataset="california_housing",
        why="Count regression on a real count outcome; exercises the Poisson "
            "log-link IRLS loop.",
        coef_names=["(Intercept)", *cols],
    )
    return X, y, spec


# ── airquality (Gamma) and quine (NegBin): emitted from R, read here ───────────

def _ensure_r_prepped() -> None:
    """Emit airquality.csv and quine_mm.csv from R if not already present.

    The CSVs are committed; this only runs on a fresh checkout / regeneration. We
    read the model matrix R itself built (``model.matrix``) for quine so Python
    sees the exact factor coding R used.
    """
    airq = _DATA / "airquality.csv"
    quine = _DATA / "quine_mm.csv"
    if airq.is_file() and quine.is_file():
        return
    if not _R_PREP.is_file():
        raise FileNotFoundError(f"R prep script missing: {_R_PREP}")
    proc = subprocess.run(["Rscript", str(_R_PREP), str(_DATA)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"R dataset prep failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")


def load_airquality_gamma() -> tuple[NDArray[np.float64], NDArray[np.float64], ModelSpec]:
    """Gamma(log) GLM: ``Ozone ~ Solar.R + Temp + Wind`` on complete cases."""
    _ensure_r_prepped()
    df = pd.read_csv(_DATA / "airquality.csv")
    cols = ["Solar.R", "Temp", "Wind"]
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in cols])
    y = df["Ozone"].to_numpy(float)
    spec = ModelSpec(
        key="glm_gamma", family="gamma", r_family="Gamma", link="log",
        formula="Ozone ~ Solar.R + Temp + Wind",
        dataset="airquality",
        why="Ozone is positive, continuous, right-skewed — the textbook Gamma "
            "regression target. 111 complete NYC daily air-quality cases.",
        coef_names=["(Intercept)", *cols],
    )
    return X, y, spec


def load_quine_negbin() -> tuple[NDArray[np.float64], NDArray[np.float64], ModelSpec]:
    """Negative-binomial GLM: ``Days ~ Eth + Sex + Age + Lrn`` (R model matrix)."""
    _ensure_r_prepped()
    df = pd.read_csv(_DATA / "quine_mm.csv")
    predictors = [c for c in df.columns if c != "Days"]
    X = np.column_stack([np.ones(len(df))] +
                        [df[c].to_numpy(float) for c in predictors])
    y = df["Days"].to_numpy(float)
    spec = ModelSpec(
        key="glm_negbin", family="negative.binomial", r_family="negbin", link="log",
        formula="Days ~ Eth + Sex + Age + Lrn",
        dataset="quine",
        why="THE canonical negative-binomial dataset (McCullagh & Nelder): 146 "
            "Australian schoolchildren, overdispersed absence counts.",
        coef_names=["(Intercept)", *predictors],
    )
    return X, y, spec


# Registry: stable key → loader. Drives the correctness sweep.
LOADERS = {
    "ols": load_california_ols,
    "glm_binomial": load_california_binomial,
    "glm_poisson": load_california_poisson,
    "glm_gamma": load_airquality_gamma,
    "glm_negbin": load_quine_negbin,
}
