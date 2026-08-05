"""Curate the canonical generalized-linear-mixed-model (GLMM) validation datasets.

One job: load the GLMM reference designs from the centralized HDF5 store (R17)
and package each as a :class:`GLMMDataset` carrying EVERYTHING both engines need
to fit the identical model:

- the matrix-API inputs for ``pystatistics.mixed.glmm`` (``y``, fixed design
  ``X`` with an explicit intercept column, ``groups`` of integer codes,
  ``random_effects`` term lists, ``random_data`` slope arrays, a ``family`` /
  ``link`` selector), and
- the R bridge: a flat data frame (response + fixed vars + grouping factors +
  slope vars) plus the matching ``lme4::glmer`` formula + family/link.

``glmm()`` takes a single response VECTOR with unit prior weights (no aggregated
binomial, no offset), so ``cbpp`` is loaded ALREADY EXPANDED to Bernoulli rows
(see ``Dev/datasets/generate_glmm_datasets.py``); ``glmer`` is fit on that same
expanded frame, so the two Laplace fits are directly comparable.

Source of truth is the centralized store (R17): ``cbpp.h5``, ``grouseticks.h5``,
``glmm_slope_synth.h5`` live in the ``pystatistics`` namespace of the central
store, reached via ``DATASETS_ROOT``. Continuous
columns are promoted to float64 on load; grouping / fixed factors are read back
as integer codes. Because the loaded array is the single source handed to BOTH
engines (Python fits it; R reads the identical values via the CSV the driver
dumps), agreement is to fp64 round-off of the SAME numbers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from store_io import store_h5_path  # noqa: E402


@dataclass(frozen=True)
class GLMMDataset:
    """A curated GLMM design + the model both engines fit.

    ``X``/``y``/``groups``/``random_effects``/``random_data``/``family``/``link``
    feed ``pystatistics.mixed.glmm`` directly. ``r_frame`` + ``r_formula`` +
    ``r_family``/``r_link`` feed ``lme4::glmer`` (Laplace, ``nAGQ=1``).
    ``expect_singular`` flags an R10 boundary case.
    """

    key: str
    why: str
    y: NDArray                              # (n,)
    X: NDArray                             # (n, p) fixed design (col 0 = intercept)
    fixed_names: list[str]                  # length p
    groups: dict[str, NDArray]             # name -> int group codes (n,)
    random_effects: dict[str, list[str]]    # name -> term list, e.g. {'g': ['1','x']}
    random_data: dict[str, NDArray]         # slope-term name -> array (n,)
    family: str                             # 'binomial' | 'poisson' | 'gaussian'
    link: str                               # 'logit' | 'log' | 'identity' | 'probit'
    r_frame: pd.DataFrame                   # flat frame R fits
    r_formula: str                          # glmer formula, e.g. "y ~ x + (x | g)"
    r_family: str                           # R family name
    r_link: str                             # R link name
    r_factor_cols: tuple[str, ...] = ()     # ALL columns R must as.factor()
    #                                         (grouping factors + fixed factors)
    expect_singular: bool = False

    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def p(self) -> int:
        return int(self.X.shape[1])


def _store_h5(stem: str) -> Path:
    """Locate ``<stem>.h5`` in the central store, else fail loud.

    Store resolution lives in ``drivers/_shared/store_io.py`` — the single place
    that knows where the store is. Do not reintroduce a local search path here.
    """
    return store_h5_path(stem)


def _load_columns(stem: str) -> dict[str, NDArray]:
    """Read each column by name, promoting continuous to float64 and nominal to
    integer codes. Fails loud on any non-finite cell."""
    import h5py

    path = _store_h5(stem)
    with h5py.File(path, "r") as f:
        values = np.asarray(f["data/values"][:], dtype=np.float64)
        names = [s.decode() if isinstance(s, bytes) else str(s)
                 for s in f["columns/raw_names"][:]]
        dtypes = [s.decode() if isinstance(s, bytes) else str(s)
                  for s in f["columns/data_types"][:]]
    if not np.isfinite(values).all():
        n_bad = int((~np.isfinite(values)).sum())
        raise ValueError(f"{stem}.h5: {n_bad} non-finite cell(s); the GLMM "
                         f"reference datasets must be fully observed.")
    out: dict[str, NDArray] = {}
    for j, (nm, dt) in enumerate(zip(names, dtypes)):
        col = values[:, j]
        out[nm] = (np.rint(col).astype(np.int64)
                   if dt in ("nominal", "ordinal") else col)
    return out


def _treatment_dummies(codes: NDArray) -> tuple[NDArray, list[int]]:
    """Treatment-coded dummies for a factor (reference = lowest code), matching
    R's default contrasts on a sorted factor. Returns (dummies (n, k-1), levels)."""
    levels = sorted(int(v) for v in np.unique(codes))
    non_ref = levels[1:]
    D = np.column_stack([(codes == lv).astype(float) for lv in non_ref]) \
        if non_ref else np.empty((codes.size, 0))
    return D, non_ref


def load_cbpp() -> GLMMDataset:
    """Bernoulli-expanded lme4::cbpp. y ~ period + (1|herd): binomial/logit."""
    c = _load_columns("cbpp")
    y = c["y"].astype(float)
    period, herd = c["period"], c["herd"]
    D, non_ref = _treatment_dummies(period)
    X = np.column_stack([np.ones(y.size), D])
    fixed_names = ["(Intercept)"] + [f"period{lv + 1}" for lv in non_ref]
    frame = pd.DataFrame({"y": y, "period": period, "herd": herd})
    return GLMMDataset(
        key="cbpp",
        why="Binomial/logit GLMM with a random intercept and a 4-level fixed "
            "factor (period); lme4's canonical glmer example, expanded to "
            "Bernoulli so glmm()'s vector response matches glmer.",
        y=y, X=X, fixed_names=fixed_names,
        groups={"herd": herd}, random_effects={"herd": ["1"]}, random_data={},
        family="binomial", link="logit",
        r_frame=frame, r_formula="y ~ period + (1 | herd)",
        r_family="binomial", r_link="logit",
        r_factor_cols=("period", "herd"))


def load_cbpp_probit() -> GLMMDataset:
    """cbpp with a PROBIT link (Family instance path). y ~ period + (1|herd)."""
    ds = load_cbpp()
    return GLMMDataset(
        key="cbpp_probit",
        why="Same cbpp design under a PROBIT link — exercises glmm()'s "
            "Binomial(link='probit') Family-instance path (no string shortcut) "
            "vs glmer(family=binomial('probit')).",
        y=ds.y, X=ds.X, fixed_names=ds.fixed_names,
        groups=ds.groups, random_effects=ds.random_effects, random_data={},
        family="binomial", link="probit",
        r_frame=ds.r_frame, r_formula=ds.r_formula,
        r_family="binomial", r_link="probit",
        r_factor_cols=("period", "herd"))


def load_grouseticks() -> GLMMDataset:
    """lme4::grouseticks. TICKS ~ cHEIGHT + (1|BROOD): Poisson/log."""
    c = _load_columns("grouseticks")
    y = c["TICKS"].astype(float)
    cheight, brood = c["cHEIGHT"].astype(float), c["BROOD"]
    X = np.column_stack([np.ones(y.size), cheight])
    frame = pd.DataFrame({"TICKS": y, "cHEIGHT": cheight, "BROOD": brood})
    return GLMMDataset(
        key="grouseticks",
        why="Poisson/log GLMM with a random intercept and one continuous "
            "covariate; lme4's canonical count example (118 broods).",
        y=y, X=X, fixed_names=["(Intercept)", "cHEIGHT"],
        groups={"BROOD": brood}, random_effects={"BROOD": ["1"]}, random_data={},
        family="poisson", link="log",
        r_frame=frame, r_formula="TICKS ~ cHEIGHT + (1 | BROOD)",
        r_family="poisson", r_link="log",
        r_factor_cols=("BROOD",))


def load_slope_synth() -> GLMMDataset:
    """Deterministic synthetic binomial GLMM. y ~ x + (1 + x | g): correlated
    random intercept + slope (the multi-term RE Laplace path)."""
    c = _load_columns("glmm_slope_synth")
    y, x, g = c["y"].astype(float), c["x"].astype(float), c["g"]
    X = np.column_stack([np.ones(y.size), x])
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return GLMMDataset(
        key="glmm_slope_synth",
        why="Binomial/logit GLMM with a CORRELATED random intercept+slope over "
            "40 groups — exercises the 2x2 RE covariance block under Laplace, "
            "the case with no small canonical Bernoulli dataset in lme4.",
        y=y, X=X, fixed_names=["(Intercept)", "x"],
        groups={"g": g}, random_effects={"g": ["1", "x"]}, random_data={"x": x},
        family="binomial", link="logit",
        r_frame=frame, r_formula="y ~ x + (x | g)",
        r_family="binomial", r_link="logit",
        r_factor_cols=("g",))


# Registry: stable key -> loader. Drives the GLMM correctness sweep.
GLMM_LOADERS = {
    "cbpp": load_cbpp,
    "grouseticks": load_grouseticks,
    "glmm_slope_synth": load_slope_synth,
    "cbpp_probit": load_cbpp_probit,
}
