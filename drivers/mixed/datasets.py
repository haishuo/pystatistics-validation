"""Curate the canonical linear-mixed-model (LMM) validation datasets.

One job: load the five textbook ``lme4`` datasets from the centralized HDF5 store
(R17) and package each as a :class:`MixedDataset` carrying EVERYTHING both engines
need to fit the identical model:

- the matrix-API inputs for ``pystatistics.mixed.lmm`` (``y``, fixed design ``X``
  with an explicit intercept column, ``groups`` of integer codes,
  ``random_effects`` term lists, ``random_data`` slope arrays), and
- the R bridge: a flat data frame (response + fixed vars + grouping factors +
  slope vars) plus the matching ``lme4::lmer`` formula string.

No fitting, no timing, no R invocation here — just the data and the model recipe,
so a coefficient-for-coefficient / variance-component comparison is meaningful.

Source of truth is the centralized store (R17): ``sleepstudy.h5``,
``penicillin.h5``, ``dyestuff.h5``, ``dyestuff2.h5``, ``pastes.h5`` live in
the ``pystatistics`` namespace of the central store, reached via
``DATASETS_ROOT``. ``/data/values`` is float32; continuous columns are promoted
to float64 on load and grouping factors are read back as integer codes from the
column ``category_map``. Because the loaded array is the single source handed to
BOTH engines (Python fits it directly; R reads the identical values via the CSV
the driver dumps), agreement is to fp64 round-off of the SAME numbers — not
contaminated by an input mismatch (the multivariate precedent).
"""

from __future__ import annotations

import json
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
class MixedDataset:
    """A curated LMM design + the model both engines fit.

    ``X``/``y``/``groups``/``random_effects``/``random_data`` feed
    ``pystatistics.mixed.lmm`` directly. ``r_frame`` + ``r_formula`` feed
    ``lme4::lmer`` / ``lmerTest::lmer``. ``reml`` selects the criterion (both
    engines). ``expect_singular`` flags the R10 boundary case (lme4 emits
    'boundary (singular) fit').
    """

    key: str
    why: str
    y: NDArray                              # (n,)
    X: NDArray                             # (n, p) fixed design (col 0 = intercept)
    fixed_names: list[str]                  # length p, e.g. ['(Intercept)', 'Days']
    groups: dict[str, NDArray]             # name -> int group codes (n,)
    random_effects: dict[str, list[str]]    # name -> term list, e.g. {'Subject': ['1','Days']}
    random_data: dict[str, NDArray]         # slope-term name -> array (n,)
    r_frame: pd.DataFrame                   # flat frame R fits
    r_formula: str                          # lme4 formula, e.g. "y ~ Days + (Days | Subject)"
    reml: bool = True
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
    integer codes. Fails loud on any non-finite cell (these designs are fully
    observed; a NaN would silently corrupt the fit)."""
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
        raise ValueError(f"{stem}.h5: {n_bad} non-finite cell(s); the LMM "
                         f"reference datasets must be fully observed.")
    out: dict[str, NDArray] = {}
    for j, (nm, dt) in enumerate(zip(names, dtypes)):
        col = values[:, j]
        out[nm] = np.rint(col).astype(np.int64) if dt == "nominal" else col
    return out


def load_sleepstudy() -> MixedDataset:
    """Reaction ~ Days + (Days | Subject): random slopes, correlated int/slope."""
    c = _load_columns("sleepstudy")
    y, days, subj = c["Reaction"].astype(float), c["Days"].astype(float), c["Subject"]
    X = np.column_stack([np.ones(y.size), days])
    frame = pd.DataFrame({"Reaction": y, "Days": days, "Subject": subj})
    return MixedDataset(
        key="sleepstudy",
        why="Random slopes with a correlated intercept/slope -- the canonical "
            "(Days|Subject) example; exercises the 2x2 RE covariance block.",
        y=y, X=X, fixed_names=["(Intercept)", "Days"],
        groups={"Subject": subj},
        random_effects={"Subject": ["1", "Days"]},
        random_data={"Days": days},
        r_frame=frame, r_formula="Reaction ~ Days + (Days | Subject)")


def load_penicillin() -> MixedDataset:
    """diameter ~ 1 + (1|plate) + (1|sample): fully crossed random intercepts."""
    c = _load_columns("penicillin")
    y, plate, sample = c["diameter"].astype(float), c["plate"], c["sample"]
    X = np.ones((y.size, 1))
    frame = pd.DataFrame({"diameter": y, "plate": plate, "sample": sample})
    return MixedDataset(
        key="penicillin",
        why="Fully CROSSED random effects (plate x sample): two independent "
            "grouping factors, intercept-only fixed part.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"plate": plate, "sample": sample},
        random_effects={"plate": ["1"], "sample": ["1"]},
        random_data={},
        r_frame=frame, r_formula="diameter ~ 1 + (1 | plate) + (1 | sample)")


def load_dyestuff() -> MixedDataset:
    """Yield ~ 1 + (1|Batch): the simplest one-way random intercept."""
    c = _load_columns("dyestuff")
    y, batch = c["Yield"].astype(float), c["Batch"]
    X = np.ones((y.size, 1))
    frame = pd.DataFrame({"Yield": y, "Batch": batch})
    return MixedDataset(
        key="dyestuff",
        why="Simplest one-way random-intercept LMM; between-batch variance well "
            "away from the boundary -- the clean baseline.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"Batch": batch}, random_effects={"Batch": ["1"]}, random_data={},
        r_frame=frame, r_formula="Yield ~ 1 + (1 | Batch)")


def load_dyestuff2() -> MixedDataset:
    """Yield ~ 1 + (1|Batch): SINGULAR fit (between-batch variance -> 0). R10."""
    c = _load_columns("dyestuff2")
    y, batch = c["Yield"].astype(float), c["Batch"]
    X = np.ones((y.size, 1))
    frame = pd.DataFrame({"Yield": y, "Batch": batch})
    return MixedDataset(
        key="dyestuff2",
        why="Variance component AT THE BOUNDARY: between-batch variance estimates "
            "to 0, so lmer emits 'boundary (singular) fit'. The R10 singular case "
            "-- pystatistics must match the behaviour, not just easy-data numbers.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"Batch": batch}, random_effects={"Batch": ["1"]}, random_data={},
        r_frame=frame, r_formula="Yield ~ 1 + (1 | Batch)",
        expect_singular=True)


def load_pastes() -> MixedDataset:
    """strength ~ 1 + (1|batch) + (1|sample): NESTED (sample nested in batch)."""
    c = _load_columns("pastes")
    y, batch, sample = c["strength"].astype(float), c["batch"], c["sample"]
    X = np.ones((y.size, 1))
    frame = pd.DataFrame({"strength": y, "batch": batch, "sample": sample})
    return MixedDataset(
        key="pastes",
        why="NESTED random effects expressed as two crossed factors: `sample` is "
            "the batch:cask label (globally unique), so (1|batch)+(1|sample) is "
            "the nested model -- exactly how the matrix API expresses nesting.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"batch": batch, "sample": sample},
        random_effects={"batch": ["1"], "sample": ["1"]}, random_data={},
        r_frame=frame, r_formula="strength ~ 1 + (1 | batch) + (1 | sample)")


# Registry: stable key -> loader. Drives the LMM correctness sweep.
LMM_LOADERS = {
    "sleepstudy": load_sleepstudy,
    "penicillin": load_penicillin,
    "dyestuff": load_dyestuff,
    "dyestuff2": load_dyestuff2,
    "pastes": load_pastes,
}
