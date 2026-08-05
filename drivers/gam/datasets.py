"""GAM validation datasets — real (HDF5 store) + seeded simulated.

Real datasets come from the central HDF5 store (``MVNMLE_DATA_DIR``); the
simulated multi-smooth designs are generated deterministically by a seeded R
``mgcv::gamSim`` call so BOTH engines fit the identical dumped values (the
seed and mgcv version are recorded in every artifact). No CSVs live in the
repo; each case dumps its data to a temp CSV the R reference and pystatistics
both read.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from store_io import store_h5_path  # noqa: E402


@dataclass(frozen=True)
class GamDataset:
    """One validation dataset: the response and the smooth/parametric data."""

    key: str
    why: str
    y: NDArray[np.floating]
    smooth_data: dict[str, NDArray[np.floating]]
    parametric: dict[str, NDArray[np.floating]] = field(default_factory=dict)
    family: str = "gaussian"
    seed: int | None = None


def _store_h5(stem: str) -> Path:
    """Locate ``<stem>.h5`` in the central store, else fail loud.

    Store resolution lives in ``drivers/_shared/store_io.py`` — the single place
    that knows where the store is. Do not reintroduce a local search path here.
    """
    return store_h5_path(stem)


def load_mcycle() -> GamDataset:
    """MASS::mcycle — the canonical 1-D smoothing benchmark (n=133)."""
    import h5py

    with h5py.File(_store_h5("mcycle"), "r") as f:
        v = np.asarray(f["data/values"][:], dtype=np.float64)
    return GamDataset(
        key="mcycle",
        why="The canonical 1-D penalised-spline benchmark: heteroscedastic, "
            "a sharp trough. R15 default-invocation case.",
        y=v[:, 1], smooth_data={"times": v[:, 0]}, family="gaussian",
    )


_GAMSIM_R = r"""
suppressMessages(library(mgcv))
args <- commandArgs(trailingOnly = TRUE)
seed <- as.integer(args[[1]]); dist <- args[[2]]
n <- as.integer(args[[3]]); scale <- as.numeric(args[[4]]); out <- args[[5]]
set.seed(seed)
d <- gamSim(1, n = n, dist = dist, scale = scale, verbose = FALSE)
write.csv(d[, c("y", "x0", "x1", "x2", "x3")], out, row.names = FALSE)
"""


def gamsim(dist: str, *, seed: int, n: int, scale: float) -> GamDataset:
    """mgcv::gamSim eg.1 — 4 smooths (x0,x1,x2,x3), the Gu-Wahba functions.

    x2 is strongly nonlinear, x3 is null (flat). Deterministic given seed +
    the installed mgcv version (both recorded in the artifact env).
    """
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "gs.R"
        script.write_text(_GAMSIM_R)
        csv = Path(td) / "gs.csv"
        subprocess.run(
            ["Rscript", str(script), str(seed), dist, str(n), str(scale),
             str(csv)],
            check=True, capture_output=True, text=True,
        )
        arr = np.genfromtxt(csv, delimiter=",", names=True)
    fam = {"normal": "gaussian", "poisson": "poisson",
           "binary": "binomial"}[dist]
    return GamDataset(
        key=f"gamsim_{dist}",
        why=f"mgcv::gamSim 4-smooth {fam} design (seed {seed}): recovers "
            f"known f0-f3 incl. a strongly nonlinear (x2) and a null (x3) "
            f"term; multi-smooth correctness.",
        y=arr["y"].astype(float),
        smooth_data={f"x{i}": arr[f"x{i}"].astype(float) for i in range(4)},
        family=fam, seed=seed,
    )


def write_case_csv(ds: GamDataset, path: Path, response: str = "y") -> None:
    """Dump a dataset to CSV (full fp64 precision) for the R reference."""
    cols: dict[str, NDArray] = {response: ds.y}
    cols.update(ds.smooth_data)
    cols.update(ds.parametric)
    names = list(cols)
    with open(path, "w", newline="") as fh:
        fh.write(",".join(names) + "\n")
        rows = zip(*[cols[n] for n in names])
        for row in rows:
            fh.write(",".join(repr(float(v)) for v in row) + "\n")
