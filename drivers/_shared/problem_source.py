"""One resolver for MVN MLE benchmark problems, by dataset name.

One job: turn a dataset name + p into a problem object, hiding whether the
problem is simulated (``mvn_sim``: ``simw``/``simg``) or store-backed (a
``surveys/<name>.h5`` file curated by ``survey_io``). Every mvnmle driver
resolves its inputs through here, so simulated and store-backed problems are
interchangeable in the grids.

Store-backed resolution has one special case: when ``p`` equals the file's full
column count, the file *is* the curated problem (e.g. ``nhanes_cardio``, an
11-variable extract curated at build time), so the eligibility band and
column-ranking machinery — designed to pick p of ~600 survey items — is
bypassed and all columns are read in file order. Fully-observed columns (an
``age`` recorded for every respondent) would otherwise be rejected by the
observed-rate band, which exists to keep *survey items* comparable, not to
forbid complete demographic columns in a purpose-built extract.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from mvn_sim import PROFILE_SEEDS, SimProblem, make_sim_problem
from store_io import SURVEY_NAMESPACE, store_root
from survey_io import MVNProblem, build_mvn_problem

SIM_NAMES = tuple(sorted(PROFILE_SEEDS))


def resolve_problem(name: str, p: int, *,
                    max_abs_corr: float | None = 0.99) -> MVNProblem | SimProblem:
    """Return the ``(name, p)`` benchmark problem, simulated or store-backed."""
    if name in PROFILE_SEEDS:
        return make_sim_problem(name, p)

    path = store_root() / SURVEY_NAMESPACE / f"{name}.h5"
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset {name!r} is neither a sim profile ({', '.join(SIM_NAMES)}) "
            f"nor a store file ({path})")

    with h5py.File(path, "r") as f:
        n_cols = f["/data/values"].shape[1]
    if p == n_cols:
        return _whole_file_problem(path, name)
    return build_mvn_problem(path, p, max_abs_corr=max_abs_corr)


def _whole_file_problem(path: Path, name: str) -> MVNProblem:
    """Read every column of a purpose-built extract, in file order."""
    with h5py.File(path, "r") as f:
        values = f["/data/values"][:].astype(np.float32)
        missing = f["/data/missing_state"][:] != 0
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f["/columns/display_names"][:]]
        source = str(f["/metadata"].attrs.get("source", path.name))
    values[missing] = np.nan
    keep = ~np.all(np.isnan(values), axis=1)
    x = values[keep]
    return MVNProblem(
        dataset_name=name,
        source=source,
        X=x,
        column_indices=np.arange(x.shape[1], dtype=np.intp),
        column_names=names,
        n_rows_original=values.shape[0],
        overall_missing_frac=float(np.isnan(x).mean()),
    )
