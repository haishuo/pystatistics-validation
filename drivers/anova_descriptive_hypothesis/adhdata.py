"""Curate the datasets for the anova / descriptive / hypothesis validation.

One job: turn the centralized HDF5 datasets (R17) into the exact numeric arrays
the drivers analyse — and deterministically synthesize the red-team hard cases
(ties, zero cells, tiny expected counts, an UNBALANCED factorial where SS
Type I/II/III diverge, a repeated-measures design) that are NOT fixed reference
data.

Shared-input discipline: continuous columns are promoted float32->float64 on
load, and the SAME promoted array is handed to R (via the JSON job the rref
bridge dumps). Agreement is therefore fp64 round-off of identical numbers, never
contaminated by using R's private in-memory copy of a teaching dataset.

Source of truth is the centralized store (R17): ``sleep.h5`` / ``iris.h5`` /
``ToothGrowth.h5`` / ``PlantGrowth.h5`` / ``InsectSprays.h5`` / ``mtcars.h5``
live in the central store under the ``pystatistics`` namespace, reached via
``DATASETS_ROOT`` (resolution lives in ``drivers/_shared/store_io.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

_SHARED = Path(__file__).resolve().parent.parent / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from store_io import DEFAULT_NAMESPACE, store_root  # noqa: E402


def _store_dir() -> Path:
    """Directory holding this project's curated datasets.

    Store resolution lives in ``drivers/_shared/store_io.py`` — the single place
    that knows where the store is. Do not reintroduce a local search path here.
    """
    return store_root() / DEFAULT_NAMESPACE


def load_frame(stem: str) -> dict[str, NDArray]:
    """Load ``<stem>.h5`` as a dict of column-name -> fp64 array.

    Nominal/factor columns are returned as their integer codes AND, for
    convenience, a companion ``"<name>__labels"`` string array (decoded via the
    stored category_map, preserving R's alphabetical level order).
    """
    path = _store_dir() / f"{stem}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"{stem}.h5 not in store {path.parent}")
    out: dict[str, NDArray] = {}
    with h5py.File(path, "r") as f:
        vals = f["data/values"][:].astype(np.float64)
        names = [x.decode() for x in f["columns/raw_names"][:]]
        dtypes = [x.decode() for x in f["columns/data_types"][:]]
        cmaps = [json.loads(x.decode()) for x in f["columns/category_maps"][:]]
    for j, name in enumerate(names):
        col = vals[:, j]
        out[name] = col
        if dtypes[j] in ("nominal", "ordinal", "binary") and cmaps[j]:
            labels = np.array(
                [cmaps[j].get(str(int(v)), str(int(v))) for v in col], dtype=object
            )
            out[f"{name}__labels"] = labels
    return out


# ---------------------------------------------------------------------------
# Deterministic red-team constructions (NOT stored reference data; documented
# generative recipes so both engines see identical numbers).
# ---------------------------------------------------------------------------


def unbalanced_toothgrowth() -> dict[str, NDArray]:
    """ToothGrowth with a deterministic subset of rows removed so the two-way
    design (supp * dose) is UNBALANCED — the case where SS Type I, II and III
    diverge (the silent-type-substitution trap). Removal rule is fixed: drop the
    first 4 rows of the OJ/dose=2 cell and the first 2 rows of the VC/dose=0.5
    cell (indices computed from the sorted design, no RNG)."""
    tg = load_frame("ToothGrowth")
    supp = tg["supp__labels"]
    dose = tg["dose"]
    keep = np.ones(len(tg["len"]), dtype=bool)
    oj_d2 = np.where((supp == "OJ") & (dose == 2.0))[0]
    vc_d05 = np.where((supp == "VC") & (dose == 0.5))[0]
    keep[oj_d2[:4]] = False
    keep[vc_d05[:2]] = False
    return {"len": tg["len"][keep], "supp": supp[keep], "dose": dose[keep]}


def rm_design(k: int = 4, n: int = 12) -> dict[str, NDArray]:
    """A balanced one-way repeated-measures design: ``n`` subjects each measured
    under ``k`` within-subject conditions. Deterministically generated (fixed
    integer recipe, no RNG): subject baseline b_s = s, condition effect c_j =
    2*j, plus a fixed 'noise' lattice so sphericity is non-trivial. Both engines
    receive the identical wide matrix Y (n x k)."""
    Y = np.empty((n, k), dtype=np.float64)
    for s in range(n):
        for j in range(k):
            # deterministic, reproducible, no RNG; a mild subject*condition
            # interaction breaks compound symmetry so GG/HF < 1.
            Y[s, j] = float(s) + 2.0 * j + ((s * 7 + j * 13) % 5) * 0.5 \
                + (j * j) * (0.3 if s % 2 == 0 else 0.1)
    return {"Y": Y, "n": n, "k": k}


def contingency_tables() -> dict[str, NDArray]:
    """Constructed integer contingency tables for chisq/fisher.

    - ``hair_eye``: the classic 4x4-ish Hair x Eye table (aggregated over sex
      from R's HairEyeColor), a large-count independence case.
    - ``lady_tea``: Fisher's 2x2 tea-tasting table (small counts, exact regime).
    - ``zero_cell``: a 2x2 with a zero cell (fisher OR boundary).
    - ``tiny_expected``: a 2x2 with expected counts < 5 (chisq warning regime).
    - ``gof_dice``: observed counts of a die for a goodness-of-fit test.
    """
    hair_eye = np.array([
        [68, 20, 15, 5],
        [119, 84, 54, 29],
        [26, 17, 14, 14],
        [7, 94, 10, 16],
    ], dtype=np.float64)          # rows Black/Brown/Red/Blond x Brown/Blue/Hazel/Green
    lady_tea = np.array([[3, 1], [1, 3]], dtype=np.float64)
    zero_cell = np.array([[10, 0], [3, 8]], dtype=np.float64)
    tiny_expected = np.array([[2, 8], [7, 3]], dtype=np.float64)
    gof_dice = np.array([22, 24, 38, 30, 46, 44], dtype=np.float64)
    # small r x c (2x3) with modest counts -> exact fisher is tractable in both
    # engines (the hair_eye table is astronomically expensive for exact fisher).
    small_rxc = np.array([[3, 5, 2], [7, 1, 4]], dtype=np.float64)
    return {
        "hair_eye": hair_eye,
        "lady_tea": lady_tea,
        "zero_cell": zero_cell,
        "tiny_expected": tiny_expected,
        "gof_dice": gof_dice,
        "small_rxc": small_rxc,
    }


def tie_vectors() -> dict[str, NDArray]:
    """Small vectors with ties, for wilcox/ks/kendall tie handling + the
    exact->normal-approx switch and the ties warning."""
    return {
        # two-sample with ties -> wilcox forces normal approx + ties warning
        "x_ties": np.array([1.0, 2, 2, 3, 4, 4, 5], dtype=np.float64),
        "y_ties": np.array([2.0, 3, 3, 4, 5, 5, 6], dtype=np.float64),
        # small no-ties -> wilcox exact regime
        "x_small": np.array([1.83, 0.50, 1.62, 2.48, 1.68], dtype=np.float64),
        "y_small": np.array([0.878, 0.647, 0.598, 2.05, 1.06], dtype=np.float64),
        # paired signed-rank with a zero difference (dropped) + ties
        "p1": np.array([125.0, 115, 130, 140, 140, 115, 140, 125, 140, 135]),
        "p2": np.array([110.0, 122, 125, 120, 140, 124, 123, 137, 135, 145]),
    }
