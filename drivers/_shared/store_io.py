"""Read curated reference tables from the central HDF5 store (schema 1.0.0).

One job: locate ``<name>.h5`` in the central store (via ``MVNMLE_DATA_DIR``, with
the Mac/Forge mirrors as fallbacks) and return a requested set of columns as a
float64 matrix, in the requested order. This is the R17 load path — validation
drivers read reference designs from the one central store, never from
driver-local CSVs.

The store keeps ``/data/values`` as float32 (survey/GPU datasets) or float64
(CPU R-parity tables whose values are not fp32-exact; see the datasets store's
``SCHEMA.md``). We read the dataset's native dtype and upcast to float64: for an
fp32-exact table (all-integer / 0-1 designs) the upcast is lossless; for an
fp64-stored table the bytes are preserved exactly. Either way the driver hands
pystatistics and R the same values.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# Mirrors of the central store, tried after ``MVNMLE_DATA_DIR`` (Mac, then Forge).
_STORE_FALLBACKS = (
    Path("/Volumes/Archive/Documents/Dropbox/Dev/datasets"),
    Path("/mnt/data/pystatistics-datasets"),
)


def store_h5_path(name: str) -> Path:
    """Locate ``<name>.h5`` via ``MVNMLE_DATA_DIR`` then the mirrors, else fail loud."""
    candidates: list[Path] = []
    root = os.environ.get("MVNMLE_DATA_DIR")
    if root:
        candidates.append(Path(root) / f"{name}.h5")
    candidates += [base / f"{name}.h5" for base in _STORE_FALLBACKS]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"{name}.h5 not found in the central dataset store. Set MVNMLE_DATA_DIR "
        f"to the store dir (tried: {[str(c) for c in candidates]}).")


def store_column_names(name: str) -> list[str]:
    """Return the ``raw_names`` of ``<name>.h5`` in on-disk column order."""
    import h5py

    path = store_h5_path(name)
    with h5py.File(path, "r") as f:
        return [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["/columns/raw_names"][:]]


def read_store_matrix(name: str, columns: list[str],
                      ) -> NDArray[np.float64]:
    """Return the named ``columns`` of ``<name>.h5`` as an ``(n, len(columns))``
    float64 matrix, in the given order. Fails loud if a column is absent."""
    import h5py

    path = store_h5_path(name)
    with h5py.File(path, "r") as f:
        values = np.asarray(f["/data/values"][:], dtype=np.float64)
        raw = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f["/columns/raw_names"][:]]
    idx = {nm: i for i, nm in enumerate(raw)}
    missing = [c for c in columns if c not in idx]
    if missing:
        raise ValueError(
            f"{name}.h5 is missing columns {missing} (have {raw})")
    return np.column_stack([values[:, idx[c]] for c in columns])
