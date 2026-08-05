"""Locate and read curated reference tables from the central HDF5 store (schema 1.0.0).

One job: resolve the central dataset store, locate ``<namespace>/<name>.h5``
inside it, and return requested columns as a float64 matrix in the requested
order. This is the R17 load path — validation drivers read reference designs
from the one central store, never from driver-local CSVs.

**This module is the single place that knows where the store lives.** Every
driver-local dataset module resolves through :func:`store_root` or
:func:`store_h5_path`; none of them may carry its own copy of the search path.
Ten copies of that path previously existed, all of them stale, and the drivers
only kept working because the environment variable was set by hand.

Layout
------
The store is namespaced by consuming project::

    <store root>/<namespace>/<name>.h5

The store root is resolved from ``DATASETS_ROOT``, falling back to the known
machine mirrors. Resolution fails loudly with every path it tried — it never
substitutes a default, a sample, or an empty frame.

``MVNMLE_DATA_DIR`` is accepted as a deprecated alias for ``DATASETS_ROOT``: the
store is shared across projects and should not be named after one subsystem of
one package. Setting both to different values is ambiguous and is an error.

Dtypes
------
The store keeps ``/data/values`` as float32 (survey/GPU datasets) or float64
(CPU R-parity tables whose values are not fp32-exact; see the datasets store's
``SCHEMA.md``). We read the dataset's native dtype and upcast to float64: for an
fp32-exact table (all-integer / 0-1 designs) the upcast is lossless; for an
fp64-stored table the bytes are preserved exactly. Either way the driver hands
pystatistics and R the same values.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

#: Environment variable naming the store root.
STORE_ROOT_ENV = "DATASETS_ROOT"

#: Deprecated alias for :data:`STORE_ROOT_ENV`, honored during migration.
STORE_ROOT_ENV_LEGACY = "MVNMLE_DATA_DIR"

#: Namespace holding this project's curated validation fixtures — the small
#: derived datasets built by the store's own generators. See the store's
#: SCHEMA.md.
DEFAULT_NAMESPACE = "pystatistics"

#: Namespace holding third-party survey microdata (GSS, WVS, CSES,
#: Afrobarometer, …). Kept separate from the fixtures because it is a different
#: kind of thing: not reproducible from any generator, and licence-restricted —
#: several of these programmes prohibit redistribution. One namespace boundary,
#: three meanings, so "may this leave the machine?" has a directory-level answer.
SURVEY_NAMESPACE = "surveys"

# Known mirrors of the central store, tried in order after the environment
# variable. Both carry the namespaced layout and were verified identical to the
# manifest on 2026-08-05 (48/48 curated files). A machine that has neither still
# fails loudly rather than guessing.
_STORE_MIRRORS = (
    Path("/Volumes/Archive/Documents/Dev/datasets"),   # Mac
    Path("/mnt/data/pystatistics-datasets"),           # Forge
)


def _root_from_env() -> Path | None:
    """Return the store root named by the environment, or ``None`` if unset.

    Fails loud if both the current and deprecated variables are set to different
    directories — that state is ambiguous and silently picking one would be
    exactly the kind of invisible choice this module exists to prevent.
    """
    current = os.environ.get(STORE_ROOT_ENV)
    legacy = os.environ.get(STORE_ROOT_ENV_LEGACY)

    if current and legacy and Path(current) != Path(legacy):
        raise RuntimeError(
            f"{STORE_ROOT_ENV}={current!r} and {STORE_ROOT_ENV_LEGACY}={legacy!r} "
            f"disagree. {STORE_ROOT_ENV_LEGACY} is a deprecated alias; unset it "
            f"or set both to the same directory.")

    if current:
        return Path(current)
    if legacy:
        warnings.warn(
            f"{STORE_ROOT_ENV_LEGACY} is deprecated; use {STORE_ROOT_ENV}. The "
            f"dataset store is shared across projects and is no longer named "
            f"after the mvnmle subsystem.",
            DeprecationWarning, stacklevel=3)
        return Path(legacy)
    return None


def store_root() -> Path:
    """Return the central store root, else fail loud listing every path tried."""
    candidates: list[Path] = []
    from_env = _root_from_env()
    if from_env is not None:
        candidates.append(from_env)
    candidates.extend(_STORE_MIRRORS)

    for cand in candidates:
        if cand.is_dir():
            return cand

    raise FileNotFoundError(
        f"Central dataset store not found. Set {STORE_ROOT_ENV} to the store "
        f"root (tried: {[str(c) for c in candidates]}).")


def store_h5_path(name: str, namespace: str = DEFAULT_NAMESPACE) -> Path:
    """Return the path to ``<store root>/<namespace>/<name>.h5``, else fail loud."""
    path = store_root() / namespace / f"{name}.h5"
    if not path.is_file():
        raise FileNotFoundError(
            f"{namespace}/{name}.h5 not found in the central dataset store "
            f"(looked in {store_root()}). Set {STORE_ROOT_ENV} to the store root.")
    return path


def store_column_names(name: str, namespace: str = DEFAULT_NAMESPACE) -> list[str]:
    """Return the ``raw_names`` of ``<name>.h5`` in on-disk column order."""
    import h5py

    path = store_h5_path(name, namespace)
    with h5py.File(path, "r") as f:
        return [x.decode() if isinstance(x, bytes) else str(x)
                for x in f["/columns/raw_names"][:]]


def read_store_matrix(name: str, columns: list[str],
                      namespace: str = DEFAULT_NAMESPACE,
                      ) -> NDArray[np.float64]:
    """Return the named ``columns`` of ``<name>.h5`` as an ``(n, len(columns))``
    float64 matrix, in the given order. Fails loud if a column is absent."""
    import h5py

    path = store_h5_path(name, namespace)
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
