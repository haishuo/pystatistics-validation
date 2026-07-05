"""Bridge to the montecarlo R reference dispatcher (r_reference.R).

One job: dump a JSON job {func, data (n x p), params}, invoke
``Rscript r_reference.R``, and return the parsed reference. The data travels as
JSON so R runs boot / permutation on the identical fp64 numbers pystatistics
does (R17 shared-input discipline). For the tight tier, the R side returns the
resample index matrix so pystatistics can reproduce R's replicates exactly.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_HERE = Path(__file__).resolve().parent
_R_SCRIPT = _HERE / "r_reference.R"


def _to_jsonable(data: NDArray[np.floating]) -> list[Any]:
    """(n,) or (n,p) fp64 array -> nested JSON list, NaN -> None (R reads NA)."""
    a = np.asarray(data, dtype=float)
    if a.ndim == 1:
        return [None if (v != v) else float(v) for v in a]
    return [[None if (v != v) else float(v) for v in row] for row in a]


def r_reference(func: str, data: NDArray[np.floating], **params: Any) -> dict[str, Any]:
    """Compute the montecarlo R reference for ``func`` on ``data``; parsed JSON."""
    job = {"func": func, "data": _to_jsonable(data), "params": params}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as jf:
        json.dump(job, jf)
        job_path = jf.name
    try:
        proc = subprocess.run(
            ["Rscript", str(_R_SCRIPT), job_path],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"R reference '{func}' failed (rc={proc.returncode}):\n"
                f"{proc.stderr.strip()}")
        return json.loads(proc.stdout)
    finally:
        Path(job_path).unlink(missing_ok=True)


def r_package_versions(pkgs: tuple[str, ...]) -> dict[str, str]:
    """Return installed R package versions (for the env manifest)."""
    expr = ";".join(
        f'cat("{p}", as.character(packageVersion("{p}")), "\\n")' for p in pkgs)
    proc = subprocess.run(["Rscript", "-e", expr], capture_output=True, text=True)
    out = {}
    for line in proc.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out
