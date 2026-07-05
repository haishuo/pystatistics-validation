"""Bridge to the R reference dispatcher (r_reference.R).

One job: given a function name, the fp64 data array, and params, dump a JSON job,
invoke ``Rscript r_reference.R``, and return the parsed reference result. The data
travels as JSON so R fits the identical numbers pystatistics does (shared-input
discipline). NaN is serialized as JSON null and read back by R as NA.
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


def _to_jsonable(y: NDArray[np.floating]) -> list[Any]:
    """fp64 array -> JSON list with NaN -> None (R reads as NA)."""
    return [None if (v != v) else float(v) for v in np.asarray(y, dtype=float)]


def r_reference(func: str, data: NDArray[np.floating], *,
                period: int = 1, **params: Any) -> dict[str, Any]:
    """Compute the R reference for ``func`` on ``data``; return parsed JSON."""
    job = {"func": func, "data": _to_jsonable(data),
           "period": int(period), "params": params}
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
    proc = subprocess.run(["Rscript", "-e", expr],
                          capture_output=True, text=True)
    out = {}
    for line in proc.stdout.strip().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out
