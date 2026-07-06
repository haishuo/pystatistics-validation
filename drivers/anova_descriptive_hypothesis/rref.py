"""Bridge to the R reference dispatcher (``r_reference.R``).

One job: given a function name and its arguments, dump a JSON job, invoke
``Rscript r_reference.R``, and return the parsed reference result. Data travels
as JSON so R analyses the identical numbers pystatistics does (shared-input
discipline). NaN serializes to JSON null (R reads NA).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_R_SCRIPT = _HERE / "r_reference.R"


def _jsonable(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, float) and (v != v):
        return None
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if f != f else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


_INF_TAGS = {"__Inf__": float("inf"), "__-Inf__": float("-inf"),
             "__NaN__": float("nan")}


def _restore_inf(v: Any) -> Any:
    """Undo r_reference.R's Inf/-Inf/NaN string tagging back to floats."""
    if isinstance(v, str) and v in _INF_TAGS:
        return _INF_TAGS[v]
    if isinstance(v, list):
        return [_restore_inf(x) for x in v]
    if isinstance(v, dict):
        return {k: _restore_inf(x) for k, x in v.items()}
    return v


def r_ref(func: str, **kwargs: Any) -> dict[str, Any]:
    """Compute the R reference for ``func``; return parsed JSON dict."""
    job = {"func": func}
    job.update({k: _jsonable(v) for k, v in kwargs.items()})
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
        return _restore_inf(json.loads(proc.stdout))
    finally:
        Path(job_path).unlink(missing_ok=True)


def r_versions() -> dict[str, str]:
    """R + package versions, for the artifact provenance block."""
    script = (
        'cat(jsonlite::toJSON(list('
        'R=R.version.string,'
        'car=as.character(packageVersion("car")),'
        'afex=as.character(packageVersion("afex"))), auto_unbox=TRUE))'
    )
    proc = subprocess.run(["Rscript", "-e", script],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"R version probe failed:\n{proc.stderr}")
    return json.loads(proc.stdout)
