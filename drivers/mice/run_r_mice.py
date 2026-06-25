"""Run and time the R ``mice`` reference → a canonical validation record.

One job: cross the language boundary once — write an incomplete matrix to CSV, call
the pinned R worker (``_r/mice_run.R``), and return R's elapsed time + per-column
imputed values, wrapped as a ``validation-run/v1`` record (timing) plus the raw
imputed values (for the fidelity comparison, which lives in the study generator).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.record import make_record

_R_WORKER = Path(__file__).resolve().parent / "_r" / "mice_run.R"


def _column_names(p: int) -> list[str]:
    return [f"x{j}" for j in range(p)]


def _kind_args(column_kinds, names: list[str]) -> tuple[str, str]:
    if column_kinds is None:
        return "none", "none"
    ordered = [names[j] for j, k in enumerate(column_kinds) if k == "ordered"]
    factor = [names[j] for j, k in enumerate(column_kinds)
              if k in ("binary", "categorical")]
    return (",".join(ordered) or "none", ",".join(factor) or "none")


def run_r_mice_record(incomplete: NDArray[np.floating], *,
                      dataset: str,
                      m: int = 5,
                      maxit: int = 5,
                      method: str = "pmm",
                      seed: int,
                      column_kinds: tuple[str, ...] | None = None,
                      rscript: str = "Rscript") -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run R mice on ``incomplete`` (NaN = missing).

    Returns ``(record, per_col)`` — ``record`` is the canonical timing record
    (engine ``r:mice``); ``per_col`` maps each incomplete column name to R's
    flattened imputed values (for distributional fidelity vs pystatistics).

    Raises ``RuntimeError`` if the R worker fails (a missing reference is fatal for
    the head-to-head, unlike a recorded skip).
    """
    n, p = incomplete.shape
    names = _column_names(p)
    tmp = Path(tempfile.mkdtemp(prefix="rmice_"))
    csv_in, json_out = tmp / "incomplete.csv", tmp / "r_result.json"
    np.savetxt(csv_in, incomplete, delimiter=",", header=",".join(names),
               comments="", fmt="%.12g")
    ordered_arg, factor_arg = _kind_args(column_kinds, names)

    proc = subprocess.run(
        [rscript, str(_R_WORKER), str(csv_in), str(json_out),
         str(m), str(maxit), method, str(seed), ordered_arg, factor_arg],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"R mice worker failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")

    raw = json.loads(json_out.read_text())
    elapsed = float(raw["elapsed_s"])
    per_col = {k: np.asarray(v, dtype=float).ravel() for k, v in raw["per_col"].items()}
    record = make_record(
        engine="r:mice", dataset=dataset, n=n, p=p, backend_name="r_mice",
        wall={"median_s": elapsed, "min_s": elapsed, "max_s": elapsed,
              "times_s": [elapsed]},
        extra={"m_imputations": int(m), "maxit": int(maxit), "method": method,
               "r_version": raw.get("mice_version")})
    return record, per_col
