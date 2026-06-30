"""Run an R prcomp / factanal reference fit -> a canonical validation record.

One job: hand R the EXACT float64 matrix pystatistics analysed (dumped to a temp
CSV at full round-trip precision, so neither engine sees a different number),
invoke the matching ``_r/*.R`` script, and parse its JSON into a
``validation-run/v1`` record comparable field-for-field with the pystatistics
record.

The temp CSV is ephemeral (R17: no committed CSV — the .h5 store is the single
source of truth; this is just the wire format to hand identical bytes to R).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pystatsval.record import make_record

_HERE = Path(__file__).resolve().parent
_R_DIR = _HERE / "_r"


def _dump_design(X: NDArray[np.floating], names: list[str], path: Path) -> None:
    """Write X to CSV at full float64 round-trip precision (clean header, no index)."""
    df = pd.DataFrame(np.asarray(X, dtype=np.float64), columns=list(names))
    # pandas writes float64 with round-trip-faithful repr by default, so R reads
    # back the identical double values pystatistics fit.
    df.to_csv(path, index=False)


def _run_rscript(script: str, args: list[str]) -> dict[str, Any]:
    out_json = args[-2]  # by convention the out_json path is the 2nd-last arg
    proc = subprocess.run(
        ["Rscript", str(_R_DIR / script), *args],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    return json.loads(Path(out_json).read_text())


def run_r_pca_record(
    X: NDArray[np.floating], names: list[str], *,
    dataset: str, center: bool, scale: bool, reps: int = 7,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """prcomp reference for matrix ``X``. Returns (record, raw_json)."""
    n, p = X.shape
    with tempfile.TemporaryDirectory() as td:
        x_csv = Path(td) / "X.csv"
        out_json = Path(td) / "out.json"
        _dump_design(X, names, x_csv)
        raw = _run_rscript("pca_run.R", [
            str(x_csv), "1" if center else "0", "1" if scale else "0",
            str(out_json), str(reps)])
    wall = {"wall_median_s": raw["elapsed_s"],
            "wall_times_s": raw.get("elapsed_times_s")}
    summary = {
        "sdev": [float(v) for v in raw["sdev"]],
        "explained_variance": [float(v) for v in raw["explained_variance"]],
        "explained_variance_ratio": [float(v) for v in raw["explained_variance_ratio"]],
        "rotation": [[float(v) for v in row] for row in raw["rotation"]],
        "scores": [[float(v) for v in row] for row in raw["scores"]],
        "center": [float(v) for v in raw["center"]],
        "scale": [float(v) for v in raw["scale"]],
        "n_components": len(raw["sdev"]),
    }
    rec = make_record(
        engine="R:prcomp", dataset=dataset, n=n, p=p,
        wall={"median_s": raw["elapsed_s"], "times_s": raw.get("elapsed_times_s")},
        backend_name="prcomp", precision="fp64",
        summary=summary,
        extra={"analysis": "pca", "center": center, "scale": scale,
               "r_version": raw.get("r_version")})
    # make_record stores wall under standardized keys; expose median for speedups.
    rec["wall_median_s"] = raw["elapsed_s"]
    rec["wall_times_s"] = raw.get("elapsed_times_s")
    return rec, raw


def run_r_fa_record(
    X: NDArray[np.floating], names: list[str], *,
    dataset: str, n_factors: int, rotation: str = "varimax", reps: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """factanal reference for matrix ``X``. Returns (record, raw_json)."""
    n, p = X.shape
    with tempfile.TemporaryDirectory() as td:
        x_csv = Path(td) / "X.csv"
        out_json = Path(td) / "out.json"
        _dump_design(X, names, x_csv)
        raw = _run_rscript("factanal_run.R", [
            str(x_csv), str(n_factors), rotation, str(out_json), str(reps)])
    summary = {
        "loadings": [[float(v) for v in row] for row in raw["loadings"]],
        "uniquenesses": [float(v) for v in raw["uniquenesses"]],
        "communalities": [float(v) for v in raw["communalities"]],
        "objective": float(raw["objective"]),
        "chi_sq": (None if raw["chi_sq"] is None else float(raw["chi_sq"])),
        "p_value": (None if raw["p_value"] is None else float(raw["p_value"])),
        "dof": int(raw["dof"]),
    }
    rec = make_record(
        engine="R:factanal", dataset=dataset, n=n, p=p,
        wall={"median_s": raw["elapsed_s"], "times_s": raw.get("elapsed_times_s")},
        backend_name="factanal", precision="fp64",
        converged=bool(raw.get("converged", 1)),
        summary=summary,
        extra={"analysis": "factor_analysis", "n_factors": n_factors,
               "rotation_method": rotation, "r_warning": raw.get("warning", ""),
               "r_version": raw.get("r_version")})
    rec["wall_median_s"] = raw["elapsed_s"]
    rec["wall_times_s"] = raw.get("elapsed_times_s")
    return rec, raw
