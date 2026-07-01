"""Run an R rrBLUP::mixed.solve reference GRM fit -> a canonical validation record.

One job: hand R the EXACT numbers grm_lmm fit (y, X, and the GRM K = W Wᵀ / M,
dumped to temp CSVs at full float64 precision), invoke ``_r/grm_run.R``, and parse
its JSON into a ``validation-run/v1`` record comparable field-for-field with the
pystatistics GRM record.

rrBLUP is the canonical GBLUP/GRM reference. NOTE on logLik: rrBLUP's restricted
log-likelihood uses a different additive-constant convention than pystatistics, so
the two logLik VALUES are not directly comparable — agreement is asserted on the
estimable quantities (β, variance components, heritability, genetic-value BLUPs),
all of which are convention-independent.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pystatsval.record import make_record

_HERE = Path(__file__).resolve().parent
_R_DIR = _HERE / "_r"


def run_r_grm_record(ds) -> tuple[dict[str, Any], dict[str, Any]]:
    """rrBLUP::mixed.solve reference for a :class:`GRMDataset`. Returns (record, raw)."""
    n, p = ds.n, ds.p
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        pd.DataFrame({"y": ds.y}).to_csv(tdp / "y.csv", index=False)
        pd.DataFrame(ds.X, columns=[f"c{i}" for i in range(p)]).to_csv(
            tdp / "X.csv", index=False)
        pd.DataFrame(ds.K).to_csv(tdp / "K.csv", index=False)
        out_json = tdp / "out.json"
        proc = subprocess.run(
            ["Rscript", str(_R_DIR / "grm_run.R"),
             str(tdp / "y.csv"), str(tdp / "X.csv"), str(tdp / "K.csv"),
             "1" if ds.reml else "0", str(out_json)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"grm_run.R failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
        raw = json.loads(out_json.read_text())

    def _L(x): return x if isinstance(x, list) else [x]
    summary: dict[str, Any] = {
        "coefficients": [float(v) for v in _L(raw["coefficients"])],
        "standard_errors": [float(v) for v in _L(raw["standard_errors"])],
        "var_genetic": float(raw["var_genetic"]),
        "var_residual": float(raw["var_residual"]),
        "heritability": float(raw["heritability"]),
        "variance_ratio": float(raw["variance_ratio"]),
        "genetic_values": [float(v) for v in _L(raw["genetic_values"])],
        "log_likelihood": float(raw["log_likelihood"]),
    }
    rec = make_record(
        engine="R:rrBLUP", dataset=ds.key, n=n, p=p,
        wall={"median_s": raw["elapsed_s"]},
        backend_name="rrBLUP_mixed.solve", precision="fp64",
        loglik=float(raw["log_likelihood"]), summary=summary,
        extra={"analysis": "grm", "method": raw["method"],
               "r_version": raw.get("r_version"),
               "rrBLUP_version": raw.get("rrBLUP_version")})
    rec["wall_median_s"] = raw["elapsed_s"]
    return rec, raw
