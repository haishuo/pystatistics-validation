"""Run an R lme4/lmerTest reference LMM fit -> a canonical validation record.

One job: hand R the EXACT data frame pystatistics analysed (dumped to a temp CSV
at full float64 round-trip precision, so neither engine sees a different number),
invoke ``_r/lmm_run.R`` with the matching ``lmer`` formula, and parse its JSON into
a ``validation-run/v1`` record comparable field-for-field with the pystatistics
record.

The temp CSV is ephemeral (R17: no committed CSV -- the .h5 store is the single
source of truth; this is just the wire format to hand identical bytes to R).
lmerTest is the reference for the Satterthwaite df + p-values (lme4 deliberately
omits them); the point estimates / SEs / variance components / logLik are lme4's.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from pystatsval.record import make_record

_HERE = Path(__file__).resolve().parent
_R_DIR = _HERE / "_r"


def _run_rscript(script: str, args: list[str], out_json: Path) -> dict[str, Any]:
    proc = subprocess.run(
        ["Rscript", str(_R_DIR / script), *args],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} failed (exit {proc.returncode}):\n{proc.stderr[-2000:]}")
    return json.loads(out_json.read_text())


def _var_components_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Reshape R's as.data.frame(VarCorr(m)) parallel arrays into the same
    {group, name, variance, std_dev, corr} list the pystatistics runner emits.

    R rows: var2 == "" -> a variance (sdcor = std dev); var2 != "" -> a
    correlation between var1 and var2 (sdcor = correlation, vcov = covariance).
    The Residual row has grp == 'Residual'. We attach each correlation onto the
    matching variance entry's ``corr`` so the shapes line up with pystatistics
    (which carries corr on the second term of a correlated block).
    """
    grp = raw["vc_grp"]; v1 = raw["vc_var1"]; v2 = raw["vc_var2"]
    vcov = raw["vc_vcov"]; sdcor = raw["vc_sdcor"]
    # normalize to lists (jsonlite auto-unboxes length-1 arrays to scalars)
    def _L(x):
        return x if isinstance(x, list) else [x]
    grp, v1, v2 = _L(grp), _L(v1), _L(v2)
    vcov, sdcor = _L(vcov), _L(sdcor)

    out: list[dict[str, Any]] = []
    corrs: list[tuple[str, str, float]] = []  # (group, second-term, correlation)
    for g, a, b, vc, sd in zip(grp, v1, v2, vcov, sdcor):
        if b == "":  # a variance row
            name = "" if g == "Residual" else a
            out.append({"group": g, "name": name,
                        "variance": float(vc), "std_dev": float(sd),
                        "corr": None})
        else:        # a correlation row between terms a and b
            corrs.append((g, b, float(sd)))
    # attach correlations to the matching (group, second-term) variance entry
    for g, second, c in corrs:
        for e in out:
            if e["group"] == g and e["name"] == second:
                e["corr"] = c
                break
    return out


def run_r_lmm_record(ds, *, reps: int = 7) -> tuple[dict[str, Any], dict[str, Any]]:
    """lme4/lmerTest reference for a :class:`MixedDataset`. Returns (record, raw)."""
    n, p = ds.n, ds.p
    factor_cols = ",".join(ds.groups.keys())
    with tempfile.TemporaryDirectory() as td:
        data_csv = Path(td) / "frame.csv"
        out_json = Path(td) / "out.json"
        ds.r_frame.to_csv(data_csv, index=False)
        raw = _run_rscript("lmm_run.R", [
            str(data_csv), ds.r_formula, "1" if ds.reml else "0",
            factor_cols, str(out_json), str(reps)], out_json)

    summary: dict[str, Any] = {
        "coefficients": [float(v) for v in _as_list(raw["coefficients"])],
        "standard_errors": [float(v) for v in _as_list(raw["standard_errors"])],
        "t_values": [float(v) for v in _as_list(raw["t_values"])],
        "df_satterthwaite": [float(v) for v in _as_list(raw["df_satterthwaite"])],
        "p_values": [float(v) for v in _as_list(raw["p_values"])],
        "coef_names": _as_list(raw["coef_names"]),
        "var_components": _var_components_from_raw(raw),
        "blups": {k: [[float(x) for x in row] for row in _as_matrix(v)]
                  for k, v in raw["blups"].items()},
        "log_likelihood": float(raw["log_likelihood"]),
        "reml_criterion": float(raw["reml_criterion"]),
        "aic": float(raw["aic"]),
        "bic": float(raw["bic"]),
    }
    rec = make_record(
        engine="R:lmer", dataset=ds.key, n=n, p=p,
        wall={"median_s": raw["elapsed_s"], "times_s": raw.get("elapsed_times_s")},
        backend_name="lmer", precision="fp64",
        loglik=float(raw["log_likelihood"]),
        summary=summary,
        extra={"analysis": "lmm", "reml": bool(raw["reml"]),
               "is_singular": bool(raw["is_singular"]),
               # jsonlite auto-unboxes a length-1 array to a scalar; normalize so
               # a single diagnostic stays one list element (not char-iterated).
               "r_warnings": _as_list(raw.get("warnings", [])) if raw.get("warnings") else [],
               "r_version": raw.get("r_version"),
               "lme4_version": raw.get("lme4_version"),
               "lmerTest_version": raw.get("lmerTest_version")})
    rec["wall_median_s"] = raw["elapsed_s"]
    rec["wall_times_s"] = raw.get("elapsed_times_s")
    return rec, raw


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def _as_matrix(v: Any) -> list[list[float]]:
    """R's blups[[grp]] is a matrix; jsonlite may emit row-lists or, for a single
    column, a flat vector. Normalize to a list of rows."""
    if not isinstance(v, list):
        return [[float(v)]]
    if v and isinstance(v[0], list):
        return v
    return [[float(x)] for x in v]  # single-column -> column vector of rows
