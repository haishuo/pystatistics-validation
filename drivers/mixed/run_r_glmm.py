"""Run an R lme4::glmer reference GLMM fit -> a canonical validation record.

One job: hand R the EXACT data frame pystatistics analysed (dumped to a temp CSV
at full float64 round-trip precision), invoke ``_r/glmm_run.R`` with the matching
``glmer`` formula + family/link (Laplace, nAGQ=1), and parse its JSON into a
``validation-run/v1`` record comparable field-for-field with the pystatistics
GLMM record.

The temp CSV is ephemeral (R17: the .h5 store is the single source of truth; this
is just the wire format to hand identical bytes to R). glmer with ``nAGQ=1`` is
the Laplace reference matching glmm()'s approximation order.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

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


def _as_list(v: Any) -> list:
    return v if isinstance(v, list) else [v]


def _as_matrix(v: Any) -> list[list[float]]:
    if not isinstance(v, list):
        return [[float(v)]]
    if v and isinstance(v[0], list):
        return v
    return [[float(x)] for x in v]


def _var_components_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Reshape R's as.data.frame(VarCorr(m)) parallel arrays into the same
    {group, name, variance, std_dev, corr} list the pystatistics runner emits.
    GLMM has no Residual row (dispersion fixed at 1); lme4 omits it too."""
    def _L(x):
        return x if isinstance(x, list) else [x]
    grp, v1, v2 = _L(raw["vc_grp"]), _L(raw["vc_var1"]), _L(raw["vc_var2"])
    vcov, sdcor = _L(raw["vc_vcov"]), _L(raw["vc_sdcor"])

    out: list[dict[str, Any]] = []
    corrs: list[tuple[str, str, float]] = []
    for g, a, b, vc, sd in zip(grp, v1, v2, vcov, sdcor):
        if g == "Residual":
            continue  # GLMM: no residual dispersion row
        if b == "":
            out.append({"group": g, "name": a, "variance": float(vc),
                        "std_dev": float(sd), "corr": None})
        else:
            corrs.append((g, b, float(sd)))
    for g, second, c in corrs:
        for e in out:
            if e["group"] == g and e["name"] == second:
                e["corr"] = c
                break
    return out


def run_r_glmm_record(ds, *, reps: int = 7) -> tuple[dict[str, Any], dict[str, Any]]:
    """lme4::glmer (Laplace, nAGQ=1) reference for a :class:`GLMMDataset`.
    Returns (record, raw)."""
    n, p = ds.n, ds.p
    # R must coerce BOTH grouping factors and fixed factors (e.g. cbpp's period)
    # to as.factor(); r_factor_cols lists them (fall back to groups if unset).
    factor_cols = ",".join(ds.r_factor_cols or tuple(ds.groups.keys()))
    with tempfile.TemporaryDirectory() as td:
        data_csv = Path(td) / "frame.csv"
        out_json = Path(td) / "out.json"
        ds.r_frame.to_csv(data_csv, index=False)
        raw = _run_rscript("glmm_run.R", [
            str(data_csv), ds.r_formula, ds.r_family, ds.r_link,
            factor_cols, str(out_json), str(reps)], out_json)

    summary: dict[str, Any] = {
        "coefficients": [float(v) for v in _as_list(raw["coefficients"])],
        "standard_errors": [float(v) for v in _as_list(raw["standard_errors"])],
        "z_values": [float(v) for v in _as_list(raw["z_values"])],
        "p_values": [float(v) for v in _as_list(raw["p_values"])],
        "coef_names": _as_list(raw["coef_names"]),
        "var_components": _var_components_from_raw(raw),
        "blups": {k: [[float(x) for x in row] for row in _as_matrix(v)]
                  for k, v in raw["blups"].items()},
        "log_likelihood": float(raw["log_likelihood"]),
        "deviance": float(raw["deviance"]),
        "aic": float(raw["aic"]),
        "bic": float(raw["bic"]),
    }
    rec = make_record(
        engine="R:glmer", dataset=ds.key, n=n, p=p,
        wall={"median_s": raw["elapsed_s"], "times_s": raw.get("elapsed_times_s")},
        backend_name="glmer", precision="fp64",
        loglik=float(raw["log_likelihood"]),
        summary=summary,
        extra={"analysis": "glmm", "family": raw.get("family"),
               "link": raw.get("link"),
               "is_singular": bool(raw["is_singular"]),
               "r_warnings": _as_list(raw.get("warnings", [])) if raw.get("warnings") else [],
               "r_version": raw.get("r_version"),
               "lme4_version": raw.get("lme4_version")})
    rec["wall_median_s"] = raw["elapsed_s"]
    rec["wall_times_s"] = raw.get("elapsed_times_s")
    return rec, raw
