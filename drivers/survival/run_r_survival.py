"""Run the R survival reference for one procedure → a comparable record.

One job: cross the language boundary once per procedure — call the pinned R
worker (``_r/survival_run.R``) on the committed lung CSV (KM / log-rank / Cox)
or on a Python-reconstructed person-period design (discrete-time), and return
R's elapsed time plus the full estimate vectors wrapped as a
``validation-run/v1`` record.

The per-quantity agreement reductions live in the study generator, not here —
this driver only produces R's numbers.
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

_R_WORKER = Path(__file__).resolve().parent / "_r" / "survival_run.R"


def _run_worker(mode: str, csv_args: list[str], reps: int) -> dict[str, Any]:
    if not _R_WORKER.is_file():
        raise FileNotFoundError(f"R worker missing: {_R_WORKER}")
    out_json = Path(tempfile.mkdtemp(prefix="rsurv_")) / "r.json"
    cmd = ["Rscript", str(_R_WORKER), mode, *csv_args, str(out_json), str(reps)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"R survival worker failed for mode={mode} (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}")
    return json.loads(out_json.read_text())


def _wall(raw: dict[str, Any]) -> dict[str, Any]:
    elapsed = float(raw["elapsed_s"])
    raw_times = raw.get("elapsed_times_s", [elapsed])
    # jsonlite(auto_unbox=TRUE) collapses a length-1 elapsed vector (reps=1) to a
    # scalar; normalize back to a list so single-rep R runs parse like the rest.
    if not isinstance(raw_times, (list, tuple)):
        raw_times = [raw_times]
    times = [float(t) for t in raw_times]
    return {"median_s": elapsed, "min_s": min(times), "max_s": max(times),
            "times_s": times}


def run_r_km(km_csv: Path, *, reps: int = 5) -> tuple[dict[str, Any], dict[str, Any]]:
    """R survfit(Surv(time,event)~1) on lung → (record, raw)."""
    raw = _run_worker("km", [str(km_csv)], reps)
    summary = {k: [float(x) for x in raw[k]]
               for k in ("time", "survival", "n_risk", "n_events",
                         "std_err", "ci_lower", "ci_upper")}
    summary["median_survival"] = float(raw["median_survival"])
    rec = make_record(
        engine="r:survfit", dataset="lung_km", n=int(raw["n"]),
        backend_name="r_km", precision="fp64", wall=_wall(raw),
        summary=summary,
        extra={"procedure": "kaplan_meier",
               "n_events_total": int(raw["n_events_total"]),
               "r_version": raw.get("r_version")})
    return rec, raw


def run_r_logrank(km_csv: Path, *, reps: int = 5) -> tuple[dict[str, Any], dict[str, Any]]:
    """R survdiff(Surv(time,event)~sex) on lung → (record, raw)."""
    raw = _run_worker("logrank", [str(km_csv)], reps)
    summary = {
        "statistic": float(raw["statistic"]),
        "df": int(raw["df"]),
        "p_value": float(raw["p_value"]),
        "observed": [float(x) for x in raw["observed"]],
        "expected": [float(x) for x in raw["expected"]],
    }
    rec = make_record(
        engine="r:survdiff", dataset="lung_km", n=int(raw["n"]),
        backend_name="r_logrank", precision="fp64", wall=_wall(raw),
        summary=summary,
        extra={"procedure": "survdiff",
               "group_labels": raw.get("group_labels"),
               "r_version": raw.get("r_version")})
    return rec, raw


def run_r_coxph(cox_csv: Path, *, reps: int = 5) -> tuple[dict[str, Any], dict[str, Any]]:
    """R coxph(Surv(time,event)~age+sex+ph.ecog, ties='efron') on lung → (record, raw)."""
    raw = _run_worker("coxph", [str(cox_csv)], reps)
    summary = {
        "coefficients": [float(x) for x in raw["coefficients"]],
        "hazard_ratios": [float(x) for x in raw["hazard_ratios"]],
        "standard_errors": [float(x) for x in raw["standard_errors"]],
        "z_values": [float(x) for x in raw["z_values"]],
        "p_values": [float(x) for x in raw["p_values"]],
        "concordance": float(raw["concordance"]),
        "loglik_null": float(raw["loglik_null"]),
        "loglik_model": float(raw["loglik_model"]),
    }
    rec = make_record(
        engine="r:coxph", dataset="lung_coxph", n=int(raw["n"]),
        p=len(raw["coefficients"]), backend_name="r_cox", precision="fp64",
        wall=_wall(raw),
        summary=summary,
        extra={"procedure": "coxph", "n_events": int(raw["n_events"]),
               "coef_names": raw.get("coef_names"), "ties": raw.get("ties"),
               "r_version": raw.get("r_version")})
    return rec, raw


def run_r_discrete(X_pp: NDArray, y_pp: NDArray, *, n_intervals: int,
                   covariate_names: list[str], reps: int = 5,
                   ) -> tuple[dict[str, Any], dict[str, Any]]:
    """R glm(binomial) on the person-period design → (record, raw).

    ``X_pp`` is ``[interval dummies | covariates]`` (no intercept). The returned
    raw vectors cover ALL columns; the covariate quantities are the last
    ``len(covariate_names)`` entries (the generator slices them).
    """
    X_pp = np.asarray(X_pp, dtype=np.float64)
    y_pp = np.asarray(y_pp, dtype=np.float64).ravel()
    p_total = X_pp.shape[1]
    col_names = [f"I{j}" for j in range(n_intervals)] + list(covariate_names)
    if len(col_names) != p_total:
        raise ValueError(f"col-name count {len(col_names)} != design cols {p_total}")

    tmp = Path(tempfile.mkdtemp(prefix="rsurv_glm_"))
    x_csv, y_csv = tmp / "x.csv", tmp / "y.csv"
    np.savetxt(x_csv, X_pp, delimiter=",", header=",".join(col_names),
               comments="", fmt="%.17g")
    np.savetxt(y_csv, y_pp, fmt="%.17g")

    raw = _run_worker("glmfit", [str(x_csv), str(y_csv)], reps)
    n_cov = len(covariate_names)
    summary = {
        "coefficients": [float(x) for x in raw["coefficients"][-n_cov:]],
        "standard_errors": [float(x) for x in raw["standard_errors"][-n_cov:]],
        "z_values": [float(x) for x in raw["z_values"][-n_cov:]],
        "p_values": [float(x) for x in raw["p_values"][-n_cov:]],
        "deviance": float(raw["deviance"]),
        "aic": float(raw["aic"]),
    }
    rec = make_record(
        engine="r:glm_discrete", dataset="lung_discrete",
        n=int(raw["n_rows"]), p=n_cov, backend_name="r_discrete",
        precision="fp64", wall=_wall(raw),
        summary=summary,
        extra={"procedure": "discrete_time", "n_intervals": n_intervals,
               "covariate_names": covariate_names,
               "r_version": raw.get("r_version")})
    return rec, raw


def run_r_glmdiag(X_pp: NDArray, y_pp: NDArray, *, n_intervals: int,
                  covariate_names: list[str]) -> dict[str, Any]:
    """R glm(binomial) BEHAVIOUR diagnostics on a person-period design (R15).

    Returns R's own convergence flag, iteration count, glm warnings, max
    |coefficient|, deviance/AIC, and the covariate-tail coef/SE/z/p — the
    reference BEHAVIOUR the default ``discrete_time(intervals=None)`` must match.
    Untimed (a single fit); the standard ``run_r_discrete`` carries timing.
    """
    X_pp = np.asarray(X_pp, dtype=np.float64)
    y_pp = np.asarray(y_pp, dtype=np.float64).ravel()
    p_total = X_pp.shape[1]
    col_names = [f"I{j}" for j in range(n_intervals)] + list(covariate_names)
    if len(col_names) != p_total:
        raise ValueError(f"col-name count {len(col_names)} != design cols {p_total}")

    tmp = Path(tempfile.mkdtemp(prefix="rsurv_glmdiag_"))
    x_csv, y_csv = tmp / "x.csv", tmp / "y.csv"
    np.savetxt(x_csv, X_pp, delimiter=",", header=",".join(col_names),
               comments="", fmt="%.17g")
    np.savetxt(y_csv, y_pp, fmt="%.17g")

    raw = _run_worker("glmdiag", [str(x_csv), str(y_csv)], reps=1)
    n_cov = len(covariate_names)
    warns = raw.get("warnings") or []
    if not isinstance(warns, (list, tuple)):
        warns = [warns]
    return {
        "coefficients": [float(x) for x in raw["coefficients"][-n_cov:]],
        "standard_errors": [float(x) for x in raw["standard_errors"][-n_cov:]],
        "z_values": [float(x) for x in raw["z_values"][-n_cov:]],
        "p_values": [float(x) for x in raw["p_values"][-n_cov:]],
        "deviance": float(raw["deviance"]),
        "aic": float(raw["aic"]),
        "converged": bool(raw["converged"]),
        "n_iter": int(raw["n_iter"]),
        "n_warnings": int(raw["n_warnings"]),
        "warnings": [str(w) for w in warns],
        "max_abs_coef": float(raw["max_abs_coef"]),
        "n_rows": int(raw["n_rows"]),
        "r_version": raw.get("r_version"),
    }


def _run_worker_feat(mode: str, csv_path: str, extra: list[str],
                     reps: int) -> dict[str, Any]:
    """Worker call for the feature modes (coxfeat/kmfeat), whose extra args
    (ties / robust / zph transform / conf.type) follow ``reps``."""
    if not _R_WORKER.is_file():
        raise FileNotFoundError(f"R worker missing: {_R_WORKER}")
    out_json = Path(tempfile.mkdtemp(prefix="rsurv_")) / "r.json"
    cmd = ["Rscript", str(_R_WORKER), mode, csv_path, str(out_json),
           str(reps), *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"R survival worker failed for mode={mode} (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}")
    return json.loads(out_json.read_text())


def run_r_coxfeat(csv_path: str, *, ties: str = "efron", robust: bool = False,
                  zph_transform: str = "", reps: int = 5,
                  dataset: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """R reference for the A1+VA-8 Cox feature cluster.

    The CSV carries ``time``, ``event``, covariates, and any of the reserved
    columns ``.start`` (counting-process), ``.strata``, ``.cluster``. Returns
    ``(record, raw)`` — the ``validation-run/v1`` record plus R's raw payload
    (which includes the zph table when ``zph_transform`` is set and R's own
    warnings, for behaviour matching).
    """
    raw = _run_worker_feat(
        "coxfeat", csv_path,
        [ties, "1" if robust else "0", zph_transform], reps)
    summary = {
        "coefficients": [float(x) for x in np.atleast_1d(raw["coefficients"])],
        "standard_errors": [float(x)
                            for x in np.atleast_1d(raw["standard_errors"])],
        "loglik_model": float(raw["loglik_model"]),
        "concordance": float(raw["concordance"]),
    }
    def _vec(key):
        # jsonlite renders an R NULL as {} — treat dict/None/absent as missing.
        v = raw.get(key)
        if v is None or isinstance(v, dict):
            return None
        return [float(x) for x in np.atleast_1d(v)]

    if _vec("naive_se") is not None:
        summary["naive_se"] = _vec("naive_se")
    if _vec("zph_chisq") is not None:
        summary["zph_chisq"] = _vec("zph_chisq")
        summary["zph_p"] = _vec("zph_p")
    record = make_record(
        engine="r_survival",
        dataset=dataset or Path(csv_path).stem,
        n=int(raw["n"]), p=len(summary["coefficients"]),
        loglik=float(raw["loglik_model"]),
        n_iter=int(raw["n_iter"]), converged=True,
        wall=_wall(raw),
        backend_name="r_coxph",
        precision="fp64",
        parameterization=f"ties={ties};robust={raw.get('robust')}",
        summary=summary,
        extra={"procedure": "coxfeat", "r_version": raw.get("r_version"),
               "warnings": raw.get("warnings") or []},
    )
    return record, raw


def run_r_kmfeat(csv_path: str, *, conf_type: str = "log", reps: int = 5,
                 dataset: str = "") -> tuple[dict[str, Any], dict[str, Any]]:
    """R reference KM for left truncation / strata (see ``run_r_coxfeat``)."""
    raw = _run_worker_feat("kmfeat", csv_path, [conf_type], reps)
    summary = {
        "time": [float(x) for x in np.atleast_1d(raw["time"])],
        "survival": [float(x) for x in np.atleast_1d(raw["survival"])],
        "n_risk": [float(x) for x in np.atleast_1d(raw["n_risk"])],
        "se": [float(x) for x in np.atleast_1d(raw["se"])],
        "strata": [str(x) for x in np.atleast_1d(raw["strata"])],
    }
    for key in ("ci_lower", "ci_upper"):
        if raw.get(key) is not None:
            summary[key] = [float(x) for x in np.atleast_1d(raw[key])]
    record = make_record(
        engine="r_survival",
        dataset=dataset or Path(csv_path).stem,
        n=int(raw["n"]), p=0,
        loglik=None, n_iter=None, converged=True,
        wall=_wall(raw),
        backend_name="r_survfit",
        precision="fp64",
        parameterization=f"conf_type={conf_type}",
        summary=summary,
        extra={"procedure": "kmfeat", "r_version": raw.get("r_version")},
    )
    return record, raw
