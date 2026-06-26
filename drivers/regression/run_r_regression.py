"""Run the R reference (lm/glm/glm.nb) for one regression model → a record.

One job: cross the language boundary once — write the numeric design Python built
to CSV, call the pinned R worker (``_r/regression_run.R``), and return R's elapsed
time + the full estimate vectors, wrapped as a ``validation-run/v1`` record.

The per-coefficient agreement reduction (max |Δ|, max relΔ) lives in the study
generator, not here — this driver only produces R's numbers.
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

_R_WORKER = Path(__file__).resolve().parent / "_r" / "regression_run.R"


def run_r_regression_record(
    X: NDArray[np.floating], y: NDArray[np.floating], *,
    r_family: str,
    dataset: str,
    model_key: str,
    rscript: str = "Rscript",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the R reference for one model on design ``X`` (intercept = column 0).

    Parameters
    ----------
    X
        Design matrix WITH a leading intercept column (column 0 = ones). The
        intercept column is stripped before handing the design to R, which adds
        its own intercept — so R's coefficient vector is ``[(Intercept), betas]``.
    r_family
        Family as the R worker names it: ``lm`` / ``binomial`` / ``poisson`` /
        ``Gamma`` / ``negbin``.

    Returns ``(record, raw)`` — ``record`` is the canonical timing+estimate record
    (engine ``r:<family>``); ``raw`` is the parsed R JSON (full vectors + model
    summary) for the agreement comparison.

    Raises ``RuntimeError`` if the R worker fails (a missing reference is fatal for
    a correctness head-to-head).
    """
    if not _R_WORKER.is_file():
        raise FileNotFoundError(f"R worker missing: {_R_WORKER}")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    x_noint = X[:, 1:]                                  # drop intercept; R adds it
    names = [f"x{j}" for j in range(x_noint.shape[1])]

    tmp = Path(tempfile.mkdtemp(prefix="rreg_"))
    x_csv, y_csv, out_json = tmp / "x.csv", tmp / "y.csv", tmp / "r.json"
    np.savetxt(x_csv, x_noint, delimiter=",", header=",".join(names),
               comments="", fmt="%.17g")
    np.savetxt(y_csv, y, fmt="%.17g")

    proc = subprocess.run(
        [rscript, str(_R_WORKER), str(x_csv), str(y_csv), r_family, str(out_json)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"R regression worker failed for {model_key} (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}")

    raw = json.loads(out_json.read_text())
    elapsed = float(raw["elapsed_s"])
    times = [float(t) for t in raw.get("elapsed_times_s", [elapsed])]
    engine = f"r:{r_family}"

    summary = {
        "coefficients": [float(c) for c in raw["coefficients"]],
        "standard_errors": [float(s) for s in raw["standard_errors"]],
        "test_statistic": [float(t) for t in raw["test_statistic"]],
        "p_values": [float(pv) for pv in raw["p_values"]],
        "stat_name": raw.get("stat_name"),
        "coef_names": raw.get("coef_names"),
    }
    for k in ("r_squared", "adjusted_r_squared", "residual_std_error",
              "deviance", "null_deviance", "aic", "dispersion", "theta"):
        if k in raw:
            summary[k] = float(raw[k])

    record = make_record(
        engine=engine, dataset=dataset, n=n, p=p, backend_name=f"r_{r_family}",
        precision="fp64",
        wall={"median_s": elapsed, "min_s": min(times), "max_s": max(times),
              "times_s": times},
        summary=summary,
        extra={"model_key": model_key, "r_version": raw.get("r_version")})
    return record, raw
