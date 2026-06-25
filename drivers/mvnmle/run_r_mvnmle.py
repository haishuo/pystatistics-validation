"""Run and time R's ``mvnmle::mlest`` on a curated problem.

One job: hand a matrix to the R reference implementation (the incumbent), time
it under the same warmup+reps protocol (executed inside R so R's startup cost is
excluded), and return a uniform benchmark record.

R's ``mvnmle`` has a hard 50-variable limit, so problems with ``p > 50`` are not
run — the record notes the limit explicitly (it is itself a headline result:
the incumbent cannot enter that regime).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pystatsval.estimates import summarize_covariance
from pystatsval.record import make_record

R_VAR_LIMIT = 50
_R_SCRIPT = Path(__file__).resolve().parent / "r_mvnmle_timing.R"


def run_r_mvnmle(X: NDArray[np.floating],
                 *,
                 survey: str,
                 n_patterns: int,
                 missing_frac: float,
                 reps: int = 5,
                 warmup: int = 1,
                 timeout_s: float = 1800.0) -> dict[str, Any]:
    """Fit + time R ``mvnmle::mlest`` on ``X``. Returns a uniform record.

    Records (rather than raises) for the expected non-fatal cases: ``p`` over the
    R 50-variable limit, missing ``Rscript``, R error, or timeout.
    """
    n, p = X.shape

    common = {"n_patterns": int(n_patterns), "missing_frac": float(missing_frac)}

    def _skip(reason: str) -> dict[str, Any]:
        return make_record(
            engine="r:mvnmle", dataset=survey, p=p, n=n, wall=None,
            summary=summarize_covariance(None),
            extra={**common, "error": reason},
        )

    if p > R_VAR_LIMIT:
        return _skip(f"skipped: p={p} exceeds R mvnmle {R_VAR_LIMIT}-variable limit")
    if shutil.which("Rscript") is None:
        return _skip("skipped: Rscript not found on PATH")

    tmp = Path(tempfile.mkdtemp(prefix="rmvnmle_"))
    csv_in, json_out = tmp / "data.csv", tmp / "out.json"
    try:
        # Write CSV with literal "NA" for missing; no header, no index.
        arr = np.asarray(X, dtype=np.float64)
        np.savetxt(csv_in, arr, delimiter=",",
                   fmt="%.10g")  # NaN -> "nan"; fix to NA below
        text = csv_in.read_text().replace("nan", "NA")
        csv_in.write_text(text)

        proc = subprocess.run(
            ["Rscript", str(_R_SCRIPT), str(csv_in), str(json_out),
             str(warmup), str(reps)],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if proc.returncode != 0:
            return _skip(f"R error (rc={proc.returncode}): {proc.stderr.strip()[:300]}")

        out = json.loads(json_out.read_text())
        sigma = np.asarray(out["sigmahat"], dtype=float).reshape(p, p)
        mu = np.asarray(out["muhat"], dtype=float)
        wall = {
            "median_s": out["wall_median_s"],
            "min_s": out["wall_min_s"],
            "max_s": out["wall_max_s"],
            "times_s": out["wall_times_s"],
        }
        # R mvnmle's `value` is -2*loglik (the minimised objective); convert.
        value = out.get("value")
        loglik = (-0.5 * value) if value is not None else None
        return make_record(
            engine="r:mvnmle", dataset=survey, p=p, n=n, loglik=loglik,
            n_iter=out.get("iterations"), converged=None, wall=wall,
            backend_name="r_mvnmle", precision="fp64",
            parameterization="inverse_cholesky",
            summary=summarize_covariance(sigma),
            extra={
                **common,
                "mu": np.asarray(mu, float).tolist(),
                "r_value_neg2ll": value,
                "stop_code": out.get("stop_code"),
            },
        )
    except subprocess.TimeoutExpired:
        return _skip(f"timeout after {timeout_s:.0f}s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
