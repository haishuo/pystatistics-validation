"""Head-to-head: pystatistics GPU MVN MLE vs. semopy FIML on the same estimand.

semopy targets structural equation models; the analog of estimating the mean and
covariance of a multivariate normal under missingness is a *saturated* model. A
saturated model is not directly expressible as covariances-only in semopy (which
registers variables only from the structural part), so we use the equivalent
fully recursive (triangular) regression x_i ~ x_{i+1} + ... + x_{p-1}, which is
just-identified (0 df) and whose model-implied covariance equals the saturated
FIML estimate. We compare that implied covariance, and the wall-clock, against
``pystatistics.mlest`` on the same standardized survey data.

Run in an environment with semopy + pystatistics installed:
    python semopy_bench.py nhanes_cardio 11
    python semopy_bench.py simw 10,25

Usage:  python semopy_bench.py SURVEY PS
"""

import sys
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]
from problem_source import resolve_problem         # noqa: E402
from curate import standardize_columns             # noqa: E402

SURVEY = sys.argv[1] if len(sys.argv) > 1 else "simw"
PS = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [10, 25]
import os                                          # noqa: E402
_OUT = Path(os.environ.get("VALIDATION_ARTIFACT_ROOT",
                           str(_HERE.parent.parent / "results"))) / f"semopy_vs_pystat_{SURVEY}.json"


def main():
    import semopy
    from pystatistics.mvnmle import mlest
    import pystatistics

    recs = []
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    for p in PS:
        prob = resolve_problem(SURVEY, p)
        X = standardize_columns(prob.X).astype(np.float64)
        n = X.shape[0]
        cols = [f"x{i}" for i in range(p)]
        df = pd.DataFrame(X, columns=cols)
        rec = {"survey": SURVEY, "p": p, "n": n,
               "pystatistics_version": pystatistics.__version__}

        # pystatistics GPU
        try:
            t = time.perf_counter()
            rg = mlest(X, backend="gpu", method="direct")
            rec["pystat_s"] = time.perf_counter() - t
            sig_ps = np.asarray(rg.sigmahat)
            rec["pystat_loglik"] = float(rg.loglik)
        except Exception as e:  # noqa: BLE001
            rec["pystat_error"] = f"{type(e).__name__}: {e}"; sig_ps = None

        # semopy FIML, recursive saturated model
        try:
            desc = "\n".join(f"{cols[i]} ~ " + " + ".join(cols[i + 1:])
                             for i in range(p - 1))
            m = semopy.Model(desc)
            t = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(df, obj="FIML")
            rec["semopy_s"] = time.perf_counter() - t
            sig_sem = np.asarray(m.calc_sigma()[0])
            if sig_ps is not None and sig_sem.shape == sig_ps.shape:
                d = np.abs(sig_sem - sig_ps)
                rec["sigma_max_abs_diff"] = float(d.max())
                rec["sigma_rel_fro"] = float(np.linalg.norm(sig_sem - sig_ps)
                                             / np.linalg.norm(sig_ps))
        except Exception as e:  # noqa: BLE001
            rec["semopy_error"] = f"{type(e).__name__}: {e}"

        if "semopy_s" in rec and "pystat_s" in rec:
            rec["speedup_semopy_over_pystat"] = rec["semopy_s"] / rec["pystat_s"]
        recs.append(rec)
        print(f"{SURVEY} p={p}: pystat={rec.get('pystat_s')}s semopy={rec.get('semopy_s')}s "
              f"speedup={rec.get('speedup_semopy_over_pystat')} "
              f"Sigma rel_fro={rec.get('sigma_rel_fro')} "
              f"{rec.get('semopy_error','')}{rec.get('pystat_error','')}", flush=True)
        _OUT.write_text(json.dumps({"records": recs}, indent=2))
    print("wrote", _OUT)


if __name__ == "__main__":
    raise SystemExit(main())
