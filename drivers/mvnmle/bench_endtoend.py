"""End-to-end MVN MLE fits for the cross-architecture comparison.

Times full ``mlest(backend='gpu', method='direct')`` fits across surveys and
p, on whatever GPU the host provides (CUDA on the Forge box, Apple MPS on a Mac).
On 3.11.0 a plain GPU fit already uses each backend's optimal configuration: the
trace term is dispatched per device (blocked inverse on Metal, triangular solve
on CUDA) and the gradient is the closed-form analytic gradient everywhere. So
this single harness, run once per machine, yields the optimal end-to-end numbers
for both architectures.

Identical code path on both backends (only the resolved device differs), which is
what the cross-architecture claim requires.

The optional ``backend`` argument runs the same sweep on the FP64 CPU path
instead, which is how the double-precision *anchor* for a GPU sweep is produced:
the FP32 optima are only interpretable against the FP64 optimum of the identical
curated problem, and only this harness curates it identically. ``warmup`` is
exposed for that case — the CPU path has no device warmup to discard, and an
extra discarded fit at p=100 costs half an hour.

Usage:  python bench_endtoend.py [tag] [surveys] [ps] [reps] [max_iter] [backend] [warmup]
        e.g.  python bench_endtoend.py mps_endtoend wvs,gss 50,100 2
              python bench_endtoend.py cpu_anchor gss 50,100 1 500 cpu 0
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]
from survey_io import build_mvn_problem            # noqa: E402
from curate import standardize_columns             # noqa: E402
from run_pystatistics import run_pystatistics       # noqa: E402
import pystatistics                                 # noqa: E402

GPU_DEVICE = ("cuda" if torch.cuda.is_available() else
              ("mps" if torch.backends.mps.is_available() else "cpu"))
TAG = sys.argv[1] if len(sys.argv) > 1 else f"{GPU_DEVICE}_endtoend"
SURVEYS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["wvs", "gss"]
PS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [50, 100]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
MAXIT = int(sys.argv[5]) if len(sys.argv) > 5 else None
BACKEND = sys.argv[6] if len(sys.argv) > 6 else "gpu"
WARMUP = int(sys.argv[7]) if len(sys.argv) > 7 else 1

if BACKEND not in ("gpu", "cpu"):
    raise SystemExit(f"backend must be 'gpu' or 'cpu', got {BACKEND!r}")
# The device stamp must name what actually ran: a CPU sweep on a Mac is 'cpu',
# not the MPS device that happens to be present.
DEVICE = GPU_DEVICE if BACKEND == "gpu" else "cpu"

from store_io import DEFAULT_NAMESPACE, store_root  # noqa: E402

_DATA = store_root() / DEFAULT_NAMESPACE
# KNOWN DEFECT (pre-existing, deliberately unchanged): this writes beside the
# store rather than into artifacts/mvnmle/<version>/runs/ where every other
# driver's evidence lives, and that directory does not exist. It is pinned to
# store_root().parent so it resolves exactly where it did before the store was
# namespaced — deriving it from _DATA would now point *inside* the store. Where
# these exploratory runs belong is a layout decision, not a mechanical fix.
_OUT = store_root().parent / "results" / f"{TAG}.json"


def main() -> int:
    recs = []
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    for survey in SURVEYS:
        for p in PS:
            try:
                prob = build_mvn_problem(str(_DATA / f"{survey}.h5"), p)
                X = standardize_columns(prob.X)
                npat = int(np.unique(np.isnan(X), axis=0).shape[0])
                rec = run_pystatistics(
                    X, backend=BACKEND, survey=survey, n_patterns=npat,
                    missing_frac=prob.overall_missing_frac, reps=REPS,
                    warmup=WARMUP, max_iter=MAXIT)
            except Exception as e:  # noqa: BLE001
                rec = {"survey": survey, "p": p,
                       "error": f"{type(e).__name__}: {e}"}
            rec["pystatistics_version"] = pystatistics.__version__
            rec["device"] = DEVICE
            recs.append(rec)
            print(f"{survey} p={p}: wall={rec.get('wall_median_s')}s "
                  f"conv={rec.get('converged')} iter={rec.get('n_iter')} "
                  f"ll={rec.get('loglik')} {rec.get('error', '')}", flush=True)
            _OUT.write_text(json.dumps({"records": recs}, indent=2, default=str))
    print("wrote", _OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
