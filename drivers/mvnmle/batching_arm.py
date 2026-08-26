"""Batching arm: per-pattern-loop baseline for the batching reformulation.

The component-factorial ablation (factorial_ablation.py) holds batching ON and
attributes the trace and gradient reformulations within the batched kernel. This
script measures the remaining arm: the per-pattern Python loop that the batched
kernel replaced. It times one objective evaluation and one gradient evaluation
of the LOOPED GPU objective from pystatistics v3.5.1 (the last release before
batching shipped in 3.6.0).

The loop uses triangular-solve traces and reverse-mode autodiff gradients --- the
same math as the factorial's ``solve`` + ``autodiff`` cell. The only difference
is loop-over-patterns vs. batched-over-patterns, so

    (loop here)  vs.  (factorial solve+autodiff cell)

isolates the batching contribution, uncounfounded by the trace/gradient changes.

Run with pystatistics 3.5.1 installed (from PyPI) in a separate
environment — the sim generator needs only numpy — e.g.

    KMP_DUPLICATE_LIB_OK=TRUE python batching_arm.py mps batching_mps 25,50 5

Usage:  python batching_arm.py [mps|cuda|cpu] [tag] [p-list] [n]
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]
from problem_source import resolve_problem         # noqa: E402
from curate import standardize_columns             # noqa: E402
from pystatistics.mvnmle._objectives.gpu_fp32 import GPUObjectiveFP32  # noqa: E402
import pystatistics                                 # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else (
    "cuda" if torch.cuda.is_available() else
    ("mps" if torch.backends.mps.is_available() else "cpu"))
TAG = sys.argv[2] if len(sys.argv) > 2 else f"batching_{DEV}"
PS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [25, 50, 100]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 5

_OUT = Path(os.environ.get("VALIDATION_ARTIFACT_ROOT",
                           str(_HERE.parent.parent / "results"))) / f"{TAG}.json"

# The redistributable problem set (simulated survey profiles; see mvn_sim.py).
SURVEYS = ["simw", "simg"]


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def _time(fn, n):
    fn(); _sync()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    _sync()
    return (time.perf_counter() - t) / n * 1000.0


def main():
    recs = []
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    for survey in SURVEYS:
        for p in PS:
            try:
                prob = resolve_problem(survey, p)
                X = standardize_columns(prob.X)
                obj = GPUObjectiveFP32(X, device=DEV)
                theta = obj.get_initial_parameters()
                obj_ms = _time(lambda: obj.compute_objective(theta), N)
                g_ms = _time(lambda: obj.compute_gradient(theta), N)
                rec = dict(backend=DEV, survey=survey, p=p, mode="loop",
                           objective_ms=round(obj_ms, 2),
                           gradient_ms=round(g_ms, 2),
                           version=pystatistics.__version__)
                print(f"{survey} p={p:3d} loop: obj={obj_ms:10.1f}ms  "
                      f"grad={g_ms:11.1f}ms", flush=True)
            except Exception as e:  # noqa: BLE001
                rec = {"backend": DEV, "survey": survey, "p": p, "mode": "loop",
                       "error": f"{type(e).__name__}: {e}",
                       "version": pystatistics.__version__}
                print(f"{survey} p={p}: ERROR {e}", flush=True)
            recs.append(rec)
            _OUT.write_text(json.dumps({"records": recs}, indent=2))  # incremental
    print("wrote", _OUT)


if __name__ == "__main__":
    raise SystemExit(main())
