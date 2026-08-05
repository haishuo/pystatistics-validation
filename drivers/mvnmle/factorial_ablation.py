"""Component-factorial ablation of the two trace/gradient reformulations.

For each (survey, p) and each cell of the 2x2 factorial
    trace    in {solve, blocked}
    gradient in {autodiff, analytic}
this times one objective evaluation and one gradient evaluation on the backend it
is launched on (MPS on a Mac, CUDA on a CUDA box). Per-evaluation timing is the
quantity the attribution claims concern ("the analytical gradient alone removes
~20 of the 22s") and isolates each reformulation independently --- not confounded
by version bundling or by optimizer convergence. Run once per backend and merge.

Holds batching ON (the batched objective); the batching contribution is measured
separately against the per-pattern-loop baseline (pystatistics v3.5.1).

Usage:  python factorial_ablation.py [mps|cuda|cpu] [tag]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
# Paper harness modules live beside this file (Forge flat bundle) or one level up
# in code/ (dev layout); pystatistics comes from PYTHONPATH / the installed pkg.
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]
from survey_io import build_mvn_problem            # noqa: E402
from curate import standardize_columns             # noqa: E402
from pystatistics.mvnmle._objectives.gpu_fp32 import GPUObjectiveFP32  # noqa: E402
from pystatistics.mvnmle._objectives._batched_cholesky import (  # noqa: E402
    objective_value, analytic_gradient, accumulate_gradient)
import pystatistics                                 # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else (
    "cuda" if torch.cuda.is_available() else
    ("mps" if torch.backends.mps.is_available() else "cpu"))
TAG = sys.argv[2] if len(sys.argv) > 2 else f"factorial_{DEV}"
# data/ sits beside this file (Forge flat bundle) or at the repo root (dev)
from store_io import DEFAULT_NAMESPACE, store_root  # noqa: E402

_DATA = store_root() / DEFAULT_NAMESPACE
# KNOWN DEFECT (pre-existing, deliberately unchanged): this writes beside the
# store rather than into artifacts/mvnmle/<version>/runs/ where every other
# driver's evidence lives, and that directory does not exist. It is pinned to
# store_root().parent so it resolves exactly where it did before the store was
# namespaced — deriving it from _DATA would now point *inside* the store. Where
# these exploratory runs belong is a layout decision, not a mechanical fix.
_OUT = store_root().parent / "results" / f"{TAG}.json"

MATRIX = [("wvs", [25, 50, 100]), ("gss", [25, 50, 100])]
TRACES = ["solve", "blocked"]
GRADS = ["autodiff", "analytic"]


def _sync():
    if DEV == "cuda":
        torch.cuda.synchronize()
    elif DEV == "mps":
        torch.mps.synchronize()


def _time(fn, n=5):
    fn(); _sync()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    _sync()
    return (time.perf_counter() - t) / n * 1000.0


def main():
    recs = []
    for survey, ps in MATRIX:
        for p in ps:
            try:
                prob = build_mvn_problem(str(_DATA / f"{survey}.h5"), p)
                X = standardize_columns(prob.X)
                obj = GPUObjectiveFP32(X, device=DEV)
                theta = obj.get_initial_parameters()
                consts, eps, cs = obj._consts, obj.eps, obj.chunk_size
                with torch.no_grad():
                    mu, sigma = obj._unpack_gpu(
                        torch.tensor(theta, device=obj.device, dtype=obj.dtype))
            except Exception as e:  # noqa: BLE001
                recs.append({"survey": survey, "p": p, "error": f"{type(e).__name__}: {e}"})
                print(f"{survey} p={p}: ERROR {e}", flush=True)
                continue
            for trace in TRACES:
                # objective eval (forward) under the chosen trace
                obj_ms = _time(lambda tr=trace:
                               _obj_call(torch, mu, sigma, consts, eps, cs, tr))
                for grad in GRADS:
                    th = torch.tensor(theta, device=obj.device, dtype=obj.dtype,
                                      requires_grad=True)
                    if grad == "analytic":
                        g_ms = _time(lambda tr=trace: analytic_gradient(
                            torch, th, obj._unpack_gpu, consts, eps, cs, method=tr))
                    else:
                        g_ms = _time(lambda tr=trace: accumulate_gradient(
                            torch, th, obj._unpack_gpu, consts, eps, cs, method=tr))
                    rec = dict(backend=DEV, survey=survey, p=p, trace=trace,
                               grad=grad, objective_ms=round(obj_ms, 2),
                               gradient_ms=round(g_ms, 2),
                               version=pystatistics.__version__)
                    recs.append(rec)
                    print(f"{survey} p={p:3d} {trace:7s}+{grad:8s}: "
                          f"obj={obj_ms:8.1f}ms  grad={g_ms:9.1f}ms", flush=True)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps({"records": recs}, indent=2))
    print("wrote", _OUT)


def _obj_call(torch, mu, sigma, consts, eps, cs, trace):
    with torch.no_grad():
        return objective_value(torch, mu, sigma, consts, eps, cs, method=trace).item()


if __name__ == "__main__":
    raise SystemExit(main())
