"""fp32 GPU no-silent-wrong verification + R10 separation stress for mice (R12/R15).

mice was blessed (correctness studies) under the pre-red-team 3.16.3 regime and
its fp32 GPU path — batched chained-equation draws in float32 — was never run
through R12. Unlike mvnmle, mice's CF-1 exposure is already *pre-mitigated* in the
code: the separation-prone discrete fits (logreg/polyreg/polr) are computed in
FP64 on CUDA/CPU (``discrete_glm_compute_dtype``), and the numeric norm/pmm draw
uses a ridged Gram + an end-of-sweep non-finite fail-loud guard. This driver
VERIFIES those claims hold at 4.6.12:

  1. Numeric fp32 gate (R12): on a known-truth linear DGP, pool each backend's
     imputations by Rubin's rules and check the pooled slopes + SEs agree and
     cover the truth. The clean precision-isolation is ``gpu_fp64`` vs ``gpu``
     (fp32) on CUDA — same torch RNG, only precision differs; any systematic gap
     is the fp32 effect. ``cpu`` (fp64) is the reference.
  2. Separation hard case (R10): a categorical column driven to (quasi-)complete
     separation under the chained equations. The fit must NOT silently collapse
     every imputed cell onto one category — it either stays well-behaved (fp64
     discrete fit) or fails loud (non-finite guard). Checked cpu vs gpu.
  3. Default invocation (R15): the naive ``mice(data, seed=0)`` call.

Device = installed torch (MPS on Mac, CUDA on Forge). CUDA is authoritative.

Usage:
    python mice_gate.py [tag]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path[:0] = [str(_HERE.parent / "_shared")]
from dgp import make_complete, make_mixed, impose_mcar  # noqa: E402

import pystatistics  # noqa: E402

# Guard artifact writes: refuse to clobber evidence committed to git unless
# PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1. See drivers/_shared/artifact_guard.py.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import guard_artifact_path  # noqa: E402

from artifact_root import artifact_root  # noqa: E402

_OUTDIR = artifact_root(_REPO) / "mice" / f"v{pystatistics.__version__}" / "runs"
_SEED = 20260707


# --------------------------------------------------------------------------- #
# Rubin-pooled OLS analysis of a completed dataset.
# --------------------------------------------------------------------------- #
def _ols(y, X):
    """OLS of y on [1, X]: return (beta (k,), se^2 (k,)) with k = 1 + X.shape[1]."""
    Z = np.column_stack([np.ones(len(y)), X])
    XtX = Z.T @ Z
    XtXinv = np.linalg.inv(XtX)
    beta = XtXinv @ (Z.T @ y)
    resid = y - Z @ beta
    dof = max(len(y) - Z.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    var = sigma2 * np.diag(XtXinv)
    return beta, var


def _pooled_fit(sol):
    """Pool a MICESolution's completed datasets by an analysis of y (col 0) on
    X (cols 1..). Returns PooledSolution over the k = p coefficients."""
    from pystatistics.mice import pool
    ests, vars = [], []
    for D in sol.completed_datasets():
        b, v = _ols(D[:, 0], D[:, 1:])
        ests.append(b)
        vars.append(v)
    n = sol.completed(0).shape[0]
    k = np.asarray(ests).shape[1]
    return pool(np.asarray(ests), np.asarray(vars), df_complete=n - k)


# --------------------------------------------------------------------------- #
# Study 1 — numeric fp32 gate.
# --------------------------------------------------------------------------- #
def numeric_gate() -> list[dict]:
    import torch
    from pystatistics.mice import mice

    n, p, M = 3000, 5, 30
    cd = make_complete(n, p, seed=_SEED, rho=0.5)   # [y|X], known beta (p+1,)
    truth = cd.beta                                  # intercept first
    data = impose_mcar(cd.matrix, rate=0.25, seed=_SEED + 1)

    backends = ["cpu", "gpu"]
    if torch.cuda.is_available():
        backends.append("gpu_fp64")

    out = []
    for method in ("norm", "pmm"):
        ref_est = None
        for bk in backends:
            try:
                sol = mice(data, n_imputations=M, max_iter=5, method=method,
                           seed=_SEED + 2, backend=bk)
                pooled = _pooled_fit(sol)
                est = np.asarray(pooled.estimate)
                se = np.asarray(pooled.standard_errors)
                lo, hi = np.asarray(pooled.ci_lower), np.asarray(pooled.ci_upper)
                covered = bool(np.all((truth >= lo) & (truth <= hi)))
                rec = {"study": "numeric_gate", "method": method, "backend": bk,
                       "backend_name": sol.backend_name,
                       "max_abs_bias_vs_truth": float(np.max(np.abs(est - truth))),
                       "covers_truth": covered,
                       "estimate": est.tolist(), "se": se.tolist(),
                       "fmi_max": float(np.max(pooled.fmi))}
                if ref_est is None and bk == "cpu":
                    ref_est, ref_se = est, se
                if ref_est is not None:
                    rec["max_abs_est_diff_vs_cpu"] = float(np.max(np.abs(est - ref_est)))
                    rec["max_rel_se_diff_vs_cpu"] = float(
                        np.max(np.abs(se - ref_se) / np.abs(ref_se)))
                out.append(rec)
                print(f"  {method:5s} {bk:9s} {sol.backend_name:18s} "
                      f"bias_vs_truth={rec['max_abs_bias_vs_truth']:.4f} "
                      f"covers={covered} "
                      f"est_diff_vs_cpu={rec.get('max_abs_est_diff_vs_cpu')} "
                      f"se_rel_diff={rec.get('max_rel_se_diff_vs_cpu')}", flush=True)
            except Exception as exc:  # noqa: BLE001
                out.append({"study": "numeric_gate", "method": method,
                            "backend": bk, "error": f"{type(exc).__name__}: {exc}"})
                print(f"  {method:5s} {bk:9s} ERROR {type(exc).__name__}: {exc}",
                      flush=True)
    return out


# --------------------------------------------------------------------------- #
# Study 1b — Rubin coverage validity: does fp32 GPU preserve ~95% coverage?
# --------------------------------------------------------------------------- #
def coverage_study(reps: int = 40) -> list[dict]:
    """The direct R12 validity test, RNG-confound-free: over ``reps`` independent
    datasets, does each backend's Rubin-pooled 95% CI cover the true slopes at the
    nominal rate? A silently-wrong fp32 path would under- or over-cover. Coverage
    is a rate over reps, so the numpy-vs-torch RNG difference is irrelevant — each
    backend is judged against the TRUTH, not against each other."""
    import torch
    from pystatistics.mice import mice

    n, p, M = 1500, 4, 10
    backends = ["cpu", "gpu"]
    if torch.cuda.is_available():
        backends.append("gpu_fp64")

    agg = {bk: {"cover": 0, "total": 0, "se_sum": 0.0} for bk in backends}
    for r in range(reps):
        cd = make_complete(n, p, seed=_SEED + 100 + r, rho=0.5)
        truth = cd.beta
        data = impose_mcar(cd.matrix, rate=0.25, seed=_SEED + 500 + r)
        for bk in backends:
            try:
                sol = mice(data, n_imputations=M, max_iter=5, method="norm",
                           seed=_SEED + 900 + r, backend=bk)
                pooled = _pooled_fit(sol)
                lo, hi = np.asarray(pooled.ci_lower), np.asarray(pooled.ci_upper)
                cov = (truth >= lo) & (truth <= hi)          # per-coefficient
                agg[bk]["cover"] += int(cov.sum())
                agg[bk]["total"] += cov.size
                agg[bk]["se_sum"] += float(np.mean(pooled.standard_errors))
            except Exception:  # noqa: BLE001
                pass
    out = []
    for bk in backends:
        a = agg[bk]
        rate = a["cover"] / a["total"] if a["total"] else None
        rec = {"study": "coverage", "backend": bk, "reps": reps,
               "coverage_rate": rate,
               "mean_se": a["se_sum"] / reps if reps else None,
               "n_coef_checked": a["total"]}
        out.append(rec)
        print(f"  coverage {bk:9s} rate={rate:.3f} (target 0.95) "
              f"mean_se={rec['mean_se']:.4f} over {reps} reps", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Study 2 — separation hard case (R10).
# --------------------------------------------------------------------------- #
def separation_case() -> list[dict]:
    """A binary column perfectly separated by a numeric predictor, with a few
    values missing. Under chained equations the logreg fit is (quasi-)separated;
    the imputations must not silently collapse onto one class."""
    from pystatistics.mice import mice
    from pystatistics.mice.design import MICEDesign
    import torch

    rng = np.random.default_rng(_SEED + 10)
    n = 1500
    x = rng.standard_normal(n)
    b = (x > 0.0).astype(float)          # perfectly separated by x (sign)
    # a second correlated numeric column so the chain has something to fit
    z = 0.7 * x + 0.3 * rng.standard_normal(n)
    data = np.column_stack([x, b, z])
    # knock out 12% of the binary column (the separated one)
    miss = rng.random(n) < 0.12
    data[miss, 1] = np.nan

    out = []
    backends = ["cpu", "gpu"]
    if torch.cuda.is_available():
        backends.append("gpu_fp64")
    for bk in backends:
        try:
            design = MICEDesign.from_array(
                data, method="auto", column_kinds=["numeric", "binary", "numeric"])
            sol = mice(design, n_imputations=10, max_iter=5, seed=_SEED + 11,
                       backend=bk)
            imp = sol.imputations(1)          # imputed binary values across chains
            uniq = np.unique(imp)
            frac1 = float(np.mean(imp == 1))
            collapsed = uniq.size == 1        # every imputed cell one class -> collapse
            rec = {"study": "separation", "backend": bk,
                   "backend_name": sol.backend_name,
                   "n_imputed": int(imp.size), "imputed_frac_class1": frac1,
                   "unique_imputed_values": uniq.tolist(),
                   "silently_collapsed": collapsed,
                   "warnings": "; ".join(sol.warnings) if sol.warnings else ""}
            print(f"  separation {bk:9s} frac1={frac1:.3f} "
                  f"collapsed={collapsed} uniq={uniq.tolist()} "
                  f"warn={rec['warnings'][:60]}", flush=True)
        except Exception as exc:  # noqa: BLE001 — a loud failure is an acceptable outcome
            rec = {"study": "separation", "backend": bk,
                   "outcome": "raised", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  separation {bk:9s} RAISED {type(exc).__name__}: {str(exc)[:80]}",
                  flush=True)
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
# Study 3 — R15 default invocation.
# --------------------------------------------------------------------------- #
def default_invocation() -> list[dict]:
    from pystatistics.mice import mice
    cd = make_complete(2000, 4, seed=_SEED + 20)
    data = impose_mcar(cd.matrix, rate=0.2, seed=_SEED + 21)
    try:
        sol = mice(data, seed=0)              # the naive day-one call
        rec = {"study": "default", "n_imputations": sol.n_imputations,
               "max_iter": sol.max_iter, "backend_name": sol.backend_name,
               "n_completed": len(sol.completed_datasets()),
               "warnings": "; ".join(sol.warnings) if sol.warnings else "",
               "outcome": "ok"}
        print(f"  default mice(data, seed=0): n_imp={sol.n_imputations} "
              f"maxit={sol.max_iter} backend={sol.backend_name} ok", flush=True)
    except Exception as exc:  # noqa: BLE001
        rec = {"study": "default", "outcome": "raised",
               "error": f"{type(exc).__name__}: {exc}"}
        print(f"  default RAISED {type(exc).__name__}: {exc}", flush=True)
    return [rec]


def main() -> int:
    import platform
    tag = sys.argv[1] if len(sys.argv) > 1 else "mice_gate"
    print("== numeric fp32 gate =="); recs = numeric_gate()
    print("== rubin coverage validity =="); recs += coverage_study()
    print("== separation hard case =="); recs += separation_case()
    print("== default invocation =="); recs += default_invocation()

    try:
        import torch
        dev = (f"cuda:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available()
               else ("mps" if torch.backends.mps.is_available() else "cpu"))
        tver = torch.__version__
    except Exception:  # noqa: BLE001
        dev, tver = "unknown", None
    import pystatistics
    payload = {"study": "mice_gate", "kind": "r12-r10-r15-fp32-separation-default",
               "pystatistics_version": pystatistics.__version__,
               "device": dev, "torch_version": tver, "host": platform.node(),
               "records": recs}
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    out = _OUTDIR / f"{tag}.json"
    guard_artifact_path(out)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out} (device={dev})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
