"""CPU-vs-R fidelity reproduce for mice at 4.6.12 (R1/R3 correctness, Guarantee 3).

mice imputation is stochastic, so agreement with R ``mice`` is DISTRIBUTIONAL, not
exact: for each incomplete column we compare pystatistics' pooled imputed values
against R's, by mean / std / quantile difference and 1-Wasserstein distance. The
CPU path is git-unchanged since the 3.16.3 correctness bless, so the distributional
agreement must reproduce at the PyPI-pinned 4.6.12.

Mixed-type DGP exercises the full method set: numeric (pmm), binary (logreg),
ordered (polr). Reference: R ``mice`` (pinned in ``_r/mice_run.R``).

Usage:
    python compare_r_reproduce.py [tag]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]
from dgp import make_mixed, impose_mcar  # noqa: E402
from run_r_mice import run_r_mice_record  # noqa: E402

import pystatistics  # noqa: E402

# Guard artifact writes: refuse to clobber evidence committed to git unless
# PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1. See drivers/_shared/artifact_guard.py.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import guard_artifact_path  # noqa: E402

from artifact_root import artifact_root  # noqa: E402

_OUTDIR = artifact_root(_REPO) / "mice" / f"v{pystatistics.__version__}" / "runs"
_SEED = 20260707


def _wasserstein1(a, b) -> float:
    """1-Wasserstein between two 1-D empirical samples (sorted-quantile form)."""
    a = np.sort(np.asarray(a, float)); b = np.sort(np.asarray(b, float))
    q = np.linspace(0, 1, 200)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def main() -> int:
    from pystatistics.mice import mice
    from pystatistics.mice.design import MICEDesign

    tag = sys.argv[1] if len(sys.argv) > 1 else "compare_r_reproduce"
    n = 2000
    kinds = ("numeric", "numeric", "numeric", "binary", "ordered")
    md = make_mixed(n, seed=_SEED, n_numeric=3,
                    cat_specs=(("binary", 2), ("ordered", 4)))
    data = impose_mcar(md.matrix, rate=0.20, seed=_SEED + 1)

    # pystatistics CPU (fp64 reference path), live API.
    design = MICEDesign.from_array(data, method="auto", column_kinds=list(kinds))
    sol = mice(design, n_imputations=5, max_iter=5, seed=_SEED + 2, backend="cpu")

    # R mice on the identical matrix.
    r_rec, r_percol = run_r_mice_record(
        data, dataset="mixed_fidelity", m=5, maxit=5, method="pmm",
        seed=_SEED + 2, column_kinds=kinds)

    records = []
    names = [f"x{j}" for j in range(data.shape[1])]
    for col in range(data.shape[1]):
        if not np.isnan(data[:, col]).any():
            continue
        py_imp = np.asarray(sol.imputations(col), float).ravel()
        r_imp = np.asarray(r_percol.get(names[col], []), float).ravel()
        if r_imp.size == 0:
            continue
        rec = {"column": names[col], "kind": kinds[col],
               "py_mean": float(py_imp.mean()), "r_mean": float(r_imp.mean()),
               "py_std": float(py_imp.std()), "r_std": float(r_imp.std()),
               "mean_abs_diff": float(abs(py_imp.mean() - r_imp.mean())),
               "wasserstein1": _wasserstein1(py_imp, r_imp),
               "py_n": int(py_imp.size), "r_n": int(r_imp.size)}
        records.append(rec)
        print(f"  {names[col]:4s} {kinds[col]:8s} "
              f"py_mean={rec['py_mean']:.3f} r_mean={rec['r_mean']:.3f} "
              f"|dmean|={rec['mean_abs_diff']:.3f} W1={rec['wasserstein1']:.3f}",
              flush=True)

    import pystatistics
    payload = {"study": "compare_r_reproduce", "kind": "cpu-vs-r-fidelity",
               "pystatistics_version": pystatistics.__version__,
               "py_backend": sol.backend_name, "r_wall_s": r_rec.get("wall_median_s"),
               "records": records}
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    out = _OUTDIR / f"{tag}.json"
    guard_artifact_path(out)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out} ({len(records)} incomplete columns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
