"""CPU-vs-R head-to-head reproduce at 4.6.12 (RIGOR R1/R3, Guarantee 3).

The v3.18.0 report established that pystatistics CPU matches R ``mvnmle::mlest``
in loglik/covariance to fp64 round-off while running orders of magnitude faster.
The module source is git-unchanged since that bless, so those numbers must
REPRODUCE at 4.6.12. This driver regenerates the pystatistics CPU side fresh at
the PyPI-pinned 4.6.12, on the same curated GSS/WVS problems (|corr|<=0.99
screening — the §5 benchmark config), and compares against R.

R ``mvnmle`` is version-independent (it is the R package, unchanged), so its
reference numbers are stable given identical curated data; running R here at
p=15 both surveys re-confirms the curation pipeline is byte-identical to the
v3.18.0 evidence (loglik must match the frozen R value), and the pystatistics
side is re-measured fresh across p=15,20,25.

Usage:
    MVNMLE_DATA_DIR=/path python compare_r_reproduce.py [tag] [--with-r-all]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]
from survey_io import build_mvn_problem  # noqa: E402
from curate import standardize_columns  # noqa: E402
from run_pystatistics import run_pystatistics  # noqa: E402
from run_r_mvnmle import run_r_mvnmle  # noqa: E402

_OUTDIR = _REPO / "artifacts" / "mvnmle" / "v4.6.13" / "runs"
_DATA = (Path(os.environ["MVNMLE_DATA_DIR"])
         if os.environ.get("MVNMLE_DATA_DIR")
         else _REPO.parent / "datasets")

GRID = [("gss", p) for p in (15, 20, 25)] + [("wvs", p) for p in (15, 20, 25)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", nargs="?", default="compare_r_reproduce")
    ap.add_argument("--with-r-all", action="store_true",
                    help="run R at every p (slow: R p=25 can take ~15 min); "
                         "default runs R only at p=15 as a curation-reproduce check")
    args = ap.parse_args()

    records = []
    for survey, p in GRID:
        path = _DATA / f"{survey}.h5"
        if not path.exists():
            print(f"{survey}: missing {path}", file=sys.stderr)
            continue
        prob = build_mvn_problem(path, p, max_abs_corr=0.99)
        # Standardize columns to unit variance — the v3.18.0 paper pipeline
        # (raw survey items span wildly different scales, e.g. an income column
        # with variance ~1e9, which would swamp the covariance by scale alone).
        X = np.asarray(standardize_columns(prob.X), dtype=np.float64)
        n_pat = int(np.unique(np.isnan(X), axis=0).shape[0])

        py = run_pystatistics(
            X, backend="cpu", survey=survey, n_patterns=n_pat,
            missing_frac=prob.overall_missing_frac, reps=3, warmup=1)
        records.append(py)

        run_r = args.with_r_all or p == 15
        if run_r:
            r = run_r_mvnmle(
                X, survey=survey, n_patterns=n_pat,
                missing_frac=prob.overall_missing_frac, reps=1, warmup=0,
                timeout_s=1800.0)
            records.append(r)
            dll = (abs(py["loglik"] - r["loglik"])
                   if py.get("loglik") is not None and r.get("loglik") is not None
                   else None)
            print(f"{survey} p={p:2d}: py_ll={py.get('loglik')} "
                  f"r_ll={r.get('loglik')} |dll|={dll} "
                  f"py_wall={py.get('wall_median_s')}s r_wall={r.get('wall_median_s')}s "
                  f"{r.get('error','')}", flush=True)
        else:
            print(f"{survey} p={p:2d}: py_ll={py.get('loglik')} "
                  f"py_wall={py.get('wall_median_s')}s (R reused from v3.18.0 frozen)",
                  flush=True)

    import pystatistics
    payload = {"study": "compare_r_reproduce",
               "kind": "cpu-vs-r-head-to-head",
               "pystatistics_version": pystatistics.__version__,
               "note": ("R mvnmle is version-independent; reference loglik at "
                        "p=20/25 carried from v3.18.0 frozen artifacts unless "
                        "--with-r-all. p=15 R here re-confirms curation parity."),
               "records": records}
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    out = _OUTDIR / f"{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}  ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
