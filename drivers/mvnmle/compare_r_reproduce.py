"""CPU-vs-R head-to-head on the redistributable benchmark problems.

Regenerates the pystatistics CPU side fresh at the PyPI-pinned version and
compares against R ``mvnmle::mlest`` on the same problems: the simulated
survey-profile problems (``simw``/``simg``, |corr|<=0.99-screened by
construction) and the NHANES cardiometabolic extract (``nhanes_cardio``,
p = 11 — the real-data anchor). These replaced the registration-gated GSS/WVS
problems (JSS open-data policy, 2026-08); R references for the retired
problems live in the frozen v3.18.0-v6.0.1 artifacts.

R runs at every grid cell — the problems are new, so there is no frozen R
reference to carry forward. R p=25 can take ~15 min.

Usage:
    DATASETS_ROOT=/path python compare_r_reproduce.py [tag]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path[:0] = [str(_HERE), str(_HERE.parent / "_shared")]
from problem_source import resolve_problem  # noqa: E402
from curate import standardize_columns  # noqa: E402
from run_pystatistics import run_pystatistics  # noqa: E402
from run_r_mvnmle import run_r_mvnmle  # noqa: E402

import pystatistics as _ps  # noqa: E402

from artifact_root import artifact_root  # noqa: E402
_OUTDIR = artifact_root(_REPO) / "mvnmle" / f"v{_ps.__version__}" / "runs"

# Guard artifact writes: refuse to clobber evidence committed to git unless
# PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1. See drivers/_shared/artifact_guard.py.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import guard_artifact_path  # noqa: E402


GRID = ([("simg", p) for p in (15, 20, 25)]
        + [("simw", p) for p in (15, 20, 25)]
        + [("nhanes_cardio", 11)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tag", nargs="?", default="compare_r_reproduce")
    ap.add_argument("--grid", default=None,
                    help="override the grid: comma-separated dataset:p1+p2+... "
                         "specs, e.g. 'simw:5+10+15+20+25+50,nhanes_cardio:11' "
                         "(used on Forge for the single-machine reference sweep)")
    args = ap.parse_args()

    grid = GRID
    if args.grid:
        grid = [(name, int(p))
                for spec in args.grid.split(",")
                for name, ps in [spec.split(":")]
                for p in ps.split("+")]

    records = []
    for survey, p in grid:
        prob = resolve_problem(survey, p, max_abs_corr=0.99)
        # Standardize columns to unit variance — the v3.18.0 paper pipeline
        # (raw survey items span wildly different scales, e.g. an income column
        # with variance ~1e9, which would swamp the covariance by scale alone).
        X = np.asarray(standardize_columns(prob.X), dtype=np.float64)
        n_pat = int(np.unique(np.isnan(X), axis=0).shape[0])

        py = run_pystatistics(
            X, backend="cpu", survey=survey, n_patterns=n_pat,
            missing_frac=prob.overall_missing_frac, reps=3, warmup=1)
        records.append(py)

        # R runs at every cell: the problems are new, no frozen reference exists.
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

    import pystatistics
    payload = {"study": "compare_r_reproduce",
               "kind": "cpu-vs-r-head-to-head",
               "pystatistics_version": pystatistics.__version__,
               "note": ("Redistributable problems (simw/simg synthetic profiles + "
                        "nhanes_cardio extract); R run fresh at every cell."),
               "records": records}
    _OUTDIR.mkdir(parents=True, exist_ok=True)
    out = _OUTDIR / f"{args.tag}.json"
    guard_artifact_path(out)
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}  ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
