"""Collinearity-screening census: how many columns the benchmark replaces.

For each curated (survey, p) problem, reports how many of the top-``p`` ranked
columns are dropped by the near-collinearity screen (``|corr| > max_abs_corr``)
and back-filled from deeper in the ranked pool, so the problem still has exactly
``p`` columns. This is the quantity behind the paper's Setup claim that screening
touches "only a few" columns, not a substantial fraction.

The metric is **curation-only and version-independent**: it depends solely on the
survey ``.h5`` files and the screening configuration (the ``build_mvn_problem``
defaults), NOT on the installed ``pystatistics`` — this driver imports no
``pystatistics``. It is filed under the mvnmle v3.18.0 artifact set because that
is the evidence set the paper consumes; the number would be identical against any
library version.

Definition. Let ``ranked`` be the eligible columns in descending quality order
and ``kept`` (positions into ``ranked``, ascending) the survivors of the greedy
collinearity screen. The p-variable problem is ``kept[:p]``. The p-th survivor
sits at rank position ``kept[p-1]``, so reaching p independents consumes the top
``kept[p-1] + 1`` ranked columns, of which ``p`` are kept; hence

    n_replaced(p) = kept[p-1] - (p - 1).

Usage:  python collinearity_screening.py [tag]
        MVNMLE_DATA_DIR=/path/to/h5 python collinearity_screening.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import h5py

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]                    # drivers/mvnmle -> repo root
sys.path[:0] = [str(_HERE.parent / "_shared")]
from curate import (rank_eligible_columns, prune_collinear,  # noqa: E402
                    to_nan_matrix)
from survey_io import survey_column_stats, _VALUES, _MISSING  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "collinearity_screening"

# build_mvn_problem defaults (the config that produced the benchmark problems).
CONFIG = dict(allowed_types=("continuous", "ordinal"),
              min_asked_frac=0.30, min_observed_among_asked=0.50,
              observed_frac_band=(0.50, 0.95))
MAX_ABS_CORR = 0.99

from store_io import DEFAULT_NAMESPACE, store_root  # noqa: E402

_DATA = store_root() / DEFAULT_NAMESPACE
_OUTDIR = _REPO / "artifacts" / "mvnmle" / "v3.18.0" / "runs"

MATRIX = [("wvs", [5, 10, 15, 20, 25, 50, 100]),
          ("gss", [5, 10, 15, 20, 25, 50, 100]),
          ("cses", [5, 10, 15, 20, 25, 50]),
          ("afrobarometer", [5, 10, 15, 20, 25, 50])]


def screen(path: Path):
    """Return (ranked_size, kept_positions) for a survey's ranked pool."""
    stats, data_types, _names = survey_column_stats(path)
    ranked = rank_eligible_columns(stats, data_types, **CONFIG)
    pool_sorted = np.sort(ranked)
    with h5py.File(path, "r") as f:
        values = f[_VALUES][:, list(pool_sorted)]
        mask = f[_MISSING][:, list(pool_sorted)]
    pos = {int(c): i for i, c in enumerate(pool_sorted)}
    rank_pos = [pos[int(c)] for c in ranked]
    x_pool = to_nan_matrix(values, mask)[:, rank_pos]   # columns in rank order
    kept = prune_collinear(x_pool, MAX_ABS_CORR)
    return int(ranked.size), kept


def main():
    records = []
    for survey, ps in MATRIX:
        path = _DATA / f"{survey}.h5"
        if not path.exists():
            print(f"{survey}: missing {path}", file=sys.stderr)
            continue
        n_eligible, kept = screen(path)
        n_independent = int(kept.size)
        for p in ps:
            rec = dict(survey=survey, p=p, n_eligible=n_eligible,
                       n_independent=n_independent,
                       max_abs_corr=MAX_ABS_CORR)
            if n_eligible < p:
                rec.update(feasible=False, reason="p>eligible",
                           n_replaced=None, replaced_frac=None)
            elif n_independent < p:
                rec.update(feasible=False, reason="p>independent",
                           n_replaced=None, replaced_frac=None)
            else:
                n_replaced = int(kept[p - 1]) - (p - 1)
                rec.update(feasible=True, reason="",
                           n_replaced=n_replaced,
                           replaced_frac=round(n_replaced / p, 6))
            records.append(rec)
            print(f"{survey:14s} p={p:3d}  eligible={n_eligible:3d}  "
                  f"independent={n_independent:3d}  "
                  f"replaced={rec['n_replaced']}")

    _OUTDIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": "collinearity_screening",
        "kind": "curation-census",
        "version_independent": True,
        "config": {**CONFIG, "max_abs_corr": MAX_ABS_CORR,
                   "definition": "n_replaced(p) = kept[p-1] - (p-1)"},
        "records": records,
    }
    (_OUTDIR / f"{TAG}.json").write_text(json.dumps(payload, indent=2))
    fields = ["survey", "p", "n_eligible", "n_independent",
              "n_replaced", "replaced_frac", "feasible", "reason",
              "max_abs_corr"]
    with (_OUTDIR / f"{TAG}_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in fields})
    print(f"\nwrote {_OUTDIR / (TAG + '.json')}")
    print(f"wrote {_OUTDIR / (TAG + '_summary.csv')}")


if __name__ == "__main__":
    raise SystemExit(main())
