"""Generate the LMM CPU-vs-R scaling artifacts (RIGOR priority 3 / R1).

The CPU path is the R-equivalent path and must never lag R (RIGOR priority 3).
This module fits the SAME random-effects model with pystatistics ``mixed.lmm`` and
``lme4::lmer`` across a spread of problem sizes — driven by the NUMBER OF GROUPS
(the dimension that grows the random-effects design) at a fixed group size — for
two RE complexities (random intercept; random intercept + slope), times both, and
estimates the empirical complexity exponent of each engine.

This is the canonical place an R1 complexity-class gap surfaces (cf. the coxph
O(n^2)-vs-O(n) incident). pystatistics builds a DENSE Z (shape n x sum(J_k*q_k))
and solves a dense penalized-least-squares system, whereas lme4 uses a SPARSE Z
and a sparse Cholesky — so a gap that widens with the number of groups is the
expected, and important, finding to measure honestly rather than sweep.

R's BLAS is named in the artifact (R11): on Powerhouse R links Apple Accelerate
(vecLib) — the same BLAS numpy links — so the CPU-vs-R comparison is same-BLAS.

All designs are SYNTHETIC + DETERMINISTIC (seeded), generated in-driver (the
scaling precedent; R17). ``mixed`` is CPU-only — there is NO GPU scaling pivot
(the GPU question is a separate investigation; see the report's priority-4
section).

    python -m drivers.mixed.generate_cpu_speed --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run

# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
# Drivers hardcode their artifact dir, so an ordinary run would otherwise
# silently destroy the artifacts a report was blessed against.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from drivers.mixed.datasets import MixedDataset
from drivers.mixed.run_pystatistics import run_lmm_record, strip_arrays
from drivers.mixed.run_r_mixed import run_r_lmm_record

from artifact_root import artifact_root  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260630
# Group counts to sweep (fixed group size). Wide enough to reveal the slope; the
# top end (G=2000) is where the 4.4.1 dense path collapsed (minutes / OOM) and
# where the 4.5.0 structure-exploiting solver must now hold near-linear vs lmer.
_GROUP_COUNTS = [10, 25, 50, 100, 200, 400, 800, 2000]
_GROUP_SIZE = 8


def _make(G: int, *, slope: bool, seed: int) -> MixedDataset:
    rng = np.random.default_rng(seed)
    ni = _GROUP_SIZE
    g = np.repeat(np.arange(G), ni)
    n = g.size
    x = rng.standard_normal(n)
    b0 = rng.normal(0.0, 2.0, size=G)
    if slope:
        b1 = rng.normal(0.0, 1.0, size=G)
        y = 1.0 + 0.5 * x + b0[g] + b1[g] * x + rng.standard_normal(n)
        re = {"g": ["1", "x"]}; rdata = {"x": x}
        formula = "y ~ x + (1 + x | g)"
    else:
        y = 1.0 + 0.5 * x + b0[g] + rng.standard_normal(n)
        re = {"g": ["1"]}; rdata = {}
        formula = "y ~ x + (1 | g)"
    X = np.column_stack([np.ones(n), x])
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return MixedDataset(
        key=f"G{G}_{'slope' if slope else 'intercept'}",
        why="scaling design", y=y, X=X, fixed_names=["(Intercept)", "x"],
        groups={"g": g}, random_effects=re, random_data=rdata,
        r_frame=frame, r_formula=formula)


def _slope(ns: list[float], ts: list[float]) -> float:
    """Empirical complexity exponent = slope of log(t) vs log(n) (least squares)."""
    ln_n = np.log(np.asarray(ns, float)); ln_t = np.log(np.asarray(ts, float))
    return float(np.polyfit(ln_n, ln_t, 1)[0])


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    curves: dict[str, dict[str, list[float]]] = {}

    for slope in (False, True):
        complexity = "intercept+slope" if slope else "intercept"
        curves[complexity] = {"n": [], "G": [], "py": [], "r": []}
        for G in _GROUP_COUNTS:
            ds = _make(G, slope=slope, seed=_SEED + G + (1000 if slope else 0))
            sut = run_lmm_record(ds, repeats=repeats, warmup=1,
                                 compute_satterthwaite=True)
            if sut.get("error"):
                raise RuntimeError(f"pystatistics failed {ds.key}: {sut['error']}")
            ref, _raw = run_r_lmm_record(ds, reps=repeats)
            records.extend(strip_arrays([sut, ref]))
            tp = sut["wall_median_s"]; tr = ref["wall_median_s"]
            curves[complexity]["n"].append(ds.n)
            curves[complexity]["G"].append(G)
            curves[complexity]["py"].append(tp)
            curves[complexity]["r"].append(tr)
            row = {
                "complexity": complexity, "G": G, "group_size": _GROUP_SIZE,
                "n": ds.n, "p": ds.p,
                "pystat_s": tp, "r_lmer_s": tr,
                "speedup_vs_r": (tr / tp if tp else None),
            }
            rows.append(row)
            print(f"  {complexity:16s} G={G:4d} n={ds.n:5d}  "
                  f"pystat={tp*1e3:8.1f}ms  lmer={tr*1e3:7.1f}ms  "
                  f"speedup_vs_r={row['speedup_vs_r']:.2f}x")

    # Empirical complexity exponents (slope of log-time vs log-n) per engine.
    exps = {}
    for complexity, c in curves.items():
        exps[complexity] = {
            "pystatistics_exponent": _slope(c["n"], c["py"]),
            "r_lmer_exponent": _slope(c["n"], c["r"]),
        }
        print(f"  [{complexity}] empirical exponent  pystat={exps[complexity]['pystatistics_exponent']:.2f}"
              f"  lmer={exps[complexity]['r_lmer_exponent']:.2f}")

    config = {
        "study": "cpu_speed_vs_r",
        "group_counts": _GROUP_COUNTS, "group_size": _GROUP_SIZE,
        "complexities": ["intercept", "intercept+slope"],
        "backend": "cpu", "repeats": repeats, "seed": _SEED,
        "reference": "R lme4::lmer (lmerTest for df/p); R BLAS = Apple Accelerate "
                     "(vecLib), same BLAS numpy links on this host (R11)",
        "empirical_exponents": exps,
        "scaling_dimension": "number of groups G at fixed group size "
                             f"({_GROUP_SIZE}); n = G*group_size",
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = artifact_root(_REPO) / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"cpu_speed_{host}.json", run)
    _write_summary_csv(out_dir / f"cpu_speed_{host}_summary.csv", rows)
    print(f"\nwrote {run_path}")
    return run_path


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
