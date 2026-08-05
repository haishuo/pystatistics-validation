"""Generate the GLMM CPU-vs-glmer scaling artifacts (RIGOR priority 3 / R1).

The CPU path is the R-equivalent path and must never lag R (RIGOR priority 3).
This fits the SAME binomial random-intercept GLMM with pystatistics ``mixed.glmm``
(Laplace, nAGQ=1) and ``lme4::glmer`` (Laplace, nAGQ=1) across a spread of group
counts (the dimension that grows the random-effects design) at a fixed group size,
times both, and estimates each engine's empirical complexity exponent — the
canonical place an R1 complexity-class gap would surface.

``mixed`` is CPU-only: there is NO GPU pivot for the general GLMM (a documented
decision matching lme4 / glmmTMB; the GPU question was settled in the LMM/GRM GPU
investigation). R's BLAS is named in the artifact (R11): on Powerhouse R links
Apple Accelerate — the same BLAS numpy links — so the comparison is same-BLAS.

All designs are SYNTHETIC + DETERMINISTIC (seeded), generated in-driver (the
scaling precedent; R17).

    python -m drivers.mixed.generate_glmm_speed --host powerhouse
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

from drivers.mixed.glmm_datasets import GLMMDataset
from drivers.mixed.run_glmm import run_glmm_record
from drivers.mixed.run_r_glmm import run_r_glmm_record

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260701
_PER = 15                       # observations per group (fixed)
_GROUP_COUNTS = [10, 25, 50, 100, 200, 400]


def _make(G: int) -> GLMMDataset:
    rng = np.random.default_rng(_SEED + G)
    n = G * _PER
    g = np.repeat(np.arange(G), _PER)
    b = rng.normal(0, 0.7, G)
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.2 + 0.6 * x + b[g])))).astype(float)
    X = np.column_stack([np.ones(n), x])
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return GLMMDataset(
        key=f"binom_G{G}", why="binomial random-intercept scaling point",
        y=y, X=X, fixed_names=["(Intercept)", "x"],
        groups={"g": g}, random_effects={"g": ["1"]}, random_data={},
        family="binomial", link="logit",
        r_frame=frame, r_formula="y ~ x + (1 | g)",
        r_family="binomial", r_link="logit", r_factor_cols=("g",))


def _exponent(ns, ts) -> float:
    ns = np.asarray(ns, float); ts = np.asarray(ts, float)
    m = ts > 0
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(ns[m]), np.log(ts[m]), 1)[0])


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    ns, py_t, r_t = [], [], []

    for G in _GROUP_COUNTS:
        ds = _make(G)
        sut = run_glmm_record(ds, repeats=repeats, warmup=1)
        if sut.get("error"):
            raise RuntimeError(f"glmm failed at G={G}: {sut['error']}")
        ref, _raw = run_r_glmm_record(ds, reps=repeats)
        py = float(sut["wall_median_s"]); rr = float(ref["wall_median_s"])
        speedup = rr / py if py > 0 else float("nan")
        # keep records slim (no per-fit arrays for a timing study)
        for rec in (sut, ref):
            for k in ("blups", "var_components", "coefficients",
                      "standard_errors", "z_values", "p_values"):
                rec.pop(k, None)
        records.extend([sut, ref])
        rows.append({"G": G, "n": ds.n, "pystat_s": round(py, 4),
                     "r_glmer_s": round(rr, 4), "speedup_vs_r": round(speedup, 3)})
        ns.append(ds.n); py_t.append(py); r_t.append(rr)
        print(f"  G={G:4d} n={ds.n:5d}  py={py:.4f}s  glmer={rr:.4f}s  "
              f"speedup={speedup:.2f}x")

    py_exp, r_exp = _exponent(ns, py_t), _exponent(ns, r_t)
    print(f"  empirical exponent: pystatistics ~O(n^{py_exp:.2f}), "
          f"glmer ~O(n^{r_exp:.2f})")

    config = {
        "study": "glmm_cpu_speed_vs_glmer",
        "backend": "cpu",
        "repeats": repeats,
        "reference": "R lme4::glmer (Laplace, nAGQ=1)",
        "per_group": _PER,
        "empirical_exponent": {"pystatistics": py_exp, "glmer": r_exp},
        "blas": "Apple Accelerate (numpy + R), same-BLAS",
        "note": "Binomial random-intercept GLMM; group-count sweep at fixed "
                "group size. CPU-only (no GPU pivot for the general GLMM).",
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"glmm_speed_cpu_{host}.json", run)
    _write_summary_csv(out_dir / f"glmm_speed_{host}_summary.csv", rows)
    print(f"\nwrote {run_path}")
    return run_path


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
