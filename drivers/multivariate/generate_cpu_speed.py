"""Generate the CPU-vs-R speed + empirical-complexity artifacts (priority 3).

One job: the MANDATORY priority-3 study -- pystatistics on the CPU path must NEVER
lag R (RIGOR R1/R3). Fit PCA (prcomp) and the FA ML core (factanal) across a
spread of n and p, timing pystatistics CPU vs R, and estimate the empirical
complexity exponent for each so a complexity-class gap (the coxph incident) cannot
hide. The reference BLAS each engine links is recorded (R11): on this Mac both
numpy and R use Apple Accelerate BLAS; R uses reference LAPACK (libRlapack),
numpy uses Accelerate LAPACK.

FA speed uses rotation='none' on BOTH engines: the varimax rotation is under a
gathered fix (finding F1) and is cheap relative to the ML optimisation, so the
core FA cost is the fair, F1-independent comparison.

    python -m drivers.multivariate.generate_cpu_speed --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run

# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
# Drivers hardcode their artifact dir, so an ordinary run would otherwise
# silently destroy the artifacts a report was blessed against.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from drivers.multivariate.run_pystatistics import (
    run_pca_record, run_fa_record, strip_arrays as _strip_arrays)
from drivers.multivariate.run_r_multivariate import run_r_pca_record, run_r_fa_record

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260630

# PCA sweeps: an n-sweep at fixed p and a p-sweep at fixed n (PCA scales with both).
_PCA_N_SWEEP = [(n, 50) for n in (1000, 5000, 20000, 100000)]
_PCA_P_SWEEP = [(10000, p) for p in (10, 50, 200, 500)]
# FA sweep: n-sweep at fixed p (FA cost = corrcoef O(n p^2) + optimiser on p).
_FA_N_SWEEP = [(n, 12) for n in (500, 2000, 10000, 50000)]


def _design(n: int, p: int) -> np.ndarray:
    """Deterministic Gaussian design (seed varies with shape for independence)."""
    rng = np.random.default_rng(_SEED + n * 131 + p)
    return rng.standard_normal((n, p))


def _slope(ns: list[float], ts: list[float]) -> float:
    """Empirical complexity exponent: slope of log(time) vs log(n)."""
    x = np.log(np.asarray(ns, float))
    y = np.log(np.asarray(ts, float))
    return float(np.polyfit(x, y, 1)[0])


def _pca_speed_rows(records: list[dict[str, Any]], sweep, repeats: int,
                    label: str) -> list[dict[str, Any]]:
    rows = []
    for n, p in sweep:
        X = _design(n, p)
        sut = run_pca_record(X, backend="cpu", dataset=f"synth_{n}x{p}",
                             scale=False, repeats=repeats, warmup=1)
        ref, _raw = run_r_pca_record(X, [f"x{i}" for i in range(p)],
                                     dataset=f"synth_{n}x{p}", center=True,
                                     scale=False, reps=repeats)
        records.extend([sut, ref])
        ws, wr = sut["wall_median_s"], ref["wall_median_s"]
        rows.append({
            "analysis": "pca", "sweep": label, "n": n, "p": p,
            "wall_pystat_s": ws, "wall_r_s": wr,
            "speedup_vs_r": (wr / ws) if (ws and wr) else None,
        })
        print(f"  PCA {label:6s} n={n:>7} p={p:>4}  "
              f"py={ws*1e3:8.2f}ms  R={wr*1e3:8.2f}ms  "
              f"speedup={rows[-1]['speedup_vs_r']:.2f}x")
    return rows


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    rows += _pca_speed_rows(records, _PCA_N_SWEEP, repeats, "n")
    rows += _pca_speed_rows(records, _PCA_P_SWEEP, repeats, "p")

    # FA core speed (rotation='none', both engines).
    fa_rows = []
    for n, p in _FA_N_SWEEP:
        X = _design(n, p)
        sut = run_fa_record(X, n_factors=2, dataset=f"synth_{n}x{p}",
                            rotation="none", repeats=max(3, repeats // 2), warmup=1)
        if sut.get("error"):
            raise RuntimeError(f"FA speed fit failed at n={n} p={p}: {sut['error']}")
        ref, _raw = run_r_fa_record(X, [f"x{i}" for i in range(p)],
                                    dataset=f"synth_{n}x{p}", n_factors=2,
                                    rotation="none", reps=max(3, repeats // 2))
        records.extend([sut, ref])
        ws, wr = sut["wall_median_s"], ref["wall_median_s"]
        fa_rows.append({
            "analysis": "factor_analysis", "sweep": "n", "n": n, "p": p,
            "rotation": "none",
            "wall_pystat_s": ws, "wall_r_s": wr,
            "speedup_vs_r": (wr / ws) if (ws and wr) else None,
        })
        print(f"  FA   n={n:>7} p={p:>4}  py={ws*1e3:8.2f}ms  R={wr*1e3:8.2f}ms  "
              f"speedup={fa_rows[-1]['speedup_vs_r']:.2f}x")
    rows += fa_rows

    # Empirical complexity exponents over the n-sweeps.
    pca_n = [r for r in rows if r["analysis"] == "pca" and r["sweep"] == "n"]
    complexity = {
        "pca_pystat_exponent_n": _slope([r["n"] for r in pca_n],
                                        [r["wall_pystat_s"] for r in pca_n]),
        "pca_r_exponent_n": _slope([r["n"] for r in pca_n],
                                   [r["wall_r_s"] for r in pca_n]),
        "fa_pystat_exponent_n": _slope([r["n"] for r in fa_rows],
                                       [r["wall_pystat_s"] for r in fa_rows]),
        "fa_r_exponent_n": _slope([r["n"] for r in fa_rows],
                                  [r["wall_r_s"] for r in fa_rows]),
    }
    print("\n  empirical complexity exponents (time ~ n^k):")
    for k, v in complexity.items():
        print(f"    {k:28s} {v:.3f}")

    min_speedup = min(r["speedup_vs_r"] for r in rows if r["speedup_vs_r"])
    print(f"\n  minimum speedup_vs_r across all sizes: {min_speedup:.2f}x "
          f"({'PASS: CPU never lags R' if min_speedup >= 1.0 else 'FAIL: a size lags R'})")

    config = {
        "study": "cpu_speed_vs_r",
        "seed": _SEED,
        "reference": "R stats::prcomp / stats::factanal",
        "blas": {"numpy": "Apple Accelerate (BLAS+LAPACK)",
                 "r_blas": "Apple Accelerate vecLib",
                 "r_lapack": "reference libRlapack"},
        "complexity": complexity,
        "min_speedup_vs_r": min_speedup,
        "pca_n_sweep": _PCA_N_SWEEP, "pca_p_sweep": _PCA_P_SWEEP,
        "fa_n_sweep": _FA_N_SWEEP,
    }
    # Drop the bulky per-fit arrays (scores n x k, rotation p x k) from the frozen
    # records: this is a timing study, the renderer reads the scalar summary CSV,
    # and full scores at n=1e5 would bloat the artifact to ~GB.
    records = _strip_arrays(records)
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "multivariate" / f"v{env['pystatistics_version']}" / "runs"
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
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
