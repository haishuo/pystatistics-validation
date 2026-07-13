"""CPU-vs-R speed across problem sizes — the first-class CPU-speed deliverable.

RIGOR priority 3 (and R1): the CPU path is the R-equivalent path and **must never
lag R**. A single size hides the slope, so this sweeps wall-clock for pystatistics
CPU vs R across `n` from small (where fixed Python dispatch overhead bites) to
large, for OLS and EVERY GLM family, and reports per-point speedup plus the
empirical complexity slope of each engine. Any size where pystatistics lags R is a
finding that must be explained (what is the user buying?) or flagged as a defect.

Fairness: both sides are timed as the full user-facing call — pystatistics
``fit(X, y, family=...)`` (which builds its design internally) and R's
``lm``/``glm``/``glm.nb`` from a formula (which builds the model matrix). The data
frame / arrays are prepared once, outside the timed region, on both sides. Small-n
fits are averaged over a loop (R) / many repeats (pystatistics) so sub-millisecond
times are resolved rather than lost to timer granularity.

Binding host: Powerhouse, whose R links Apple Accelerate (a fast vendor BLAS) — the
honest "must never lag R" test. (Forge's R links the reference netlib BLAS, which
would flatter pystatistics.)

    python -m drivers.regression.generate_cpu_speed --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.measure import measure
from pystatsval.serialize import build_run, write_run

_REPO = Path(__file__).resolve().parent.parent.parent
_R_WORKER = Path(__file__).resolve().parent / "_r" / "cpu_speed_run.R"

# (model_key, pystatistics family arg, R family tag, max n for this family)
_MODELS = (
    ("ols",         None,                 "lm",       500_000),
    ("glm_binomial", "binomial",          "binomial", 500_000),
    ("glm_poisson",  "poisson",           "poisson",  500_000),
    ("glm_gamma",    "gamma",             "Gamma",    500_000),
    ("glm_negbin",   "negative-binomial", "negbin",    10_000),  # glm.nb profiling is slow
)
_N_GRID = (50, 200, 1_000, 10_000, 100_000, 500_000)
_P = 8


def _reps(n: int) -> int:
    """More repeats at small n so sub-millisecond fits are resolved."""
    if n <= 200:
        return 300
    if n <= 1_000:
        return 100
    if n <= 10_000:
        return 30
    if n <= 100_000:
        return 8
    return 4


def _fam(family: str | None):
    from pystatistics.regression import Gamma
    if family == "gamma":
        return Gamma(link="log")
    return family


def _synth(model_key: str, family: str | None, n: int, *, seed: int):
    """Deterministic design (intercept + p predictors) and a family-appropriate y."""
    rng = np.random.default_rng(seed)
    Xp = rng.standard_normal((n, _P))
    X = np.column_stack([np.ones(n), Xp])
    beta = rng.standard_normal(_P + 1) * 0.3 / np.sqrt(_P)
    eta = np.clip(X @ beta, -6, 6)
    if family is None:
        y = X @ (rng.standard_normal(_P + 1) * 0.5) + rng.standard_normal(n)
    elif family == "binomial":
        y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
    elif family == "poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    elif family == "gamma":
        y = rng.gamma(shape=2.0, scale=np.exp(eta) / 2.0) + 1e-6
    else:  # negative binomial — overdispersed counts
        mu = np.exp(eta)
        theta = 2.0
        y = rng.negative_binomial(theta, theta / (theta + mu)).astype(np.float64)
    return X, y


def _time_pystat(X, y, family, *, repeats):
    from pystatistics.regression import fit

    def _call():
        return fit(X, y, family=_fam(family), backend="cpu")

    wall, _ = measure(_call, device="cpu", repeats=repeats, warmup=2)
    return float(wall["median_s"])


def _time_r(X, y, r_family, reps) -> float:
    tmp = Path(tempfile.mkdtemp(prefix="rspeed_"))
    xc, yc, oj = tmp / "x.csv", tmp / "y.csv", tmp / "o.json"
    names = [f"x{j}" for j in range(X.shape[1] - 1)]
    np.savetxt(xc, X[:, 1:], delimiter=",", header=",".join(names), comments="", fmt="%.17g")
    np.savetxt(yc, y, fmt="%.17g")
    p = subprocess.run(["Rscript", str(_R_WORKER), str(xc), str(yc), r_family,
                        str(reps), str(oj)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"R speed worker failed ({r_family}, n={len(y)}):\n{p.stderr[-1500:]}")
    return float(json.loads(oj.read_text())["elapsed_per_fit_s"])


def _slope(ns: list[int], walls: list[float]) -> float | None:
    """Empirical complexity exponent: slope of log(wall) vs log(n) for n>=1000
    (above the fixed-overhead floor). None if too few points."""
    pts = [(n, w) for n, w in zip(ns, walls) if n >= 1000 and w and w > 0]
    if len(pts) < 2:
        return None
    lx = np.log10([n for n, _ in pts])
    ly = np.log10([w for _, w in pts])
    return float(np.polyfit(lx, ly, 1)[0])


def generate(host: str) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    slope_rows: list[dict[str, Any]] = []
    seed = 0
    for model_key, family, r_family, n_max in _MODELS:
        ns_seen, py_walls, r_walls = [], [], []
        for n in _N_GRID:
            if n > n_max:
                continue
            seed += 1
            X, y = _synth(model_key, family, n, seed=seed)
            reps = _reps(n)
            py = _time_pystat(X, y, family, repeats=reps)
            r = _time_r(X, y, r_family, reps)
            speedup = (r / py) if (py and r) else None
            row = {
                "model_key": model_key, "family": (family or "gaussian"),
                "n": n, "p": _P, "reps": reps,
                "wall_pystat_s": py, "wall_r_s": r,
                "speedup_vs_r": speedup,
                "pystat_lags_r": (bool(speedup is not None and speedup < 1.0)),
            }
            rows.append(row)
            records.append({"engine": "cpu_speed", "dataset": model_key, **row})
            ns_seen.append(n); py_walls.append(py); r_walls.append(r)
            tag = "LAGS R" if row["pystat_lags_r"] else "ok"
            print(f"  {model_key:13} n={n:>7} reps={reps:>4}: "
                  f"pystat={py:.3e}s R={r:.3e}s  speedup={speedup:.2f}x [{tag}]", flush=True)
        slope_rows.append({
            "model_key": model_key, "family": (family or "gaussian"),
            "slope_pystat": _slope(ns_seen, py_walls),
            "slope_r": _slope(ns_seen, r_walls),
            "n_points_for_slope": sum(1 for n in ns_seen if n >= 1000),
        })

    config = {"study": "cpu_speed_vs_r", "n_grid": list(_N_GRID), "p": _P,
              "models": [m for m, *_ in _MODELS], "reference": "R lm()/glm()/glm.nb()",
              "host_blas": "powerhouse=Apple Accelerate (fast vendor BLAS)"}
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = write_run(out_dir / f"cpu_speed_{host}.json", run)
    _csv(out_dir / f"cpu_speed_{host}_summary.csv", rows,
         ["model_key", "family", "n", "p", "reps", "wall_pystat_s", "wall_r_s",
          "speedup_vs_r", "pystat_lags_r"])
    _csv(out_dir / f"cpu_speed_slopes_{host}_summary.csv", slope_rows,
         ["model_key", "family", "slope_pystat", "slope_r", "n_points_for_slope"])
    print(f"\nwrote {run_path}")
    lags = [r for r in rows if r["pystat_lags_r"]]
    print(f"points where pystatistics CPU LAGS R: {len(lags)}")
    for r in lags:
        print(f"  {r['model_key']} n={r['n']}: speedup={r['speedup_vs_r']:.2f}x")
    return run_path


def _csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    args = ap.parse_args()
    generate(args.host)


if __name__ == "__main__":
    main()
