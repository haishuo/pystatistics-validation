"""Generate the Cox PH CPU-scaling artifacts (the performance / optimization pass).

One job: survival is correctness-dominant and, per the PyStatistics Constitution,
Cox PH is NOT a GPU target (the partial likelihood is inherently sequential). So
the honest performance story is CPU-side. This sweep fits ``coxph`` on a
synthetic proportional-hazards DGP at growing n and records pystatistics wall,
R ``coxph`` wall, and their ratio.

Why the ratio matters: pystatistics' current Cox implementation has two
quadratic CPU hotspots —

  O1: Harrell's concordance is an O(n^2) nested Python loop (R uses an
      O(n log n) balanced-tree count);
  O2: the partial-likelihood / score / information loops recompute the risk
      mask ``time >= t_j`` and re-slice the design for every unique event time,
      every Newton iteration — roughly O(n_events * n).

Both are O(n^2). R's coxph is ~O(n log n). So pystatistics/R wall should be flat
for small n (Python dispatch overhead dominates) and GROW with n as the quadratic
terms take over — direct, public-API evidence of the asymptotic gap, with no
library instrumentation. This study measures the symptom; the optimization
targets O1/O2 are named in the subsystem prose.

Deterministic DGP (seeded); CPU only. PyPI-only (``require_pypi``).

    python -m drivers.survival.generate_scaling --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.measure import measure
from pystatsval.record import make_record
from pystatsval.serialize import build_run, write_run

from drivers.survival.run_r_survival import _run_worker, _wall

_REPO = Path(__file__).resolve().parent.parent.parent

_N_GRID = [250, 500, 1000, 2000, 4000, 8000]
_BETA = np.array([0.5, -0.3, 0.2])          # true log-hazard ratios
_SEED = 20260627
_CENSOR_RATE = 0.45                          # tuned for ~30% censoring


def _simulate(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic proportional-hazards sample → (time, event, X).

    Exponential baseline hazard with covariate effects (Bender et al. 2005
    inversion): T = -log(U) / (lambda0 * exp(X beta)); independent exponential
    censoring. Continuous times (no manufactured ties).
    """
    rng = np.random.default_rng(seed)
    p = len(_BETA)
    X = rng.standard_normal((n, p))
    eta = X @ _BETA
    u = rng.uniform(size=n)
    t_event = -np.log(u) / np.exp(eta)              # lambda0 = 1
    t_cens = rng.exponential(scale=1.0 / _CENSOR_RATE, size=n)
    time = np.minimum(t_event, t_cens)
    event = (t_event <= t_cens).astype(np.float64)
    return time, event, X


def _run_pystat(time, event, X, *, reps, warmup) -> dict[str, Any]:
    from pystatistics.survival import coxph

    def _call():
        return coxph(time, event, X)

    wall, sol = measure(_call, device="cpu", repeats=reps, warmup=warmup)
    return make_record(
        engine="pystatistics:cpu", dataset=f"sim_n{len(time)}", n=len(time),
        p=X.shape[1], backend_name="cpu_cox", precision="fp64",
        n_iter=int(sol.n_iter), converged=bool(sol.converged), wall=wall,
        extra={"procedure": "coxph", "n_events": int(sol.n_events)})


def _run_r(time, event, X, *, reps) -> dict[str, Any]:
    p = X.shape[1]
    cols = ["time", "event"] + [f"x{j}" for j in range(p)]
    data = np.column_stack([time, event, X])
    tmp = Path(tempfile.mkdtemp(prefix="rcoxgrid_")) / "cox.csv"
    np.savetxt(tmp, data, delimiter=",", header=",".join(cols), comments="",
               fmt="%.17g")
    raw = _run_worker("coxfit", [str(tmp)], reps)
    return make_record(
        engine="r:coxph", dataset=f"sim_n{len(time)}", n=int(raw["n"]),
        p=p, backend_name="r_cox", precision="fp64", wall=_wall(raw),
        extra={"procedure": "coxph", "n_events": int(raw["n_events"])})


def generate(host: str, *, reps: int, warmup: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for n in _N_GRID:
        time, event, X = _simulate(n, _SEED)
        sut = _run_pystat(time, event, X, reps=reps, warmup=warmup)
        ref = _run_r(time, event, X, reps=reps)
        records += [sut, ref]
        ws, wr = sut["wall_median_s"], ref["wall_median_s"]
        ratio = (ws / wr) if (ws and wr) else None
        rows.append({
            "n": n, "p": X.shape[1], "n_events": sut.get("n_events"),
            "n_iter": sut.get("n_iter"),
            "wall_pystat_s": ws, "wall_r_s": wr,
            "ratio_pystat_over_r": ratio,
        })
        ratio_str = f"{ratio:.1f}x" if ratio else "n/a"
        print(f"  n={n:6d}  events={sut.get('n_events'):5d}  "
              f"pystat={ws:.4f}s  R={wr:.4f}s  ratio={ratio_str}")

    config = {
        "study": "coxph_cpu_scaling",
        "dataset": "synthetic proportional-hazards DGP (exponential baseline)",
        "n_grid": _N_GRID, "true_beta": _BETA.tolist(), "seed": _SEED,
        "censor_rate": _CENSOR_RATE, "backend": "cpu", "repeats": reps,
        "reference": "R survival::coxph (ties='efron')",
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "survival" / f"v{env['pystatistics_version']}" / "runs"
    stem = f"scaling_cpu_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    cols = ["n", "p", "n_events", "n_iter", "wall_pystat_s", "wall_r_s",
            "ratio_pystat_over_r"]
    csv_path = out_dir / f"{stem}_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {run_path}\nwrote {csv_path}")
    return run_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    generate(args.host, reps=args.reps, warmup=args.warmup)


if __name__ == "__main__":
    main()
