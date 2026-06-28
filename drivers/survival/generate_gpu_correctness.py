"""Generate the discrete-time GPU correctness artifacts on real flchain data.

One job: on the real survival::flchain person-period design, establish that the
discrete-time GPU path is correct to a defensible tolerance, and characterize the
float32 acceptance boundary honestly. Three things are frozen, per host:

1. CPU-vs-R on real data — ``discrete_time`` (CPU, fp64) vs R ``glm(binomial)`` on
   the IDENTICAL flchain person-period design (yearly bins, ~83k rows): the
   coefficient / SE / z / p agreement, extending the lung-only CPU correctness to
   a large real design. Guarded by the ``person_period_n`` reconstruction check.

2. Faithfulness — that ``regression.fit(binomial)`` on the reconstructed design
   reproduces the public ``discrete_time`` covariate coefficients to round-off, so
   the fit-isolated scaling measurement is measuring the real thing.

3. Device cross-precision — for each accelerator backend available on the host
   (MPS fp32; CUDA fp32 and the exact ``gpu_fp64``), the covariate-coefficient
   agreement vs the CPU fp64 fit, swept across bin granularities. Where the
   float32 acceptance gate REJECTS a fit, the RIGOR-R9 forced-fp32 accuracy is
   also recorded — so a rejection is shown to be a genuinely-wrong fit (correct
   fail-loud, A6), not a presumed-broken backend.

PyPI-only (``require_pypi``). Run from the dedicated PyPI venv.

    # Mac: CPU reference + MPS
    python -m drivers.survival.generate_gpu_correctness --host powerhouse --backends gpu
    # Forge: CPU reference + CUDA fp32 + CUDA fp64
    python -m drivers.survival.generate_gpu_correctness --host forge --backends gpu gpu_fp64
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.survival import agreement, datasets
from drivers.survival._person_period import build_person_period
from drivers.survival.run_pystatistics import run_discrete_record
from drivers.survival.run_pystatistics_gpu import (
    fit_pp_glm_record, forced_fp32_accuracy)
from drivers.survival.run_r_survival import run_r_discrete

_REPO = Path(__file__).resolve().parent.parent.parent

# Bin grid for the cross-precision sweep (label, bin width in days). Spans the
# regime where MPS float32 is accurate (coarse, few baseline dummies) into where
# it is not (fine). Days: ~5yr, ~2.5yr, yearly, ~half-year, quarterly.
_GRANULARITIES = [("5yr", 1826.0), ("2_5yr", 913.0), ("yearly", 365.25),
                  ("half_yr", 182.0), ("quarterly", 91.3)]
_FAITHFULNESS_TOL = 1e-9


def _faithfulness(time, event, X, names, bounds, cpu_cov_coef) -> float:
    """Max rel diff between discrete_time covariate coefs and the fit on the
    reconstruction — the guard that the fit-isolated measurement is faithful."""
    from pystatistics.survival import discrete_time
    sol = discrete_time(time, event, X, names=names, intervals=bounds, backend="cpu")
    dt = np.asarray(sol.coefficients, float).ravel()
    cov = np.asarray(cpu_cov_coef, float).ravel()
    return float((np.abs(dt - cov) / np.maximum(np.abs(dt), 1e-12)).max())


def generate(host: str, *, backends: list[str], reps: int, with_r: bool) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    time_, event, X, names = datasets.load_flchain()
    n_cov = len(names)

    records: list[dict[str, Any]] = []
    cpu_r_rows: list[dict[str, Any]] = []     # real-data CPU-vs-R agreement
    device_rows: list[dict[str, Any]] = []    # cross-precision agreement

    # --- 1+2. CPU-vs-R + faithfulness on yearly bins (R-tractable real design) ---
    if with_r:
        bounds_y = datasets.flchain_interval_bounds(time_, event, 365.25)
        dt_sut = run_discrete_record(time_, event, X, names, bounds_y, reps=reps)
        X_pp, y_pp, n_int = build_person_period(time_, event, X, bounds_y)
        if len(y_pp) != dt_sut["person_period_n"]:
            raise RuntimeError(
                f"person-period reconstruction mismatch: {len(y_pp)} vs "
                f"{dt_sut['person_period_n']} — update _person_period.py")
        dt_ref, _ = run_r_discrete(X_pp, y_pp, n_intervals=n_int,
                                   covariate_names=names, reps=reps)
        records += [dt_sut, dt_ref]
        for r in agreement.discrete_rows(dt_sut, dt_ref):
            r["dataset"] = "flchain_yearly"
            r["person_period_n"] = len(y_pp)
            cpu_r_rows.append(r)
        print(f"  CPU-vs-R (flchain yearly, {len(y_pp)} rows): "
              + "  ".join(f"{r['quantity']}={r['max_rel']:.1e}"
                          for r in cpu_r_rows))

    # --- 3. Device cross-precision sweep ---
    for label, bin_days in _GRANULARITIES:
        bounds = datasets.flchain_interval_bounds(time_, event, bin_days)
        X_pp, y_pp, n_int = build_person_period(time_, event, X, bounds)
        cols = X_pp.shape[1]
        dataset = f"flchain_{label}"

        cpu_rec = fit_pp_glm_record(X_pp, y_pp, n_cov=n_cov, backend="cpu",
                                    dataset=dataset, repeats=reps)
        records.append(cpu_rec)
        cpu_full = np.asarray(cpu_rec["all_coefficients"], float)
        cpu_cov = np.asarray(cpu_rec["covariate_coefficients"], float)
        faith = _faithfulness(time_, event, X, names, bounds, cpu_cov)
        if faith > _FAITHFULNESS_TOL:
            raise RuntimeError(
                f"{dataset}: regression.fit on the reconstruction diverges from "
                f"discrete_time by {faith:.2e} (> {_FAITHFULNESS_TOL}) — the "
                "fit-isolated measurement is no longer faithful to the public path")

        for backend in backends:
            acc = fit_pp_glm_record(X_pp, y_pp, n_cov=n_cov, backend=backend,
                                    dataset=dataset, repeats=reps)
            records.append(acc)
            row: dict[str, Any] = {
                "dataset": dataset, "person_period_n": X_pp.shape[0],
                "cols": cols, "n_intervals": n_int, "backend": backend,
                "precision": acc["precision"], "faithfulness_rel": faith}
            if acc.get("error"):
                row["gate_result"] = "rejected"
                row["coef_max_rel"] = None
                # R9: only fp32 ('gpu') has a meaningful forced comparison.
                if backend == "gpu":
                    fr = forced_fp32_accuracy(X_pp, y_pp, cpu_full, backend="gpu")
                    row["forced_max_rel"] = fr["forced_max_rel"]
                    row["forced_max_abs"] = fr["forced_max_abs"]
                row["note"] = acc["error"].split(":", 1)[0]
            else:
                acc_cov = np.asarray(acc["covariate_coefficients"], float)
                rel = float((np.abs(acc_cov - cpu_cov)
                             / np.maximum(np.abs(cpu_cov), 1e-12)).max())
                row["gate_result"] = "accepted"
                row["coef_max_rel"] = rel
                row["forced_max_rel"] = None
            device_rows.append(row)
            tag = row["gate_result"]
            extra = (f"coef_rel={row.get('coef_max_rel'):.1e}"
                     if row.get("coef_max_rel") is not None
                     else f"forced_rel={row.get('forced_max_rel')}")
            print(f"  {dataset:18s} {backend:9s} {tag:9s} {extra}")

    config = {
        "study": "gpu_correctness",
        "dataset": "survival::flchain person-period (discrete-time)",
        "covariates": names, "backends_tested": backends,
        "granularities": [g[0] for g in _GRANULARITIES],
        "repeats": reps, "faithfulness_tol": _FAITHFULNESS_TOL,
        "reference": "CPU fp64 discrete-time fit; R glm(binomial) for CPU-vs-R",
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "survival" / f"v{env['pystatistics_version']}" / "runs"
    stem = f"gpu_correctness_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    if cpu_r_rows:
        _write_csv(out_dir / f"{stem}_cpu_vs_r.csv", cpu_r_rows,
                   ["dataset", "person_period_n", "procedure", "quantity",
                    "n_elements", "max_abs", "max_rel"])
    _write_csv(out_dir / f"{stem}_device.csv", device_rows,
               ["dataset", "person_period_n", "cols", "n_intervals", "backend",
                "precision", "gate_result", "coef_max_rel", "forced_max_rel",
                "faithfulness_rel", "note"])
    print(f"\nwrote {run_path}")
    return run_path


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--backends", nargs="+", default=["gpu"],
                    help="accelerator backends to test (gpu, gpu_fp64)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-r", action="store_true", help="skip the R head-to-head")
    args = ap.parse_args()
    generate(args.host, backends=args.backends, reps=args.reps,
             with_r=not args.no_r)


if __name__ == "__main__":
    main()
