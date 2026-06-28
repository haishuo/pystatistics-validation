"""Generate the discrete-time GPU scaling / device-pivot artifacts.

One job: measure how the ONE GPU-capable survival path (``discrete_time`` →
``regression.fit(family='binomial')`` over the person-period expansion) scales on
a device, and where the GPU wins. Per invocation one device is swept (cpu / mps /
cuda, the last in fp32 or the exact gpu_fp64); the renderer's ``device_pivot``
later pairs the CPU run against each accelerator run by (problem, cols).

What is measured, and why it is split:

- **Fit-isolated device pivot** (real flchain bin sweep + a synthetic person-period
  scale-up grid): times ``regression.fit(binomial, backend=)`` on a PRE-BUILT
  person-period design. The GPU accelerates the FIT, not the (pure-Python,
  device-independent) expansion, so the speedup claim is made on the fit alone.
- **Expansion-vs-fit breakdown** (CPU run only): times the person-period expansion,
  the CPU fit, and the end-to-end ``discrete_time`` call, so the report can show
  the expansion is a small fraction of end-to-end at scale (not the bottleneck).
- **R-choke** (CPU run only): times R ``glm(binomial)`` on the large flchain
  person-period design and records it honestly (slow / may OOM) — the "scale where
  R chokes" evidence.

Deterministic synthetic data; PyPI-only (``require_pypi``).

    # Powerhouse: CPU (carries breakdown + R-choke) then MPS
    python -m drivers.survival.generate_gpu_scaling --host powerhouse --device cpu
    python -m drivers.survival.generate_gpu_scaling --host powerhouse --device mps
    # Forge: CPU then CUDA fp32 then CUDA fp64
    python -m drivers.survival.generate_gpu_scaling --host forge --device cuda --precision fp32
    python -m drivers.survival.generate_gpu_scaling --host forge --device cuda --precision fp64
"""

from __future__ import annotations

import argparse
import csv
import socket
import tempfile
import time as _time
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.survival import datasets
from drivers.survival._person_period import build_person_period
from drivers.survival._pp_synth import make_pp_design
from drivers.survival.run_pystatistics_gpu import (
    discrete_time_e2e_record, fit_pp_glm_record)

_REPO = Path(__file__).resolve().parent.parent.parent

# flchain bin sweep: coarse (where MPS float32 holds) through fine (~1M rows).
_FLCHAIN_BINS = [("5yr", 1826.0), ("2_5yr", 913.0), ("yearly", 365.25),
                 ("quarterly", 91.3), ("monthly", 30.4)]
# Synthetic person-period scale-up: fixed columns, growing rows past flchain.
_SYNTH_INTERVALS = 60
_SYNTH_COV = 8
_SYNTH_N = [130_000, 260_000, 520_000, 1_040_000, 2_080_000]
_SYNTH_SEED = 20260628
# R-choke: a growth sweep showing R glm's wall climbing into the minutes at ~1M
# person-period rows (the model matrix is multi-GB; finer binning pushes the R
# bridge past practical memory). CPU run only.
_RCHOICE_BINS = [("yearly", 365.25), ("quarterly", 91.3), ("monthly", 30.4)]


def _resolve(device: str, precision: str) -> tuple[str, str]:
    """Return (backend, devtag) for the run-file name and engine tag."""
    if device == "cpu":
        return "cpu", "cpu"
    if device == "mps":
        return "gpu", "mps"
    if device == "cuda":
        return ("gpu_fp64", "cuda_fp64") if precision == "fp64" else ("gpu", "cuda")
    raise ValueError(f"unknown device {device!r}")


def _row(rec: dict[str, Any], device: str) -> dict[str, Any]:
    return {"dataset": rec["dataset"], "n": rec.get("n"), "cols": rec.get("p"),
            "engine": rec["engine"], "device": device,
            "precision": rec.get("precision"),
            "wall_median_s": rec.get("wall_median_s"),
            "peak_mem_mb": rec.get("peak_mem_mb"),
            "converged": rec.get("converged"), "error": rec.get("error")}


def generate(host: str, device: str, *, precision: str, reps: int, warmup: int,
             cpu_extras: bool = True) -> Path:
    backend, devtag = _resolve(device, precision)
    env = env_manifest(device=("cpu" if device == "cpu" else device), host=host)
    require_pypi(env)

    time_, event, X, names = datasets.load_flchain()
    n_cov = len(names)
    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    # --- Fit-isolated device pivot: real flchain bin sweep ---
    for label, bin_days in _FLCHAIN_BINS:
        bounds = datasets.flchain_interval_bounds(time_, event, bin_days)
        X_pp, y_pp, _ = build_person_period(time_, event, X, bounds)
        rec = fit_pp_glm_record(X_pp, y_pp, n_cov=n_cov, backend=backend,
                                dataset=f"flchain_{label}", repeats=reps,
                                warmup=warmup)
        records.append(rec)
        rows.append(_row(rec, devtag))
        _log(rec)

    # --- Fit-isolated device pivot: synthetic scale-up (fixed cols) ---
    for n_pp in _SYNTH_N:
        X_pp, y_pp = make_pp_design(n_pp, _SYNTH_INTERVALS, _SYNTH_COV,
                                    seed=_SYNTH_SEED)
        rec = fit_pp_glm_record(X_pp, y_pp, n_cov=_SYNTH_COV, backend=backend,
                                dataset=f"synth_pp_n{n_pp}", repeats=reps,
                                warmup=warmup)
        records.append(rec)
        rows.append(_row(rec, devtag))
        _log(rec)

    config: dict[str, Any] = {
        "study": "gpu_scaling", "device": devtag, "backend": backend,
        "precision": precision if device != "cpu" else "fp64",
        "flchain_bins": [b[0] for b in _FLCHAIN_BINS],
        "synth_intervals": _SYNTH_INTERVALS, "synth_cov": _SYNTH_COV,
        "synth_n": _SYNTH_N, "repeats": reps, "warmup": warmup,
    }
    out_dir = _REPO / "artifacts" / "survival" / f"v{env['pystatistics_version']}" / "runs"
    stem = f"scaling_discrete_{devtag}_{host}"
    run = build_run(env=env, config=config, records=records)
    run_path = write_run(out_dir / f"{stem}.json", run)
    _write_csv(out_dir / f"{stem}_summary.csv", rows,
               ["dataset", "n", "cols", "engine", "device", "precision",
                "wall_median_s", "peak_mem_mb", "converged", "error"])

    # --- CPU-only studies: breakdown + R-choke (powerhouse carries these) ---
    if device == "cpu" and cpu_extras:
        _breakdown(time_, event, X, names, out_dir, host, env, reps)
        _r_choke(time_, event, X, names, out_dir, host, env)

    print(f"\nwrote {run_path}")
    return run_path


def _breakdown(time_, event, X, names, out_dir, host, env, reps) -> None:
    """Expansion vs fit vs end-to-end on CPU, to show expansion is not the cost."""
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    n_cov = len(names)
    for label, bin_days in _FLCHAIN_BINS:
        bounds = datasets.flchain_interval_bounds(time_, event, bin_days)
        t0 = _time.perf_counter()
        X_pp, y_pp, _ = build_person_period(time_, event, X, bounds)
        expansion_s = _time.perf_counter() - t0

        # Single rep: these designs are large and the timings are stable; the
        # breakdown is illustrative (expansion fraction), not the headline number.
        fit_rec = fit_pp_glm_record(X_pp, y_pp, n_cov=n_cov, backend="cpu",
                                    dataset=f"flchain_{label}", repeats=1,
                                    warmup=0)
        e2e_rec = discrete_time_e2e_record(time_, event, X, names, bounds,
                                           backend="cpu",
                                           dataset=f"flchain_{label}", repeats=1,
                                           warmup=0)
        records += [fit_rec, e2e_rec]
        fit_s = fit_rec.get("wall_median_s")
        e2e_s = e2e_rec.get("wall_median_s")
        rows.append({
            "dataset": f"flchain_{label}", "person_period_n": X_pp.shape[0],
            "cols": X_pp.shape[1], "expansion_s": round(expansion_s, 4),
            "fit_cpu_s": fit_s, "e2e_discrete_s": e2e_s,
            "expansion_frac_of_e2e": (round(expansion_s / e2e_s, 4)
                                      if e2e_s else None)})
        print(f"  breakdown {label:10s} exp={expansion_s:.3f}s fit={fit_s:.3f}s "
              f"e2e={e2e_s:.3f}s frac={rows[-1]['expansion_frac_of_e2e']}")
    run = build_run(env=env, config={"study": "expansion_breakdown",
                                     "backend": "cpu"}, records=records)
    write_run(out_dir / f"breakdown_cpu_{host}.json", run)
    _write_csv(out_dir / f"breakdown_cpu_{host}_summary.csv", rows,
               ["dataset", "person_period_n", "cols", "expansion_s", "fit_cpu_s",
                "e2e_discrete_s", "expansion_frac_of_e2e"])


def _r_choke(time_, event, X, names, out_dir, host, env) -> None:
    """Time R glm(binomial) on the large flchain person-period design, honestly."""
    from drivers.survival.run_r_survival import run_r_discrete
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for label, bin_days in _RCHOICE_BINS:
        bounds = datasets.flchain_interval_bounds(time_, event, bin_days)
        X_pp, y_pp, n_int = build_person_period(time_, event, X, bounds)
        try:
            rec, _ = run_r_discrete(X_pp, y_pp, n_intervals=n_int,
                                    covariate_names=names, reps=1)
            records.append(rec)
            wr = rec.get("wall_median_s")
            rows.append({"dataset": f"flchain_{label}",
                         "person_period_n": X_pp.shape[0], "cols": X_pp.shape[1],
                         "engine": "r:glm", "wall_r_s": wr, "status": "ok"})
            print(f"  R-choke {label:10s} rows={X_pp.shape[0]} cols={X_pp.shape[1]} "
                  f"R glm wall={wr:.2f}s")
        except Exception as exc:  # noqa: BLE001 - record OOM / failure honestly
            rows.append({"dataset": f"flchain_{label}",
                         "person_period_n": X_pp.shape[0], "cols": X_pp.shape[1],
                         "engine": "r:glm", "wall_r_s": None,
                         "status": f"{type(exc).__name__}: {str(exc)[:200]}"})
            print(f"  R-choke {label:10s} FAILED: {type(exc).__name__}")
    if records:
        run = build_run(env=env, config={"study": "r_choke", "backend": "r"},
                        records=records)
        write_run(out_dir / f"r_choke_{host}.json", run)
    _write_csv(out_dir / f"r_choke_{host}_summary.csv", rows,
               ["dataset", "person_period_n", "cols", "engine", "wall_r_s",
                "status"])


def _log(rec: dict[str, Any]) -> None:
    tag = rec.get("error") or f"{rec.get('wall_median_s'):.4g}s"
    mem = rec.get("peak_mem_mb")
    print(f"  {rec['dataset']:18s} cols={rec.get('p'):>4} {rec['precision']:6s} "
          f"{tag}" + (f"  mem={mem}MB" if mem else ""))


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
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp64"],
                    help="CUDA precision: fp32 (backend gpu) or fp64 (gpu_fp64)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--no-cpu-extras", action="store_true",
                    help="skip the breakdown + R-choke studies (CPU runs only)")
    args = ap.parse_args()
    generate(args.host, args.device, precision=args.precision, reps=args.reps,
             warmup=args.warmup, cpu_extras=not args.no_cpu_extras)


if __name__ == "__main__":
    main()
