"""Generate the survival correctness-vs-R artifacts (CPU, one host).

One job: fit BOTH pystatistics and the R ``survival`` reference on the canonical
``survival::lung`` designs for all four procedures (KM, log-rank, Cox PH,
discrete-time), reduce each to per-quantity agreement metrics, and freeze a
``validation-run/v1`` artifact plus two flat summary CSVs under
``artifacts/survival/v<ver>/runs/``:

    correctness_cpu_<host>.json          — the full records (py + R), both engines
    correctness_cpu_<host>_agreement.csv — one row per (procedure, quantity)
    correctness_cpu_<host>_timing.csv    — one row per procedure (wall + speedup)

Discrete-time is validated against R ``glm(binomial)`` on the IDENTICAL
person-period design, reconstructed here and guarded: the driver's
``person_period_n`` must equal the library's, or the run fails loud rather than
comparing mismatched designs.

PyPI-only: refuses to write a canonical run unless pystatistics is installed from
PyPI (``require_pypi``). Run it from the dedicated PyPI venv.

    python -m drivers.survival.generate_correctness --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.survival import agreement, datasets
from drivers.survival._person_period import build_person_period
from drivers.survival.run_pystatistics import (
    run_coxph_record, run_discrete_record, run_km_record, run_logrank_record)
from drivers.survival.run_r_survival import (
    run_r_coxph, run_r_discrete, run_r_km, run_r_logrank)

_REPO = Path(__file__).resolve().parent.parent.parent
_DATA = Path(__file__).resolve().parent / "data"


def _timing_row(procedure: str, dataset: str, sut: dict[str, Any],
                ref: dict[str, Any]) -> dict[str, Any]:
    ws, wr = sut.get("wall_median_s"), ref.get("wall_median_s")
    # KM stores a per-interval n_events VECTOR in its summary; the scalar total
    # lives under n_events_total. Cox/discrete store a scalar n_events.
    events = sut.get("n_events_total")
    if events is None:
        events = sut.get("n_events")
    if isinstance(events, list):
        events = int(sum(events))
    return {
        "procedure": procedure,
        "dataset": dataset,
        "n": sut.get("n"),
        "n_events": events,
        "wall_pystat_s": ws,
        "wall_r_s": wr,
        "speedup_vs_r": (wr / ws) if (ws and wr) else None,
    }


def generate(host: str, *, reps: int, n_bins: int) -> Path:
    """Run the full correctness sweep on CPU and freeze the artifacts."""
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    km_csv = _DATA / "lung_km.csv"
    cox_csv = _DATA / "lung_coxph.csv"

    t, e, sex = datasets.load_lung_km()
    tc, ec, X, names = datasets.load_lung_cox()
    bounds = datasets.discrete_interval_bounds(tc, ec, n_bins=n_bins)

    records: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    # --- Kaplan-Meier ---
    km_sut = run_km_record(t, e, reps=reps)
    km_ref, _ = run_r_km(km_csv, reps=reps)
    records += [km_sut, km_ref]
    agreement_rows += agreement.km_rows(km_sut, km_ref)
    timing_rows.append(_timing_row("kaplan_meier", "lung_km", km_sut, km_ref))

    # --- Log-rank by sex ---
    lr_sut = run_logrank_record(t, e, sex, reps=reps)
    lr_ref, _ = run_r_logrank(km_csv, reps=reps)
    records += [lr_sut, lr_ref]
    agreement_rows += agreement.logrank_rows(lr_sut, lr_ref)
    timing_rows.append(_timing_row("survdiff", "lung_km", lr_sut, lr_ref))

    # --- Cox PH ---
    cox_sut = run_coxph_record(tc, ec, X, names, reps=reps)
    cox_ref, _ = run_r_coxph(cox_csv, reps=reps)
    records += [cox_sut, cox_ref]
    agreement_rows += agreement.coxph_rows(cox_sut, cox_ref)
    timing_rows.append(_timing_row("coxph", "lung_coxph", cox_sut, cox_ref))

    # --- Discrete-time (person-period logistic), with the expansion guard ---
    dt_sut = run_discrete_record(tc, ec, X, names, bounds, reps=reps)
    X_pp, y_pp, n_int = build_person_period(tc, ec, X, bounds)
    if len(y_pp) != dt_sut["person_period_n"]:
        raise RuntimeError(
            "person-period reconstruction mismatch: driver built "
            f"{len(y_pp)} rows but pystatistics reports "
            f"{dt_sut['person_period_n']}. The library's discrete-time expansion "
            "changed — update drivers/survival/_person_period.py before trusting "
            "the discrete-time head-to-head.")
    dt_ref, _ = run_r_discrete(X_pp, y_pp, n_intervals=n_int,
                               covariate_names=names, reps=reps)
    records += [dt_sut, dt_ref]
    agreement_rows += agreement.discrete_rows(dt_sut, dt_ref)
    timing_rows.append(_timing_row("discrete_time", "lung_discrete", dt_sut, dt_ref))

    for r in agreement_rows:
        print(f"  {r['procedure']:14s} {r['quantity']:16s} "
              f"max_abs={r['max_abs']:.2e}  max_rel={r['max_rel']:.2e}")

    config = {
        "study": "correctness_vs_r",
        "procedures": ["kaplan_meier", "survdiff", "coxph", "discrete_time"],
        "dataset": "survival::lung (NCCTG advanced lung cancer)",
        "backend": "cpu",
        "repeats": reps,
        "discrete_n_bins": n_bins,
        "discrete_interval_bounds": [float(b) for b in bounds],
        "reference": "R survival::survfit / survdiff / coxph; glm(binomial) for discrete-time",
    }
    run = build_run(env=env, config=config, records=records)

    out_dir = _REPO / "artifacts" / "survival" / f"v{env['pystatistics_version']}" / "runs"
    stem = f"correctness_cpu_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    _write_csv(out_dir / f"{stem}_agreement.csv", agreement_rows,
               ["procedure", "quantity", "n_elements", "max_abs", "max_rel"])
    _write_csv(out_dir / f"{stem}_timing.csv", timing_rows,
               ["procedure", "dataset", "n", "n_events",
                "wall_pystat_s", "wall_r_s", "speedup_vs_r"])
    print(f"\nwrote {run_path}")
    return run_path


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--n-bins", type=int, default=5,
                    help="discrete-time interval count (coarse, well-posed binning)")
    args = ap.parse_args()
    generate(args.host, reps=args.reps, n_bins=args.n_bins)


if __name__ == "__main__":
    main()
