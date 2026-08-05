"""A1+VA-8 survival feature-cluster correctness study (CPU-vs-R).

Generates the frozen evidence for the stratified Cox / stratified KM /
counting-process (start,stop) Cox / left-truncation KM / cox.zph / robust &
cluster-robust SE surfaces added in the feature cluster. Each case fits the
SAME data in pystatistics and in R ``survival`` (via the ``coxfeat`` / ``kmfeat``
worker modes) and reduces the pair to per-quantity agreement rows.

Real data: flchain (``strata(sex)``, from the central HDF5 store, R17). Synthetic
data: deterministic seeded generators for the entry/cluster structure flchain
lacks — materialized to temp CSVs for R only (never committed).

    python -m drivers.survival.generate_feature_cluster --host powerhouse
    python -m drivers.survival.generate_feature_cluster --dry-run   # working tree

CLI: ``--host``, ``--reps``, ``--dry-run`` (skip the PyPI-only guard for a local
smoke of the harness before the release exists on PyPI).
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
from pystatsval.serialize import build_run

# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
# Drivers hardcode their artifact dir, so an ordinary run would otherwise
# silently destroy the artifacts a report was blessed against.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from drivers.survival import agreement, datasets
from drivers.survival.run_pystatistics_feat import (
    run_coxfeat_record, run_kmfeat_record)
from drivers.survival.run_r_survival import run_r_coxfeat, run_r_kmfeat

from artifact_root import artifact_root  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent


def _write_csv(path: Path, header: list[str], columns: dict[str, np.ndarray]) -> None:
    n = len(columns[header[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            w.writerow([columns[c][i] for c in header])


def _timing_row(procedure: str, dataset: str, sut: dict, ref: dict) -> dict:
    ws, wr = sut.get("wall_median_s"), ref.get("wall_median_s")
    return {"procedure": procedure, "dataset": dataset, "n": sut.get("n"),
            "n_events": sut.get("n_events"),
            "wall_pystat_s": ws, "wall_r_s": wr,
            "speedup_vs_r": (wr / ws) if (ws and wr) else None}


# ---- synthetic, deterministic datasets ---------------------------------------

def _synth_counting_process(seed: int = 303, n_subjects: int = 120):
    """Multi-spell (start,stop] data with a covariate that switches mid-follow."""
    rng = np.random.default_rng(seed)
    start, stop, event, x1, x2, cid = [], [], [], [], [], []
    for i in range(n_subjects):
        total = round(float(rng.exponential(12)) + 2, 1)
        xa = round(float(rng.normal()), 5)
        ev = int(rng.binomial(1, 0.75))
        if rng.random() < 0.6 and total > 3:
            cut = round(total * float(rng.uniform(0.25, 0.75)), 1)
            start += [0.0, cut]; stop += [cut, total]; event += [0, ev]
            x1 += [xa, xa]; x2 += [0.0, 1.0]; cid += [i, i]
        else:
            start += [0.0]; stop += [total]; event += [ev]
            x1 += [xa]; x2 += [0.0]; cid += [i]
    return {c: np.asarray(v, float) for c, v in
            {"time": stop, "event": event, "x1": x1, "x2": x2,
             ".start": start, ".cluster": cid}.items()}


def _synth_left_trunc(seed: int = 202, n: int = 150):
    rng = np.random.default_rng(seed)
    stop = np.round(rng.exponential(10, n), 1) + 0.5
    entry = np.round(stop * rng.uniform(0.0, 0.5, n), 1)
    return {"time": stop, "event": rng.binomial(1, 0.7, n).astype(float),
            ".entry": entry,
            ".strata": np.tile(np.array(["A", "B"]), n // 2 + 1)[:n]}


def _synth_robust_cluster(seed: int = 2, n_subjects: int = 100):
    rng = np.random.default_rng(seed)
    time, event, x1, x2, cid = [], [], [], [], []
    for i in range(n_subjects):
        k = 2 if rng.random() < 0.5 else 1
        xb = round(float(rng.normal()), 5)
        for _ in range(k):
            time.append(round(float(rng.exponential(10)), 2))
            event.append(int(rng.binomial(1, 0.7)))
            x1.append(round(float(rng.normal()), 5)); x2.append(xb)
            cid.append(i)
    return {c: np.asarray(v, float) for c, v in
            {"time": time, "event": event, "x1": x1, "x2": x2,
             ".cluster": cid}.items()}


def _flchain_strata():
    """Real flchain: age/kappa/lambda covariates, sex as the stratum."""
    time, event, X, names = datasets.load_flchain(standardize=True)
    sex_idx = names.index("sex")
    strata = X[:, sex_idx].astype(int)
    keep = [j for j in range(X.shape[1]) if names[j] in ("age", "kappa", "lambda")]
    return time, event, X[:, keep], [names[j] for j in keep], strata


def generate(host: str, *, reps: int, dry_run: bool) -> Path:
    env = env_manifest(device="cpu", host=host)
    if not dry_run:
        require_pypi(env)

    records: list[dict] = []
    agreement_rows: list[dict] = []
    timing_rows: list[dict] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # --- real flchain: stratified Cox (efron + breslow = A8 row) + KM + zph
        ft, fe, fX, fnames, fg = _flchain_strata()
        fl_csv = tmp / "flchain.csv"
        _write_csv(fl_csv, ["time", "event", *fnames, ".strata"],
                   {"time": ft, "event": fe, ".strata": fg,
                    **{n: fX[:, j] for j, n in enumerate(fnames)}})
        for ties in ("efron", "breslow"):
            proc = f"coxph_strata_{ties}"
            sut = run_coxfeat_record(ft, fe, fX, dataset="flchain", ties=ties,
                                     strata=fg, reps=reps)
            ref, _ = run_r_coxfeat(str(fl_csv), ties=ties, reps=reps,
                                   dataset="flchain")
            records += [sut, ref]
            agreement_rows += agreement.coxfeat_rows(proc, sut, ref)
            timing_rows.append(_timing_row(proc, "flchain", sut, ref))

        # stratified KM on flchain sex
        flkm_csv = tmp / "flchain_km.csv"
        _write_csv(flkm_csv, ["time", "event", ".strata"],
                   {"time": ft, "event": fe, ".strata": fg})
        sut = run_kmfeat_record(ft, fe, dataset="flchain", strata=fg, reps=reps)
        ref, _ = run_r_kmfeat(str(flkm_csv), reps=reps, dataset="flchain")
        records += [sut, ref]
        agreement_rows += agreement.kmfeat_rows("km_strata", sut, ref)
        timing_rows.append(_timing_row("km_strata", "flchain", sut, ref))

        # cox.zph on the flchain stratified fit
        sut = run_coxfeat_record(ft, fe, fX, dataset="flchain", strata=fg,
                                 zph_transform="km", reps=reps)
        ref, _ = run_r_coxfeat(str(fl_csv), zph_transform="km", reps=reps,
                               dataset="flchain")
        records += [sut, ref]
        agreement_rows += agreement.coxfeat_rows("cox_zph", sut, ref)

        # --- synthetic: counting-process Cox
        cp = _synth_counting_process()
        cp_csv = tmp / "cp.csv"
        _write_csv(cp_csv, ["time", "event", "x1", "x2", ".start"], cp)
        sut = run_coxfeat_record(cp["time"], cp["event"],
                                 np.column_stack([cp["x1"], cp["x2"]]),
                                 dataset="synth_tvc", start=cp[".start"],
                                 zph_transform="km", reps=reps)
        ref, _ = run_r_coxfeat(str(cp_csv), zph_transform="km", reps=reps,
                               dataset="synth_tvc")
        records += [sut, ref]
        agreement_rows += agreement.coxfeat_rows("cox_start_stop", sut, ref)
        timing_rows.append(_timing_row("cox_start_stop", "synth_tvc", sut, ref))

        # --- synthetic: left-truncation KM (+ strata)
        lt = _synth_left_trunc()
        lt_csv = tmp / "lt.csv"
        _write_csv(lt_csv, ["time", "event", ".entry", ".strata"], lt)
        sut = run_kmfeat_record(lt["time"], lt["event"], dataset="synth_lt",
                                entry=lt[".entry"], strata=lt[".strata"],
                                reps=reps)
        ref, _ = run_r_kmfeat(str(lt_csv), reps=reps, dataset="synth_lt")
        records += [sut, ref]
        agreement_rows += agreement.kmfeat_rows("km_left_trunc", sut, ref)

        # --- synthetic: robust + cluster Cox
        rc = _synth_robust_cluster()
        rc_csv = tmp / "rc.csv"
        _write_csv(rc_csv, ["time", "event", "x1", "x2", ".cluster"], rc)
        sut = run_coxfeat_record(rc["time"], rc["event"],
                                 np.column_stack([rc["x1"], rc["x2"]]),
                                 dataset="synth_cluster", cluster=rc[".cluster"],
                                 reps=reps)
        ref, _ = run_r_coxfeat(str(rc_csv), reps=reps, dataset="synth_cluster")
        records += [sut, ref]
        agreement_rows += agreement.coxfeat_rows("cox_cluster_robust", sut, ref)
        timing_rows.append(_timing_row("cox_cluster_robust", "synth_cluster",
                                        sut, ref))

    config = {"study": "feature_cluster_correctness", "reps": reps,
              "dry_run": dry_run}
    run = build_run(env=env, config=config, records=records)
    out_dir = (artifact_root(_REPO) / "survival"
               / f"v{env['pystatistics_version']}" / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"feature_cluster_cpu_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    _write_summary_csv(out_dir / f"{stem}_agreement.csv", agreement_rows,
                       ["procedure", "quantity", "n_elements", "max_abs",
                        "max_rel"])
    _write_summary_csv(out_dir / f"{stem}_timing.csv", timing_rows,
                       ["procedure", "dataset", "n", "n_events",
                        "wall_pystat_s", "wall_r_s", "speedup_vs_r"])
    worst = max((r["max_abs"] for r in agreement_rows), default=0.0)
    print(f"feature-cluster study: {len(agreement_rows)} agreement rows, "
          f"worst max_abs={worst:.2e} -> {run_path}")
    return run_path


def _write_summary_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the PyPI-only guard (local harness smoke)")
    args = ap.parse_args()
    generate(args.host, reps=args.reps, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
