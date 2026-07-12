"""A1+VA-8 survival feature-cluster R10 HARD-CASE study (CPU-vs-R).

Companion to ``generate_feature_cluster`` (the machine-precision surface study).
Where that driver validates the feature-cluster surfaces on realistic data, this
one drives them into the RIGOR R10 adversarial regime and checks pystatistics
still matches ``survival`` (numbers AND behaviour):

  - heavy tied event times (integer times) in STRATIFIED Cox, Efron AND Breslow
  - a singleton / zero-information stratum (a stratum contributes nothing to the
    partial likelihood -- pystatistics must handle it like coxph does)
  - cox.zph with the non-default transforms (rank, identity) -- the surface study
    only exercised the default km transform
  - left-truncation edge risk sets: entry times pressed right up against the
    event times, plus a late-entry subject
  - counting-process (start,stop] Cox with heavily tied stop times

Each case fits the SAME data in pystatistics and in R ``survival`` via the shared
``coxfeat`` / ``kmfeat`` workers and reduces the pair to per-quantity agreement
rows. Deterministic seeded synthetic data (never committed; materialised to temp
CSVs for R only). CPU only; PyPI-only (``require_pypi``).

    python -m drivers.survival.generate_feature_cluster_r10 --host powerhouse
    python -m drivers.survival.generate_feature_cluster_r10 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import socket
import tempfile
from pathlib import Path

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from drivers.survival import agreement
from drivers.survival.run_pystatistics_feat import (
    run_coxfeat_record, run_kmfeat_record)
from drivers.survival.run_r_survival import run_r_coxfeat, run_r_kmfeat

_REPO = Path(__file__).resolve().parent.parent.parent


def _write_csv(path: Path, header: list[str], columns: dict[str, np.ndarray]) -> None:
    n = len(columns[header[0]])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(n):
            w.writerow([columns[c][i] for c in header])


def _write_summary_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


# ---- adversarial, deterministic datasets -------------------------------------

def _heavy_ties_strata(seed: int = 11, n: int = 240):
    """Integer event times (few distinct -> heavy ties) with two strata."""
    rng = np.random.default_rng(seed)
    x1 = np.round(rng.normal(size=n), 5)
    x2 = np.round(rng.normal(size=n), 5)
    time = rng.integers(1, 6, n).astype(float)      # 5 distinct times => heavy ties
    event = rng.binomial(1, 0.75, n).astype(float)
    strata = np.tile(np.array(["g1", "g2"]), n // 2 + 1)[:n]
    return {"time": time, "event": event, "x1": x1, "x2": x2, ".strata": strata}


def _singleton_stratum(seed: int = 12, n_big: int = 160):
    """One informative stratum + a singleton stratum (zero partial-likelihood
    information) + a two-subject stratum. Tests that a stratum contributing no
    comparable risk-set pair is handled like coxph handles it."""
    rng = np.random.default_rng(seed)
    x1 = np.round(rng.normal(size=n_big), 5)
    x2 = np.round(rng.normal(size=n_big), 5)
    time = np.round(rng.exponential(9, n_big) + 0.3, 1)
    event = rng.binomial(1, 0.7, n_big).astype(float)
    strata = np.array(["big"] * n_big, dtype=object)
    # append a singleton (1 subject) and a pair (2 subjects) stratum
    extra_x1 = np.round(rng.normal(size=3), 5)
    extra_x2 = np.round(rng.normal(size=3), 5)
    extra_t = np.array([4.5, 7.2, 3.1])
    extra_e = np.array([1.0, 1.0, 0.0])
    extra_s = np.array(["solo", "pair", "pair"], dtype=object)
    return {"time": np.concatenate([time, extra_t]),
            "event": np.concatenate([event, extra_e]),
            "x1": np.concatenate([x1, extra_x1]),
            "x2": np.concatenate([x2, extra_x2]),
            ".strata": np.concatenate([strata, extra_s])}


def _left_trunc_edge(seed: int = 13, n: int = 180):
    """Entry pressed up against event time + a late-entry subject whose entry
    exceeds most event times (a nearly-empty risk set at its entry)."""
    rng = np.random.default_rng(seed)
    stop = np.round(rng.exponential(10, n) + 2.0, 1)
    entry = np.round(stop - 0.1, 1)                 # entry one grid-tick below event
    event = rng.binomial(1, 0.7, n).astype(float)
    # force one very-late entrant near the max time (still strictly entry < stop)
    mx = float(np.round(np.max(stop), 1))
    stop[0] = mx
    entry[0] = float(np.round(mx - 0.2, 1))
    event[0] = 1.0
    assert np.all(entry < stop), "edge generator must keep entry strictly < time"
    strata = np.tile(np.array(["A", "B"]), n // 2 + 1)[:n]
    return {"time": stop, "event": event, ".entry": entry, ".strata": strata}


def _cp_heavy_ties(seed: int = 14, n_subjects: int = 130):
    """Counting-process (start,stop] with integer stop times (heavy ties across
    spells) and a covariate that switches mid follow-up."""
    rng = np.random.default_rng(seed)
    start, stop, event, x1, x2, cid = [], [], [], [], [], []
    for i in range(n_subjects):
        total = float(rng.integers(3, 8))
        xa = round(float(rng.normal()), 5)
        ev = int(rng.binomial(1, 0.7))
        if rng.random() < 0.6:
            cut = float(rng.integers(1, int(total)))
            start += [0.0, cut]; stop += [cut, total]; event += [0, ev]
            x1 += [xa, xa]; x2 += [0.0, 1.0]; cid += [i, i]
        else:
            start += [0.0]; stop += [total]; event += [ev]
            x1 += [xa]; x2 += [0.0]; cid += [i]
    return {c: np.asarray(v, float) for c, v in
            {"time": stop, "event": event, "x1": x1, "x2": x2,
             ".start": start}.items()}


def generate(host: str, *, reps: int, dry_run: bool) -> Path:
    env = env_manifest(device="cpu", host=host)
    if not dry_run:
        require_pypi(env)

    records: list[dict] = []
    rows: list[dict] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # (1) heavy ties, STRATIFIED, efron + breslow ------------------------
        ht = _heavy_ties_strata()
        ht_csv = tmp / "ht.csv"
        _write_csv(ht_csv, ["time", "event", "x1", "x2", ".strata"], ht)
        for ties in ("efron", "breslow"):
            sut = run_coxfeat_record(
                ht["time"], ht["event"], np.column_stack([ht["x1"], ht["x2"]]),
                dataset="r10_heavy_ties", ties=ties, strata=ht[".strata"],
                reps=reps)
            ref, _ = run_r_coxfeat(str(ht_csv), ties=ties, reps=reps,
                                   dataset="r10_heavy_ties")
            records += [sut, ref]
            rows += agreement.coxfeat_rows(f"r10_heavy_ties_strata_{ties}", sut, ref)

        # (2) singleton / zero-information stratum ---------------------------
        ss = _singleton_stratum()
        ss_csv = tmp / "ss.csv"
        _write_csv(ss_csv, ["time", "event", "x1", "x2", ".strata"], ss)
        sut = run_coxfeat_record(
            ss["time"], ss["event"], np.column_stack([ss["x1"], ss["x2"]]),
            dataset="r10_singleton_stratum", strata=ss[".strata"], reps=reps)
        ref, _ = run_r_coxfeat(str(ss_csv), reps=reps,
                               dataset="r10_singleton_stratum")
        records += [sut, ref]
        rows += agreement.coxfeat_rows("r10_singleton_stratum", sut, ref)

        # (3) cox.zph non-default transforms (rank, identity) ----------------
        # reuse the heavy-ties (unstratified) design to get a stable zph table
        z = _heavy_ties_strata(seed=21)
        z_csv = tmp / "z.csv"
        _write_csv(z_csv, ["time", "event", "x1", "x2"],
                   {k: z[k] for k in ("time", "event", "x1", "x2")})
        for tr in ("rank", "identity"):
            sut = run_coxfeat_record(
                z["time"], z["event"], np.column_stack([z["x1"], z["x2"]]),
                dataset="r10_zph", zph_transform=tr, reps=reps)
            ref, _ = run_r_coxfeat(str(z_csv), zph_transform=tr, reps=reps,
                                   dataset="r10_zph")
            records += [sut, ref]
            rows += agreement.coxfeat_rows(f"r10_zph_{tr}", sut, ref)

        # (4) left-truncation edge risk sets ---------------------------------
        lt = _left_trunc_edge()
        lt_csv = tmp / "lt.csv"
        _write_csv(lt_csv, ["time", "event", ".entry", ".strata"], lt)
        sut = run_kmfeat_record(lt["time"], lt["event"], dataset="r10_lt_edge",
                                entry=lt[".entry"], strata=lt[".strata"],
                                reps=reps)
        ref, _ = run_r_kmfeat(str(lt_csv), reps=reps, dataset="r10_lt_edge")
        records += [sut, ref]
        rows += agreement.kmfeat_rows("r10_km_left_trunc_edge", sut, ref)

        # (5) counting-process with heavy ties -------------------------------
        cp = _cp_heavy_ties()
        cp_csv = tmp / "cp.csv"
        _write_csv(cp_csv, ["time", "event", "x1", "x2", ".start"], cp)
        for ties in ("efron", "breslow"):
            sut = run_coxfeat_record(
                cp["time"], cp["event"], np.column_stack([cp["x1"], cp["x2"]]),
                dataset="r10_cp_heavy_ties", ties=ties, start=cp[".start"],
                reps=reps)
            ref, _ = run_r_coxfeat(str(cp_csv), ties=ties, reps=reps,
                                   dataset="r10_cp_heavy_ties")
            records += [sut, ref]
            rows += agreement.coxfeat_rows(f"r10_cp_heavy_ties_{ties}", sut, ref)

    config = {"study": "feature_cluster_r10_hard_cases", "reps": reps,
              "dry_run": dry_run}
    run = build_run(env=env, config=config, records=records)
    out_dir = (_REPO / "artifacts" / "survival"
               / f"v{env['pystatistics_version']}" / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"feature_cluster_r10_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    _write_summary_csv(out_dir / f"{stem}_agreement.csv", rows,
                       ["procedure", "quantity", "n_elements", "max_abs",
                        "max_rel"])
    worst = max((r["max_abs"] for r in rows), default=0.0)
    print(f"feature-cluster R10 study: {len(rows)} agreement rows, "
          f"worst max_abs={worst:.2e} -> {run_path}")
    return run_path


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
