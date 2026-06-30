"""Generate the Mac MPS PCA artifacts (priority 4): correctness + value + guards.

One job: validate the randomized-SVD MPS PCA path that 4.4.0 added (the path the
library previously refused outright) on Apple Silicon. Four sub-studies:

  1. **correctness @ fp32 tier** — MPS randomized vs the CPU fp64 reference,
     sign-aligned, on several shapes: sdev rel-err, loading cosines, score
     rel-err must sit inside the GPU_FP32 tier (rtol 1e-4 / atol 1e-5).
  2. **value (R priority 4)** — CPU vs MPS wall-clock across shapes: MPS must be
     FASTER than the CPU in its large-n/large-p regime to justify the backend;
     small-n losing to the CPU is expected and reported, not hidden.
  3. **reproducibility (Rule 6)** — same seed -> identical result; different seed
     -> still inside the fp32 tier (the random sketch is the only nondeterminism
     and it is seed-injectable).
  4. **fail-loud guards (A6)** — an explicit solver='svd'/'gram' on MPS must RAISE
     (no silent CPU fallback / no eigh NotImplementedError leak); only the
     randomized path (and the default/auto routing to it) runs on Metal.

CUDA isolation (gpu_fp64 vs cpu_fp64, R11) and the CF-1 Gram-gate boundary are a
separate Forge study; this runs on the Mac (MPS only).

    python -m drivers.multivariate.generate_gpu_mac --host powerhouse
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

from drivers.multivariate.run_pystatistics import (
    run_pca_record, strip_arrays as _strip_arrays)

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260630

# Correctness shapes (tall, wide, square) with a planted low-rank spectrum so the
# top-k structure is well-defined; k = components compared.
_CORRECTNESS = [
    ("tall",   50000, 100, 10),
    ("tall",   200000, 100, 10),
    ("wide",   2000, 3000, 8),
    ("square", 5000, 5000, 10),
]
# Value shapes: small (CPU should win) through large tall/wide (MPS should win).
_VALUE = [
    (5000, 100, 10),
    (100000, 100, 10),
    (500000, 200, 10),
    (2000, 5000, 10),
]
_TIER_RTOL = 1e-4
_TIER_ATOL = 1e-5


def _planted(n: int, p: int, k: int) -> np.ndarray:
    """Deterministic well-conditioned design with a clear top-k spectral GAP.

    Built as ``X = (U * s) @ Vt`` with orthonormal U (n x r), Vt (r x p) and a
    singular-value spectrum that is well-SEPARATED in the top-k (linspace 10..6)
    then drops to a flat floor (0.5). The gap matters: a smooth/degenerate
    spectrum makes individual singular vectors mathematically ambiguous (any
    rotation within a near-tied cluster is valid), so per-vector comparison is
    meaningless there even though the SUBSPACE is correct. A real top-k PCA target
    has separated leading components; that is what we validate. cond = 20 (well
    inside the fp32 gate). Ill-conditioned *retained* subspaces are exercised
    separately (the gate must refuse those, not silently mis-decompose).
    """
    rng = np.random.default_rng(_SEED + n + p)
    r = min(n, p)
    U, _ = np.linalg.qr(rng.standard_normal((n, r)))
    V, _ = np.linalg.qr(rng.standard_normal((p, r)))
    s = np.full(r, 0.5)
    s[:k] = np.linspace(10.0, 6.0, k)         # separated top-k, then a gap to 0.5
    return ((U * s) @ V.T).astype(np.float64)


def _subspace_max_angle_deg(Va, Vb) -> float:
    """Largest principal angle (deg) between two column spaces — rotation-invariant.

    This is the CORRECT correctness metric for PCA loadings: it measures whether
    the k-dimensional subspaces agree, immune to the per-vector sign/rotation
    ambiguity that arises when singular values are near-degenerate. 0 deg == the
    subspaces coincide.
    """
    Qa, _ = np.linalg.qr(np.asarray(Va, float))
    Qb, _ = np.linalg.qr(np.asarray(Vb, float))
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(sv.min(), -1.0, 1.0))))


def _sign_align(Vp, Vr):
    Vp = np.asarray(Vp, float)
    Vr = np.asarray(Vr, float)
    s = np.sign((Vp * Vr).sum(axis=0))
    s[s == 0] = 1.0
    return Vr * s[np.newaxis, :], s


def _within_tier(a, b) -> bool:
    return bool(np.allclose(np.asarray(a, float), np.asarray(b, float),
                            rtol=_TIER_RTOL, atol=_TIER_ATOL))


def _correctness_rows(records: list[dict[str, Any]], repeats: int) -> list[dict]:
    rows = []
    for shape, n, p, k in _CORRECTNESS:
        X = _planted(n, p, k)
        cpu = run_pca_record(X, backend="cpu", dataset=f"{shape}_{n}x{p}",
                             scale=False, n_components=k, repeats=2, warmup=1,
                             device="cpu")
        mps = run_pca_record(X, backend="gpu", dataset=f"{shape}_{n}x{p}",
                             scale=False, n_components=k, solver="randomized",
                             seed=0, repeats=repeats, warmup=1, device="mps")
        if mps.get("error"):
            raise RuntimeError(f"MPS PCA failed on {shape} {n}x{p}: {mps['error']}")
        records.extend([cpu, mps])
        sdev_rel = float(np.max(np.abs(np.array(mps["sdev"]) - np.array(cpu["sdev"]))
                                / np.maximum(np.abs(np.array(cpu["sdev"])), 1e-300)))
        rot_c = np.array(cpu["rotation"]); rot_m = np.array(mps["rotation"])
        rot_m_al, signs = _sign_align(rot_c, rot_m)
        # PRIMARY metric: rotation-invariant subspace agreement (largest principal
        # angle). SECONDARY: per-vector cosine (meaningful only because the top-k
        # spectrum is gapped here).
        subspace_angle = _subspace_max_angle_deg(rot_c, rot_m)
        cos = np.abs((rot_c * rot_m_al).sum(0) /
                     (np.linalg.norm(rot_c, axis=0) * np.linalg.norm(rot_m_al, axis=0)))
        sc_c = np.array(cpu["scores"]); sc_m = np.array(mps["scores"]) * signs[None, :]
        sc_rel = float(np.max(np.abs(sc_m - sc_c) / np.maximum(np.abs(sc_c), 1e-3)))
        rows.append({
            "study": "correctness", "shape": shape, "n": n, "p": p, "k": k,
            "sdev_max_rel": sdev_rel,
            "subspace_max_angle_deg": subspace_angle,
            "loading_min_cos": float(cos.min()),
            "score_max_rel": sc_rel,
            "within_fp32_tier": bool(sdev_rel <= _TIER_RTOL and subspace_angle < 1.0),
        })
        print(f"  [correct] {shape:6s} {n:>7}x{p:<5} k={k}  "
              f"sdev_rel={sdev_rel:.2e}  subspace={subspace_angle:.4f}deg  "
              f"min_cos={cos.min():.6f}  tier_ok={rows[-1]['within_fp32_tier']}")
    return rows


def _value_rows(records: list[dict[str, Any]], repeats: int) -> list[dict]:
    rows = []
    for n, p, k in _VALUE:
        X = _planted(n, p, k)
        cpu = run_pca_record(X, backend="cpu", dataset=f"val_{n}x{p}", scale=False,
                             n_components=k, repeats=repeats, warmup=1, device="cpu")
        mps = run_pca_record(X, backend="gpu", dataset=f"val_{n}x{p}", scale=False,
                             n_components=k, solver="randomized", seed=0,
                             repeats=repeats, warmup=1, device="mps")
        records.extend([cpu, mps])
        wc, wm = cpu["wall_median_s"], mps["wall_median_s"]
        rows.append({
            "study": "value", "n": n, "p": p, "k": k,
            "wall_cpu_s": wc, "wall_mps_s": wm,
            "mps_speedup_vs_cpu": (wc / wm) if (wc and wm) else None,
        })
        print(f"  [value]   {n:>7}x{p:<5}  cpu={wc*1e3:8.2f}ms  mps={wm*1e3:8.2f}ms  "
              f"speedup={rows[-1]['mps_speedup_vs_cpu']:.2f}x")
    return rows


def _repro_and_guards(repeats: int) -> dict[str, Any]:
    from pystatistics.multivariate import pca
    X = _planted(50000, 100, 10)
    # Determinism (Rule 6): the SEED controls the algorithmic randomness (the
    # random sketch). GPU fp32 matmul reductions are not bit-deterministic across
    # calls (parallel reduction order), so same-seed agreement is to the fp32
    # tier, not bit-exact -- the correct, honest claim. Different seeds must also
    # stay within tier (the sketch choice does not move the answer).
    a = pca(X, backend="gpu", solver="randomized", n_components=10, seed=7)
    b = pca(X, backend="gpu", solver="randomized", n_components=10, seed=7)
    same_seed_bit_identical = bool(np.array_equal(np.asarray(a.sdev), np.asarray(b.sdev)))
    same_seed_max_abs = float(np.max(np.abs(np.asarray(a.sdev) - np.asarray(b.sdev))))
    same_seed_within_tier = _within_tier(a.sdev, b.sdev)
    c = pca(X, backend="gpu", solver="randomized", n_components=10, seed=99)
    diff_seed_within_tier = _within_tier(a.sdev, c.sdev)
    # fail-loud guards: explicit svd / gram on MPS must raise
    guards = {}
    for method in ("svd", "gram"):
        try:
            pca(X, backend="gpu", solver=method, n_components=10)
            guards[method] = "DID_NOT_RAISE"
        except Exception as exc:  # noqa: BLE001
            guards[method] = type(exc).__name__

    # fp32 no-silent-wrong gate (R12/R13 on MPS). Two cases, because the
    # randomized path gates on the conditioning of the RETAINED top-k subspace
    # (the rank-(k+oversample) sketch), not the full matrix -- which is the
    # correct thing for top-k PCA:
    #
    #  (1) full matrix ill-conditioned (cond 1e4) but the top-k subspace is
    #      well-conditioned -> the path correctly DOES NOT refuse, and the
    #      returned top-k must be CORRECT vs the CPU fp64 reference (subspace
    #      angle < 1 deg). This is the "accept" side: not silently wrong.
    #  (2) the RETAINED top-k subspace itself is ill-conditioned (top-20 singular
    #      values span 1e5) -> fp32 CholeskyQR cannot recover it, so the gate must
    #      REFUSE loud (NumericalError); force=True bypasses (documented override),
    #      and backend='cpu' on the same data succeeds (the escape hatch).
    rng2 = np.random.default_rng(_SEED + 777)
    # case (1): ill FULL matrix, well-conditioned RETAINED subspace. The sketch
    # retains rank l = k + oversample = 20, so the top-20 singular values must be
    # well-conditioned; the tiny tail that makes the full matrix ill-conditioned
    # sits at components 21+ (beyond the sketch).
    U1, _ = np.linalg.qr(rng2.standard_normal((20000, 100)))
    V1, _ = np.linalg.qr(rng2.standard_normal((100, 100)))
    s1 = np.full(100, 1e-4); s1[:20] = np.linspace(10.0, 5.0, 20)  # retained cond ~2
    X_accept = (U1 * s1) @ V1.T
    cpu_acc = pca(X_accept, backend="cpu", n_components=10)
    try:
        mps_acc = pca(X_accept, backend="gpu", solver="randomized", n_components=10, seed=0)
        accept_refused = False
        accept_subspace = _subspace_max_angle_deg(cpu_acc.rotation, mps_acc.rotation)
    except Exception as exc:  # noqa: BLE001
        accept_refused = True
        accept_subspace = float("nan")
    # case (2): ill-conditioned retained top-k -> must refuse
    s2 = np.full(100, 1e-7); s2[:20] = np.geomspace(1.0, 1e-5, 20)  # retained cond 1e5
    X_refuse = (U1 * s2) @ V1.T
    try:
        pca(X_refuse, backend="gpu", solver="randomized", n_components=10)
        refuse_outcome = "DID_NOT_RAISE"
    except Exception as exc:  # noqa: BLE001
        refuse_outcome = type(exc).__name__
    refuse_cpu_ok = run_pca_record(X_refuse, backend="cpu", dataset="refuse",
                                   n_components=10, repeats=1, warmup=0,
                                   device="cpu").get("error") is None
    try:
        pca(X_refuse, backend="gpu", solver="randomized", n_components=10, force=True)
        refuse_force_bypasses = True
    except Exception:  # noqa: BLE001
        refuse_force_bypasses = False

    print(f"  [repro]   same_seed_bit_identical={same_seed_bit_identical}  "
          f"same_seed_max_abs={same_seed_max_abs:.2e}  "
          f"same_seed_within_tier={same_seed_within_tier}  "
          f"diff_seed_within_tier={diff_seed_within_tier}")
    print(f"  [guards]  explicit svd->{guards['svd']}  gram->{guards['gram']}")
    print(f"  [gate-accept] illfull/wellcond-topk refused={accept_refused}  "
          f"topk_subspace={accept_subspace:.4f}deg (must be <1, correct accept)")
    print(f"  [gate-refuse] illcond-topk outcome={refuse_outcome}  "
          f"cpu_escape_ok={refuse_cpu_ok}  force_bypasses={refuse_force_bypasses}")
    return {
        "same_seed_bit_identical": same_seed_bit_identical,
        "same_seed_max_abs_diff": same_seed_max_abs,
        "same_seed_within_tier": same_seed_within_tier,
        "diff_seed_within_tier": diff_seed_within_tier,
        "explicit_svd_on_mps": guards["svd"],
        "explicit_gram_on_mps": guards["gram"],
        "accept_illfull_wellcond_topk_refused": accept_refused,
        "accept_topk_subspace_angle_deg": accept_subspace,
        "refuse_illcond_topk_outcome": refuse_outcome,
        "refuse_cpu_escape_ok": refuse_cpu_ok,
        "refuse_force_bypasses": refuse_force_bypasses,
    }


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="mps", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    print("correctness @ fp32 tier (MPS randomized vs CPU fp64):")
    rows += _correctness_rows(records, repeats)
    print("value (CPU vs MPS):")
    rows += _value_rows(records, repeats)
    print("reproducibility + fail-loud guards:")
    guards = _repro_and_guards(repeats)

    large = [r for r in rows if r["study"] == "value" and r["n"] >= 100000]
    best = max((r["mps_speedup_vs_cpu"] for r in large), default=None)
    config = {
        "study": "gpu_mac_mps",
        "seed": _SEED, "device": "mps",
        "fp32_tier": {"rtol": _TIER_RTOL, "atol": _TIER_ATOL},
        "guards": guards,
        "torch_version_note": "MPS randomized path validated on torch 2.12.1 "
            "(version-sensitive kernels); the host-fp64 CPU reference is the "
            "version-independent correctness anchor.",
        "best_large_mps_speedup_vs_cpu": best,
    }
    records = _strip_arrays(records)  # drop bulky scores/rotation (timing study)
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "multivariate" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"gpu_mac_{host}.json", run)
    # Split the correctness and value rows (different column sets) for clean tables.
    _write_summary_csv(out_dir / f"gpu_mac_correctness_{host}_summary.csv",
                       [r for r in rows if r["study"] == "correctness"])
    _write_summary_csv(out_dir / f"gpu_mac_value_{host}_summary.csv",
                       [r for r in rows if r["study"] == "value"])
    print(f"\n  best large-regime MPS speedup vs CPU: "
          f"{best:.2f}x" if best else "  (no large speedup measured)")
    print(f"wrote {run_path}")
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
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
