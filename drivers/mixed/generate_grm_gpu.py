"""Generate the GRM GPU artifacts: CF-1 no-silent-wrong gate, R11 isolation, and
the GPU-earns-its-keep speed study (RIGOR priority 4).

The grm_lmm GPU path forms an fp32 Gram (K = W Wᵀ / M) — exactly the CF-1 exposure.
This study proves the priority-4 bar for the ONE GPU-warranted model in `mixed`:

  1. **CF-1 no-silent-wrong gate (R12/R13):** sweep cond(W) across the fp32
     boundary. Every ACCEPTED fp32 fit must be correct vs the CPU fp64 optimum
     (h²/β to the fp32 tier); every design past the fp32-safe boundary must REFUSE
     LOUD (core.exceptions.NumericalError), with force=True the documented override.
     Assert silent_wrong_count == 0 (accept ⟹ correct, refuse ⟹ loud).
  2. **R11 isolation (CUDA):** gpu_fp64 vs cpu_fp64 — same precision, isolates the
     hardware effect from the fp32 effect.
  3. **Earns-its-keep:** cpu vs gpu(fp32) vs gpu_fp64 wall time at large n — the GPU
     path must beat the CPU in its regime or it has no reason to exist.

Device is chosen by --device (mps on Apple Silicon, cuda on Forge). Per the GPU
investigation, MPS is a correctness/portability path (no speedup); the speed story
is CUDA. PyPI-only (require_pypi). Needs torch.

    python -m drivers.mixed.generate_grm_gpu --host powerhouse --device mps
    python -m drivers.mixed.generate_grm_gpu --host forge --device cuda
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

from drivers.mixed.grm_datasets import make_grm, make_grm_conditioned
from drivers.mixed.run_grm import run_grm_record

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260630
# fp32-tier statistical-equivalence bound for accepted GPU fits.
_FP32_TIER = 5e-3          # |Δh²| / |Δβ| accepted-fit bound (generous fp32 tier)
_COND_SWEEP = [1e2, 1e3, 3e3, 1e4, 3e4, 1e5]   # straddles the fp32-safe boundary


def _rel(a, b) -> float:
    return float(abs(a - b) / max(abs(b), 1e-300))


def _cf1_gate(host: str, device: str, records: list) -> list[dict[str, Any]]:
    """Sweep cond(W); classify each fp32 fit as accepted-correct / refused-loud /
    SILENT-WRONG (the failure we must never see)."""
    rows = []
    for cond in _COND_SWEEP:
        ds = make_grm_conditioned(n=1500, M=300, h2=0.5, cond=cond,
                                  seed=_SEED + int(np.log10(cond) * 7))
        cpu = run_grm_record(ds, backend="cpu", device="cpu", repeats=1)
        gpu = run_grm_record(ds, backend="gpu", device=device, repeats=1)
        records.extend([cpu, gpu])
        refused = bool(gpu.get("refused"))
        if refused:
            verdict = "refused_loud"
            dh2 = None; dbeta = None
        else:
            dh2 = _rel(gpu["heritability"], cpu["heritability"])
            dbeta = float(np.max(np.abs(
                np.asarray(gpu["coefficients"], float)
                - np.asarray(cpu["coefficients"], float))))
            correct = (dh2 <= _FP32_TIER)
            verdict = "accepted_correct" if correct else "SILENT_WRONG"
        rows.append({"study": "cf1_gate", "cond": cond, "device": device,
                     "verdict": verdict, "dh2_vs_fp64": dh2, "dbeta_vs_fp64": dbeta,
                     "refused": refused})
        print(f"  CF1 cond={cond:.0e}  verdict={verdict}  "
              f"dh2={('--' if dh2 is None else f'{dh2:.2e}')}")
    silent_wrong = sum(1 for r in rows if r["verdict"] == "SILENT_WRONG")
    print(f"  >>> silent_wrong_count = {silent_wrong} (must be 0)")
    return rows


def _speed_and_isolation(host: str, device: str, records: list) -> list[dict[str, Any]]:
    rows = []
    for n, M in [(5000, 500), (20000, 1000)]:
        ds = make_grm(n=n, M=M, h2=0.5, seed=_SEED + n)
        cpu = run_grm_record(ds, backend="cpu", device="cpu", repeats=3)
        gpu = run_grm_record(ds, backend="gpu", device=device, repeats=3)
        recs = [cpu, gpu]
        row = {"study": "speed", "n": n, "M": M, "device": device,
               "cpu_s": cpu.get("wall_median_s"),
               "gpu_fp32_s": (None if gpu.get("error") else gpu.get("wall_median_s"))}
        if device == "cuda":   # R11 same-precision hardware pivot (CUDA only)
            gpu64 = run_grm_record(ds, backend="gpu_fp64", device=device, repeats=3)
            recs.append(gpu64)
            row["gpu_fp64_s"] = (None if gpu64.get("error") else gpu64.get("wall_median_s"))
            if row["gpu_fp64_s"] and row["cpu_s"]:
                row["gpu_fp64_speedup_vs_cpu"] = row["cpu_s"] / row["gpu_fp64_s"]
        if row["gpu_fp32_s"] and row["cpu_s"]:
            row["gpu_fp32_speedup_vs_cpu"] = row["cpu_s"] / row["gpu_fp32_s"]
        records.extend(recs)
        rows.append(row)
        print(f"  speed n={n} M={M}  cpu={row['cpu_s']}  "
              f"gpu_fp32={row['gpu_fp32_s']}  "
              f"speedup={row.get('gpu_fp32_speedup_vs_cpu')}")
    return rows


def generate(host: str, device: str, *, repeats: int) -> Path:
    env = env_manifest(device=device, host=host)
    require_pypi(env)
    records: list[dict[str, Any]] = []
    print(f"== CF-1 no-silent-wrong gate ({device}) ==")
    cf1 = _cf1_gate(host, device, records)
    print(f"== speed / R11 isolation ({device}) ==")
    spd = _speed_and_isolation(host, device, records)

    config = {"study": "grm_gpu", "device": device, "cond_sweep": _COND_SWEEP,
              "fp32_tier": _FP32_TIER, "seed": _SEED,
              "note": "CF-1 gate must give silent_wrong_count=0; GPU must beat CPU "
                      "in its regime (else no reason to exist)"}
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"grm_gpu_{device}_{host}.json", run)
    _write_summary_csv(out_dir / f"grm_cf1_gate_{device}_{host}_summary.csv", cf1)
    _write_summary_csv(out_dir / f"grm_speed_{device}_{host}_summary.csv", spd)
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
        w.writeheader(); w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--device", required=True, choices=["mps", "cuda"])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    generate(args.host, args.device, repeats=args.repeats)


if __name__ == "__main__":
    main()
