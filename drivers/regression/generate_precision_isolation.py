"""Isolate precision from hardware in the GPU speed claims (RIGOR R11).

The headline device-scaling tables compare fp64-CPU against fp32-GPU. That number
bundles two independent effects — the fp32-vs-fp64 win and the GPU-vs-CPU win — so
it is not attributable. R11 requires a same-precision pivot that measures the
HARDWARE effect alone: ``gpu_fp64`` vs ``cpu_fp64`` on the identical grid.

This driver times THREE backends on one host (CUDA only — ``gpu_fp64`` needs a
data-center-class double-precision GPU; Apple Silicon has no fp64) over the same
(model, n, p) grid, and emits one self-contained table carrying BOTH:

  * speedup_fp32_vs_cpu   the BUNDLED number (hardware + precision) — what the
                          existing scaling tables report, kept for honesty.
  * speedup_fp64_vs_cpu   the HARDWARE-ISOLATED number (both fp64) — the clean
                          attributable hardware effect R11 asks for.

It also records the gpu_fp64 vs cpu coefficient agreement (~1e-15: gpu_fp64 is the
numerically-exact path, so the isolated timing is on truly equivalent computation).

The CPU-vs-R column lives in the correctness study (already same-precision fp64);
the reference BLAS provenance is captured separately by ``_r/blas_info.R``.

    python -m drivers.regression.generate_precision_isolation --host forge --device cuda
"""

from __future__ import annotations

import argparse
import csv
import socket
from pathlib import Path
from typing import Any

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.measure import measure
from pystatsval.serialize import build_run

# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
# Drivers hardcode their artifact dir, so an ordinary run would otherwise
# silently destroy the artifacts a report was blessed against.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent

# Same shape as the scaling grid, trimmed to the large end where the GPU matters
# (small n is dominated by dispatch overhead — see the scaling study's limitation).
_N_GRID = (10_000, 100_000, 500_000)
_P_GRID = (32, 128)
_MODELS = (("ols", None), ("glm_poisson", "poisson"))


def _fam(family: str | None):
    from pystatistics.regression import Gamma
    if family == "gamma":
        return Gamma(link="log")
    return family


def _synth(n: int, p: int, family: str | None, *, seed: int):
    rng = np.random.default_rng(seed)
    Xp = rng.standard_normal((n, p))
    X = np.column_stack([np.ones(n), Xp])
    if family == "poisson":
        beta = rng.standard_normal(p + 1) * 0.4 / np.sqrt(p)
        y = rng.poisson(np.exp(X @ beta)).astype(np.float64)
    else:
        beta = rng.standard_normal(p + 1) * 0.5
        y = X @ beta + rng.standard_normal(n)
    return X, y


def _time_backend(X, y, family, backend, device_label, *, repeats, warmup):
    """Median wall-clock + the fitted coefficients for one backend."""
    from pystatistics.regression import Design, fit
    d = Design.from_arrays(X, y)

    def _call():
        return fit(d, family=_fam(family), backend=backend)

    wall, sol = measure(_call, device=device_label, repeats=repeats, warmup=warmup)
    return wall, np.asarray(sol.coefficients, float)


def generate(host: str, device: str, *, repeats: int, warmup: int) -> Path:
    if device != "cuda":
        raise SystemExit("precision isolation requires CUDA (gpu_fp64 has no MPS path)")

    env = env_manifest(device=device, host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seed = 0
    for model_key, family in _MODELS:
        for n in _N_GRID:
            for p in _P_GRID:
                seed += 1
                X, y = _synth(n, p, family, seed=seed)

                w_cpu, c_cpu = _time_backend(X, y, family, "cpu", "cpu",
                                             repeats=repeats, warmup=warmup)
                w_f32, c_f32 = _time_backend(X, y, family, "gpu", "cuda",
                                             repeats=repeats, warmup=warmup)
                w_f64, c_f64 = _time_backend(X, y, family, "gpu_fp64", "cuda",
                                             repeats=repeats, warmup=warmup)

                wc, wf32, wf64 = w_cpu["median_s"], w_f32["median_s"], w_f64["median_s"]
                fp64_rel = float(np.max(np.abs(c_f64 - c_cpu) /
                                        np.maximum(np.abs(c_cpu), 1e-300)))
                row = {
                    "model_key": model_key, "n": n, "p": p,
                    "wall_cpu_fp64_s": wc,
                    "wall_gpu_fp32_s": wf32,
                    "wall_gpu_fp64_s": wf64,
                    "speedup_fp32_vs_cpu": (wc / wf32) if wf32 else None,
                    "speedup_fp64_vs_cpu": (wc / wf64) if wf64 else None,
                    "gpu_fp64_coef_rel_vs_cpu": fp64_rel,
                }
                rows.append(row)
                records.append({"engine": f"isolation:{device}", "dataset": model_key,
                                "n": n, "p": p, **row})
                print(f"  {model_key:12} n={n:>7} p={p:>3}: "
                      f"cpu={wc:.4g}s fp32={wf32:.4g}s({wc/wf32:.1f}x) "
                      f"fp64={wf64:.4g}s({wc/wf64:.1f}x) fp64_rel={fp64_rel:.1e}",
                      flush=True)

    config = {"study": "precision_isolation_r11", "device": device,
              "n_grid": list(_N_GRID), "p_grid": list(_P_GRID),
              "models": [m for m, _ in _MODELS], "repeats": repeats, "warmup": warmup}
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"precision_isolation_{device}_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    cols = ["model_key", "n", "p", "wall_cpu_fp64_s", "wall_gpu_fp32_s",
            "wall_gpu_fp64_s", "speedup_fp32_vs_cpu", "speedup_fp64_vs_cpu",
            "gpu_fp64_coef_rel_vs_cpu"]
    with (out_dir / f"{stem}_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {run_path}")
    return run_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--device", default="cuda", choices=["cuda"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    generate(args.host, args.device, repeats=args.repeats, warmup=args.warmup)


if __name__ == "__main__":
    main()
