"""Generate the regression scaling / device-pivot artifacts.

One job: sweep synthetic OLS and GLM designs over a grid of (n, p), fit them on a
single device (cpu / mps / cuda), and freeze the wall-clock + peak-memory records
so the renderer's ``device_pivot`` can pair CPU against the accelerator per problem.

Correctness is NOT the question here (that is the correctness study, on real data
vs R); this study answers "how does pystatistics scale, and where does the GPU win".
Data are synthetic but deterministic (fixed seed, no randomness in the timed region).

    # Powerhouse: CPU then MPS
    python -m drivers.regression.generate_scaling --host powerhouse --device cpu
    python -m drivers.regression.generate_scaling --host powerhouse --device mps
    # Forge: CPU then CUDA
    python -m drivers.regression.generate_scaling --host forge --device cuda

Each invocation writes one device's rows; ``device_pivot`` later joins the cpu run
and the accelerator run for the same host by (problem, p).
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

from drivers.regression.run_pystatistics import run_regression_record

_REPO = Path(__file__).resolve().parent.parent.parent

# (n, p) grid. p includes the intercept-free predictor count; the design adds the
# intercept column. Kept modest at the top end so a full sweep is minutes, not hours.
_N_GRID = (1_000, 10_000, 100_000, 500_000)
_P_GRID = (8, 32, 128)
# Families swept: OLS (normal-eq/QR/Cholesky) plus the three GLM IRLS paths —
# binomial (bounded, the float32-stable case) and the two LOG-LINK families
# (Poisson, Gamma), which are the ones that expose float32 ill-conditioning at
# large n. Covering them here closes the gap that hid the 3.19.0 MPS bug.
_FAMILIES = (
    ("ols", None),
    ("glm_binomial", "binomial"),
    ("glm_poisson", "poisson"),
    ("glm_gamma", "gamma"),
)


def _synth(n: int, p: int, family: str | None, *, seed: int):
    """Deterministic synthetic design with intercept column. No timed randomness.

    GLM coefficients are scaled by 1/sqrt(p) so the linear predictor stays O(1)
    (realistic; avoids exp() overflow for log links), while the large-n float32
    behaviour of each path is still exercised.
    """
    rng = np.random.default_rng(seed)
    Xp = rng.standard_normal((n, p))
    X = np.column_stack([np.ones(n), Xp])
    if family in ("binomial", "poisson", "gamma"):
        beta = rng.standard_normal(p + 1) * 0.4 / np.sqrt(p)
        eta = X @ beta
        if family == "binomial":
            y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)
        elif family == "poisson":
            y = rng.poisson(np.exp(eta)).astype(np.float64)
        else:  # gamma (log link)
            y = rng.gamma(shape=2.0, scale=np.exp(eta) / 2.0) + 1e-3
        return X, y
    # OLS / gaussian
    beta = rng.standard_normal(p + 1) * 0.5
    eta = X @ beta
    y = eta + rng.standard_normal(n)
    return X, y


def generate(host: str, device: str, *, repeats: int, warmup: int) -> Path:
    """Sweep the (family, n, p) grid on ``device`` and freeze the records."""
    backend = "cpu" if device == "cpu" else "gpu"
    env = env_manifest(device=device, host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    seed = 0
    for model_key, family in _FAMILIES:
        for n in _N_GRID:
            for p in _P_GRID:
                seed += 1
                X, y = _synth(n, p, family, seed=seed)
                label = f"{model_key}_n{n}"
                rec = run_regression_record(
                    X, y, family=family, backend=backend,
                    dataset=label, model_key=model_key,
                    repeats=repeats, warmup=warmup)
                # Tag p explicitly so device_pivot can pair by (problem, p).
                rec["p"] = int(p)
                rec["model_key"] = model_key
                records.append(rec)
                rows.append({
                    "model_key": model_key, "n": n, "p": p,
                    "engine": rec["engine"], "device": device,
                    "wall_median_s": rec.get("wall_median_s"),
                    "peak_mem_mb": rec.get("peak_mem_mb"),
                    "n_iter": rec.get("n_iter"),
                    "error": rec.get("error"),
                })
                tag = rec.get("error") or f"{rec.get('wall_median_s'):.4g}s"
                print(f"  {label:18s} p={p:>4}  {device}: {tag}")

    config = {
        "study": "scaling", "device": device,
        "n_grid": list(_N_GRID), "p_grid": list(_P_GRID),
        "families": [f for f, _ in _FAMILIES], "repeats": repeats, "warmup": warmup,
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    stem = f"scaling_{device}_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    _write_csv(out_dir / f"{stem}_summary.csv", rows)
    print(f"\nwrote {run_path}")
    return run_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = ["model_key", "n", "p", "engine", "device",
            "wall_median_s", "peak_mem_mb", "n_iter", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    generate(args.host, args.device, repeats=args.repeats, warmup=args.warmup)


if __name__ == "__main__":
    main()
