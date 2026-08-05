"""Generate the GRM / low-rank mixed-model (grm_lmm) CPU correctness artifacts.

One job: for a grid of simulated genomic designs (heritability × size) fit BOTH
pystatistics ``grm_lmm(backend='cpu')`` and R ``rrBLUP::mixed.solve`` on the
identical numbers, reduce every estimate (β, SEs, variance components,
heritability, genetic-value BLUPs) to scalar agreement metrics, and freeze a
``validation-run/v1`` artifact + summary CSV under ``artifacts/mixed/v<ver>/runs/``.

Estimable quantities (β, σ_g², σ_e², h², BLUPs) are convention-independent and
compared directly. logLik is NOT compared across engines: rrBLUP's restricted
log-likelihood uses a different additive constant than grm_lmm (the fits agree on
every estimate regardless) — so we assert agreement on the estimates and record
both logLik values without a cross-engine tolerance on them.

PyPI-only (``require_pypi``). Run from the dedicated PyPI venv (needs R + rrBLUP).

    python -m drivers.mixed.generate_grm_correctness --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
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

from drivers.mixed.grm_datasets import make_grm
from drivers.mixed.run_grm import run_grm_record
from drivers.mixed.run_r_grm import run_r_grm_record

from artifact_root import artifact_root  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260630

# (n, M, h2) grid: three heritabilities across the range, two sizes.
_GRID = [
    (400, 80, 0.2), (400, 80, 0.5), (400, 80, 0.8),
    (800, 150, 0.5),
]


def _max_rel(a, b) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _blup_agreement(a, b) -> dict[str, float]:
    a = np.asarray(a, float); b = np.asarray(b, float)
    rms = float(np.sqrt(np.mean(b ** 2))) or 1.0
    return {"blup_corr": float(np.corrcoef(a, b)[0, 1]),
            "blup_max_scaled": float(np.max(np.abs(a - b)) / rms)}


def _agreement(sut, ref, ds) -> dict[str, Any]:
    bl = _blup_agreement(sut["genetic_values"], ref["genetic_values"])
    return {
        "dataset": ds.key, "n": ds.n, "p": ds.p, "M": ds.M,
        "true_h2": ds.true_h2,
        "beta_max_rel": _max_rel(sut["coefficients"], ref["coefficients"]),
        "beta_se_max_rel": _max_rel(sut["standard_errors"], ref["standard_errors"]),
        "var_genetic_rel": _max_rel([sut["var_genetic"]], [ref["var_genetic"]]),
        "var_residual_rel": _max_rel([sut["var_residual"]], [ref["var_residual"]]),
        "heritability_rel": _max_rel([sut["heritability"]], [ref["heritability"]]),
        "blup_corr": bl["blup_corr"], "blup_max_scaled": bl["blup_max_scaled"],
        "py_h2": sut["heritability"], "r_h2": ref["heritability"],
        "py_loglik": sut["log_likelihood"], "r_loglik": ref["log_likelihood"],
    }


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for (n, M, h2) in _GRID:
        ds = make_grm(n=n, M=M, h2=h2, seed=_SEED + n + int(h2 * 100))
        sut = run_grm_record(ds, backend="cpu", device="cpu", repeats=repeats)
        if sut.get("error"):
            raise RuntimeError(f"grm_lmm cpu failed for {ds.key}: {sut['error']}")
        ref, _raw = run_r_grm_record(ds)
        records.extend([sut, ref])
        row = _agreement(sut, ref, ds)
        rows.append(row)
        print(f"  {ds.key:22s} h2={h2}  beta={row['beta_max_rel']:.1e} "
              f"vg={row['var_genetic_rel']:.1e} ve={row['var_residual_rel']:.1e} "
              f"h2_rel={row['heritability_rel']:.1e} blup_corr={row['blup_corr']:.6f}")

    config = {
        "study": "grm_correctness_vs_rrblup",
        "grid": [{"n": n, "M": M, "h2": h2} for (n, M, h2) in _GRID],
        "backend": "cpu", "repeats": repeats, "seed": _SEED,
        "reference": "R rrBLUP::mixed.solve (GBLUP/GRM; K = W Wᵀ / M, Z = I)",
        "note": "estimable quantities (β, σ_g², σ_e², h², BLUPs) compared directly; "
                "logLik NOT compared across engines (different REML additive "
                "constant — estimates agree regardless)",
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = artifact_root(_REPO) / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"grm_correctness_cpu_{host}.json", run)
    _write_summary_csv(out_dir / f"grm_correctness_{host}_summary.csv", rows)
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
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
