"""Generate the multivariate correctness-vs-R artifacts (CPU reference path).

One job: for each canonical PCA design (iris, USArrests; covariance + correlation
scalings) fit BOTH pystatistics and R ``prcomp`` on the identical numbers,
sign-align the arbitrary eigenvector signs, reduce sdev/rotation/scores to scalar
agreement metrics, and freeze a ``validation-run/v1`` artifact plus a flat summary
CSV under ``artifacts/multivariate/v<ver>/runs/``. Also records the iris 1-factor
ML factor-analysis case vs ``factanal`` (which exercises finding F2 — the Heywood
uniqueness-floor convention difference — recorded honestly; see findings_ledger).

PyPI-only (``require_pypi``). Run from the dedicated 4.4.0 PyPI venv with
``MVNMLE_DATA_DIR`` pointing at the curated HDF5 store.

    python -m drivers.multivariate.generate_correctness --host powerhouse
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

from drivers.multivariate import datasets
from drivers.multivariate.run_pystatistics import run_pca_record, run_fa_record
from drivers.multivariate.run_r_multivariate import run_r_pca_record, run_r_fa_record

_REPO = Path(__file__).resolve().parent.parent.parent

# PCA correctness grid: (dataset_key, scale). Both scalings on both datasets;
# USArrests scale=True is the textbook correlation-vs-covariance case.
_PCA_GRID = [
    ("iris", False), ("iris", True),
    ("usarrests", False), ("usarrests", True),
]


def _max_rel(a, b) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _max_abs(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.max(np.abs(a - b)))


def _sign_align(Vp, Vr):
    """Return R's matrix with per-column signs flipped to match pystatistics.

    Eigenvector signs are arbitrary (SVD/LAPACK-dependent), so before comparing
    we flip each R column by the sign of its dot product with the corresponding
    pystatistics column. The same per-column sign applies to the scores.
    """
    Vp = np.asarray(Vp, float)
    Vr = np.asarray(Vr, float)
    signs = np.sign((Vp * Vr).sum(axis=0))
    signs[signs == 0] = 1.0
    return Vr * signs[np.newaxis, :], signs


def _pca_agreement(sut: dict[str, Any], ref: dict[str, Any],
                   key: str, scale: bool) -> dict[str, Any]:
    rot_r, signs = _sign_align(sut["rotation"], ref["rotation"])
    scores_r = np.asarray(ref["scores"], float) * signs[np.newaxis, :]
    wall_s = sut.get("wall_median_s")
    wall_r = ref.get("wall_median_s")
    return {
        "analysis": "pca",
        "dataset": key,
        "scaling": "correlation" if scale else "covariance",
        "n": sut.get("n"),
        "p": sut.get("p"),
        "k": sut.get("n_components"),
        "sdev_max_rel": _max_rel(sut["sdev"], ref["sdev"]),
        "rotation_max_abs": _max_abs(sut["rotation"], rot_r),
        "scores_max_rel": _max_rel(sut["scores"], scores_r),
        "evr_max_abs": _max_abs(sut["explained_variance_ratio"],
                                ref["explained_variance_ratio"]),
        "center_max_rel": _max_rel(sut["center"], ref["center"]),
        "scale_max_rel": (_max_rel(sut["scale"], ref["scale"])
                          if sut.get("scale") is not None else 0.0),
        "wall_pystat_s": wall_s,
        "wall_r_s": wall_r,
    }


def _fa_agreement(sut: dict[str, Any], ref: dict[str, Any],
                  key: str, n_factors: int) -> dict[str, Any]:
    load_r, _ = _sign_align(sut["loadings"], ref["loadings"])
    return {
        "analysis": "factor_analysis",
        "dataset": key,
        "n_factors": n_factors,
        "n": sut.get("n"),
        "p": sut.get("p"),
        "uniq_max_abs": _max_abs(sut["uniquenesses"], ref["uniquenesses"]),
        "loadings_max_abs": _max_abs(sut["loadings"], load_r),
        "objective_rel": _max_rel([sut["objective"]], [ref["objective"]]),
        "chi_sq_rel": (_max_rel([sut["chi_sq"]], [ref["chi_sq"]])
                       if sut.get("chi_sq") and ref.get("chi_sq") else None),
        "py_converged": sut.get("converged"),
        "r_warning": ref.get("r_warning", ""),
        "note": "F2: Heywood uniqueness-floor convention differs (py ~0 vs R "
                "lower=0.005); see findings_ledger. Gathered for 4.4.1.",
    }


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    # ---- PCA correctness grid ----
    for key, scale in _PCA_GRID:
        X, names, spec = datasets.PCA_LOADERS[key]()
        sut = run_pca_record(X, backend="cpu", dataset=key, scale=scale,
                             repeats=repeats, warmup=1)
        if sut.get("error"):
            raise RuntimeError(f"pystatistics PCA failed for {key} scale={scale}: "
                               f"{sut['error']}")
        ref, _raw = run_r_pca_record(X, names, dataset=key, center=True,
                                     scale=scale, reps=repeats)
        records.extend([sut, ref])
        row = _pca_agreement(sut, ref, key, scale)
        rows.append(row)
        print(f"  PCA {key:10s} {row['scaling']:11s}  "
              f"sdev_rel={row['sdev_max_rel']:.2e}  "
              f"rot_abs={row['rotation_max_abs']:.2e}  "
              f"scores_rel={row['scores_max_rel']:.2e}")

    # ---- FA iris 1-factor (records F2; multi-factor deferred to 4.4.1) ----
    X, names, spec = datasets.load_iris()
    fa_sut = run_fa_record(X, n_factors=spec.n_factors, dataset="iris",
                           repeats=max(3, repeats // 2), warmup=1)
    if fa_sut.get("error"):
        raise RuntimeError(f"pystatistics FA failed for iris: {fa_sut['error']}")
    fa_ref, _raw = run_r_fa_record(X, names, dataset="iris",
                                   n_factors=spec.n_factors, reps=3)
    records.extend([fa_sut, fa_ref])
    fa_row = _fa_agreement(fa_sut, fa_ref, "iris", spec.n_factors)
    rows.append(fa_row)
    print(f"  FA  iris 1-factor   uniq_abs={fa_row['uniq_max_abs']:.2e}  "
          f"obj_rel={fa_row['objective_rel']:.2e}  (F2 convention gap, gathered)")

    config = {
        "study": "correctness_vs_r",
        "pca_grid": [{"dataset": k, "scaling": "correlation" if s else "covariance"}
                     for k, s in _PCA_GRID],
        "fa": {"dataset": "iris", "n_factors": 1,
               "status": "1-factor recorded; multi-factor deferred to 4.4.1 (F1 varimax)"},
        "backend": "cpu",
        "repeats": repeats,
        "reference": "R stats::prcomp / stats::factanal",
        "sign_convention": "eigenvector signs aligned per-column before comparison",
    }
    run = build_run(env=env, config=config, records=records)

    out_dir = _REPO / "artifacts" / "multivariate" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"correctness_cpu_{host}.json", run)
    # Split PCA and FA into separate summary CSVs (different column sets) so each
    # renders as a clean table.
    pca_rows = [r for r in rows if r["analysis"] == "pca"]
    fa_rows = [r for r in rows if r["analysis"] == "factor_analysis"]
    _write_summary_csv(out_dir / f"correctness_pca_{host}_summary.csv", pca_rows)
    _write_summary_csv(out_dir / f"correctness_fa_{host}_summary.csv", fa_rows)
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
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    generate(args.host, repeats=args.repeats)


if __name__ == "__main__":
    main()
