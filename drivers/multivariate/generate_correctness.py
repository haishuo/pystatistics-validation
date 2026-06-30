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


_FA_SEED = 20260630  # documented deterministic seed for the synthetic FA designs


def _synth_factor_cases() -> list[tuple[str, np.ndarray, list[str], int]]:
    """Deterministic well-fitting multi-factor designs (the F1 varimax regime).

    Simple-structure loadings (each variable loads on one factor) + per-variable
    noise, so the ML factor model fits well and the rotated solution is identified
    — the clean multi-factor case that raised ConvergenceError before the 4.4.1
    varimax fix. Two designs: a 2-factor (8 vars) and a 3-factor (9 vars).
    """
    cases: list[tuple[str, np.ndarray, list[str], int]] = []
    # 2-factor, p=8
    rng = np.random.default_rng(_FA_SEED)
    L = np.zeros((8, 2)); L[:4, 0] = [0.8, 0.7, 0.75, 0.65]; L[4:, 1] = [0.8, 0.7, 0.75, 0.6]
    psi = np.array([0.3, 0.4, 0.35, 0.45, 0.3, 0.4, 0.35, 0.5])
    X2 = (rng.standard_normal((400, 2)) @ L.T
          + rng.standard_normal((400, 8)) * np.sqrt(psi))
    cases.append(("synth2f", X2, [f"v{i}" for i in range(8)], 2))
    # 3-factor, p=9
    rng = np.random.default_rng(_FA_SEED + 1)
    L3 = np.zeros((9, 3))
    L3[:3, 0] = [0.8, 0.7, 0.75]; L3[3:6, 1] = [0.78, 0.72, 0.7]; L3[6:, 2] = [0.8, 0.68, 0.74]
    psi3 = np.full(9, 0.4)
    X3 = (rng.standard_normal((500, 3)) @ L3.T
          + rng.standard_normal((500, 9)) * np.sqrt(psi3))
    cases.append(("synth3f", X3, [f"v{i}" for i in range(9)], 3))
    return cases


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


def _align_factor_columns(Lp, Lr):
    """Align R's loading columns to pystatistics' by greedy best-|cosine| match.

    Factor loadings are identified only up to column PERMUTATION and per-column
    SIGN (varimax fixes the rotation but not the order/sign). Greedily pair each
    pystatistics column with its best-correlated unused R column, then sign-flip.
    Returns R's loadings reordered+signed to match pystatistics' columns.
    """
    Lp = np.asarray(Lp, float)
    Lr = np.asarray(Lr, float)
    m = Lp.shape[1]
    # cosine similarity matrix between pystatistics and R columns
    norm_p = np.linalg.norm(Lp, axis=0)
    norm_r = np.linalg.norm(Lr, axis=0)
    cos = (Lp.T @ Lr) / np.maximum(np.outer(norm_p, norm_r), 1e-300)
    used = set()
    perm = [-1] * m
    for i in range(m):
        order = np.argsort(-np.abs(cos[i]))
        for j in order:
            if j not in used:
                perm[i] = int(j)
                used.add(int(j))
                break
    Lr_perm = Lr[:, perm]
    signs = np.sign((Lp * Lr_perm).sum(axis=0))
    signs[signs == 0] = 1.0
    return Lr_perm * signs[np.newaxis, :]


def _fa_agreement(sut: dict[str, Any], ref: dict[str, Any],
                  key: str, n_factors: int) -> dict[str, Any]:
    # Uniquenesses, the ML objective and chi-sq are rotation-invariant — compared
    # directly. Loadings are aligned up to column permutation + sign first.
    load_r = _align_factor_columns(sut["loadings"], ref["loadings"])
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

    # ---- FA correctness: single-factor (iris, the F2 Heywood case, now floored
    #      at lower=0.005) + multi-factor (synthetic, the F1 varimax case, now
    #      converging). Validated at 4.4.1 after the gathered F1/F2 bundle. ----
    iris_X, iris_names, iris_spec = datasets.load_iris()
    fa_cases = [("iris", iris_X, iris_names, 1)]
    fa_cases += _synth_factor_cases()
    for label, X, fnames, m in fa_cases:
        fa_sut = run_fa_record(X, n_factors=m, dataset=label,
                               repeats=max(3, repeats // 2), warmup=1)
        if fa_sut.get("error"):
            raise RuntimeError(f"pystatistics FA failed for {label} (m={m}): "
                               f"{fa_sut['error']}")
        fa_ref, _raw = run_r_fa_record(X, fnames, dataset=label, n_factors=m, reps=3)
        records.extend([fa_sut, fa_ref])
        fa_row = _fa_agreement(fa_sut, fa_ref, label, m)
        rows.append(fa_row)
        print(f"  FA  {label:8s} m={m}  conv={fa_row['py_converged']}  "
              f"uniq_abs={fa_row['uniq_max_abs']:.2e}  "
              f"load_abs={fa_row['loadings_max_abs']:.2e}  "
              f"obj_rel={fa_row['objective_rel']:.2e}")

    config = {
        "study": "correctness_vs_r",
        "pca_grid": [{"dataset": k, "scaling": "correlation" if s else "covariance"}
                     for k, s in _PCA_GRID],
        "fa_cases": [{"dataset": lbl, "n_factors": m} for lbl, _, _, m in fa_cases],
        "backend": "cpu",
        "repeats": repeats,
        "reference": "R stats::prcomp / stats::factanal",
        "sign_convention": "PCA loadings sign-aligned; FA loadings aligned up to "
                           "column permutation + sign; uniquenesses/objective/chi2 "
                           "are rotation-invariant",
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
