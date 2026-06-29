"""Generate the regression correctness-vs-R artifacts (one device, one host).

One job: for each canonical model (OLS + 5 GLM families) fit BOTH pystatistics
and the R reference on the identical design, reduce the coefficient/SE/p vectors
to scalar agreement metrics, and freeze a ``validation-run/v1`` artifact plus a
flat summary CSV under ``artifacts/regression/v<ver>/runs/``.

PyPI-only: refuses to write a canonical run unless pystatistics is installed from
PyPI (``require_pypi``). Run it from the dedicated PyPI venv.

    python -m drivers.regression.generate_correctness --host powerhouse --device cpu

The agreement reductions (max |Δ| and max relative Δ across the coefficient and
standard-error vectors, plus the model-level R²/deviance gap) are the correctness
claim; the renderer turns them into the agreement table.
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

from drivers.regression import datasets
from drivers.regression.run_pystatistics import run_regression_record
from drivers.regression.run_r_regression import run_r_regression_record

_REPO = Path(__file__).resolve().parent.parent.parent


def _rel(a: float, b: float) -> float:
    """Relative difference |a-b|/max(|b|, tiny); b is the reference (R)."""
    denom = max(abs(b), 1e-300)
    return abs(a - b) / denom


def _vec_agreement(sut: list[float], ref: list[float]) -> dict[str, float]:
    """Max absolute and max relative gap between two aligned numeric vectors."""
    s = np.asarray(sut, float)
    r = np.asarray(ref, float)
    if s.shape != r.shape:
        raise ValueError(f"vector length mismatch: sut {s.shape} vs ref {r.shape}")
    abs_d = np.abs(s - r)
    rel_d = np.array([_rel(si, ri) for si, ri in zip(s, r)])
    return {"max_abs": float(abs_d.max()), "max_rel": float(rel_d.max())}


def _agreement_record(sut: dict[str, Any], ref: dict[str, Any],
                      spec: datasets.ModelSpec) -> dict[str, Any]:
    """Reduce one pystatistics-vs-R pair to the per-model agreement summary row."""
    coef = _vec_agreement(sut["coefficients"], ref["coefficients"])
    se = _vec_agreement(sut["standard_errors"], ref["standard_errors"])
    pv = _vec_agreement(sut["p_values"], ref["p_values"])
    stat = _vec_agreement(sut["test_statistic"], ref["test_statistic"])
    wall_sut = sut.get("wall_median_s")
    wall_ref = ref.get("wall_median_s")
    speedup = (wall_ref / wall_sut) if (wall_sut and wall_ref) else None
    row: dict[str, Any] = {
        "model_key": spec.key,
        "family": spec.family,
        "dataset": spec.dataset,
        "formula": spec.formula,
        "n": sut.get("n"),
        "n_coef": len(sut["coefficients"]),
        "coef_max_abs": coef["max_abs"],
        "coef_max_rel": coef["max_rel"],
        "se_max_rel": se["max_rel"],
        "tstat_max_rel": stat["max_rel"],
        "pval_max_abs": pv["max_abs"],
        "wall_pystat_s": wall_sut,
        "wall_r_s": wall_ref,
        "speedup_vs_r": speedup,
    }
    if spec.key == "ols":
        row["r2_pystat"] = sut.get("r_squared")
        row["r2_r"] = ref.get("r_squared")
        row["r2_abs_diff"] = abs(sut.get("r_squared", 0.0) - ref.get("r_squared", 0.0))
    else:
        row["deviance_pystat"] = sut.get("deviance")
        row["deviance_r"] = ref.get("deviance")
        row["deviance_rel_diff"] = _rel(sut.get("deviance", 0.0), ref.get("deviance", 0.0))
        # AIC/BIC agreement vs R — the Gamma (4.3.0) and negbin-theta (4.3.1) AIC
        # fixes are validated here: both information criteria now match R.
        if sut.get("aic") is not None and ref.get("aic") is not None:
            row["aic_rel"] = _rel(sut["aic"], ref["aic"])
        if sut.get("bic") is not None and ref.get("bic") is not None:
            row["bic_rel"] = _rel(sut["bic"], ref["bic"])
        # Negative binomial: the auto-estimated theta is now exposed (4.3.1), so
        # compare it directly against MASS::glm.nb's theta.
        if sut.get("theta") is not None and ref.get("theta") is not None:
            row["theta_pystat"] = sut["theta"]
            row["theta_r"] = ref["theta"]
            row["theta_rel"] = _rel(sut["theta"], ref["theta"])
        row["n_iter"] = sut.get("n_iter")
    return row


def generate(host: str, device: str, *, repeats: int) -> Path:
    """Run the full correctness sweep on ``device`` and freeze the artifacts."""
    backend = "cpu" if device == "cpu" else "gpu"
    env = env_manifest(device=device, host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []

    for key, loader in datasets.LOADERS.items():
        X, y, spec = loader()
        sut = run_regression_record(
            X, y, family=spec.family, backend=backend,
            dataset=spec.dataset, model_key=key, link=spec.link,
            repeats=repeats, warmup=1)
        ref_rec, _raw = run_r_regression_record(
            X, y, r_family=spec.r_family, dataset=spec.dataset, model_key=key)
        if sut.get("error"):
            raise RuntimeError(f"pystatistics fit failed for {key}: {sut['error']}")
        records.extend([sut, ref_rec])
        agreement_rows.append(_agreement_record(sut, ref_rec, spec))
        print(f"  {key:14s} n={sut['n']:>6}  "
              f"coef_max_rel={agreement_rows[-1]['coef_max_rel']:.2e}  "
              f"se_max_rel={agreement_rows[-1]['se_max_rel']:.2e}")

    config = {
        "study": "correctness_vs_r",
        "models": list(datasets.LOADERS.keys()),
        "backend": backend,
        "repeats": repeats,
        "reference": "R lm()/glm()/MASS::glm.nb()",
    }
    run = build_run(env=env, config=config, records=records)

    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    stem = f"correctness_{device}_{host}"
    run_path = write_run(out_dir / f"{stem}.json", run)
    _write_summary_csv(out_dir / f"{stem}_summary.csv", agreement_rows)
    print(f"\nwrote {run_path}")
    print(f"wrote {out_dir / f'{stem}_summary.csv'}")
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
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()
    generate(args.host, args.device, repeats=args.repeats)


if __name__ == "__main__":
    main()
