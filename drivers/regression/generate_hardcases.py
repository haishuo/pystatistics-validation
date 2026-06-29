"""Generate the ADVERSARIAL (hard-case) regression correctness grid (RIGOR R10).

One job: probe regression where R-agreement is most likely to break AND where R
itself changes behaviour, and record whether pystatistics matches R's behaviour —
including its failures, warnings, and refusals, not only its coefficients.

The existing ``generate_correctness`` grid is all well-behaved data; it earns
"matches R when things are easy". This driver earns the harder claim: "matches R
when things are HARD, including matching R's own failures and warnings."

Six scenarios (all isolated from the negative-binomial theta gap):

- **collinear**       a condition-number SWEEP: the CPU QR fit agrees with R right
                      across the range, while the GPU Cholesky path accepts below
                      its fp32 condition gate and refuses (NumericalError) above
                      it — unless force=True. Shows the refusal boundary is
                      honoured on BOTH the accept side and the refuse side.
- **separation_***    complete and quasi logistic separation: R's glm diverges,
                      emits "fitted probabilities numerically 0 or 1 occurred",
                      and reports non-convergence. pystatistics replicates the
                      BEHAVIOUR — it diverges and flags ``converged=False`` at the
                      iteration cap (its signal for the same condition).
- **factor**          a clean Gaussian design with multi-level factors AND an
                      interaction: pystatistics' OWN treatment-contrast coding
                      (``build_terms_design`` + ``C``) is checked against R's
                      default ``model.matrix`` contrasts — same columns, same fit.
- **weights/offset**  prior weights and an offset term: pystatistics 4.2.4 does
                      not expose these in ``fit()``; the call FAILS LOUD with a
                      TypeError (a documented scope limitation per CONVENTIONS A6,
                      never a silently-different answer).
- **rank_deficient**  an exactly aliased column: R drops it to an ``NA``
                      coefficient; pystatistics drops it to ``NaN`` in the same
                      position (drop-to-NA match) while the identified
                      coefficients still agree with R.

    python -m drivers.regression.generate_hardcases --host powerhouse

Writes one run JSON (full per-case detail) plus two summary CSVs — the behaviour
grid and the collinear boundary sweep — under artifacts/regression/v<ver>/runs/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import socket
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

_REPO = Path(__file__).resolve().parent.parent.parent
_R_WORKER = Path(__file__).resolve().parent / "_r" / "hardcase_run.R"


# ── R bridge ──────────────────────────────────────────────────────────────────

def _r_numeric(X_noint: np.ndarray, y: np.ndarray, family: str) -> dict[str, Any]:
    """Fit the R reference on a numeric design (no intercept col); capture behaviour."""
    tmp = Path(tempfile.mkdtemp(prefix="rhard_"))
    x_csv, y_csv, out = tmp / "x.csv", tmp / "y.csv", tmp / "r.json"
    names = [f"x{j}" for j in range(X_noint.shape[1])]
    np.savetxt(x_csv, X_noint, delimiter=",", header=",".join(names), comments="", fmt="%.17g")
    np.savetxt(y_csv, y, fmt="%.17g")
    proc = subprocess.run(
        ["Rscript", str(_R_WORKER), "numeric", str(x_csv), str(y_csv), family, str(out)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"R numeric worker failed:\n{proc.stderr[-2000:]}")
    return json.loads(out.read_text())


def _r_frame(df: pd.DataFrame, formula: str, family: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the R reference from a raw frame + formula; return (json, model.matrix)."""
    tmp = Path(tempfile.mkdtemp(prefix="rhardf_"))
    frame_csv, out, mm_csv = tmp / "frame.csv", tmp / "r.json", tmp / "mm.csv"
    df.to_csv(frame_csv, index=False)
    proc = subprocess.run(
        ["Rscript", str(_R_WORKER), "frame", str(frame_csv), formula, family,
         str(out), str(mm_csv)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"R frame worker failed:\n{proc.stderr[-2000:]}")
    return json.loads(out.read_text()), pd.read_csv(mm_csv)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _vec_max_rel(sut: list, ref: list) -> float:
    """Max relative gap over aligned vectors, skipping positions that are None/NaN
    in EITHER vector (aliased columns are compared separately, not numerically)."""
    out = 0.0
    for s, r in zip(sut, ref):
        if s is None or r is None:
            continue
        s, r = float(s), float(r)
        if np.isnan(s) or np.isnan(r):
            continue
        out = max(out, _rel(s, r))
    return out


# ── scenario builders (deterministic) ─────────────────────────────────────────

# ── the two studies ───────────────────────────────────────────────────────────

_COND_RE = re.compile(r"condition number:\s*([0-9.eE+\-]+)")


def _fit_rel(X: np.ndarray, beta_sut: np.ndarray, beta_ref: np.ndarray,
             y: np.ndarray) -> tuple[float, float]:
    """Fit-level agreement: max relative gap in PREDICTIONS, and relative RSS gap.

    For ill-conditioned designs the individual coefficients are non-identifiable in
    fp32, so coefficient agreement degrades even when both engines minimise RSS to
    the same point. The honest accept-side claim is therefore on the FIT (the
    predictions and the residual sum of squares), not the coefficient vector."""
    f_sut, f_ref = X @ beta_sut, X @ beta_ref
    fitted_rel = float(np.max(np.abs(f_sut - f_ref)) / (np.max(np.abs(f_ref)) + 1e-30))
    rss_sut = float(np.sum((y - f_sut) ** 2))
    rss_ref = float(np.sum((y - f_ref) ** 2))
    rss_rel = abs(rss_sut - rss_ref) / max(rss_ref, 1e-300)
    return fitted_rel, rss_rel


def _run_collinear(records: list, rows: list) -> None:
    """Condition sweep showing the GPU OLS path NEVER silently returns a wrong FIT.

    Across the conditioning range:
      * CPU QR agrees with R to round-off at EVERY conditioning (the reference).
      * GPU fp32 Cholesky, where it ACCEPTS, matches the CPU fit in predictions and
        RSS; the per-coefficient gap grows with conditioning because the
        coefficients become non-identifiable in fp32 (the documented fp32 tier),
        NOT because the fit is wrong — predictions and RSS track CPU throughout.
      * Where fp32 cannot represent the solve (condition gate exceeded, or the fp32
        Cholesky loses positive-definiteness), the GPU FAILS LOUD (NumericalError),
        never returning a silently-wrong answer. force=True overrides the gate.

    The decision variable is the backend's OWN fp32-Gram condition number
    (``sol.info['condition_number']`` on accept; parsed from the error on refuse),
    not numpy's fp64 cond(X). Sorting by it makes the boundary read monotonically.
    """
    from pystatistics.regression import fit
    from pystatistics.core.exceptions import NumericalError

    n = 800
    rng = np.random.default_rng(99)
    x1 = rng.standard_normal(n)
    noise = rng.standard_normal(n)
    beta = np.array([1.0, 0.5, -0.3])
    sweep = []
    for eps in np.logspace(-1.0, -4.0, 10):
        X = np.column_stack([np.ones(n), x1, x1 + eps * noise])
        y = X @ beta + 0.01 * rng.standard_normal(n)
        npcond = float(np.linalg.cond(X))

        # CPU QR — the R-reference path; must agree with R at every conditioning.
        cpu = fit(X, y, backend="cpu")
        cpu_coef = np.asarray(cpu.coefficients, float)
        rref = _r_numeric(X[:, 1:], y, "lm")
        coef_rel = _vec_max_rel(list(cpu.coefficients), rref["coefficients"])

        # GPU Cholesky — accept below the gate, refuse (fail-loud) above it.
        lib_cond = None
        gpu_coef_rel = gpu_fit_rel = gpu_rss_rel = None
        refuse_reason = ""
        try:
            g = fit(X, y, backend="gpu")
            gpu_status = "accepted"
            lib_cond = float(g.info.get("condition_number"))
            g_coef = np.asarray(g.coefficients, float)
            gpu_coef_rel = _vec_max_rel(list(g_coef), list(cpu_coef))
            gpu_fit_rel, gpu_rss_rel = _fit_rel(X, g_coef, cpu_coef, y)
        except NumericalError as exc:
            gpu_status = "refused"
            m = _COND_RE.search(str(exc))
            lib_cond = float(m.group(1)) if m else None
            refuse_reason = ("gate>1e6" if m else "fp32-cholesky-breakdown")
        # force=True must always proceed (the documented override).
        try:
            fit(X, y, backend="gpu", force=True)
            force_status = "accepted"
        except Exception as exc:  # noqa: BLE001
            force_status = type(exc).__name__
        sweep.append({
            "scenario": "collinear",
            "np_cond": npcond,
            "lib_cond_fp32": lib_cond,
            "cpu_coef_max_rel_vs_r": coef_rel,
            "gpu_status": gpu_status,
            "gpu_refuse_reason": refuse_reason,
            "gpu_coef_rel_vs_cpu": gpu_coef_rel,
            "gpu_fitted_rel_vs_cpu": gpu_fit_rel,
            "gpu_rss_rel_vs_cpu": gpu_rss_rel,
            "gpu_force_status": force_status,
        })

    # Sort by the gate's own metric so the accept/refuse boundary reads cleanly.
    sweep.sort(key=lambda r: (r["lib_cond_fp32"] is None, r["lib_cond_fp32"] or 0.0))
    for row in sweep:
        rows.append(row)
        records.append({"engine": "hardcase:collinear", "dataset": "synthetic",
                        "n": n, "p": 3, **row})
        lc = row["lib_cond_fp32"]
        lcs = f"{lc:.2e}" if lc is not None else "n/a"
        extra = ""
        if row["gpu_status"] == "accepted":
            extra = (f" coef_rel={row['gpu_coef_rel_vs_cpu']:.2e} "
                     f"fitted_rel={row['gpu_fitted_rel_vs_cpu']:.2e} "
                     f"rss_rel={row['gpu_rss_rel_vs_cpu']:.2e}")
        else:
            extra = f" ({row['gpu_refuse_reason']})"
        print(f"  collinear np_cond={row['np_cond']:.2e} lib_cond={lcs} "
              f"cpu_rel={row['cpu_coef_max_rel_vs_r']:.2e} gpu={row['gpu_status']}{extra}")


def _run_separation(records: list, brows: list) -> None:
    """Complete + quasi logistic separation: replicate R's divergence behaviour."""
    from pystatistics.regression import fit

    n = 60
    x = np.linspace(-3.0, 3.0, n)
    Xd = np.column_stack([np.ones(n), x])
    for case, y in (("separation_complete", (x > 0).astype(float)),
                    ("separation_quasi", None)):
        if y is None:
            y = (x > 0).astype(float)
            y[n // 2] = 1.0 - y[n // 2]      # one flip → quasi-separation

        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")
            sol = fit(Xd, y, family="binomial", backend="cpu")
        py_warns = [str(w.message) for w in wlist]
        py_conv = bool(getattr(sol, "converged", True))
        py_iter = int(getattr(sol, "n_iter", 0) or 0)
        py_max_abs_coef = float(np.max(np.abs(sol.coefficients)))

        r = _r_numeric(Xd[:, 1:], y, "binomial")
        r_conv = bool(r["converged"])
        r_warns = list(r.get("warnings") or [])
        r_max_abs_coef = float(np.max(np.abs([c for c in r["coefficients"] if c is not None])))
        coef_rel = _vec_max_rel(list(sol.coefficients), r["coefficients"])

        # Behaviour match: both diverge (large coefs) AND both signal trouble
        # (pystatistics via converged=False; R via converged=False and/or a warning).
        both_diverge = py_max_abs_coef > 10 and r_max_abs_coef > 10
        both_flag = (not py_conv) and ((not r_conv) or len(r_warns) > 0)
        match = "behaviour-match" if (both_diverge and both_flag) else "MISMATCH"

        py_behavior = f"diverges (max|β|={py_max_abs_coef:.0f}), converged=False@{py_iter}"
        r_flag = "converged=False" if not r_conv else "converged=True"
        r_behavior = (f"diverges (max|β|={r_max_abs_coef:.0f}), {r_flag}, "
                      f"{len(r_warns)} warning(s)")
        brow = {
            "case": case, "n": n, "p": 2,
            "pystat_behavior": py_behavior,
            "r_behavior": r_behavior,
            "match": match,
            "coef_max_rel": coef_rel,
            "detail": (r_warns[0][:90] if r_warns else "") ,
        }
        brows.append(brow)
        records.append({"engine": f"hardcase:{case}", "dataset": "synthetic", "n": n,
                        "p": 2, "pystat_warnings": py_warns, "r_warnings": r_warns,
                        "py_converged": py_conv, "r_converged": r_conv,
                        "py_n_iter": py_iter, **brow})
        print(f"  {case}: {match}  py[{py_behavior}]  r[{r_behavior}]")


def _run_factor(records: list, brows: list) -> None:
    """Factor + interaction contrast coding vs R's default treatment contrasts."""
    from pystatistics.regression import fit
    from pystatistics.regression.terms import build_terms_design, C

    n = 400
    rng = np.random.default_rng(7)
    g3 = rng.choice(["a", "b", "c"], n)
    g2 = rng.choice(["x", "y"], n)
    z = rng.standard_normal(n)
    src = {"g3": g3, "g2": g2, "z": z}
    terms = ["z", C("g3"), C("g2"), (C("g3"), C("g2")), (C("g3"), "z")]
    X, names = build_terms_design(src, terms, intercept=True)

    beta = rng.standard_normal(X.shape[1]) * 0.5
    y = X @ beta + 0.3 * rng.standard_normal(n)
    sol = fit(X, y, backend="cpu")

    df = pd.DataFrame({"g3": g3, "g2": g2, "z": z, "yv": y})
    formula = "yv ~ z + g3 + g2 + g3:g2 + g3:z"
    r, mm = _r_frame(df, formula, "lm")

    # Design-match: every pystatistics column must equal some R model.matrix
    # column (value-for-value) and vice versa — a bijection proves identical
    # contrast coding regardless of naming/ordering.
    py_cols = [X[:, j] for j in range(X.shape[1])]
    r_cols = [mm[c].to_numpy(float) for c in mm.columns]
    matched_r = set()
    coef_pairs = []
    bijection = True
    for j, pc in enumerate(py_cols):
        hit = None
        for k, rc in enumerate(r_cols):
            if k in matched_r:
                continue
            if np.allclose(pc, rc, atol=1e-9, rtol=0):
                hit = k
                break
        if hit is None:
            bijection = False
            break
        matched_r.add(hit)
        coef_pairs.append((float(sol.coefficients[j]), float(r["coefficients"][hit]),
                           float(sol.standard_errors[j]), float(r["standard_errors"][hit])))
    bijection = bijection and len(matched_r) == len(r_cols) == len(py_cols)

    coef_rel = max((_rel(a, b) for a, b, _, _ in coef_pairs), default=float("nan"))
    se_rel = max((_rel(c, d) for _, _, c, d in coef_pairs), default=float("nan"))
    match = "yes" if (bijection and coef_rel < 1e-8) else "MISMATCH"
    brow = {
        "case": "factor_interaction", "n": n, "p": X.shape[1],
        "pystat_behavior": f"build_terms_design treatment contrasts ({X.shape[1]} cols)",
        "r_behavior": f"model.matrix default contrasts ({len(r_cols)} cols)",
        "match": match,
        "coef_max_rel": coef_rel,
        "detail": f"design bijection={'yes' if bijection else 'NO'}, se_max_rel={se_rel:.2e}",
    }
    brows.append(brow)
    records.append({"engine": "hardcase:factor", "dataset": "synthetic", "n": n,
                    "p": X.shape[1], "py_names": names, "r_names": r["coef_names"],
                    "bijection": bijection, "se_max_rel": se_rel, **brow})
    print(f"  factor_interaction: {match}  bijection={bijection} coef_rel={coef_rel:.2e}")


def _run_weights_offset(records: list, brows: list) -> None:
    """Prior weights / offset: pystatistics 4.2.4 fails loud (unsupported)."""
    from pystatistics.regression import fit

    n = 100
    rng = np.random.default_rng(11)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ np.array([1.0, 0.5]) + 0.2 * rng.standard_normal(n)
    for case, kw in (("weights", {"weights": np.ones(n)}),
                     ("offset", {"offset": np.zeros(n)})):
        try:
            fit(X, y, backend="cpu", **kw)
            outcome = "ACCEPTED"
        except TypeError as exc:
            outcome = f"TypeError: {str(exc)[:60]}"
        brow = {
            "case": case, "n": n, "p": 2,
            "pystat_behavior": "fail-loud (unsupported in fit() 4.2.4)",
            "r_behavior": f"supported via {case}=",
            "match": "documented-gap" if outcome.startswith("TypeError") else "ACCEPTED?!",
            "coef_max_rel": None,
            "detail": outcome,
        }
        brows.append(brow)
        records.append({"engine": f"hardcase:{case}", "dataset": "synthetic", "n": n,
                        "p": 2, **brow})
        print(f"  {case}: {brow['match']}  ({outcome})")


def _run_rank_deficient(records: list, brows: list) -> None:
    """Exactly-aliased column: pystatistics drops to NaN, matching R's NA."""
    from pystatistics.regression import fit

    n = 200
    rng = np.random.default_rng(5)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x1, x2, 2.0 * x1])     # col 3 ≡ 2·col 1
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + 0.1 * rng.standard_normal(n)

    sol = fit(X, y, backend="cpu", solver="qr")
    py_coef = list(sol.coefficients)
    py_nan = [i for i, c in enumerate(py_coef) if c is None or np.isnan(float(c))]

    r = _r_numeric(X[:, 1:], y, "lm")
    r_coef = r["coefficients"]
    r_nan = [i for i, c in enumerate(r_coef) if c is None]

    pos_match = py_nan == r_nan
    coef_rel = _vec_max_rel(py_coef, r_coef)
    match = "yes" if (pos_match and coef_rel < 1e-8) else "MISMATCH"
    brow = {
        "case": "rank_deficient", "n": n, "p": 4,
        "pystat_behavior": f"drops aliased col→NaN at index {py_nan}",
        "r_behavior": f"drops aliased col→NA at index {r_nan}",
        "match": match,
        "coef_max_rel": coef_rel,
        "detail": f"aliased-position match={'yes' if pos_match else 'NO'}",
    }
    brows.append(brow)
    records.append({"engine": "hardcase:rank_deficient", "dataset": "synthetic",
                    "n": n, "p": 4, "py_nan_idx": py_nan, "r_nan_idx": r_nan, **brow})
    print(f"  rank_deficient: {match}  py_nan={py_nan} r_nan={r_nan} coef_rel={coef_rel:.2e}")


def generate(host: str) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    collinear_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []

    print("collinear boundary sweep:")
    _run_collinear(records, collinear_rows)
    print("behaviour grid:")
    _run_separation(records, behavior_rows)
    _run_factor(records, behavior_rows)
    _run_weights_offset(records, behavior_rows)
    _run_rank_deficient(records, behavior_rows)

    config = {"study": "hardcases_r10",
              "scenarios": ["collinear", "separation_complete", "separation_quasi",
                            "factor_interaction", "weights", "offset", "rank_deficient"],
              "reference": "R lm()/glm()"}
    run = build_run(env=env, config=config, records=records)

    out_dir = _REPO / "artifacts" / "regression" / f"v{env['pystatistics_version']}" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = write_run(out_dir / f"hardcases_{host}.json", run)
    _write_csv(out_dir / f"hardcases_behavior_{host}_summary.csv", behavior_rows,
               ["case", "n", "p", "pystat_behavior", "r_behavior", "match",
                "coef_max_rel", "detail"])
    _write_csv(out_dir / f"hardcases_collinear_{host}_summary.csv", collinear_rows,
               ["np_cond", "lib_cond_fp32", "cpu_coef_max_rel_vs_r", "gpu_status",
                "gpu_refuse_reason", "gpu_coef_rel_vs_cpu", "gpu_fitted_rel_vs_cpu",
                "gpu_rss_rel_vs_cpu", "gpu_force_status"])
    print(f"\nwrote {run_path}")
    return run_path


def _write_csv(path: Path, rows: list[dict[str, Any]], cols: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=socket.gethostname().split(".")[0].lower())
    args = ap.parse_args()
    generate(args.host)


if __name__ == "__main__":
    main()
