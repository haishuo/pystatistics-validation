"""Generate the GLMM R10 red-team + R15 default-invocation artifacts (CPU).

One job: exercise ``mixed.glmm`` on the ADVERSARIAL regime where agreement with
``lme4::glmer`` is most likely to break AND where R itself changes behaviour, and
record — for each case — whether pystatistics matches R's BEHAVIOUR (numbers where
both fit; the same singular/near-boundary variance; the same fail-loud vs fit
decision), not just easy-data numbers. Freezes a ``validation-run/v1`` artifact +
a flat summary CSV under ``artifacts/mixed/v<ver>/runs/``.

Cases (all constructed deterministically in-driver — red-team data is not fixed
reference data and is regenerated from a fixed seed, matching the LMM hardcases):

  hc1_separation      binomial/logit with (quasi-)complete separation in a fixed
                      predictor: glmer warns + diverges; glmm must not silently
                      return a tidy finite fit — match the behaviour.
  hc2_singular_re     binomial with a near-zero true random-effect variance:
                      glmer's variance → boundary (isSingular); record whether
                      glmm's variance collapses to the same boundary.
  hc3_unbalanced      binomial with singleton / very unequal groups: both fit;
                      agreement to the two-tier contract.
  hc4_default_r15     the bare glmm(y, X, groups, family='binomial') a naive user
                      writes (random_effects=None) MUST equal the explicit
                      (1|group) spec and match glmer's default (R15).
  hc5_gaussian_loud   glmm(family='gaussian') must FAIL LOUD (dispersion≠1), not
                      return silently-wrong fit statistics (A6).
  hc6_gamma_loud      glmm(family='gamma') must FAIL LOUD likewise.

PyPI-only (``require_pypi``).

    python -m drivers.mixed.generate_glmm_hardcases --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run

# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
# Drivers hardcode their artifact dir, so an ordinary run would otherwise
# silently destroy the artifacts a report was blessed against.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402

from drivers.mixed.glmm_datasets import GLMMDataset
from drivers.mixed.run_glmm import run_glmm_record
from drivers.mixed.run_r_glmm import run_r_glmm_record

from artifact_root import artifact_root  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent


def _logit_ds(key, why, y, x, g, expect_singular=False) -> GLMMDataset:
    """A one-covariate binomial/logit random-intercept dataset from arrays."""
    X = np.column_stack([np.ones(y.size), x])
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return GLMMDataset(
        key=key, why=why, y=y.astype(float), X=X,
        fixed_names=["(Intercept)", "x"],
        groups={"g": g}, random_effects={"g": ["1"]}, random_data={},
        family="binomial", link="logit",
        r_frame=frame, r_formula="y ~ x + (1 | g)",
        r_family="binomial", r_link="logit",
        r_factor_cols=("g",), expect_singular=expect_singular)


def _case_separation() -> GLMMDataset:
    rng = np.random.default_rng(101)
    G, per = 12, 12
    n = G * per
    g = np.repeat(np.arange(G), per)
    x = rng.normal(0, 1, n)
    # QUASI-separation: y ≈ step(x) but a few points near the boundary overlap,
    # so the MLE slope is huge-but-finite and glmer converges WITH a warning
    # (complete separation makes glmer error outright — a less comparable case).
    y = (x > 0).astype(float)
    flip = rng.choice(np.where(np.abs(x) < 0.5)[0], size=4, replace=False)
    y[flip] = 1.0 - y[flip]
    return _logit_ds("hc1_quasi_separation",
                     "Quasi-separation in the fixed predictor — the slope MLE is "
                     "huge but finite and glmer converges with a warning; glmm must "
                     "match the behaviour (large coefficient), not a tidy small fit.",
                     y, x, g)


def _case_singular_re() -> GLMMDataset:
    rng = np.random.default_rng(202)
    G, per = 20, 15
    n = G * per
    g = np.repeat(np.arange(G), per)
    x = rng.normal(0, 1, n)
    # TRUE random-effect variance = 0 (no group signal): glmer -> isSingular.
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.1 + 0.8 * x)))).astype(float)
    return _logit_ds("hc2_singular_re",
                     "No true group signal — the random-effect variance is at the "
                     "boundary; glmer flags isSingular. Record whether glmm's "
                     "variance collapses to the same boundary.",
                     y, x, g, expect_singular=True)


def _case_unbalanced() -> GLMMDataset:
    rng = np.random.default_rng(303)
    # very unequal group sizes incl singletons
    sizes = [1, 1, 2, 3, 5, 8, 13, 21, 34, 40]
    g = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n = g.size
    b = rng.normal(0, 0.8, len(sizes))
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.2 + 0.6 * x + b[g])))).astype(float)
    return _logit_ds("hc3_unbalanced",
                     "Highly unbalanced groups with singletons — the small/unequal "
                     "group regime; both engines fit, agreement to the two-tier "
                     "contract.", y, x, g)


def _agreement_row(ds, sut, ref) -> dict[str, Any]:
    def _mr(a, b):
        a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
        return None if a.shape != b.shape or a.size == 0 else \
            float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    py_var = (sut["var_components"][0]["variance"]
              if sut.get("var_components") else None)
    r_var = (ref["var_components"][0]["variance"]
             if ref.get("var_components") else None)
    return {
        "case": ds.key,
        "n": ds.n,
        "coef_max_rel": _mr(sut.get("coefficients"), ref.get("coefficients")),
        "py_var": py_var,
        "r_var": r_var,
        "py_converged": sut.get("converged"),
        "py_warned": bool(sut.get("py_warnings")),
        "r_singular": ref.get("is_singular"),
        "r_warned": bool(ref.get("r_warnings")),
        "expect_singular": ds.expect_singular,
    }


def _failloud_row(case: str, family: str, sut: dict[str, Any]) -> dict[str, Any]:
    err = sut.get("error") or ""
    return {
        "case": case, "n": sut.get("n"),
        "coef_max_rel": None, "py_var": None, "r_var": None,
        "py_converged": None,
        "py_warned": None, "r_singular": None, "r_warned": None,
        "expect_singular": None,
        "failed_loud": bool(err) and "ValidationError" in err,
        "error": err[:120],
    }


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    # --- behaviour-match cases (fit both engines) ---
    for ds in (_case_separation(), _case_singular_re(), _case_unbalanced()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sut = run_glmm_record(ds, repeats=repeats, warmup=1)
        try:
            ref, _raw = run_r_glmm_record(ds, reps=repeats)
        except RuntimeError as exc:
            # glmer can ERROR OUT on a degenerate likelihood (e.g. it refuses a
            # non-converged PIRLS). That refusal IS R's behaviour — record it
            # rather than crashing the sweep.
            records.append(sut)
            rows.append({
                "case": ds.key, "n": ds.n, "coef_max_rel": None,
                "py_var": (sut["var_components"][0]["variance"]
                           if sut.get("var_components") else None),
                "r_var": None, "py_converged": sut.get("converged"),
                "py_warned": bool(sut.get("py_warnings")),
                "r_singular": None, "r_warned": True,
                "expect_singular": ds.expect_singular,
                "r_failed": True, "error": str(exc)[:120]})
            print(f"  {ds.key:18s} R refused (degenerate); py_converged="
                  f"{sut.get('converged')}")
            continue
        records.extend([sut, ref])
        row = _agreement_row(ds, sut, ref)
        rows.append(row)
        print(f"  {ds.key:18s} coef={row['coef_max_rel']} "
              f"py_var={row['py_var']:.3g} r_var={row['r_var']:.3g} "
              f"r_sing={row['r_singular']}")

    # --- R15 default-invocation: bare default == explicit (1|group) == glmer ---
    row_default, rec_default = _default_invocation_case(repeats)
    records.extend(rec_default)
    rows.append(row_default)
    print(f"  hc4_default_r15    default_vs_explicit={row_default['coef_max_rel']:.1e} "
          f"vs_glmer={row_default['vs_glmer_coef_rel']:.1e}")

    # --- fail-loud cases (A6): gaussian / gamma must raise ---
    for case, fam in (("hc5_gaussian_loud", "gaussian"), ("hc6_gamma_loud", "gamma")):
        ds = _failloud_ds(fam)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sut = run_glmm_record(ds, repeats=1, warmup=0)
        records.append(sut)
        row = _failloud_row(case, fam, sut)
        rows.append(row)
        print(f"  {case:18s} failed_loud={row['failed_loud']}")

    config = {
        "study": "glmm_hardcases_r10_r15",
        "backend": "cpu",
        "repeats": repeats,
        "reference": "R lme4::glmer (Laplace, nAGQ=1)",
        "note": "Adversarial R10 grid + R15 default invocation + A6 fail-loud "
                "families. Behaviour match, not just easy-data numbers.",
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = artifact_root(_REPO) / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"glmm_hardcases_cpu_{host}.json", run)
    _write_summary_csv(out_dir / f"glmm_hardcases_{host}_summary.csv", rows)
    print(f"\nwrote {run_path}")
    return run_path


def _default_invocation_case(repeats: int):
    """R15: bare glmm(y, X, groups, family='binomial') (random_effects=None) must
    equal the explicit (1|group) spec AND match glmer's default."""
    from pystatistics.mixed import glmm
    rng = np.random.default_rng(404)
    G, per = 25, 16
    n = G * per
    g = np.repeat(np.arange(G), per)
    b = rng.normal(0, 0.7, G)
    x = rng.normal(0, 1, n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * x + b[g])))).astype(float)
    X = np.column_stack([np.ones(n), x])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = glmm(y, X, groups={"g": g}, family="binomial")
        explicit = glmm(y, X, groups={"g": g}, family="binomial",
                        random_effects={"g": ["1"]})
    ds = _logit_ds("hc4_default_r15", "R15 default invocation", y, x, g)
    ref, _ = run_r_glmm_record(ds, reps=repeats)
    d_vs_e = float(np.max(np.abs(np.asarray(default.coefficients)
                                 - np.asarray(explicit.coefficients))))
    vs_glmer = float(np.max(np.abs(np.asarray(default.coefficients)
                                   - np.asarray(ref["coefficients"]))
                            / np.maximum(np.abs(ref["coefficients"]), 1e-300)))
    row = {
        "case": "hc4_default_r15", "n": n,
        "coef_max_rel": d_vs_e,           # default vs explicit (should be ~0)
        "vs_glmer_coef_rel": vs_glmer,    # default vs glmer default
        "py_var": float(default.var_components[0].variance),
        "r_var": ref["var_components"][0]["variance"],
        "py_converged": bool(default.converged),
        "py_warned": None, "r_singular": ref.get("is_singular"),
        "r_warned": bool(ref.get("r_warnings")), "expect_singular": False,
    }
    return row, [ref]


def _failloud_ds(family: str) -> GLMMDataset:
    rng = np.random.default_rng(505)
    G, per = 10, 12
    n = G * per
    g = np.repeat(np.arange(G), per)
    x = rng.normal(0, 1, n)
    y = np.abs(rng.normal(2.0, 1.0, n)) + 0.1  # positive (valid for gamma too)
    X = np.column_stack([np.ones(n), x])
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return GLMMDataset(
        key=f"failloud_{family}", why="fail-loud family", y=y, X=X,
        fixed_names=["(Intercept)", "x"], groups={"g": g},
        random_effects={"g": ["1"]}, random_data={},
        family=family, link=("identity" if family == "gaussian" else "inverse"),
        r_frame=frame, r_formula="y ~ x + (1 | g)",
        r_family=family, r_link="", r_factor_cols=("g",))


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
