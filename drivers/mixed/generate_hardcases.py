"""Generate the LMM R10 hard-case artifacts (the red-team grid).

Matching R on well-behaved textbook data proves little (RIGOR R10). This module
drives pystatistics ``mixed.lmm`` into the adversarial regime where R-agreement is
most likely to break AND where lme4 itself changes behaviour, and demands
pystatistics match the BEHAVIOUR — including R's singular-fit diagnostics — not
just the easy-data numbers:

  HC1  random-slope SINGULAR fit (the 2x2 RE-covariance boundary): data with no
       true slope variance, fit with (1 + x | g) -> slope variance / correlation
       collapse to the boundary; lme4 emits 'boundary (singular) fit'.
  HC2  UNBALANCED groups with singletons (group sizes 1..k): the small/odd-group
       regime that stresses the profiled deviance.
  HC3  extreme variance ratio (tiny residual, near-perfect ICC): numerical
       stability when between-group variance dwarfs within.
  HC4  the NESTING footgun: reused inner labels. Correct nested usage (form the
       batch:cask-style interaction id) must match lme4's nested fit; naive usage
       (pass reused inner labels directly) yields a CROSSED model — and lme4 does
       the SAME, so it is a matched footgun to document, not a silent deviation.

Plus the R15 default-invocation check: the bare ``lmm(y, X, groups=...)`` a naive
user writes (random_effects=None -> random intercept per group, REML, default
optimizer) must equal the explicit call and match R.

All hard-case designs are SYNTHETIC and DETERMINISTIC (seeded), generated
in-driver and handed to R via an ephemeral CSV — the central HDF5 store holds the
real reference datasets, not parameterised stress designs (the multivariate
precedent; R17).

    python -m drivers.mixed.generate_hardcases --host powerhouse
"""

from __future__ import annotations

import argparse
import csv
import socket
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

from drivers.mixed.datasets import MixedDataset
from drivers.mixed.run_pystatistics import run_lmm_record
from drivers.mixed.run_r_mixed import run_r_lmm_record

from artifact_root import artifact_root  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent.parent
_SEED = 20260630  # documented deterministic seed for every synthetic hard case


def _max_rel(a, b) -> float:
    a = np.asarray(a, float).ravel(); b = np.asarray(b, float).ravel()
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# --------------------------------------------------------------------------- #
# Synthetic designs
# --------------------------------------------------------------------------- #
def hc1_random_slope_singular() -> MixedDataset:
    """Intercept and slope PERFECTLY correlated by construction (b1 = c*b0), so
    the true 2x2 RE covariance is rank-1. Fitting (1 + x | g) drives the
    int/slope correlation estimate to +-1 -> a rank-deficient covariance ->
    lme4 'boundary (singular) fit'. The 2x2-boundary analogue of Dyestuff2."""
    rng = np.random.default_rng(_SEED)
    G, ni = 15, 8
    n = G * ni
    g = np.repeat(np.arange(G), ni)
    x = rng.standard_normal(n)
    b0 = rng.normal(0.0, 3.0, size=G)
    b1 = 0.5 * b0                               # perfect intercept/slope correlation
    y = 2.0 + 1.5 * x + b0[g] + b1[g] * x + rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    frame = pd.DataFrame({"y": y, "x": x, "g": g})
    return MixedDataset(
        key="hc1_rslope_singular",
        why="Intercept/slope perfectly correlated (rank-1 RE covariance); fitting "
            "(1 + x | g) drives the correlation estimate to +-1 -> singular 2x2 "
            "covariance -> lme4 'boundary (singular) fit'. The 2x2 analogue of "
            "the Dyestuff2 scalar boundary.",
        y=y, X=X, fixed_names=["(Intercept)", "x"],
        groups={"g": g}, random_effects={"g": ["1", "x"]}, random_data={"x": x},
        r_frame=frame, r_formula="y ~ x + (1 + x | g)", expect_singular=True)


def hc2_unbalanced_singletons() -> MixedDataset:
    """Random intercept with very unbalanced group sizes incl. singletons."""
    rng = np.random.default_rng(_SEED + 1)
    sizes = [1, 1, 2, 2, 3, 4, 5, 7, 9, 10, 1, 6]   # several singletons
    g = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n = g.size
    b0 = rng.normal(0.0, 2.0, size=len(sizes))
    y = 5.0 + b0[g] + rng.standard_normal(n)
    X = np.ones((n, 1))
    frame = pd.DataFrame({"y": y, "g": g})
    return MixedDataset(
        key="hc2_unbalanced",
        why="Highly unbalanced groups (sizes 1..10, several singletons): the "
            "small/odd-group regime that stresses the profiled deviance.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"g": g}, random_effects={"g": ["1"]}, random_data={},
        r_frame=frame, r_formula="y ~ 1 + (1 | g)")


def hc3_extreme_variance_ratio() -> MixedDataset:
    """Tiny residual variance, large between-group variance (near-perfect ICC)."""
    rng = np.random.default_rng(_SEED + 2)
    G, ni = 15, 6
    n = G * ni
    g = np.repeat(np.arange(G), ni)
    b0 = rng.normal(0.0, 10.0, size=G)
    y = 1.0 + b0[g] + rng.normal(0.0, 0.03, size=n)   # residual sd ~ 0.03
    X = np.ones((n, 1))
    frame = pd.DataFrame({"y": y, "g": g})
    return MixedDataset(
        key="hc3_extreme_ratio",
        why="Between-group variance dwarfs within (ICC ~ 0.9999): numerical "
            "stability of the profiled deviance at an extreme variance ratio.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"g": g}, random_effects={"g": ["1"]}, random_data={},
        r_frame=frame, r_formula="y ~ 1 + (1 | g)")


def _nested_dgp():
    """School/class data with REUSED inner labels (class 0..2 in every school)."""
    rng = np.random.default_rng(_SEED + 3)
    S, C, ni = 6, 3, 5
    school = np.repeat(np.arange(S), C * ni)
    cls_local = np.tile(np.repeat(np.arange(C), ni), S)         # reused 0..2
    cls_global = school * C + cls_local                         # unique batch:cask id
    n = school.size
    u = rng.normal(0.0, 4.0, size=S)
    v = rng.normal(0.0, 2.0, size=S * C)
    y = 10.0 + u[school] + v[cls_global] + rng.standard_normal(n)
    return school, cls_local, cls_global, y


def hc4_nested_correct() -> MixedDataset:
    """CORRECT nested usage: pass the globally-unique batch:cask-style inner id."""
    school, _cls_local, cls_global, y = _nested_dgp()
    n = y.size
    X = np.ones((n, 1))
    frame = pd.DataFrame({"y": y, "school": school, "sclass": cls_global})
    return MixedDataset(
        key="hc4_nested_correct",
        why="Nesting done right: the inner id is globally unique (batch:cask "
            "style), so (1|school)+(1|sclass) IS the nested model and must match "
            "lme4's nested fit.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"school": school, "sclass": cls_global},
        random_effects={"school": ["1"], "sclass": ["1"]}, random_data={},
        r_frame=frame, r_formula="y ~ 1 + (1 | school) + (1 | sclass)")


def hc4_nested_naive_crossed() -> MixedDataset:
    """FOOTGUN: pass reused inner labels directly -> a CROSSED (wrong) model.
    lme4 with (1|class) on the same reused labels does the SAME thing, so this
    documents a matched footgun, not a pystatistics-specific deviation."""
    school, cls_local, _cls_global, y = _nested_dgp()
    n = y.size
    X = np.ones((n, 1))
    frame = pd.DataFrame({"y": y, "school": school, "klass": cls_local})
    return MixedDataset(
        key="hc4_naive_crossed",
        why="The footgun: reused inner labels passed directly give a CROSSED "
            "model (class crossed with school), not nested. lme4 (1|school)+"
            "(1|klass) reproduces the SAME crossed answer -> matched behaviour, "
            "a usage caveat to document.",
        y=y, X=X, fixed_names=["(Intercept)"],
        groups={"school": school, "klass": cls_local},
        random_effects={"school": ["1"], "klass": ["1"]}, random_data={},
        r_frame=frame, r_formula="y ~ 1 + (1 | school) + (1 | klass)")


_HARD_CASES = [
    hc1_random_slope_singular, hc2_unbalanced_singletons,
    hc3_extreme_variance_ratio, hc4_nested_correct, hc4_nested_naive_crossed,
]


def _row(sut, ref, ds) -> dict[str, Any]:
    # variance components paired by (group,name); compare away-from-boundary
    rd = {(d["group"], d["name"]): d for d in ref["var_components"]}
    rel = []
    for d in sut["var_components"]:
        k = (d["group"], d["name"])
        if k in rd and abs(rd[k]["variance"]) > 1e-6:
            rel.append(abs(d["variance"] - rd[k]["variance"]) / abs(rd[k]["variance"]))
    return {
        "case": ds.key,
        "n": sut.get("n"), "p": sut.get("p"),
        "coef_max_rel": _max_rel(sut["coefficients"], ref["coefficients"]),
        "loglik_rel": _max_rel([sut["log_likelihood"]], [ref["log_likelihood"]]),
        "se_max_rel": _max_rel(sut["standard_errors"], ref["standard_errors"]),
        "varcomp_max_rel": (max(rel) if rel else 0.0),
        "py_singular": sut.get("is_singular"),
        "r_singular": ref.get("is_singular"),
        "expect_singular": ds.expect_singular,
        "r_diagnostics": " | ".join(ref.get("r_warnings", []) or []),
        "py_converged": sut.get("converged"),
    }


def _r15_default_check() -> dict[str, Any]:
    """R15: the bare lmm(y, X, groups=...) call (all defaults) must equal the
    explicit random_effects spec AND match R's default lmer."""
    from drivers.mixed.datasets import load_dyestuff
    from pystatistics.mixed import lmm
    ds = load_dyestuff()
    explicit = lmm(ds.y, ds.X, groups=ds.groups,
                   random_effects=ds.random_effects, random_data=ds.random_data)
    default = lmm(ds.y, ds.X, groups=ds.groups)   # random_effects=None default
    coef_eq = _max_rel(default.coefficients, explicit.coefficients)
    vc_eq = _max_rel([v.variance for v in default.var_components],
                     [v.variance for v in explicit.var_components])
    ref, _ = run_r_lmm_record(ds, reps=1)
    coef_vs_r = _max_rel(default.coefficients, ref["coefficients"])
    return {
        "case": "R15_default_invocation",
        "default_vs_explicit_coef_rel": coef_eq,
        "default_vs_explicit_varcomp_rel": vc_eq,
        "default_vs_r_coef_rel": coef_vs_r,
        "note": "bare lmm(y,X,groups) == explicit (1|g) spec and matches lmer",
    }


def generate(host: str, *, repeats: int) -> Path:
    env = env_manifest(device="cpu", host=host)
    require_pypi(env)

    records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for make in _HARD_CASES:
        ds = make()
        sut = run_lmm_record(ds, repeats=repeats, warmup=1)
        if sut.get("error"):
            raise RuntimeError(f"pystatistics LMM failed for {ds.key}: {sut['error']}")
        ref, _raw = run_r_lmm_record(ds, reps=repeats)
        records.extend([sut, ref])
        row = _row(sut, ref, ds)
        rows.append(row)
        print(f"  {ds.key:22s} coef={row['coef_max_rel']:.1e} ll={row['loglik_rel']:.1e} "
              f"vc={row['varcomp_max_rel']:.1e} sing py/r={row['py_singular']}/{row['r_singular']} "
              f"(expect={row['expect_singular']}) diag='{row['r_diagnostics']}'")

    r15 = _r15_default_check()
    print(f"  R15 default: default==explicit coef={r15['default_vs_explicit_coef_rel']:.1e} "
          f"vc={r15['default_vs_explicit_varcomp_rel']:.1e}  vs_R={r15['default_vs_r_coef_rel']:.1e}")

    config = {
        "study": "hardcases_r10",
        "cases": [c.__name__ for c in _HARD_CASES] + ["R15_default_invocation"],
        "backend": "cpu", "repeats": repeats, "seed": _SEED,
        "reference": "R lme4::lmer + lmerTest::lmer",
        "r15_default": r15,
    }
    run = build_run(env=env, config=config, records=records)
    out_dir = artifact_root(_REPO) / "mixed" / f"v{env['pystatistics_version']}" / "runs"
    run_path = write_run(out_dir / f"hardcases_cpu_{host}.json", run)
    _write_summary_csv(out_dir / f"hardcases_{host}_summary.csv", rows)
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
