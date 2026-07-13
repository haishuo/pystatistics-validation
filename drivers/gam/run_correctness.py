"""GAM correctness validation vs mgcv::gam (render-from-artifacts).

Emits artifacts/gam/v<ver>/runs/correctness.json — every number the report
shows. Covers:

* G1 correctness, two-tier contract:
    - TIER 1 (tight): at mgcv's selected sp fed to BOTH engines (cr smooths,
      whose penalty scaling is shared), fitted values / coefficients / EDF /
      scale / posterior SE — machine-ish agreement.
    - TIER 2 (optimizer-bounded): free selection, each engine picks its own
      sp — selected sp, per-smooth & total EDF, the GCV/UBRE/REML score.
* R15: the default invocation on mcycle.
* R10 hard cases: near-linear (lambda->inf), wiggly (lambda->0), concurvity,
  over-specified k, multi-smooth + parametric, cr AND tp, binomial AND
  poisson, Gamma.
* G2 fidelity: fail-loud on bs='cc', quasipoisson, REML+Gamma, k>n_unique,
  and the removed backend= kwarg.

Run:  MVNMLE_DATA_DIR=Dev/datasets python drivers/gam/run_correctness.py
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

import pystatistics
from pystatistics.gam import gam, s
from pystatistics.core.exceptions import ValidationError

import datasets as D  # local driver module

_HERE = Path(__file__).resolve().parent
_R_REF = _HERE / "r_reference.R"
_VER = pystatistics.__version__
_ARTIFACT = (
    _HERE.parents[1] / f"artifacts/gam/v{_VER}/runs/correctness.json"
)


# ---------------------------------------------------------------------------
# Case specification
# ---------------------------------------------------------------------------

@dataclass
class Smooth:
    var: str
    k: int = 10
    bs: str = "cr"


@dataclass
class Case:
    key: str
    why: str
    dataset: D.GamDataset
    smooths: list[Smooth]
    family: str = "gaussian"
    method: str = "GCV"
    parametric: list[str] = field(default_factory=list)  # parametric var names
    tier1: bool = True  # fixed-sp tight tier (cr-only designs)


def _r_family(fam: str, link: str | None) -> str:
    if fam == "Gamma" and link == "log":
        return "Gamma-log"
    return fam


def _formula(case: Case) -> str:
    terms = list(case.parametric)
    for sm in case.smooths:
        terms.append(f"s({sm.var},bs='{sm.bs}',k={sm.k})")
    return "y ~ " + " + ".join(terms)


def _run_r(case: Case, csv: Path, r_family: str,
           sp: list[float] | None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        spec = {
            "data_csv": str(csv), "response": "y",
            "formula": _formula(case), "family": r_family,
            "method": "GCV.Cp" if case.method == "GCV" else case.method,
        }
        if sp is not None:
            spec["sp"] = sp
        spec_path = Path(td) / "spec.json"
        out_path = Path(td) / "out.json"
        spec_path.write_text(json.dumps(spec))
        proc = subprocess.run(
            ["Rscript", str(_R_REF), str(spec_path), str(out_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"R failed for {case.key}: {proc.stderr[:400]}")
        return json.loads(out_path.read_text())


def _fit_py(case: Case, sp: list[float] | None):
    ds = case.dataset
    fam_obj: Any = case.family
    if case.family == "Gamma-log":
        from pystatistics.regression.families import resolve_family
        fam_obj = type(resolve_family("Gamma"))(link="log")
    X = None
    if case.parametric:
        cols = [np.ones(ds.y.size)] + [ds.parametric[v] for v in case.parametric]
        X = np.column_stack(cols)
    smooths = [s(sm.var, k=sm.k, bs=sm.bs) for sm in case.smooths]
    kw: dict[str, Any] = dict(
        smooths=smooths, smooth_data=ds.smooth_data, family=fam_obj,
        method=case.method,
    )
    if X is not None:
        kw["X"] = X
    if sp is not None:
        kw["sp"] = sp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gam(ds.y, **kw)


def _cmp(a: float, b: float) -> dict[str, float]:
    a, b = float(a), float(b)
    return {"py": a, "r": b, "abs": abs(a - b),
            "rel": abs(a - b) / max(abs(b), 1e-30)}


def _vec_cmp(a, b) -> dict[str, float]:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    return {"max_abs": float(np.max(np.abs(a - b))),
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "max_rel": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))}


def run_case(case: Case) -> dict[str, Any]:
    ds = case.dataset
    r_family = _r_family(case.family, "log" if case.family == "Gamma-log" else None)
    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "data.csv"
        D.write_case_csv(ds, csv)

        # --- TIER 2: free selection ---
        r_free = _run_r(case, csv, r_family, sp=None)
        py_free = _fit_py(case, sp=None)
        all_cr = all(sm.bs == "cr" for sm in case.smooths)
        rec: dict[str, Any] = {
            "key": case.key, "why": case.why, "family": case.family,
            "method": case.method, "n": r_free["n"],
            "all_cr": all_cr,
            "n_coef": {"py": int(py_free.coefficients.shape[0]),
                       "r": int(r_free["n_coef"])},
            "tier2_free": {
                "total_edf": _cmp(py_free.total_edf, r_free["total_edf"]),
                "smooth_edf": _vec_cmp(py_free.edf, r_free["smooth_edf"]),
                "score": _cmp(
                    py_free.reml_score if case.method == "REML"
                    else (py_free.ubre if py_free.params.dispersion_fixed
                          else py_free.gcv),
                    r_free["gcv_ubre"]),
                "deviance": _cmp(py_free.deviance, r_free["deviance"]),
                "fitted": _vec_cmp(py_free.fitted_values, r_free["fitted"]),
            },
        }
        # sp comparison only meaningful for all-cr designs. Report it
        # PER SMOOTH alongside that smooth's EDF: for a null / near-linear
        # term lambda -> inf is only weakly identified, so a large sp
        # ratio there is expected and benign iff the EDF (and fit) agree.
        if all_cr:
            rec["tier2_free"]["sp"] = _vec_cmp(py_free.lambdas, r_free["sp"])
            py_sp = np.atleast_1d(py_free.lambdas)
            r_sp = np.atleast_1d(r_free["sp"])
            py_e = np.atleast_1d(py_free.edf)
            r_e = np.atleast_1d(r_free["smooth_edf"])
            rec["tier2_free"]["per_smooth"] = [
                {"term": case.smooths[i].var,
                 "sp_py": float(py_sp[i]), "sp_r": float(r_sp[i]),
                 "sp_rel": abs(py_sp[i] - r_sp[i]) / max(abs(r_sp[i]), 1e-30),
                 "edf_py": float(py_e[i]), "edf_r": float(r_e[i]),
                 "edf_abs": abs(float(py_e[i]) - float(r_e[i])),
                 "near_null": bool(r_e[i] < 1.5)}
                for i in range(len(case.smooths))
            ]

        # --- TIER 1: fixed sp = mgcv's selected sp (cr designs only) ---
        if case.tier1 and all_cr:
            sp_mgcv = [float(v) for v in np.atleast_1d(r_free["sp"])]
            r_fix = _run_r(case, csv, r_family, sp=sp_mgcv)
            py_fix = _fit_py(case, sp=sp_mgcv)
            py_se = py_fix.standard_errors
            # mgcv posterior SE for the parametric block
            r_pse = np.atleast_1d(r_fix["p_se"])
            n_par = 1 + len(case.parametric)
            rec["tier1_fixedsp"] = {
                "sp": sp_mgcv,
                "fitted": _vec_cmp(py_fix.fitted_values, r_fix["fitted"]),
                "total_edf": _cmp(py_fix.total_edf, r_fix["total_edf"]),
                "scale": _cmp(py_fix.scale, r_fix["scale"]),
                "param_coef": _vec_cmp(py_fix.coefficients[:n_par],
                                       r_fix["p_coef"]),
                "param_se": _vec_cmp(py_se[:n_par], r_pse),
                "aic_abs": abs(float(py_fix.aic) - float(r_fix["aic"])),
                "smooth_stat_kind": r_fix["smooth_stat_kind"],
                "smooth_stat": _vec_cmp(
                    [si.chi_sq for si in py_fix.smooth_terms],
                    r_fix["smooth_stat"]),
            }
        return rec


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------

def _linear_dataset() -> D.GamDataset:
    rng = np.random.default_rng(101)
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    y = 2.0 + 3.0 * x + rng.normal(0, 0.3, n)
    return D.GamDataset("near_linear", "", y, {"x": x})


def _wiggly_dataset() -> D.GamDataset:
    rng = np.random.default_rng(102)
    n = 400
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * 6 * x) + rng.normal(0, 0.1, n)
    return D.GamDataset("wiggly", "", y, {"x": x})


def _concurvity_dataset() -> D.GamDataset:
    rng = np.random.default_rng(103)
    n = 400
    x1 = np.sort(rng.uniform(0, 1, n))
    x2 = x1 + rng.normal(0, 0.05, n)   # strongly correlated with x1
    y = np.sin(2 * np.pi * x1) + rng.normal(0, 0.3, n)
    return D.GamDataset("concurvity", "", y, {"x1": x1, "x2": x2})


def _parametric_dataset() -> D.GamDataset:
    rng = np.random.default_rng(104)
    n = 300
    x = np.sort(rng.uniform(0, 1, n))
    z = rng.normal(size=n)
    y = np.sin(2 * np.pi * x) + 1.5 * z + rng.normal(0, 0.3, n)
    ds = D.GamDataset("parametric_smooth", "", y, {"x": x})
    return D.GamDataset(ds.key, ds.why, y, {"x": x}, parametric={"z": z})


def build_cases() -> list[Case]:
    mcycle = D.load_mcycle()
    gsim_g = D.gamsim("normal", seed=2, n=400, scale=2.0)
    gsim_p = D.gamsim("poisson", seed=3, n=2000, scale=0.1)
    gsim_b = D.gamsim("binary", seed=4, n=600, scale=1.0)
    lin = _linear_dataset()
    wig = _wiggly_dataset()
    conc = _concurvity_dataset()
    par = _parametric_dataset()

    return [
        Case("R15_default_mcycle",
             "R15: the naive default call gam(y, smooths=[s('times')]) — "
             "default k=10, cr, GCV.", mcycle, [Smooth("times")]),
        Case("mcycle_reml", "Same fit under REML selection.",
             mcycle, [Smooth("times")], method="REML"),
        Case("mcycle_tp", "Default smooth on the tp basis (fitted/EDF tier "
             "only — tp sp units differ from mgcv).", mcycle,
             [Smooth("times", bs="tp")], tier1=False),
        Case("A3_cc_basis", "A3: cyclic cubic basis bs='cc' (seasonal/periodic "
             "smoother) vs mgcv — free-REML edf/fitted/score.", wig,
             [Smooth("x", k=10, bs="cc")], method="REML", tier1=False),
        Case("A3_ps_basis", "A3: P-spline basis bs='ps' (Eilers-Marx) vs mgcv — "
             "free-REML edf/fitted/score.", wig,
             [Smooth("x", k=10, bs="ps")], method="REML", tier1=False),
        Case("gamsim4_gaussian",
             "4-smooth Gaussian gamSim: recovers f0-f3 incl. nonlinear x2 "
             "and null x3.", gsim_g,
             [Smooth(f"x{i}") for i in range(4)]),
        Case("gamsim4_gaussian_reml", "4-smooth Gaussian under REML.",
             gsim_g, [Smooth(f"x{i}") for i in range(4)], method="REML"),
        Case("gamsim4_poisson", "4-smooth Poisson GAM (penalised IRLS, UBRE).",
             gsim_p, [Smooth(f"x{i}") for i in range(4)], family="poisson"),
        Case("gamsim4_binomial", "4-smooth Binomial GAM (penalised IRLS).",
             gsim_b, [Smooth(f"x{i}") for i in range(4)], family="binomial"),
        Case("R10_near_linear",
             "R10: near-linear truth -> lambda->inf, EDF->~1.", lin,
             [Smooth("x", k=10)]),
        Case("R10_wiggly",
             "R10: wiggly low-noise -> small lambda, EDF near k-1.", wig,
             [Smooth("x", k=20)]),
        Case("R10_concurvity",
             "R10: two strongly correlated smooths (GAM collinearity).", conc,
             [Smooth("x1"), Smooth("x2")]),
        Case("R10_overspecified_k",
             "R10: deliberately over-large k (k=30) the penalty must tame.",
             wig, [Smooth("x", k=30)]),
        Case("R10_parametric_plus_smooth",
             "R10: parametric covariate z + smooth s(x).", par,
             [Smooth("x")], parametric=["z"]),
    ]


# ---------------------------------------------------------------------------
# Fidelity (fail-loud) checks
# ---------------------------------------------------------------------------

def run_fidelity() -> list[dict[str, Any]]:
    rng = np.random.default_rng(7)
    n = 100
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    checks: list[dict[str, Any]] = []

    def record(name: str, why: str, fn, exc) -> None:
        try:
            fn()
            checks.append({"check": name, "why": why, "raised": None,
                           "pass": False})
        except exc as e:  # noqa: BLE001
            checks.append({"check": name, "why": why,
                           "raised": f"{type(e).__name__}: {str(e)[:100]}",
                           "pass": True})

    record("exotic_basis_re",
           "exotic bases (re/ds/gp/fs) are a declared scope carve-out -> fail "
           "loud on an unknown bs, no silent substitution (cc/ps ARE now "
           "implemented, validated in the correctness grid; re/ds/gp/fs raise)",
           lambda: s("x", bs="re"), ValidationError)
    record("quasipoisson_family",
           "quasipoisson not supported -> fail loud",
           lambda: gam(y, smooths=[s("x")], smooth_data={"x": x},
                       family="quasipoisson"), ValidationError)
    record("reml_gamma",
           "REML for free-dispersion Gamma -> fail loud, direct to GCV",
           lambda: gam(np.abs(y) + 1.0, smooths=[s("x")],
                       smooth_data={"x": x}, family="Gamma", method="REML"),
           ValidationError)
    record("k_exceeds_unique",
           "k > unique x values -> fail loud (mgcv parity)",
           lambda: gam(rng.normal(size=120),
                       smooths=[s("x", k=10)],
                       smooth_data={"x": np.repeat(np.linspace(0, 1, 6), 20)}),
           ValidationError)
    record("backend_removed",
           "GPU backend removed -> backend= raises TypeError (CPU-only module)",
           lambda: gam(y, smooths=[s("x")], smooth_data={"x": x},
                       backend="gpu"), TypeError)
    record("names_length_mismatch",
           "wrong-length names -> fail loud (would mislabel coefficients)",
           lambda: gam(y, np.column_stack([np.ones(n), x]),
                       smooths=[s("x")], smooth_data={"x": x},
                       names=["a", "b", "c"]), ValidationError)
    return checks


def main() -> None:
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["mgcv"] = subprocess.run(
        ["Rscript", "-e",
         'cat(as.character(packageVersion("mgcv")))'],
        capture_output=True, text=True).stdout.strip()

    records = [run_case(c) for c in build_cases()]
    fidelity = run_fidelity()

    run = build_run(
        env=env,
        config={"suite": "gam-correctness", "reference": "mgcv::gam",
                "tolerance_contract": "two-tier (fixed-sp tight / free "
                "optimizer-bounded)"},
        records=records + [{"key": "fidelity", "checks": fidelity}],
    )
    write_run(_ARTIFACT, run)
    print(f"wrote {_ARTIFACT}")
    # console summary
    for r in records:
        t2 = r["tier2_free"]
        line = (f"{r['key']:<28} edf {t2['total_edf']['abs']:.2e} "
                f"score {t2['score']['rel']:.2e} fitted {t2['fitted']['max_abs']:.2e}")
        if "sp" in t2:
            line += f" sp_rel {t2['sp']['max_rel']:.2e}"
        print(line)
    print("fidelity:", sum(c["pass"] for c in fidelity), "/", len(fidelity), "pass")


if __name__ == "__main__":
    main()
