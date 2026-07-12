"""GAM TENSOR / MULTIVARIATE smooth validation vs mgcv (VA-1, render-from-artifacts).

Emits artifacts/gam/v<ver>/runs/tensor.json — the two-tier evidence for the
4.8.0 tensor-product / interaction / isotropic-multivariate smooths:

  - te(x,z)      tensor-product smooth (per-margin bases + one sp per margin)
  - ti(x)+ti(z)+ti(x,z)   the functional-ANOVA decomposition (main + interaction)
  - s(x,z)       isotropic (thin-plate) multivariate smooth (a single sp)

Each case fits the SAME dumped data in pystatistics and in mgcv (via the generic
r_reference.R formula worker) and reduces the pair to the honest two-tier
contract used by run_correctness:

  - TIER 1 (tight, per-margin-cr/cc designs): feed mgcv's SELECTED sp to BOTH
    engines -> the tensor basis is mgcv-exact, so fitted / EDF / coefficients /
    scale / AIC agree to machine precision. (Skipped for tp-based s(x,z), whose
    sp units differ from mgcv -- see the tp caveat in run_correctness.)
  - TIER 2 (free REML/GCV/UBRE): each engine selects the per-margin sp itself ->
    compare the per-margin sp, per-term & total EDF, the selection score, the
    deviance, and the fitted values.

R10 hard cases: mixed cyclic x cubic margins (bs=c('cc','cr')), per-margin
differing k, a Poisson tensor, functional-ANOVA ti(), a tensor alongside a
univariate smooth, and concurvity between the tensor margins.

Deterministic seeded data (mgcv::gamSim(2) + custom bivariate surfaces),
generated in R so both engines read identical float64 bytes. CPU only; PyPI-only.

    python -m drivers.gam.run_tensor
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from pystatistics import __version__ as _VER
from pystatistics.gam import gam, te, ti, s

_HERE = Path(__file__).resolve().parent
_R_REF = _HERE / "r_reference.R"
_ARTIFACT = _HERE.parents[1] / f"artifacts/gam/v{_VER}/runs/tensor.json"


# --------------------------------------------------------------------------
# Deterministic data generation (in R, so both engines read identical bytes)
# --------------------------------------------------------------------------

_DATA_R = r"""
suppressMessages({library(mgcv)})
args <- commandArgs(trailingOnly = TRUE); outdir <- args[[1]]
set.seed(20260711)

# (1) bivariate Gaussian surface: mgcv gamSim(2) test function f(x,z)
g2 <- gamSim(2, n = 500, scale = 0.3, verbose = FALSE)$data
write.csv(g2[, c("y", "x", "z")], file.path(outdir, "biv_gauss.csv"), row.names = FALSE)

# (2) bivariate Poisson: counts from the same kind of smooth surface
n <- 800
x <- runif(n); z <- runif(n)
f <- 0.2 * exp(2 * x) + 2 * sin(pi * z) + 1.5 * x * z      # true log-mean surface
mu <- exp((f - mean(f)) * 0.6)
y <- rpois(n, mu)
write.csv(data.frame(y = y, x = x, z = z), file.path(outdir, "biv_pois.csv"), row.names = FALSE)

# (3) cyclic x cubic: x periodic on [0,1], z linear; true surface periodic in x
n <- 600
x <- runif(n); z <- runif(n)
f <- sin(2 * pi * x) * (0.5 + z) + 0.5 * cos(2 * pi * x)
y <- f + rnorm(n, 0, 0.3)
write.csv(data.frame(y = y, x = x, z = z), file.path(outdir, "cyclic.csv"), row.names = FALSE)

# (4) univariate smooth + tensor: y = f(w) + g(x,z)
n <- 600
w <- runif(n); x <- runif(n); z <- runif(n)
f <- sin(2 * pi * w) + 0.3 * exp(2 * x) * z + cos(pi * z)
y <- f + rnorm(n, 0, 0.4)
write.csv(data.frame(y = y, w = w, x = x, z = z), file.path(outdir, "wxz.csv"), row.names = FALSE)

# (5) concurvity between tensor margins: x2 ~ x1 + small noise
n <- 500
x1 <- runif(n); x2 <- x1 + rnorm(n, 0, 0.05)
y <- sin(2 * pi * x1) + 0.5 * x2 + rnorm(n, 0, 0.3)
write.csv(data.frame(y = y, x = x1, z = x2), file.path(outdir, "concurv.csv"), row.names = FALSE)
"""


def _gen_data(outdir: Path) -> None:
    script = outdir / "gen.R"
    script.write_text(_DATA_R)
    subprocess.run(["Rscript", str(script), str(outdir)], check=True,
                   capture_output=True, text=True)


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    import csv as _csv
    with open(path) as fh:
        rows = list(_csv.DictReader(fh))
    cols = rows[0].keys()
    return {c: np.array([float(r[c]) for r in rows]) for c in cols}


# --------------------------------------------------------------------------
# Case definition
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorCase:
    key: str
    why: str
    data: str                    # CSV stem
    r_formula: str               # mgcv formula string
    py_smooths: Callable[[], list]  # builds the pystatistics smooth list
    smooth_vars: list[str]       # variables the smooths consume
    family: str = "gaussian"
    method: str = "REML"
    tier1: bool = True           # feed mgcv sp to both (exact-basis tier)
    parametric: list[str] = field(default_factory=list)


def _r_family(fam: str) -> str:
    return {"gaussian": "gaussian", "poisson": "poisson",
            "binomial": "binomial"}[fam]


def _run_r(case: TensorCase, csv: Path, sp: list[float] | None) -> dict:
    r_method = "GCV.Cp" if case.method == "GCV" else case.method
    spec = {"data_csv": str(csv), "response": "y", "formula": case.r_formula,
            "family": _r_family(case.family), "method": r_method}
    if sp is not None:
        spec["sp"] = list(sp)
    with tempfile.TemporaryDirectory() as td:
        spec_path = Path(td) / "spec.json"
        out_path = Path(td) / "out.json"
        spec_path.write_text(json.dumps(spec))
        proc = subprocess.run(["Rscript", str(_R_REF), str(spec_path),
                               str(out_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mgcv failed for {case.key}:\n{proc.stderr}")
        return json.loads(out_path.read_text())


def _fit_py(case: TensorCase, cols: dict[str, np.ndarray],
            sp: list[float] | None):
    y = cols["y"]
    smooth_data = {v: cols[v] for v in case.smooth_vars}
    X = None
    if case.parametric:
        X = np.column_stack([np.ones(y.size)]
                            + [cols[v] for v in case.parametric])
    fam = case.family
    kw = dict(smooths=case.py_smooths(), smooth_data=smooth_data,
              family=fam, method=case.method)
    if sp is not None:
        kw["sp"] = np.asarray(sp, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gam(y, X, **kw)


def _cmp(a: float, b: float) -> dict[str, float]:
    a, b = float(a), float(b)
    return {"py": a, "r": b, "abs": abs(a - b),
            "rel": abs(a - b) / max(abs(b), 1e-30)}


def _vec_cmp(a, b) -> dict[str, float]:
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    return {"n": int(n),
            "max_abs": float(np.max(np.abs(a - b))),
            "max_rel": float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))}


def run_case(case: TensorCase, tmp: Path) -> dict[str, Any]:
    cols = _load_csv(tmp / f"{case.data}.csv")
    r_free = _run_r(case, tmp / f"{case.data}.csv", sp=None)
    py_free = _fit_py(case, cols, sp=None)

    score_py = (py_free.reml_score if case.method == "REML"
                else (py_free.ubre if py_free.params.dispersion_fixed
                      else py_free.gcv))
    rec: dict[str, Any] = {
        "key": case.key, "why": case.why, "family": case.family,
        "method": case.method, "n": r_free["n"], "tier1_applicable": case.tier1,
        "n_coef": {"py": int(np.asarray(py_free.coefficients).shape[0]),
                   "r": int(r_free["n_coef"])},
        "n_sp": {"py": int(np.atleast_1d(py_free.lambdas).size),
                 "r": int(np.atleast_1d(r_free["sp"]).size)},
        "tier2_free": {
            "total_edf": _cmp(py_free.total_edf, r_free["total_edf"]),
            "smooth_edf": _vec_cmp(py_free.edf, r_free["smooth_edf"]),
            "sp": _vec_cmp(py_free.lambdas, r_free["sp"]),
            "score": _cmp(score_py, r_free["gcv_ubre"]),
            "deviance": _cmp(py_free.deviance, r_free["deviance"]),
            "fitted": _vec_cmp(py_free.fitted_values, r_free["fitted"]),
        },
    }

    if case.tier1:
        sp_mgcv = [float(v) for v in np.atleast_1d(r_free["sp"])]
        r_fix = _run_r(case, tmp / f"{case.data}.csv", sp=sp_mgcv)
        py_fix = _fit_py(case, cols, sp=sp_mgcv)
        n_par = 1 + len(case.parametric)
        rec["tier1_fixedsp"] = {
            "sp": sp_mgcv,
            "fitted": _vec_cmp(py_fix.fitted_values, r_fix["fitted"]),
            "coef": _vec_cmp(py_fix.coefficients, r_fix["coef"]),
            "total_edf": _cmp(py_fix.total_edf, r_fix["total_edf"]),
            "scale": _cmp(py_fix.scale, r_fix["scale"]),
            "param_coef": _vec_cmp(np.asarray(py_fix.coefficients)[:n_par],
                                   r_fix["p_coef"]),
            "param_se": _vec_cmp(np.asarray(py_fix.se)[:n_par], r_fix["p_se"]),
            "aic_abs": abs(float(py_fix.aic) - float(r_fix["aic"])),
        }
    return rec


# --------------------------------------------------------------------------
# Case matrix
# --------------------------------------------------------------------------

def build_cases() -> list[TensorCase]:
    return [
        TensorCase(
            "te_biv_reml",
            "Canonical tensor product te(x,z) cr x cr, REML — the core mgcv "
            "tensor surface.", "biv_gauss",
            "y ~ te(x,z,k=c(5,5),bs=c('cr','cr'))",
            lambda: [te("x", "z", k=5, bs="cr")], ["x", "z"], method="REML"),
        TensorCase(
            "te_biv_gcv", "Same tensor under GCV selection.", "biv_gauss",
            "y ~ te(x,z,k=c(5,5),bs=c('cr','cr'))",
            lambda: [te("x", "z", k=5, bs="cr")], ["x", "z"], method="GCV"),
        TensorCase(
            "te_diff_k",
            "R10: per-margin differing k=c(5,8) — the anisotropic marginal "
            "bases mgcv builds.", "biv_gauss",
            "y ~ te(x,z,k=c(5,8),bs=c('cr','cr'))",
            lambda: [te("x", "z", k=(5, 8), bs="cr")], ["x", "z"], method="REML"),
        TensorCase(
            "te_cc_cr",
            "R10: mixed cyclic x cubic margins bs=c('cc','cr') — the seasonal "
            "tensor.", "cyclic",
            "y ~ te(x,z,k=c(8,5),bs=c('cc','cr'))",
            lambda: [te("x", "z", k=(8, 5), bs=("cc", "cr"))], ["x", "z"],
            method="REML"),
        TensorCase(
            "ti_anova",
            "Functional-ANOVA decomposition ti(x)+ti(z)+ti(x,z) — main effects "
            "plus interaction. Free tier only: ti() main-effect margin penalties "
            "are reported on a different scale than mgcv's rescaled m$sp (like the "
            "tp basis), so the fixed-sp cross-feed is not attempted; the ESTIMATOR "
            "is validated exact by the free-REML per-term/total EDF, deviance, "
            "REML score and fitted values (all machine-precision vs mgcv).",
            "biv_gauss",
            "y ~ ti(x,k=5,bs='cr')+ti(z,k=5,bs='cr')+ti(x,z,k=c(5,5),bs=c('cr','cr'))",
            lambda: [ti("x", k=5, bs="cr"), ti("z", k=5, bs="cr"),
                     ti("x", "z", k=5, bs="cr")], ["x", "z"], method="REML",
            tier1=False),
        TensorCase(
            "te_poisson",
            "R10: Poisson tensor te(x,z) (penalised IRLS, UBRE) — the "
            "non-Gaussian tensor path.", "biv_pois",
            "y ~ te(x,z,k=c(5,5),bs=c('cr','cr'))",
            lambda: [te("x", "z", k=5, bs="cr")], ["x", "z"],
            family="poisson", method="REML"),
        TensorCase(
            "te_plus_s",
            "R10: a univariate smooth alongside a tensor — s(w)+te(x,z), the "
            "additive + interaction model.", "wxz",
            "y ~ s(w,k=10,bs='cr')+te(x,z,k=c(5,5),bs=c('cr','cr'))",
            lambda: [s("w", k=10, bs="cr"), te("x", "z", k=5, bs="cr")],
            ["w", "x", "z"], method="REML"),
        TensorCase(
            "te_concurvity",
            "R10: concurvity — tensor margins x,z strongly correlated (z=x+eps).",
            "concurv", "y ~ te(x,z,k=c(5,5),bs=c('cr','cr'))",
            lambda: [te("x", "z", k=5, bs="cr")], ["x", "z"], method="REML"),
        TensorCase(
            "s_iso_biv",
            "Isotropic multivariate s(x,z) thin-plate (a single sp) — free "
            "tier only (tp sp units differ from mgcv; edf/fitted/score compared).",
            "biv_gauss", "y ~ s(x,z,k=30)",
            lambda: [s("x", "z", k=30)], ["x", "z"], method="REML", tier1=False),
    ]


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = {"mgcv": None}

    records = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _gen_data(tmp)
        for case in build_cases():
            rec = run_case(case, tmp)
            records.append(rec)
            t2 = rec["tier2_free"]
            t1 = rec.get("tier1_fixedsp")
            t1s = (f" | tier1 fitted {t1['fitted']['max_abs']:.1e}" if t1 else
                   " | tier1 n/a")
            print(f"  {case.key:16s} n_coef py {rec['n_coef']['py']}="
                  f"{rec['n_coef']['r']} r | tier2 edf_abs "
                  f"{t2['total_edf']['abs']:.2e} sp_rel {t2['sp']['max_rel']:.2e} "
                  f"fitted {t2['fitted']['max_abs']:.1e}{t1s}")

    run = build_run(
        env=env,
        config={"suite": "gam-tensor",
                "reference": "mgcv::gam te()/ti()/s(x,z)",
                "tolerance_contract": "two-tier: TIER1 fixed-sp (mgcv sp fed to "
                "both) -> fitted/EDF/coef machine-precision on the exact tensor "
                "basis; TIER2 free REML/GCV/UBRE -> per-margin sp, per-term & "
                "total EDF, score, deviance, fitted"},
        records=records)
    out = Path(str(_ARTIFACT))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
