"""VA-2 by= / VA-3 nb() gam validation vs mgcv (render-from-artifacts).

Emits artifacts/gam/v<ver>/runs/by_nb.json — the two-tier evidence for the two
gam surfaces that shipped in 4.7.0 with no rendered validation report:

  - VA-2  s(x, by=z)   continuous varying-coefficient smooth (by * f(x))
  - VA-3  family='nb'  negative-binomial GAM with estimated dispersion theta

Contract, per surface:

  VA-2 (Gaussian by=), two-tier like run_tensor:
    - TIER 1  feed mgcv's SELECTED sp to BOTH engines -> the by-multiplied cr
      basis + penalty is mgcv-exact, so fitted / EDF / coef / scale agree to
      machine precision. (Unlike the tp/ti margins, the by= sp is on the SAME
      scale as mgcv's -- established here, not assumed: the same sp fed to both
      yields the identical fit.)
    - TIER 2  free REML -> per-smooth & total EDF, the selected sp (directly
      comparable for cr by=), the REML score, deviance and fitted values.

  VA-3 (nb), free REML only (nb REML is the supported path; GCV fails loud):
    - total & per-smooth EDF, deviance, REML score, fitted values, and the
      estimated dispersion THETA vs mgcv's m$family$getTheta(TRUE). pystatistics
      exposes NO theta accessor (family_name is the bare string
      'negative.binomial'), so py's effective theta is RECOVERED by inverting the
      NB deviance at the fitted mean -- see finding VA3-F1 in the report.

R10 hard cases: by= with negative values and a zero-variance stretch; by=
alongside a second smooth; a large-magnitude by-variable (sp-scale probe); nb
with small counts, with a large theta near the Poisson limit, and the
fail-loud nb+GCV guard. Plus the factor-by fidelity probe (finding VA2-F1):
a categorical by-column is silently taken as continuous (no per-level smooths,
no warning) -- recorded as a fail-loud gap.

Deterministic seeded data generated in R so both engines read identical float64
bytes. CPU only; PyPI-only.

    python -m drivers.gam.run_by_nb
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
from scipy.optimize import brentq

from pystatsval.device import env_manifest, require_pypi
from pystatsval.serialize import build_run, write_run

from pystatistics import __version__ as _VER
from pystatistics.gam import gam, s
from pystatistics.core.exceptions import ValidationError

_HERE = Path(__file__).resolve().parent
_ARTIFACT = _HERE.parents[1] / f"artifacts/gam/v{_VER}/runs/by_nb.json"


# --------------------------------------------------------------------------
# Deterministic data generation (in R, so both engines read identical bytes)
# --------------------------------------------------------------------------

_DATA_R = r"""
suppressMessages({library(mgcv)})
outdir <- commandArgs(trailingOnly = TRUE)[[1]]

# (1) continuous by-variable, positive-ish: y = z * f(x) + eps
set.seed(42)
n <- 400; x <- runif(n); z <- rnorm(n, 2, 1)
y <- z * sin(2 * pi * x) + rnorm(n, 0, 0.3)
write.csv(data.frame(y = y, x = x, z = z),
          file.path(outdir, "by_cont.csv"), row.names = FALSE)

# (2) R10: by-variable with NEGATIVE values and a zero-variance stretch
set.seed(11)
n <- 400; x <- runif(n); z <- runif(n, -3, 3)
z[1:40] <- 0                                   # a flat zero stretch
y <- z * cos(2 * pi * x) + rnorm(n, 0, 0.3)
write.csv(data.frame(y = y, x = x, z = z),
          file.path(outdir, "by_neg.csv"), row.names = FALSE)

# (3) R10: by= alongside a SECOND ordinary smooth: y = z*f(x) + g(w)
set.seed(99)
n <- 500; x <- runif(n); w <- runif(n); z <- rnorm(n, 1.5, 0.8)
y <- z * sin(2 * pi * x) + exp(1.2 * w) + rnorm(n, 0, 0.4)
write.csv(data.frame(y = y, x = x, w = w, z = z),
          file.path(outdir, "by_two.csv"), row.names = FALSE)

# (4) R10: LARGE-magnitude by-variable (sp-scale probe: does py lambda stay
#     comparable to mgcv sp when the by-column is ~1e3?)
set.seed(5)
n <- 400; x <- runif(n); z <- rnorm(n, 1000, 200)
y <- z * sin(2 * pi * x) + rnorm(n, 0, 50)
write.csv(data.frame(y = y, x = x, z = z),
          file.path(outdir, "by_big.csv"), row.names = FALSE)

# (5) nb default: moderate theta
set.seed(7)
n <- 500; x <- runif(n); eta <- 1.2 + 1.5 * sin(2 * pi * x)
mu <- exp(eta - mean(eta)); y <- rnbinom(n, size = 3.0, mu = mu)
write.csv(data.frame(y = y, x = x),
          file.path(outdir, "nb_default.csv"), row.names = FALSE)

# (6) R10 nb: large theta -> near the Poisson limit
set.seed(21)
n <- 500; x <- runif(n); mu <- exp(0.5 + 1.0 * sin(2 * pi * x))
y <- rnbinom(n, size = 50.0, mu = mu)
write.csv(data.frame(y = y, x = x),
          file.path(outdir, "nb_bigtheta.csv"), row.names = FALSE)

# (7) R10 nb: small counts (low mean, small theta)
set.seed(21)
n <- 500; x <- runif(n); mu <- exp(0.5 + 1.0 * sin(2 * pi * x)) * 0.25
y <- rnbinom(n, size = 2.0, mu = mu)
write.csv(data.frame(y = y, x = x),
          file.path(outdir, "nb_small.csv"), row.names = FALSE)

# (8) factor-by fidelity probe: a 3-level factor coded {0,1,2}
set.seed(0)
n <- 300; x <- runif(n); g <- sample(0:2, n, replace = TRUE)
y <- ifelse(g == 0, sin(2 * pi * x),
     ifelse(g == 1, cos(2 * pi * x), 0.5 * x)) + rnorm(n, 0, 0.2)
write.csv(data.frame(y = y, x = x, g = g),
          file.path(outdir, "fac.csv"), row.names = FALSE)
"""


# The dedicated R reference worker: supports gaussian by= and nb(), and returns
# the estimated theta for nb (getTheta) -- the shared r_reference.R does not.
_R_WORKER = r"""
suppressMessages({library(mgcv); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE)
spec <- fromJSON(a[[1]]); out_path <- a[[2]]
d <- read.csv(spec$data_csv)
fam <- if (spec$family == "nb") nb() else gaussian()
form <- as.formula(spec$formula)
fit_args <- list(formula = form, data = d, family = fam, method = spec$method)
if (!is.null(spec$sp)) fit_args$sp <- as.numeric(spec$sp)
m <- do.call(gam, fit_args)
sm <- summary(m)
theta <- if (spec$family == "nb") m$family$getTheta(TRUE) else NA
out <- list(
  coef = as.numeric(coef(m)),
  p_coef = as.numeric(sm$p.coeff),
  p_se = as.numeric(sm$se[seq_along(sm$p.coeff)]),
  sp = as.numeric(m$sp),
  smooth_edf = as.numeric(summary(m)$s.table[, "edf"]),
  total_edf = sum(m$edf),
  reml = m$gcv.ubre,
  deviance = deviance(m),
  scale = m$sig2,
  fitted = as.numeric(fitted(m)),
  theta = theta,
  n = nrow(d), n_coef = length(coef(m)), n_smooth = length(m$smooth),
  mgcv_version = as.character(packageVersion("mgcv"))
)
write_json(out, out_path, auto_unbox = TRUE, digits = 14, na = "null")
"""


def _gen_data(outdir: Path) -> None:
    (outdir / "gen.R").write_text(_DATA_R)
    subprocess.run(["Rscript", str(outdir / "gen.R"), str(outdir)], check=True,
                   capture_output=True, text=True)


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    import csv as _csv
    with open(path) as fh:
        rows = list(_csv.DictReader(fh))
    return {c: np.array([float(r[c]) for r in rows]) for c in rows[0].keys()}


def _run_r(worker: Path, csv: Path, formula: str, family: str, method: str,
           sp: list[float] | None) -> dict:
    spec: dict[str, Any] = {"data_csv": str(csv), "formula": formula,
                            "family": family, "method": method}
    if sp is not None:
        spec["sp"] = list(sp)
    with tempfile.TemporaryDirectory() as td:
        sp_path = Path(td) / "spec.json"
        out_path = Path(td) / "out.json"
        sp_path.write_text(json.dumps(spec))
        proc = subprocess.run(["Rscript", str(worker), str(sp_path),
                               str(out_path)], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mgcv failed ({formula}):\n{proc.stderr}")
        return json.loads(out_path.read_text())


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


def _nb_theta_from_deviance(y: np.ndarray, mu: np.ndarray,
                            deviance: float) -> float:
    """Recover py's EFFECTIVE nb theta by inverting the NB deviance at the
    fitted mean (pystatistics exposes no theta accessor -- finding VA3-F1).

    Unit deviance 2*sum[ y log(y/mu) - (y+theta) log((y+theta)/(mu+theta)) ]
    is strictly monotone in theta at fixed (y, mu), so the inversion is unique.
    """
    mm = np.maximum(mu, 1e-10)

    def dev(theta: float) -> float:
        t1 = np.where(y > 0, y * np.log(y / mm), 0.0)
        t2 = (y + theta) * np.log((y + theta) / (mm + theta))
        return 2.0 * float(np.sum(t1 - t2))

    return float(brentq(lambda th: dev(th) - deviance, 1e-3, 1e6, xtol=1e-10))


# --------------------------------------------------------------------------
# VA-2 by= cases (Gaussian, two-tier)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ByCase:
    key: str
    why: str
    data: str
    r_formula: str
    py_smooths: Callable[[], list]
    smooth_vars: list[str]


def _fit_py_by(case: ByCase, cols: dict[str, np.ndarray],
               sp: list[float] | None):
    y = cols["y"]
    smooth_data = {v: cols[v] for v in case.smooth_vars}
    kw: dict[str, Any] = dict(smooths=case.py_smooths(), smooth_data=smooth_data,
                              family="gaussian", method="REML")
    if sp is not None:
        kw["sp"] = np.asarray(sp, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gam(y, None, **kw)


def run_by_case(case: ByCase, worker: Path, tmp: Path) -> dict[str, Any]:
    csv = tmp / f"{case.data}.csv"
    cols = _load_csv(csv)
    r_free = _run_r(worker, csv, case.r_formula, "gaussian", "REML", sp=None)
    py_free = _fit_py_by(case, cols, sp=None)

    rec: dict[str, Any] = {
        "key": case.key, "why": case.why, "surface": "VA-2 by=",
        "n": r_free["n"],
        "n_coef": {"py": int(np.asarray(py_free.coefficients).shape[0]),
                   "r": int(r_free["n_coef"])},
        "n_sp": {"py": int(np.atleast_1d(py_free.lambdas).size),
                 "r": int(np.atleast_1d(r_free["sp"]).size)},
        "tier2_free": {
            "total_edf": _cmp(py_free.total_edf, r_free["total_edf"]),
            "smooth_edf": _vec_cmp(py_free.edf, r_free["smooth_edf"]),
            "sp": _vec_cmp(py_free.lambdas, r_free["sp"]),
            "sp_values": {"py": [float(v) for v in np.atleast_1d(py_free.lambdas)],
                          "r": [float(v) for v in np.atleast_1d(r_free["sp"])]},
            "reml": _cmp(py_free.reml_score, r_free["reml"]),
            "deviance": _cmp(py_free.deviance, r_free["deviance"]),
            "scale": _cmp(py_free.scale, r_free["scale"]),
            "fitted": _vec_cmp(py_free.fitted_values, r_free["fitted"]),
        },
    }
    # TIER 1 -- feed mgcv's selected sp to both engines.
    sp_mgcv = [float(v) for v in np.atleast_1d(r_free["sp"])]
    r_fix = _run_r(worker, csv, case.r_formula, "gaussian", "REML", sp=sp_mgcv)
    py_fix = _fit_py_by(case, cols, sp=sp_mgcv)
    rec["tier1_fixedsp"] = {
        "sp": sp_mgcv,
        "fitted": _vec_cmp(py_fix.fitted_values, r_fix["fitted"]),
        "coef": _vec_cmp(py_fix.coefficients, r_fix["coef"]),
        "total_edf": _cmp(py_fix.total_edf, r_fix["total_edf"]),
        "scale": _cmp(py_fix.scale, r_fix["scale"]),
        "param_se": _vec_cmp(np.asarray(py_fix.standard_errors)[:1], r_fix["p_se"]),
    }
    return rec


def build_by_cases() -> list[ByCase]:
    return [
        ByCase("by_cont",
               "Canonical continuous by=: y = z*f(x), z ~ N(2,1). The core "
               "varying-coefficient surface.",
               "by_cont", "y ~ s(x, by=z, k=10, bs='cr')",
               lambda: [s("x", k=10, bs="cr", by="z")], ["x", "z"]),
        ByCase("by_neg",
               "R10: by-variable with NEGATIVE values and a zero-variance "
               "stretch (z in [-3,3], first 40 obs z=0).",
               "by_neg", "y ~ s(x, by=z, k=10, bs='cr')",
               lambda: [s("x", k=10, bs="cr", by="z")], ["x", "z"]),
        ByCase("by_two",
               "R10: by= alongside a SECOND ordinary smooth -- s(x,by=z)+s(w). "
               "Two penalties, two smoothing parameters.",
               "by_two", "y ~ s(x, by=z, k=10, bs='cr') + s(w, k=10, bs='cr')",
               lambda: [s("x", k=10, bs="cr", by="z"), s("w", k=10, bs="cr")],
               ["x", "w", "z"]),
        ByCase("by_big",
               "R10: LARGE-magnitude by-variable (z ~ N(1000,200)) -- probes "
               "whether py lambda stays directly comparable to mgcv sp when the "
               "by-column dominates the design scale.",
               "by_big", "y ~ s(x, by=z, k=10, bs='cr')",
               lambda: [s("x", k=10, bs="cr", by="z")], ["x", "z"]),
    ]


# --------------------------------------------------------------------------
# VA-3 nb cases (free REML; theta recovered by deviance inversion)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NbCase:
    key: str
    why: str
    data: str


def run_nb_case(case: NbCase, worker: Path, tmp: Path) -> dict[str, Any]:
    csv = tmp / f"{case.data}.csv"
    cols = _load_csv(csv)
    formula = "y ~ s(x, k=10, bs='cr')"
    r_free = _run_r(worker, csv, formula, "nb", "REML", sp=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py = gam(cols["y"], None, smooths=[s("x", k=10, bs="cr")],
                 smooth_data={"x": cols["x"]}, family="nb", method="REML")
    # Read the estimated theta from the PUBLIC accessor (VA3-F1 fix, 4.8.1);
    # cross-check it against the deviance-inversion recovery for defence in depth.
    py_theta = float(py.theta)
    py_theta_recovered = _nb_theta_from_deviance(
        cols["y"], py.fitted_values, py.deviance)
    r_theta = float(r_free["theta"])
    return {
        "key": case.key, "why": case.why, "surface": "VA-3 nb",
        "n": r_free["n"],
        "n_coef": {"py": int(np.asarray(py.coefficients).shape[0]),
                   "r": int(r_free["n_coef"])},
        "theta": {"py_accessor": py_theta, "py_recovered": py_theta_recovered,
                  "r_getTheta": r_theta, "abs": abs(py_theta - r_theta),
                  "rel": abs(py_theta - r_theta) / max(abs(r_theta), 1e-30),
                  "accessor_vs_recovered_abs": abs(py_theta - py_theta_recovered)},
        "theta_accessor_exposed": _theta_accessor_present(py),
        "total_edf": _cmp(py.total_edf, r_free["total_edf"]),
        "smooth_edf": _vec_cmp(py.edf, r_free["smooth_edf"]),
        "reml": _cmp(py.reml_score, r_free["reml"]),
        "deviance": _cmp(py.deviance, r_free["deviance"]),
        "fitted": _vec_cmp(py.fitted_values, r_free["fitted"]),
    }


# --------------------------------------------------------------------------
# VA-2 FACTOR-by cases (per-level smooths; free REML) -- VA2-F1 fix (4.8.1)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FactorByCase:
    key: str
    why: str
    data: str
    var: str          # smooth variable
    by: str           # integer-coded factor column
    bs: str
    k: int


def run_factor_by_case(case: FactorByCase, worker: Path,
                       tmp: Path) -> dict[str, Any]:
    """s(x, by=g, by_type='factor') -> a smooth per level, vs mgcv's
    s(x, by=factor(g)) + factor(g). Free-REML estimator-invariant comparison
    (total & per-level EDF, deviance, REML score, fitted, coefficient count)."""
    csv = tmp / f"{case.data}.csv"
    cols = _load_csv(csv)
    n_levels = int(np.unique(cols[case.by]).size)
    r_formula = (f"y ~ s({case.var}, by=factor({case.by}), k={case.k}, "
                 f"bs='{case.bs}') + factor({case.by})")
    r_free = _run_r(worker, csv, r_formula, "gaussian", "REML", sp=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py = gam(cols["y"], None,
                 smooths=[s(case.var, k=case.k, bs=case.bs, by=case.by,
                            by_type="factor")],
                 smooth_data={case.var: cols[case.var], case.by: cols[case.by]},
                 method="REML")
    return {
        "key": case.key, "why": case.why, "surface": "VA-2 by= (factor)",
        "n": r_free["n"], "n_levels": n_levels,
        "n_coef": {"py": int(np.asarray(py.coefficients).shape[0]),
                   "r": int(r_free["n_coef"])},
        "n_smooth": {"py": int(np.atleast_1d(py.edf).size),
                     "r": int(r_free["n_smooth"])},
        "total_edf": _cmp(py.total_edf, r_free["total_edf"]),
        "smooth_edf": _vec_cmp(sorted(np.atleast_1d(py.edf)),
                               sorted(np.atleast_1d(r_free["smooth_edf"]))),
        "reml": _cmp(py.reml_score, r_free["reml"]),
        "deviance": _cmp(py.deviance, r_free["deviance"]),
        "fitted": _vec_cmp(py.fitted_values, r_free["fitted"]),
    }


def build_factor_by_cases() -> list[FactorByCase]:
    return [
        FactorByCase(
            "factor_by_tp",
            "VA2-F1 fix: s(x, by=g, by_type='factor') builds a separate smooth "
            "per level of a 3-level factor (mgcv's s(x, by=factor(g))+factor(g)); "
            "tp basis, mgcv's factor-by default.",
            "fac", "x", "g", "tp", 10),
        FactorByCase(
            "factor_by_cr",
            "VA2-F1: the same per-level factor-by on a cr basis.",
            "fac", "x", "g", "cr", 10),
    ]


def _theta_accessor_present(py) -> bool:
    """Is the estimated nb theta reachable through the public solution API?
    (No: family_name is the bare 'negative.binomial' and no scalar attr holds
    it -- finding VA3-F1. Probed here so the artifact records it as evidence.)"""
    for obj in (py, py.params):
        for attr in dir(obj):
            if "theta" in attr.lower():
                return True
    fam = str(py.params.family_name)
    return any(ch.isdigit() for ch in fam)  # e.g. "Negative Binomial(3.05)"


# --------------------------------------------------------------------------
# Fail-loud probes: factor-by (VA2-F1) and nb+GCV guard
# --------------------------------------------------------------------------

def run_fidelity(worker: Path, tmp: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # (a) VA2-F1 guard: a categorical-looking by-column with by_type UNSET must
    # now FAIL LOUD (4.8.1) instead of silently fitting a continuous varying
    # coefficient. (The per-level factor-by surface itself is validated for
    # agreement in run_factor_by_case.)
    cols = _load_csv(tmp / "fac.csv")
    try:
        gam(cols["y"], None, smooths=[s("x", k=10, bs="cr", by="g")],
            smooth_data={"x": cols["x"], "g": cols["g"]}, method="REML")
        guard = {"raised": False}
    except (ValidationError, Exception) as exc:  # noqa: BLE001
        guard = {"raised": True, "error_type": type(exc).__name__,
                 "message": str(exc)[:200]}
    out.append({
        "key": "factor_by_guard", "surface": "VA-2 by= (fail-loud)",
        "finding": "VA2-F1 (resolved)",
        "why": "A categorical-looking by-column (3-level factor coded {0,1,2}) "
        "with by_type unset must fail loud, telling the user to pick "
        "by_type='factor' or 'continuous' -- not silently fit it as a continuous "
        "varying coefficient (the pre-4.8.1 silent misinterpretation).",
        "py": guard, "fails_loud": bool(guard.get("raised"))})

    # (b) nb + GCV must fail loud (estimated theta needs REML).
    cols = _load_csv(tmp / "nb_default.csv")
    try:
        gam(cols["y"], None, smooths=[s("x", k=10, bs="cr")],
            smooth_data={"x": cols["x"]}, family="nb", method="GCV")
        guard = {"raised": False}
    except (ValidationError, Exception) as exc:  # noqa: BLE001
        guard = {"raised": True, "error_type": type(exc).__name__,
                 "message": str(exc)[:200]}
    out.append({
        "key": "nb_gcv_guard", "surface": "VA-3 nb (fail-loud)",
        "why": "family='nb' with estimated theta requires method='REML'; GCV "
        "profiling of theta is degenerate. Must fail loud, not silently fit.",
        "py": guard, "fails_loud": bool(guard.get("raised"))})
    return out


# --------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = {"mgcv": None}

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _gen_data(tmp)
        worker = tmp / "worker.R"
        worker.write_text(_R_WORKER)

        for case in build_by_cases():
            rec = run_by_case(case, worker, tmp)
            records.append(rec)
            t1, t2 = rec["tier1_fixedsp"], rec["tier2_free"]
            print(f"  {case.key:10s} VA-2 | tier1 fitted {t1['fitted']['max_abs']:.1e} "
                  f"edf {t1['total_edf']['abs']:.1e} | tier2 edf {t2['total_edf']['abs']:.1e} "
                  f"sp_rel {t2['sp']['max_rel']:.1e} fitted {t2['fitted']['max_abs']:.1e}")

        for case in [NbCase("nb_default", "Canonical nb, theta~3.", "nb_default"),
                     NbCase("nb_bigtheta",
                            "R10: large theta -> near the Poisson limit.",
                            "nb_bigtheta"),
                     NbCase("nb_small", "R10: small counts, low mean.",
                            "nb_small")]:
            rec = run_nb_case(case, worker, tmp)
            records.append(rec)
            th = rec["theta"]
            print(f"  {case.key:11s} VA-3 | theta py {th['py_accessor']:.5f} "
                  f"r {th['r_getTheta']:.5f} (rel {th['rel']:.1e}) | edf "
                  f"{rec['total_edf']['abs']:.1e} fitted {rec['fitted']['max_abs']:.1e} "
                  f"| theta_accessor={rec['theta_accessor_exposed']}")

        for fcase in build_factor_by_cases():
            rec = run_factor_by_case(fcase, worker, tmp)
            records.append(rec)
            print(f"  {fcase.key:11s} VA-2 factor | n_coef py {rec['n_coef']['py']}="
                  f"{rec['n_coef']['r']} r, {rec['n_levels']} levels | "
                  f"total_edf {rec['total_edf']['abs']:.1e} fitted "
                  f"{rec['fitted']['max_abs']:.1e} deviance {rec['deviance']['abs']:.1e}")

        fidelity = run_fidelity(worker, tmp)
        records.extend(fidelity)
        for f in fidelity:
            print(f"  {f['key']:11s} {f['surface']:18s} | "
                  + (f"silent_misinterp={f['silent_misinterpretation']}"
                     if "silent_misinterpretation" in f
                     else f"fails_loud={f['fails_loud']}"))

    run = build_run(
        env=env,
        config={"suite": "gam-by-nb",
                "reference": "mgcv::gam s(x,by=z) [continuous] and family=nb()",
                "tolerance_contract":
                "VA-2 by=: TIER1 mgcv sp fed to both -> fitted/EDF/coef "
                "machine-precision on the by-multiplied cr basis (sp on the same "
                "scale as mgcv, established not assumed); TIER2 free REML -> "
                "per-smooth & total EDF, comparable sp, REML score, deviance, "
                "fitted. VA-3 nb: free REML -> EDF/deviance/REML/fitted and the "
                "estimated theta vs getTheta (py theta recovered by deviance "
                "inversion; no accessor -- VA3-F1). Fidelity: factor-by (VA2-F1) "
                "and nb+GCV guard."},
        records=records)
    out = Path(str(_ARTIFACT))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
