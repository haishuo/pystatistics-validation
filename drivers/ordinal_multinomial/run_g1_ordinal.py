"""G1 correctness — ordinal polr vs MASS::polr (both whole surface).

Two-tier contract (matched link between engines):
  TIER A (optimizer): coefficients beta, ordered thresholds zeta, their standard
    errors (observed information at the optimum), log-likelihood, AIC/deviance.
  TIER B (tight): the implied fitted class probabilities. polr exposes no predict/
    fitted-probability accessor, so we compute P(Y=j|x) from the library's OWN
    reported coef+thresholds and confirm they reproduce R's predict(type="probs")
    — i.e. the coefficient agreement carries through to the response scale.

Covers: housing across all three links (logistic/probit/cloglog); a well-
conditioned continuous+factor DGP (clean coefficient tier); the R15 bare default
call; and the R10 hard cases (complete separation -> loud failure matching R's
divergence; a rare interior category).

Emits artifacts/ordinal_multinomial/v<ver>/runs/g1_ordinal.json.
Run: MVNMLE_DATA_DIR=Dev/datasets python drivers/ordinal_multinomial/run_g1_ordinal.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.special import expit, ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omdata  # noqa: E402
from omref import r_polr, r_package_versions  # noqa: E402
from omcompare import arr_cmp, scalar_cmp, within, to_native  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run  # noqa: E402
# write_run comes from the guard, not pystatsval: it refuses to overwrite
# evidence that is committed to git unless PYSTATSVAL_ALLOW_ARTIFACT_OVERWRITE=1.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_guard import write_run  # noqa: E402
from pystatistics.ordinal import polr  # noqa: E402
from pystatistics.core.exceptions import ConvergenceError  # noqa: E402

_ART = (Path(__file__).resolve().parents[2]
        / "artifacts/ordinal_multinomial/v{ver}/runs/g1_ordinal.json")


def _linkinv(name: str, x: np.ndarray) -> np.ndarray:
    if name == "logistic":
        return expit(x)
    if name == "probit":
        return ndtr(x)
    if name == "cloglog":
        return 1.0 - np.exp(-np.exp(np.clip(x, -500, 500)))
    raise ValueError(name)


def _implied_probs(coef, zeta, X, link) -> np.ndarray:
    """P(Y=j|x) = F(zeta_j - x'b) - F(zeta_{j-1} - x'b), matching MASS::polr."""
    eta = X @ np.asarray(coef, float)
    cum = _linkinv(link, zeta[None, :] - eta[:, None])       # (n, K-1)
    ext = np.column_stack([np.zeros(len(X)), cum, np.ones(len(X))])
    return ext[:, 1:] - ext[:, :-1]


# The shared statistical-link label ("logistic") is what MASS::polr's method= and
# the local link helpers use; pystatistics 5.0 renamed that link's argument value
# to "logit". Decouple: keep `link` as the matched-engine label (R + helpers +
# JSON) and translate ONLY at the pystatistics call site.
_PYLINK = {"logistic": "logit", "probit": "probit", "cloglog": "cloglog"}


def _fit_case(des, link: str, tier_a_abs=2e-3, se_rel=2e-2, ll_abs=2e-3,
              prob_abs=1e-4) -> dict:
    sol = polr(des.y, des.X, link=_PYLINK[link])
    r = r_polr(des.y, des.X, link)
    if not r.get("ok"):
        return {"group": "polr_fit", "dataset": des.key, "link": link,
                "r_error": r.get("error"), "r_warnings": r.get("warnings"),
                "pass": False}
    coef = arr_cmp(sol.coefficients, r["coef"])
    zeta = arr_cmp(sol.threshold_values, r["zeta"])
    se_b = arr_cmp(sol.standard_errors, r["se_beta"])
    se_z = arr_cmp(sol.threshold_standard_errors, r["se_zeta"])
    ll = scalar_cmp(sol.log_likelihood, r["loglik"])
    aic = scalar_cmp(sol.aic, r["aic"])
    probs_py = _implied_probs(sol.coefficients, sol.threshold_values, des.X, link)
    r_fit = np.asarray(r["fitted"], float)
    prob = arr_cmp(probs_py, r_fit)
    ok = (within(coef, abs_tol=tier_a_abs) and within(zeta, abs_tol=tier_a_abs)
          and within(se_b, rel_tol=se_rel) and within(se_z, rel_tol=se_rel)
          and within(ll, abs_tol=ll_abs) and within(prob, abs_tol=prob_abs))
    return {"group": "polr_fit", "dataset": des.key, "link": link,
            "n": len(des.y), "p": des.X.shape[1], "K": des.n_levels,
            "coef": coef, "zeta": zeta, "se_beta": se_b, "se_zeta": se_z,
            "loglik": ll, "aic": aic, "implied_probs": prob,
            "converged": bool(sol.converged),
            "r_warnings": r.get("warnings"), "pass": bool(ok)}


def run_correctness() -> list[dict]:
    recs = []
    housing = omdata.load_housing()
    for link in ("logistic", "probit", "cloglog"):
        recs.append(_fit_case(housing, link))
    # well-conditioned continuous+factor DGP (clean coefficient tier)
    synth = omdata.synth_ordinal()
    recs.append(_fit_case(synth, "logistic"))
    # rare interior category
    coll = omdata.collapsed_ordinal()
    recs.append(_fit_case(coll, "logistic"))
    return recs


def run_default() -> list[dict]:
    """R15: the bare default call. polr(y, X) must default to the logit link
    (renamed from 'logistic' in 5.0) and reproduce the explicit-logit fit."""
    housing = omdata.load_housing()
    sol_default = polr(housing.y, housing.X)          # no link= -> default
    sol_logit = polr(housing.y, housing.X, link="logit")
    same = arr_cmp(sol_default.coefficients, sol_logit.coefficients)
    return [{"group": "r15_default", "dataset": "housing",
             "default_link": sol_default.link,
             "coef_vs_explicit_logistic": same,
             "pass": (sol_default.link == "logit"
                      and within(same, abs_tol=1e-12))}]


def run_hardcases() -> list[dict]:
    """R10: complete separation must fail LOUD (ConvergenceError), matching R's
    divergence — record R's coefficient blow-up / warnings for the report."""
    recs = []
    sep = omdata.sep_ordinal_complete()
    py_raised, py_msg = False, ""
    try:
        polr(sep.y, sep.X, link="logit")
    except ConvergenceError as e:
        py_raised, py_msg = True, str(e)[:160]
    r = r_polr(sep.y, sep.X, "logistic")
    r_maxcoef = (float(np.max(np.abs(r["coef"]))) if r.get("ok") else None)
    # R10 pass = pystatistics fails loud AND R does not return a clean converged
    # fit either (it errors or diverges) — both engines refuse a separated fit.
    r_refuses = (not r.get("ok")) or bool(r.get("warnings"))
    recs.append({"group": "r10_separation", "dataset": sep.key,
                 "py_fail_loud": py_raised, "py_message": py_msg,
                 "r_ok": r.get("ok"), "r_error": r.get("error"),
                 "r_maxabs_coef": r_maxcoef, "r_warnings": r.get("warnings"),
                 "r_refuses": r_refuses,
                 "note": "complete separation: pystatistics rejects the non-PD "
                         "observed information with ConvergenceError (fail-loud); "
                         "MASS::polr also refuses — glm.fit emits the "
                         "'fitted probabilities numerically 0 or 1' separation "
                         "warnings and polr then errors. Both engines refuse a "
                         "separated fit; neither returns a silently-finite result.",
                 "pass": bool(py_raised and r_refuses)})
    return recs


def main() -> None:
    warnings.filterwarnings("ignore")
    env = env_manifest(device="cpu")
    require_pypi(env)
    env["r_packages"] = r_package_versions()

    corr = run_correctness()
    default = run_default()
    hard = run_hardcases()

    run = build_run(
        env=env,
        config={"suite": "ordinal-g1", "reference": "R MASS::polr",
                "tolerance_contract": "two-tier: coef/zeta/loglik optimizer-tier "
                "(matched link), SE rel-tier, implied fitted probs tight"},
        records=to_native([{"key": "correctness", "checks": corr},
                           {"key": "default", "checks": default},
                           {"key": "hardcases", "checks": hard}]),
    )
    out = Path(str(_ART).format(ver=env["pystatistics_version"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    write_run(out, run)
    print(f"wrote {out}")
    for name, grp in [("correct", corr), ("default", default), ("hard", hard)]:
        print(f"  {name:8s} {sum(bool(r.get('pass')) for r in grp)}/{len(grp)} pass")


if __name__ == "__main__":
    main()
