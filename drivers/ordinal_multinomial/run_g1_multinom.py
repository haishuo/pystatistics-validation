"""G1 correctness — multinomial multinom vs nnet::multinom (whole surface).

Two-tier contract (matched convention: decay=0, baseline releveled so nnet's
first-level baseline coincides with pystatistics' last-code reference):
  TIER A (optimizer): the (K-1)x(1+p) coefficient matrix, its standard errors,
    log-likelihood, AIC/deviance — validated tightly only where the MLE is
    IDENTIFIED (well-conditioned designs).
  TIER B (tight): the fitted class probabilities (multinom exposes fitted_probs
    directly; compared to nnet's fitted(), reordered to code order).

Separation regime (R10): iris (setosa linearly separable), fgl (small classes),
and a constructed complete-separation design are QUASI/COMPLETE separation cases
where the MLE is at infinity. There both engines report 'convergence' but the
separating coefficients are non-identified — they agree on LOG-LIKELIHOOD and
FITTED PROBABILITIES, not on the runaway coefficients. This matches nnet's own
behaviour (it does not fail loud on separation); the validation records the
loglik/probability agreement and the coefficient divergence rather than
demanding a coefficient match that does not exist.

Emits artifacts/ordinal_multinomial/v<ver>/runs/g1_multinom.json.
Run: MVNMLE_DATA_DIR=Dev/datasets python drivers/ordinal_multinomial/run_g1_multinom.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import omdata  # noqa: E402
from omref import r_multinom, reorder_fitted, r_package_versions  # noqa: E402
from omcompare import arr_cmp, scalar_cmp, within, to_native  # noqa: E402

from pystatsval.device import env_manifest, require_pypi  # noqa: E402
from pystatsval.serialize import build_run, write_run  # noqa: E402
from pystatistics.multinomial import multinom  # noqa: E402
from pystatistics.core.exceptions import (  # noqa: E402
    ConvergenceError, NotPositiveDefiniteError)

_ART = (Path(__file__).resolve().parents[2]
        / "artifacts/ordinal_multinomial/v{ver}/runs/g1_multinom.json")


def _fit(des, max_iter=3000):
    try:
        sol = multinom(des.y, des.X, max_iter=max_iter)
        return sol, None
    except (ConvergenceError, NotPositiveDefiniteError) as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"


def _well_conditioned(des, coef_abs=5e-3, se_rel=5e-2, ll_abs=5e-3,
                      prob_abs=1e-3) -> dict:
    """Identified MLE: coefficients, SEs, loglik, and fitted probs all tight."""
    sol, err = _fit(des)
    r = r_multinom(des.y, des.X[:, 1:], des.r_levels)   # strip intercept for nnet
    if err or not r.get("ok"):
        return {"group": "multinom_well", "dataset": des.key,
                "py_error": err, "r_error": r.get("error"), "pass": False}
    coef = arr_cmp(sol.coefficient_matrix, r["coef"])
    se = arr_cmp(sol.standard_errors, r["se"])
    ll = scalar_cmp(sol.log_likelihood, r["loglik"])
    aic = scalar_cmp(sol.aic, r["aic"])
    fit_r = reorder_fitted(r["fitted"], r["fitted_cols"], des.n_classes)
    prob = arr_cmp(sol.fitted_probs, fit_r)
    ok = (within(coef, abs_tol=coef_abs) and within(se, rel_tol=se_rel)
          and within(ll, abs_tol=ll_abs) and within(prob, abs_tol=prob_abs))
    return {"group": "multinom_well", "dataset": des.key,
            "n": len(des.y), "p": des.X.shape[1], "K": des.n_classes,
            "coef": coef, "se": se, "loglik": ll, "aic": aic,
            "fitted_probs": prob, "converged": bool(sol.converged),
            "r_conv": r.get("conv"), "pass": bool(ok)}


def _separation(des, ll_abs=1e-1, prob_abs=5e-3) -> dict:
    """Non-identified MLE (separation): loglik + fitted probs agree; coefficients
    diverge in BOTH engines. Pass on the identified quantities, not coefficients."""
    sol, err = _fit(des)
    r = r_multinom(des.y, des.X[:, 1:], des.r_levels)
    if not r.get("ok"):
        return {"group": "multinom_separation", "dataset": des.key,
                "py_error": err, "r_error": r.get("error"),
                "r_warnings": r.get("warnings"), "pass": False}
    rec = {"group": "multinom_separation", "dataset": des.key,
           "why": des.why, "n": len(des.y), "K": des.n_classes,
           "r_conv": r.get("conv"), "r_warnings": r.get("warnings")}
    if err:
        # pystatistics failed loud where nnet limped to a non-identified fit.
        rec.update({"py_fail_loud": True, "py_error": err,
                    "r_maxabs_coef": float(np.max(np.abs(r["coef"]))),
                    "note": "pystatistics fails loud; nnet returns a non-identified "
                            "large-coefficient fit (convergence flag "
                            f"{r.get('conv')}).",
                    "pass": True})
        return rec
    ll = scalar_cmp(sol.log_likelihood, r["loglik"])
    fit_r = reorder_fitted(r["fitted"], r["fitted_cols"], des.n_classes)
    prob = arr_cmp(sol.fitted_probs, fit_r)
    py_max = float(np.max(np.abs(sol.coefficient_matrix)))
    r_max = float(np.max(np.abs(r["coef"])))
    rec.update({"loglik": ll, "fitted_probs": prob,
                "py_maxabs_coef": py_max, "r_maxabs_coef": r_max,
                "coef_identified": False,
                "note": "quasi-separation: loglik/probs agree; max|coef| diverges "
                        f"(py {py_max:.0f} vs R {r_max:.0f}) — MLE at infinity, "
                        "coefficients non-identified in both engines.",
                "pass": bool(within(ll, abs_tol=ll_abs)
                             and within(prob, abs_tol=prob_abs))})
    return rec


def run_correctness() -> list[dict]:
    recs = []
    # well-conditioned: clean 3-class anchor + a many-category (5-class) synth
    recs.append(_well_conditioned(omdata.load_multinom_synth()))
    recs.append(_well_conditioned(omdata.synth_multinom(n=2000, K=5, p=4)))
    recs.append(_well_conditioned(omdata.unbalanced_multinom()))
    # real-data separation cases
    recs.append(_separation(omdata.load_iris_multinom()))
    recs.append(_separation(omdata.load_fgl()))
    return recs


def run_default() -> list[dict]:
    """R15: bare multinom(y, X). decay defaults to 0 (the unpenalized MLE); the
    default fit must reproduce the explicit fit and match nnet(decay=0)."""
    des = omdata.load_multinom_synth()
    sol = multinom(des.y, des.X)                 # all defaults
    r = r_multinom(des.y, des.X[:, 1:], des.r_levels)
    coef = arr_cmp(sol.coefficient_matrix, r["coef"])
    ll = scalar_cmp(sol.log_likelihood, r["loglik"])
    return [{"group": "r15_default", "dataset": des.key,
             "coef": coef, "loglik": ll, "converged": bool(sol.converged),
             "pass": within(coef, abs_tol=5e-3) and within(ll, abs_tol=5e-3)}]


def run_hardcases() -> list[dict]:
    """R10 complete separation (constructed): one class perfectly predicted, so the
    MLE is at infinity and the observed information is not positive definite. As of
    4.6.9 multinom FAILS LOUD here (NotPositiveDefiniteError) — matching polr and
    the library's fail-loud contract — rather than returning a converged fit with a
    meaningless (negative-variance) covariance. nnet instead returns the degenerate
    fit with a non-convergence flag; we record both postures."""
    des = omdata.sep_multinom_complete()
    sol, err = _fit(des, max_iter=2000)
    r = r_multinom(des.y, des.X[:, 1:], des.r_levels)
    rec = {"group": "r10_complete_separation", "dataset": des.key,
           "r_conv": r.get("conv"), "r_warnings": r.get("warnings"),
           "r_maxabs_coef": (float(np.max(np.abs(r["coef"]))) if r.get("ok")
                             else None),
           "r_maxabs_se": (float(np.max(np.abs(r["se"]))) if r.get("ok")
                           else None)}
    py_fail_loud = err is not None
    rec.update({
        "py_fail_loud": py_fail_loud, "py_error": err,
        "note": "complete separation: multinom raises NotPositiveDefiniteError "
                "(the observed information is not positive definite; the "
                "coefficient covariance is not identified) — fail-loud, matching "
                f"polr. nnet returns the degenerate fit (conv={r.get('conv')}, "
                f"|coef|~{rec['r_maxabs_coef']:.0f} if finite) with a non-"
                "convergence flag; pystatistics refuses rather than return a "
                "meaningless variance-covariance.",
        "pass": bool(py_fail_loud)})
    return [rec]


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
        config={"suite": "multinom-g1", "reference": "R nnet::multinom (decay=0)",
                "tolerance_contract": "two-tier: coef/SE/loglik optimizer-tier "
                "where identified; fitted probs tight; separation cases validated "
                "on loglik+probs (coefficients non-identified, matching nnet)"},
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
