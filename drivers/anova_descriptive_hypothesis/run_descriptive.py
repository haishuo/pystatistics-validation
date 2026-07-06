"""G1 correctness sweep for the ``descriptive`` module vs R (+ scipy third ref).

Covers the whole public surface: var, cov, cor{pearson,spearman,kendall},
quantile (all 9 R types), describe/summary (mean, sd, var, median, skewness &
kurtosis matched to e1071 type 2). Every quantity is deterministic closed-form,
so the bar is machine precision (TOL_EXACT) on identical inputs.

Writes artifacts/anova_descriptive_hypothesis/v<ver>/runs/descriptive_g1.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.stats as sps

import adhdata as D
from compare import TOL_EXACT, TOL_TIGHT, case, summarize, write_artifact
from rref import r_ref, r_versions

from pystatistics.descriptive import cor, cov, describe, quantile, var
from pystatistics import __version__ as PYVER

ART = Path(__file__).resolve().parents[2] / \
    "artifacts/anova_descriptive_hypothesis" / f"v{PYVER}" / "runs"


def run() -> list[dict]:
    cases: list[dict] = []
    mt = D.load_frame("mtcars")
    mpg = mt["mpg"]
    hp = mt["hp"]
    wt = mt["wt"]

    # ---- var / sd (n-1 denominator) ----------------------------------------
    for col in ("mpg", "hp", "wt", "qsec", "disp"):
        x = mt[col]
        cases.append(case(f"var:mtcars.{col}", "variance",
                          var(x).variance, r_ref("var", x=x)["value"], TOL_EXACT))
    cases.append(case("sd:mtcars.mpg", "sd",
                      describe(mpg).sd, r_ref("sd", x=mpg)["value"], TOL_EXACT))

    # ---- cov: pairwise + full matrix ---------------------------------------
    cases.append(case("cov:mpg~hp", "covariance",
                      cov(mpg, hp).covariance_matrix[0, 1],
                      r_ref("cov_pair", x=mpg, y=hp)["value"], TOL_EXACT))
    X = np.column_stack([mpg, hp, wt, mt["qsec"]])
    cases.append(case("cov:matrix(mpg,hp,wt,qsec)", "cov_matrix",
                      np.asarray(cov(X).covariance_matrix).ravel(),
                      r_ref("cov_matrix", x=X.ravel(), nrow=X.shape[0])["value"],
                      TOL_EXACT))

    # ---- cor: three methods, pairwise + matrix, incl. scipy third ref ------
    for method, sp in (("pearson", lambda a, b: sps.pearsonr(a, b)[0]),
                       ("spearman", lambda a, b: sps.spearmanr(a, b)[0]),
                       ("kendall", lambda a, b: sps.kendalltau(a, b)[0])):
        c = cor(mpg, hp, method=method)
        cases.append(case(f"cor:mpg~hp:{method}", f"cor_{method}",
                          c.correlation_matrix[0, 1],
                          r_ref("cor_pair", x=mpg, y=hp, method=method)["value"],
                          TOL_EXACT, scipy=sp(mpg, hp)))
        cm = cor(X, method=method)
        cases.append(case(f"cor:matrix:{method}", f"cor_matrix_{method}",
                          np.asarray(cm.correlation_matrix).ravel(),
                          r_ref("cor_matrix", x=X.ravel(), nrow=X.shape[0],
                                method=method)["value"], TOL_EXACT))

    # ---- kendall tau-b with TIES (the tie-correction gotcha) ---------------
    tv = D.tie_vectors()
    xt, yt = tv["x_ties"], tv["y_ties"]
    cases.append(case("cor:kendall:ties(tau-b)", "cor_kendall_ties",
                      cor(xt, yt, method="kendall").correlation_matrix[0, 1],
                      r_ref("cor_pair", x=xt, y=yt, method="kendall")["value"],
                      TOL_EXACT, scipy=sps.kendalltau(xt, yt)[0]))
    cases.append(case("cor:spearman:ties", "cor_spearman_ties",
                      cor(xt, yt, method="spearman").correlation_matrix[0, 1],
                      r_ref("cor_pair", x=xt, y=yt, method="spearman")["value"],
                      TOL_EXACT, scipy=sps.spearmanr(xt, yt)[0]))

    # ---- quantile: ALL 9 R types (default is type 7) -----------------------
    probs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for t in range(1, 10):
        q = quantile(mpg, probs=probs, quantile_type=t)
        cases.append(case(f"quantile:mtcars.mpg:type{t}", f"quantile_type{t}",
                          np.asarray(q.quantiles).ravel(),
                          r_ref("quantile", x=mpg, probs=probs, type=t)["value"],
                          TOL_EXACT))
    # default (no type arg) must reproduce type 7
    qd = quantile(mpg)
    cases.append(case("quantile:mtcars.mpg:DEFAULT", "quantile_default_is_type7",
                      np.asarray(qd.quantiles).ravel(),
                      r_ref("quantile", x=mpg, probs=[0, .25, .5, .75, 1],
                            type=7)["value"], TOL_EXACT))

    # ---- describe / summary: moments (skew/kurt = e1071 type 2) ------------
    for col in ("mpg", "wt"):
        x = mt[col]
        d = describe(x)
        rm = r_ref("moments", x=x)
        cases.append(case(f"describe:{col}:mean", "mean", d.mean, rm["mean"], TOL_EXACT))
        cases.append(case(f"describe:{col}:var", "var", d.variance, rm["var"], TOL_EXACT))
        cases.append(case(f"describe:{col}:sd", "sd", d.sd, rm["sd"], TOL_EXACT))
        cases.append(case(f"describe:{col}:skewness", "skewness_e1071_t2",
                          d.skewness, rm["skewness"], TOL_TIGHT,
                          scipy=sps.skew(x, bias=False)))  # scipy type-2 skew
        cases.append(case(f"describe:{col}:kurtosis", "kurtosis_e1071_t2",
                          d.kurtosis, rm["kurtosis"], TOL_TIGHT))
    return cases


def main() -> None:
    cases = run()
    summ = summarize(cases)
    payload = {
        "module": "descriptive", "guarantee": "G1_correctness",
        "engine": "pystatistics", "py_version": PYVER,
        "r_versions": r_versions(), "reference": "R stats + e1071; scipy third ref",
        "summary": summ, "cases": cases,
    }
    out = write_artifact(ART / "descriptive_g1.json", payload)
    print(f"descriptive G1: clean {summ['n_clean_pass']}/{summ['n_clean']} match R "
          f"(worst_abs={summ['worst_clean_max_abs']:.2e}); "
          f"findings={summ['findings']} -> {out}")
    for c in cases:
        if not c["pass"] and not c.get("finding"):
            print("  UNEXPECTED FAIL", c["case"], c["quantity"], "abs=%.3e rel=%.3e"
                  % (c["max_abs"], c["max_rel"] or -1))


if __name__ == "__main__":
    main()
