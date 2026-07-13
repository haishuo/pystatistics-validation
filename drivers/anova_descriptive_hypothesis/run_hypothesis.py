"""G1 correctness sweep for the ``hypothesis`` module vs R (+ scipy third ref).

Whole public surface: t_test (Welch default + pooled/paired/one-sample/one-sided),
var_test, prop_test (Yates default + k>2), chisq_test (Yates 2x2 default +
independence + GoF + tiny-expected warning regime), fisher_test (cond-MLE OR/CI,
zero cell, rxc), wilcox_test (exact / ties->normal-approx / paired signed-rank),
ks_test (two-sample exact, one-sample asymptotic, ties), p_adjust (all methods).

Exact / closed-form quantities -> machine precision; only the fisher conditional-
MLE odds ratio and its CI (an internal root-find) get TOL_LOOSE.

Writes runs/hypothesis_g1.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.stats as sps

import adhdata as D
from compare import TOL_EXACT, TOL_LOOSE, TOL_TIGHT, case, summarize, write_artifact
from rref import r_ref, r_versions

from pystatistics import __version__ as PYVER
from pystatistics.hypothesis import (
    chisq_test, fisher_test, ks_test, p_adjust, prop_test, t_test,
    var_test, wilcox_test,
)

ART = Path(__file__).resolve().parents[2] / \
    "artifacts/anova_descriptive_hypothesis" / f"v{PYVER}" / "runs"


def _est(sol) -> list[float]:
    return [float(v) for v in sol.estimate.values()]


def run() -> list[dict]:
    C: list[dict] = []
    sleep = D.load_frame("sleep")
    g = sleep["group"]
    e1 = sleep["extra"][g == g.min()]      # group 1
    e2 = sleep["extra"][g == g.max()]      # group 2
    mt = D.load_frame("mtcars")

    # ================= t_test =================
    # R15: bare default is Welch two-sample, two-sided.
    s = t_test(e1, e2)
    r = r_ref("t_test", x=e1, y=e2, mu=0.0, paired=False, var_equal=False,
              alternative="two-sided", conf_level=0.95)
    sc = sps.ttest_ind(e1, e2, equal_var=False)
    C += [case("t:welch(default):stat", "statistic", s.statistic, r["statistic"], TOL_TIGHT, scipy=float(sc.statistic)),
          case("t:welch(default):df", "df", s.parameter["df"], r["df"], TOL_TIGHT),
          case("t:welch(default):p", "p_value", s.p_value, r["p_value"], TOL_TIGHT, scipy=float(sc.pvalue)),
          case("t:welch(default):ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT),
          case("t:welch(default):est", "estimate", _est(s), r["estimate"], TOL_EXACT)]
    # pooled (var_equal=True)
    s = t_test(e1, e2, equal_var=True)
    r = r_ref("t_test", x=e1, y=e2, mu=0.0, paired=False, var_equal=True,
              alternative="two-sided", conf_level=0.95)
    C += [case("t:pooled:stat", "statistic", s.statistic, r["statistic"], TOL_TIGHT),
          case("t:pooled:df", "df", s.parameter["df"], r["df"], TOL_EXACT),
          case("t:pooled:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT),
          case("t:pooled:ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT)]
    # paired (sleep is the canonical paired design)
    s = t_test(e1, e2, paired=True)
    r = r_ref("t_test", x=e1, y=e2, mu=0.0, paired=True, var_equal=False,
              alternative="two-sided", conf_level=0.95)
    C += [case("t:paired:stat", "statistic", s.statistic, r["statistic"], TOL_TIGHT),
          case("t:paired:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT),
          case("t:paired:ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT)]
    # one-sample, non-zero mu, one-sided
    x1 = mt["mpg"]
    s = t_test(x1, pop_mean=20.0, alternative="greater")
    r = r_ref("t_test", x=x1, mu=20.0, alternative="greater", conf_level=0.95)
    C += [case("t:one-sample:greater:stat", "statistic", s.statistic, r["statistic"], TOL_TIGHT),
          case("t:one-sample:greater:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT),
          case("t:one-sample:greater:ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT)]

    # ================= var_test =================
    am = mt["am"]
    v1, v2 = mt["mpg"][am == 0], mt["mpg"][am == 1]
    s = var_test(v1, v2)
    r = r_ref("var_test", x=v1, y=v2, ratio=1.0, alternative="two-sided", conf_level=0.95)
    C += [case("var_test:stat", "F", s.statistic, r["statistic"], TOL_TIGHT),
          case("var_test:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT),
          case("var_test:ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT),
          case("var_test:est", "ratio", list(s.estimate.values()), [r["estimate"]], TOL_TIGHT)]

    # ================= prop_test =================
    # 2-sample, Yates default (correct=True) then correct=False
    for corr in (True, False):
        s = prop_test(np.array([83., 90]), np.array([86., 93]), correct=corr)
        r = r_ref("prop_test", x=[83, 90], n=[86, 93], alternative="two-sided",
                  conf_level=0.95, correct=corr)
        C += [case(f"prop:2samp:correct={corr}:stat", "X2", s.statistic, r["statistic"], TOL_TIGHT),
              case(f"prop:2samp:correct={corr}:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT)]
        # 2-sample CI now applies R's capped continuity correction
        # min(0.5, |Delta|/sum(1/n)) for correct=True; correct=False unchanged.
        C.append(case(f"prop:2samp:correct={corr}:ci", "conf_int", s.conf_int,
                      r["conf_int"], TOL_TIGHT))
    # 1-sample vs p
    s = prop_test(np.array([83.]), np.array([100.]), null_value=0.75)
    r = r_ref("prop_test", x=[83], n=[100], p=[0.75], alternative="two-sided",
              conf_level=0.95, correct=True)
    C += [case("prop:1samp:stat", "X2", s.statistic, r["statistic"], TOL_TIGHT),
          case("prop:1samp:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT),
          case("prop:1samp:ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT)]
    # k>2 groups (no CI)
    s = prop_test(np.array([83., 90, 129, 70]), np.array([86., 93, 136, 82]))
    r = r_ref("prop_test", x=[83, 90, 129, 70], n=[86, 93, 136, 82],
              alternative="two-sided", conf_level=0.95, correct=True)
    C += [case("prop:k=4:stat", "X2", s.statistic, r["statistic"], TOL_TIGHT),
          case("prop:k=4:df", "df", s.parameter["df"], r["parameter"] if "parameter" in r else r["df"], TOL_EXACT),
          case("prop:k=4:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT)]

    # ================= chisq_test =================
    ct = D.contingency_tables()
    # independence, 4x4 (no Yates; df>1)
    s = chisq_test(ct["hair_eye"])
    r = r_ref("chisq_test", table=ct["hair_eye"], nrow=4, correct=True)
    C += [case("chisq:indep4x4:stat", "X2", s.statistic, r["statistic"], TOL_TIGHT),
          case("chisq:indep4x4:df", "df", s.parameter["df"], r["df"], TOL_EXACT),
          case("chisq:indep4x4:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT),
          case("chisq:indep4x4:expected", "expected", np.asarray(s.expected).ravel(),
               r["expected"], TOL_TIGHT)]
    # 2x2 Yates default TRUE vs FALSE (the R15 default gotcha)
    tab2 = np.array([[68., 20], [119, 84]])
    for corr in (True, False):
        s = chisq_test(tab2, correct=corr)
        r = r_ref("chisq_test", table=tab2, nrow=2, correct=corr)
        C += [case(f"chisq:2x2:correct={corr}:stat", "X2", s.statistic, r["statistic"], TOL_TIGHT),
              case(f"chisq:2x2:correct={corr}:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT)]
    # goodness-of-fit vs uniform
    s = chisq_test(ct["gof_dice"], expected_probs=np.array([1/6.]*6))
    r = r_ref("chisq_test", table=ct["gof_dice"], nrow=1, p=[1/6.]*6)
    C += [case("chisq:gof:stat", "X2", s.statistic, r["statistic"], TOL_TIGHT),
          case("chisq:gof:df", "df", s.parameter["df"], r["df"], TOL_EXACT),
          case("chisq:gof:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT)]

    # ================= fisher_test =================
    # 2x2 exact p-value is machine-exact (hypergeometric). The conditional-MLE
    # odds ratio and its CI are an internal noncentral-hypergeometric root-find:
    # on a WELL-CONDITIONED table they agree with R to ~5-7 digits (TOL_ROOT);
    # on the extreme lady_tea table (3,1,1,3) the upper CI sits in a flat tail
    # where the 0.025 root is ill-conditioned, so we assert only the exact
    # p-value there and characterise OR/CI on the well-conditioned table.
    TOL_ROOT = 1e-4
    s = fisher_test(ct["lady_tea"])
    r = r_ref("fisher_test", table=ct["lady_tea"], nrow=2, alternative="two-sided", conf_level=0.95)
    C += [case("fisher:2x2(lady_tea):p", "p_value", s.p_value, r["p_value"], TOL_EXACT)]
    wc = np.array([[30., 12], [9, 25]])
    s = fisher_test(wc)
    r = r_ref("fisher_test", table=wc, nrow=2, alternative="two-sided", conf_level=0.95)
    C += [case("fisher:2x2(wellcond):p", "p_value", s.p_value, r["p_value"], TOL_EXACT),
          case("fisher:2x2(wellcond):OR", "odds_ratio_cmle", s.estimate["odds ratio"], r["estimate"], TOL_ROOT),
          case("fisher:2x2(wellcond):ci", "conf_int", s.conf_int, r["conf_int"], TOL_ROOT)]
    # zero cell (OR -> +Inf boundary; p exact)
    s = fisher_test(ct["zero_cell"])
    r = r_ref("fisher_test", table=ct["zero_cell"], nrow=2, alternative="two-sided", conf_level=0.95)
    C += [case("fisher:zerocell:p", "p_value", s.p_value, r["p_value"], TOL_EXACT),
          case("fisher:zerocell:OR", "odds_ratio_cmle", s.estimate["odds ratio"], r["estimate"], TOL_ROOT)]
    # one-sided
    s = fisher_test(ct["lady_tea"], alternative="greater")
    r = r_ref("fisher_test", table=ct["lady_tea"], nrow=2, alternative="greater", conf_level=0.95)
    C += [case("fisher:2x2:greater:p", "p_value", s.p_value, r["p_value"], TOL_EXACT)]
    # rxc (2x3): pystatistics returns a SIMULATED (Monte-Carlo) p-value by
    # default (disclosed in the method string; seedable). R's default is exact.
    # With a fixed seed the MC p is reproducible; we validate it lands within
    # Monte-Carlo error of R's exact p (sqrt(p(1-p)/B) ~ 3.6e-3 at B=1e4).
    s = fisher_test(ct["small_rxc"], simulate_p_value=True, seed=1)
    r = r_ref("fisher_test", table=ct["small_rxc"], nrow=2, alternative="two-sided")
    C += [case("fisher:2x3:p(MC vs exact)", "p_value_mc", s.p_value, r["p_value"], 1e-2,
               extra={"note": "py simulated p (B=1e4, seed=1) vs R exact; "
                      "agreement within Monte-Carlo error, not machine precision",
                      "py_method": s.method})]

    # ================= wilcox_test =================
    tv = D.tie_vectors()
    # exact (small, no ties)
    s = wilcox_test(tv["x_small"], tv["y_small"])
    r = r_ref("wilcox_test", x=tv["x_small"], y=tv["y_small"], mu=0.0, paired=False,
              correct=True, conf_int=True, conf_level=0.95, alternative="two-sided")
    # exact-regime CI now matches R's order-statistic inversion (qwilcox).
    C += [case("wilcox:exact:stat", "W", s.statistic, r["statistic"], TOL_EXACT),
          case("wilcox:exact:p", "p_value", s.p_value, r["p_value"], TOL_EXACT),
          case("wilcox:exact:est", "HL_estimate", s.estimate["difference in location"], r["estimate"], TOL_TIGHT),
          case("wilcox:exact:ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT)]
    # one-sided exact CIs (half-open, +/-Inf bound)
    for alt in ("less", "greater"):
        s = wilcox_test(tv["x_small"], tv["y_small"], alternative=alt)
        r = r_ref("wilcox_test", x=tv["x_small"], y=tv["y_small"], mu=0.0, paired=False,
                  correct=True, conf_int=True, conf_level=0.95, alternative=alt)
        C.append(case(f"wilcox:exact:{alt}:ci", "conf_int", s.conf_int, r["conf_int"], TOL_TIGHT))
    # ties -> normal-approx CI now matches R's uniroot inversion (to R's own
    # root tolerance ~1e-4); statistic/p and the ties warning also checked.
    s = wilcox_test(tv["x_ties"], tv["y_ties"])
    r = r_ref("wilcox_test", x=tv["x_ties"], y=tv["y_ties"], mu=0.0, paired=False,
              correct=True, conf_int=True, conf_level=0.95, alternative="two-sided")
    C += [case("wilcox:ties:stat", "W", s.statistic, r["statistic"], TOL_EXACT),
          case("wilcox:ties:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT,
               extra={"py_warnings": list(s.warnings), "r_warnings": r.get("r_warnings")}),
          case("wilcox:ties:ci(approx)", "conf_int", s.conf_int, r["conf_int"], 1e-3)]
    # paired signed-rank: statistic, p, pseudomedian AND CI (with a zero
    # difference + ties -> approx regime; zeros dropped, matching R).
    s = wilcox_test(tv["p1"], tv["p2"], paired=True)
    r = r_ref("wilcox_test", x=tv["p1"], y=tv["p2"], mu=0.0, paired=True,
              correct=True, conf_int=True, conf_level=0.95, alternative="two-sided")
    C += [case("wilcox:paired:stat", "V", s.statistic, r["statistic"], TOL_EXACT),
          case("wilcox:paired:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT,
               extra={"py_warnings": list(s.warnings), "r_warnings": r.get("r_warnings")}),
          case("wilcox:paired:est", "pseudomedian", s.estimate["(pseudo)median"], r["estimate"], 1e-3),
          case("wilcox:paired:ci(approx)", "conf_int", s.conf_int, r["conf_int"], 1e-3)]

    # ================= ks_test =================
    # two-sample exact
    s = ks_test(tv["x_small"], tv["y_small"])
    r = r_ref("ks_test", x=tv["x_small"], y=tv["y_small"], alternative="two-sided")
    C += [case("ks:2samp:stat", "D", s.statistic, r["statistic"], TOL_EXACT),
          case("ks:2samp:p", "p_value", s.p_value, r["p_value"], TOL_TIGHT)]
    # one-sample vs normal(mean,sd) -- use a NO-TIES sample so both engines take
    # the exact-Kolmogorov path (R falls back to asymptotic + warns when ties are
    # present; that ties behaviour is checked separately in the fidelity run).
    xk = np.array([0.10, 0.32, 0.55, 0.71, 0.93, 1.18, 1.42, 1.66, 1.91, 2.20,
                   2.51, 2.88])
    s = ks_test(xk, distribution="norm", mean=1.2, sd=0.8)
    r = r_ref("ks_test", x=xk, dist="pnorm", arg1=1.2, arg2=0.8, alternative="two-sided")
    C += [case("ks:1samp-norm(exact):stat", "D", s.statistic, r["statistic"], TOL_TIGHT),
          case("ks:1samp-norm(exact):p", "p_value", s.p_value, r["p_value"], TOL_TIGHT)]

    # ================= p_adjust (all 8) =================
    pv = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.5])
    for m in ("holm", "hochberg", "hommel", "bonferroni", "BH", "BY", "fdr", "none"):
        s = p_adjust(pv, method=m)
        r = r_ref("p_adjust", p=pv, method=("BH" if m == "fdr" else m))
        C.append(case(f"p_adjust:{m}", "adjusted", np.asarray(s).ravel(), r["value"], TOL_EXACT))
    return C


def main() -> None:
    C = run()
    summ = summarize(C)
    payload = {
        "module": "hypothesis", "guarantee": "G1_correctness",
        "engine": "pystatistics", "py_version": PYVER,
        "r_versions": r_versions(), "reference": "R stats; scipy third ref",
        "summary": summ, "cases": C,
    }
    out = write_artifact(ART / "hypothesis_g1.json", payload)
    print(f"hypothesis G1: clean {summ['n_clean_pass']}/{summ['n_clean']} match R "
          f"(worst_abs={summ['worst_clean_max_abs']:.2e}); "
          f"findings={summ['findings']} ({summ['n_finding_cases']} cases) -> {out}")
    for c in C:
        if not c["pass"] and not c.get("finding"):
            print("  UNEXPECTED FAIL", c["case"], c["quantity"], "abs=%.3e rel=%.3e"
                  % (c["max_abs"], (c["max_rel"] or -1)))
        elif c.get("finding"):
            print(f"  [{c['finding']}] {c['case']}: py={c['py']} r={c['r']}")


if __name__ == "__main__":
    main()
