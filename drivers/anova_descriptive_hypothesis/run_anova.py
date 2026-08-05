"""G1 correctness sweep for the ``anova`` module vs R (car / afex / TukeyHSD).

Whole public surface: anova_oneway (Type I), anova (Type II default, + I/III),
levene_test (center=median default, car Brown-Forsythe), anova_posthoc (TukeyHSD
studentized-range q), anova_rm (within-F + Mauchly + GG/HF vs afex).

Red-team (R10): an UNBALANCED two-way design where SS Type I, II and III DIVERGE
-- each pystatistics ss_type is validated against the MATCHING R type, proving no
silent SS-type substitution (the classic trap).

Writes runs/anova_g1.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import adhdata as D
from compare import TOL_EXACT, TOL_TIGHT, case, summarize, write_artifact
from rref import r_ref, r_versions

from pystatistics import __version__ as PYVER
from pystatistics.anova import (
    anova, anova_oneway, anova_posthoc, anova_rm, levene_test,
)

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_root import artifact_root  # noqa: E402

ART = artifact_root(Path(__file__).resolve().parents[2]) / \
    "anova_descriptive_hypothesis" / f"v{PYVER}" / "runs"


def _canon(term: str) -> str:
    """Canonicalize an interaction term so 'dose:supp' == 'supp:dose'."""
    return ":".join(sorted(term.split(":")))


def _py_terms(sol) -> dict[str, dict]:
    """Map an AnovaSolution's non-residual rows by canonical term name."""
    out = {}
    for row in sol.table:
        if row.term == "Residuals":
            continue
        out[_canon(row.term)] = {"df": row.df, "ss": row.sum_sq,
                                 "F": row.f_value, "p": row.p_value}
    return out


def _r_terms(r: dict) -> dict[str, dict]:
    terms = r["terms"] if isinstance(r["terms"], list) else [r["terms"]]
    def col(k):
        v = r[k]
        return v if isinstance(v, list) else [v]
    df, ss, F, p = col("df"), col("ss"), col("F"), col("p_value")
    return {_canon(t): {"df": df[i], "ss": ss[i], "F": F[i], "p": p[i]}
            for i, t in enumerate(terms)}


def run() -> list[dict]:
    C: list[dict] = []
    pg = D.load_frame("PlantGrowth")
    isp = D.load_frame("InsectSprays")
    tg = D.load_frame("ToothGrowth")

    # ================= anova_oneway (Type I) =================
    for name, fr, resp, grp in (("PlantGrowth", pg, "weight", "group__labels"),
                                ("InsectSprays", isp, "count", "spray__labels")):
        o = anova_oneway(fr[resp], fr[grp])
        r = r_ref("anova_oneway", y=fr[resp], group=fr[grp])
        row = o.table[0]
        C += [case(f"oneway:{name}:df", "df", row.df, r["df"], TOL_EXACT),
              case(f"oneway:{name}:ss", "sum_sq", row.sum_sq, r["ss"], TOL_TIGHT),
              case(f"oneway:{name}:F", "F", row.f_value, r["F"], TOL_TIGHT),
              case(f"oneway:{name}:p", "p_value", row.p_value, r["p_value"], TOL_TIGHT),
              case(f"oneway:{name}:ss_resid", "ss_resid", o.residual_ss, r["ss_resid"], TOL_TIGHT)]

    # A factorial comparison: drive R's formula order from py's OWN term order so
    # the sequential Type I is validated on the identical sequence (py orders
    # main effects alphabetically; Type II/III are order-invariant).
    def factorial_cases(prefix, y, factors, t):
        o = anova(y, factors, ss_type=t)
        main = [r.term for r in o.table
                if r.term != "Residuals" and ":" not in r.term]      # py's order
        rhs = "*".join(main)
        r = r_ref("anova_factorial", y=y, factors=factors,
                  formula_rhs=rhs, ss_type=t)
        pyt, rt = _py_terms(o), _r_terms(r)
        assert set(pyt) == set(rt), f"term mismatch {set(pyt)} vs {set(rt)}"
        rows = []
        for term in pyt:
            rows += [case(f"{prefix}:type{t}:{term}:ss", "sum_sq", pyt[term]["ss"], rt[term]["ss"], TOL_TIGHT),
                     case(f"{prefix}:type{t}:{term}:F", "F", pyt[term]["F"], rt[term]["F"], TOL_TIGHT),
                     case(f"{prefix}:type{t}:{term}:p", "p_value", pyt[term]["p"], rt[term]["p"], TOL_TIGHT)]
        return rows, {term: pyt[term]["ss"] for term in pyt}

    # ================= anova two-way: BALANCED ToothGrowth =================
    # balanced -> Type I = II = III coincide; each py ss_type matches its R type.
    yb = tg["len"]
    supp_b = tg["supp__labels"]
    dose_b = np.array([str(d) for d in tg["dose"]], dtype=object)
    for t in (1, 2, 3):
        rows, _ = factorial_cases("balanced", yb, {"supp": supp_b, "dose": dose_b}, t)
        C += rows

    # ================= RED-TEAM: UNBALANCED two-way, types DIVERGE =========
    ub = D.unbalanced_toothgrowth()
    yu = ub["len"]
    supp_u = ub["supp"]
    dose_u = np.array([str(d) for d in ub["dose"]], dtype=object)
    type_ss = {}   # capture the main-effect SS per type to PROVE divergence
    for t in (1, 2, 3):
        rows, ss_by_term = factorial_cases("UNBAL", yu, {"supp": supp_u, "dose": dose_u}, t)
        C += rows
        type_ss[t] = ss_by_term
    # assert the types genuinely diverge on the 'supp' main effect (else the
    # red-team would be vacuous) -- record it as evidence.
    # Evidence the three SS types genuinely diverge: 'dose' is entered FIRST, so
    # its Type I SS (unadjusted) != Type II (adjusted for supp) != Type III. The
    # 'supp' main effect shows Type I == Type II here because supp is entered LAST
    # among main effects (Type II SS == Type I-when-entered-last, a known
    # identity) -- itself a correctness signal that the types are implemented
    # right, not swapped.
    d1, d2, d3 = (type_ss[t]["dose"] for t in (1, 2, 3))
    diverge = abs(d1 - d2) > 1e-6 and abs(d2 - d3) > 1e-6 and abs(d1 - d3) > 1e-6
    C.append({"case": "UNBAL:divergence-check", "quantity": "dose_SS_by_type[I,II,III]",
              "py": [d1, d2, d3], "r": None, "tol": None,
              "max_abs": 0.0, "max_rel": 0.0, "pass": bool(diverge),
              "extra": {"supp_SS_by_type": [type_ss[t]["supp"] for t in (1, 2, 3)],
                        "note": "dose SS diverges across Type I/II/III; each type "
                        "matched its R counterpart above -> no silent SS-type "
                        "substitution. supp I==II is the entered-last identity."}})

    # ================= levene_test (center=median default) =================
    for name, fr, resp, grp in (("InsectSprays", isp, "count", "spray__labels"),
                                ("PlantGrowth", pg, "weight", "group__labels")):
        for loc in ("median", "mean"):
            lv = levene_test(fr[resp], fr[grp], location=loc)
            r = r_ref("levene", y=fr[resp], group=fr[grp], center=loc)
            C += [case(f"levene:{name}:{loc}:F", "F", lv.f_value, r["F"], TOL_TIGHT),
                  case(f"levene:{name}:{loc}:p", "p_value", lv.p_value, r["p_value"], TOL_TIGHT),
                  case(f"levene:{name}:{loc}:df", "df_between",
                       lv.df_between, r["df1"], TOL_EXACT)]

    # ================= anova_posthoc TukeyHSD =================
    for name, fr, resp, grp in (("PlantGrowth", pg, "weight", "group__labels"),):
        o = anova_oneway(fr[resp], fr[grp])
        ph = anova_posthoc(o, method="tukey")
        r = r_ref("tukey", y=fr[resp], group=fr[grp], conf_level=0.95)
        # align by comparison label ("g2-g1" in R)
        r_map = {}
        comps = r["comparisons"] if isinstance(r["comparisons"], list) else [r["comparisons"]]
        for i, comp in enumerate(comps):
            r_map[comp] = {k: (r[k][i] if isinstance(r[k], list) else r[k])
                           for k in ("diff", "lwr", "upr", "p_adj")}
        for pc in ph.comparisons:
            key = f"{pc.group2}-{pc.group1}"
            assert key in r_map, f"missing R comparison {key} in {list(r_map)}"
            rr = r_map[key]
            C += [case(f"tukey:{name}:{key}:diff", "diff", pc.diff, rr["diff"], TOL_TIGHT),
                  case(f"tukey:{name}:{key}:lwr", "ci_lower", pc.ci_lower, rr["lwr"], TOL_TIGHT),
                  case(f"tukey:{name}:{key}:upr", "ci_upper", pc.ci_upper, rr["upr"], TOL_TIGHT),
                  case(f"tukey:{name}:{key}:padj", "p_adj", pc.p_value, rr["p_adj"], TOL_TIGHT)]

    # ================= anova_rm (Mauchly + GG/HF vs afex) =================
    rm = D.rm_design()
    n, k, Y = rm["n"], rm["k"], rm["Y"]
    y = Y.ravel(order="F")
    subj = np.tile(np.arange(1, n + 1), k).astype(str)
    cond = np.repeat(np.arange(1, k + 1), n).astype(str)
    rr = anova_rm(y, subj, within={"cond": cond})
    r = r_ref("anova_rm", Y=Y, n=n, k=k)
    row = rr.table[0]
    sph = rr.sphericity[0]
    # HF epsilon: pystatistics reports the value actually USED (capped at 1.0,
    # per Huynh-Feldt); afex reports the raw uncapped estimate (here 1.117).
    # Validate py == min(1, raw); the HF-CORRECTED p-value (rm:p_hf) is what
    # affects any decision and matches R directly.
    hf_capped = min(1.0, r["hf_eps"])
    C += [case("rm:within:df1", "df1", row.df, r["df1"], TOL_EXACT),
          case("rm:within:F", "F", row.f_value, r["F"], TOL_TIGHT),
          case("rm:within:p_uncorr", "p_value", row.p_value, r["p_value"], TOL_TIGHT),
          case("rm:mauchly:W", "mauchly_w", sph.mauchly_w, r["mauchly_W"], TOL_TIGHT),
          # Mauchly p now includes R's second-order (Box) correction -> matches.
          case("rm:mauchly:p", "mauchly_p", sph.p_value, r["mauchly_p"], TOL_TIGHT),
          case("rm:gg_epsilon", "gg_eps", sph.gg_epsilon, r["gg_eps"], TOL_TIGHT),
          case("rm:hf_epsilon(capped)", "hf_eps_min1", sph.hf_epsilon, hf_capped, TOL_TIGHT,
               extra={"note": "py reports capped HF=min(1,raw); afex raw=%.6f" % r["hf_eps"]}),
          case("rm:p_gg", "gg_p_value", row.gg_p_value, r["p_gg"], TOL_TIGHT),
          case("rm:p_hf", "hf_p_value", row.hf_p_value, r["p_hf"], TOL_TIGHT)]
    return C


def main() -> None:
    C = run()
    summ = summarize(C)
    payload = {
        "module": "anova", "guarantee": "G1_correctness",
        "engine": "pystatistics", "py_version": PYVER,
        "r_versions": r_versions(),
        "reference": "R stats(aov) + car(Anova/leveneTest) + TukeyHSD + afex(aov_ez)",
        "summary": summ, "cases": C,
    }
    out = write_artifact(ART / "anova_g1.json", payload)
    print(f"anova G1: clean {summ['n_clean_pass']}/{summ['n_clean']} match R "
          f"(worst_abs={summ['worst_clean_max_abs']:.2e}); findings={summ['findings']} -> {out}")
    for c in C:
        if not c["pass"] and not c.get("finding"):
            print("  UNEXPECTED FAIL", c["case"], c["quantity"], "abs=%.3e rel=%.3e"
                  % (c["max_abs"] or -1, (c["max_rel"] or -1)))


if __name__ == "__main__":
    main()
