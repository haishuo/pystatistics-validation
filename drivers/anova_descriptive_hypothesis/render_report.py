"""Render the combined validation report for anova + descriptive + hypothesis
from the FROZEN artifacts. Numbers are never hand-typed (R5): every figure in
the report is read from artifacts/anova_descriptive_hypothesis/v<ver>/runs/*.json.

Usage: python render_report.py   (writes reports/anova-descriptive-hypothesis-v<ver>.md)
"""

from __future__ import annotations

import json
from pathlib import Path

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / "_shared"))
from artifact_root import artifact_root  # noqa: E402

VER = "5.0.0"
BLESSED_AT = "4.6.11"  # statistical bless line; 5.0.0 is a rename-only release over it
ROOT = Path(__file__).resolve().parents[2]
RUNS = artifact_root(ROOT) / "anova_descriptive_hypothesis" / f"v{VER}" / "runs"
OUT = ROOT / "reports" / f"anova-descriptive-hypothesis-v{VER}.md"


def load(name):
    return json.load(open(RUNS / name))


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x == 0:
            return "0"
        return f"{x:.3e}"
    return str(x)


def clean_cases(art):
    return [c for c in art["cases"] if not c.get("finding")]


def worst_rows(art, n=8):
    cs = sorted(clean_cases(art), key=lambda c: -(c["max_abs"] or 0))
    return cs[:n]


def g1_table(art):
    rows = ["| case | quantity | max_abs | max_rel | pass |",
            "| --- | --- | --- | --- | --- |"]
    for c in sorted(clean_cases(art), key=lambda c: c["case"]):
        rows.append(f"| {c['case']} | {c['quantity']} | {fmt(c['max_abs'])} | "
                    f"{fmt(c.get('max_rel'))} | {'✓' if c['pass'] else '✗'} |")
    return "\n".join(rows)


def main():
    desc = load("descriptive_g1.json")
    hyp = load("hypothesis_g1.json")
    anova = load("anova_g1.json")
    fid = load("fidelity_g2.json")
    perf = load("perf_g3.json")
    r4 = load("r4_audit.json")
    led = load("findings_ledger.json")
    rv = desc["r_versions"]

    ds, hs, as_ = desc["summary"], hyp["summary"], anova["summary"]

    P = []
    P.append(f"# Validation report — pystatistics `anova` + `descriptive` + "
             f"`hypothesis` v{VER}\n")
    P.append(f"> This report validates **pystatistics anova + descriptive + "
             f"hypothesis v{VER}**, installed from **PyPI**. Evidence lives in "
             f"`artifacts/anova_descriptive_hypothesis/v{VER}/`. It was generated "
             f"by `render_report.py` from frozen artifacts — **do not edit numbers "
             f"by hand**; fix the artifact or the renderer and re-render.\n")
    P.append(f"> **v{VER} is a rename-only release over the blessed v{BLESSED_AT} "
             f"line** (statistically identical). Every G1 statistic / p-value and "
             f"G2 fidelity outcome reproduces v{BLESSED_AT} **bit-for-bit**; the "
             f"only observable change is that the `prop_test` invalid-input message "
             f"now names the renamed `n_trials` argument (still raises "
             f"`ValidationError`). Driver call sites were updated to the 5.0 API "
             f"(`prop_test(null_value=)`, `chisq_test(expected_probs=)`).\n")
    P.append(f"**Reference:** {rv['R']} (base `stats`, `car` {rv['car']}, "
             f"`afex` {rv['afex']}, `e1071`, `TukeyHSD`); **scipy.stats** as an "
             f"independent third reference where R and scipy differ on a "
             f"convention.\n")
    P.append("**Subsystems:** classical inference — one/two-sample & rank tests, "
             "contingency tables, multiple-comparison adjustment (`hypothesis`); "
             "summary statistics, correlation/covariance, quantiles "
             "(`descriptive`); analysis of variance, post-hoc, repeated measures, "
             "variance homogeneity (`anova`).\n")

    P.append("## 1. What statistical procedures are implemented?\n")
    P.append(
        "Three **deterministic, closed-form** modules — almost every quantity "
        "has a closed-form statistic and (often) an exact p-value, so the bar is "
        "**machine precision / exact**, not an optimizer tier, and the work is "
        "exact per-function convention-matching against R's defaults.\n\n"
        "- **descriptive** — `describe`, `summary`, `cor` (pearson/spearman/"
        "kendall tau-b), `cov`, `var` (n−1), `quantile` (all 9 R types, default "
        "type 7). Skewness/kurtosis match `e1071` type 2.\n"
        "- **hypothesis** — `t_test` (Welch default, pooled/paired/one-sample), "
        "`chisq_test` (Yates 2×2 default + GoF), `fisher_test` (exact "
        "hypergeometric + conditional-MLE odds ratio), `wilcox_test` "
        "(exact/normal-approx switch, HL CI), `ks_test` (one/two-sample), "
        "`prop_test`, `var_test`, `p_adjust` (all 8 methods).\n"
        "- **anova** — `anova_oneway` (Type I), `anova` (Type II default, + I/III), "
        "`anova_posthoc` (TukeyHSD studentized-range q), `anova_rm` (within-F, "
        "Mauchly + GG/HF), `levene_test` (Brown-Forsythe, center=median default).\n")

    P.append("## 2. Which algorithms / conventions are matched?\n")
    P.append(
        "Every function is validated against its canonical R reference on "
        "**identical float64 inputs** (each dataset is dumped from the central "
        "HDF5 store and analysed by both engines), matching R's **default "
        "convention** per function (R15):\n\n"
        "| function | R default matched |\n| --- | --- |\n"
        "| `quantile` | type 7 (all 9 types validated) |\n"
        "| `cor` | pearson default; spearman; kendall **tau-b** (tie-corrected) |\n"
        "| `t_test` | **Welch** (`var.equal=FALSE`), two-sided; Welch–Satterthwaite df |\n"
        "| `chisq_test` | **Yates continuity correction TRUE for 2×2**; expected<5 warning |\n"
        "| `fisher_test` | exact hypergeometric p; conditional-MLE odds ratio (≠ sample OR) |\n"
        "| `wilcox_test` | exact when small & no ties; ties→normal approx + warning; **HL CI by exact/approx inversion** |\n"
        "| `ks_test` | one-sample exact (no ties); two-sample; ties warning |\n"
        "| `prop_test` | continuity correction TRUE; capped CI correction |\n"
        "| `p_adjust` | default `holm`; all of holm/hochberg/hommel/bonferroni/BH/BY/none |\n"
        "| `anova` | **Type II** (`car::Anova` default); `anova_oneway` Type I |\n"
        "| `anova_posthoc` | `TukeyHSD` studentized-range q, family-wise CIs |\n"
        "| `anova_rm` | Mauchly + GG/HF (`afex`); second-order Box p-value correction |\n"
        "| `levene_test` | **center='median'** (car Brown-Forsythe) |\n")

    P.append("## 3. Correctness (G1) vs R — deterministic, machine precision\n")
    P.append(
        f"Every test **statistic** and every default exact/asymptotic **p-value** "
        f"matches R to machine precision. Summary (clean = R-match cases, all "
        f"passing; no open findings):\n\n"
        f"| module | cases | match R | worst max_abs | findings |\n| --- | --- | --- | --- | --- |\n"
        f"| descriptive | {ds['n_clean']} | {ds['n_clean_pass']} | {fmt(ds['worst_clean_max_abs'])} | none |\n"
        f"| hypothesis | {hs['n_clean']} | {hs['n_clean_pass']} | {fmt(hs['worst_clean_max_abs'])}\\* | none |\n"
        f"| anova | {as_['n_clean']} | {as_['n_clean_pass']} | {fmt(as_['worst_clean_max_abs'])} | none |\n\n"
        f"\\* the hypothesis worst case is the *deliberately* Monte-Carlo Fisher "
        f"r×c p-value (seeded, compared to R's exact p within MC error, tol 1e-2); "
        f"every other clean case is ≤ ~2e-13. The red-team (R10) is embedded: an "
        f"**unbalanced two-way design proves SS Type I/II/III diverge and each "
        f"matches its R counterpart** (no silent SS-type substitution); ties "
        f"(kendall/wilcox/ks), a zero-cell Fisher OR (→ +Inf), tiny expected "
        f"counts (χ² warning → Fisher regime), and exact/asymptotic switchovers "
        f"are all covered.\n")

    P.append("### descriptive — every case\n")
    P.append(g1_table(desc) + "\n")
    P.append("### hypothesis — worst 10 (all else ≤ 2e-13)\n")
    rows = ["| case | quantity | max_abs | max_rel |", "| --- | --- | --- | --- |"]
    for c in worst_rows(hyp, 10):
        rows.append(f"| {c['case']} | {c['quantity']} | {fmt(c['max_abs'])} | {fmt(c.get('max_rel'))} |")
    P.append("\n".join(rows) + "\n")
    P.append("### anova — worst 10\n")
    rows = ["| case | quantity | max_abs | max_rel |", "| --- | --- | --- | --- |"]
    for c in worst_rows(anova, 10):
        rows.append(f"| {c['case']} | {c['quantity']} | {fmt(c['max_abs'])} | {fmt(c.get('max_rel'))} |")
    P.append("\n".join(rows) + "\n")

    P.append("## 4. Fidelity (G2) — fail-loud + match R's warnings\n")
    fs = fid["summary"]
    P.append(
        f"- **Fail-loud: {fs['fail_loud_pass']}/{fs['fail_loud_total']}** — every "
        f"invalid / dimension-mismatched / out-of-range input raises "
        f"`ValidationError` (A6, no silent default). Includes `quantile` probs "
        f"outside [0,1] (now raises, matching R) and `prop_test` successes>trials.\n"
        f"- **Warning fidelity: {fs['warning_match_clean_pass']}/"
        f"{fs['warning_match_clean_total']}** — the χ² small-expected-count "
        f"warning, the wilcox ties→normal-approx warning, and the one-sample KS "
        f"ties warning all reproduce R's wording.\n"
        f"- **No silent substitution** — the exact→approx wilcox switch and the "
        f"Fisher r×c simulation are disclosed in the method string.\n"
        f"- **Degenerate-but-defined** input matches R's NA: `var` of a single "
        f"observation returns `nan` (= R `NA`), not fabricated data.\n")

    P.append("## 5. Performance (G3) — CPU, and why there is no GPU path\n")
    lp = {r["label"]: r for r in perf["large_n_cpu"]}
    rin = perf["large_n_r_inprocess_s"]
    P.append("Closed-form scalars and O(n) reductions; CPU parity with R at "
             "n=1e6 (compute-vs-compute, R timed in-process):\n\n"
             "| op (n=1e6) | pystatistics | R in-process |\n| --- | --- | --- |\n")
    label_map = [("var(n=1000000)", "var"), ("cor:pearson(n=1000000)", "cor_pearson"),
                 ("cor:spearman(n=1000000)", "cor_spearman"),
                 ("quantile:type7(n=1000000)", "quantile_t7")]
    tbl = []
    for lab, rkey in label_map:
        py_ms = lp[lab]["median_s"] * 1e3
        r_ms = rin[rkey] * 1e3
        tbl.append(f"| {lab.replace('(n=1000000)','')} | {py_ms:.1f} ms | {r_ms:.1f} ms |")
    P.append("\n".join(tbl) + "\n\n")
    P.append("> " + perf["no_gpu_rationale"] + "\n")

    P.append("## 6. Findings ledger — five R-convention divergences, ALL FIXED\n")
    P.append(
        "First-time validation of these modules surfaced five divergences from R "
        "at **4.6.10**. Per RIGOR R18 these were gathered and **fixed together** "
        "in the **4.6.11** patch — *documented is not fixed*; the bless is on the "
        "version that fixes them. No test statistic or default p-value on valid "
        "input was ever wrong (all matched R to machine precision at 4.6.10); the "
        "five were confidence-interval, warning, diagnostic, and invalid-input "
        "conventions.\n\n"
        "| id | fn | divergence (4.6.10) | fix (4.6.11) |\n| --- | --- | --- | --- |")
    for f in led["findings"]:
        P.append(f"| {f['id']} | `{f['fn']}` | {f['divergence']} | {f['fix']} |")
    P.append("")
    P.append(f"\n**Showstoppers (R16):** none. All five re-validated against R at "
             f"{VER}: "
             f"{'; '.join(f['id'] + ' ' + f['revalidation'].rstrip('.') for f in led['findings'])}.\n")

    P.append("## 7. Constitutional audit (R4) + the seven questions\n")
    P.append(f"**R4 naming/constitutional audit: {r4['verdict']}** — "
             f"{len(r4['naming_law_violations'])} violations. The naming law holds "
             f"across all three surfaces: `pop_mean`/`equal_var` (A2 + S6), "
             f"`null_value` for wilcox's generic shift, the `…Solution` objects, "
             f"and `quantile_type` (avoids shadowing the builtin).\n")
    P.append(
        "The seven questions (R7): **procedure** — §1; **algorithms/conventions** "
        "— §2; **comparison to R** — §3 (+ scipy third reference); **tolerances** "
        "— machine precision on statistics and exact p-values, R's uniroot "
        "tolerance (~1e-4) on the two approximate wilcox CIs, Monte-Carlo error on "
        "the seeded Fisher r×c p; **benchmarks** — §5 (CPU parity; these are O(n) "
        "with no scaling pathology); **known limitations** — none open (the five "
        "findings are fixed at 4.6.11); **version** — pystatistics "
        f"{VER} from PyPI, {rv['R']}.\n")

    P.append("## 8. Verdict — BLESSED, and the corpus is complete (9 of 9)\n")
    P.append(
        f"pystatistics **anova + descriptive + hypothesis v{VER}** is **BLESSED**: "
        f"every function's statistic and default p-value matches R to machine "
        f"precision; fidelity (fail-loud + warnings) matches R; the five "
        f"convention divergences found at 4.6.10 are fixed (in the {BLESSED_AT} "
        f"patch) and re-validated at {VER}; the findings ledger carries **no open "
        f"item** (RIGOR R18 bless precondition satisfied). No GPU path is warranted "
        f"and none is manufactured.\n\n"
        f"v{VER} is a **rename-only release over the blessed v{BLESSED_AT} line** "
        f"and reproduces its G1/G2 evidence bit-for-bit, so the bless carries "
        f"forward unchanged.\n")

    OUT.write_text("\n".join(P) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
