"""
Cross-validation of pystatsbio against R reference results.

Compares every numerical output from pystatsbio against pre-computed R
results (stored in data/pystatsbio_r_results.json).

Prerequisites:
  1. Run `Rscript tests/pystatsbio_reference_r.R` to generate reference data
  2. pip install pystatsbio pandas

R packages used for reference:
  - NonCompart  (NCA: AUC, Cmax, half-life, CL, Vz)
  - drc         (dose-response: LL.4 fit, ED50)
  - pROC        (ROC curves: AUC, DeLong CI, DeLong test)
  - pwr         (power: t-test, ANOVA, proportions)
  - PowerTOST   (crossover bioequivalence)

Usage:
  python tests/test_pystatsbio_vs_r.py
"""

import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

passed = 0
failed = 0

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pystatsbio_r"

# Tolerances
RTOL_TIGHT = 1e-6    # For algorithms that should match R closely (NCA, power)
RTOL_FIT   = 1e-3    # For optimization-dependent results (curve fitting params)
ATOL       = 1e-10   # Absolute tolerance


def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def check(label, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {label}")
        passed += 1
    except Exception:
        print(f"  [FAIL] {label}")
        traceback.print_exc()
        failed += 1


def assert_close(py_val, r_val, label, rtol=RTOL_TIGHT, atol=ATOL):
    """Compare scalar or array; print both on failure."""
    py_arr = np.atleast_1d(np.asarray(py_val, dtype=float))
    r_arr = np.atleast_1d(np.asarray(r_val, dtype=float))
    np.testing.assert_allclose(
        py_arr, r_arr, rtol=rtol, atol=atol,
        err_msg=f"{label}: pystatsbio={py_arr} vs R={r_arr}",
    )


# ===================================================================
# Load data & R reference
# ===================================================================
section("Loading data & R reference")

R_RESULTS = json.loads((DATA_DIR / "pystatsbio_r_results.json").read_text())
theoph = pd.read_csv(DATA_DIR / "theoph.csv")
ryegrass = pd.read_csv(DATA_DIR / "ryegrass.csv")
asah = pd.read_csv(DATA_DIR / "asah.csv")

print(f"  R results keys: {len(R_RESULTS)}")
print(f"  Theoph: {theoph.shape[0]} rows")
print(f"  ryegrass: {ryegrass.shape[0]} rows")
print(f"  aSAH: {asah.shape[0]} rows")


# ===================================================================
# 1. PK — NCA Subject 1 vs R (NonCompart)
# ===================================================================
section("PK: NCA Subject 1 vs R")

s1 = theoph[theoph["Subject"] == 1]
time_s1 = s1["Time"].values.astype(float)
conc_s1 = s1["conc"].values.astype(float)
dose_s1 = float(s1["Dose"].iloc[0])


def test_nca_cmax_vs_r():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    r = R_RESULTS["nca_subject1"]
    assert_close(result.cmax, r["cmax"], "Cmax")
    assert_close(result.tmax, r["tmax"], "Tmax")
    print(f"    Cmax: py={result.cmax:.4f}, R={r['cmax']}")
    print(f"    Tmax: py={result.tmax:.4f}, R={r['tmax']}")


def test_nca_auc_vs_r():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    r = R_RESULTS["nca_subject1"]
    assert_close(result.auc_last, r["auc_last"], "AUC(0-last)")
    assert_close(result.auc_inf, r["auc_inf"], "AUC(0-inf)")
    assert_close(result.auc_pct_extrap, r["pct_extrap"], "% extrapolated")
    print(f"    AUC(0-last): py={result.auc_last:.4f}, R={r['auc_last']}")
    print(f"    AUC(0-inf):  py={result.auc_inf:.4f}, R={r['auc_inf']}")


def test_nca_terminal_vs_r():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    r = R_RESULTS["nca_subject1"]
    assert_close(result.lambda_z, r["lambda_z"], "lambda_z")
    assert_close(result.half_life, r["half_life"], "t1/2")
    print(f"    lambda_z: py={result.lambda_z:.6f}, R={r['lambda_z']}")
    print(f"    t1/2:     py={result.half_life:.4f}, R={r['half_life']}")


def test_nca_cl_vz_vs_r():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    r = R_RESULTS["nca_subject1"]
    # Unit convention difference:
    # NonCompart assumes dose in mg, concentration in µg/L, and applies a
    # ×1000 factor to CL and Vz (converting the raw dose/AUC ratio to L/h).
    # pystatsbio is unit-agnostic: CL = dose / AUC directly, so the user
    # must ensure consistent units.  The underlying formula is identical;
    # the ×1000 is purely NonCompart's default unit scaling.
    assert_close(result.clearance * 1000, r["cl_f"], "CL/F (×1000 for R units)")
    assert_close(result.vz * 1000, r["vz_f"], "Vz/F (×1000 for R units)")
    print(f"    CL/F: py={result.clearance:.6f}, R={r['cl_f']:.4f}")
    print(f"    Vz/F: py={result.vz:.6f}, R={r['vz_f']:.4f}")
    print(f"      (R = py × 1000; NonCompart default µg/L→mg unit scaling)")


check("NCA Cmax & Tmax vs R", test_nca_cmax_vs_r)
check("NCA AUC(0-last), AUC(0-inf) vs R", test_nca_auc_vs_r)
check("NCA lambda_z & t1/2 vs R", test_nca_terminal_vs_r)
check("NCA CL/F & Vz/F vs R", test_nca_cl_vz_vs_r)


# ===================================================================
# 2. PK — NCA all 12 subjects vs R
# ===================================================================
section("PK: NCA all subjects vs R")


def test_nca_all_cmax_vs_r():
    from pystatsbio.pk import nca
    r_all = R_RESULTS["nca_all_subjects"]
    for subj in sorted(theoph["Subject"].unique()):
        si = theoph[theoph["Subject"] == subj]
        result = nca(
            si["Time"].values.astype(float),
            si["conc"].values.astype(float),
            dose=float(si["Dose"].iloc[0]),
            route="ev",
        )
        r = r_all[str(subj)]
        assert_close(result.cmax, r["cmax"], f"Subject {subj} Cmax")
        assert_close(result.auc_last, r["auc_last"], f"Subject {subj} AUC(0-last)")
    print(f"    All 12 subjects: Cmax & AUC(0-last) match R")


def test_nca_all_terminal_vs_r():
    from pystatsbio.pk import nca
    r_all = R_RESULTS["nca_all_subjects"]
    matched = 0
    for subj in sorted(theoph["Subject"].unique()):
        si = theoph[theoph["Subject"] == subj]
        result = nca(
            si["Time"].values.astype(float),
            si["conc"].values.astype(float),
            dose=float(si["Dose"].iloc[0]),
            route="ev",
        )
        r = r_all[str(subj)]
        if result.half_life is not None and r["half_life"] is not None:
            # Terminal phase point selection can differ between implementations,
            # leading to slightly different lambda_z / t1/2 for some subjects.
            assert_close(result.half_life, r["half_life"],
                         f"Subject {subj} t1/2", rtol=0.05)
            matched += 1
    print(f"    {matched}/12 subjects: t1/2 matches R")


check("NCA all subjects: Cmax & AUC vs R", test_nca_all_cmax_vs_r)
check("NCA all subjects: t1/2 vs R", test_nca_all_terminal_vs_r)


# ===================================================================
# 3. Dose-response — LL.4 fit vs R (drc)
# ===================================================================
section("Dose-response: LL.4 fit vs R")

dose_rg = ryegrass["conc"].values.astype(float)
resp_rg = ryegrass["rootl"].values.astype(float)


def test_drm_params_vs_r():
    from pystatsbio.doseresponse import fit_drm
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    r = R_RESULTS["drm_ll4"]
    p = result.params
    # drc convention: b=hill (always positive), c=lower limit, d=upper limit, e=ec50
    # pystatsbio: hill can be negative (for decreasing), bottom=lower, top=upper
    # So: |py.hill| ≈ R.b, py.bottom ≈ R.c, py.top ≈ R.d, py.ec50 ≈ R.e
    assert_close(abs(p.hill), abs(r["hill"]), "|hill| (b)", rtol=RTOL_FIT)
    assert_close(p.bottom, r["bottom"], "bottom (c)", rtol=RTOL_FIT)
    assert_close(p.top, r["top"], "top (d)", rtol=RTOL_FIT)
    assert_close(p.ec50, r["ec50"], "ec50 (e)", rtol=RTOL_FIT)
    print(f"    |hill|: py={abs(p.hill):.6f}, R={r['hill']}")
    print(f"    bottom: py={p.bottom:.6f}, R={r['bottom']}")
    print(f"    top:    py={p.top:.6f}, R={r['top']}")
    print(f"    ec50:   py={p.ec50:.6f}, R={r['ec50']}")


def test_drm_rss_vs_r():
    from pystatsbio.doseresponse import fit_drm
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    r = R_RESULTS["drm_ll4"]
    assert_close(result.rss, r["rss"], "RSS", rtol=RTOL_FIT)
    print(f"    RSS: py={result.rss:.6f}, R={r['rss']}")


def test_drm_predictions_vs_r():
    from pystatsbio.doseresponse import fit_drm
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    r = R_RESULTS["drm_predictions"]
    pred_doses = np.array(r["doses"], dtype=float)
    py_pred = result.predict(pred_doses)
    r_pred = np.array(r["predicted"], dtype=float)
    assert_close(py_pred, r_pred, "predictions at unique doses", rtol=RTOL_FIT)
    print(f"    Predictions match R at {len(pred_doses)} dose levels")


check("LL.4 parameters vs R (drc)", test_drm_params_vs_r)
check("LL.4 RSS vs R", test_drm_rss_vs_r)
check("LL.4 predictions vs R", test_drm_predictions_vs_r)


# ===================================================================
# 4. Dose-response — EC50 vs R
# ===================================================================
section("Dose-response: EC50 vs R")


def test_ec50_vs_r():
    from pystatsbio.doseresponse import fit_drm, ec50
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    e = ec50(result)
    r = R_RESULTS["drm_ec50"]
    assert_close(e.estimate, r["estimate"], "EC50 estimate", rtol=RTOL_FIT)
    assert_close(e.se, r["se"], "EC50 SE", rtol=RTOL_FIT)
    # CI: both use raw-scale t-CI with residual df (matching drc::ED).
    # Small differences from SE (optimizer covariance matrix) propagate.
    assert_close(e.ci_lower, r["ci_lower"], "EC50 CI lower", rtol=RTOL_FIT)
    assert_close(e.ci_upper, r["ci_upper"], "EC50 CI upper", rtol=RTOL_FIT)
    print(f"    EC50: py={e.estimate:.6f}, R={r['estimate']}")
    print(f"    SE:   py={e.se:.6f}, R={r['se']}")
    print(f"    CI:   py=[{e.ci_lower:.4f}, {e.ci_upper:.4f}]")
    print(f"          R =[{r['ci_lower']:.4f}, {r['ci_upper']:.4f}]")


check("EC50 estimate, SE & CI vs R", test_ec50_vs_r)


# ===================================================================
# 5. Diagnostic — ROC AUC vs R (pROC)
# ===================================================================
section("Diagnostic: ROC AUC vs R")

response_asah = (asah["outcome"] == "Poor").astype(int).values
s100b = asah["s100b"].values.astype(float)
ndka = asah["ndka"].values.astype(float)


def test_roc_auc_vs_r():
    from pystatsbio.diagnostic import roc
    result = roc(response_asah, s100b, direction="<")
    r = R_RESULTS["roc_s100b"]
    assert_close(result.auc, r["auc"], "AUC (s100b)")
    print(f"    AUC: py={result.auc:.10f}, R={r['auc']}")


def test_roc_ci_vs_r():
    from pystatsbio.diagnostic import roc
    result = roc(response_asah, s100b, direction="<")
    r = R_RESULTS["roc_s100b"]
    # Both use DeLong normal CI (Wald interval on original scale)
    assert_close(result.auc_ci_lower, r["ci_lower"], "AUC CI lower")
    assert_close(result.auc_ci_upper, r["ci_upper"], "AUC CI upper")
    print(f"    CI lower: py={result.auc_ci_lower:.10f}, R={r['ci_lower']}")
    print(f"    CI upper: py={result.auc_ci_upper:.10f}, R={r['ci_upper']}")


def test_roc_counts_vs_r():
    from pystatsbio.diagnostic import roc
    result = roc(response_asah, s100b, direction="<")
    r = R_RESULTS["roc_s100b"]
    assert result.n_positive == r["n_positive"]
    assert result.n_negative == r["n_negative"]


check("ROC AUC vs R (pROC)", test_roc_auc_vs_r)
check("ROC DeLong CI vs R", test_roc_ci_vs_r)
check("ROC sample counts vs R", test_roc_counts_vs_r)


# ===================================================================
# 6. Diagnostic — Youden cutoff vs R
# ===================================================================
section("Diagnostic: Youden cutoff vs R")


def test_youden_vs_r():
    from pystatsbio.diagnostic import roc, optimal_cutoff
    roc_result = roc(response_asah, s100b, direction="<")
    cut = optimal_cutoff(roc_result, method="youden")
    r = R_RESULTS["roc_youden"]
    # Sensitivity and specificity at the Youden-optimal point must match exactly
    assert_close(cut.sensitivity, r["sensitivity"], "Youden sensitivity")
    assert_close(cut.specificity, r["specificity"], "Youden specificity")
    # Cutoff value convention differs: pystatsbio uses the observed data value,
    # pROC uses midpoint between adjacent observations.  Both give the same
    # classification (no data falls between 0.19 and 0.22).
    print(f"    cutoff: py={cut.cutoff:.4f}, R={r['cutoff']}")
    print(f"      (convention: py=observed value, pROC=midpoint of gap)")
    print(f"    sens:   py={cut.sensitivity:.10f}, R={r['sensitivity']}")
    print(f"    spec:   py={cut.specificity:.10f}, R={r['specificity']}")


check("Youden cutoff sens/spec vs R", test_youden_vs_r)


# ===================================================================
# 7. Diagnostic — accuracy at cutoff=0.205 vs R
# ===================================================================
section("Diagnostic: accuracy at cutoff=0.205 vs R")


def test_diag_accuracy_vs_r():
    from pystatsbio.diagnostic import diagnostic_accuracy
    result = diagnostic_accuracy(response_asah, s100b, cutoff=0.205, direction="<")
    r = R_RESULTS["diag_accuracy"]
    assert_close(result.sensitivity, r["sensitivity"], "sensitivity")
    assert_close(result.specificity, r["specificity"], "specificity")
    assert_close(result.ppv, r["ppv"], "PPV")
    assert_close(result.npv, r["npv"], "NPV")
    assert_close(result.lr_positive, r["lr_positive"], "LR+")
    assert_close(result.dor, r["dor"], "DOR")
    print(f"    Sens: py={result.sensitivity:.6f}, R={r['sensitivity']}")
    print(f"    Spec: py={result.specificity:.6f}, R={r['specificity']}")
    print(f"    PPV:  py={result.ppv:.6f}, R={r['ppv']}")
    print(f"    NPV:  py={result.npv:.6f}, R={r['npv']}")
    print(f"    LR+:  py={result.lr_positive:.6f}, R={r['lr_positive']}")
    print(f"    DOR:  py={result.dor:.6f}, R={r['dor']}")


check("Diagnostic accuracy vs R", test_diag_accuracy_vs_r)


# ===================================================================
# 8. Diagnostic — DeLong test vs R
# ===================================================================
section("Diagnostic: DeLong test vs R")


def test_delong_vs_r():
    from pystatsbio.diagnostic import roc, roc_test
    roc1 = roc(response_asah, s100b, direction="<")
    roc2 = roc(response_asah, ndka, direction="<")
    result = roc_test(
        roc1, roc2,
        predictor1=s100b, predictor2=ndka,
        response=response_asah,
    )
    r = R_RESULTS["delong_test"]
    # Note: sign of Z may differ (depends on subtraction order);
    # compare absolute value
    assert_close(abs(result.statistic), abs(r["statistic"]),
                 "DeLong |Z|", rtol=1e-4)
    assert_close(result.p_value, r["p_value"], "DeLong p-value", rtol=1e-4)
    print(f"    |Z|:  py={abs(result.statistic):.6f}, R={abs(r['statistic'])}")
    print(f"    p:    py={result.p_value:.6f}, R={r['p_value']}")


check("DeLong test Z & p vs R", test_delong_vs_r)


# ===================================================================
# 9. Power — t-test vs R (pwr)
# ===================================================================
section("Power: t-test vs R")


def test_power_ttest_n_vs_r():
    from pystatsbio.power import power_t_test
    result = power_t_test(d=0.5, power=0.80, alpha=0.05)
    r = R_RESULTS["power_ttest_n"]
    assert result.n == r["n_ceil"], (
        f"n: py={result.n}, R={r['n_ceil']}"
    )
    print(f"    n: py={result.n}, R={r['n_ceil']} (exact R={r['n_exact']:.4f})")


def test_power_ttest_power_vs_r():
    from pystatsbio.power import power_t_test
    result = power_t_test(n=64, d=0.5, alpha=0.05)
    r = R_RESULTS["power_ttest_power"]
    assert_close(result.power, r["power"], "power (n=64, d=0.5)")
    print(f"    power: py={result.power:.10f}, R={r['power']}")


def test_power_ttest_effect_vs_r():
    from pystatsbio.power import power_t_test
    result = power_t_test(n=100, power=0.80, alpha=0.05)
    r = R_RESULTS["power_ttest_effect"]
    # Root-finding convergence can differ slightly between implementations
    assert_close(result.effect_size, r["d"], "d (n=100, power=0.80)", rtol=1e-4)
    print(f"    d: py={result.effect_size:.10f}, R={r['d']}")


def test_power_ttest_onesided_vs_r():
    from pystatsbio.power import power_t_test
    result = power_t_test(d=0.5, power=0.80, alpha=0.05, alternative="greater")
    r = R_RESULTS["power_ttest_onesided"]
    assert result.n == r["n_ceil"], (
        f"n (one-sided): py={result.n}, R={r['n_ceil']}"
    )
    print(f"    n (one-sided): py={result.n}, R={r['n_ceil']}")


def test_power_ttest_onesample_vs_r():
    from pystatsbio.power import power_t_test
    result = power_t_test(d=0.5, power=0.80, alpha=0.05, type="one.sample")
    r = R_RESULTS["power_ttest_onesample"]
    assert result.n == r["n_ceil"], (
        f"n (one-sample): py={result.n}, R={r['n_ceil']}"
    )
    print(f"    n (one-sample): py={result.n}, R={r['n_ceil']}")


def test_power_ttest_paired_vs_r():
    from pystatsbio.power import power_paired_t_test
    result = power_paired_t_test(d=0.5, power=0.80, alpha=0.05)
    r = R_RESULTS["power_ttest_paired"]
    assert result.n == r["n_ceil"], (
        f"n (paired): py={result.n}, R={r['n_ceil']}"
    )
    print(f"    n (paired): py={result.n}, R={r['n_ceil']}")


check("t-test n vs R (pwr)", test_power_ttest_n_vs_r)
check("t-test power vs R", test_power_ttest_power_vs_r)
check("t-test effect size vs R", test_power_ttest_effect_vs_r)
check("t-test one-sided n vs R", test_power_ttest_onesided_vs_r)
check("t-test one-sample n vs R", test_power_ttest_onesample_vs_r)
check("t-test paired n vs R", test_power_ttest_paired_vs_r)


# ===================================================================
# 10. Power — ANOVA vs R
# ===================================================================
section("Power: ANOVA vs R")


def test_power_anova_n_vs_r():
    from pystatsbio.power import power_anova_oneway
    result = power_anova_oneway(f=0.25, k=3, power=0.80, alpha=0.05)
    r = R_RESULTS["power_anova"]
    assert result.n == r["n_ceil"], (
        f"n: py={result.n}, R={r['n_ceil']}"
    )
    print(f"    n: py={result.n}, R={r['n_ceil']} (exact R={r['n_exact']:.4f})")


def test_power_anova_power_vs_r():
    from pystatsbio.power import power_anova_oneway
    result = power_anova_oneway(f=0.25, k=3, n=53, alpha=0.05)
    r = R_RESULTS["power_anova_power"]
    assert_close(result.power, r["power"], "ANOVA power")
    print(f"    power: py={result.power:.10f}, R={r['power']}")


check("ANOVA n vs R (pwr)", test_power_anova_n_vs_r)
check("ANOVA power vs R", test_power_anova_power_vs_r)


# ===================================================================
# 11. Power — proportions vs R
# ===================================================================
section("Power: proportions vs R")


def test_power_prop_n_vs_r():
    from pystatsbio.power import power_prop_test
    result = power_prop_test(h=0.5, power=0.80, alpha=0.05)
    r = R_RESULTS["power_prop"]
    assert result.n == r["n_ceil"], (
        f"n: py={result.n}, R={r['n_ceil']}"
    )
    print(f"    n: py={result.n}, R={r['n_ceil']} (exact R={r['n_exact']:.4f})")


check("Proportions n vs R (pwr)", test_power_prop_n_vs_r)


# ===================================================================
# 12. Power — crossover BE vs R (PowerTOST)
# ===================================================================
section("Power: crossover BE vs R")


def test_power_be_n_vs_r():
    from pystatsbio.power import power_crossover_be
    # Both pystatsbio and PowerTOST now use alpha as per-test level
    result = power_crossover_be(cv=0.25, theta0=1.0, power=0.80, alpha=0.05)
    r = R_RESULTS["power_be"]
    assert result.n == r["n"], (
        f"n: py={result.n}, R={r['n']}"
    )
    print(f"    n: py={result.n}, R={r['n']}")


def test_power_be_power_vs_r():
    from pystatsbio.power import power_crossover_be
    result = power_crossover_be(cv=0.25, theta0=1.0, n=24, alpha=0.05)
    r = R_RESULTS["power_be"]
    assert_close(result.power, r["power"], "BE power at n=24")
    print(f"    power: py={result.power:.10f}, R={r['power']}")


check("Crossover BE n vs R (PowerTOST)", test_power_be_n_vs_r)
check("Crossover BE power vs R", test_power_be_power_vs_r)


# ===================================================================
# Summary
# ===================================================================
print(f"\n{'='*60}")
total = passed + failed
print(f"  CROSS-VALIDATION RESULTS: {passed}/{total} passed, {failed} failed")
if failed == 0:
    print(f"  All pystatsbio results match R within tolerance")
print(f"{'='*60}\n")

sys.exit(1 if failed else 0)
