"""
Integration test for pystatsbio using real biomedical datasets.

Datasets:
  - Theoph      — Theophylline PK (12 subjects, built-in R dataset)
  - ryegrass    — Ferulic acid dose-response (drc package)
  - aSAH        — Aneurysmal subarachnoid hemorrhage diagnostics (pROC package)

Tests every public function in pystatsbio against real data and verifies
that outputs are numerically sensible (positive where expected, CIs ordered,
etc.).  Does NOT compare against R — see test_pystatsbio_vs_r.py for that.

Prerequisites:
  pip install pystatsbio pandas

Usage:
  python tests/test_pystatsbio_analysis.py
"""

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

passed = 0
failed = 0

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pystatsbio_r"


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


# ===================================================================
# 0. Load datasets
# ===================================================================
section("Data loading")

theoph = pd.read_csv(DATA_DIR / "theoph.csv")
ryegrass = pd.read_csv(DATA_DIR / "ryegrass.csv")
asah = pd.read_csv(DATA_DIR / "asah.csv")


def test_theoph_loaded():
    assert theoph.shape[0] == 132, f"expected 132 rows, got {theoph.shape[0]}"
    assert set(theoph.columns) >= {"Subject", "Wt", "Dose", "Time", "conc"}
    n_subj = theoph["Subject"].nunique()
    assert n_subj == 12, f"expected 12 subjects, got {n_subj}"
    print(f"    Theoph: {theoph.shape[0]} rows, {n_subj} subjects")


def test_ryegrass_loaded():
    assert ryegrass.shape[0] == 24, f"expected 24 rows, got {ryegrass.shape[0]}"
    assert "rootl" in ryegrass.columns and "conc" in ryegrass.columns
    print(f"    ryegrass: {ryegrass.shape[0]} rows, doses: {sorted(ryegrass['conc'].unique())}")


def test_asah_loaded():
    assert asah.shape[0] == 113, f"expected 113 rows, got {asah.shape[0]}"
    assert "outcome" in asah.columns and "s100b" in asah.columns
    n_poor = (asah["outcome"] == "Poor").sum()
    n_good = (asah["outcome"] == "Good").sum()
    print(f"    aSAH: {asah.shape[0]} rows, {n_poor} Poor / {n_good} Good")


check("Theoph dataset loaded", test_theoph_loaded)
check("ryegrass dataset loaded", test_ryegrass_loaded)
check("aSAH dataset loaded", test_asah_loaded)


# ===================================================================
# 1. PK — NCA on Theoph Subject 1
# ===================================================================
section("PK: NCA on Theoph Subject 1")

s1 = theoph[theoph["Subject"] == 1].copy()
time_s1 = s1["Time"].values.astype(float)
conc_s1 = s1["conc"].values.astype(float)
dose_s1 = float(s1["Dose"].iloc[0])


def test_nca_s1_basic():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    assert result.cmax > 0
    assert result.tmax >= 0
    assert result.auc_last > 0
    assert result.route == "ev"
    print(f"    Cmax={result.cmax:.2f} at Tmax={result.tmax:.2f}h")
    print(f"    AUC(0-last)={result.auc_last:.2f}")


def test_nca_s1_terminal():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    assert result.half_life is not None and result.half_life > 0
    assert result.lambda_z is not None and result.lambda_z > 0
    assert result.auc_inf is not None
    assert result.auc_inf > result.auc_last
    expected_t12 = math.log(2) / result.lambda_z
    assert abs(result.half_life - expected_t12) < 1e-10
    print(f"    t1/2={result.half_life:.2f}h, lambda_z={result.lambda_z:.6f}")
    print(f"    AUC(0-inf)={result.auc_inf:.2f}, extrap={result.auc_pct_extrap:.1f}%")


def test_nca_s1_pk_params():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    assert result.clearance is not None and result.clearance > 0
    assert result.vz is not None and result.vz > 0
    print(f"    CL/F={result.clearance:.4f}, Vz/F={result.vz:.4f}")


def test_nca_s1_summary():
    from pystatsbio.pk import nca
    result = nca(time_s1, conc_s1, dose=dose_s1, route="ev")
    s = result.summary()
    assert isinstance(s, str)
    assert "Cmax" in s
    assert "AUC" in s
    assert "t1/2" in s
    assert "CL/F" in s


check("NCA Subject 1: Cmax, Tmax, AUC(0-last)", test_nca_s1_basic)
check("NCA Subject 1: terminal phase", test_nca_s1_terminal)
check("NCA Subject 1: CL/F & Vz/F", test_nca_s1_pk_params)
check("NCA Subject 1: summary()", test_nca_s1_summary)


# ===================================================================
# 2. PK — NCA on all 12 subjects
# ===================================================================
section("PK: NCA on all 12 Theoph subjects")


def test_nca_all_subjects():
    from pystatsbio.pk import nca
    results_list = []
    for subj in sorted(theoph["Subject"].unique()):
        si = theoph[theoph["Subject"] == subj]
        r = nca(
            si["Time"].values.astype(float),
            si["conc"].values.astype(float),
            dose=float(si["Dose"].iloc[0]),
            route="ev",
        )
        results_list.append(r)
        assert r.cmax > 0
        assert r.auc_last > 0

    cmax_vals = [r.cmax for r in results_list]
    auc_vals = [r.auc_last for r in results_list]
    print(f"    12 subjects analyzed")
    print(f"    Cmax range: [{min(cmax_vals):.2f}, {max(cmax_vals):.2f}]")
    print(f"    AUC  range: [{min(auc_vals):.2f}, {max(auc_vals):.2f}]")


check("NCA all 12 subjects", test_nca_all_subjects)


# ===================================================================
# 3. Dose-response — LL.4 fit
# ===================================================================
section("Dose-response: LL.4 fit (ryegrass)")

dose_rg = ryegrass["conc"].values.astype(float)
resp_rg = ryegrass["rootl"].values.astype(float)


def test_drm_fit():
    from pystatsbio.doseresponse import fit_drm
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    assert result.converged
    assert result.model == "LL.4"
    p = result.params
    # Sanity: ec50 > 0, hill != 0
    # Note: ryegrass is a decreasing response, so hill is negative and
    # bottom > top (pystatsbio convention: bottom = dose→0, top = dose→∞)
    assert p.ec50 > 0, f"ec50={p.ec50}"
    assert p.hill != 0, f"hill={p.hill}"
    print(f"    hill={p.hill:.4f}, bottom={p.bottom:.4f}")
    print(f"    top={p.top:.4f}, ec50={p.ec50:.4f}")
    print(f"    RSS={result.rss:.4f}, AIC={result.aic:.2f}")


def test_drm_se():
    from pystatsbio.doseresponse import fit_drm
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    assert len(result.se) == 4
    assert np.all(result.se > 0), "SEs should be positive"
    print(f"    SEs: {result.se}")


def test_drm_predict():
    from pystatsbio.doseresponse import fit_drm
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    pred = result.predict()
    assert len(pred) == len(dose_rg)
    # Predictions at dose=0 should be near the high-response asymptote
    zero_idx = dose_rg == 0
    assert np.all(pred[zero_idx] > 5), "predictions at dose=0 should be high (untreated control)"


def test_drm_summary():
    from pystatsbio.doseresponse import fit_drm
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    s = result.summary()
    assert "LL.4" in s
    assert "RSS" in s
    assert "ec50" in s


check("LL.4 fit converges", test_drm_fit)
check("LL.4 standard errors positive", test_drm_se)
check("LL.4 predictions sensible", test_drm_predict)
check("LL.4 summary()", test_drm_summary)


# ===================================================================
# 4. Dose-response — EC50
# ===================================================================
section("Dose-response: EC50 (ryegrass)")


def test_ec50():
    from pystatsbio.doseresponse import fit_drm, ec50
    result = fit_drm(dose_rg, resp_rg, model="LL.4")
    e = ec50(result)
    assert e.estimate > 0
    assert e.ci_lower < e.estimate < e.ci_upper
    assert e.se > 0
    assert e.conf_level == 0.95
    print(f"    EC50={e.estimate:.4f} ({e.ci_lower:.4f}, {e.ci_upper:.4f})")
    print(f"    SE={e.se:.4f}")


check("EC50 with delta-method CI", test_ec50)


# ===================================================================
# 5. Diagnostic — ROC on s100b
# ===================================================================
section("Diagnostic: ROC on aSAH s100b")

response_asah = (asah["outcome"] == "Poor").astype(int).values
s100b = asah["s100b"].values.astype(float)
ndka = asah["ndka"].values.astype(float)


def test_roc_s100b():
    from pystatsbio.diagnostic import roc
    result = roc(response_asah, s100b, direction="<")
    assert 0.5 < result.auc < 1.0
    assert result.n_positive == 41
    assert result.n_negative == 72
    assert result.auc_ci_lower < result.auc < result.auc_ci_upper
    print(f"    AUC={result.auc:.4f} [{result.auc_ci_lower:.4f}, {result.auc_ci_upper:.4f}]")
    print(f"    n+={result.n_positive}, n-={result.n_negative}")


def test_roc_curve_shape():
    from pystatsbio.diagnostic import roc
    result = roc(response_asah, s100b, direction="<")
    assert abs(result.tpr[0]) < 1e-9
    assert abs(result.fpr[0]) < 1e-9
    assert abs(result.tpr[-1] - 1.0) < 1e-9
    assert abs(result.fpr[-1] - 1.0) < 1e-9
    assert len(result.tpr) == len(result.fpr)


def test_roc_summary():
    from pystatsbio.diagnostic import roc
    result = roc(response_asah, s100b, direction="<")
    s = result.summary()
    assert "AUC" in s
    assert "CI" in s


check("ROC (s100b): AUC & CI", test_roc_s100b)
check("ROC (s100b): curve endpoints", test_roc_curve_shape)
check("ROC (s100b): summary()", test_roc_summary)


# ===================================================================
# 6. Diagnostic — optimal cutoff
# ===================================================================
section("Diagnostic: optimal cutoff (Youden)")


def test_optimal_cutoff():
    from pystatsbio.diagnostic import roc, optimal_cutoff
    roc_result = roc(response_asah, s100b, direction="<")
    cut = optimal_cutoff(roc_result, method="youden")
    assert cut.cutoff > 0
    assert 0 < cut.sensitivity <= 1
    assert 0 < cut.specificity <= 1
    assert cut.method == "youden"
    print(f"    Youden cutoff={cut.cutoff:.4f}")
    print(f"    Sens={cut.sensitivity:.4f}, Spec={cut.specificity:.4f}")


check("Optimal cutoff (Youden)", test_optimal_cutoff)


# ===================================================================
# 7. Diagnostic — diagnostic accuracy at cutoff
# ===================================================================
section("Diagnostic: accuracy at cutoff=0.205")


def test_diag_accuracy():
    from pystatsbio.diagnostic import diagnostic_accuracy
    result = diagnostic_accuracy(response_asah, s100b, cutoff=0.205, direction="<")
    assert 0 <= result.sensitivity <= 1
    assert 0 <= result.specificity <= 1
    assert 0 <= result.ppv <= 1
    assert 0 <= result.npv <= 1
    assert result.lr_positive > 0
    assert result.dor > 0
    print(f"    Sens={result.sensitivity:.4f}, Spec={result.specificity:.4f}")
    print(f"    PPV={result.ppv:.4f}, NPV={result.npv:.4f}")
    print(f"    LR+={result.lr_positive:.4f}, DOR={result.dor:.4f}")


def test_diag_accuracy_summary():
    from pystatsbio.diagnostic import diagnostic_accuracy
    result = diagnostic_accuracy(response_asah, s100b, cutoff=0.205, direction="<")
    s = result.summary()
    assert "Sensitivity" in s
    assert "Specificity" in s
    assert "PPV" in s


check("Diagnostic accuracy at cutoff=0.205", test_diag_accuracy)
check("DiagnosticResult summary()", test_diag_accuracy_summary)


# ===================================================================
# 8. Diagnostic — DeLong test (s100b vs ndka)
# ===================================================================
section("Diagnostic: DeLong test (s100b vs ndka)")


def test_delong_test():
    from pystatsbio.diagnostic import roc, roc_test
    roc1 = roc(response_asah, s100b, direction="<")
    roc2 = roc(response_asah, ndka, direction="<")
    result = roc_test(
        roc1, roc2,
        predictor1=s100b, predictor2=ndka,
        response=response_asah,
    )
    assert hasattr(result, "statistic")
    assert hasattr(result, "p_value")
    assert 0 <= result.p_value <= 1
    print(f"    Z={result.statistic:.4f}, p={result.p_value:.4f}")
    print(f"    AUC1={result.auc1:.4f}, AUC2={result.auc2:.4f}")


check("DeLong test (s100b vs ndka)", test_delong_test)


# ===================================================================
# 9. Diagnostic — batch AUC
# ===================================================================
section("Diagnostic: batch AUC")


def test_batch_auc():
    from pystatsbio.diagnostic import batch_auc
    predictors = np.column_stack([s100b, ndka])
    result = batch_auc(response_asah, predictors, backend="cpu")
    assert result.n_markers == 2
    assert np.all(result.auc > 0.5)
    print(f"    s100b AUC={result.auc[0]:.4f}, ndka AUC={result.auc[1]:.4f}")


check("Batch AUC (s100b + ndka)", test_batch_auc)


# ===================================================================
# 10. Power — t-test
# ===================================================================
section("Power: t-test")


def test_power_ttest_n():
    from pystatsbio.power import power_t_test
    r = power_t_test(d=0.5, power=0.80, alpha=0.05)
    assert r.n is not None and r.n > 10
    print(f"    n={r.n} per group (d=0.5, two-sided)")


def test_power_ttest_power():
    from pystatsbio.power import power_t_test
    r = power_t_test(n=64, d=0.5, alpha=0.05)
    assert r.power is not None
    assert 0.75 < r.power < 0.85
    print(f"    power={r.power:.6f} (n=64, d=0.5)")


def test_power_ttest_effect():
    from pystatsbio.power import power_t_test
    r = power_t_test(n=100, power=0.80, alpha=0.05)
    assert r.effect_size is not None
    assert 0.3 < r.effect_size < 0.5
    print(f"    d={r.effect_size:.6f} (n=100, power=0.80)")


check("t-test: solve for n", test_power_ttest_n)
check("t-test: solve for power", test_power_ttest_power)
check("t-test: solve for effect size", test_power_ttest_effect)


# ===================================================================
# 11. Power — ANOVA
# ===================================================================
section("Power: ANOVA")


def test_power_anova():
    from pystatsbio.power import power_anova_oneway
    r = power_anova_oneway(f=0.25, k=3, power=0.80, alpha=0.05)
    assert r.n is not None and r.n > 10
    print(f"    n={r.n} per group (f=0.25, k=3)")


check("ANOVA: solve for n", test_power_anova)


# ===================================================================
# 12. Power — proportions
# ===================================================================
section("Power: proportions")


def test_power_prop():
    from pystatsbio.power import power_prop_test
    r = power_prop_test(h=0.5, power=0.80, alpha=0.05)
    assert r.n is not None and r.n > 10
    print(f"    n={r.n} per group (h=0.5)")


check("Proportions: solve for n", test_power_prop)


# ===================================================================
# 13. Power — crossover bioequivalence
# ===================================================================
section("Power: crossover bioequivalence")


def test_power_be():
    from pystatsbio.power import power_crossover_be
    r = power_crossover_be(cv=0.25, power=0.80)
    assert r.n is not None and r.n > 10
    print(f"    n={r.n} total (CV=25%)")


check("Crossover BE: solve for n", test_power_be)


# ===================================================================
# Summary
# ===================================================================
print(f"\n{'='*60}")
total = passed + failed
print(f"  PYSTATSBIO ANALYSIS RESULTS: {passed}/{total} passed, {failed} failed")
print(f"{'='*60}\n")

sys.exit(1 if failed else 0)
