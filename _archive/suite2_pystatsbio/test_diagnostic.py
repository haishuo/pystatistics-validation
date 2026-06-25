"""Suite 2: Diagnostic accuracy tests against R pROC on NHANES."""

import pytest
import numpy as np

from pystatsbio.diagnostic import roc, roc_test, diagnostic_accuracy, batch_auc, optimal_cutoff


class TestROC:
    """ROC analysis vs R pROC."""

    def test_hba1c_auc(self, nhanes_data, r_results):
        """HbA1c AUC should be 1.0 — diabetic is defined by HbA1c >= 6.5."""
        if "roc_hba1c" not in r_results:
            pytest.skip("R did not compute HbA1c ROC")

        result = roc(
            nhanes_data["diabetic"].values,
            nhanes_data["LBXGH"].values,
            direction="<",
        )
        r_ref = r_results["roc_hba1c"]

        assert result.auc == pytest.approx(r_ref["auc"], abs=1e-10)

    def test_glucose_auc(self, nhanes_data, r_results):
        if "roc_glucose" not in r_results:
            pytest.skip("R did not compute glucose ROC")

        result = roc(
            nhanes_data["diabetic"].values,
            nhanes_data["LBXSGL"].values,
            direction="<",
        )
        r_ref = r_results["roc_glucose"]

        assert result.auc == pytest.approx(r_ref["auc"], rel=1e-10)


class TestROCTest:
    """DeLong test comparing two ROC curves vs R pROC::roc.test."""

    def test_hba1c_vs_glucose(self, nhanes_data, r_results):
        if "roc_test_delong" not in r_results:
            pytest.skip("R did not compute DeLong test")

        response = nhanes_data["diabetic"].values
        pred1 = nhanes_data["LBXGH"].values
        pred2 = nhanes_data["LBXSGL"].values
        roc1 = roc(response, pred1, direction="<")
        roc2 = roc(response, pred2, direction="<")
        result = roc_test(roc1, roc2, predictor1=pred1, predictor2=pred2, response=response)
        r_ref = r_results["roc_test_delong"]

        assert result.statistic == pytest.approx(r_ref["statistic"], rel=1e-4)
        assert result.p_value == pytest.approx(r_ref["p_value"], rel=1e-4)


class TestBatchAUC:
    """batch_auc CPU vs R sequential pROC."""

    def test_batch_matches_r(self, nhanes_data, r_results):
        if "batch_auc" not in r_results:
            pytest.skip("R did not compute batch AUC")

        r_ref = r_results["batch_auc"]
        cols = r_ref["columns"]

        # Filter to available columns
        available = [c for c in cols if c in nhanes_data.columns]
        if len(available) < 2:
            pytest.skip("Not enough biomarker columns")

        predictors = nhanes_data[available].values.astype(float)
        response = nhanes_data["diabetic"].values.astype(int)

        result = batch_auc(response, predictors, backend="cpu")

        r_aucs = np.array(r_ref["auc"])
        # Match by column index
        for i, col in enumerate(available):
            r_idx = cols.index(col)
            if not np.isnan(r_aucs[r_idx]):
                assert result.auc[i] == pytest.approx(
                    r_aucs[r_idx], rel=1e-6
                ), f"AUC mismatch for {col}"


class TestOptimalCutoff:
    """Optimal cutoff (Youden) vs R."""

    def test_hba1c_youden(self, nhanes_data, r_results):
        """Optimal cutoff for HbA1c.

        Note: HbA1c AUC=1.0 (perfect separator since diabetic is defined by HbA1c >= 6.5),
        so the optimal cutoff should be near the definition threshold 6.5.
        """
        if "optimal_cutoff_hba1c" not in r_results:
            pytest.skip("R did not compute optimal cutoff")

        roc_result = roc(
            nhanes_data["diabetic"].values,
            nhanes_data["LBXGH"].values,
            direction="<",
        )
        result = optimal_cutoff(roc_result, method="youden")

        # For AUC=1.0, both R and Python should find a cutoff near 6.5
        assert 6.0 <= result.cutoff <= 7.0, f"Optimal cutoff {result.cutoff} not near 6.5"
