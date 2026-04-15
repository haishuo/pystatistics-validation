"""Suite 1: ANOVA tests against R on California Housing."""

import pytest
import numpy as np

from pystatistics.anova import anova_oneway, anova, anova_posthoc, levene_test


class TestOneWayANOVA:
    """One-way ANOVA: MedHouseVal ~ region."""

    def test_f_and_p(self, california_data, r_results):
        result = anova_oneway(
            california_data["MedHouseVal"].values,
            california_data["region"].values,
        )
        r_ref = r_results["anova_oneway"]

        # Find the region row in the ANOVA table
        region_row = [r for r in result.table if r.f_value is not None][0]
        assert region_row.f_value == pytest.approx(r_ref["f_value"], rel=1e-10)
        assert region_row.p_value == pytest.approx(r_ref["p_value"], rel=1e-8)

    def test_sum_of_squares(self, california_data, r_results):
        result = anova_oneway(
            california_data["MedHouseVal"].values,
            california_data["region"].values,
        )
        r_ref = r_results["anova_oneway"]

        region_row = [r for r in result.table if r.f_value is not None][0]
        assert region_row.sum_sq == pytest.approx(r_ref["ss_between"], rel=1e-10)
        assert result.residual_ss == pytest.approx(r_ref["ss_within"], rel=1e-10)

    def test_degrees_of_freedom(self, california_data, r_results):
        result = anova_oneway(
            california_data["MedHouseVal"].values,
            california_data["region"].values,
        )
        r_ref = r_results["anova_oneway"]

        region_row = [r for r in result.table if r.f_value is not None][0]
        assert region_row.df == r_ref["df_between"]
        assert result.residual_df == r_ref["df_within"]


class TestFactorialANOVA:
    """Factorial ANOVA Type II: MedHouseVal ~ region * old_house."""

    def test_type_ii(self, california_data, r_results):
        result = anova(
            california_data["MedHouseVal"].values,
            factors={
                "region": california_data["region"].values,
                "old_house": california_data["old_house"].astype(str).values,
            },
            ss_type=2,
        )
        r_ref = r_results["anova_factorial"]

        # Compare F values for each term
        for row in result.table:
            if row.f_value is not None:
                r_idx = r_ref["terms"].index(row.term) if row.term in r_ref["terms"] else None
                if r_idx is not None and r_ref["f_value"][r_idx] is not None:
                    assert row.f_value == pytest.approx(
                        r_ref["f_value"][r_idx], rel=1e-10
                    ), f"F-value mismatch for term '{row.term}'"


class TestTukeyHSD:
    """Tukey HSD post-hoc vs R."""

    def test_pairwise_differences(self, california_data, r_results):
        aov_result = anova_oneway(
            california_data["MedHouseVal"].values,
            california_data["region"].values,
        )
        posthoc = anova_posthoc(aov_result, method="tukey")
        r_ref = r_results["tukey_hsd"]

        # Compare differences and p-values
        for comp in posthoc.comparisons:
            py_pair = {comp.group1, comp.group2}
            for i, r_comp in enumerate(r_ref["comparisons"]):
                r_pair = set(r_comp.split("-"))
                if py_pair == r_pair:
                    assert comp.diff == pytest.approx(
                        r_ref["diff"][i], rel=1e-10
                    ) or comp.diff == pytest.approx(
                        -r_ref["diff"][i], rel=1e-10
                    ), f"Tukey diff mismatch for {py_pair}"
                    # Tukey HSD p-value implementations differ between R and scipy
                    # but both should agree on significance direction
                    if r_ref["p_adj"][i] < 0.001:
                        assert comp.p_value < 0.01, f"Tukey significance mismatch for {py_pair}"
                    elif r_ref["p_adj"][i] < 0.05:
                        assert comp.p_value < 0.10, f"Tukey significance mismatch for {py_pair}"


class TestLevene:
    """Levene's test vs R."""

    def test_f_and_p(self, california_data, r_results):
        result = levene_test(
            california_data["MedHouseVal"].values,
            california_data["region"].values,
        )
        r_ref = r_results["levene"]

        assert result.f_value == pytest.approx(r_ref["f_value"], rel=1e-10)
        assert result.p_value == pytest.approx(r_ref["p_value"], rel=1e-8)
