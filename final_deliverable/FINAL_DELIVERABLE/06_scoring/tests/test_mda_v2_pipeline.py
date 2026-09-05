import csv
from pathlib import Path

from audit_mda_initial_scores import audit_dimension_row, locator_exists
from clean_mda_texts_v2 import clean_one_mda_text, paragraph_ids
from compare_mda_v1_v2_scores import compare_score_rows, full_rescore_decision
from mda_numeric_checker import analyze_text


def test_cleaning_does_not_modify_financial_numbers_and_removes_page_number():
    raw = "MANAGEMENT DISCUSSION\nRevenue increased from RM460 million to RM500 million.\n12\nProfit margin was 8.7%."

    cleaned, mapping = clean_one_mda_text("MDA_TEST", raw)

    assert "RM460 million" in cleaned
    assert "RM500 million" in cleaned
    assert "8.7%" in cleaned
    assert "\n12\n" not in cleaned
    assert any(row["is_removed"] == "1" and row["removal_reason"] == "isolated_page_number" for row in mapping)


def test_cleaning_paragraph_ids_are_continuous_and_mapping_links_raw_clean():
    raw = "Financial Review\nRevenue was RM500 million.\n\nBusiness Review\nMarket demand improved."

    cleaned, mapping = clean_one_mda_text("MDA_TEST", raw)

    assert paragraph_ids(cleaned) == ["MDA_P001", "MDA_P002"]
    kept = [row for row in mapping if row["is_removed"] == "0"]
    assert kept[0]["raw_text_snippet"]
    assert kept[0]["clean_text"]


def test_evidence_locator_found_in_cleaned_text():
    text = "[SECTION: Financial Review]\n\n[MDA_P001]\nRevenue was RM500 million."

    assert locator_exists("[MDA_P001]", text) is True
    assert locator_exists("[MDA_P999]", text) is False


def test_ar02_external_financial_statement_claim_is_flagged():
    row = {
        "document_id": "MDA_TEST",
        "dimension_code": "AR02",
        "raw_score": "4",
        "evidence_locator": "[MDA_P001]",
        "evidence_text_short": "The numbers match audited financial statements.",
        "reason": "Matches audited financial statements.",
        "confidence_level": "high",
        "weighted_score": "75",
    }
    result = audit_dimension_row(row, "[MDA_P001]\nRevenue was RM500 million.", {"num_material_discrepancies": 0, "num_severe_direction_conflicts": 0})

    assert result["ar02_external_verification_claim_flag"] == 1
    assert result["needs_review"] == 1


def test_numeric_checker_growth_rate_and_material_discrepancy():
    consistent = analyze_text("Revenue increased by 8.7% from RM460 million to RM500 million.", "MDA_TEST")
    inconsistent = analyze_text("Revenue increased by 15% from RM460 million to RM500 million.", "MDA_TEST")

    assert consistent["num_consistent_checks"] == 1
    assert inconsistent["num_material_discrepancies"] == 1


def test_numeric_checker_direction_conflict():
    result = analyze_text("Revenue decreased by 8.7% from RM460 million to RM500 million.", "MDA_TEST")

    assert result["num_severe_direction_conflicts"] == 1


def test_v1_v2_comparison_flags_material_change():
    row = compare_score_rows(
        {"document_id": "MDA_TEST", "AR01": "3", "AR02": "2", "AR03": "3", "AR04": "3", "AR05": "3", "total_score_100": "50"},
        {"document_id": "MDA_TEST", "AR01": "3", "AR02": "4", "AR03": "3", "AR04": "3", "AR05": "3", "total_score_100": "65"},
    )

    assert row["AR02_diff"] == 2
    assert row["material_change_flag"] == 1


def test_full_rescore_recommended_when_material_change_ratio_high():
    rows = [
        {"material_change_flag": 1, "AR02_diff": 0, "evidence_locator_invalid_flag": 0},
        {"material_change_flag": 0, "AR02_diff": 0, "evidence_locator_invalid_flag": 0},
        {"material_change_flag": 0, "AR02_diff": 0, "evidence_locator_invalid_flag": 0},
        {"material_change_flag": 0, "AR02_diff": 0, "evidence_locator_invalid_flag": 0},
        {"material_change_flag": 0, "AR02_diff": 0, "evidence_locator_invalid_flag": 0},
    ]

    decision = full_rescore_decision(rows)

    assert decision["full_rescore_recommended"] == 1
    assert "material change ratio" in decision["reason"]
