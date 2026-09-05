from mda_numeric_checker import (
    analyze_text,
    calculate_change_pct,
    classify_percentage_delta,
    detect_direction_conflict,
)


def test_growth_rate_calculation_uses_absolute_previous_value():
    assert calculate_change_pct(120, 100) == 20.0
    assert calculate_change_pct(-80, -100) == 20.0


def test_previous_value_zero_is_not_calculable():
    result = calculate_change_pct(100, 0)
    assert result["status"] == "not_calculable_previous_zero"


def test_discrepancy_classification_follows_percentage_point_thresholds():
    assert classify_percentage_delta(stated_pct=20, calculated_pct=20.6) == "consistent"
    assert classify_percentage_delta(stated_pct=20, calculated_pct=22.0) == "minor_discrepancy"
    assert classify_percentage_delta(stated_pct=20, calculated_pct=25.0) == "material_discrepancy"


def test_direction_conflict_detects_increase_language_with_negative_calculation():
    assert detect_direction_conflict("revenue increased", calculated_pct=-5.0) is True
    assert detect_direction_conflict("profit decreased", calculated_pct=-5.0) is False


def test_analyze_text_flags_consistent_and_material_discrepancy_examples():
    consistent = analyze_text(
        "Revenue increased by 20% from RM100 million to RM120 million. "
        "Profit decreased by 10% from RM50 million to RM45 million.",
        document_id="MDA_TEST_2024",
    )
    inconsistent = analyze_text(
        "Revenue increased by 50% from RM100 million to RM120 million.",
        document_id="MDA_BAD_2024",
    )

    assert consistent["num_calculable_changes"] >= 2
    assert consistent["num_material_discrepancies"] == 0
    assert inconsistent["num_material_discrepancies"] >= 1


def test_analyze_text_does_not_cross_sentence_boundaries_between_metrics():
    result = analyze_text(
        "Revenue increased by 20% from RM100 million to RM120 million. "
        "Net profit increased by 10% from RM10 million to RM11 million.",
        document_id="MDA_BOUNDARY_2024",
    )

    assert result["num_calculable_changes"] == 2
    assert result["num_material_discrepancies"] == 0


def test_analyze_text_handles_percent_before_metric_as_compared_with_pattern():
    result = analyze_text(
        "The Group recorded 18% increase in revenue of RM515.615 million in FY2021 "
        "as compared with RM436.166 million in FY2020.",
        document_id="MDA_3A_2021",
    )

    assert result["num_calculable_changes"] == 1
    assert result["num_consistent_checks"] == 1
    assert result["num_material_discrepancies"] == 0


def test_analyze_text_checks_margin_against_profit_and_revenue():
    consistent = analyze_text(
        "[MDA_P001] Gross profit margin was 20% with gross profit of RM20 million and revenue of RM100 million.",
        document_id="MDA_MARGIN_OK",
    )
    inconsistent = analyze_text(
        "[MDA_P001] Gross profit margin was 30% with gross profit of RM20 million and revenue of RM100 million.",
        document_id="MDA_MARGIN_BAD",
    )

    assert consistent["num_calculable_changes"] == 1
    assert consistent["num_consistent_checks"] == 1
    assert consistent["checks"][0]["evidence_locator"] == "[MDA_P001]"
    assert inconsistent["num_material_discrepancies"] == 1


def test_analyze_text_checks_segment_sum_against_total():
    consistent = analyze_text(
        "[MDA_P002] Segment revenue from food was RM60 million and beverages was RM40 million, for total segment revenue of RM100 million.",
        document_id="MDA_SEGMENT_OK",
    )
    inconsistent = analyze_text(
        "[MDA_P002] Segment revenue from food was RM60 million and beverages was RM40 million, for total segment revenue of RM120 million.",
        document_id="MDA_SEGMENT_BAD",
    )

    assert consistent["num_calculable_changes"] == 1
    assert consistent["num_consistent_checks"] == 1
    assert consistent["checks"][0]["evidence_locator"] == "[MDA_P002]"
    assert inconsistent["num_material_discrepancies"] == 1
