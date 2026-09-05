from scoring_utils import document_total, grade_label, standardize_score, weighted_score


def test_standardize_score_maps_1_to_0_and_5_to_100():
    assert standardize_score(1) == 0.0
    assert standardize_score(5) == 100.0
    assert standardize_score(3) == 50.0


def test_weighted_total_is_reproducible_from_raw_scores_and_weights():
    raw_scores = {"AR01": 4, "AR02": 3, "AR03": 5, "AR04": 2, "AR05": 3}
    weights = {"AR01": 0.20, "AR02": 0.25, "AR03": 0.25, "AR04": 0.20, "AR05": 0.10}

    total = document_total(raw_scores, weights)

    assert total == 62.5


def test_grade_boundaries_are_inclusive():
    assert grade_label(85) == "A"
    assert grade_label(70) == "B"
    assert grade_label(55) == "C"
    assert grade_label(54.99) == "D"


def test_weighted_score_multiplies_standardized_score_by_weight():
    assert weighted_score(4, 0.25) == 18.75
