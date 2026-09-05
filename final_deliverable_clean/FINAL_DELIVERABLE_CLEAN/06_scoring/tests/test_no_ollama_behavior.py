import csv
from pathlib import Path

from run_scoring import handle_model_unavailable


def test_no_ollama_creates_manual_template_without_fake_scores(tmp_path):
    root = tmp_path / "06_scoring"
    rows = [
        {
            "document_id": "MDA_1234_TEST_2024",
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "1234",
            "ticker": "TEST",
            "company_name": "Test Food Berhad",
            "report_year": "2024",
        }
    ]

    handle_model_unavailable("mda", rows, root, "Local Ollama qwen3:8b is unavailable")

    template = root / "07_review" / "mda_manual_scoring_template.csv"
    ratings = root / "06_ratings" / "mda_ratings_long.csv"
    with template.open(newline="", encoding="utf-8") as fh:
        template_rows = list(csv.DictReader(fh))
    with ratings.open(newline="", encoding="utf-8") as fh:
        rating_rows = list(csv.DictReader(fh))

    assert template_rows[0]["document_id"] == "MDA_1234_TEST_2024"
    assert template_rows[0]["manual_required_reason"] == "Local Ollama qwen3:8b is unavailable"
    assert rating_rows == []
