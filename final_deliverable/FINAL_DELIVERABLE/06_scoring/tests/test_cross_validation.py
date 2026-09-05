import csv
from pathlib import Path

from scoring_utils import MDA_DOCUMENT_HEADERS, NEWS_DOCUMENT_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows
from run_cross_validation import run_cross_validation


def _write_fixture(root: Path) -> None:
    mda_dir = root / "02_extracted_text" / "mda"
    news_dir = root / "02_extracted_text" / "news"
    mda_dir.mkdir(parents=True)
    news_dir.mkdir(parents=True)
    mda_text = (
        "[MDA_P001]\nRevenue increased in 2024 as domestic demand improved.\n\n"
        "[MDA_P002]\nRaw material costs increased and pressured margins.\n\n"
        "[MDA_P003]\nThe company maintains a cautious outlook for 2025.\n\n"
        "[MDA_P004]\nThe group expanded capacity through a new production line."
    )
    (mda_dir / "MDA_TEST_2024.txt").write_text(mda_text, encoding="utf-8")
    news_items = {
        "NEWS_RAW_2024": "Industry data showed F&B raw material costs increased in 2024, adding pressure to producers.",
        "NEWS_GROWTH_2025": "Analysts said Test Food expects explosive growth in 2025 after recent expansion.",
        "NEWS_SAFETY_2024": "Regulators reported a food safety issue affecting Test Food products in 2024.",
        "NEWS_AR_2024": "According to the annual report, Test Food said raw material costs increased during the year.",
        "NEWS_REV_2024": "Market report said Test Food revenue decreased in 2024 despite management saying revenue increased.",
    }
    for document_id, text in news_items.items():
        (news_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
    mda_rows = [
        {
            "document_id": "MDA_TEST_2024",
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "9999",
            "ticker": "TEST",
            "company_name": "Test Food Berhad",
            "report_year": "2024",
            "extraction_status": "success",
            "text_path": "02_extracted_text/mda/MDA_TEST_2024.txt",
            "source_path": "",
            "source_file_sha256": "",
            "notes": "",
        }
    ]
    news_rows = []
    for idx, document_id in enumerate(news_items, start=1):
        news_rows.append(
            {
                "document_id": document_id,
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Test Food Berhad",
                "ticker": "TEST",
                "source_name": "Synthetic News",
                "publish_date": f"2024-0{idx}-15",
                "title": document_id,
                "url": "",
                "text_path": f"02_extracted_text/news/{document_id}.txt",
                "notes": "",
            }
        )
    write_csv_rows(root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], mda_rows)
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], news_rows)
    write_csv_rows(
        root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            {
                "document_id": "MDA_TEST_2024",
                "doc_type": "mda",
                "ticker": "TEST",
                "company_name": "Test Food Berhad",
                "report_year": "2024",
                "total_score_100": "75",
            }
        ],
    )
    write_csv_rows(
        root / "06_ratings" / "news_document_scores.csv",
        NEWS_DOCUMENT_HEADERS,
        [
            {
                "document_id": row["document_id"],
                "doc_type": "news",
                "ticker": "TEST",
                "company_name": "Test Food Berhad",
                "publish_date": row["publish_date"],
                "total_score_100": "65",
            }
            for row in news_rows
        ],
    )


def test_cross_validation_classifies_bidirectional_relation_cases(tmp_path):
    root = tmp_path / "06_scoring"
    _write_fixture(root)

    summary = run_cross_validation("full", root=root, mock_mode=True)

    assert summary["mda_claim_count"] >= 4
    assert summary["news_claim_count"] >= 5
    links = read_csv_rows(root / "09_cross_validation" / "claim_links.csv")
    relation_by_news = {row["news_document_id"]: row for row in links}
    assert relation_by_news["NEWS_RAW_2024"]["relation_type"] in {"contextual_support", "strong_support"}
    assert relation_by_news["NEWS_RAW_2024"]["direction"] == "news_supports_mda"
    assert relation_by_news["NEWS_GROWTH_2025"]["relation_type"] in {"qualification", "possible_overstatement"}
    assert relation_by_news["NEWS_GROWTH_2025"]["news_overstatement_flag"] == "1"
    assert relation_by_news["NEWS_SAFETY_2024"]["relation_type"] == "possible_omission"
    assert relation_by_news["NEWS_SAFETY_2024"]["mda_omission_flag"] == "1"
    assert relation_by_news["NEWS_SAFETY_2024"]["review_required"] == "true"
    assert relation_by_news["NEWS_AR_2024"]["relation_type"] == "same_source_repetition"
    assert relation_by_news["NEWS_AR_2024"]["source_independence"] == "low"
    assert relation_by_news["NEWS_AR_2024"]["support_level"] != "strong"
    assert relation_by_news["NEWS_REV_2024"]["relation_type"] == "contradiction"
    assert relation_by_news["NEWS_REV_2024"]["contradiction_flag"] == "1"


def test_cross_validation_writes_all_required_outputs_and_preserves_scores(tmp_path):
    root = tmp_path / "06_scoring"
    _write_fixture(root)
    mda_scores_before = (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8")
    news_scores_before = (root / "06_ratings" / "news_document_scores.csv").read_text(encoding="utf-8")

    run_cross_validation("pilot", root=root, mock_mode=True)

    required = [
        "mda_claims.csv",
        "news_claims.csv",
        "claim_links.csv",
        "mda_news_cross_validation_pairs.csv",
        "company_year_cross_validation_summary.csv",
        "cross_validation_report.md",
    ]
    for name in required:
        assert (root / "09_cross_validation" / name).exists()
    assert (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8") == mda_scores_before
    assert (root / "06_ratings" / "news_document_scores.csv").read_text(encoding="utf-8") == news_scores_before
    pair_rows = read_csv_rows(root / "09_cross_validation" / "mda_news_cross_validation_pairs.csv")
    assert pair_rows[0]["cross_validation_score"] in {"1", "2"}
    review_rows = list(csv.DictReader((root / "07_review" / "review_log.csv").open(encoding="utf-8")))
    review_reasons = {row["review_reason"] for row in review_rows}
    assert "cross_validation_contradiction" in review_reasons
    assert "cross_validation_possible_omission" in review_reasons
