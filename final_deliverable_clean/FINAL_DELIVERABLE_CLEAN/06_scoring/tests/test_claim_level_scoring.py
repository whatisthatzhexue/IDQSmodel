import json
from pathlib import Path

from aggregate_claim_scores import aggregate_claim_scores
from claim_utils import map_claim_to_dimensions
from compare_claim_vs_document_scoring import compare_claim_vs_document_scoring
from extract_mda_claims import extract_mda_claims
from extract_news_claims import extract_news_claims
from run_claim_scoring import run_claim_scoring
from scoring_utils import MDA_DOCUMENT_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows
from self_check_pipeline import claim_round_names


def _write_claim_fixture(root: Path) -> None:
    mda_dir = root / "02_extracted_text" / "mda_clean_v2"
    news_dir = root / "02_extracted_text" / "news"
    mda_dir.mkdir(parents=True, exist_ok=True)
    news_dir.mkdir(parents=True, exist_ok=True)
    (mda_dir / "MDA_CLAIM_2024.txt").write_text(
        "[MDA_P001]\nRevenue increased from 460 to 500 due to export demand and lower domestic sales.\n\n"
        "[MDA_P002]\nRaw material cost pressure increased and management expects a cautious outlook.",
        encoding="utf-8",
    )
    (news_dir / "NEWS_CLAIM_2024.txt").write_text(
        "[TITLE]\nTest Food revenue rises as exports improve\n\n"
        "[NEWS_P001]\nAnalysts said Test Food revenue increased after stronger export demand, while domestic sales declined.",
        encoding="utf-8",
    )
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        REGISTRY_HEADERS["mda"],
        [
            {
                "document_id": "MDA_CLAIM_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "9999",
                "ticker": "CLAIM",
                "company_name": "Claim Food Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda_clean_v2/MDA_CLAIM_2024.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_CLAIM_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Claim Food Berhad",
                "ticker": "CLAIM",
                "source_name": "Synthetic News",
                "publish_date": "2024-06-01",
                "title": "Test Food revenue rises as exports improve",
                "url": "",
                "text_path": "02_extracted_text/news/NEWS_CLAIM_2024.txt",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            {
                "document_id": "MDA_CLAIM_2024",
                "doc_type": "mda",
                "total_score_100": "62.5",
                "AR02": "3",
                "AR02_std": "50",
                "model_name": "legacy_document_level",
            }
        ],
    )


def test_extract_mda_claims_decomposes_sentence_into_minimal_semantic_units(tmp_path):
    root = tmp_path / "06_scoring"
    _write_claim_fixture(root)

    summary = extract_mda_claims(root=root)

    path = root / "10_claims" / "mda_claims" / "MDA_CLAIM_2024.json"
    claims = json.loads(path.read_text(encoding="utf-8"))["claims"]
    claim_types = {claim["claim_type"] for claim in claims}
    claim_texts = [claim["claim_text"] for claim in claims]
    assert summary["document_count"] == 1
    assert summary["claim_count"] >= 5
    assert "financial_claim" in claim_types
    assert "causal_claim" in claim_types
    assert "cost_claim" in claim_types
    assert "outlook_claim" in claim_types
    assert any("460" in text and "500" in text for text in claim_texts)
    assert any("export demand" in text.lower() for text in claim_texts)
    assert any("domestic sales" in text.lower() and claim["direction"] == "decrease" for claim, text in zip(claims, claim_texts))
    assert all(claim["source_type"] == "mda" for claim in claims)
    assert all(claim["evidence_locator"].startswith("[MDA_P") for claim in claims)


def test_extract_news_claims_writes_required_claim_schema(tmp_path):
    root = tmp_path / "06_scoring"
    _write_claim_fixture(root)

    summary = extract_news_claims(root=root)

    claims = json.loads((root / "10_claims" / "news_claims" / "NEWS_CLAIM_2024.json").read_text(encoding="utf-8"))["claims"]
    assert summary["claim_count"] >= 3
    required = {
        "claim_id",
        "document_id",
        "source_type",
        "claim_type",
        "topic",
        "claim_text",
        "value",
        "unit",
        "direction",
        "metric_name",
        "evidence_locator",
        "confidence",
    }
    assert required.issubset(claims[0])
    assert {claim["source_type"] for claim in claims} == {"news"}
    assert any(claim["claim_type"] == "analyst_claim" for claim in claims)


def test_claim_mapping_and_mock_scoring_produce_per_claim_scores(tmp_path):
    root = tmp_path / "06_scoring"
    _write_claim_fixture(root)
    extract_mda_claims(root=root)
    financial_claim = json.loads((root / "10_claims" / "mda_claims" / "MDA_CLAIM_2024.json").read_text(encoding="utf-8"))["claims"][0]

    assert "AR02" in map_claim_to_dimensions(financial_claim)

    summary = run_claim_scoring(root=root, source_type="mda", mock=True)

    score_path = root / "10_claim_scoring" / "mda_claim_scores" / "MDA_CLAIM_2024.json"
    scores = json.loads(score_path.read_text(encoding="utf-8"))["claim_scores"]
    assert summary["claim_count"] == len(scores)
    assert summary["success_count"] == len(scores)
    assert all(1 <= int(row["score"]) <= 5 for row in scores)
    assert all(row["dimension"].startswith("AR") for row in scores)


def test_claim_aggregation_preserves_document_scores_and_writes_comparison_report(tmp_path):
    root = tmp_path / "06_scoring"
    _write_claim_fixture(root)
    original = (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8")
    extract_mda_claims(root=root)
    run_claim_scoring(root=root, source_type="mda", mock=True)

    aggregate_summary = aggregate_claim_scores(root=root, source_type="mda")
    comparison_summary = compare_claim_vs_document_scoring(root=root)

    assert (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8") == original
    claim_scores = read_csv_rows(root / "10_claim_mapping" / "mda_document_scores_claim_level.csv")
    assert aggregate_summary["document_count"] == 1
    assert claim_scores[0]["document_id"] == "MDA_CLAIM_2024"
    assert claim_scores[0]["scoring_level"] == "claim"
    assert all(claim_scores[0][code] for code in ["AR01", "AR02", "AR03", "AR04", "AR05"])
    report = (root / "08_reports" / "claim_vs_document_comparison.md").read_text(encoding="utf-8")
    assert "JSON success rate" in report
    assert "AR02 error rate" in report
    assert "claim-level" in comparison_summary["recommended_primary_system"]


def test_self_check_registers_five_claim_rounds():
    assert claim_round_names() == [
        "Round claim-1",
        "Round claim-2",
        "Round claim-3",
        "Round claim-4",
        "Round claim-5",
    ]
