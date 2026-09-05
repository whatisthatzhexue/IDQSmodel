from __future__ import annotations

import json
from pathlib import Path

from build_news_minimal_pipeline import build_news_minimal_pipeline
from news_pipeline_status_check import check_news_pipeline_status
from run_final_data_freeze import run_final_data_freeze
from scoring_utils import MDA_DOCUMENT_HEADERS, NEWS_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows


def test_news_status_check_reports_missing_pipeline_without_fabricating_data(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], [])

    summary = check_news_pipeline_status(root=root)

    assert summary["news_registry"]["exists"] is True
    assert summary["news_registry"]["row_count"] == 0
    assert summary["news_text_extraction"]["row_count"] == 0
    assert summary["news_scoring"]["document_score_count"] == 0
    report = root / "08_reports" / "news_pipeline_status_report.md"
    assert report.exists()
    assert "missing" in report.read_text(encoding="utf-8").lower()


def test_build_news_minimal_pipeline_creates_synthetic_pilot_scores_when_news_missing(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], [])

    summary = build_news_minimal_pipeline(root=root)

    assert summary["created_synthetic_placeholder"] is True
    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    score_rows = read_csv_rows(root / "06_ratings" / "news_document_scores.csv")
    ratings_rows = read_csv_rows(root / "06_ratings" / "news_ratings_long.csv")
    assert len(registry_rows) >= 2
    assert len(score_rows) == len(registry_rows)
    assert len(ratings_rows) == len(registry_rows) * 5
    assert all("synthetic_flag=true" in row["notes"] for row in registry_rows)
    assert all(row["schema_validation_status"] == "valid" for row in ratings_rows)
    assert all(row["parse_status"] == "parsed" for row in ratings_rows)
    assert all(row["evidence_locator"].startswith("[NEWS_P") for row in ratings_rows)


def test_news_status_check_does_not_treat_synthetic_scores_as_ready(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], [])
    build_news_minimal_pipeline(root=root)

    summary = check_news_pipeline_status(root=root)

    assert summary["overall_status"] == "synthetic_only"
    assert summary["news_registry"]["real_included_count"] == 0
    assert summary["news_scoring"]["real_document_score_count"] == 0
    assert summary["news_stability"]["can_run"] is False


def test_news_status_check_reads_current_final_news_paths(tmp_path):
    root = tmp_path / "06_scoring"
    text_path = root / "02_extracted_text" / "news_clean" / "NEWS_REAL_2024.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("Alpha Foods announced a new product expansion plan.", encoding="utf-8")
    raw_dir = root / "05_raw_model_outputs" / "news_final_full"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "NEWS_REAL_2024_N01.json").write_text('{"score": 4}\n', encoding="utf-8")
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_REAL_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "synthetic_flag": "false",
                "company_name": "Alpha Foods",
                "ticker": "AAA",
                "source_name": "Business Wire",
                "publish_date": "2024-03-01",
                "title": "Alpha Foods launches expansion",
                "text_path": "02_extracted_text/news_clean/NEWS_REAL_2024.txt",
                "notes": "synthetic_flag=false",
            }
        ],
    )
    write_csv_rows(
        root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv",
        NEWS_DOCUMENT_HEADERS,
        [_news_doc("NEWS_REAL_2024", "AAA", "Alpha Foods", "2024-03-01", "Business Wire", "Alpha Foods launches expansion", 78)],
    )
    write_csv_rows(
        root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv",
        RATINGS_LONG_HEADERS,
        _rating_rows("news", "NEWS_REAL_2024", ["N01", "N02", "N03", "N04", "N05"]),
    )

    summary = check_news_pipeline_status(root=root)

    assert summary["overall_status"] == "ready"
    assert summary["news_raw"]["path"].endswith("05_raw_model_outputs/news_final_full")
    assert summary["news_text_extraction"]["path"].endswith("02_extracted_text/news_clean")
    assert summary["news_text_extraction"]["row_count"] == 1
    assert summary["news_scoring"]["document_scores_path"].endswith("06_ratings/news_final_full/news_final_document_scores.csv")
    assert summary["news_scoring"]["real_document_score_count"] == 1
    assert summary["news_scoring"]["real_rating_row_count"] == 5


def test_freeze_gate_uses_news_exists_and_schema_valid_but_still_blocks_low_mda_stability(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_stability(root, selected=2, success=1)
    build_news_minimal_pipeline(root=root)

    summary = run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert summary["freeze_allowed"] is False
    assert metadata["quality_gates"]["mda_stability_success_rate"]["passed"] is False
    assert metadata["quality_gates"]["news_exists"]["passed"] is False
    assert metadata["quality_gates"]["news_schema_valid"]["passed"] is False
    assert metadata["final_dataset_size"]["news"] == 0


def test_freeze_gate_blocks_synthetic_news_even_when_mda_stability_and_schema_pass(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_stability(root, selected=2, success=2)
    build_news_minimal_pipeline(root=root)

    summary = run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert summary["freeze_allowed"] is False
    assert metadata["quality_gates"]["mda_stability_success_rate"]["passed"] is True
    assert metadata["quality_gates"]["news_exists"]["passed"] is False
    assert metadata["quality_gates"]["news_schema_valid"]["passed"] is False
    assert metadata["quality_gates"]["cross_validation_claim_links"]["passed"] is False


def _write_mda_stability(root: Path, selected: int, success: int) -> None:
    selected_rows = [{"document_id": f"MDA_{idx:03d}"} for idx in range(selected)]
    success_rows = [_mda_doc(f"MDA_{idx:03d}", "AAA", "Alpha Foods", "2024", 72) for idx in range(success)]
    write_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv", ["document_id"], selected_rows)
    write_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv", MDA_DOCUMENT_HEADERS, success_rows)
    rating_rows = []
    for row in success_rows:
        rating_rows.extend(_rating_rows("mda", row["document_id"], ["AR01", "AR02", "AR03", "AR04", "AR05"]))
    write_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_ratings_long.csv", RATINGS_LONG_HEADERS, rating_rows)


def _mda_doc(document_id: str, ticker: str, company: str, year: str, total: int) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "mda", "ticker": ticker, "company_name": company, "report_year": year, "dimension_count": "5", "total_score_100": str(total), "AR01": "4", "AR02": "4", "AR03": "4", "AR04": "4", "AR05": "4"})
    return row


def _news_doc(document_id: str, ticker: str, company: str, publish_date: str, source_name: str, title: str, total: int) -> dict[str, str]:
    row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "news", "ticker": ticker, "company_name": company, "source_name": source_name, "publish_date": publish_date, "title": title, "dimension_count": "5", "total_score_100": str(total), "N01": "4", "N02": "4", "N03": "4", "N04": "4", "N05": "4"})
    return row


def _rating_rows(doc_type: str, document_id: str, dimensions: list[str]) -> list[dict[str, str]]:
    prefix = "MDA" if doc_type == "mda" else "NEWS"
    rows = []
    for idx, code in enumerate(dimensions, start=1):
        row = {field: "" for field in RATINGS_LONG_HEADERS}
        row.update(
            {
                "rating_id": f"{document_id}_{code}",
                "document_id": document_id,
                "doc_type": doc_type,
                "dimension_code": code,
                "raw_score": "4",
                "std_score_100": "75",
                "weighted_score": "15",
                "evidence_locator": f"[{prefix}_P{idx:03d}]",
                "confidence_level": "high",
                "parse_status": "parsed",
                "schema_validation_status": "valid",
            }
        )
        rows.append(row)
    return rows
