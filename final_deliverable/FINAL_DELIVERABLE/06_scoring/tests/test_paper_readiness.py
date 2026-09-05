from __future__ import annotations

import json
from pathlib import Path

from paper_utils import PAPER_STATUS_FIELDS
from run_paper_readiness_check import run_paper_readiness_check
from scoring_utils import NEWS_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, REGISTRY_HEADERS, save_json, write_csv_rows


def test_paper_readiness_blocks_synthetic_news_and_reports_blockers(tmp_path):
    root = tmp_path / "06_scoring"
    _write_freeze_metadata(root, freeze_allowed=False, mda_size=2, news_size=2, synthetic_count=2, claim_links=0)
    _write_synthetic_news_scores(root)

    summary = run_paper_readiness_check(root=root)

    assert summary["paper_ready"] is False
    assert summary["news_ready"] is False
    assert "news_final_dataset_or_real_registry_not_ready" in summary["blocking_reasons"]
    assert not any(action.startswith("repair MD&A") for action in summary["recommended_next_actions"])
    assert set(PAPER_STATUS_FIELDS).issubset(summary.keys())
    report = root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_report.md"
    text = report.read_text(encoding="utf-8")
    assert "not paper-ready" in text.lower()
    assert "synthetic news" in text.lower()


def test_paper_readiness_records_missing_inputs_without_crashing(tmp_path):
    root = tmp_path / "06_scoring"

    summary = run_paper_readiness_check(root=root)

    assert summary["paper_ready"] is False
    assert summary["missing_inputs"]
    assert (root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json").exists()
    assert "missing input" in (root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_report.md").read_text(encoding="utf-8").lower()


def test_paper_readiness_recommends_real_news_input_before_scoring_when_registry_missing(tmp_path):
    root = tmp_path / "06_scoring"
    _write_freeze_metadata(root, freeze_allowed=False, mda_size=149, news_size=0, synthetic_count=0, claim_links=0)

    summary = run_paper_readiness_check(root=root)

    assert summary["mda_ready"] is True
    assert summary["news_ready"] is False
    assert summary["recommended_next_actions"][0].startswith("provide real news_raw")


def test_paper_readiness_status_fields_are_complete(tmp_path):
    root = tmp_path / "06_scoring"
    _write_freeze_metadata(root, freeze_allowed=False, mda_size=1, news_size=0, synthetic_count=0, claim_links=0)

    summary = run_paper_readiness_check(root=root)

    assert list(PAPER_STATUS_FIELDS) == [
        "paper_ready",
        "freeze_allowed",
        "mda_structural_ready",
        "mda_semantic_ready",
        "mda_human_validation_ready",
        "mda_ready",
        "news_ready",
        "cross_validation_ready",
        "cross_validation_pipeline_ready",
        "cross_validation_empirical_ready",
        "mock_mode",
        "systemic_conflict_flag",
        "reproducibility_ready",
        "mda_final_dataset_size",
        "news_final_dataset_size",
        "news_company_focused_dataset_size",
        "synthetic_news_count",
        "claim_link_count",
        "eligible_company_year_pair_count",
        "mda_stability_success_rate",
        "system_level_error_rate",
        "blocking_reasons",
        "advisory_warnings",
        "recommended_next_actions",
    ]
    assert set(PAPER_STATUS_FIELDS).issubset(summary)
    assert "mda_structural_ready" in summary
    assert "mda_semantic_ready" in summary
    assert "mda_human_validation_ready" in summary


def test_paper_readiness_records_mock_mode_and_systemic_conflict(tmp_path):
    root = tmp_path / "06_scoring"
    _write_freeze_metadata(root, freeze_allowed=True, mda_size=1, news_size=1, synthetic_count=0, claim_links=1)
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    metadata["cross_validation"]["systemic_conflict_flag"] = True
    save_json(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json", metadata)
    save_json(
        root / "09_cross_validation" / "final_real_run" / "cross_validation_run_metadata.json",
        {
            "mock_mode": True,
            "systemic_conflict_flag": True,
            "cross_validation_pipeline_ready": True,
            "cross_validation_empirical_ready": False,
            "claim_link_count": 1,
        },
    )
    (root / "09_cross_validation").mkdir(parents=True, exist_ok=True)
    (root / "09_cross_validation" / "cross_validation_report.md").write_text("mock_mode=true\n", encoding="utf-8")

    summary = run_paper_readiness_check(root=root)
    saved = json.loads((root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json").read_text(encoding="utf-8"))

    assert summary["mock_mode"] is True
    assert summary["systemic_conflict_flag"] is True
    assert saved["mock_mode"] is True
    assert saved["systemic_conflict_flag"] is True
    assert summary["cross_validation_empirical_ready"] is False
    assert "systematic_matching_failure_detected" in summary["blocking_reasons"]


def test_paper_readiness_json_preserves_cross_validation_counts(tmp_path):
    root = tmp_path / "06_scoring"
    _write_freeze_metadata(root, freeze_allowed=False, mda_size=1, news_size=0, synthetic_count=0, claim_links=0)

    run_paper_readiness_check(root=root)
    saved = json.loads((root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json").read_text(encoding="utf-8"))

    assert saved["eligible_company_year_pair_count"] == 0
    assert saved["cross_validation_pipeline_ready"] is False
    assert saved["cross_validation_empirical_ready"] is False


def test_paper_readiness_falls_back_when_rebuilt_registry_is_empty(tmp_path):
    root = tmp_path / "06_scoring"
    _write_freeze_metadata(root, freeze_allowed=True, mda_size=1, news_size=1, synthetic_count=0, claim_links=1)
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_LEGACY_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Legacy Foods",
                "ticker": "LEG",
                "source_name": "Legacy Source",
                "publish_date": "2024-03-01",
                "title": "Legacy article",
                "url": "https://example.com/legacy",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        root / "01_registry" / "news_registry_rebuilt.csv",
        [
            "document_id",
            "doc_type",
            "ticker",
            "company_name",
            "publish_date",
            "source_name",
            "title",
            "url",
            "file_path",
            "article_text_hash",
            "text_length",
            "synthetic_flag",
            "include_flag",
            "exclude_reason",
            "created_at",
        ],
        [],
    )

    summary = run_paper_readiness_check(root=root)

    assert summary["paper_ready"] is False
    assert summary["news_ready"] is True
    assert "news_registry_real" not in summary["blocking_reasons"]
    assert "real_news_exists" not in summary["blocking_reasons"]
    report = (root / "08_reports" / "news_registry_selection_report.md").read_text(encoding="utf-8")
    assert "fallback" in report.lower()


def _write_freeze_metadata(root: Path, *, freeze_allowed: bool, mda_size: int, news_size: int, synthetic_count: int, claim_links: int) -> None:
    save_json(
        root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json",
        {
            "freeze_allowed": freeze_allowed,
            "final_dataset_size": {"mda": mda_size, "news": news_size},
            "system_level_error_rate": 0.0,
            "quality_gates": {
                "mda_stability_success_rate": {"passed": mda_size > 0, "value": 0.9 if mda_size > 0 else 0.0},
                "mda_no_schema_failure_critical": {"passed": True, "value": 0},
                "mda_evidence_locator_success": {"passed": True, "value": 1.0},
                "mda_ar02_numeric_review_controlled": {"passed": True, "value": mda_size},
                "news_exists": {"passed": news_size > 0 and synthetic_count == 0, "value": news_size},
                "news_schema_valid": {"passed": news_size > 0 and synthetic_count == 0, "value": 1.0 if news_size else 0.0},
                "news_json_parse_failure": {"passed": news_size > 0 and synthetic_count == 0, "value": 0.0 if news_size else 1.0},
                "news_registry_real": {"passed": synthetic_count == 0 and news_size > 0, "value": news_size},
                "cross_validation_claim_links": {"passed": claim_links > 0, "value": claim_links},
                "system_level_error_rate": {"passed": True, "value": 0.0},
            },
            "news": {"extra": {"synthetic_count": synthetic_count, "real_count": 0 if synthetic_count else news_size}},
            "cross_validation": {"claim_link_count": claim_links, "systemic_conflict_flag": False},
        },
    )
    (root / "FINAL_OUTPUT" / "final_quality_report.md").parent.mkdir(parents=True, exist_ok=True)
    (root / "FINAL_OUTPUT" / "final_quality_report.md").write_text("# Final Quality Report\n", encoding="utf-8")


def _write_synthetic_news_scores(root: Path) -> None:
    registry_rows = [
        {
            "document_id": "NEWS_SYN_001",
            "doc_type": "news",
            "include_flag": "yes",
            "company_name": "Synthetic Co",
            "ticker": "SYN",
            "source_name": "Synthetic Placeholder",
            "publish_date": "2024-01-01",
            "title": "Synthetic title",
            "url": "",
            "text_path": "02_extracted_text/news/NEWS_SYN_001.txt",
            "notes": "synthetic_flag=true",
        }
    ]
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], registry_rows)
    score_row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
    score_row.update({"document_id": "NEWS_SYN_001", "doc_type": "news", "ticker": "SYN", "company_name": "Synthetic Co", "total_score_100": "50", "N01": "3", "N02": "3", "N03": "3", "N04": "3", "N05": "3"})
    write_csv_rows(root / "06_ratings" / "news_document_scores.csv", NEWS_DOCUMENT_HEADERS, [score_row])
    rating_rows = []
    for code in ["N01", "N02", "N03", "N04", "N05"]:
        row = {field: "" for field in RATINGS_LONG_HEADERS}
        row.update({"document_id": "NEWS_SYN_001", "dimension_code": code, "parse_status": "parsed", "schema_validation_status": "valid", "evidence_locator": "[NEWS_P001]"})
        rating_rows.append(row)
    write_csv_rows(root / "06_ratings" / "news_ratings_long.csv", RATINGS_LONG_HEADERS, rating_rows)
