from __future__ import annotations

import json
from pathlib import Path

from build_blocker_status_reports import build_blocker_status_reports
from scoring_utils import REGISTRY_HEADERS, save_json, write_csv_rows


def test_blocker_status_reports_write_news_and_final_blockers(tmp_path):
    root = tmp_path / "06_scoring"
    save_json(
        root / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json",
        {
            "paper_ready": False,
            "freeze_allowed": False,
            "mda_ready": False,
            "news_ready": False,
            "cross_validation_ready": True,
            "reproducibility_ready": True,
            "mda_final_dataset_size": 16,
            "news_final_dataset_size": 0,
            "synthetic_news_count": 0,
            "claim_link_count": 1,
            "mda_stability_success_rate": 0.592593,
            "system_level_error_rate": 0.0,
            "blocking_reasons": ["mda_stability_success_rate", "news_final_dataset_size"],
            "advisory_warnings": [],
            "recommended_next_actions": ["score real News documents"],
        },
    )
    save_json(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json", {"quality_gates": {"mda_stability_success_rate": {"passed": False}, "system_level_error_rate": {"passed": True}}})
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_REAL_001",
                "doc_type": "news",
                "include_flag": "yes",
                "company_name": "Alpha",
                "ticker": "AAA",
                "source_name": "Source",
                "publish_date": "2026-01-01",
                "title": "Real title",
                "url": "",
                "text_path": "02_extracted_text/news/NEWS_REAL_001.txt",
                "notes": "synthetic_flag=false",
            }
        ],
    )

    result = build_blocker_status_reports(root=root, ollama_status={"ollama_available": False, "errors": ["HTTP 502"]})

    assert result["news_ready"] is False
    news_blocker = (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md")
    assert news_blocker.exists()
    blocker_text = news_blocker.read_text(encoding="utf-8")
    assert "news_ready remains false" in blocker_text
    assert "cross-validation cannot be interpreted empirically" in blocker_text
    assert "synthetic news is not used" in blocker_text
    assert "news_raw.xlsx or news_raw.csv" in blocker_text
    assert "ticker, company_name, publish_date, source_name, title, url, article_text" in blocker_text
    assert (root / "PAPER_OUTPUT" / "00_status" / "news_ready_report.md").exists()
    assert (root / "PAPER_OUTPUT" / "00_status" / "cross_validation_blocker_diagnosis.md").exists()
    final_text = (root / "PAPER_OUTPUT" / "00_status" / "final_blocker_list.md").read_text(encoding="utf-8")
    assert "mda_stability_success_rate" in final_text
    assert "final empirical results" in final_text.lower()
