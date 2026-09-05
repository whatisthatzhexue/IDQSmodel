from __future__ import annotations

import json

from build_final_deliverable_with_news import build_final_deliverable_with_news
from scoring_utils import save_json


def test_build_final_deliverable_with_news_packages_blocker_state(tmp_path):
    root = tmp_path / "project" / "06_scoring"
    project_root = root.parent
    project_root.mkdir(parents=True)
    (project_root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (project_root / "run_reproduce.sh").write_text("#!/usr/bin/env bash\npython3 -m pytest 06_scoring/tests -q\n", encoding="utf-8")
    (root / "FINAL_OUTPUT").mkdir(parents=True)
    (root / "FINAL_OUTPUT" / "mda_final_dataset.csv").write_text("document_id,total_score_100\nMDA_A,80\n", encoding="utf-8")
    (root / "06_ratings" / "mda_final_full").mkdir(parents=True)
    (root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv").write_text("document_id,dimension_code,raw_score\nMDA_A,AR01,4\n", encoding="utf-8")
    (root / "FINAL_OUTPUT" / "news_final_dataset.csv").write_text("document_id\n", encoding="utf-8")
    (root / "FINAL_OUTPUT" / "final_quality_report.md").write_text("# Quality\n", encoding="utf-8")
    save_json(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json", {"freeze_allowed": False})
    save_json(
        root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json",
        {
            "mda_ready": True,
            "news_ready": False,
            "cross_validation_ready": False,
            "reproducibility_ready": True,
            "freeze_allowed": False,
            "paper_ready": False,
            "news_final_dataset_size": 0,
            "synthetic_news_count": 0,
            "eligible_company_year_pair_count": 0,
            "claim_link_count": 0,
            "system_level_error_rate": 0.0,
            "blocking_reasons": ["real_news_exists", "news_final_dataset_size"],
            "recommended_next_actions": ["provide real news_raw.csv or news_raw.xlsx"],
        },
    )
    (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md").write_text(
        "# Real News Missing Blocker\n\nMD&A final is complete and frozen.\n",
        encoding="utf-8",
    )

    summary = build_final_deliverable_with_news(root=root, test_results="133 passed")
    out = root / "FINAL_DELIVERABLE_WITH_NEWS"
    status = json.loads((out / "FINAL_STATUS.json").read_text(encoding="utf-8"))

    assert summary["status_path"] == str(out / "FINAL_STATUS.json")
    assert status["mda_ready"] is True
    assert status["news_ready"] is False
    assert status["paper_ready"] is False
    assert (out / "MDA" / "mda_final_document_scores.csv").exists()
    assert (out / "NEWS" / "real_news_missing_blocker.md").exists()
    assert (out / "REPRODUCIBILITY" / "requirements.txt").read_text(encoding="utf-8") == "pytest\n"
    assert "133 passed" in (out / "REPRODUCIBILITY" / "test_results.txt").read_text(encoding="utf-8")
    assert "paper_ready: false" in (out / "FINAL_PACKAGE_MANIFEST.md").read_text(encoding="utf-8")


def test_build_final_deliverable_with_news_does_not_package_missing_blocker_when_news_ready(tmp_path):
    root = tmp_path / "project" / "06_scoring"
    project_root = root.parent
    project_root.mkdir(parents=True)
    (project_root / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (project_root / "run_reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    (root / "FINAL_OUTPUT").mkdir(parents=True)
    (root / "FINAL_OUTPUT" / "mda_final_dataset.csv").write_text("document_id,total_score_100\nMDA_A,80\n", encoding="utf-8")
    (root / "FINAL_OUTPUT" / "news_final_dataset.csv").write_text("document_id,total_score_100\nNEWS_A,70\n", encoding="utf-8")
    (root / "06_ratings" / "mda_final_full").mkdir(parents=True)
    (root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv").write_text("document_id,dimension_code,raw_score\nMDA_A,AR01,4\n", encoding="utf-8")
    (root / "06_ratings" / "news_final_full").mkdir(parents=True)
    (root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv").write_text("document_id,dimension_code,raw_score\nNEWS_A,N01,4\n", encoding="utf-8")
    save_json(
        root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json",
        {
            "mda_ready": True,
            "news_ready": True,
            "cross_validation_ready": False,
            "reproducibility_ready": True,
            "freeze_allowed": True,
            "paper_ready": False,
            "news_final_dataset_size": 1,
            "synthetic_news_count": 0,
            "eligible_company_year_pair_count": 0,
            "claim_link_count": 0,
            "system_level_error_rate": 0.0,
            "blocking_reasons": ["cross_validation_not_interpretable"],
            "recommended_next_actions": ["review cross-validation"],
        },
    )
    (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md").write_text(
        "# Historical News Missing Blocker\n",
        encoding="utf-8",
    )

    build_final_deliverable_with_news(root=root, test_results="ok")
    out = root / "FINAL_DELIVERABLE_WITH_NEWS"
    status = json.loads((out / "FINAL_STATUS.json").read_text(encoding="utf-8"))

    assert status["news_ready"] is True
    assert status["real_news_missing_blocker_exists"] is False
    assert not (out / "NEWS" / "real_news_missing_blocker.md").exists()
