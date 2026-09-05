from __future__ import annotations

import json

from build_final_deliverable_clean import build_final_deliverable_clean
from scoring_utils import save_json, write_csv_rows


def test_build_final_deliverable_clean_writes_blocker_aware_report_and_sections(tmp_path):
    root = tmp_path / "06_scoring"
    save_json(
        root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json",
        {
            "paper_ready": False,
            "freeze_allowed": False,
            "mda_ready": True,
            "news_ready": False,
            "cross_validation_ready": False,
            "reproducibility_ready": True,
            "mda_final_dataset_size": 149,
            "news_final_dataset_size": 0,
            "blocking_reasons": ["real_news_exists"],
            "recommended_next_actions": ["provide real news_raw.csv"],
        },
    )
    write_csv_rows(root / "FINAL_OUTPUT" / "mda_final_dataset.csv", ["document_id"], [{"document_id": "MDA_1"}])
    write_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv", ["document_id"], [])
    write_csv_rows(root / "01_registry" / "news_registry_rebuilt.csv", ["document_id", "include_flag"], [])
    (root / "PAPER_OUTPUT" / "00_status").mkdir(parents=True, exist_ok=True)
    (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md").write_text("# Blocked\n", encoding="utf-8")
    (root / "HUMAN_VALIDATION").mkdir(parents=True, exist_ok=True)
    (root / "HUMAN_VALIDATION" / "mda_human_validation_instructions.md").write_text("# Instructions\n", encoding="utf-8")
    (root.parent / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (root.parent / "run_reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    summary = build_final_deliverable_clean(root=root)

    out = root / "FINAL_DELIVERABLE_CLEAN"
    assert summary["paper_ready"] is False
    assert (out / "MDA" / "mda_final_document_scores.csv").exists()
    assert (out / "NEWS" / "real_news_missing_blocker.md").exists()
    assert (out / "HUMAN_VALIDATION" / "mda_human_validation_instructions.md").exists()
    report = (out / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "MD&A-only scoring is complete as a model-generated candidate dataset." in report
    assert "News empirical analysis is blocked due to absence of real company-specific full-text News." in report
    assert "Cross-validation is framework-ready but not empirically interpretable." in report
    assert "Combined paper_ready remains false." in report
    assert "MD&A-only results may be used only with caveat and recommended human validation." in report
    status = json.loads((out / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    assert status["deliverable_type"] == "final_deliverable_clean_blocker_aware"


def test_final_deliverable_clean_embeds_current_news_scripts_and_tests(tmp_path):
    root = tmp_path / "06_scoring"
    save_json(
        root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json",
        {
            "paper_ready": False,
            "freeze_allowed": False,
            "mda_ready": True,
            "news_ready": True,
            "cross_validation_ready": False,
            "reproducibility_ready": True,
            "mda_final_dataset_size": 149,
            "news_final_dataset_size": 145,
            "synthetic_news_count": 0,
            "claim_link_count": 0,
            "blocking_reasons": ["cross_validation_run_complete"],
        },
    )
    write_csv_rows(root / "FINAL_OUTPUT" / "mda_final_dataset.csv", ["document_id"], [{"document_id": "MDA_1"}])
    write_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv", ["document_id"], [{"document_id": "NEWS_1"}])
    (root.parent / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    (root.parent / "run_reproduce.sh").write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    build_final_deliverable_clean(root=root)

    packaged = root / "FINAL_DELIVERABLE_CLEAN" / "06_scoring"
    out = root / "FINAL_DELIVERABLE_CLEAN"
    assert not (out / "NEWS" / "real_news_missing_blocker.md").exists()
    report = (out / "FINAL_REPORT.md").read_text(encoding="utf-8")
    assert "News scoring is ready with 145 real non-synthetic final rows." in report
    for rel in [
        "scripts/audit_filter_news_relevance.py",
        "scripts/build_news_registry_from_filter_news.py",
        "scripts/extract_filter_news_texts.py",
        "scripts/run_filter_news_final_scoring.py",
        "scripts/run_final_data_freeze.py",
        "scripts/paper_utils.py",
        "scripts/run_paper_readiness_check.py",
        "tests/test_news_refilter_pipeline.py",
        "04_prompts/scoring_output_schema.json",
        "04_prompts/mda_scoring_prompt.txt",
        "04_prompts/news_scoring_prompt.txt",
    ]:
        assert (packaged / rel).exists()
    assert "news_registry_rebuilt.csv" not in (packaged / "scripts" / "paper_utils.py").read_text(encoding="utf-8").split("def news_registry_path", 1)[1].split("def read_news_registry_rows", 1)[0].strip().splitlines()[0]
