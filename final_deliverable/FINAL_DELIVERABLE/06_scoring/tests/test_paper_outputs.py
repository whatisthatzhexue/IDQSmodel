from __future__ import annotations

import json
from pathlib import Path

from build_paper_data_section import build_paper_data_section
from build_paper_methods_section import build_paper_methods_section
from build_paper_ready_report import build_paper_ready_report
from build_paper_results_tables import build_paper_results_tables
from build_paper_scoring_framework import build_paper_scoring_framework
from paper_utils import ensure_paper_output_dirs
from scoring_utils import MDA_DOCUMENT_HEADERS, save_json, write_csv_rows


def test_paper_output_directories_and_core_tables_are_created(tmp_path):
    root = tmp_path / "06_scoring"
    ensure_paper_output_dirs(root)
    _write_minimal_mda_final(root)
    _write_status(root, paper_ready=False, freeze_allowed=False)

    build_paper_methods_section(root=root)
    build_paper_data_section(root=root)
    build_paper_scoring_framework(root=root)
    build_paper_results_tables(root=root)
    status = build_paper_ready_report(root=root)

    assert status["paper_ready"] is False
    for rel_path in [
        "PAPER_OUTPUT/01_methods/methods_section_cn.md",
        "PAPER_OUTPUT/01_methods/methods_section_en.md",
        "PAPER_OUTPUT/02_data/data_section_en.md",
        "PAPER_OUTPUT/tables/table_1_sample_construction.csv",
        "PAPER_OUTPUT/tables/table_2_variable_definitions.md",
        "PAPER_OUTPUT/tables/table_3_mda_dimensions.csv",
        "PAPER_OUTPUT/tables/table_4_news_dimensions.md",
        "PAPER_OUTPUT/tables/table_5_weighting_schemes.csv",
        "PAPER_OUTPUT/tables/table_6_mda_score_summary.csv",
        "PAPER_OUTPUT/tables/table_11_final_freeze_summary.md",
        "PAPER_OUTPUT/PAPER_READY_REPORT.md",
        "PAPER_OUTPUT/PAPER_READY_STATUS.json",
    ]:
        assert (root / rel_path).exists(), rel_path


def test_missing_news_creates_empty_news_table_with_missing_note(tmp_path):
    root = tmp_path / "06_scoring"
    _write_minimal_mda_final(root)
    _write_status(root, paper_ready=False, freeze_allowed=False)

    build_paper_data_section(root=root)
    build_paper_results_tables(root=root)

    news_summary = (root / "PAPER_OUTPUT" / "tables" / "table_7_news_score_summary.md").read_text(encoding="utf-8")
    data_section = (root / "PAPER_OUTPUT" / "02_data" / "data_section_en.md").read_text(encoding="utf-8")
    assert "missing" in news_summary.lower()
    assert "news-based analyses and cross-validation are not available" in data_section.lower()


def test_paper_builders_do_not_overwrite_scoring_files(tmp_path):
    root = tmp_path / "06_scoring"
    original = "document_id,total_score_100\nKEEP,50\n"
    path = root / "06_ratings" / "mda_document_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(original, encoding="utf-8")
    _write_status(root, paper_ready=False, freeze_allowed=False)

    build_paper_methods_section(root=root)
    build_paper_data_section(root=root)
    build_paper_scoring_framework(root=root)

    assert path.read_text(encoding="utf-8") == original


def _write_minimal_mda_final(root: Path) -> None:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update({"document_id": "MDA_001", "doc_type": "mda", "ticker": "AAA", "company_name": "Alpha Foods", "report_year": "2024", "total_score_100": "75", "grade_label": "B", "AR01": "4", "AR02": "4", "AR03": "4", "AR04": "4", "AR05": "4"})
    write_csv_rows(root / "FINAL_OUTPUT" / "mda_final_dataset.csv", [*MDA_DOCUMENT_HEADERS, "final_dataset_version"], [row])


def _write_status(root: Path, *, paper_ready: bool, freeze_allowed: bool) -> None:
    save_json(
        root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json",
        {
            "paper_ready": paper_ready,
            "freeze_allowed": freeze_allowed,
            "mda_ready": False,
            "news_ready": False,
            "cross_validation_ready": False,
            "reproducibility_ready": True,
            "mda_final_dataset_size": 1,
            "news_final_dataset_size": 0,
            "synthetic_news_count": 0,
            "claim_link_count": 0,
            "mda_stability_success_rate": 0.59,
            "system_level_error_rate": 0.0,
            "blocking_reasons": ["mda_stability_success_rate", "news_final_dataset_or_real_registry_not_ready"],
            "advisory_warnings": [],
            "recommended_next_actions": ["score real News documents"],
        },
    )
