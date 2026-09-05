from __future__ import annotations

from pathlib import Path

from build_paper_cross_validation_section import build_paper_cross_validation_section
from build_paper_limitations import build_paper_limitations
from build_paper_methods_section import build_paper_methods_section
from build_paper_ready_report import build_paper_ready_report
from scoring_utils import save_json, write_csv_rows


def test_methods_use_mda_disclosure_quality_not_annual_report_quality(tmp_path):
    root = tmp_path / "06_scoring"
    _write_status(root)

    build_paper_methods_section(root=root)

    text = (root / "PAPER_OUTPUT" / "01_methods" / "methods_section_en.md").read_text(encoding="utf-8").lower()
    assert "md&a disclosure quality" in text
    assert "financial statement verification" not in text
    assert "annual report quality" not in text
    assert "internal numeric consistency" in text


def test_reports_do_not_claim_final_empirical_results_when_not_ready(tmp_path):
    root = tmp_path / "06_scoring"
    _write_status(root)

    build_paper_ready_report(root=root)
    build_paper_limitations(root=root)

    report = (root / "PAPER_OUTPUT" / "PAPER_READY_REPORT.md").read_text(encoding="utf-8").lower()
    limitations = (root / "PAPER_OUTPUT" / "07_limitations" / "limitations_en.md").read_text(encoding="utf-8").lower()
    assert "final empirical conclusions are not allowed" in report
    assert "pipeline validation materials rather than final empirical findings" in limitations
    assert "results can be submitted: no" in report


def test_cross_validation_without_news_is_not_interpreted_as_empirical_evidence(tmp_path):
    root = tmp_path / "06_scoring"
    _write_status(root)
    write_csv_rows(root / "09_cross_validation" / "claim_links.csv", ["link_id"], [])

    build_paper_cross_validation_section(root=root)

    text = (root / "PAPER_OUTPUT" / "06_cross_validation" / "cross_validation_section_en.md").read_text(encoding="utf-8").lower()
    assert "does not alter md&a or news scores" in text
    assert "cannot be interpreted as empirical evidence" in text
    assert "should not be described as passed" in text
    assert "validates truth" not in text
    assert "news proves md&a true" not in text


def _write_status(root: Path) -> None:
    save_json(
        root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json",
        {
            "paper_ready": False,
            "freeze_allowed": False,
            "mda_ready": False,
            "news_ready": False,
            "cross_validation_ready": False,
            "reproducibility_ready": True,
            "mda_final_dataset_size": 16,
            "news_final_dataset_size": 0,
            "synthetic_news_count": 0,
            "claim_link_count": 0,
            "mda_stability_success_rate": 0.592593,
            "system_level_error_rate": 0.0,
            "blocking_reasons": ["mda_stability_success_rate", "news_final_dataset_or_real_registry_not_ready"],
            "advisory_warnings": [],
            "recommended_next_actions": ["restore qwen3:8b and score real News documents"],
        },
    )
