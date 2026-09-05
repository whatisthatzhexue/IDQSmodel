from __future__ import annotations

from pathlib import Path

from check_real_news_availability import check_real_news_availability
from diagnose_mda_stability_blocker import CASE_HEADERS
from repair_mda_stability_failures import repair_mda_stability_failures
from scoring_utils import write_csv_rows


def test_repair_mda_stability_failures_repairs_json_without_overwriting_history(tmp_path):
    root = tmp_path / "06_scoring"
    raw_path = root / "05_raw_model_outputs" / "mda_v2_stability_batch" / "MDA_A_AR01.raw.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text('```json\n{"score": 4, "evidence": "MDA_P001", "reason": "Clear explanation."}\n```', encoding="utf-8")
    text_path = root / "02_extracted_text" / "mda" / "MDA_A.txt"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text("[MDA_P001] Management explains operating performance.", encoding="utf-8")
    old_parsed = root / "06_ratings" / "mda_v2_stability_batch" / "per_dimension" / "MDA_A_AR01_parsed.json"
    old_parsed.parent.mkdir(parents=True, exist_ok=True)
    old_parsed.write_text('{"status":"failed"}', encoding="utf-8")
    write_csv_rows(
        root / "PAPER_OUTPUT" / "00_status" / "mda_stability_blocker_cases.csv",
        CASE_HEADERS,
        [
            {
                "document_id": "MDA_A",
                "ticker": "AAA",
                "company_name": "Alpha Foods",
                "report_year": "2024",
                "dimension_code": "AR01",
                "stability_case_type": "stability_batch",
                "failure_reason": "json_parse_failure",
                "source_file": "mda_v2_stability_batch",
                "raw_output_path": str(raw_path.relative_to(root)),
                "parsed_output_path": str(old_parsed.relative_to(root)),
                "evidence_locator": "MDA_P001",
                "evidence_exists": "true",
                "score_current": "",
                "score_comparison": "",
                "score_diff": "",
                "needs_rescore": "false",
                "recommended_action": "repair_json_only",
            }
        ],
    )

    summary = repair_mda_stability_failures(root=root)

    repaired_path = root / "06_ratings" / "mda_stability_repair" / "per_dimension" / "MDA_A_AR01_parsed.json"
    assert summary["number_of_cases_repaired"] == 1
    assert repaired_path.exists()
    assert '"status": "success"' in repaired_path.read_text(encoding="utf-8")
    assert old_parsed.read_text(encoding="utf-8") == '{"status":"failed"}'
    actions = (root / "PAPER_OUTPUT" / "00_status" / "mda_stability_repair_actions.csv").read_text(encoding="utf-8")
    assert "repair_json_only" in actions
    assert "repaired" in actions
    assert (root / "PAPER_OUTPUT" / "00_status" / "mda_stability_repair_log.md").exists()


def test_check_real_news_availability_writes_missing_blocker_without_synthetic_data(tmp_path):
    root = tmp_path / "06_scoring"

    summary = check_real_news_availability(root=root)

    status_dir = root / "PAPER_OUTPUT" / "00_status"
    assert summary["real_news_file_count"] == 0
    assert summary["real_news_available"] is False
    blocker = (status_dir / "real_news_missing_blocker.md").read_text(encoding="utf-8")
    assert "news_ready remains false" in blocker
    assert "synthetic news is not used" in blocker
    assert "ticker, company_name, publish_date, source_name, title, url, article_text" in blocker
    assert (status_dir / "real_news_files_found.csv").exists()


def test_check_real_news_availability_finds_canonical_news_csv(tmp_path):
    root = tmp_path / "06_scoring"
    news_csv = root / "input_news" / "news_raw.csv"
    news_csv.parent.mkdir(parents=True, exist_ok=True)
    news_csv.write_text("ticker,company_name,publish_date,source_name,title,url,article_text\nAAA,Alpha,2024-01-01,Source,Title,,Body\n", encoding="utf-8")

    summary = check_real_news_availability(root=root)

    status_dir = root / "PAPER_OUTPUT" / "00_status"
    assert summary["real_news_file_count"] == 1
    assert summary["real_news_available"] is True
    assert "news_raw.csv" in (status_dir / "real_news_files_found.csv").read_text(encoding="utf-8")
    assert not (status_dir / "real_news_missing_blocker.md").exists()
