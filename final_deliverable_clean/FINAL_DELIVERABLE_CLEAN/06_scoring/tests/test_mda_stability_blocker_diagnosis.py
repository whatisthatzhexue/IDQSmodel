from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from diagnose_mda_stability_blocker import CASE_HEADERS, diagnose_mda_stability_blocker
from scoring_utils import RATINGS_LONG_HEADERS, REGISTRY_HEADERS, save_json, write_csv_rows


def test_mda_stability_blocker_diagnosis_writes_cases_and_report(tmp_path):
    root = tmp_path / "06_scoring"
    batch_dir = root / "06_ratings" / "mda_v2_stability_batch"
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        REGISTRY_HEADERS["mda"],
        [
            {
                "document_id": "MDA_A",
                "ticker": "AAA",
                "company_name": "Alpha Foods",
                "report_year": "2024",
            },
            {
                "document_id": "MDA_B",
                "ticker": "BBB",
                "company_name": "Beta Drinks",
                "report_year": "2023",
            },
            {
                "document_id": "MDA_C",
                "ticker": "CCC",
                "company_name": "Cafe Co",
                "report_year": "2022",
            },
        ],
    )
    write_csv_rows(batch_dir / "mda_v2_stability_batch_selection.csv", ["document_id"], [{"document_id": "MDA_A"}, {"document_id": "MDA_B"}, {"document_id": "MDA_C"}])
    write_csv_rows(batch_dir / "mda_v2_stability_document_scores.csv", ["document_id", "total_score_100", "AR01", "AR02", "AR03", "AR04", "AR05"], [{"document_id": "MDA_A", "total_score_100": "75", "AR01": "4", "AR02": "4", "AR03": "4", "AR04": "4", "AR05": "4"}])
    rows = []
    for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
        row = {field: "" for field in RATINGS_LONG_HEADERS}
        row.update({"document_id": "MDA_A", "dimension_code": code, "parse_status": "parsed", "schema_validation_status": "valid", "evidence_locator": "[MDA_P001]", "confidence_level": "high"})
        rows.append(row)
    write_csv_rows(batch_dir / "mda_v2_stability_ratings_long.csv", RATINGS_LONG_HEADERS, rows)
    save_json(batch_dir / "per_document" / "MDA_B_parsed.json", {"status": "failed", "result": {"parse_status": "parse_failed", "schema_validation_status": "invalid", "error_type": "json_parse", "raw_output_path": "05_raw_model_outputs/mda_v2_stability_batch/MDA_B.raw.txt"}})
    save_json(batch_dir / "per_document" / "MDA_C_parsed.json", {"status": "failed", "result": {"error_type": "missing_output"}})
    write_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv", ["document_id", "num_material_discrepancies", "num_severe_direction_conflicts"], [{"document_id": "MDA_C", "num_material_discrepancies": "1", "num_severe_direction_conflicts": "0"}])

    summary = diagnose_mda_stability_blocker(root=root)

    assert summary["selected_count"] == 3
    assert summary["success_count"] == 1
    assert summary["success_fraction"] == "1/3"
    assert summary["is_16_over_27"] is False
    assert summary["success_rate"] == 0.333333
    assert summary["failure_type_distribution"]["json_parse_failure"] == 1
    assert summary["failure_type_distribution"]["schema_validation_failure"] == 1
    assert summary["failure_type_distribution"]["AR02_numeric_discrepancy"] == 1
    assert summary["top_unstable_documents"][0]["document_id"] in {"MDA_B", "MDA_C"}
    assert (root / "PAPER_OUTPUT" / "00_status" / "mda_stability_blocker_diagnosis.md").exists()
    case_path = root / "PAPER_OUTPUT" / "00_status" / "mda_stability_blocker_cases.csv"
    cases = case_path.read_text(encoding="utf-8")
    assert case_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(CASE_HEADERS)
    assert "json_parse_failure" in cases
    assert "repair_json_only" in cases
    assert "rerun_failed_dimension_only" in cases
    assert "recommended_action" in cases.splitlines()[0]
    assert "Beta Drinks" in cases


def test_run_scoring_accepts_scoring_strategy_argument():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_scoring.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--scoring-strategy" in completed.stdout
