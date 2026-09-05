import csv
import json
from pathlib import Path

from audit_ar02_numeric_discrepancies import audit_ar02_numeric_discrepancies
from diagnose_mda_v2_sample_failures import diagnose_mda_v2_sample_failures
from generate_news_missing_next_steps import generate_news_missing_next_steps
from mda_numeric_checker import analyze_text
from run_mda_v2_full_rescore import run_mda_v2_full_rescore
from run_mda_v2_stability_batch import should_recommend_full_rescore, summarize_stability_results
from scoring_utils import MDA_DOCUMENT_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows


def _write_minimal_contracts(root: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for rel in ["04_prompts/scoring_output_schema.json", "04_prompts/mda_scoring_prompt.txt"]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((source_root / rel).read_text(encoding="utf-8"), encoding="utf-8")


def _write_mda_rows(root: Path, doc_ids: list[str]) -> None:
    rows = []
    text_dir = root / "02_extracted_text" / "mda_clean_v2"
    text_dir.mkdir(parents=True, exist_ok=True)
    for idx, doc_id in enumerate(doc_ids, start=1):
        text = (
            "[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.\n\n"
            "[MDA_P002]\nManagement discussed raw material costs, risk, and cautious outlook."
        )
        (text_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        rows.append(
            {
                "document_id": doc_id,
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": f"{idx:04d}",
                "ticker": "TST",
                "company_name": "Test Food Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": f"02_extracted_text/mda_clean_v2/{doc_id}.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        )
    write_csv_rows(root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], rows)


def test_failure_diagnosis_classifies_mock_failed_cases(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_rows(root, ["MDA_FAIL_JSON", "MDA_FAIL_SCHEMA"])
    sample_dir = root / "06_ratings" / "mda_v2_sample_rescore"
    parsed_dir = sample_dir / "per_document"
    parsed_dir.mkdir(parents=True)
    for doc_id, parse_status, schema_status in [
        ("MDA_FAIL_JSON", "parse_failed", "invalid"),
        ("MDA_FAIL_SCHEMA", "parsed", "invalid"),
    ]:
        (parsed_dir / f"{doc_id}_parsed.json").write_text(
            json.dumps(
                {
                    "document_id": doc_id,
                    "doc_type": "mda",
                    "status": "failed",
                    "result": {
                        "error_type": "scoring_failed",
                        "error_message": "json_parse_failed" if parse_status == "parse_failed" else "schema validation failed",
                        "parse_status": parse_status,
                        "schema_validation_status": schema_status,
                        "attempts_used": 3,
                    },
                }
            ),
            encoding="utf-8",
        )
    write_csv_rows(
        root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv",
        ["document_id", "num_material_discrepancies"],
        [{"document_id": "MDA_FAIL_JSON", "num_material_discrepancies": "2"}],
    )

    summary = diagnose_mda_v2_sample_failures(root)

    rows = read_csv_rows(sample_dir / "mda_v2_failure_cases.csv")
    by_doc = {row["document_id"]: row for row in rows}
    assert summary["failure_count"] == 2
    assert by_doc["MDA_FAIL_JSON"]["failure_stage"] == "json_parse"
    assert by_doc["MDA_FAIL_JSON"]["recommended_action"] == "repair_json_and_retry"
    assert by_doc["MDA_FAIL_SCHEMA"]["failure_stage"] == "schema_validation"
    assert by_doc["MDA_FAIL_SCHEMA"]["recommended_action"] == "fix_prompt_and_retry"


def test_numeric_discrepancy_audit_labels_real_and_parser_false_positive(tmp_path):
    root = tmp_path / "06_scoring"
    out_dir = root / "03_numeric_checks" / "mda_v2"
    out_dir.mkdir(parents=True)
    payload = {
        "document_id": "MDA_NUMERIC",
        "checks": [
            {
                "metric": "revenue",
                "current_value": 120,
                "previous_value": 100,
                "stated_pct": 50,
                "calculated_change_pct": 20,
                "status": "material_discrepancy",
                "source_text": "[MDA_P001] Revenue increased by 50% from RM100 million to RM120 million.",
                "evidence_locator": "[MDA_P001]",
            },
            {
                "metric": "segment revenue",
                "current_value": 4054.6,
                "previous_value": 38.46,
                "stated_pct": 0,
                "calculated_change_pct": 4054.6,
                "status": "material_discrepancy",
                "source_text": "[MDA_P002] Segment table shows multiple years and a total row.",
                "evidence_locator": "[MDA_P002]",
                "note": "segment revenue: components sum 4054.60, stated total 38.46",
            },
        ],
    }
    (out_dir / "MDA_NUMERIC_numeric_checks.json").write_text(json.dumps(payload), encoding="utf-8")
    write_csv_rows(
        root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv",
        ["document_id", "num_material_discrepancies"],
        [{"document_id": "MDA_NUMERIC", "num_material_discrepancies": "2"}],
    )

    summary = audit_ar02_numeric_discrepancies(root, sample_size=30)

    rows = read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_discrepancy_audit.csv")
    labels = {row["metric_name"]: row["audit_label"] for row in rows}
    assert summary["sample_count"] == 2
    assert labels["revenue"] == "real_discrepancy"
    assert labels["segment revenue"] == "parser_false_positive"


def test_numeric_checker_handles_units_negative_base_and_percentage_points():
    text = (
        "[MDA_P001]\nRevenue increased by 20% from RM'000 100,000 to RM 120 million.\n\n"
        "[MDA_P002]\nMargin increased by 2 percentage points from 8% to 10%.\n\n"
        "[MDA_P003]\nProfit improved from (RM10 million) to RM5 million."
    )
    result = analyze_text(text, "MDA_UNIT")
    statuses = {check["status"] for check in result["checks"]}
    assert "consistent" in statuses
    assert "special_case_negative_base" in statuses
    assert result["num_material_discrepancies"] == 0


def test_full_rescore_writes_independent_outputs_and_final_candidate_without_overwriting_v1(tmp_path):
    root = tmp_path / "06_scoring"
    _write_minimal_contracts(root)
    _write_mda_rows(root, ["MDA_FULL_2024"])
    write_csv_rows(
        root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [{"document_id": "MDA_V1_ONLY", "doc_type": "mda", "total_score_100": "50"}],
    )
    before = (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8")

    summary = run_mda_v2_full_rescore(root=root, mock=True, llm_workers=1, cpu_workers=2)

    assert summary["success_count"] == 1
    assert (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8") == before
    full_scores = root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv"
    failed_cases = root / "06_ratings" / "mda_v2_full" / "mda_v2_full_failed_cases.csv"
    candidate = root / "06_ratings" / "mda_document_scores_final_candidate.csv"
    assert full_scores.exists()
    assert failed_cases.exists()
    assert candidate.exists()
    candidate_rows = read_csv_rows(candidate)
    assert candidate_rows[0]["score_version"] == "v2_full"
    assert candidate_rows[0]["text_version"] == "mda_clean_v2"


def test_stability_gate_requires_success_rate_and_zero_scope_violations():
    assert should_recommend_full_rescore({"success_rate": 0.85, "scope_violation_count": 0}) is True
    assert should_recommend_full_rescore({"success_rate": 0.84, "scope_violation_count": 0}) is False
    assert should_recommend_full_rescore({"success_rate": 0.95, "scope_violation_count": 1}) is False
    report = summarize_stability_results(
        [
            {"status": "success", "duration_seconds": 10, "result": {}},
            {"status": "failed", "duration_seconds": 20, "result": {"parse_status": "parse_failed"}},
        ]
    )
    assert report["success_rate"] == 0.5
    assert report["recommendation_for_full_rescore"] == "continue_fixing_failures_do_not_full_rescore"


def test_news_missing_report_and_empty_cross_validation_claims(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(root / "01_registry" / "news_registry.csv", ["document_id", "doc_type", "include_flag"], [])

    path = generate_news_missing_next_steps(root)

    text = path.read_text(encoding="utf-8")
    assert "news_registry.csv is empty" in text
    assert "run_cross_validation.py --mode full" in text
