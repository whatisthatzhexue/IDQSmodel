from __future__ import annotations

import json
import subprocess
from pathlib import Path

from build_final_deliverable_clean import build_final_deliverable_clean
from build_human_validation_sample import build_human_validation_sample
from build_mda_final_tables_v2 import build_mda_final_tables_v2
from compute_human_model_agreement import compute_human_model_agreement
from run_mda_semantic_repair_v2 import run_mda_semantic_repair_v2
from scoring_utils import MDA_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, REGISTRY_HEADERS, read_csv_rows, save_json, write_csv_rows


def test_semantic_repair_v2_reports_only_actual_replacements(tmp_path):
    root = tmp_path / "06_scoring"
    _write_minimal_mda_fixture(root, doc_count=2)
    write_csv_rows(
        root / "08_reports" / "mda_semantic_evidence_audit.csv",
        _audit_headers(),
        [
            _audit_row("MDA_DOC_001", "AR02", "invalid", "rerun_dimension_only"),
            _audit_row("MDA_DOC_002", "AR03", "invalid", "check_text_cleaning"),
        ],
    )

    summary = run_mda_semantic_repair_v2(root=root)

    actions = read_csv_rows(root / "08_reports" / "mda_semantic_repair_v2_actions.csv")
    assert summary["flagged_dimension_count"] == 2
    assert summary["repaired_dimension_count"] == 0
    assert summary["manual_required_count"] == 2
    assert {row["repair_status"] for row in actions} == {"manual_required"}
    assert all(row["raw_output_path"] == "" for row in actions)


def test_final_tables_v2_rebuilds_scores_and_review_status_from_v2_audit(tmp_path):
    root = tmp_path / "06_scoring"
    _write_minimal_mda_fixture(root, doc_count=2)
    write_csv_rows(
        root / "08_reports" / "mda_semantic_evidence_audit_v2.csv",
        _audit_headers() + ["repair_status"],
        [
            _audit_row("MDA_DOC_001", "AR01", "weak", "manual_required", repair_status="manual_required"),
            _audit_row("MDA_DOC_001", "AR02", "acceptable", "manual_review", issue="numeric_review_required"),
        ],
    )

    summary = build_mda_final_tables_v2(root=root)

    ratings = read_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_ratings_long.csv")
    docs = read_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_document_scores.csv")
    doc1 = next(row for row in docs if row["document_id"] == "MDA_DOC_001")
    assert summary["ratings_long_count"] == 10
    assert summary["document_scores_count"] == 2
    assert len(ratings) == 10
    assert len(docs) == 2
    assert doc1["needs_review"] == "true"
    assert "semantic_manual_required:AR01" in doc1["review_reasons"]
    assert "numeric_review_required:AR02" in doc1["review_reasons"]
    assert float(doc1["total_score_100"]) == 50.0


def test_human_validation_v2_uses_30_unique_documents_and_agreement_not_ready(tmp_path):
    root = tmp_path / "06_scoring"
    _write_minimal_mda_fixture(root, doc_count=31)
    summary = build_human_validation_sample(root=root, sample_size=30, v2=True)
    agreement = compute_human_model_agreement(root=root, sample_path=Path(summary["csv_path"]))

    sample_rows = read_csv_rows(root / "HUMAN_VALIDATION" / "mda_human_validation_sample_v2.csv")
    assert summary["unique_document_count"] == 30
    assert summary["sample_rows"] == 150
    assert len({row["document_id"] for row in sample_rows}) == 30
    assert len(sample_rows) == 150
    assert agreement["mda_human_validation_ready"] is False
    assert "weighted_kappa_by_dimension" in agreement


def test_clean_deliverable_is_standalone_and_clears_stale_cross_validation(tmp_path):
    root = tmp_path / "06_scoring"
    _write_minimal_mda_fixture(root, doc_count=2)
    save_json(
        root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json",
        {
            "paper_ready": False,
            "freeze_allowed": False,
            "mda_structural_ready": True,
            "mda_semantic_ready": False,
            "mda_human_validation_ready": False,
            "mda_ready": False,
            "news_ready": False,
            "cross_validation_ready": False,
            "cross_validation_empirical_ready": False,
            "reproducibility_ready": True,
            "mda_final_dataset_size": 2,
            "news_final_dataset_size": 0,
            "claim_link_count": 7,
            "eligible_company_year_pair_count": 5,
            "blocking_reasons": ["real_news_exists"],
            "advisory_warnings": [],
            "recommended_next_actions": [],
        },
    )
    write_csv_rows(root / "FINAL_OUTPUT" / "mda_final_dataset.csv", MDA_DOCUMENT_HEADERS, [_mda_doc("MDA_DOC_001"), _mda_doc("MDA_DOC_002")])
    write_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_ratings_long.csv", RATINGS_LONG_HEADERS, _rating_rows("MDA_DOC_001") + _rating_rows("MDA_DOC_002"))
    write_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_doc("MDA_DOC_001"), _mda_doc("MDA_DOC_002")])
    write_csv_rows(root / "09_cross_validation" / "claim_links.csv", ["link_id"], [{"link_id": "OLD"}])
    (root.parent / "requirements.txt").write_text("pytest\n", encoding="utf-8")

    summary = build_final_deliverable_clean(root=root)

    out = Path(summary["output_dir"])
    status = json.loads((out / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    assert (out / "run_reproduce.sh").exists()
    assert (out / "06_scoring" / "scripts" / "run_final_data_freeze.py").exists()
    assert (out / "MDA" / "mda_final_ratings_long.csv").exists()
    assert status["claim_link_count"] == 0
    assert status["eligible_company_year_pair_count"] == 0
    assert status["cross_validation_pipeline_ready"] is True
    assert status["cross_validation_empirical_ready"] is False
    assert not list(out.rglob("__pycache__"))
    assert not list(out.rglob(".pytest_cache"))
    assert "<LOCAL_USER>" not in (out / "MDA" / "mda_final_ratings_long.csv").read_text(encoding="utf-8")

    completed = subprocess.run(["bash", "run_reproduce.sh"], cwd=out, env={"SKIP_VENV": "1"}, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def _write_minimal_mda_fixture(root: Path, doc_count: int) -> None:
    docs = [_mda_doc(f"MDA_DOC_{idx:03d}") for idx in range(1, doc_count + 1)]
    ratings = []
    registry = []
    for idx in range(1, doc_count + 1):
        doc_id = f"MDA_DOC_{idx:03d}"
        ratings.extend(_rating_rows(doc_id))
        registry.append({"document_id": doc_id, "doc_type": "mda", "include_flag": "Yes", "ticker": "AAA", "company_name": "Alpha Foods", "report_year": "2024", "text_path": f"02_extracted_text/mda/{doc_id}.txt"})
        text_path = root / "02_extracted_text" / "mda" / f"{doc_id}.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text("[MDA_P001] Revenue and profit improved because market demand increased. Risk outlook remains manageable. The discussion is structured.\n", encoding="utf-8")
    write_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv", MDA_DOCUMENT_HEADERS, docs)
    write_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv", RATINGS_LONG_HEADERS, ratings)
    write_csv_rows(root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], registry)


def _mda_doc(document_id: str) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update(
        {
            "document_id": document_id,
            "doc_type": "mda",
            "scoring_basis": "mda_disclosure_quality",
            "ticker": "AAA",
            "company_name": "Alpha Foods",
            "report_year": "2024",
            "dimension_count": "5",
            "total_score_100": "50",
            "grade_label": "D",
            "AR01": "3",
            "AR02": "3",
            "AR03": "3",
            "AR04": "3",
            "AR05": "3",
            "AR01_std": "50",
            "AR02_std": "50",
            "AR03_std": "50",
            "AR04_std": "50",
            "AR05_std": "50",
            "needs_review": "false",
            "review_reasons": "",
        }
    )
    return row


def _rating_rows(document_id: str) -> list[dict[str, str]]:
    rows = []
    for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
        row = {field: "" for field in RATINGS_LONG_HEADERS}
        row.update(
            {
                "rating_id": f"{document_id}_{code}",
                "document_id": document_id,
                "doc_type": "mda",
                "dimension_code": code,
                "dimension_name": code,
                "raw_score": "3",
                "scale_min": "1",
                "scale_max": "5",
                "weight_value": {"AR01": "0.15", "AR02": "0.25", "AR03": "0.30", "AR04": "0.20", "AR05": "0.10"}[code],
                "std_score_100": "50",
                "weighted_score": {"AR01": "7.5", "AR02": "12.5", "AR03": "15.0", "AR04": "10.0", "AR05": "5.0"}[code],
                "evidence_locator": "[MDA_P001]",
                "evidence_text_short": "Revenue and profit improved because market demand increased.",
                "confidence_level": "medium",
                "comment_short": "Baseline comment",
                "numeric_check_path": "03_numeric_checks/example.json" if code == "AR02" else "",
                "parse_status": "parsed",
                "schema_validation_status": "valid",
            }
        )
        rows.append(row)
    return rows


def _audit_headers() -> list[str]:
    return [
        "document_id",
        "ticker",
        "company_name",
        "report_year",
        "dimension_code",
        "raw_score",
        "evidence_locator",
        "evidence_exists",
        "evidence_text_short",
        "evidence_semantic_status",
        "evidence_issue_type",
        "score_evidence_alignment",
        "needs_review",
        "recommended_action",
    ]


def _audit_row(document_id: str, dimension_code: str, status: str, action: str, *, issue: str = "missing_financial_content", repair_status: str = "") -> dict[str, str]:
    row = {
        "document_id": document_id,
        "ticker": "AAA",
        "company_name": "Alpha Foods",
        "report_year": "2024",
        "dimension_code": dimension_code,
        "raw_score": "3",
        "evidence_locator": "[MDA_P001]",
        "evidence_exists": "true",
        "evidence_text_short": "Evidence text",
        "evidence_semantic_status": status,
        "evidence_issue_type": issue,
        "score_evidence_alignment": "misaligned" if status == "invalid" else "partial",
        "needs_review": "true" if status in {"weak", "invalid"} or action != "accept" else "false",
        "recommended_action": action,
    }
    if repair_status:
        row["repair_status"] = repair_status
    return row
