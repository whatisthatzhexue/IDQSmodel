from __future__ import annotations

from audit_mda_semantic_evidence import audit_mda_semantic_evidence, classify_evidence
from scoring_utils import RATINGS_LONG_HEADERS, write_csv_rows


def test_classify_evidence_flags_dimension_specific_issues():
    ar02 = classify_evidence("AR02", "The board thanked shareholders for their support.", "reason")
    ar03 = classify_evidence("AR03", "The business environment was challenging.", "business was challenging")
    ar04 = classify_evidence("AR04", "The group faces raw material price risk and uncertain outlook.", "risk outlook")

    assert ar02["evidence_semantic_status"] == "invalid"
    assert ar02["evidence_issue_type"] == "missing_financial_content"
    assert ar03["evidence_semantic_status"] == "weak"
    assert ar03["evidence_issue_type"] == "missing_causal_explanation"
    assert ar04["evidence_semantic_status"] in {"strong", "acceptable"}
    assert ar04["evidence_issue_type"] == "none"


def test_audit_mda_semantic_evidence_writes_review_pool(tmp_path):
    root = tmp_path / "06_scoring"
    text_dir = root / "02_extracted_text" / "mda_clean_v2"
    text_dir.mkdir(parents=True)
    (text_dir / "MDA_A.txt").write_text(
        "[MDA_P001]\nRevenue increased by 10% due to higher demand.\n\n"
        "[MDA_P002]\nThe board thanked shareholders for their support.\n",
        encoding="utf-8",
    )
    write_csv_rows(
        root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv",
        ["document_id", "ticker", "company_name", "report_year", "total_score_100"],
        [{"document_id": "MDA_A", "ticker": "AAA", "company_name": "Alpha", "report_year": "2024", "total_score_100": "80"}],
    )
    rows = []
    for code, locator in [("AR02", "[MDA_P002]"), ("AR03", "[MDA_P001]")]:
        row = {field: "" for field in RATINGS_LONG_HEADERS}
        row.update(
            {
                "document_id": "MDA_A",
                "dimension_code": code,
                "raw_score": "4",
                "evidence_locator": locator,
                "evidence_text_short": "The board thanked shareholders" if code == "AR02" else "Revenue increased",
                "comment_short": "short reason",
            }
        )
        rows.append(row)
    write_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv", RATINGS_LONG_HEADERS, rows)

    summary = audit_mda_semantic_evidence(root=root)

    assert summary["total_dimension_rows"] == 2
    assert summary["invalid_count"] == 1
    assert (root / "08_reports" / "mda_semantic_evidence_audit.csv").exists()
    review_pool = (root / "07_review" / "mda_semantic_evidence_review_pool.csv").read_text(encoding="utf-8")
    assert "MDA_A" in review_pool
    assert "missing_financial_content" in review_pool
