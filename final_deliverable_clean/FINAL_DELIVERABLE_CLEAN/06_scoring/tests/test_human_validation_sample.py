from __future__ import annotations

from pathlib import Path

from build_human_validation_sample import build_human_validation_sample
from scoring_utils import MDA_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, read_csv_rows, write_csv_rows


def test_human_validation_sample_writes_xlsx_and_instructions(tmp_path):
    root = tmp_path / "06_scoring"
    ratings = []
    docs = []
    audit_rows = []
    for idx in range(1, 9):
        document_id = f"MDA_DOC_{idx:03d}"
        docs.append(_mda_doc(document_id, "AAA", "Alpha Foods", "2024", 40 + idx * 5))
        for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
            ratings.append(_rating(document_id, code, str(1 + idx % 5), numeric=code == "AR02" and idx % 2 == 0))
            audit_rows.append(
                {
                    "document_id": document_id,
                    "ticker": "AAA",
                    "company_name": "Alpha Foods",
                    "report_year": "2024",
                    "dimension_code": code,
                    "raw_score": str(1 + idx % 5),
                    "evidence_locator": f"[MDA_P{idx:03d}]",
                    "evidence_exists": "true",
                    "evidence_text_short": f"Evidence {idx} {code}",
                    "evidence_semantic_status": "invalid" if idx == 1 and code == "AR02" else "strong",
                    "evidence_issue_type": "missing_financial_content" if idx == 1 and code == "AR02" else "none",
                    "score_evidence_alignment": "misaligned" if idx == 1 and code == "AR02" else "aligned",
                    "needs_review": "true" if idx == 1 and code == "AR02" else "false",
                    "recommended_action": "rerun_dimension_only" if idx == 1 and code == "AR02" else "accept",
                }
            )
    write_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv", MDA_DOCUMENT_HEADERS, docs)
    write_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv", RATINGS_LONG_HEADERS, ratings)
    write_csv_rows(
        root / "08_reports" / "mda_semantic_evidence_audit.csv",
        list(audit_rows[0].keys()),
        audit_rows,
    )

    summary = build_human_validation_sample(root=root, sample_size=30)

    out_dir = root / "HUMAN_VALIDATION"
    assert summary["sample_rows"] == 30
    assert (out_dir / "mda_human_validation_sample.xlsx").exists()
    assert (out_dir / "mda_human_validation_instructions.md").exists()
    assert len(read_csv_rows(out_dir / "mda_human_validation_sample.csv")) == 30
    text = (out_dir / "mda_human_validation_instructions.md").read_text(encoding="utf-8")
    assert "MD&A-only" in text
    assert "Do not edit the model score columns" in text


def _mda_doc(document_id: str, ticker: str, company: str, year: str, total: int) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update(
        {
            "document_id": document_id,
            "doc_type": "mda",
            "ticker": ticker,
            "company_name": company,
            "report_year": year,
            "dimension_count": "5",
            "total_score_100": str(total),
            "grade_label": "B",
            "AR01": "3",
            "AR02": "3",
            "AR03": "3",
            "AR04": "3",
            "AR05": "3",
        }
    )
    return row


def _rating(document_id: str, code: str, raw_score: str, *, numeric: bool = False) -> dict[str, str]:
    row = {field: "" for field in RATINGS_LONG_HEADERS}
    row.update(
        {
            "rating_id": f"{document_id}_{code}",
            "document_id": document_id,
            "doc_type": "mda",
            "dimension_code": code,
            "dimension_name": code,
            "raw_score": raw_score,
            "evidence_locator": "[MDA_P001]",
            "evidence_text_short": f"Evidence for {document_id} {code}",
            "confidence_level": "medium",
            "comment_short": "Comment",
            "numeric_check_path": "03_numeric_checks/example.json" if numeric else "",
            "parse_status": "parsed",
            "schema_validation_status": "valid",
        }
    )
    return row
