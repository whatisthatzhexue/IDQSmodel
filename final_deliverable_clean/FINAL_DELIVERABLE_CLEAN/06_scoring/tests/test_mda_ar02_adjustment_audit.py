from __future__ import annotations

import json
from pathlib import Path

from audit_mda_ar02_adjustments import audit_mda_ar02_adjustments
from scoring_utils import RATINGS_LONG_HEADERS, read_csv_rows, write_csv_rows


def test_ar02_adjustment_audit_keeps_original_and_adjusted_scores(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    raw_path = root / "05_raw_model_outputs" / "mda_final_v2" / "MDA_A_AR02_raw.json"
    numeric_path = root / "03_numeric_checks" / "mda" / "MDA_A_AR02.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    numeric_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "document_id": "MDA_A",
                "dimension_code": "AR02",
                "score": 5,
                "comment_short": "Model gave a high traceability score.",
            }
        ),
        encoding="utf-8",
    )
    numeric_path.write_text(
        json.dumps(
            {
                "result": "adjusted_down",
                "rule": "numeric_checker_postprocess",
                "reason": "reported arithmetic issue in the evidence span",
            }
        ),
        encoding="utf-8",
    )
    row = {field: "" for field in RATINGS_LONG_HEADERS}
    row.update(
        {
            "rating_id": "MDA_A_AR02",
            "document_id": "MDA_A",
            "doc_type": "mda",
            "dimension_code": "AR02",
            "raw_score": "4",
            "raw_output_path": str(raw_path.relative_to(root)),
            "numeric_check_path": str(numeric_path.relative_to(root)),
            "parse_status": "parsed",
            "schema_validation_status": "valid",
        }
    )
    write_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_ratings_long.csv", RATINGS_LONG_HEADERS, [row])

    summary = audit_mda_ar02_adjustments(root)

    assert summary["ar02_rows"] == 1
    assert summary["adjustment_count"] == 1
    enriched = read_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_ratings_long.csv")
    assert enriched[0]["original_model_raw_score"] == "5"
    assert enriched[0]["adjusted_raw_score"] == "4"
    assert enriched[0]["adjustment_applied"] == "true"
    assert enriched[0]["adjustment_rule"] == "numeric_checker_postprocess"
    assert enriched[0]["adjustment_reason"] == "reported arithmetic issue in the evidence span"
    assert "adjusted_down" in enriched[0]["numeric_checker_result"]

    audit_rows = read_csv_rows(root / "08_reports" / "mda_ar02_adjustment_audit.csv")
    assert len(audit_rows) == 1
    assert audit_rows[0]["raw_output_path"] == str(raw_path.relative_to(root))
