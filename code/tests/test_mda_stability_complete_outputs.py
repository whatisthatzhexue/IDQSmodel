from pathlib import Path

from build_mda_stability_complete_outputs import build_mda_stability_complete_outputs
from scoring_utils import MDA_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, read_csv_rows, write_csv_rows


def _rating(doc_id: str, code: str) -> dict:
    return {
        "rating_id": f"{doc_id}_{code}",
        "document_id": doc_id,
        "doc_type": "mda",
        "dimension_code": code,
        "dimension_name": code,
        "raw_score": "3",
        "parse_status": "parsed",
        "schema_validation_status": "valid",
    }


def _document(doc_id: str) -> dict:
    return {
        "document_id": doc_id,
        "doc_type": "mda",
        "dimension_count": "5",
        "total_score_100": "50",
        "AR01": "3",
        "AR02": "3",
        "AR03": "3",
        "AR04": "3",
        "AR05": "3",
    }


def test_build_complete_stability_outputs_prefers_repair_rows_and_keeps_empty_failed_header(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv",
        ["document_id"],
        [{"document_id": "MDA_A"}, {"document_id": "MDA_B"}],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_ratings_long.csv",
        RATINGS_LONG_HEADERS,
        [_rating("MDA_A", code) for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [_document("MDA_A")],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_context_repair" / "mda_context_repair_ratings_long.csv",
        RATINGS_LONG_HEADERS,
        [_rating("MDA_B", code) for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_context_repair" / "mda_context_repair_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [_document("MDA_B")],
    )

    summary = build_mda_stability_complete_outputs(root)

    out_dir = root / "06_ratings" / "mda_context_repair"
    ratings = read_csv_rows(out_dir / "mda_stability_ratings_long_complete.csv")
    docs = read_csv_rows(out_dir / "mda_stability_document_scores_complete.csv")
    failed_path = out_dir / "mda_stability_failed_cases_complete.csv"
    assert summary["ratings_long_rows"] == 10
    assert summary["document_scores_rows"] == 2
    assert summary["failed_cases_rows"] == 0
    assert len(ratings) == 10
    assert len(docs) == 2
    assert failed_path.read_text(encoding="utf-8").strip().startswith("document_id")
