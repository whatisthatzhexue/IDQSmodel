from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scoring_utils import (
    MDA_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    SCORING_ROOT,
    dimensions_for,
    read_csv_rows,
    save_json,
    write_csv_rows,
)


FAILED_CASE_HEADERS = [
    "document_id",
    "doc_type",
    "status",
    "error_type",
    "error_message",
    "parse_status",
    "schema_validation_status",
    "dimension_count",
    "failed_dimensions",
    "raw_output_path",
]

COMPLETE_RATINGS_FILENAME = "mda_stability_ratings_long_complete.csv"
COMPLETE_DOCUMENT_SCORES_FILENAME = "mda_stability_document_scores_complete.csv"
COMPLETE_FAILED_CASES_FILENAME = "mda_stability_failed_cases_complete.csv"
COMPLETE_SUMMARY_FILENAME = "mda_stability_complete_summary.json"


def _read_headers(path: Path, fallback: list[str]) -> list[str]:
    if not path.exists():
        return list(fallback)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        try:
            headers = next(reader)
        except StopIteration:
            return list(fallback)
    return headers or list(fallback)


def _index_first_by_document(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        doc_id = row.get("document_id", "").strip()
        if doc_id and doc_id not in indexed:
            indexed[doc_id] = row
    return indexed


def _group_ratings(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        doc_id = row.get("document_id", "").strip()
        if doc_id:
            grouped.setdefault(doc_id, []).append(row)
    return grouped


def _complete_dimension_rows(
    rows: list[dict[str, str]],
    dimension_codes: list[str],
) -> list[dict[str, str]] | None:
    rows_by_code: dict[str, dict[str, str]] = {}
    for row in rows:
        code = row.get("dimension_code", "").strip()
        if code in dimension_codes and code not in rows_by_code:
            rows_by_code[code] = row
    if set(rows_by_code) != set(dimension_codes):
        return None
    return [rows_by_code[code] for code in dimension_codes]


def _failed_row(
    doc_id: str,
    dimension_codes: list[str],
    rating_rows: list[dict[str, str]],
    has_document_score: bool,
) -> dict[str, str]:
    present = {row.get("dimension_code", "").strip() for row in rating_rows}
    missing_dimensions = [code for code in dimension_codes if code not in present]
    reasons: list[str] = []
    if missing_dimensions:
        reasons.append(f"missing dimensions: {', '.join(missing_dimensions)}")
    if not has_document_score:
        reasons.append("missing document score row")
    return {
        "document_id": doc_id,
        "doc_type": "mda",
        "status": "failed",
        "error_type": "incomplete_stability_record",
        "error_message": "; ".join(reasons) or "incomplete stability record",
        "parse_status": "",
        "schema_validation_status": "",
        "dimension_count": str(len(present)),
        "failed_dimensions": "|".join(missing_dimensions),
        "raw_output_path": "",
    }


def build_mda_stability_complete_outputs(root: Path = SCORING_ROOT) -> dict[str, Any]:
    root = Path(root)
    batch_dir = root / "06_ratings" / "mda_v2_stability_batch"
    repair_dir = root / "06_ratings" / "mda_context_repair"
    selection_path = batch_dir / "mda_v2_stability_batch_selection.csv"

    selected_document_ids = [
        row["document_id"].strip()
        for row in read_csv_rows(selection_path)
        if row.get("document_id", "").strip()
    ]
    if not selected_document_ids:
        raise ValueError(f"No selected document ids found in {selection_path}")

    dimension_codes = [item["code"] for item in dimensions_for("mda")]

    batch_ratings = _group_ratings(read_csv_rows(batch_dir / "mda_v2_stability_ratings_long.csv"))
    repair_ratings = _group_ratings(read_csv_rows(repair_dir / "mda_context_repair_ratings_long.csv"))
    batch_documents = _index_first_by_document(
        read_csv_rows(batch_dir / "mda_v2_stability_document_scores.csv")
    )
    repair_documents = _index_first_by_document(
        read_csv_rows(repair_dir / "mda_context_repair_document_scores.csv")
    )

    complete_ratings: list[dict[str, str]] = []
    complete_documents: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    document_sources: dict[str, str] = {}
    source_counts = {"mda_context_repair": 0, "mda_v2_stability_batch": 0, "failed": 0}

    for doc_id in selected_document_ids:
        repair_complete_rows = _complete_dimension_rows(
            repair_ratings.get(doc_id, []),
            dimension_codes,
        )
        batch_complete_rows = _complete_dimension_rows(
            batch_ratings.get(doc_id, []),
            dimension_codes,
        )

        if repair_complete_rows is not None and doc_id in repair_documents:
            complete_ratings.extend(repair_complete_rows)
            complete_documents.append(repair_documents[doc_id])
            document_sources[doc_id] = "mda_context_repair"
            source_counts["mda_context_repair"] += 1
            continue

        if batch_complete_rows is not None and doc_id in batch_documents:
            complete_ratings.extend(batch_complete_rows)
            complete_documents.append(batch_documents[doc_id])
            document_sources[doc_id] = "mda_v2_stability_batch"
            source_counts["mda_v2_stability_batch"] += 1
            continue

        available_rows = repair_ratings.get(doc_id, []) or batch_ratings.get(doc_id, [])
        has_document_score = doc_id in repair_documents or doc_id in batch_documents
        failed_rows.append(_failed_row(doc_id, dimension_codes, available_rows, has_document_score))
        document_sources[doc_id] = "failed"
        source_counts["failed"] += 1

    failed_headers = _read_headers(
        repair_dir / "mda_context_repair_failed_cases.csv",
        FAILED_CASE_HEADERS,
    )
    ratings_path = repair_dir / COMPLETE_RATINGS_FILENAME
    documents_path = repair_dir / COMPLETE_DOCUMENT_SCORES_FILENAME
    failed_path = repair_dir / COMPLETE_FAILED_CASES_FILENAME
    summary_path = repair_dir / COMPLETE_SUMMARY_FILENAME

    write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, complete_ratings)
    write_csv_rows(documents_path, MDA_DOCUMENT_HEADERS, complete_documents)
    write_csv_rows(failed_path, failed_headers, failed_rows)

    summary: dict[str, Any] = {
        "status": "complete" if not failed_rows else "incomplete",
        "expected_documents": len(selected_document_ids),
        "expected_rating_rows": len(selected_document_ids) * len(dimension_codes),
        "ratings_long_rows": len(complete_ratings),
        "document_scores_rows": len(complete_documents),
        "failed_cases_rows": len(failed_rows),
        "complete_documents": len(complete_documents),
        "dimension_codes": dimension_codes,
        "source_counts": source_counts,
        "document_sources": document_sources,
        "outputs": {
            "ratings_long": str(ratings_path),
            "document_scores": str(documents_path),
            "failed_cases": str(failed_path),
            "summary": str(summary_path),
        },
        "notes": [
            "This step only combines existing stability-batch and context-repair rows.",
            "It does not rescore documents, recalculate document totals, or modify dimensions, weights, freeze gates, or JSON repair settings.",
        ],
    }
    save_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build complete 27-document MD&A stability output tables from existing batch and repair rows."
    )
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    summary = build_mda_stability_complete_outputs(args.root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
