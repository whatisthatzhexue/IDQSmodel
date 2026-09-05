from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_scoring import _build_document_score_row, _build_rating_rows, _review_reasons, now_iso
from scoring_utils import (
    FLAG_FIELDS,
    RATINGS_LONG_HEADERS,
    REVIEW_LOG_HEADERS,
    SCOREBOOK_VERSION,
    SCORING_ROOT,
    dimensions_for,
    document_headers_for,
    read_csv_rows,
    resolve_text_path,
    save_json,
    scoring_basis_for,
    write_csv_rows,
)
from validate_model_scores import normalize_scoring_payload, validate_scoring_payload


FAILED_CASE_HEADERS = [
    "document_id",
    "status",
    "error_type",
    "error_message",
    "parse_status",
    "schema_validation_status",
    "dimension_count",
    "failed_dimensions",
    "raw_output_path",
]


def aggregate_dimensionwise_mda_scores(
    root: Path = SCORING_ROOT,
    output_name: str = "mda_v2_dimensionwise",
    ratings_prefix: str = "mda_v2_dimensionwise",
    doc_ids: list[str] | None = None,
    model_name: str = "qwen3:8b",
) -> dict[str, Any]:
    out_dir = root / "06_ratings" / output_name
    per_document_dir = out_dir / "per_document"
    per_document_dir.mkdir(parents=True, exist_ok=True)
    registry = {row.get("document_id", ""): row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv")}
    selected_ids = doc_ids or _doc_ids_from_per_dimension(out_dir)
    rating_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    expected_codes = [item["code"] for item in dimensions_for("mda")]

    for doc_id in selected_ids:
        row = registry.get(doc_id, {"document_id": doc_id, "doc_type": "mda"})
        records = _dimension_records(out_dir, doc_id)
        success_records = {record.get("dimension_code"): record for record in records if record.get("status") == "success"}
        failed_codes = [code for code in expected_codes if code not in success_records]
        raw_paths = {
            code: str(success_records.get(code, {}).get("result", {}).get("raw_output_path", ""))
            for code in expected_codes
        }
        if failed_codes:
            failed = _failed_case(doc_id, records, failed_codes, "missing_dimension_scores")
            failed_rows.append(failed)
            save_json(per_document_dir / f"{doc_id}_parsed.json", _failed_per_document_record(doc_id, failed))
            continue
        dimension_scores = [success_records[code]["payload"] for code in expected_codes]
        payload = normalize_scoring_payload(
            {
                "document_id": doc_id,
                "doc_type": "mda",
                "scoring_basis": scoring_basis_for("mda"),
                "scorebook_version": SCOREBOOK_VERSION,
                "evidence_limitation": "Dimension-wise local qwen3:8b scoring; Python aggregated five AR dimensions.",
                "dimension_scores": dimension_scores,
                "flags": {field: 0 for field in FLAG_FIELDS},
                "overall_comment": "Aggregated from five dimension-wise MD&A scoring calls.",
                "needs_review": _needs_review(dimension_scores),
                "review_reasons": _payload_review_reasons(dimension_scores),
            },
            "mda",
            doc_id,
        )
        text_path = resolve_text_path(row, "mda", root)
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
        validation = validate_scoring_payload(payload, text, "mda")
        if not validation.is_valid:
            failed = _failed_case(doc_id, records, [], "schema_validation_failed", ";".join(validation.issues))
            failed_rows.append(failed)
            save_json(per_document_dir / f"{doc_id}_parsed.json", _failed_per_document_record(doc_id, failed))
            continue
        started_at = min(str(record.get("result", {}).get("started_at", now_iso())) for record in records)
        finished_at = max(str(record.get("result", {}).get("finished_at", now_iso())) for record in records)
        record_model = str(records[0].get("result", {}).get("model_name") or model_name)
        numeric_check_path = _numeric_check_path(records)
        joined_raw_paths = ";".join(path for path in raw_paths.values() if path)
        doc_score_row = _build_document_score_row(payload, row, "mda", record_model)
        rows = _build_rating_rows(
            payload,
            "mda",
            started_at,
            finished_at,
            record_model,
            joined_raw_paths,
            numeric_check_path,
            "parsed",
            "valid",
        )
        for rating_row in rows:
            rating_row["raw_output_path"] = raw_paths.get(rating_row["dimension_code"], joined_raw_paths)
        reviews = _review_reasons(
            payload,
            text,
            row,
            "mda",
            float(doc_score_row["total_score_100"]),
            _numeric_payload(numeric_check_path),
            numeric_check_path,
            joined_raw_paths,
        )
        result = {
            "document_id": doc_id,
            "doc_type": "mda",
            "success": True,
            "raw_output_path": joined_raw_paths,
            "parsed_output_path": str(per_document_dir / f"{doc_id}_parsed.json"),
            "parse_status": "parsed",
            "schema_validation_status": "valid",
            "error_type": "",
            "error_message": "",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": _duration_seconds(records),
            "attempts_used": sum(int(record.get("result", {}).get("attempts_used") or 0) for record in records),
            "needs_review": bool(payload.get("needs_review")) or bool(reviews),
            "review_reasons": payload.get("review_reasons", []),
            "status": "success",
            "worker_id": "dimensionwise-aggregator",
        }
        save_json(
            per_document_dir / f"{doc_id}_parsed.json",
            {
                "document_id": doc_id,
                "doc_type": "mda",
                "status": "success",
                "payload": payload,
                "metadata": row,
                "result": result,
                "ratings_rows": rows,
                "document_score_row": doc_score_row,
                "review_rows": reviews,
            },
        )
        rating_rows.extend(rows)
        document_rows.append(doc_score_row)
        review_rows.extend(reviews)

    write_csv_rows(out_dir / f"{ratings_prefix}_ratings_long.csv", RATINGS_LONG_HEADERS, rating_rows)
    write_csv_rows(out_dir / f"{ratings_prefix}_document_scores.csv", document_headers_for("mda"), document_rows)
    write_csv_rows(out_dir / f"{ratings_prefix}_review_log.csv", REVIEW_LOG_HEADERS, review_rows)
    write_csv_rows(out_dir / f"{ratings_prefix}_failed_cases.csv", FAILED_CASE_HEADERS, failed_rows)
    summary = {
        "doc_type": "mda",
        "scoring_strategy": "dimensionwise",
        "output_name": output_name,
        "ratings_prefix": ratings_prefix,
        "selected_count": len(selected_ids),
        "success_count": len(document_rows),
        "failure_count": len(failed_rows),
        "dimension_count_expected": 5,
        "document_scores_path": str(out_dir / f"{ratings_prefix}_document_scores.csv"),
        "ratings_long_path": str(out_dir / f"{ratings_prefix}_ratings_long.csv"),
        "failed_cases_path": str(out_dir / f"{ratings_prefix}_failed_cases.csv"),
    }
    save_json(out_dir / f"{ratings_prefix}_aggregation_summary.json", summary)
    return summary


def _doc_ids_from_per_dimension(out_dir: Path) -> list[str]:
    doc_ids: list[str] = []
    for path in sorted((out_dir / "per_dimension").glob("*_parsed.json")):
        suffix = path.name.split("_")[-2:]
        if len(suffix) < 2:
            continue
        code = suffix[0]
        doc_id = path.name.removesuffix(f"_{code}_parsed.json")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def _dimension_records(out_dir: Path, doc_id: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((out_dir / "per_dimension").glob(f"{doc_id}_AR*_parsed.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            records.append({"document_id": doc_id, "status": "failed", "dimension_code": "", "result": {"error_type": "json_decode_error"}})
    return records


def _failed_case(
    doc_id: str,
    records: list[dict[str, Any]],
    failed_codes: list[str],
    error_type: str,
    error_message: str = "",
) -> dict[str, Any]:
    result_by_code = {record.get("dimension_code"): record.get("result", {}) for record in records}
    parse_status = "parsed" if records and all(result.get("parse_status") == "parsed" for result in result_by_code.values()) else "parse_failed"
    schema_status = "valid" if records and all(result.get("schema_validation_status") == "valid" for result in result_by_code.values()) else "invalid"
    raw_paths = [str(result.get("raw_output_path", "")) for result in result_by_code.values() if result.get("raw_output_path")]
    return {
        "document_id": doc_id,
        "status": "failed",
        "error_type": error_type,
        "error_message": error_message,
        "parse_status": parse_status,
        "schema_validation_status": schema_status,
        "dimension_count": len([record for record in records if record.get("status") == "success"]),
        "failed_dimensions": ";".join(failed_codes),
        "raw_output_path": ";".join(raw_paths),
    }


def _failed_per_document_record(doc_id: str, failed: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": doc_id,
        "doc_type": "mda",
        "status": "failed",
        "result": {
            "error_type": failed["error_type"],
            "error_message": failed["error_message"],
            "parse_status": failed["parse_status"],
            "schema_validation_status": failed["schema_validation_status"],
            "raw_output_path": failed["raw_output_path"],
            "review_reasons": [failed["error_type"]],
        },
    }


def _needs_review(dimension_scores: list[dict[str, Any]]) -> bool:
    return any(item.get("confidence_level") == "low" or item.get("missing_type") != "none" for item in dimension_scores)


def _payload_review_reasons(dimension_scores: list[dict[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if any(item.get("confidence_level") == "low" for item in dimension_scores):
        reasons.append("dimension_low_confidence")
    if any(item.get("missing_type") != "none" for item in dimension_scores):
        reasons.append("dimension_missing_type_not_none")
    return reasons


def _numeric_check_path(records: list[dict[str, Any]]) -> str:
    for record in records:
        path = record.get("numeric_check_path") or record.get("result", {}).get("numeric_check_path")
        if path:
            return str(path)
    return ""


def _numeric_payload(path_text: str) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _duration_seconds(records: list[dict[str, Any]]) -> float:
    return round(sum(float(record.get("result", {}).get("duration_seconds") or 0) for record in records), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate five dimension-wise MD&A JSON scores into document-level CSVs.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--output-name", default="mda_v2_dimensionwise")
    parser.add_argument("--ratings-prefix", default="mda_v2_dimensionwise")
    args = parser.parse_args()
    print(json.dumps(aggregate_dimensionwise_mda_scores(args.root, args.output_name, args.ratings_prefix), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
