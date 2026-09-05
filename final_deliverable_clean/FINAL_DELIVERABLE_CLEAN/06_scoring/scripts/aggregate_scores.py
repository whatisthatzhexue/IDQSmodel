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
    load_json,
    read_csv_rows,
    registry_path,
    resolve_text_path,
    save_json,
    scoring_basis_for,
    write_csv_rows,
)
from validate_model_scores import normalize_scoring_payload, validate_scoring_payload


SINGLE_DIMENSION_FAILED_HEADERS = [
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


def aggregate_scores(doc_type: str, root: Path = SCORING_ROOT) -> dict[str, Any]:
    per_doc_dir = root / "06_ratings" / "per_document" / doc_type
    rating_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(per_doc_dir.glob("*_parsed.json")):
        record = load_json(path)
        if record.get("status") != "success":
            continue
        result = record.get("result", {})
        if result.get("schema_validation_status") != "valid":
            continue
        document_id = record.get("document_id")
        if document_id in seen:
            continue
        seen.add(document_id)
        rating_rows.extend(record.get("ratings_rows", []))
        document_rows.append(record.get("document_score_row", {}))
    write_csv_rows(root / "06_ratings" / f"{doc_type}_ratings_long.csv", RATINGS_LONG_HEADERS, rating_rows)
    write_csv_rows(root / "06_ratings" / f"{doc_type}_document_scores.csv", document_headers_for(doc_type), document_rows)
    return {"doc_type": doc_type, "rating_count": len(rating_rows), "document_count": len(document_rows)}


def aggregate_single_dimension_scores(
    doc_type: str,
    root: Path = SCORING_ROOT,
    output_name: str = "single_dimension",
    ratings_prefix: str = "single_dimension",
    doc_ids: list[str] | None = None,
    model_name: str = "qwen3:8b",
) -> dict[str, Any]:
    if doc_type not in {"mda", "news"}:
        raise ValueError("doc_type must be mda or news")
    out_dir = root / "06_ratings" / output_name
    per_document_dir = out_dir / "per_document"
    per_document_dir.mkdir(parents=True, exist_ok=True)
    registry = {row.get("document_id", ""): row for row in read_csv_rows(registry_path(doc_type, root))}
    selected_ids = doc_ids or _doc_ids_from_per_dimension(out_dir)
    expected_codes = [item["code"] for item in dimensions_for(doc_type)]
    rating_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for doc_id in selected_ids:
        row = registry.get(doc_id, {"document_id": doc_id, "doc_type": doc_type})
        records = _dimension_records(out_dir, doc_id, expected_codes)
        success_records = {record.get("dimension_code"): record for record in records if record.get("status") == "success"}
        failed_codes = [code for code in expected_codes if code not in success_records]
        raw_paths = {
            code: str(success_records.get(code, {}).get("result", {}).get("raw_output_path", ""))
            for code in expected_codes
        }
        if failed_codes:
            failed = _single_failed_case(doc_id, doc_type, records, failed_codes, "missing_dimension_scores")
            failed_rows.append(failed)
            save_json(per_document_dir / f"{doc_id}_parsed.json", _failed_per_document_record(failed))
            continue
        dimension_scores = [success_records[code]["payload"] for code in expected_codes]
        payload = normalize_scoring_payload(
            {
                "document_id": doc_id,
                "doc_type": doc_type,
                "scoring_basis": scoring_basis_for(doc_type),
                "scorebook_version": SCOREBOOK_VERSION,
                "evidence_limitation": "Single-dimension local qwen3:8b scoring; Python-only aggregation.",
                "dimension_scores": dimension_scores,
                "flags": {field: 0 for field in FLAG_FIELDS},
                "overall_comment": "Aggregated from five single-dimension scoring calls.",
                "needs_review": _needs_review(dimension_scores),
                "review_reasons": _payload_review_reasons(dimension_scores),
            },
            doc_type,
            doc_id,
        )
        text_path = resolve_text_path(row, doc_type, root)
        text = text_path.read_text(encoding="utf-8", errors="replace") if text_path.exists() else ""
        validation = validate_scoring_payload(payload, text, doc_type)
        if not validation.is_valid:
            failed = _single_failed_case(doc_id, doc_type, records, [], "schema_validation_failed", ";".join(validation.issues))
            failed_rows.append(failed)
            save_json(per_document_dir / f"{doc_id}_parsed.json", _failed_per_document_record(failed))
            continue
        started_at = min(str(record.get("result", {}).get("started_at", now_iso())) for record in records)
        finished_at = max(str(record.get("result", {}).get("finished_at", now_iso())) for record in records)
        record_model = str(records[0].get("result", {}).get("model_name") or model_name)
        numeric_check_path = _numeric_check_path(records)
        joined_raw_paths = ";".join(path for path in raw_paths.values() if path)
        doc_score_row = _build_document_score_row(payload, row, doc_type, record_model)
        rows = _build_rating_rows(payload, doc_type, started_at, finished_at, record_model, joined_raw_paths, numeric_check_path, "parsed", "valid")
        for rating_row in rows:
            rating_row["raw_output_path"] = raw_paths.get(rating_row["dimension_code"], joined_raw_paths)
        reviews = _review_reasons(payload, text, row, doc_type, float(doc_score_row["total_score_100"]), _numeric_payload(numeric_check_path), numeric_check_path, joined_raw_paths)
        result = {
            "document_id": doc_id,
            "doc_type": doc_type,
            "success": True,
            "status": "success",
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
            "worker_id": "single-dimension-aggregator",
        }
        save_json(
            per_document_dir / f"{doc_id}_parsed.json",
            {
                "document_id": doc_id,
                "doc_type": doc_type,
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
    write_csv_rows(out_dir / f"{ratings_prefix}_document_scores.csv", document_headers_for(doc_type), document_rows)
    write_csv_rows(out_dir / f"{ratings_prefix}_review_log.csv", REVIEW_LOG_HEADERS, review_rows)
    write_csv_rows(out_dir / f"{ratings_prefix}_failed_cases.csv", SINGLE_DIMENSION_FAILED_HEADERS, failed_rows)
    summary = {
        "doc_type": doc_type,
        "scoring_strategy": "single_dimension",
        "output_name": output_name,
        "ratings_prefix": ratings_prefix,
        "selected_count": len(selected_ids),
        "success_count": len(document_rows),
        "failure_count": len(failed_rows),
        "dimension_count_expected": len(expected_codes),
        "document_scores_path": str(out_dir / f"{ratings_prefix}_document_scores.csv"),
        "ratings_long_path": str(out_dir / f"{ratings_prefix}_ratings_long.csv"),
        "failed_cases_path": str(out_dir / f"{ratings_prefix}_failed_cases.csv"),
    }
    save_json(out_dir / f"{ratings_prefix}_aggregation_summary.json", summary)
    return summary


def _doc_ids_from_per_dimension(out_dir: Path) -> list[str]:
    doc_ids: list[str] = []
    for path in sorted((out_dir / "per_dimension").glob("*_parsed.json")):
        stem = path.name.removesuffix("_parsed.json")
        parts = stem.split("_")
        if len(parts) < 2:
            continue
        code = parts[-1]
        doc_id = stem.removesuffix(f"_{code}")
        if doc_id and doc_id not in doc_ids:
            doc_ids.append(doc_id)
    return doc_ids


def _dimension_records(out_dir: Path, doc_id: str, expected_codes: list[str]) -> list[dict[str, Any]]:
    records = []
    for code in expected_codes:
        path = out_dir / "per_dimension" / f"{doc_id}_{code}_parsed.json"
        if not path.exists():
            continue
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            records.append({"document_id": doc_id, "status": "failed", "dimension_code": code, "result": {"error_type": "json_decode_error"}})
    return records


def _single_failed_case(
    doc_id: str,
    doc_type: str,
    records: list[dict[str, Any]],
    failed_codes: list[str],
    error_type: str,
    error_message: str = "",
) -> dict[str, Any]:
    results = [record.get("result", {}) if isinstance(record.get("result"), dict) else {} for record in records]
    parse_status = "parsed" if records and all(result.get("parse_status") == "parsed" for result in results) else "parse_failed"
    schema_status = "valid" if records and all(result.get("schema_validation_status") == "valid" for result in results) else "invalid"
    raw_paths = [str(result.get("raw_output_path", "")) for result in results if result.get("raw_output_path")]
    return {
        "document_id": doc_id,
        "doc_type": doc_type,
        "status": "failed",
        "error_type": error_type,
        "error_message": error_message,
        "parse_status": parse_status,
        "schema_validation_status": schema_status,
        "dimension_count": len([record for record in records if record.get("status") == "success"]),
        "failed_dimensions": ";".join(failed_codes),
        "raw_output_path": ";".join(raw_paths),
    }


def _failed_per_document_record(failed: dict[str, Any]) -> dict[str, Any]:
    return {
        "document_id": failed["document_id"],
        "doc_type": failed["doc_type"],
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
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _duration_seconds(records: list[dict[str, Any]]) -> float:
    return round(sum(float(record.get("result", {}).get("duration_seconds") or 0) for record in records), 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate per-document parsed scoring JSON into CSV outputs.")
    parser.add_argument("--doc-type", choices=["mda", "news"], required=True)
    args = parser.parse_args()
    print(aggregate_scores(args.doc_type))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
