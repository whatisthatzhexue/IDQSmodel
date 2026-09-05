from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows


SIMPLE_JSON_FIELDS = {"score", "evidence", "reason"}

VALIDATION_HEADERS = [
    "document_id",
    "doc_type",
    "dimension_code",
    "status",
    "parse_status",
    "schema_validation_status",
    "evidence_locator_status",
    "score_range_status",
    "issues",
    "parsed_output_path",
]


def simple_dimension_schema(doc_type: str) -> dict[str, Any]:
    prefix = _evidence_prefix(doc_type)
    return {
        "type": "object",
        "required": ["score", "evidence", "reason"],
        "properties": {
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "evidence": {"type": "string", "pattern": f"^\\[?{prefix}_P\\d{{3}}\\]?$"},
            "reason": {"type": "string", "minLength": 1, "maxLength": 240},
        },
        "additionalProperties": False,
    }


def validate_simple_dimension_json(payload: Any, doc_type: str, dimension_code: str, text: str) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload_not_object"]
    issues: list[str] = []
    for key, value in payload.items():
        if key not in SIMPLE_JSON_FIELDS:
            issues.append(f"unexpected_field_{key}")
        if isinstance(value, (dict, list)):
            issues.append(f"nested_value_{key}")
    for field in sorted(SIMPLE_JSON_FIELDS):
        if field not in payload:
            issues.append(f"missing_field_{field}")
    try:
        payload["score"] = int(payload.get("score"))
    except (TypeError, ValueError):
        issues.append("score_not_integer")
    if payload.get("score") not in {1, 2, 3, 4, 5}:
        issues.append("score_out_of_range")
    evidence = normalize_simple_evidence(str(payload.get("evidence", "")), doc_type)
    payload["evidence"] = evidence
    if not evidence:
        issues.append("empty_evidence")
    elif f"[{evidence}]" not in text:
        issues.append("evidence_locator_not_found")
    reason = " ".join(str(payload.get("reason", "")).split())
    payload["reason"] = reason[:240].rstrip()
    if not reason:
        issues.append("empty_reason")
    if len(reason) > 240:
        issues.append("reason_too_long")
    if re.search(r"\d+\s*[+\-*/]\s*\d+", reason):
        issues.append("reason_contains_arithmetic")
    return sorted(set(issues))


def normalize_simple_evidence(locator: str, doc_type: str) -> str:
    prefix = _evidence_prefix(doc_type)
    match = re.search(rf"{prefix}_P\d{{3}}", locator)
    return match.group(0) if match else ""


def validate_json_outputs(root: Path = SCORING_ROOT, output_name: str = "single_dimension", doc_type: str | None = None) -> dict[str, Any]:
    out_dir = root / "06_ratings" / output_name
    rows = []
    for path in sorted((out_dir / "per_dimension").glob("*_parsed.json")):
        record = _load_json(path)
        record_doc_type = str(record.get("doc_type") or doc_type or "")
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        evidence_locator = str(payload.get("evidence_locator", ""))
        issues = []
        if record.get("status") != "success":
            issues.append(result.get("error_type") or "dimension_failed")
        if result.get("parse_status") != "parsed":
            issues.append("json_parse_failed")
        if result.get("schema_validation_status") != "valid":
            issues.append("schema_validation_failed")
        if not evidence_locator:
            issues.append("empty_evidence_locator")
        rows.append(
            {
                "document_id": record.get("document_id", path.name),
                "doc_type": record_doc_type,
                "dimension_code": record.get("dimension_code", ""),
                "status": record.get("status", ""),
                "parse_status": result.get("parse_status", ""),
                "schema_validation_status": result.get("schema_validation_status", ""),
                "evidence_locator_status": "found" if evidence_locator else "missing",
                "score_range_status": "valid" if payload.get("raw_score") in {1, 2, 3, 4, 5} else "invalid",
                "issues": ";".join(sorted(set(str(item) for item in issues if item))),
                "parsed_output_path": str(path),
            }
        )
    total = len(rows)
    json_success = sum(row["parse_status"] == "parsed" for row in rows)
    schema_success = sum(row["schema_validation_status"] == "valid" for row in rows)
    evidence_success = sum(row["evidence_locator_status"] == "found" for row in rows)
    failed = sum(row["status"] != "success" for row in rows)
    report_dir = root / "08_reports"
    write_csv_rows(report_dir / f"{output_name}_json_validation.csv", VALIDATION_HEADERS, rows)
    summary = {
        "output_name": output_name,
        "doc_type": doc_type or "",
        "dimension_record_count": total,
        "json_success_rate": round(json_success / total, 6) if total else 0.0,
        "schema_success_rate": round(schema_success / total, 6) if total else 0.0,
        "evidence_match_rate": round(evidence_success / total, 6) if total else 0.0,
        "system_failure_rate": round(failed / total, 6) if total else 0.0,
        "validation_csv_path": str(report_dir / f"{output_name}_json_validation.csv"),
    }
    save_json(report_dir / f"{output_name}_json_validation_summary.json", summary)
    return summary


def _evidence_prefix(doc_type: str) -> str:
    if doc_type == "mda":
        return "MDA"
    if doc_type == "news":
        return "NEWS"
    raise ValueError("doc_type must be mda or news")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "result": {"error_type": "json_decode_error"}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate single-dimension parsed JSON outputs.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--doc-type", choices=["mda", "news"])
    args = parser.parse_args()
    print(json.dumps(validate_json_outputs(args.root, args.output_name, args.doc_type), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
