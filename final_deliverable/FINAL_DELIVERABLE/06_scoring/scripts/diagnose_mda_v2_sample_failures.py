from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


FAILURE_HEADERS = [
    "document_id",
    "company_name",
    "ticker",
    "report_year",
    "failure_stage",
    "error_type",
    "error_message",
    "text_word_count",
    "paragraph_count",
    "numeric_issue_count",
    "json_parse_status",
    "schema_validation_status",
    "evidence_locator_status",
    "scope_violation_status",
    "retry_count",
    "manual_required",
    "recommended_action",
]


def diagnose_mda_v2_sample_failures(root: Path = SCORING_ROOT) -> dict[str, Any]:
    sample_dir = root / "06_ratings" / "mda_v2_sample_rescore"
    parsed_dir = sample_dir / "per_document"
    registry = {row.get("document_id", ""): row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv")}
    numeric_summary = {
        row.get("document_id", ""): row for row in read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv")
    }
    rows: list[dict[str, Any]] = []
    for parsed_path in sorted(parsed_dir.glob("*_parsed.json")):
        record = _load_json(parsed_path)
        if record.get("status") == "success":
            continue
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        doc_id = record.get("document_id") or result.get("document_id") or parsed_path.name.removesuffix("_parsed.json")
        meta = registry.get(doc_id, {})
        text = _read_clean_text(root, doc_id)
        numeric_issues = _int(numeric_summary.get(doc_id, {}).get("num_material_discrepancies", 0)) + _int(
            numeric_summary.get(doc_id, {}).get("num_severe_direction_conflicts", 0)
        )
        parse_status = result.get("parse_status") or result.get("json_parse_status") or ""
        schema_status = result.get("schema_validation_status") or ""
        error_type = result.get("error_type", "")
        error_message = result.get("error_message", "")
        attempts = _int(result.get("attempts_used", 0))
        evidence_status = _evidence_locator_status(result)
        scope_status = _scope_violation_status(result)
        stage = _classify_stage(text, numeric_issues, parse_status, schema_status, error_type, error_message, evidence_status, scope_status)
        rows.append(
            {
                "document_id": doc_id,
                "company_name": meta.get("company_name", ""),
                "ticker": meta.get("ticker", ""),
                "report_year": meta.get("report_year", ""),
                "failure_stage": stage,
                "error_type": error_type,
                "error_message": error_message,
                "text_word_count": len(text.split()),
                "paragraph_count": text.count("[MDA_P") or len([part for part in text.split("\n\n") if part.strip()]),
                "numeric_issue_count": numeric_issues,
                "json_parse_status": parse_status,
                "schema_validation_status": schema_status,
                "evidence_locator_status": evidence_status,
                "scope_violation_status": scope_status,
                "retry_count": max(0, attempts - 1),
                "manual_required": "1",
                "recommended_action": _recommended_action(stage),
            }
        )
    write_csv_rows(sample_dir / "mda_v2_failure_cases.csv", FAILURE_HEADERS, rows)
    report_path = root / "08_reports" / "mda_v2_sample_failure_diagnosis.md"
    _write_report(report_path, rows)
    return {
        "failure_count": len(rows),
        "stage_counts": dict(Counter(row["failure_stage"] for row in rows)),
        "error_type_counts": dict(Counter(row["error_type"] for row in rows)),
        "auto_fix_count": sum(1 for row in rows if row["recommended_action"] in {"repair_json_and_retry", "fix_prompt_and_retry", "reduce_context_and_retry"}),
        "manual_review_count": sum(1 for row in rows if row["recommended_action"] in {"manual_review", "numeric_checker_review"}),
        "report_path": str(report_path),
    }


def _classify_stage(
    text: str,
    numeric_issues: int,
    parse_status: str,
    schema_status: str,
    error_type: str,
    error_message: str,
    evidence_status: str,
    scope_status: str,
) -> str:
    lowered = f"{error_type} {error_message}".lower()
    if "timeout" in lowered:
        return "ollama_timeout"
    if parse_status == "parse_failed" or "json_parse" in lowered:
        return "json_parse"
    if schema_status == "invalid" or "schema" in lowered:
        return "schema_validation"
    if len(text.split()) < 80:
        return "text_quality"
    if evidence_status == "missing_or_invalid":
        return "missing_evidence"
    if scope_status == "scope_violation":
        return "scope_violation"
    if numeric_issues > 0:
        return "numeric_check"
    if "manual" in lowered:
        return "manual_required"
    return "unknown"


def _recommended_action(stage: str) -> str:
    return {
        "text_quality": "reduce_context_and_retry",
        "numeric_check": "numeric_checker_review",
        "ollama_timeout": "reduce_context_and_retry",
        "json_parse": "repair_json_and_retry",
        "schema_validation": "fix_prompt_and_retry",
        "missing_evidence": "fix_prompt_and_retry",
        "scope_violation": "manual_review",
        "manual_required": "manual_review",
        "unknown": "manual_review",
    }[stage]


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    stage_counts = Counter(row["failure_stage"] for row in rows)
    error_counts = Counter(row["error_type"] for row in rows)
    auto_fix = sum(1 for row in rows if row["recommended_action"] in {"repair_json_and_retry", "fix_prompt_and_retry", "reduce_context_and_retry"})
    manual = sum(1 for row in rows if row["recommended_action"] in {"manual_review", "numeric_checker_review"})
    blockers = [stage for stage in ["json_parse", "schema_validation", "missing_evidence", "scope_violation"] if stage_counts.get(stage)]
    lines = [
        "# MD&A v2 Sample Failure Diagnosis",
        "",
        f"- failure total: {len(rows)}",
        f"- auto-fix candidates: {auto_fix}",
        f"- manual/numeric review required: {manual}",
        "",
        "## Failure Stage Counts",
        *[f"- {key}: {value}" for key, value in sorted(stage_counts.items())],
        "",
        "## Error Type Counts",
        *[f"- {key or 'blank'}: {value}" for key, value in sorted(error_counts.items())],
        "",
        "## Full Rescore Blockers",
        *(f"- {item}" for item in blockers),
        "",
    ]
    if not blockers:
        lines.append("- none identified from failed sample records")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_clean_text(root: Path, document_id: str) -> str:
    path = root / "02_extracted_text" / "mda_clean_v2" / f"{document_id}.txt"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _evidence_locator_status(result: dict[str, Any]) -> str:
    text = json.dumps(result, ensure_ascii=False)
    return "missing_or_invalid" if "evidence" in text.lower() and "missing" in text.lower() else "unknown"


def _scope_violation_status(result: dict[str, Any]) -> str:
    text = json.dumps(result, ensure_ascii=False).lower()
    return "scope_violation" if "external financial statement" in text or "scope_violation" in text else "none"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"document_id": path.name.removesuffix("_parsed.json"), "status": "failed", "result": {"error_type": "json_decode_error"}}


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose MD&A v2 sample failed/manual-required cases.")
    parser.add_argument("--root", default=str(SCORING_ROOT))
    args = parser.parse_args()
    print(diagnose_mda_v2_sample_failures(Path(args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
