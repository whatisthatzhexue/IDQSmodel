from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from json_stabilization import is_parser_repairable, parse_model_json
from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


DIAGNOSIS_HEADERS = [
    "document_id",
    "source_run",
    "raw_output_path",
    "failure_type",
    "has_markdown_fence",
    "has_think_block",
    "has_leading_text",
    "has_trailing_text",
    "has_multiple_json_objects",
    "has_truncated_json",
    "missing_opening_brace",
    "missing_closing_brace",
    "invalid_escape",
    "invalid_quote",
    "missing_comma",
    "wrong_root_type",
    "missing_required_fields",
    "missing_dimension_scores",
    "dimension_count",
    "raw_output_length",
    "recommended_fix",
]

RUN_SPECS = [
    {
        "source_run": "mda_v2_sample",
        "raw_dir": "05_raw_model_outputs/mda_v2_sample",
        "ratings_dir": "06_ratings/mda_v2_sample_rescore",
    },
    {
        "source_run": "mda_v2_stability_batch",
        "raw_dir": "05_raw_model_outputs/mda_v2_stability_batch",
        "ratings_dir": "06_ratings/mda_v2_stability_batch",
    },
]


def diagnose_qwen_json_failures(root: Path = SCORING_ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in RUN_SPECS:
        rows.extend(_diagnose_run(root, spec["source_run"], root / spec["raw_dir"], root / spec["ratings_dir"]))

    output_csv = root / "08_reports" / "qwen_json_failure_diagnosis.csv"
    write_csv_rows(output_csv, DIAGNOSIS_HEADERS, rows)
    summary = _summarize(rows)
    _write_report(root / "08_reports" / "qwen_json_failure_diagnosis.md", summary)
    return summary


def _diagnose_run(root: Path, source_run: str, raw_dir: Path, ratings_dir: Path) -> list[dict[str, Any]]:
    records = _load_failed_records(ratings_dir)
    rows: list[dict[str, Any]] = []
    for record_path, record in records:
        document_id = str(record.get("document_id") or record_path.name.removesuffix("_parsed.json"))
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        raw_path = _resolve_raw_path(root, raw_dir, document_id, result.get("raw_output_path", ""))
        raw_text = _read_raw_model_text(raw_path)
        parsed = parse_model_json(raw_text)
        row = _diagnosis_row(document_id, source_run, raw_path, parsed)
        rows.append(row)
    return rows


def _load_failed_records(ratings_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    per_document = ratings_dir / "per_document"
    candidates = sorted(per_document.glob("*_parsed.json")) if per_document.exists() else []
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            record = {"document_id": path.name.removesuffix("_parsed.json"), "status": "failed", "result": {"parse_status": "parse_failed"}}
        if _is_json_failure(record):
            records.append((path, record))
    return records


def _is_json_failure(record: dict[str, Any]) -> bool:
    result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
    combined = json.dumps(record, ensure_ascii=False).lower()
    if record.get("status") != "success":
        return True
    if result.get("parse_status") == "parse_failed":
        return True
    if result.get("error_type") in {"json_parse_failed", "scoring_failed", "JSONDecodeError"}:
        return True
    return "json_parse" in combined or "scoring_failed" in combined


def _resolve_raw_path(root: Path, raw_dir: Path, document_id: str, raw_output_path: str) -> Path:
    if raw_output_path:
        candidate = Path(raw_output_path)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate
    matches = sorted(raw_dir.glob(f"{document_id}*attempt*.json"))
    if matches:
        return matches[-1]
    matches = sorted(raw_dir.glob(f"{document_id}*.json"))
    if matches:
        return matches[-1]
    return raw_dir / f"{document_id}_attempt_1.json"


def _read_raw_model_text(raw_path: Path) -> str:
    if not raw_path.exists():
        return ""
    text = raw_path.read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        for key in ["content", "response", "raw_text"]:
            value = payload.get(key)
            if isinstance(value, str):
                return value
        message = payload.get("raw_response", {}).get("message", {}) if isinstance(payload.get("raw_response"), dict) else {}
        if isinstance(message.get("content"), str):
            return message["content"]
        response = payload.get("raw_response", {}).get("response") if isinstance(payload.get("raw_response"), dict) else None
        if isinstance(response, str):
            return response
    return text


def _diagnosis_row(document_id: str, source_run: str, raw_path: Path, parsed: dict[str, Any]) -> dict[str, Any]:
    payload = parsed.get("payload")
    missing_required_fields = bool(parsed.get("missing_required_fields"))
    missing_dimension_scores = bool(parsed.get("missing_dimension_scores"))
    wrong_root_type = bool(parsed.get("wrong_root_type"))
    dimension_count = int(parsed.get("dimension_count") or 0)
    if parsed.get("ok") and isinstance(payload, dict) and "dimension_code" not in payload:
        required = {
            "document_id",
            "doc_type",
            "scoring_basis",
            "scorebook_version",
            "evidence_limitation",
            "dimension_scores",
            "flags",
            "overall_comment",
            "needs_review",
            "review_reasons",
        }
        missing_required_fields = bool(required - set(payload))
        dimension_scores = payload.get("dimension_scores")
        missing_dimension_scores = not isinstance(dimension_scores, list)
        dimension_count = len(dimension_scores) if isinstance(dimension_scores, list) else 0
    failure_type = parsed.get("failure_type") or ""
    recommended_fix = parsed.get("recommended_fix") or ""
    if parsed.get("ok") and not failure_type and wrong_root_type:
        failure_type = "wrong_schema"
        recommended_fix = "rescore_dimensionwise"
    elif parsed.get("ok") and not failure_type and missing_required_fields:
        failure_type = "missing_required_fields"
        recommended_fix = "rescore_dimensionwise"
    return {
        "document_id": document_id,
        "source_run": source_run,
        "raw_output_path": str(raw_path),
        "failure_type": failure_type or "unknown",
        "has_markdown_fence": bool(parsed.get("has_markdown_fence")),
        "has_think_block": bool(parsed.get("has_think_block")),
        "has_leading_text": bool(parsed.get("has_leading_text")),
        "has_trailing_text": bool(parsed.get("has_trailing_text")),
        "has_multiple_json_objects": bool(parsed.get("has_multiple_json_objects")),
        "has_truncated_json": bool(parsed.get("has_truncated_json")),
        "missing_opening_brace": bool(parsed.get("missing_opening_brace")),
        "missing_closing_brace": bool(parsed.get("missing_closing_brace")),
        "invalid_escape": bool(parsed.get("invalid_escape")),
        "invalid_quote": bool(parsed.get("invalid_quote")),
        "missing_comma": bool(parsed.get("missing_comma")),
        "wrong_root_type": wrong_root_type,
        "missing_required_fields": missing_required_fields,
        "missing_dimension_scores": missing_dimension_scores,
        "dimension_count": dimension_count,
        "raw_output_length": int(parsed.get("raw_output_length") or 0),
        "recommended_fix": recommended_fix or "manual_required",
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(row["failure_type"] for row in rows)
    fix_counts = Counter(row["recommended_fix"] for row in rows)
    parser_repairable = sum(1 for row in rows if _row_is_parser_repairable(row))
    rescore_needed = len(rows) - parser_repairable
    summary = {
        "failure_count": len(rows),
        "failure_type_counts": dict(sorted(type_counts.items())),
        "recommended_fix_counts": dict(sorted(fix_counts.items())),
        "parser_repairable_count": parser_repairable,
        "rescore_needed_count": rescore_needed,
        "recommend_dimensionwise": len(rows) > 0 and (rescore_needed > 0 or parser_repairable < len(rows)),
        "diagnosis_csv": "06_scoring/08_reports/qwen_json_failure_diagnosis.csv",
    }
    return summary


def _row_is_parser_repairable(row: dict[str, Any]) -> bool:
    if row["recommended_fix"] in {"strip_markdown_and_parse", "strip_think_block_and_parse", "extract_largest_json_object"}:
        return True
    if row["recommended_fix"] == "repair_json":
        return row["failure_type"] in {"invalid_escape_or_quote", "missing_comma_or_brace"}
    return False


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen JSON Failure Diagnosis",
        "",
        f"- 失败总数: {summary['failure_count']}",
        f"- 能通过 parser 修复的数量: {summary['parser_repairable_count']}",
        f"- 需要重新评分的数量: {summary['rescore_needed_count']}",
        f"- 建议切换到 dimensionwise scoring: {'是' if summary['recommend_dimensionwise'] else '否'}",
        "",
        "## Failure Type Counts",
    ]
    for failure_type, count in summary["failure_type_counts"].items():
        lines.append(f"- {failure_type}: {count}")
    lines.extend(["", "## Recommended Fix Counts"])
    for fix, count in summary["recommended_fix_counts"].items():
        lines.append(f"- {fix}: {count}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "如果 parser 可修复数量较高，应先运行 repair_existing_failed_outputs.py 生成隔离修复结果；"
            "如果仍有 schema 缺失、空响应或截断输出，应优先切换到 dimensionwise scoring，减少单次 JSON 复杂度。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose qwen3:8b JSON failures in MD&A v2 sample/stability runs.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(diagnose_qwen_json_failures(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
