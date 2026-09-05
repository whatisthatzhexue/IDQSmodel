from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from check_ollama import run_check
from diagnose_mda_stability_blocker import CASE_HEADERS
from json_stabilization import parse_model_json
from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from stability_common import write_markdown
from validate_json_outputs import normalize_simple_evidence, validate_simple_dimension_json


ACTION_HEADERS = [
    *CASE_HEADERS,
    "action_status",
    "action_detail",
    "repair_raw_output_path",
    "repair_parsed_output_path",
    "created_at",
]

DIMENSION_CODES = {"AR01", "AR02", "AR03", "AR04", "AR05"}


def repair_mda_stability_failures(root: Path = SCORING_ROOT) -> dict[str, Any]:
    cases_path = root / "PAPER_OUTPUT" / "00_status" / "mda_stability_blocker_cases.csv"
    cases = read_csv_rows(cases_path)
    out_dir = root / "PAPER_OUTPUT" / "00_status"
    repair_ratings_dir = root / "06_ratings" / "mda_stability_repair"
    repair_raw_dir = root / "05_raw_model_outputs" / "mda_stability_repair"
    (repair_ratings_dir / "per_dimension").mkdir(parents=True, exist_ok=True)
    (repair_ratings_dir / "per_document").mkdir(parents=True, exist_ok=True)
    repair_raw_dir.mkdir(parents=True, exist_ok=True)

    ollama_status: dict[str, Any] | None = None
    actions: list[dict[str, Any]] = []
    for case in cases:
        action = case.get("recommended_action", "")
        if action == "repair_json_only":
            result = _repair_json_case(root, case, repair_ratings_dir, repair_raw_dir)
        elif action in {"rerun_single_dimension", "rerun_failed_dimension_only", "rerun_document_all_dimensions"}:
            if ollama_status is None:
                ollama_status = run_check(root=root)
            result = _rerun_case(case, ollama_status)
        elif action == "fix_evidence_mapping":
            result = _evidence_mapping_case(root, case)
        elif action == "fix_numeric_checker":
            result = _numeric_checker_case(case)
        elif action == "exclude_from_comparison_as_not_comparable":
            result = _non_comparable_case(case)
        elif action == "manual_review":
            result = _manual_case(case)
        elif action == "no_action":
            result = {"action_status": "no_action", "action_detail": "No repair action was required.", "repair_raw_output_path": "", "repair_parsed_output_path": ""}
        else:
            result = {"action_status": "manual_required", "action_detail": f"Unknown recommended_action: {action}", "repair_raw_output_path": "", "repair_parsed_output_path": ""}
        actions.append({**case, **result, "created_at": _now_iso()})

    write_csv_rows(out_dir / "mda_stability_repair_actions.csv", ACTION_HEADERS, actions)
    summary = _summary(cases, actions)
    _write_log(out_dir / "mda_stability_repair_log.md", summary, actions, ollama_status)
    _write_after_repair_report(out_dir / "mda_stability_after_repair.md", root, summary)
    save_json(out_dir / "mda_stability_repair_summary.json", summary)
    return summary


def _repair_json_case(root: Path, case: dict[str, str], repair_ratings_dir: Path, repair_raw_dir: Path) -> dict[str, str]:
    raw_path = _resolve_path(root, case.get("raw_output_path", ""))
    document_id = case.get("document_id", "")
    dimension_code = case.get("dimension_code", "")
    if not raw_path or not raw_path.exists():
        return {
            "action_status": "manual_required",
            "action_detail": "raw_output_path missing or not found; cannot repair without source output",
            "repair_raw_output_path": "",
            "repair_parsed_output_path": "",
        }
    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_model_json(raw_text)
    repair_raw_path = repair_raw_dir / f"{document_id}_{dimension_code or 'ALL'}_raw.txt"
    repair_raw_path.write_text(raw_text, encoding="utf-8")
    if not parsed.get("ok"):
        parsed_path = _repair_parsed_path(repair_ratings_dir, document_id, dimension_code)
        save_json(
            parsed_path,
            {
                "status": "manual_required",
                "doc_type": "mda",
                "document_id": document_id,
                "dimension_code": dimension_code,
                "result": {
                    "parse_status": "parse_failed",
                    "schema_validation_status": "invalid",
                    "error_type": parsed.get("failure_type", "json_parse_failure"),
                    "error_message": parsed.get("parse_error", ""),
                    "raw_output_path": str(repair_raw_path),
                },
            },
        )
        return {
            "action_status": "manual_required",
            "action_detail": f"parser failed: {parsed.get('failure_type', 'unknown')}",
            "repair_raw_output_path": str(repair_raw_path),
            "repair_parsed_output_path": str(parsed_path),
        }

    payload = parsed.get("payload")
    parsed_path = _repair_parsed_path(repair_ratings_dir, document_id, dimension_code)
    if dimension_code in DIMENSION_CODES and isinstance(payload, dict):
        text = _mda_text(root, document_id)
        simple_payload = dict(payload)
        issues = validate_simple_dimension_json(simple_payload, "mda", dimension_code, text)
        if not issues:
            evidence = normalize_simple_evidence(str(simple_payload.get("evidence", "")), "mda")
            expanded = {
                **simple_payload,
                "raw_score": simple_payload["score"],
                "evidence_locator": f"[{evidence}]",
                "comment_short": simple_payload.get("reason", ""),
                "confidence_level": "repaired",
            }
            save_json(
                parsed_path,
                {
                    "status": "success",
                    "doc_type": "mda",
                    "document_id": document_id,
                    "dimension_code": dimension_code,
                    "payload": expanded,
                    "result": {
                        "parse_status": "parsed",
                        "schema_validation_status": "valid",
                        "error_type": "",
                        "error_message": "",
                        "raw_output_path": str(repair_raw_path),
                        "parser_failure_type": parsed.get("failure_type", ""),
                    },
                },
            )
            return {
                "action_status": "repaired",
                "action_detail": "JSON parsed, schema validated, evidence locator found, and score range valid.",
                "repair_raw_output_path": str(repair_raw_path),
                "repair_parsed_output_path": str(parsed_path),
            }
        save_json(
            parsed_path,
            {
                "status": "manual_required",
                "doc_type": "mda",
                "document_id": document_id,
                "dimension_code": dimension_code,
                "payload": payload,
                "result": {
                    "parse_status": "parsed",
                    "schema_validation_status": "invalid",
                    "error_type": "schema_or_evidence_validation_failed",
                    "error_message": ";".join(issues),
                    "raw_output_path": str(repair_raw_path),
                },
            },
        )
        return {
            "action_status": "manual_required",
            "action_detail": "parser recovered JSON but validation failed: " + ";".join(issues),
            "repair_raw_output_path": str(repair_raw_path),
            "repair_parsed_output_path": str(parsed_path),
        }

    save_json(
        parsed_path,
        {
            "status": "manual_required",
            "doc_type": "mda",
            "document_id": document_id,
            "dimension_code": dimension_code or "ALL",
            "payload": payload,
            "result": {
                "parse_status": "parsed",
                "schema_validation_status": "invalid",
                "error_type": "not_single_dimension_payload",
                "error_message": "Recovered JSON is not a strict single-dimension payload.",
                "raw_output_path": str(repair_raw_path),
            },
        },
    )
    return {
        "action_status": "manual_required",
        "action_detail": "parser recovered JSON, but the payload is not a strict single-dimension object.",
        "repair_raw_output_path": str(repair_raw_path),
        "repair_parsed_output_path": str(parsed_path),
    }


def _rerun_case(case: dict[str, str], ollama_status: dict[str, Any]) -> dict[str, str]:
    if not ollama_status.get("ollama_available") or not ollama_status.get("model_available"):
        errors = "; ".join(str(item) for item in ollama_status.get("errors", []))
        return {
            "action_status": "blocked_ollama_unavailable",
            "action_detail": f"qwen3:8b rerun was not attempted because Ollama/model is unavailable. {errors}".strip(),
            "repair_raw_output_path": "",
            "repair_parsed_output_path": "",
        }
    return {
        "action_status": "manual_required",
        "action_detail": "Ollama is available, but this safety script does not overwrite historical outputs; run run_single_dimension_scoring with output_name=mda_stability_repair for the listed document/dimension.",
        "repair_raw_output_path": "",
        "repair_parsed_output_path": "",
    }


def _evidence_mapping_case(root: Path, case: dict[str, str]) -> dict[str, str]:
    evidence = case.get("evidence_locator", "")
    document_id = case.get("document_id", "")
    text = _mda_text(root, document_id)
    exists = bool(evidence and all(locator in text for locator in _locators(evidence)))
    if exists:
        return {
            "action_status": "repaired",
            "action_detail": "Evidence locator is present in current extracted text; case can be revalidated without changing scores.",
            "repair_raw_output_path": "",
            "repair_parsed_output_path": "",
        }
    return {
        "action_status": "manual_required",
        "action_detail": "Evidence locator still missing in extracted text; paragraph mapping requires review.",
        "repair_raw_output_path": "",
        "repair_parsed_output_path": "",
    }


def _numeric_checker_case(case: dict[str, str]) -> dict[str, str]:
    return {
        "action_status": "manual_required",
        "action_detail": "AR02 numeric discrepancy is marked for numeric-checker review only; no LLM math or score change was performed.",
        "repair_raw_output_path": "",
        "repair_parsed_output_path": "",
    }


def _non_comparable_case(case: dict[str, str]) -> dict[str, str]:
    return {
        "action_status": "excluded_from_stability_comparison_only",
        "action_detail": "Case is documented as non-comparable for the relevant stability comparison; it is not deleted from final data.",
        "repair_raw_output_path": "",
        "repair_parsed_output_path": "",
    }


def _manual_case(case: dict[str, str]) -> dict[str, str]:
    return {
        "action_status": "manual_required",
        "action_detail": "Manual review required; no score was generated.",
        "repair_raw_output_path": "",
        "repair_parsed_output_path": "",
    }


def _repair_parsed_path(repair_ratings_dir: Path, document_id: str, dimension_code: str) -> Path:
    if dimension_code in DIMENSION_CODES:
        return repair_ratings_dir / "per_dimension" / f"{document_id}_{dimension_code}_parsed.json"
    return repair_ratings_dir / "per_document" / f"{document_id}_parsed.json"


def _mda_text(root: Path, document_id: str) -> str:
    direct = root / "02_extracted_text" / "mda" / f"{document_id}.txt"
    if direct.exists():
        return direct.read_text(encoding="utf-8", errors="replace")
    for row in read_csv_rows(root / "01_registry" / "mda_registry.csv"):
        if row.get("document_id") != document_id or not row.get("text_path"):
            continue
        candidate = root / row["text_path"]
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace")
    return ""


def _resolve_path(root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _locators(evidence: str) -> list[str]:
    import re

    return re.findall(r"MDA_P\d{3}", evidence or "")


def _summary(cases: list[dict[str, str]], actions: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(action.get("action_status", "") for action in actions)
    action_counts = Counter(action.get("recommended_action", "") for action in actions)
    repaired = status_counts.get("repaired", 0)
    still_failed = sum(count for status, count in status_counts.items() if status not in {"repaired", "no_action", "excluded_from_stability_comparison_only"})
    return {
        "case_count": len(cases),
        "action_counts": dict(action_counts),
        "action_status_counts": dict(status_counts),
        "number_of_cases_repaired": repaired,
        "number_of_cases_still_failed": still_failed,
        "remaining_blockers": [
            status
            for status in sorted(status_counts)
            if status not in {"repaired", "no_action", "excluded_from_stability_comparison_only"}
        ],
    }


def _write_log(path: Path, summary: dict[str, Any], actions: list[dict[str, Any]], ollama_status: dict[str, Any] | None) -> None:
    lines = [
        "# MD&A Stability Repair Log",
        "",
        f"- case_count: {summary['case_count']}",
        f"- number_of_cases_repaired: {summary['number_of_cases_repaired']}",
        f"- number_of_cases_still_failed: {summary['number_of_cases_still_failed']}",
        f"- remaining_blockers: {', '.join(summary['remaining_blockers']) or 'none'}",
        "",
        "## Action Status Counts",
    ]
    for status, count in summary["action_status_counts"].items():
        lines.append(f"- {status}: {count}")
    if ollama_status is not None:
        lines.extend(
            [
                "",
                "## Ollama Status For Reruns",
                f"- ollama_available: {ollama_status.get('ollama_available')}",
                f"- model_available: {ollama_status.get('model_available')}",
                f"- errors: {'; '.join(str(item) for item in ollama_status.get('errors', [])) or 'none'}",
            ]
        )
    lines.extend(["", "## Guardrails", "- Historical v1/v2/pilot/sample outputs were not overwritten.", "- No scores were fabricated."])
    write_markdown(path, lines)


def _write_after_repair_report(path: Path, root: Path, summary: dict[str, Any]) -> None:
    paper_status_path = root / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json"
    status = {}
    if paper_status_path.exists():
        try:
            status = json.loads(paper_status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            status = {}
    before = status.get("mda_stability_success_rate", "")
    after = status.get("mda_stability_success_rate", "")
    lines = [
        "# MD&A Stability After Repair",
        "",
        f"- mda_stability_before: {before}",
        f"- mda_stability_after: {after}",
        f"- number_of_cases_repaired: {summary['number_of_cases_repaired']}",
        f"- number_of_cases_still_failed: {summary['number_of_cases_still_failed']}",
        f"- remaining_blockers: {', '.join(summary['remaining_blockers']) or 'none'}",
        f"- whether_mda_ready_now: {str(bool(after and float(after) >= 0.85 and summary['number_of_cases_still_failed'] == 0)).lower() if after != '' else 'false'}",
        "",
        "This report is refreshed by the repair script before downstream stability/freeze commands are rerun; downstream status files remain authoritative after those commands finish.",
    ]
    write_markdown(path, lines)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply targeted, non-overwriting repairs for MD&A stability blocker cases.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(repair_mda_stability_failures(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
