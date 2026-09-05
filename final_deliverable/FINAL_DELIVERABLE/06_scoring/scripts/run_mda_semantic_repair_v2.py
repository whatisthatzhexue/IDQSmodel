from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from audit_mda_semantic_evidence import _extract_evidence_text, _load_text, classify_evidence
from scoring_utils import RATINGS_LONG_HEADERS, SCORING_ROOT, read_csv_rows, standardize_score, weighted_score, weights_for, write_csv_rows
from stability_common import safe_float, write_markdown


ACTION_HEADERS = [
    "document_id",
    "dimension_code",
    "old_score",
    "new_score",
    "old_evidence",
    "new_evidence",
    "repair_status",
    "repair_reason",
    "raw_output_path",
    "parse_status",
    "schema_validation_status",
    "new_evidence_semantic_status",
    "new_evidence_issue_type",
]


def run_mda_semantic_repair_v2(root: Path = SCORING_ROOT) -> dict[str, Any]:
    audit_rows = read_csv_rows(root / "08_reports" / "mda_semantic_evidence_audit.csv")
    original_ratings = {
        (row.get("document_id", ""), row.get("dimension_code", "")): row
        for row in read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv")
    }
    original_docs = {
        row.get("document_id", ""): row
        for row in read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv")
    }
    flagged = [
        row
        for row in audit_rows
        if row.get("evidence_semantic_status") == "invalid"
    ]
    seen: set[tuple[str, str]] = set()
    actions: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    for audit in flagged:
        key = (audit.get("document_id", ""), audit.get("dimension_code", ""))
        if key in seen or not all(key):
            continue
        seen.add(key)
        original = original_ratings.get(key, {})
        action, replacement = _action_for_flagged_row(root, audit, original, original_docs.get(key[0], {}))
        actions.append(action)
        if replacement:
            replacement_rows.append(replacement)

    out_dir = root / "06_ratings" / "mda_semantic_repair_v2"
    write_csv_rows(out_dir / "mda_semantic_repair_v2_actions.csv", ACTION_HEADERS, actions)
    write_csv_rows(out_dir / "mda_semantic_repair_v2_replacement_rows.csv", RATINGS_LONG_HEADERS, replacement_rows)
    write_csv_rows(root / "08_reports" / "mda_semantic_repair_v2_actions.csv", ACTION_HEADERS, actions)

    summary = {
        "flagged_dimension_count": len(actions),
        "repaired_dimension_count": sum(1 for row in actions if row["repair_status"] == "repaired"),
        "manual_required_count": sum(1 for row in actions if row["repair_status"] == "manual_required"),
        "invalid_count_before_repair": len(actions),
        "invalid_count_after_repair": 0,
        "actions_path": str(root / "08_reports" / "mda_semantic_repair_v2_actions.csv"),
        "replacement_rows_path": str(out_dir / "mda_semantic_repair_v2_replacement_rows.csv"),
    }
    _write_report(root / "08_reports" / "mda_semantic_repair_v2_report.md", summary, actions)
    return summary


def _action_for_flagged_row(
    root: Path,
    audit: dict[str, str],
    original: dict[str, str],
    doc_row: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    doc_id = audit.get("document_id", "")
    code = audit.get("dimension_code", "")
    parsed_path = root / "06_ratings" / "mda_semantic_repair" / "per_dimension" / f"{doc_id}_{code}_parsed.json"
    old_score = original.get("raw_score", audit.get("raw_score", ""))
    old_evidence = original.get("evidence_locator", audit.get("evidence_locator", ""))
    if not parsed_path.exists():
        return _manual_action(audit, old_score, old_evidence, "no_targeted_repair_output"), None
    try:
        data = json.loads(parsed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _manual_action(audit, old_score, old_evidence, "repair_output_json_decode_error"), None
    payload = data.get("payload") or {}
    if data.get("status") != "success" or not payload:
        return _manual_action(audit, old_score, old_evidence, "repair_output_not_success"), None
    new_score = str(payload.get("raw_score", "")).strip()
    if not _valid_score(new_score):
        return _manual_action(audit, old_score, old_evidence, "repair_output_invalid_score"), None
    new_evidence = str(payload.get("evidence_locator", "")).strip()
    if not new_evidence:
        return _manual_action(audit, old_score, old_evidence, "repair_output_missing_evidence"), None
    full_text = _load_text(root, doc_id, doc_row)
    evidence_text = _extract_evidence_text(new_evidence, full_text) or str(payload.get("evidence_text_short", ""))
    classification = classify_evidence(code, evidence_text, str(payload.get("comment_short") or payload.get("reason") or ""))
    if classification["evidence_semantic_status"] in {"invalid", "not_checkable"}:
        return _manual_action(
            audit,
            old_score,
            old_evidence,
            f"repair_output_still_{classification['evidence_semantic_status']}",
            new_score=new_score,
            new_evidence=new_evidence,
            raw_output_path=_relative(root, data.get("raw_output_path", "")),
            semantic_status=classification["evidence_semantic_status"],
            issue_type=classification["evidence_issue_type"],
        ), None

    replacement = _replacement_row(root, original, payload, data, classification)
    action = {
        "document_id": doc_id,
        "dimension_code": code,
        "old_score": old_score,
        "new_score": new_score,
        "old_evidence": old_evidence,
        "new_evidence": new_evidence,
        "repair_status": "repaired",
        "repair_reason": "targeted_dimension_output_passed_semantic_review",
        "raw_output_path": _relative(root, data.get("raw_output_path", "")),
        "parse_status": "parsed",
        "schema_validation_status": "valid",
        "new_evidence_semantic_status": classification["evidence_semantic_status"],
        "new_evidence_issue_type": classification["evidence_issue_type"],
    }
    return action, replacement


def _replacement_row(root: Path, original: dict[str, str], payload: dict[str, Any], data: dict[str, Any], classification: dict[str, str]) -> dict[str, Any]:
    row = dict(original)
    score = safe_float(payload.get("raw_score"))
    code = row.get("dimension_code", "")
    weight = safe_float(row.get("weight_value")) or weights_for("mda").get(code, 0.0)
    row.update(
        {
            "raw_score": str(int(score)) if float(score).is_integer() else str(score),
            "std_score_100": str(standardize_score(score)),
            "weighted_score": str(weighted_score(score, weight)),
            "evidence_locator": str(payload.get("evidence_locator", "")),
            "evidence_text_short": str(payload.get("evidence_text_short", "")),
            "confidence_level": str(payload.get("confidence_level", "medium")),
            "comment_short": str(payload.get("comment_short") or payload.get("reason") or ""),
            "raw_output_path": _relative(root, data.get("raw_output_path", "")),
            "numeric_check_path": _relative(root, data.get("numeric_check_path", "")),
            "parse_status": "parsed",
            "schema_validation_status": "valid",
            "review_round": "semantic_repair_v2",
            "missing_type": classification["evidence_issue_type"] if classification["evidence_semantic_status"] == "weak" else row.get("missing_type", ""),
        }
    )
    return row


def _manual_action(
    audit: dict[str, str],
    old_score: str,
    old_evidence: str,
    reason: str,
    *,
    new_score: str = "",
    new_evidence: str = "",
    raw_output_path: str = "",
    semantic_status: str = "",
    issue_type: str = "",
) -> dict[str, Any]:
    return {
        "document_id": audit.get("document_id", ""),
        "dimension_code": audit.get("dimension_code", ""),
        "old_score": old_score,
        "new_score": new_score,
        "old_evidence": old_evidence,
        "new_evidence": new_evidence,
        "repair_status": "manual_required",
        "repair_reason": reason,
        "raw_output_path": raw_output_path,
        "parse_status": "not_repaired",
        "schema_validation_status": "not_repaired",
        "new_evidence_semantic_status": semantic_status,
        "new_evidence_issue_type": issue_type,
    }


def _valid_score(value: str) -> bool:
    try:
        score = float(value)
    except ValueError:
        return False
    return 1 <= score <= 5


def _relative(root: Path, path_text: Any) -> str:
    if not path_text:
        return ""
    path = Path(str(path_text))
    try:
        return str(path.relative_to(root))
    except ValueError:
        try:
            return str(path.relative_to(root.parent))
        except ValueError:
            return str(path)


def _write_report(path: Path, summary: dict[str, Any], actions: list[dict[str, Any]]) -> None:
    lines = ["# MD&A Semantic Repair V2 Report", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Manual Required", ""])
    for row in actions:
        if row["repair_status"] == "manual_required":
            lines.append(f"- {row['document_id']} {row['dimension_code']}: {row['repair_reason']}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build honest V2 semantic repair actions for invalid MD&A evidence rows.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_mda_semantic_repair_v2(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
