from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scoring_utils import DIMENSIONS, FLAG_FIELDS, SCORING_ROOT, document_total, weights_for
from scoring_utils import SCOREBOOK_VERSION, scoring_basis_for


@dataclass
class ValidationResult:
    is_valid: bool
    issues: list[str]


ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_MISSING = {"none", "structural_missing", "operational_missing"}
SCOPE_VIOLATION_PATTERNS = [
    "match the audited financial statements",
    "matches the audited financial statements",
    "matched the audited financial statements",
    "audited financial statements match",
    "matches the financial statements",
    "full annual report quality",
]


def normalize_scoring_payload(payload: dict[str, Any], doc_type: str, document_id: str) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["document_id"] = document_id
    normalized["doc_type"] = doc_type
    normalized["scoring_basis"] = scoring_basis_for(doc_type)
    normalized["scorebook_version"] = SCOREBOOK_VERSION
    dimension_scores = []
    for item in normalized.get("dimension_scores", []):
        if not isinstance(item, dict):
            dimension_scores.append(item)
            continue
        copied = dict(item)
        copied["evidence_locator"] = _normalize_evidence_locator(str(copied.get("evidence_locator", "")))
        dimension_scores.append(copied)
    normalized["dimension_scores"] = dimension_scores
    return normalized


def validate_scoring_payload(
    payload: dict[str, Any] | str,
    input_text: str,
    expected_doc_type: str | None = None,
) -> ValidationResult:
    issues: list[str] = []
    if isinstance(payload, str):
        stripped = payload.strip()
        if stripped.startswith("```"):
            issues.append("markdown_wrapped_json")
        try:
            payload = json.loads(_strip_markdown_fence(stripped))
        except json.JSONDecodeError:
            return ValidationResult(False, issues + ["json_parse_failed"])
    if not isinstance(payload, dict):
        return ValidationResult(False, issues + ["payload_not_object"])

    doc_type = payload.get("doc_type")
    if expected_doc_type and doc_type != expected_doc_type:
        issues.append("doc_type_mismatch")
    if doc_type not in {"mda", "news"}:
        issues.append("invalid_doc_type")
        doc_type = expected_doc_type or "mda"

    required_top = [
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
    ]
    for key in required_top:
        if key not in payload:
            issues.append(f"missing_top_field_{key}")

    dimension_scores = payload.get("dimension_scores", [])
    if not isinstance(dimension_scores, list):
        issues.append("dimension_scores_not_list")
        dimension_scores = []
    if len(dimension_scores) != 5:
        issues.append("dimension_count_not_5")

    expected_codes = [item["code"] for item in DIMENSIONS.get(doc_type, [])]
    seen_codes = []
    for item in dimension_scores:
        if not isinstance(item, dict):
            issues.append("dimension_entry_not_object")
            continue
        seen_codes.append(str(item.get("dimension_code", "")))
        _validate_dimension_item(item, input_text, issues)

    if expected_codes and seen_codes != expected_codes:
        issues.append("dimension_codes_not_expected_order")

    flags = payload.get("flags", {})
    if not isinstance(flags, dict):
        issues.append("flags_not_object")
        flags = {}
    for field in FLAG_FIELDS:
        if flags.get(field) not in {0, 1}:
            issues.append(f"invalid_flag_{field}")

    if doc_type == "mda":
        combined_text = json.dumps(payload, ensure_ascii=False).lower()
        for pattern in SCOPE_VIOLATION_PATTERNS:
            if pattern in combined_text:
                issues.append("mda_scope_violation_audited_financial_statement_match")
                break

    if _has_unrelated_chatter(payload):
        issues.append("model_output_unrelated_chatter")

    return ValidationResult(len(issues) == 0, sorted(set(issues)))


def validate_total_score(
    dimension_scores: list[dict[str, Any]],
    doc_type: str,
    observed_total: float | int | str,
) -> ValidationResult:
    raw_scores = {item["dimension_code"]: int(item["raw_score"]) for item in dimension_scores}
    expected = document_total(raw_scores, weights_for(doc_type))
    if abs(float(observed_total) - expected) > 0.000001:
        return ValidationResult(False, ["total_score_formula_mismatch"])
    return ValidationResult(True, [])


def _validate_dimension_item(item: dict[str, Any], input_text: str, issues: list[str]) -> None:
    required = [
        "dimension_code",
        "dimension_name",
        "raw_score",
        "reason",
        "evidence_locator",
        "evidence_text_short",
        "confidence_level",
        "missing_type",
        "comment_short",
    ]
    for key in required:
        if key not in item:
            issues.append(f"missing_dimension_field_{key}")
    raw_score = item.get("raw_score")
    if raw_score not in {1, 2, 3, 4, 5}:
        issues.append("raw_score_out_of_range")
    if item.get("confidence_level") not in ALLOWED_CONFIDENCE:
        issues.append("invalid_confidence_level")
    if item.get("missing_type") not in ALLOWED_MISSING:
        issues.append("invalid_missing_type")

    evidence_locator = str(item.get("evidence_locator", "")).strip()
    evidence_text = str(item.get("evidence_text_short", "")).strip()
    if not evidence_locator:
        issues.append("empty_evidence_locator")
    if not evidence_text:
        issues.append("empty_evidence_text_short")
    if evidence_locator and not _locator_exists(evidence_locator, input_text):
        issues.append("evidence_locator_not_found")
    if raw_score in {4, 5} and (not evidence_locator or len(evidence_text) < 12):
        issues.append("high_score_without_specific_evidence")
    if raw_score == 1 and _looks_like_positive_coverage(item):
        issues.append("raw_score_reason_contradiction")


def _locator_exists(locator: str, input_text: str) -> bool:
    locators = re.findall(r"\[[A-Z0-9_]+\]", locator)
    if not locators:
        return False
    return all(item in input_text for item in locators)


def _normalize_evidence_locator(locator: str) -> str:
    tokens = re.findall(r"\[?[A-Z]+_P\d{3}\]?|\[TITLE\]|\[LEAD\]", locator)
    if not tokens:
        return locator.strip()
    normalized = []
    for token in tokens:
        bare = token.strip("[]")
        normalized.append(f"[{bare}]")
    return ", ".join(normalized)


def _strip_markdown_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    return text


def _has_unrelated_chatter(payload: dict[str, Any]) -> bool:
    chatter = ["as an ai language model", "i cannot", "here is the json"]
    combined = json.dumps(payload, ensure_ascii=False).lower()
    return any(item in combined for item in chatter)


def _looks_like_positive_coverage(item: dict[str, Any]) -> bool:
    combined = " ".join(
        str(item.get(key, ""))
        for key in ["reason", "comment_short"]
    ).lower()
    positive_terms = [
        "covers",
        "comprehensive",
        "detailed",
        "clear",
        "specific",
        "good",
        "strong",
        "adequate",
        "includes revenue",
        "includes sections",
        "provides a comprehensive",
    ]
    return any(re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", combined) for term in positive_terms)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate one scoring JSON output.")
    parser.add_argument("json_path")
    parser.add_argument("input_text_path")
    parser.add_argument("--doc-type", choices=["mda", "news"])
    args = parser.parse_args()
    payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    input_text = Path(args.input_text_path).read_text(encoding="utf-8", errors="replace")
    result = validate_scoring_payload(payload, input_text, args.doc_type)
    print(json.dumps({"is_valid": result.is_valid, "issues": result.issues}, indent=2))
    return 0 if result.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
