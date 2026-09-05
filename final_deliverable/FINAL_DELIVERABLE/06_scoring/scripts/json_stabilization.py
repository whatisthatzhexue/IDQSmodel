from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, asdict
from typing import Any


PARSER_REPAIR_FIXES = {
    "strip_markdown_and_parse",
    "strip_think_block_and_parse",
    "extract_largest_json_object",
    "repair_json",
}


@dataclass
class ParsedModelJson:
    ok: bool
    payload: Any | None
    failure_type: str
    recommended_fix: str
    has_markdown_fence: bool
    has_think_block: bool
    has_leading_text: bool
    has_trailing_text: bool
    has_multiple_json_objects: bool
    has_truncated_json: bool
    missing_opening_brace: bool
    missing_closing_brace: bool
    invalid_escape: bool
    invalid_quote: bool
    missing_comma: bool
    wrong_root_type: bool
    missing_required_fields: bool
    missing_dimension_scores: bool
    dimension_count: int
    raw_output_length: int
    parse_error: str = ""
    repaired_text: str = ""


def parse_model_json(raw_text: str) -> dict[str, Any]:
    """Parse common qwen JSON wrappers before deciding that a rescore is needed."""
    text = raw_text or ""
    stripped = text.strip()
    base = _base_result(text)
    if not stripped:
        return asdict(
            ParsedModelJson(
                **base,
                ok=False,
                payload=None,
                failure_type="empty_response",
                recommended_fix="rescore_dimensionwise",
                parse_error="empty response",
            )
        )

    attempts: list[tuple[str, str, str]] = [("direct", stripped, "")]
    markdown_text = _strip_markdown_fence(stripped)
    if markdown_text != stripped:
        attempts.append(("markdown_wrapped_json", markdown_text, "strip_markdown_and_parse"))
    think_text = _strip_think_block(stripped)
    if think_text != stripped:
        attempts.append(("think_block_before_json", think_text, "strip_think_block_and_parse"))
    markdown_then_think = _strip_think_block(markdown_text)
    if markdown_then_think != markdown_text:
        attempts.append(("think_block_before_json", markdown_then_think, "strip_think_block_and_parse"))
    think_then_markdown = _strip_markdown_fence(think_text)
    if think_then_markdown != think_text:
        attempts.append(("markdown_wrapped_json", think_then_markdown, "strip_markdown_and_parse"))

    json_objects = extract_json_objects(stripped)
    if json_objects:
        largest = max(json_objects, key=len)
        attempts.append((_object_failure_type(base), largest, _object_recommended_fix(base)))

    for failure_type, candidate, recommended_fix in attempts:
        payload, error = _loads(candidate)
        if error:
            continue
        return _success_result(base, payload, candidate, failure_type, recommended_fix)

    repaired_candidates = _repair_candidates(stripped)
    last_error = ""
    for candidate in repaired_candidates:
        payload, error = _loads(candidate)
        if not error:
            return _success_result(base, payload, candidate, "missing_comma_or_brace", "repair_json")
        last_error = error

    literal_candidate = _python_literal_candidate(stripped)
    if literal_candidate is not None:
        return _success_result(base, literal_candidate, "", "invalid_escape_or_quote", "repair_json")

    failure_type, recommended_fix = _failure_from_text(base, stripped, last_error)
    return asdict(
        ParsedModelJson(
            **base,
            ok=False,
            payload=None,
            failure_type=failure_type,
            recommended_fix=recommended_fix,
            parse_error=last_error,
        )
    )


def extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            current = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif current == "\\":
                    escape = True
                elif current == '"':
                    in_string = False
                continue
            if current == '"':
                in_string = True
            elif current == "{":
                depth += 1
            elif current == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start : idx + 1])
                    break
    return objects


def is_parser_repairable(result: dict[str, Any]) -> bool:
    return bool(result.get("ok")) and result.get("recommended_fix") in PARSER_REPAIR_FIXES


def _base_result(text: str) -> dict[str, Any]:
    stripped = text.strip()
    objects = extract_json_objects(stripped)
    largest = max(objects, key=len) if objects else ""
    leading = False
    trailing = False
    if largest:
        start = stripped.find(largest)
        end = start + len(largest)
        leading = bool(stripped[:start].strip())
        trailing = bool(stripped[end:].strip())
    missing_opening = "{" not in stripped
    missing_closing = _has_unclosed_brace(stripped)
    return {
        "has_markdown_fence": "```" in stripped,
        "has_think_block": bool(re.search(r"</?think\b", stripped, flags=re.IGNORECASE)),
        "has_leading_text": leading,
        "has_trailing_text": trailing,
        "has_multiple_json_objects": len(objects) > 1,
        "has_truncated_json": bool(stripped) and (missing_closing or (stripped.count("{") > stripped.count("}"))),
        "missing_opening_brace": missing_opening,
        "missing_closing_brace": missing_closing,
        "invalid_escape": bool(re.search(r"\\(?![\"\\/bfnrtu])", stripped)),
        "invalid_quote": _looks_like_single_quoted_json(stripped),
        "missing_comma": False,
        "wrong_root_type": False,
        "missing_required_fields": False,
        "missing_dimension_scores": False,
        "dimension_count": 0,
        "raw_output_length": len(text),
    }


def _success_result(
    base: dict[str, Any],
    payload: Any,
    repaired_text: str,
    failure_type: str,
    recommended_fix: str,
) -> dict[str, Any]:
    root_is_wrong = not isinstance(payload, dict)
    dimension_scores = payload.get("dimension_scores") if isinstance(payload, dict) else None
    dimension_count = len(dimension_scores) if isinstance(dimension_scores, list) else 0
    missing_dimension_scores = isinstance(payload, dict) and "dimension_scores" not in payload and "dimension_code" not in payload
    missing_required_fields = False
    if isinstance(payload, dict) and "dimension_code" not in payload:
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
    clean_failure_type = "" if failure_type == "direct" else failure_type
    clean_recommended = "" if failure_type == "direct" else recommended_fix
    return asdict(
        ParsedModelJson(
            **{
                **base,
                "wrong_root_type": root_is_wrong,
                "missing_required_fields": missing_required_fields,
                "missing_dimension_scores": missing_dimension_scores,
                "dimension_count": dimension_count,
            },
            ok=True,
            payload=payload,
            failure_type=clean_failure_type,
            recommended_fix=clean_recommended,
            repaired_text=repaired_text,
        )
    )


def _failure_from_text(base: dict[str, Any], text: str, error: str) -> tuple[str, str]:
    lowered = error.lower()
    base["missing_comma"] = "expecting ',' delimiter" in lowered or "expecting property name" in lowered
    if base["missing_closing_brace"] or base["has_truncated_json"]:
        return "truncated_json", "repair_json"
    if base["invalid_escape"] or base["invalid_quote"]:
        return "invalid_escape_or_quote", "repair_json"
    if base["missing_comma"]:
        return "missing_comma_or_brace", "repair_json"
    if base["missing_opening_brace"]:
        if "timeout" in text.lower() or "timed out" in text.lower():
            return "timeout_or_no_response", "rescore_dimensionwise"
        return "not_json", "rescore_dimensionwise"
    return "unknown", "rescore_dimensionwise"


def _object_failure_type(base: dict[str, Any]) -> str:
    if base["has_multiple_json_objects"]:
        return "multiple_json_objects"
    if base["has_leading_text"] or base["has_trailing_text"]:
        return "leading_or_trailing_text"
    return "leading_or_trailing_text"


def _object_recommended_fix(base: dict[str, Any]) -> str:
    if base["has_multiple_json_objects"] or base["has_leading_text"] or base["has_trailing_text"]:
        return "extract_largest_json_object"
    return "extract_largest_json_object"


def _loads(text: str) -> tuple[Any | None, str]:
    try:
        return json.loads(text), ""
    except (TypeError, json.JSONDecodeError) as exc:
        return None, str(exc)


def _strip_markdown_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _strip_think_block(text: str) -> str:
    without_blocks = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if without_blocks != text.strip():
        return without_blocks
    close_idx = text.lower().find("</think>")
    if close_idx >= 0:
        return text[close_idx + len("</think>") :].strip()
    return text


def _repair_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for source in [text, _strip_markdown_fence(text), _strip_think_block(text)]:
        extracted = max(extract_json_objects(source), key=len) if extract_json_objects(source) else source
        if "{" in extracted:
            start = extracted.find("{")
            extracted = extracted[start:]
        repaired = _repair_invalid_escapes(extracted)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        if repaired and repaired not in candidates:
            candidates.append(repaired)
    return candidates


def _repair_invalid_escapes(text: str) -> str:
    return re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", text)


def _python_literal_candidate(text: str) -> Any | None:
    candidate = _strip_markdown_fence(_strip_think_block(text)).strip()
    objects = extract_json_objects(candidate)
    if objects:
        candidate = max(objects, key=len)
    try:
        payload = ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        return None
    return payload if isinstance(payload, (dict, list)) else None


def _has_unclosed_brace(text: str) -> bool:
    depth = 0
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth = max(0, depth - 1)
    return depth > 0


def _looks_like_single_quoted_json(text: str) -> bool:
    return bool(re.search(r"\{\s*'[^']+'\s*:", text) or re.search(r":\s*'[^']+'", text))
