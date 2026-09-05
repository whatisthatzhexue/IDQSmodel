from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import RATINGS_LONG_HEADERS, SCORING_ROOT, read_csv_rows, write_csv_rows
from stability_common import safe_float, write_markdown


AUDIT_HEADERS = [
    "rating_id",
    "document_id",
    "dimension_code",
    "original_model_raw_score",
    "adjusted_raw_score",
    "adjustment_applied",
    "adjustment_rule",
    "adjustment_reason",
    "numeric_checker_result",
    "raw_output_path",
    "numeric_check_path",
    "comment_short",
]


def audit_mda_ar02_adjustments(
    root: Path = SCORING_ROOT,
    ratings_path: Path | None = None,
) -> dict[str, Any]:
    ratings_path = ratings_path or root / "06_ratings" / "mda_final_v2" / "mda_final_ratings_long.csv"
    rows = read_csv_rows(ratings_path)
    enriched_rows = []
    audit_rows = []
    adjustment_count = 0
    ar02_count = 0

    for row in rows:
        out = {field: row.get(field, "") for field in RATINGS_LONG_HEADERS}
        if row.get("doc_type", "mda").lower() == "mda" and row.get("dimension_code") == "AR02":
            ar02_count += 1
            original_score = _model_raw_score(root, row.get("raw_output_path", ""))
            adjusted_score = _format_score(row.get("raw_score", ""))
            numeric_payload = _load_json(root, row.get("numeric_check_path", ""))
            numeric_result = _numeric_checker_result(numeric_payload)
            adjustment_applied = _score_changed(original_score, adjusted_score)
            if adjustment_applied:
                adjustment_count += 1
            adjustment_rule = _adjustment_rule(adjustment_applied, numeric_payload)
            adjustment_reason = _adjustment_reason(numeric_payload, row)
            out.update(
                {
                    "original_model_raw_score": original_score,
                    "adjusted_raw_score": adjusted_score,
                    "adjustment_applied": str(adjustment_applied).lower(),
                    "adjustment_rule": adjustment_rule,
                    "adjustment_reason": adjustment_reason,
                    "numeric_checker_result": numeric_result,
                }
            )
            audit_rows.append({field: out.get(field, row.get(field, "")) for field in AUDIT_HEADERS})
        else:
            for field in [
                "original_model_raw_score",
                "adjusted_raw_score",
                "adjustment_applied",
                "adjustment_rule",
                "adjustment_reason",
                "numeric_checker_result",
            ]:
                out.setdefault(field, "")
        enriched_rows.append(out)

    if ratings_path.exists():
        write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, enriched_rows)
    audit_path = root / "08_reports" / "mda_ar02_adjustment_audit.csv"
    write_csv_rows(audit_path, AUDIT_HEADERS, audit_rows)
    _write_report(root, ratings_path, ar02_count, adjustment_count)
    return {
        "ratings_path": str(ratings_path),
        "audit_path": str(audit_path),
        "ar02_rows": ar02_count,
        "adjustment_count": adjustment_count,
    }


def _model_raw_score(root: Path, raw_output_path: str) -> str:
    payload = _load_json(root, raw_output_path)
    if not payload:
        return ""
    candidates = [
        ("parsed_simple", "score"),
        ("parsed", "score"),
        ("result", "score"),
        ("result", "raw_score"),
        ("dimension_score",),
        ("score",),
        ("raw_score",),
    ]
    for path in candidates:
        value: Any = payload
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        score = _format_score(value)
        if score:
            return score
    content = payload.get("content") if isinstance(payload, dict) else None
    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            return _model_raw_score_from_payload(json.loads(content))
        except json.JSONDecodeError:
            return ""
    return _model_raw_score_from_payload(payload)


def _model_raw_score_from_payload(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in ["score", "raw_score", "dimension_score"]:
            score = _format_score(payload.get(key))
            if score:
                return score
        for value in payload.values():
            score = _model_raw_score_from_payload(value)
            if score:
                return score
    if isinstance(payload, list):
        for value in payload:
            score = _model_raw_score_from_payload(value)
            if score:
                return score
    return ""


def _load_json(root: Path, path_text: str) -> dict[str, Any]:
    if not path_text:
        return {}
    first_path = str(path_text).split(";", 1)[0].strip()
    path = Path(first_path)
    if not path.is_absolute():
        path = root / path
    if not path.exists() or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _numeric_checker_result(payload: dict[str, Any]) -> str:
    if not payload:
        return ""
    if isinstance(payload.get("result"), str):
        return payload["result"]
    parts = []
    for key in [
        "overall_numeric_consistency",
        "num_material_discrepancies",
        "num_minor_discrepancies",
        "num_severe_direction_conflicts",
        "num_financial_mentions",
        "num_not_calculable",
    ]:
        if key in payload:
            parts.append(f"{key}={payload.get(key)}")
    if payload.get("checks"):
        parts.append(f"checks={len(payload.get('checks') or [])}")
    return "; ".join(parts) or json.dumps(payload, ensure_ascii=False, sort_keys=True)[:500]


def _adjustment_rule(adjusted: bool, payload: dict[str, Any]) -> str:
    if not adjusted:
        return "none"
    if payload:
        return str(payload.get("rule") or "numeric_checker_postprocess")
    return "score_postprocess_unknown"


def _adjustment_reason(payload: dict[str, Any], row: dict[str, str]) -> str:
    for key in ["reason", "adjustment_reason"]:
        if payload.get(key):
            return str(payload[key])
    notes = payload.get("internal_check_notes")
    if isinstance(notes, list) and notes:
        return "; ".join(str(item) for item in notes[:3])
    if payload.get("overall_numeric_consistency"):
        return f"overall_numeric_consistency={payload['overall_numeric_consistency']}"
    return row.get("comment_short", "")


def _score_changed(original_score: str, adjusted_score: str) -> bool:
    if not original_score or not adjusted_score:
        return False
    return safe_float(original_score, -999) != safe_float(adjusted_score, -999)


def _format_score(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not 1 <= number <= 5:
        return ""
    if number.is_integer():
        return str(int(number))
    return str(number).rstrip("0").rstrip(".")


def _write_report(root: Path, ratings_path: Path, ar02_count: int, adjustment_count: int) -> None:
    write_markdown(
        root / "08_reports" / "mda_ar02_adjustment_audit.md",
        [
            "# MDA AR02 Adjustment Audit",
            "",
            f"- ratings_path: {ratings_path}",
            f"- ar02_rows: {ar02_count}",
            f"- adjustment_applied_rows: {adjustment_count}",
            "",
            "The `raw_score` column is retained for backward compatibility. For AR02, "
            "`original_model_raw_score` records the score parsed from the raw qwen output, "
            "while `adjusted_raw_score` records the score used in final tables after numeric-checker postprocessing.",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MDA AR02 model score vs numeric-checker adjusted score.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--ratings-path", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(audit_mda_ar02_adjustments(args.root, args.ratings_path), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
