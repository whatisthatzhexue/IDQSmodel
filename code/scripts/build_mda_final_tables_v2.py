from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scoring_utils import (
    MDA_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    SCORING_ROOT,
    grade_label,
    read_csv_rows,
    standardize_score,
    weighted_score,
    weights_for,
    write_csv_rows,
)
from stability_common import safe_float, write_markdown


AUDIT_V2_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "report_year",
    "dimension_code",
    "raw_score",
    "evidence_locator",
    "evidence_exists",
    "evidence_text_short",
    "evidence_semantic_status",
    "evidence_issue_type",
    "score_evidence_alignment",
    "needs_review",
    "recommended_action",
    "repair_status",
    "repair_reason",
    "numeric_issue_status",
]


def build_mda_final_tables_v2(root: Path = SCORING_ROOT) -> dict[str, Any]:
    base_ratings = read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv")
    base_docs = {
        row.get("document_id", ""): row
        for row in read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv")
    }
    replacements = {
        (row.get("document_id", ""), row.get("dimension_code", "")): row
        for row in read_csv_rows(root / "06_ratings" / "mda_semantic_repair_v2" / "mda_semantic_repair_v2_replacement_rows.csv")
    }
    actions = {
        (row.get("document_id", ""), row.get("dimension_code", "")): row
        for row in read_csv_rows(root / "08_reports" / "mda_semantic_repair_v2_actions.csv")
    }
    audit_source_rows = read_csv_rows(root / "08_reports" / "mda_semantic_evidence_audit_v2.csv") or read_csv_rows(root / "08_reports" / "mda_semantic_evidence_audit.csv")
    audit_v1 = {
        (row.get("document_id", ""), row.get("dimension_code", "")): row
        for row in audit_source_rows
    }
    merged_ratings = []
    for row in base_ratings:
        key = (row.get("document_id", ""), row.get("dimension_code", ""))
        merged = dict(replacements.get(key, row))
        merged["raw_output_path"] = _relative(root, merged.get("raw_output_path", ""))
        merged["numeric_check_path"] = _relative(root, merged.get("numeric_check_path", ""))
        merged_ratings.append(merged)

    audit_rows = _audit_v2_rows(merged_ratings, base_docs, audit_v1, actions)
    review_rows = [row for row in audit_rows if row["needs_review"] == "true"]
    document_rows = _document_scores(root, merged_ratings, base_docs, audit_rows)

    out_dir = root / "06_ratings" / "mda_final_v2"
    write_csv_rows(out_dir / "mda_final_ratings_long.csv", RATINGS_LONG_HEADERS, merged_ratings)
    write_csv_rows(out_dir / "mda_final_document_scores.csv", MDA_DOCUMENT_HEADERS, document_rows)
    write_csv_rows(root / "08_reports" / "mda_semantic_evidence_audit_v2.csv", AUDIT_V2_HEADERS, audit_rows)
    write_csv_rows(root / "07_review" / "mda_semantic_evidence_review_pool_v2.csv", AUDIT_V2_HEADERS, review_rows)
    summary = _summary(audit_rows, merged_ratings, document_rows)
    _write_audit_report(root / "08_reports" / "mda_semantic_evidence_audit_v2.md", summary)
    write_markdown(
        root / "08_reports" / "mda_final_tables_v2_report.md",
        ["# MD&A Final Tables V2", "", *[f"- {key}: {value}" for key, value in summary.items() if not isinstance(value, dict)]],
    )
    return summary


def _audit_v2_rows(
    ratings: list[dict[str, str]],
    docs: dict[str, dict[str, str]],
    audit_v1: dict[tuple[str, str], dict[str, str]],
    actions: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, Any]]:
    rows = []
    for rating in ratings:
        key = (rating.get("document_id", ""), rating.get("dimension_code", ""))
        old = audit_v1.get(key, {})
        action = actions.get(key, {})
        status = old.get("evidence_semantic_status") or "acceptable"
        issue = old.get("evidence_issue_type") or "none"
        alignment = old.get("score_evidence_alignment") or "aligned"
        recommended = old.get("recommended_action") or "accept"
        repair_status = action.get("repair_status", "") or old.get("repair_status", "")
        repair_reason = action.get("repair_reason", "") or old.get("repair_reason", "")
        if repair_status == "repaired":
            status = action.get("new_evidence_semantic_status") or "acceptable"
            issue = action.get("new_evidence_issue_type") or "none"
            alignment = "aligned" if status in {"strong", "acceptable"} else "partial"
            recommended = "manual_review" if status == "weak" else "accept"
        elif status == "invalid":
            status = "weak"
            alignment = "partial"
            recommended = "manual_required"
            repair_status = repair_status or "manual_required"
            repair_reason = repair_reason or "unresolved_invalid_requires_manual_validation"
        numeric_status = _numeric_issue_status(rating, old)
        needs_review = status == "weak" or recommended not in {"accept", ""} or numeric_status in {"confirmed_numeric_failure", "numeric_review_required", "not_enough_numbers"} or rating.get("confidence_level", "").lower() == "low"
        doc = docs.get(key[0], {})
        rows.append(
            {
                "document_id": key[0],
                "ticker": doc.get("ticker", old.get("ticker", "")),
                "company_name": doc.get("company_name", old.get("company_name", "")),
                "report_year": doc.get("report_year", old.get("report_year", "")),
                "dimension_code": key[1],
                "raw_score": rating.get("raw_score", old.get("raw_score", "")),
                "evidence_locator": rating.get("evidence_locator", old.get("evidence_locator", "")),
                "evidence_exists": old.get("evidence_exists", "true"),
                "evidence_text_short": rating.get("evidence_text_short", old.get("evidence_text_short", "")),
                "evidence_semantic_status": status,
                "evidence_issue_type": issue,
                "score_evidence_alignment": alignment,
                "needs_review": str(needs_review).lower(),
                "recommended_action": recommended,
                "repair_status": repair_status,
                "repair_reason": repair_reason,
                "numeric_issue_status": numeric_status,
            }
        )
    return rows


def _numeric_issue_status(rating: dict[str, str], audit: dict[str, str]) -> str:
    if rating.get("dimension_code") != "AR02":
        return ""
    text = " ".join(
        [
            rating.get("comment_short", ""),
            rating.get("evidence_text_short", ""),
            rating.get("missing_type", ""),
            audit.get("evidence_issue_type", ""),
            audit.get("recommended_action", ""),
        ]
    ).lower()
    if "confirmed_numeric_failure" in text or "numeric failure" in text:
        return "confirmed_numeric_failure"
    if "numeric_review_required" in text or "manual_review" in text or "numeric review" in text:
        return "numeric_review_required"
    if "not enough numbers" in text or "not enough number" in text or "insufficient numeric" in text:
        return "not_enough_numbers"
    return "no_numeric_issue"


def _document_scores(
    root: Path,
    ratings: list[dict[str, str]],
    docs: dict[str, dict[str, str]],
    audit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    audit_by_key = {(row["document_id"], row["dimension_code"]): row for row in audit_rows}
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in ratings:
        grouped[row.get("document_id", "")][row.get("dimension_code", "")] = row
    weights = weights_for("mda")
    out = []
    for doc_id in sorted(grouped):
        dim_rows = grouped[doc_id]
        base = dict(docs.get(doc_id, {}))
        if not base:
            base = {field: "" for field in MDA_DOCUMENT_HEADERS}
            base.update({"document_id": doc_id, "doc_type": "mda"})
        total = 0.0
        review_reasons: list[str] = []
        for code, weight in weights.items():
            rating = dim_rows.get(code, {})
            score = safe_float(rating.get("raw_score", ""))
            if not rating:
                review_reasons.append(f"missing_dimension:{code}")
                continue
            base[code] = str(int(score)) if float(score).is_integer() else str(score)
            base[f"{code}_std"] = str(standardize_score(score))
            total += weighted_score(score, weight)
            audit = audit_by_key.get((doc_id, code), {})
            if audit.get("repair_status") == "manual_required":
                review_reasons.append(f"semantic_manual_required:{code}")
            elif audit.get("evidence_semantic_status") == "weak":
                review_reasons.append(f"semantic_weak:{code}")
            numeric_status = audit.get("numeric_issue_status", "")
            if numeric_status in {"confirmed_numeric_failure", "numeric_review_required", "not_enough_numbers"}:
                review_reasons.append(f"{numeric_status}:{code}")
            if rating.get("confidence_level", "").lower() == "low":
                review_reasons.append(f"low_confidence:{code}")
        base["dimension_count"] = str(len(dim_rows))
        base["total_score_100"] = str(round(total, 6))
        base["grade_label"] = grade_label(total)
        base["needs_review"] = str(bool(review_reasons)).lower()
        base["review_reasons"] = "; ".join(dict.fromkeys(review_reasons))
        base["final_version"] = "mda_final_v2_semantic_review"
        out.append(base)
    return out


def _summary(audit_rows: list[dict[str, Any]], ratings: list[dict[str, str]], docs: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["evidence_semantic_status"] for row in audit_rows)
    numeric_counts = Counter(row["numeric_issue_status"] for row in audit_rows if row["numeric_issue_status"])
    return {
        "ratings_long_count": len(ratings),
        "document_scores_count": len(docs),
        "strong_count": status_counts.get("strong", 0),
        "acceptable_count": status_counts.get("acceptable", 0),
        "weak_count": status_counts.get("weak", 0),
        "invalid_count_after_repair": status_counts.get("invalid", 0),
        "needs_review_count": sum(1 for row in audit_rows if row["needs_review"] == "true"),
        "manual_required_count": sum(1 for row in audit_rows if row["repair_status"] == "manual_required"),
        "confirmed_numeric_failure_count": numeric_counts.get("confirmed_numeric_failure", 0),
        "numeric_review_required_count": numeric_counts.get("numeric_review_required", 0),
        "not_enough_numbers_count": numeric_counts.get("not_enough_numbers", 0),
        "no_numeric_issue_count": numeric_counts.get("no_numeric_issue", 0),
    }


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


def _write_audit_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# MD&A Semantic Evidence Audit V2", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final MD&A V2 ratings/document tables from targeted semantic repair actions.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_mda_final_tables_v2(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
