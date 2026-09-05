from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, document_total, read_csv_rows, weights_for, write_csv_rows


AUDIT_HEADERS = [
    "document_id",
    "dimension_code",
    "raw_score_v1",
    "evidence_locator_v1",
    "evidence_valid_in_clean_v2",
    "scope_violation_flag",
    "ar02_external_verification_claim_flag",
    "numeric_check_conflict_flag",
    "empty_evidence_flag",
    "score_recalculation_error_flag",
    "needs_review",
    "review_reasons",
]


def locator_exists(locator: str, text: str) -> bool:
    locators = re.findall(r"\[MDA_P\d{3}\]", locator or "")
    if not locators:
        return False
    return all(item in text for item in locators)


def audit_dimension_row(row: dict[str, str], cleaned_text: str, numeric_summary: dict[str, Any], document_total_error: bool = False) -> dict[str, Any]:
    combined = " ".join(str(row.get(field, "")) for field in ["reason", "comment_short", "evidence_text_short"]).lower()
    dimension = row.get("dimension_code", "")
    raw_score = row.get("raw_score", "")
    evidence = row.get("evidence_locator", "")
    evidence_valid = locator_exists(evidence, cleaned_text)
    external_claim = int(dimension == "AR02" and bool(re.search(r"matches? (the )?(audited )?financial statements?", combined)))
    scope_violation = int(bool(re.search(r"audit report|corporate governance|notes to the financial statements|full annual report", combined)))
    empty_evidence = int(not evidence.strip() or not row.get("evidence_text_short", "").strip())
    numeric_conflict = int(dimension == "AR02" and int(float(raw_score or 0)) >= 4 and (int(numeric_summary.get("num_material_discrepancies", 0)) > 0 or int(numeric_summary.get("num_severe_direction_conflicts", 0)) > 0))
    reasons = []
    try:
        raw_int = int(float(raw_score))
        if raw_int < 1 or raw_int > 5:
            reasons.append("raw_score_out_of_range")
    except ValueError:
        reasons.append("raw_score_invalid")
    if not evidence_valid:
        reasons.append("evidence_locator_invalid")
    if external_claim:
        reasons.append("ar02_external_verification_claim")
    if scope_violation:
        reasons.append("scope_violation")
    if empty_evidence:
        reasons.append("empty_evidence")
    if numeric_conflict:
        reasons.append("numeric_check_conflict")
    if row.get("confidence_level", "").lower() == "low":
        reasons.append("low_confidence")
    if document_total_error:
        reasons.append("score_recalculation_error")
    if dimension == "AR02" and "not found in annual report" in combined:
        reasons.append("possible_mda_scope_violation")
    return {
        "document_id": row.get("document_id", ""),
        "dimension_code": dimension,
        "raw_score_v1": raw_score,
        "evidence_locator_v1": evidence,
        "evidence_valid_in_clean_v2": int(evidence_valid),
        "scope_violation_flag": scope_violation,
        "ar02_external_verification_claim_flag": external_claim,
        "numeric_check_conflict_flag": numeric_conflict,
        "empty_evidence_flag": empty_evidence,
        "score_recalculation_error_flag": int(document_total_error),
        "needs_review": int(bool(reasons)),
        "review_reasons": ";".join(reasons),
    }


def audit_mda_initial_scores(root: Path = SCORING_ROOT) -> dict[str, Any]:
    archive = root / "archive" / "v1_initial_mda_scoring"
    ratings = read_csv_rows(archive / "mda_ratings_long.csv")
    docs = {row["document_id"]: row for row in read_csv_rows(archive / "mda_document_scores.csv")}
    numeric = {row["document_id"]: row for row in read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv")}
    grouped: dict[str, dict[str, int]] = {}
    for row in ratings:
        try:
            grouped.setdefault(row["document_id"], {})[row["dimension_code"]] = int(float(row["raw_score"]))
        except (ValueError, KeyError):
            pass
    total_errors: set[str] = set()
    weights = weights_for("mda")
    for doc_id, scores in grouped.items():
        if set(scores) >= {"AR01", "AR02", "AR03", "AR04", "AR05"}:
            total = document_total(scores, weights)
            original = float(docs.get(doc_id, {}).get("total_score_100", total))
            if abs(total - original) > 0.01:
                total_errors.add(doc_id)
    audit_rows = []
    for row in ratings:
        doc_id = row.get("document_id", "")
        cleaned_path = root / "02_extracted_text" / "mda_clean_v2" / f"{doc_id}.txt"
        cleaned_text = cleaned_path.read_text(encoding="utf-8", errors="replace") if cleaned_path.exists() else ""
        audit_rows.append(audit_dimension_row(row, cleaned_text, numeric.get(doc_id, {}), doc_id in total_errors))
    out_dir = root / "06_ratings" / "mda_v1_audit"
    write_csv_rows(out_dir / "mda_initial_score_audit.csv", AUDIT_HEADERS, audit_rows)
    needs = sum(int(row["needs_review"]) for row in audit_rows)
    report = out_dir / "mda_initial_score_audit_report.md"
    report.write_text(f"# MDA Initial Score Audit Report\n\n- audited_dimension_rows: {len(audit_rows)}\n- needs_review_rows: {needs}\n", encoding="utf-8")
    return {"audited_dimension_rows": len(audit_rows), "needs_review_rows": needs}


def main() -> int:
    print(audit_mda_initial_scores())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
