from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json
from stability_common import write_markdown


def diagnose_mda_stability_failures(root: Path = SCORING_ROOT) -> dict[str, Any]:
    batch_dir = root / "06_ratings" / "mda_v2_stability_batch"
    selection = read_csv_rows(batch_dir / "mda_v2_stability_batch_selection.csv")
    scores = read_csv_rows(batch_dir / "mda_v2_stability_document_scores.csv")
    ratings = read_csv_rows(batch_dir / "mda_v2_stability_ratings_long.csv")
    numeric_rows = {
        row.get("document_id", ""): row
        for row in read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv")
    }
    selected_ids = [row.get("document_id", "") for row in selection if row.get("document_id")]
    scored_ids = {row.get("document_id", "") for row in scores}
    ratings_by_doc: dict[str, list[dict[str, str]]] = {}
    for row in ratings:
        ratings_by_doc.setdefault(row.get("document_id", ""), []).append(row)

    reason_counter: Counter[str] = Counter()
    failed_documents: list[dict[str, Any]] = []
    dimension_mismatch_count = 0
    evidence_failure_count = 0
    ar02_issue_count = 0
    json_failure_count = 0
    schema_failure_count = 0

    for doc_id in selected_ids:
        per_doc = _load_per_document(batch_dir / "per_document" / f"{doc_id}_parsed.json")
        doc_ratings = ratings_by_doc.get(doc_id, [])
        numeric = numeric_rows.get(doc_id, {})
        doc_reasons: list[str] = []

        if doc_id not in scored_ids or per_doc.get("status") != "success":
            result = per_doc.get("result", {}) if isinstance(per_doc.get("result"), dict) else {}
            text = " ".join(str(result.get(field, "")) for field in ["error_type", "error_message", "parse_status", "schema_validation_status"]).lower()
            if "json" in text or result.get("parse_status") == "parse_failed":
                doc_reasons.append("json_parse_failed")
                json_failure_count += 1
            if "schema" in text or result.get("schema_validation_status") == "invalid":
                doc_reasons.append("schema_validation_failed")
                schema_failure_count += 1
            if "dimension_count" in text or "dimension_codes" in text or "missing_dimension" in text:
                doc_reasons.append("dimension_mismatch")
                dimension_mismatch_count += 1
            if not doc_reasons:
                doc_reasons.append(result.get("error_type") or "scoring_failed")

        if doc_ratings and len(doc_ratings) != 5:
            doc_reasons.append("dimension_mismatch")
            dimension_mismatch_count += 1
        if doc_ratings and any(not row.get("evidence_locator", "").strip() for row in doc_ratings):
            doc_reasons.append("evidence_locator_failed")
            evidence_failure_count += 1
        if _numeric_has_issue(numeric):
            ar02_issue_count += 1
            if doc_id not in scored_ids:
                doc_reasons.append("ar02_numeric_inconsistency_present")

        for reason in sorted(set(doc_reasons)):
            reason_counter[reason] += 1
        if doc_reasons:
            failed_documents.append(
                {
                    "document_id": doc_id,
                    "status": per_doc.get("status", "missing_per_document"),
                    "reasons": sorted(set(doc_reasons)),
                    "error_message": str((per_doc.get("result") or {}).get("error_message", ""))[:500] if isinstance(per_doc.get("result"), dict) else "",
                }
            )

    total = len(selected_ids)
    success_count = len(scored_ids & set(selected_ids))
    summary = {
        "selected_count": total,
        "success_count": success_count,
        "failure_count": max(0, total - success_count),
        "success_rate": round(success_count / total, 6) if total else 0.0,
        "json_failure_rate": round(json_failure_count / total, 6) if total else 0.0,
        "schema_failure_rate": round(schema_failure_count / total, 6) if total else 0.0,
        "ar02_inconsistency_rate": round(ar02_issue_count / total, 6) if total else 0.0,
        "evidence_locator_failure_rate": round(evidence_failure_count / total, 6) if total else 0.0,
        "dimension_mismatch_rate": round(dimension_mismatch_count / total, 6) if total else 0.0,
        "top_root_causes": [
            {"reason": reason, "count": count, "share_of_batch": round(count / total, 6) if total else 0.0}
            for reason, count in reason_counter.most_common(3)
        ],
        "failed_documents": failed_documents,
        "recommended_fix": _recommend_fix(reason_counter),
    }
    save_json(root / "08_reports" / "mda_stability_failure_root_cause.json", summary)
    _write_report(root / "08_reports" / "mda_stability_failure_root_cause.md", summary)
    return summary


def _load_per_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing_per_document", "result": {"error_type": "missing_per_document"}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "result": {"error_type": "per_document_json_invalid", "parse_status": "parse_failed"}}


def _numeric_has_issue(row: dict[str, str]) -> bool:
    try:
        material = int(float(row.get("num_material_discrepancies") or 0))
        severe = int(float(row.get("num_severe_direction_conflicts") or 0))
    except ValueError:
        return False
    return material > 0 or severe > 0


def _recommend_fix(counter: Counter[str]) -> str:
    if counter.get("json_parse_failed") or counter.get("schema_validation_failed"):
        return "rerun failed MD&A documents through single-dimension scoring with temperature=0; parser repair cannot create missing dimension scores."
    if counter.get("ar02_numeric_inconsistency_present"):
        return "review numeric checker outputs and AR02 review flags; do not ask LLM to calculate."
    if counter.get("evidence_locator_failed"):
        return "repair paragraph mapping before scoring."
    return "manual_required"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MD&A Stability Failure Root Cause",
        "",
        f"- selected_count: {summary['selected_count']}",
        f"- success_count: {summary['success_count']}",
        f"- failure_count: {summary['failure_count']}",
        f"- success_rate: {summary['success_rate']}",
        f"- JSON failure rate: {summary['json_failure_rate']}",
        f"- schema failure rate: {summary['schema_failure_rate']}",
        f"- AR02 inconsistency rate: {summary['ar02_inconsistency_rate']}",
        f"- evidence_locator failure rate: {summary['evidence_locator_failure_rate']}",
        f"- dimension mismatch rate: {summary['dimension_mismatch_rate']}",
        "",
        "## TOP 3 Root Causes",
    ]
    if summary["top_root_causes"]:
        lines.extend(f"- {item['reason']}: {item['count']} ({item['share_of_batch']})" for item in summary["top_root_causes"])
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended Fix", f"- {summary['recommended_fix']}", "", "## Failed Documents"])
    for item in summary["failed_documents"]:
        lines.append(f"- {item['document_id']}: {', '.join(item['reasons'])}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose MD&A v2 stability-batch failure root causes.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(diagnose_mda_stability_failures(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
