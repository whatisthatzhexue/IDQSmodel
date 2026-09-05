from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


AUDIT_HEADERS = [
    "document_id",
    "paragraph_id",
    "raw_text",
    "metric_name",
    "current_value",
    "previous_value",
    "stated_change_value",
    "calculated_change_pct",
    "difference_pct_points",
    "original_consistency_status",
    "audit_label",
    "likely_reason",
    "needs_manual_review",
    "recommended_numeric_checker_fix",
]


def audit_ar02_numeric_discrepancies(root: Path = SCORING_ROOT, sample_size: int = 30) -> dict[str, Any]:
    checks_dir = root / "03_numeric_checks" / "mda_v2"
    discrepancies = []
    for path in sorted(checks_dir.glob("*_numeric_checks.json")):
        payload = _load_json(path)
        for check in payload.get("checks", []):
            if check.get("status") == "material_discrepancy":
                discrepancies.append(_audit_row(payload.get("document_id", path.name.split("_numeric_checks")[0]), check))
    total = len(discrepancies)
    sample = discrepancies[: min(sample_size, total)] if total > sample_size else discrepancies
    write_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_discrepancy_audit.csv", AUDIT_HEADERS, sample)
    report_path = root / "08_reports" / "ar02_numeric_discrepancy_audit_report.md"
    _write_report(report_path, total, sample)
    counts = Counter(row["audit_label"] for row in sample)
    return {
        "material_discrepancy_total": total,
        "sample_count": len(sample),
        "audit_label_counts": dict(counts),
        "numeric_checker_needs_fix": any(row["audit_label"] == "parser_false_positive" for row in sample),
        "report_path": str(report_path),
    }


def _audit_row(document_id: str, check: dict[str, Any]) -> dict[str, Any]:
    label, reason, fix = _classify_discrepancy(check)
    current = check.get("current_value", "")
    previous = check.get("previous_value", "")
    stated = check.get("stated_pct", "")
    calculated = check.get("calculated_change_pct", "")
    diff = _difference(stated, calculated)
    return {
        "document_id": document_id,
        "paragraph_id": check.get("evidence_locator", ""),
        "raw_text": check.get("source_text", ""),
        "metric_name": check.get("metric", ""),
        "current_value": current,
        "previous_value": previous,
        "stated_change_value": stated,
        "calculated_change_pct": calculated,
        "difference_pct_points": diff,
        "original_consistency_status": check.get("status", ""),
        "audit_label": label,
        "likely_reason": reason,
        "needs_manual_review": "1" if label == "needs_manual_review" else "0",
        "recommended_numeric_checker_fix": fix,
    }


def _classify_discrepancy(check: dict[str, Any]) -> tuple[str, str, str]:
    text = " ".join([str(check.get("source_text", "")), str(check.get("note", "")), str(check.get("metric", ""))]).lower()
    previous = _float(check.get("previous_value"))
    calculated = _float(check.get("calculated_change_pct"))
    stated = _float(check.get("stated_pct"))
    if "segment" in text and ("components sum" in text or "total" in text):
        return "parser_false_positive", "segment_total_mismatch", "mark large segment-table pairings as needs_manual_review"
    if "rm'000" in text or "rm000" in text or "million" in text or "billion" in text:
        if abs(calculated) > 500:
            return "parser_false_positive", "unit_mismatch", "normalize RM'000/RM million/RM billion before comparison"
    if "percentage point" in text:
        return "parser_false_positive", "percentage_points_vs_percent", "compare point delta instead of percent growth"
    if previous is not None and previous < 0:
        return "needs_manual_review", "negative_base", "mark previous negative base as special_case_negative_base"
    if "loss" in text and "profit" in text:
        return "needs_manual_review", "profit_loss_swing", "do not force ordinary growth rate on profit/loss swing"
    if len(set(__import__("re").findall(r"\b20\d{2}\b", text))) >= 3:
        return "parser_false_positive", "multiple_years_confusion", "mark multi-year extraction as needs_manual_review"
    if stated is not None and calculated is not None and abs(abs(stated) - abs(calculated)) <= 3:
        return "parser_false_positive", "rounding", "relax rounding tolerance or preserve sign context"
    return "real_discrepancy", "true_math_error", "none"


def _write_report(path: Path, total: int, sample: list[dict[str, Any]]) -> None:
    counts = Counter(row["audit_label"] for row in sample)
    denom = max(1, len(sample))
    parser_ratio = counts.get("parser_false_positive", 0) / denom
    lines = [
        "# AR02 Numeric Discrepancy Audit Report",
        "",
        f"- material discrepancy total: {total}",
        f"- sampled discrepancies: {len(sample)}",
        f"- real_discrepancy ratio: {counts.get('real_discrepancy', 0) / denom:.3f}",
        f"- parser_false_positive ratio: {parser_ratio:.3f}",
        f"- needs_manual_review ratio: {counts.get('needs_manual_review', 0) / denom:.3f}",
        f"- numeric checker needs revision: {parser_ratio > 0}",
        f"- AR02 full scoring impact: {'review numeric checker outputs before full scoring' if parser_ratio > 0 else 'no broad parser issue detected in sample'}",
        "",
        "## Label Counts",
        *[f"- {key}: {value}" for key, value in sorted(counts.items())],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _difference(stated: Any, calculated: Any) -> str:
    left = _float(stated)
    right = _float(calculated)
    if left is None or right is None:
        return ""
    return round(abs(abs(left) - abs(right)), 6)


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AR02 material numeric discrepancies.")
    parser.add_argument("--root", default=str(SCORING_ROOT))
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()
    print(audit_ar02_numeric_discrepancies(Path(args.root), args.sample_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
