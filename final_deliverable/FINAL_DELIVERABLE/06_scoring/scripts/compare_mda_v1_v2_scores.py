from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


COMPARISON_HEADERS = [
    "document_id",
    "AR01_v1", "AR01_v2", "AR01_diff",
    "AR02_v1", "AR02_v2", "AR02_diff",
    "AR03_v1", "AR03_v2", "AR03_diff",
    "AR04_v1", "AR04_v2", "AR04_diff",
    "AR05_v1", "AR05_v2", "AR05_diff",
    "total_score_v1", "total_score_v2", "total_score_diff",
    "max_dimension_diff", "material_change_flag", "evidence_locator_invalid_flag", "full_rescore_recommended", "reason",
]


def compare_score_rows(v1: dict[str, str], v2: dict[str, str]) -> dict[str, Any]:
    row: dict[str, Any] = {"document_id": v1.get("document_id") or v2.get("document_id")}
    diffs = []
    for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
        a = int(float(v1.get(code, 0) or 0))
        b = int(float(v2.get(code, 0) or 0))
        diff = b - a
        row[f"{code}_v1"] = a
        row[f"{code}_v2"] = b
        row[f"{code}_diff"] = diff
        diffs.append(abs(diff))
    total_v1 = float(v1.get("total_score_100", 0) or 0)
    total_v2 = float(v2.get("total_score_100", 0) or 0)
    total_diff = total_v2 - total_v1
    row["total_score_v1"] = total_v1
    row["total_score_v2"] = total_v2
    row["total_score_diff"] = total_diff
    row["max_dimension_diff"] = max(diffs)
    row["material_change_flag"] = int(max(diffs) >= 2 or abs(total_diff) >= 15)
    row["full_rescore_recommended"] = 0
    row["reason"] = "material dimension/total change" if row["material_change_flag"] else "no material change"
    row["evidence_locator_invalid_flag"] = int(v2.get("evidence_locator_invalid_flag", 0) or 0)
    return row


def full_rescore_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"full_rescore_recommended": 0, "reason": "no v2 sample comparison rows"}
    material_ratio = sum(int(row.get("material_change_flag", 0)) for row in rows) / len(rows)
    ar02_avg_diff = statistics.mean(abs(float(row.get("AR02_diff", 0))) for row in rows)
    evidence_fail_ratio = sum(int(row.get("evidence_locator_invalid_flag", 0)) for row in rows) / len(rows)
    reasons = []
    if material_ratio >= 0.20:
        reasons.append(f"material change ratio {material_ratio:.2f} >= 0.20")
    if ar02_avg_diff >= 1:
        reasons.append(f"AR02 average difference {ar02_avg_diff:.2f} >= 1")
    if evidence_fail_ratio >= 0.20:
        reasons.append(f"evidence locator failure ratio {evidence_fail_ratio:.2f} >= 0.20")
    return {"full_rescore_recommended": int(bool(reasons)), "reason": "; ".join(reasons) or "v1/v2 differences are small"}


def compare_mda_v1_v2_scores(root: Path = SCORING_ROOT) -> dict[str, Any]:
    archive = root / "archive" / "v1_initial_mda_scoring"
    v1_rows = {row["document_id"]: row for row in read_csv_rows(archive / "mda_document_scores.csv")}
    v2_rows = {row["document_id"]: row for row in read_csv_rows(root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_document_scores_sample.csv")}
    comparisons = [compare_score_rows(v1_rows[doc], v2_rows[doc]) for doc in sorted(v1_rows.keys() & v2_rows.keys())]
    decision = full_rescore_decision(comparisons)
    for row in comparisons:
        row["full_rescore_recommended"] = decision["full_rescore_recommended"]
        row["reason"] = decision["reason"] if row["material_change_flag"] else row["reason"]
    out = root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v1_v2_score_comparison.csv"
    write_csv_rows(out, COMPARISON_HEADERS, comparisons)
    report = root / "08_reports" / "mda_v1_v2_comparison_report.md"
    report.write_text(f"# MDA v1 v2 Comparison Report\n\n- compared_documents: {len(comparisons)}\n- full_rescore_recommended: {decision['full_rescore_recommended']}\n- reason: {decision['reason']}\n", encoding="utf-8")
    return {"compared_documents": len(comparisons), **decision}


def main() -> int:
    print(compare_mda_v1_v2_scores())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
