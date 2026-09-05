from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from stability_common import (
    discover_inputs,
    ensure_stability_dirs,
    fmt,
    safe_float,
    stability_root,
    try_make_hist,
    write_empty_csv,
    write_markdown,
    write_plot_skip,
)


VERSION_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "report_year",
    "score_version_a",
    "score_version_b",
    "total_score_a",
    "total_score_b",
    "total_score_diff",
    "abs_total_score_diff",
    "AR01_diff",
    "AR02_diff",
    "AR03_diff",
    "AR04_diff",
    "AR05_diff",
    "max_dimension_diff",
    "version_stability_flag",
    "review_required",
    "review_reasons",
]


def run_version_comparison(root: Path = SCORING_ROOT, include_mda: bool = True, make_plots: bool = True, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    manifest = manifest or discover_inputs(root)
    selected = manifest.get("selected_inputs", {})
    if include_mda:
        summary, cases = _analyze_mda(root, Path(selected.get("mda_v1") or ""), Path(selected.get("mda_v2") or ""))
    else:
        summary, cases = {"status": "skipped", "reason": "include_mda false"}, []
        write_empty_csv(base / "outputs" / "mda_version_comparison.csv", VERSION_HEADERS)
    output = {"mda": summary, "high_volatility_cases": cases}
    if make_plots:
        rows = read_csv_rows(base / "outputs" / "mda_version_comparison.csv")
        try_make_hist(root, "version_score_diff_distribution.png", [safe_float(row.get("abs_total_score_diff")) for row in rows], "Version score differences", "absolute total score difference")
    else:
        write_plot_skip(root, "version_score_diff_distribution.png", "make_plots false")
    _write_report(base / "reports" / "mda_version_stability_report.md", output)
    save_json(base / "outputs" / "version_comparison_summary.json", output)
    return output


def _analyze_mda(root: Path, v1_path: Path, v2_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_path = stability_root(root) / "outputs" / "mda_version_comparison.csv"
    if not str(v1_path) or not str(v2_path) or not v1_path.is_file() or not v2_path.is_file():
        write_empty_csv(output_path, VERSION_HEADERS)
        return {
            "status": "missing-input",
            "v1_path": str(v1_path) if str(v1_path) != "." else "",
            "v2_path": str(v2_path) if str(v2_path) != "." else "",
            "output_path": str(output_path),
            "document_count": 0,
            "reason": "version comparison not available because v1 or v2 scores are missing.",
        }, []
    v1 = {row.get("document_id", ""): row for row in read_csv_rows(v1_path)}
    v2 = {row.get("document_id", ""): row for row in read_csv_rows(v2_path)}
    rows = []
    cases = []
    dimensions = ["AR01", "AR02", "AR03", "AR04", "AR05"]
    focus_dimensions = {"AR02", "AR03", "AR04"}
    for document_id in sorted(set(v1) & set(v2)):
        left = v1[document_id]
        right = v2[document_id]
        total_a = safe_float(left.get("total_score_100"))
        total_b = safe_float(right.get("total_score_100"))
        diffs = {code: safe_float(right.get(code)) - safe_float(left.get(code)) for code in dimensions}
        max_dimension_diff = max(abs(value) for value in diffs.values()) if diffs else 0.0
        abs_total = abs(total_b - total_a)
        reasons = []
        if abs_total > 15:
            reasons.append("major_version_effect_review_required")
        elif abs_total > 10:
            reasons.append("moderate_version_effect")
        if max_dimension_diff >= 2:
            reasons.append("dimension_shift_review_required")
        focus_shift = [code for code in focus_dimensions if abs(diffs.get(code, 0.0)) >= 2]
        if focus_shift:
            reasons.append("focus_dimension_shift:" + ",".join(sorted(focus_shift)))
        row = {
            "document_id": document_id,
            "ticker": right.get("ticker") or left.get("ticker", ""),
            "company_name": right.get("company_name") or left.get("company_name", ""),
            "report_year": right.get("report_year") or left.get("report_year", ""),
            "score_version_a": "v1_initial",
            "score_version_b": "clean_v2_full",
            "total_score_a": fmt(total_a),
            "total_score_b": fmt(total_b),
            "total_score_diff": fmt(total_b - total_a),
            "abs_total_score_diff": fmt(abs_total),
            "max_dimension_diff": fmt(max_dimension_diff),
            "version_stability_flag": _flag(abs_total),
            "review_required": "yes" if reasons else "no",
            "review_reasons": ";".join(reasons),
        }
        for code in dimensions:
            row[f"{code}_diff"] = fmt(diffs[code])
        rows.append(row)
        base_case = {
            "document_id": document_id,
            "doc_type": "mda",
            "ticker": row["ticker"],
            "company_name": row["company_name"],
            "year_or_publish_date": row["report_year"],
            "source_file": str(output_path),
        }
        if abs_total > 10:
            cases.append({**base_case, "issue_type": "version_comparison_abs_diff_gt_10", "severity": "high" if abs_total > 15 else "medium", "score_impact": fmt(abs_total), "rank_impact": "", "dimension_affected": "", "recommended_action": "manual_review"})
        if max_dimension_diff >= 2:
            cases.append({**base_case, "issue_type": "version_comparison_max_dimension_diff_ge_2", "severity": "medium", "score_impact": fmt(abs_total), "rank_impact": "", "dimension_affected": _max_dimension(diffs), "recommended_action": "check_text_extraction"})
    write_csv_rows(output_path, VERSION_HEADERS, rows)
    return {
        "status": "success" if rows else "empty-overlap",
        "v1_path": str(v1_path),
        "v2_path": str(v2_path),
        "output_path": str(output_path),
        "document_count": len(rows),
        "major_review_count": sum(safe_float(row.get("abs_total_score_diff")) > 15 for row in rows),
        "moderate_or_major_count": sum(safe_float(row.get("abs_total_score_diff")) > 10 for row in rows),
        "dimension_shift_count": sum(safe_float(row.get("max_dimension_diff")) >= 2 for row in rows),
    }, cases


def _flag(abs_total: float) -> str:
    if abs_total <= 5:
        return "stable"
    if abs_total <= 10:
        return "minor_version_effect"
    if abs_total <= 15:
        return "moderate_version_effect"
    return "major_version_effect_review_required"


def _max_dimension(diffs: dict[str, float]) -> str:
    return max(diffs, key=lambda code: abs(diffs[code])) if diffs else ""


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    item = summary.get("mda", {})
    lines = [
        "# MD&A Version Stability Report",
        "",
        "- Compares existing v1 and clean-v2/full scores only.",
        f"- status: {item.get('status', '')}",
        f"- document_count: {item.get('document_count', 0)}",
        f"- moderate_or_major_count: {item.get('moderate_or_major_count', 0)}",
        f"- dimension_shift_count: {item.get('dimension_shift_count', 0)}",
        f"- reason: {item.get('reason', '')}",
        "",
    ]
    write_markdown(path, lines)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MD&A version stability comparison.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--include-mda", type=parse_bool, default=True)
    parser.add_argument("--make-plots", type=parse_bool, default=True)
    args = parser.parse_args()
    print(json.dumps(run_version_comparison(args.root, args.include_mda, args.make_plots), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
