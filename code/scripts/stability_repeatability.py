from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from stability_common import (
    discover_inputs,
    ensure_stability_dirs,
    fmt,
    pearson,
    safe_float,
    spearman,
    stability_root,
    try_make_hist,
    write_empty_csv,
    write_markdown,
    write_plot_skip,
)


REPEAT_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "run1_total_score",
    "run2_total_score",
    "total_score_diff",
    "abs_total_score_diff",
    "max_dimension_diff",
    "dimension_diff_count_ge_2",
    "repeatability_flag",
    "review_required",
    "review_reasons",
]


def run_repeatability_analysis(root: Path = SCORING_ROOT, include_mda: bool = True, include_news: bool = True, make_plots: bool = True, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    manifest = manifest or discover_inputs(root)
    selected = manifest.get("selected_inputs", {})
    summary: dict[str, Any] = {}
    high_volatility: list[dict[str, Any]] = []
    if include_mda:
        item, cases = _analyze(root, "mda", Path(selected.get("mda_run1") or ""), Path(selected.get("mda_run2") or ""), ["AR01", "AR02", "AR03", "AR04", "AR05"])
        summary["mda"] = item
        high_volatility.extend(cases)
    else:
        summary["mda"] = {"status": "skipped", "reason": "include_mda false"}
    if include_news:
        item, cases = _analyze(root, "news", Path(selected.get("news_run1") or ""), Path(selected.get("news_run2") or ""), ["N01", "N02", "N03", "N04", "N05"])
        summary["news"] = item
        high_volatility.extend(cases)
    else:
        summary["news"] = {"status": "skipped", "reason": "include_news false"}
    summary["high_volatility_cases"] = high_volatility
    if make_plots:
        diffs = []
        for doc_type in ["mda", "news"]:
            diffs.extend([safe_float(row.get("abs_total_score_diff")) for row in read_csv_rows(base / "outputs" / f"{doc_type}_repeatability_comparison.csv")])
        try_make_hist(root, "repeatability_score_diff_distribution.png", diffs, "Repeatability score differences", "absolute total score difference")
    else:
        write_plot_skip(root, "repeatability_score_diff_distribution.png", "make_plots false")
    _write_report(base / "reports" / "repeatability_report.md", summary)
    save_json(base / "outputs" / "repeatability_summary.json", summary)
    return summary


def _analyze(root: Path, doc_type: str, run1_path: Path, run2_path: Path, dimensions: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_path = stability_root(root) / "outputs" / f"{doc_type}_repeatability_comparison.csv"
    if not str(run1_path) or not str(run2_path) or not run1_path.is_file() or not run2_path.is_file():
        write_empty_csv(output_path, REPEAT_HEADERS)
        return _missing(doc_type, run1_path, run2_path, output_path), []
    run1 = {row.get("document_id", ""): row for row in read_csv_rows(run1_path)}
    run2 = {row.get("document_id", ""): row for row in read_csv_rows(run2_path)}
    rows = []
    cases = []
    for document_id in sorted(set(run1) & set(run2)):
        left = run1[document_id]
        right = run2[document_id]
        total1 = safe_float(left.get("total_score_100"))
        total2 = safe_float(right.get("total_score_100"))
        dim_diffs = [abs(safe_float(left.get(code)) - safe_float(right.get(code))) for code in dimensions]
        max_dim_diff = max(dim_diffs) if dim_diffs else 0.0
        dim_ge_2 = sum(diff >= 2 for diff in dim_diffs)
        abs_diff = abs(total2 - total1)
        reasons = []
        if abs_diff > 10:
            reasons.append("total_score_abs_diff_gt_10")
        if dim_ge_2:
            reasons.append("dimension_diff_ge_2")
        row = {
            "document_id": document_id,
            "ticker": left.get("ticker", ""),
            "company_name": left.get("company_name", ""),
            "run1_total_score": fmt(total1),
            "run2_total_score": fmt(total2),
            "total_score_diff": fmt(total2 - total1),
            "abs_total_score_diff": fmt(abs_diff),
            "max_dimension_diff": fmt(max_dim_diff),
            "dimension_diff_count_ge_2": dim_ge_2,
            "repeatability_flag": _repeatability_flag(abs_diff),
            "review_required": "yes" if reasons else "no",
            "review_reasons": ";".join(reasons),
        }
        rows.append(row)
        base_case = {
            "document_id": document_id,
            "doc_type": doc_type,
            "ticker": left.get("ticker", ""),
            "company_name": left.get("company_name", ""),
            "year_or_publish_date": left.get("report_year", left.get("publish_date", "")),
            "source_file": str(output_path),
        }
        if abs_diff > 10:
            cases.append({**base_case, "issue_type": "repeat_run_abs_diff_gt_10", "severity": "high", "score_impact": fmt(abs_diff), "rank_impact": "", "dimension_affected": "", "recommended_action": "manual_review"})
        if max_dim_diff >= 2:
            cases.append({**base_case, "issue_type": "repeat_run_max_dimension_diff_ge_2", "severity": "medium", "score_impact": fmt(abs_diff), "rank_impact": "", "dimension_affected": _max_diff_dimension(left, right, dimensions), "recommended_action": "manual_review"})
    write_csv_rows(output_path, REPEAT_HEADERS, rows)
    abs_values = [safe_float(row.get("abs_total_score_diff")) for row in rows]
    totals1 = [safe_float(row.get("run1_total_score")) for row in rows]
    totals2 = [safe_float(row.get("run2_total_score")) for row in rows]
    summary = {
        "status": "success" if rows else "empty-overlap",
        "doc_type": doc_type,
        "run1_path": str(run1_path),
        "run2_path": str(run2_path),
        "output_path": str(output_path),
        "document_count": len(rows),
        "pearson": pearson(totals1, totals2),
        "spearman": spearman(totals1, totals2),
        "mean_absolute_difference": round(statistics.mean(abs_values), 6) if abs_values else 0.0,
        "median_absolute_difference": round(statistics.median(abs_values), 6) if abs_values else 0.0,
        "max_absolute_difference": max(abs_values) if abs_values else 0.0,
        "proportion_abs_diff_gt_5": round(sum(value > 5 for value in abs_values) / len(abs_values), 6) if abs_values else 0.0,
        "proportion_abs_diff_gt_10": round(sum(value > 10 for value in abs_values) / len(abs_values), 6) if abs_values else 0.0,
        "proportion_dimension_diff_ge_2": round(sum(safe_float(row.get("dimension_diff_count_ge_2")) > 0 for row in rows) / len(rows), 6) if rows else 0.0,
        "icc": _icc_two_runs(totals1, totals2),
    }
    return summary, cases


def _repeatability_flag(abs_diff: float) -> str:
    if abs_diff <= 5:
        return "stable"
    if abs_diff <= 10:
        return "moderate_difference"
    return "unstable_review_required"


def _max_diff_dimension(left: dict[str, str], right: dict[str, str], dimensions: list[str]) -> str:
    if not dimensions:
        return ""
    return max(dimensions, key=lambda code: abs(safe_float(left.get(code)) - safe_float(right.get(code))))


def _icc_two_runs(values1: list[float], values2: list[float]) -> float:
    if len(values1) < 2 or len(values1) != len(values2):
        return 0.0
    means = [(a + b) / 2 for a, b in zip(values1, values2)]
    grand = statistics.mean(means)
    between = sum((mean - grand) ** 2 for mean in means) * 2 / (len(means) - 1)
    within = sum((a - mean) ** 2 + (b - mean) ** 2 for a, b, mean in zip(values1, values2, means)) / len(means)
    denominator = between + within
    return round(max(-1.0, min(1.0, (between - within) / denominator)), 6) if denominator else 0.0


def _missing(doc_type: str, run1_path: Path, run2_path: Path, output_path: Path) -> dict[str, Any]:
    return {
        "status": "missing-input",
        "doc_type": doc_type,
        "run1_path": str(run1_path) if str(run1_path) != "." else "",
        "run2_path": str(run2_path) if str(run2_path) != "." else "",
        "output_path": str(output_path),
        "document_count": 0,
        "reason": "repeatability analysis not available because repeat-run scores are missing.",
        "pearson": 0.0,
        "spearman": 0.0,
        "icc": 0.0,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Repeatability Report", "", "No repeat scoring is performed by this module.", ""]
    for doc_type in ["mda", "news"]:
        item = summary.get(doc_type, {})
        lines.extend(
            [
                f"## {doc_type.upper()}",
                f"- status: {item.get('status', 'skipped')}",
                f"- document_count: {item.get('document_count', 0)}",
                f"- Pearson: {item.get('pearson', 0)}",
                f"- Spearman: {item.get('spearman', 0)}",
                f"- ICC: {item.get('icc', 0)}",
                f"- reason: {item.get('reason', '')}",
                "",
            ]
        )
    write_markdown(path, lines)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run repeatability analysis for optional repeat-run scores.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--include-mda", type=parse_bool, default=True)
    parser.add_argument("--include-news", type=parse_bool, default=True)
    parser.add_argument("--make-plots", type=parse_bool, default=True)
    args = parser.parse_args()
    print(json.dumps(run_repeatability_analysis(args.root, args.include_mda, args.include_news, args.make_plots), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
