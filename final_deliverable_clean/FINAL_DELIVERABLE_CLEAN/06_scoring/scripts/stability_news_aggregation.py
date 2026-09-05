from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
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
    weighted_total,
    write_empty_csv,
    write_markdown,
    write_plot_skip,
)
from stability_weight_sensitivity import NEWS_WEIGHTS


NEWS_AGG_HEADERS = [
    "ticker",
    "company_name",
    "year",
    "num_news",
    "mean_score",
    "median_score",
    "trimmed_mean_score",
    "high_source_reliability_mean",
    "high_information_value_mean",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "aggregation_score_range",
    "aggregation_stability_flag",
    "review_required",
    "review_reasons",
]


def run_news_aggregation_stability(root: Path = SCORING_ROOT, include_news: bool = True, make_plots: bool = True, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    output_path = base / "outputs" / "news_company_year_aggregation_stability.csv"
    if not include_news:
        write_empty_csv(output_path, NEWS_AGG_HEADERS)
        summary = {"status": "skipped", "reason": "include_news false", "output_path": str(output_path), "group_count": 0, "high_volatility_cases": []}
        save_json(base / "outputs" / "news_aggregation_summary.json", summary)
        return summary
    manifest = manifest or discover_inputs(root)
    selected = manifest.get("selected_inputs", {})
    scores_path = Path(selected.get("news_scores") or "")
    registry_path = Path(selected.get("news_registry") or "")
    if not str(scores_path) or not scores_path.is_file() or not read_csv_rows(scores_path):
        write_empty_csv(output_path, NEWS_AGG_HEADERS)
        summary = {"status": "missing-news", "reason": "news scores are missing or empty", "news_scores_path": str(scores_path) if str(scores_path) != "." else "", "output_path": str(output_path), "group_count": 0, "high_volatility_cases": []}
        _write_report(base / "reports" / "news_aggregation_stability_report.md", summary)
        save_json(base / "outputs" / "news_aggregation_summary.json", summary)
        write_plot_skip(root, "news_aggregation_range_distribution.png", "missing news")
        return summary
    registry = {row.get("document_id", ""): row for row in read_csv_rows(registry_path)} if registry_path.exists() else {}
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(scores_path):
        document_id = row.get("document_id", "")
        enriched = {**registry.get(document_id, {}), **row}
        year = str(enriched.get("publish_date", ""))[:4] or "unknown"
        groups[(enriched.get("ticker", ""), enriched.get("company_name", ""), year)].append(enriched)
    rows = []
    cases = []
    for (ticker, company, year), items in sorted(groups.items()):
        scores = [_news_score(row) for row in items]
        methods = {
            "mean_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "trimmed_mean_score": _trimmed_mean(scores),
        }
        source_scores = [_news_score(row) for row in items if safe_float(row.get("N02")) >= 4]
        info_scores = [_news_score(row) for row in items if safe_float(row.get("N04")) >= 4]
        if source_scores:
            methods["high_source_reliability_mean"] = statistics.mean(source_scores)
        if info_scores:
            methods["high_information_value_mean"] = statistics.mean(info_scores)
        ci_low, ci_high = _bootstrap_ci(scores) if len(scores) >= 5 else ("", "")
        method_values = [safe_float(value) for value in methods.values()]
        score_range = max(method_values) - min(method_values) if method_values else 0.0
        reasons = []
        if len(items) < 3:
            reasons.append("low_news_count_flag")
        if score_range > 10:
            reasons.append("aggregation_sensitive_review_required")
        elif score_range > 5:
            reasons.append("moderately_sensitive")
        if ci_low != "" and safe_float(ci_high) - safe_float(ci_low) > 15:
            reasons.append("unstable_news_signal")
        row = {
            "ticker": ticker,
            "company_name": company,
            "year": year,
            "num_news": len(items),
            "mean_score": fmt(methods["mean_score"]),
            "median_score": fmt(methods["median_score"]),
            "trimmed_mean_score": fmt(methods["trimmed_mean_score"]),
            "high_source_reliability_mean": fmt(methods["high_source_reliability_mean"]) if "high_source_reliability_mean" in methods else "",
            "high_information_value_mean": fmt(methods["high_information_value_mean"]) if "high_information_value_mean" in methods else "",
            "bootstrap_ci_low": fmt(ci_low) if ci_low != "" else "",
            "bootstrap_ci_high": fmt(ci_high) if ci_high != "" else "",
            "aggregation_score_range": fmt(score_range),
            "aggregation_stability_flag": _flag(score_range),
            "review_required": "yes" if reasons else "no",
            "review_reasons": ";".join(reasons),
        }
        rows.append(row)
        if score_range > 10:
            cases.append(
                {
                    "document_id": f"{ticker}_{year}",
                    "doc_type": "news",
                    "ticker": ticker,
                    "company_name": company,
                    "year_or_publish_date": year,
                    "issue_type": "news_aggregation_score_range_gt_10",
                    "severity": "high",
                    "score_impact": fmt(score_range),
                    "rank_impact": "",
                    "dimension_affected": "",
                    "recommended_action": "check_news_source",
                    "source_file": str(output_path),
                }
            )
    write_csv_rows(output_path, NEWS_AGG_HEADERS, rows)
    if make_plots:
        try_make_hist(root, "news_aggregation_range_distribution.png", [safe_float(row.get("aggregation_score_range")) for row in rows], "News aggregation range", "aggregation score range")
    else:
        write_plot_skip(root, "news_aggregation_range_distribution.png", "make_plots false")
    summary = {
        "status": "success" if rows else "missing-news",
        "news_scores_path": str(scores_path),
        "news_registry_path": str(registry_path) if str(registry_path) != "." else "",
        "output_path": str(output_path),
        "group_count": len(rows),
        "sensitive_group_count": sum(safe_float(row.get("aggregation_score_range")) > 10 for row in rows),
        "low_news_count_group_count": sum(safe_float(row.get("num_news")) < 3 for row in rows),
        "high_volatility_cases": cases,
    }
    _write_report(base / "reports" / "news_aggregation_stability_report.md", summary)
    save_json(base / "outputs" / "news_aggregation_summary.json", summary)
    return summary


def _news_score(row: dict[str, str]) -> float:
    if row.get("total_score_100"):
        return safe_float(row.get("total_score_100"))
    return weighted_total(row, NEWS_WEIGHTS["W2_main_recommended"])


def _trimmed_mean(values: list[float]) -> float:
    if len(values) < 10:
        return statistics.mean(values)
    trim = max(1, int(len(values) * 0.10))
    return statistics.mean(sorted(values)[trim:-trim])


def _bootstrap_ci(values: list[float], reps: int = 500) -> tuple[float, float]:
    rng = random.Random(42)
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(statistics.mean(sample))
    means.sort()
    return means[int(reps * 0.025)], means[int(reps * 0.975)]


def _flag(score_range: float) -> str:
    if score_range <= 5:
        return "stable"
    if score_range <= 10:
        return "moderately_sensitive"
    return "aggregation_sensitive_review_required"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# News Aggregation Stability Report",
        "",
        "News is aggregated by company-year only. MD&A and News scores are not averaged.",
        f"- status: {summary.get('status', '')}",
        f"- group_count: {summary.get('group_count', 0)}",
        f"- sensitive_group_count: {summary.get('sensitive_group_count', 0)}",
        f"- low_news_count_group_count: {summary.get('low_news_count_group_count', 0)}",
        f"- reason: {summary.get('reason', '')}",
        "",
    ]
    write_markdown(path, lines)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run News company-year aggregation stability analysis.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--include-news", type=parse_bool, default=True)
    parser.add_argument("--make-plots", type=parse_bool, default=True)
    args = parser.parse_args()
    print(json.dumps(run_news_aggregation_stability(args.root, args.include_news, args.make_plots), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
