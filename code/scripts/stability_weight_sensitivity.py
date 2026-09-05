from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from stability_common import (
    dimension_correlation_matrix,
    dimension_stats,
    discover_inputs,
    descending_ranks,
    ensure_stability_dirs,
    fmt,
    high_correlation_pairs,
    pearson,
    safe_float,
    safe_int,
    spearman,
    stability_judgement,
    stability_root,
    top_bottom_overlap,
    try_make_hist,
    try_make_scatter,
    valid_dimension_rows,
    weighted_total,
    write_empty_csv,
    write_markdown,
)


MDA_WEIGHTS = {
    "W1_baseline": {"AR01": 0.20, "AR02": 0.25, "AR03": 0.25, "AR04": 0.20, "AR05": 0.10},
    "W2_main_recommended": {"AR01": 0.15, "AR02": 0.25, "AR03": 0.30, "AR04": 0.20, "AR05": 0.10},
    "W3_conservative_financial": {"AR01": 0.15, "AR02": 0.30, "AR03": 0.25, "AR04": 0.20, "AR05": 0.10},
}

NEWS_WEIGHTS = {
    "W1_baseline": {"N01": 0.30, "N02": 0.20, "N03": 0.20, "N04": 0.15, "N05": 0.15},
    "W2_main_recommended": {"N01": 0.30, "N02": 0.25, "N03": 0.15, "N04": 0.20, "N05": 0.10},
    "W3_source_reliability": {"N01": 0.30, "N02": 0.30, "N03": 0.15, "N04": 0.15, "N05": 0.10},
}

MDA_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "report_year",
    "AR01",
    "AR02",
    "AR03",
    "AR04",
    "AR05",
    "score_W1_baseline",
    "score_W2_main_recommended",
    "score_W3_conservative_financial",
    "rank_W1",
    "rank_W2",
    "rank_W3",
    "rank_change_W1_W2",
    "rank_change_W2_W3",
    "max_rank_change",
    "score_change_W1_W2",
    "score_change_W2_W3",
    "max_score_change",
]

NEWS_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "publish_date",
    "source_name",
    "title",
    "N01",
    "N02",
    "N03",
    "N04",
    "N05",
    "score_W1_baseline",
    "score_W2_main_recommended",
    "score_W3_source_reliability",
    "rank_W1",
    "rank_W2",
    "rank_W3",
    "rank_change_W1_W2",
    "rank_change_W2_W3",
    "max_rank_change",
    "score_change_W1_W2",
    "score_change_W2_W3",
    "max_score_change",
]


def pearson_correlation(values_a: list[float], values_b: list[float]) -> float:
    return pearson(values_a, values_b)


def spearman_correlation(values_a: list[float], values_b: list[float]) -> float:
    return spearman(values_a, values_b)


def run_weight_stability(
    root: Path = SCORING_ROOT,
    include_mda: bool = True,
    include_news: bool = True,
    make_plots: bool = True,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    manifest = manifest or discover_inputs(root)
    selected = manifest.get("selected_inputs", {})
    summaries: dict[str, Any] = {}
    high_volatility: list[dict[str, Any]] = []

    if include_mda:
        summary, cases = _analyze_doc_type(
            root=root,
            doc_type="mda",
            input_path=Path(selected.get("mda_scores") or ""),
            weights=MDA_WEIGHTS,
            headers=MDA_HEADERS,
            dimensions=list(next(iter(MDA_WEIGHTS.values())).keys()),
            metadata_fields=["ticker", "company_name", "report_year"],
            output_name="mda_weight_sensitivity_scores.csv",
            make_plots=make_plots,
        )
        summaries["mda"] = summary
        high_volatility.extend(cases)
    else:
        summaries["mda"] = {"status": "skipped", "reason": "include_mda false"}

    if include_news:
        summary, cases = _analyze_doc_type(
            root=root,
            doc_type="news",
            input_path=Path(selected.get("news_scores") or ""),
            weights=NEWS_WEIGHTS,
            headers=NEWS_HEADERS,
            dimensions=list(next(iter(NEWS_WEIGHTS.values())).keys()),
            metadata_fields=["ticker", "company_name", "publish_date", "source_name", "title"],
            output_name="news_weight_sensitivity_scores.csv",
            make_plots=make_plots,
        )
        summaries["news"] = summary
        high_volatility.extend(cases)
    else:
        summaries["news"] = {"status": "skipped", "reason": "include_news false"}

    summaries["high_volatility_cases"] = high_volatility
    _write_report(base / "reports" / "weight_sensitivity_report.md", summaries)
    save_json(base / "outputs" / "weight_sensitivity_summary.json", summaries)
    return summaries


def _analyze_doc_type(
    *,
    root: Path,
    doc_type: str,
    input_path: Path,
    weights: dict[str, dict[str, float]],
    headers: list[str],
    dimensions: list[str],
    metadata_fields: list[str],
    output_name: str,
    make_plots: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_path = stability_root(root) / "outputs" / output_name
    missing_status = "missing-news" if doc_type == "news" else "missing-input"
    if not str(input_path) or not input_path.is_file():
        for plot_name in _plot_names_for(doc_type):
            from stability_common import write_plot_skip

            write_plot_skip(root, plot_name, "score input missing")
        write_empty_csv(output_path, headers)
        return _missing_summary(doc_type, input_path, output_path, missing_status, "score input missing"), []
    rows = valid_dimension_rows(read_csv_rows(input_path), dimensions)
    if not rows:
        for plot_name in _plot_names_for(doc_type):
            from stability_common import write_plot_skip

            write_plot_skip(root, plot_name, "no scored rows")
        write_empty_csv(output_path, headers)
        status = "missing-news" if doc_type == "news" else "empty-input"
        return _missing_summary(doc_type, input_path, output_path, status, "no scored rows with all dimension scores"), []

    scored_rows = _score_rows(rows, dimensions, weights, metadata_fields)
    write_csv_rows(output_path, headers, scored_rows)
    pairwise = _pairwise_metrics(scored_rows, list(weights.keys()))
    dim_stats = dimension_stats(rows, dimensions)
    dim_matrix = dimension_correlation_matrix(rows, dimensions)
    low_variance = [item for item in dim_stats if item["count"] >= 2 and item["std"] < 0.30]
    high_corr = high_correlation_pairs(dim_matrix)
    rank_change_count = sum(abs(safe_int(row.get("max_rank_change"))) > 10 for row in scored_rows)
    score_change_count = sum(safe_float(row.get("max_score_change")) > 10 for row in scored_rows)
    recommendation = _recommendation(doc_type, pairwise, low_variance, high_corr)
    if make_plots:
        _make_weight_plots(root, doc_type, scored_rows, list(weights.keys()))
    else:
        for plot_name in _plot_names_for(doc_type):
            from stability_common import write_plot_skip

            write_plot_skip(root, plot_name, "make_plots false")
    summary = {
        "status": "success",
        "doc_type": doc_type,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "document_count": len(scored_rows),
        "skipped_invalid_rows": len(read_csv_rows(input_path)) - len(rows),
        "pairwise_metrics": pairwise,
        "dimension_stats": dim_stats,
        "dimension_correlation_matrix": dim_matrix,
        "low_variance_dimensions": low_variance,
        "high_correlation_pairs": high_corr,
        "rank_change_gt_10_count": rank_change_count,
        "score_change_gt_10_count": score_change_count,
        "recommendation": recommendation,
        "sensitive_documents": _sensitive_documents(scored_rows, metadata_fields),
    }
    return summary, _weight_high_volatility_cases(doc_type, scored_rows, metadata_fields, str(output_path), high_corr)


def _score_rows(
    rows: list[dict[str, str]],
    dimensions: list[str],
    weights: dict[str, dict[str, float]],
    metadata_fields: list[str],
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        scored = {"document_id": row.get("document_id", "")}
        for field in metadata_fields:
            scored[field] = row.get(field, "")
        for dimension in dimensions:
            scored[dimension] = fmt(safe_float(row.get(dimension)))
        for scheme, scheme_weights in weights.items():
            scored[f"score_{scheme}"] = fmt(weighted_total(row, scheme_weights))
        output.append(scored)
    scheme_names = list(weights.keys())
    for idx, scheme in enumerate(scheme_names, start=1):
        ranks = descending_ranks(output, f"score_{scheme}")
        for row in output:
            row[f"rank_W{idx}"] = ranks.get(row["document_id"], 0)
    for row in output:
        rank_w1 = safe_int(row.get("rank_W1"))
        rank_w2 = safe_int(row.get("rank_W2"))
        rank_w3 = safe_int(row.get("rank_W3"))
        score_w1 = safe_float(row.get(f"score_{scheme_names[0]}"))
        score_w2 = safe_float(row.get(f"score_{scheme_names[1]}"))
        score_w3 = safe_float(row.get(f"score_{scheme_names[2]}"))
        row["rank_change_W1_W2"] = rank_w2 - rank_w1
        row["rank_change_W2_W3"] = rank_w3 - rank_w2
        row["max_rank_change"] = max(abs(rank_w2 - rank_w1), abs(rank_w3 - rank_w2), abs(rank_w3 - rank_w1))
        row["score_change_W1_W2"] = fmt(score_w2 - score_w1)
        row["score_change_W2_W3"] = fmt(score_w3 - score_w2)
        row["max_score_change"] = fmt(max(abs(score_w2 - score_w1), abs(score_w3 - score_w2), abs(score_w3 - score_w1)))
    return output


def _pairwise_metrics(rows: list[dict[str, Any]], scheme_names: list[str]) -> list[dict[str, Any]]:
    metrics = []
    if len(rows) < 2:
        return metrics
    for left, right in itertools.combinations(scheme_names, 2):
        left_scores = [safe_float(row.get(f"score_{left}")) for row in rows]
        right_scores = [safe_float(row.get(f"score_{right}")) for row in rows]
        left_ranks = [safe_float(row.get(f"rank_W{scheme_names.index(left) + 1}")) for row in rows]
        right_ranks = [safe_float(row.get(f"rank_W{scheme_names.index(right) + 1}")) for row in rows]
        spearman_value = pearson(left_ranks, right_ranks)
        metrics.append(
            {
                "scheme_a": left,
                "scheme_b": right,
                "pearson": pearson(left_scores, right_scores),
                "spearman": spearman_value,
                "top_20_overlap": top_bottom_overlap(rows, f"score_{left}", f"score_{right}", top=True),
                "bottom_20_overlap": top_bottom_overlap(rows, f"score_{left}", f"score_{right}", top=False),
                "stability_judgement": stability_judgement(spearman_value),
            }
        )
    return metrics


def _recommendation(doc_type: str, pairwise: list[dict[str, Any]], low_variance: list[dict[str, Any]], high_corr: list[dict[str, Any]]) -> dict[str, Any]:
    min_spearman = min((safe_float(item.get("spearman")) for item in pairwise), default=None)
    third = "W3_conservative_financial" if doc_type == "mda" else "W3_source_reliability"
    if min_spearman is None:
        return {
            "main_weight": "W2_main_recommended",
            "robustness_weights": ["W1_baseline", third],
            "adopt_w2": False,
            "reason": "insufficient scored documents for correlation analysis",
            "stability": "insufficient",
        }
    if min_spearman >= 0.90:
        reason = "results are stable across weight schemes"
        adopt = True
    elif min_spearman >= 0.80:
        reason = "results are moderately stable; disclose weight sensitivity"
        adopt = True
    else:
        reason = "results are weight-sensitive and need review"
        adopt = False
    notes = []
    if high_corr:
        notes.append("possible redundancy")
    if low_variance:
        notes.append("low discrimination warning")
    return {
        "main_weight": "W2_main_recommended",
        "robustness_weights": ["W1_baseline", third],
        "adopt_w2": adopt,
        "min_spearman": min_spearman,
        "reason": reason,
        "stability": stability_judgement(min_spearman),
        "warnings": notes,
    }


def _sensitive_documents(rows: list[dict[str, Any]], metadata_fields: list[str]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        item = {
            "document_id": row.get("document_id", ""),
            "max_rank_change": safe_int(row.get("max_rank_change")),
            "max_score_change": safe_float(row.get("max_score_change")),
        }
        for field in metadata_fields:
            item[field] = row.get(field, "")
        output.append(item)
    return sorted(output, key=lambda item: (-item["max_score_change"], -item["max_rank_change"], item["document_id"]))[:20]


def _weight_high_volatility_cases(
    doc_type: str,
    rows: list[dict[str, Any]],
    metadata_fields: list[str],
    source_file: str,
    high_corr: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cases = []
    date_field = "report_year" if doc_type == "mda" else "publish_date"
    for row in rows:
        base = {
            "document_id": row.get("document_id", ""),
            "doc_type": doc_type,
            "ticker": row.get("ticker", ""),
            "company_name": row.get("company_name", ""),
            "year_or_publish_date": row.get(date_field, ""),
            "source_file": source_file,
        }
        if safe_float(row.get("max_score_change")) > 10:
            cases.append({**base, "issue_type": "weight_score_change_gt_10", "severity": "high", "score_impact": row.get("max_score_change", ""), "rank_impact": row.get("max_rank_change", ""), "dimension_affected": "", "recommended_action": "manual_review"})
        if abs(safe_int(row.get("max_rank_change"))) > 10:
            cases.append({**base, "issue_type": "weight_rank_change_gt_10", "severity": "medium", "score_impact": row.get("max_score_change", ""), "rank_impact": row.get("max_rank_change", ""), "dimension_affected": "", "recommended_action": "manual_review"})
        if high_corr and metadata_fields:
            pair = high_corr[0]
            cases.append({**base, "issue_type": "dimension_correlation_warning", "severity": "low", "score_impact": "", "rank_impact": "", "dimension_affected": f"{pair['dimension_a']},{pair['dimension_b']}", "recommended_action": "manual_review"})
    return cases


def _make_weight_plots(root: Path, doc_type: str, rows: list[dict[str, Any]], scheme_names: list[str]) -> None:
    prefix = "mda" if doc_type == "mda" else "news"
    try_make_scatter(root, f"{prefix}_weight_scheme_scatter_W1_W2.png", rows, f"score_{scheme_names[0]}", f"score_{scheme_names[1]}", f"{prefix.upper()} W1 vs W2")
    try_make_scatter(root, f"{prefix}_weight_scheme_scatter_W2_W3.png", rows, f"score_{scheme_names[1]}", f"score_{scheme_names[2]}", f"{prefix.upper()} W2 vs W3")
    try_make_hist(root, f"{prefix}_rank_change_distribution.png", [abs(safe_float(row.get("max_rank_change"))) for row in rows], f"{prefix.upper()} rank change", "max rank change")


def _plot_names_for(doc_type: str) -> list[str]:
    prefix = "mda" if doc_type == "mda" else "news"
    return [
        f"{prefix}_weight_scheme_scatter_W1_W2.png",
        f"{prefix}_weight_scheme_scatter_W2_W3.png",
        f"{prefix}_rank_change_distribution.png",
    ]


def _missing_summary(doc_type: str, input_path: Path, output_path: Path, status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status,
        "doc_type": doc_type,
        "input_path": str(input_path) if str(input_path) != "." else "",
        "output_path": str(output_path),
        "document_count": 0,
        "pairwise_metrics": [],
        "dimension_stats": [],
        "dimension_correlation_matrix": {},
        "low_variance_dimensions": [],
        "high_correlation_pairs": [],
        "rank_change_gt_10_count": 0,
        "score_change_gt_10_count": 0,
        "recommendation": {
            "main_weight": "W2_main_recommended",
            "adopt_w2": False,
            "reason": reason,
            "stability": "unavailable",
        },
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Weight Sensitivity Report",
        "",
        "This analysis recalculates totals from existing dimension scores only. It does not call qwen3:8b and does not rewrite original scoring files.",
        "",
    ]
    for doc_type in ["mda", "news"]:
        item = summary.get(doc_type, {})
        lines.extend(
            [
                f"## {doc_type.upper()} Weight Sensitivity",
                f"- status: {item.get('status', 'skipped')}",
                f"- input_path: {item.get('input_path', '')}",
                f"- document_count: {item.get('document_count', 0)}",
                f"- recommendation: {item.get('recommendation', {}).get('reason', '')}",
                f"- W2_main_recommended adopted: {item.get('recommendation', {}).get('adopt_w2', False)}",
                f"- possible redundancy warnings: {len(item.get('high_correlation_pairs', []))}",
                f"- low variance warnings: {len(item.get('low_variance_dimensions', []))}",
                "",
            ]
        )
        for metric in item.get("pairwise_metrics", []):
            lines.append(
                f"- {metric['scheme_a']} vs {metric['scheme_b']}: Pearson={metric['pearson']}, Spearman={metric['spearman']}, {metric['stability_judgement']}"
            )
        if item.get("pairwise_metrics"):
            lines.append("")
    write_markdown(path, lines)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MD&A and News weight sensitivity under 11_stability_analysis.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--include-mda", type=parse_bool, default=True)
    parser.add_argument("--include-news", type=parse_bool, default=True)
    parser.add_argument("--make-plots", type=parse_bool, default=True)
    args = parser.parse_args()
    print(json.dumps(run_weight_stability(args.root, args.include_mda, args.include_news, args.make_plots), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
