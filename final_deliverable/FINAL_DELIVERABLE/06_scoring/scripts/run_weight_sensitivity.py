from __future__ import annotations

import argparse
import itertools
import math
import statistics
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, ensure_directories, read_csv_rows, standardize_score, write_csv_rows


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

MDA_OUTPUT_HEADERS = [
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
]

NEWS_OUTPUT_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "publish_date",
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
]


def run_weight_sensitivity(
    root: Path = SCORING_ROOT,
    mda_input_path: Path | None = None,
    news_input_path: Path | None = None,
) -> dict[str, Any]:
    ensure_directories(root)
    out_dir = root / "06_ratings" / "weight_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    mda_path = mda_input_path or _default_mda_input(root)
    news_path = news_input_path or root / "06_ratings" / "news_document_scores.csv"
    mda_summary = _analyze_doc_type(
        doc_type="mda",
        rows=read_csv_rows(mda_path),
        input_path=mda_path,
        output_path=out_dir / "mda_weight_sensitivity_scores.csv",
        dimensions=list(next(iter(MDA_WEIGHTS.values())).keys()),
        weights=MDA_WEIGHTS,
        output_headers=MDA_OUTPUT_HEADERS,
        metadata_fields=["ticker", "company_name", "report_year"],
        root=root,
    )
    news_summary = _analyze_doc_type(
        doc_type="news",
        rows=read_csv_rows(news_path),
        input_path=news_path,
        output_path=out_dir / "news_weight_sensitivity_scores.csv",
        dimensions=list(next(iter(NEWS_WEIGHTS.values())).keys()),
        weights=NEWS_WEIGHTS,
        output_headers=NEWS_OUTPUT_HEADERS,
        metadata_fields=["ticker", "company_name", "publish_date"],
        root=root,
    )
    summary = {
        "mda": mda_summary,
        "news": news_summary,
        "report_path": str(root / "08_reports" / "weight_sensitivity_report.md"),
    }
    _write_report(root / "08_reports" / "weight_sensitivity_report.md", summary)
    return summary


def _default_mda_input(root: Path) -> Path:
    v2_path = root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv"
    if read_csv_rows(v2_path):
        return v2_path
    return root / "06_ratings" / "mda_document_scores.csv"


def _analyze_doc_type(
    *,
    doc_type: str,
    rows: list[dict[str, str]],
    input_path: Path,
    output_path: Path,
    dimensions: list[str],
    weights: dict[str, dict[str, float]],
    output_headers: list[str],
    metadata_fields: list[str],
    root: Path,
) -> dict[str, Any]:
    valid_rows = [_row for _row in rows if _has_dimension_scores(_row, dimensions)]
    scored_rows = _score_rows(valid_rows, dimensions, weights, metadata_fields)
    write_csv_rows(output_path, output_headers, scored_rows)
    score_columns = [f"score_{name}" for name in weights]
    dimension_stats = _dimension_stats(valid_rows, dimensions)
    dimension_correlations = _dimension_correlation_matrix(valid_rows, dimensions)
    pairwise = _scheme_pairwise_metrics(scored_rows, list(weights.keys()))
    sensitive_documents = _sensitive_documents(scored_rows, metadata_fields)
    reliability_issues = _scoring_reliability_issues(root, doc_type, len(valid_rows), dimensions)
    summary = {
        "doc_type": doc_type,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "document_count": len(valid_rows),
        "skipped_invalid_rows": len(rows) - len(valid_rows),
        "pairwise_metrics": pairwise,
        "dimension_stats": dimension_stats,
        "dimension_correlation_matrix": dimension_correlations,
        "low_variance_dimensions": [item for item in dimension_stats if item["count"] >= 2 and item["std"] < 0.30],
        "high_correlation_pairs": _high_correlation_pairs(dimension_correlations),
        "rank_change_gt_10_count": sum(_safe_int(row.get("max_rank_change")) > 10 for row in scored_rows),
        "score_change_gt_10_count": _score_change_gt_10_count(scored_rows, score_columns),
        "sensitive_documents": sensitive_documents,
        "scoring_reliability_issues": reliability_issues,
    }
    summary["recommendation"] = _recommendation(summary)
    return summary


def _score_rows(
    rows: list[dict[str, str]],
    dimensions: list[str],
    weights: dict[str, dict[str, float]],
    metadata_fields: list[str],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        raw_scores = {dimension: _safe_float(row.get(dimension)) for dimension in dimensions}
        out = {"document_id": row.get("document_id", "")}
        for field in metadata_fields:
            out[field] = row.get(field, "")
        for dimension in dimensions:
            out[dimension] = _format_number(raw_scores[dimension])
        for scheme_name, scheme_weights in weights.items():
            out[f"score_{scheme_name}"] = _format_number(_weighted_total(raw_scores, scheme_weights))
        scored.append(out)
    scheme_names = list(weights.keys())
    for idx, scheme_name in enumerate(scheme_names, start=1):
        ranks = _descending_ranks(scored, f"score_{scheme_name}")
        for row in scored:
            row[f"rank_W{idx}"] = ranks[row["document_id"]]
    for row in scored:
        rank_w1 = _safe_int(row.get("rank_W1"))
        rank_w2 = _safe_int(row.get("rank_W2"))
        rank_w3 = _safe_int(row.get("rank_W3"))
        row["rank_change_W1_W2"] = rank_w2 - rank_w1
        row["rank_change_W2_W3"] = rank_w3 - rank_w2
        row["max_rank_change"] = max(abs(rank_w2 - rank_w1), abs(rank_w3 - rank_w2), abs(rank_w3 - rank_w1))
    return scored


def _weighted_total(raw_scores: dict[str, float], weights: dict[str, float]) -> float:
    return round(sum(standardize_score(raw_scores[dimension]) * weight for dimension, weight in weights.items()), 6)


def _descending_ranks(rows: list[dict[str, Any]], score_field: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (-_safe_float(row.get(score_field)), row.get("document_id", "")))
    ranks: dict[str, int] = {}
    previous_score: float | None = None
    current_rank = 0
    for index, row in enumerate(ordered, start=1):
        score = _safe_float(row.get(score_field))
        if previous_score is None or score != previous_score:
            current_rank = index
            previous_score = score
        ranks[row["document_id"]] = current_rank
    return ranks


def _scheme_pairwise_metrics(rows: list[dict[str, Any]], scheme_names: list[str]) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    metrics = []
    for left, right in itertools.combinations(scheme_names, 2):
        left_scores = [_safe_float(row.get(f"score_{left}")) for row in rows]
        right_scores = [_safe_float(row.get(f"score_{right}")) for row in rows]
        left_ranks = [_rank_for_scheme(row, scheme_names, left) for row in rows]
        right_ranks = [_rank_for_scheme(row, scheme_names, right) for row in rows]
        top_overlap = _top_bottom_overlap(rows, scheme_names, left, right, top=True)
        bottom_overlap = _top_bottom_overlap(rows, scheme_names, left, right, top=False)
        spearman = _pearson(left_ranks, right_ranks)
        metrics.append(
            {
                "scheme_a": left,
                "scheme_b": right,
                "pearson": round(_pearson(left_scores, right_scores), 6),
                "spearman": round(spearman, 6),
                "top_20_overlap": top_overlap,
                "bottom_20_overlap": bottom_overlap,
                "stability_judgement": _stability_judgement(spearman),
            }
        )
    return metrics


def _rank_for_scheme(row: dict[str, Any], scheme_names: list[str], scheme_name: str) -> int:
    return _safe_int(row.get(f"rank_W{scheme_names.index(scheme_name) + 1}"))


def _top_bottom_overlap(rows: list[dict[str, Any]], scheme_names: list[str], left: str, right: str, *, top: bool) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "denominator": 0, "ratio": 0.0}
    size = max(1, math.ceil(len(rows) * 0.20))
    reverse = top
    left_ids = _edge_ids(rows, f"score_{left}", size, reverse)
    right_ids = _edge_ids(rows, f"score_{right}", size, reverse)
    count = len(left_ids & right_ids)
    return {"count": count, "denominator": size, "ratio": round(count / size, 6)}


def _edge_ids(rows: list[dict[str, Any]], score_field: str, size: int, reverse: bool) -> set[str]:
    return {
        row["document_id"]
        for row in sorted(rows, key=lambda item: (_safe_float(item.get(score_field)), item.get("document_id", "")), reverse=reverse)[:size]
    }


def _dimension_stats(rows: list[dict[str, str]], dimensions: list[str]) -> list[dict[str, Any]]:
    stats = []
    for dimension in dimensions:
        values = [_safe_float(row.get(dimension)) for row in rows if _is_number(row.get(dimension))]
        stats.append(
            {
                "dimension": dimension,
                "mean": round(statistics.mean(values), 6) if values else 0.0,
                "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
                "count": len(values),
            }
        )
    return stats


def _dimension_correlation_matrix(rows: list[dict[str, str]], dimensions: list[str]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for left in dimensions:
        matrix[left] = {}
        for right in dimensions:
            left_values = [_safe_float(row.get(left)) for row in rows if _has_pair(row, left, right)]
            right_values = [_safe_float(row.get(right)) for row in rows if _has_pair(row, left, right)]
            if left == right and left_values:
                matrix[left][right] = 1.0
            else:
                matrix[left][right] = round(_pearson(left_values, right_values, constant_result=0.0), 6)
    return matrix


def _high_correlation_pairs(matrix: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    pairs = []
    dimensions = list(matrix.keys())
    for left, right in itertools.combinations(dimensions, 2):
        corr = matrix.get(left, {}).get(right, 0.0)
        if corr > 0.85:
            pairs.append({"dimension_a": left, "dimension_b": right, "correlation": corr, "warning": "possible redundancy"})
    return pairs


def _score_change_gt_10_count(rows: list[dict[str, Any]], score_columns: list[str]) -> int:
    count = 0
    for row in rows:
        scores = [_safe_float(row.get(column)) for column in score_columns]
        if scores and max(scores) - min(scores) > 10:
            count += 1
    return count


def _sensitive_documents(rows: list[dict[str, Any]], metadata_fields: list[str], limit: int = 10) -> list[dict[str, Any]]:
    score_fields = [field for field in rows[0].keys() if field.startswith("score_")] if rows else []
    ranked = []
    for row in rows:
        scores = [_safe_float(row.get(field)) for field in score_fields]
        item = {
            "document_id": row.get("document_id", ""),
            "max_rank_change": _safe_int(row.get("max_rank_change")),
            "max_score_change": round(max(scores) - min(scores), 6) if scores else 0.0,
        }
        for field in metadata_fields:
            item[field] = row.get(field, "")
        ranked.append(item)
    return sorted(ranked, key=lambda item: (-item["max_rank_change"], -item["max_score_change"], item["document_id"]))[:limit]


def _scoring_reliability_issues(root: Path, doc_type: str, document_count: int, dimensions: list[str]) -> list[dict[str, Any]]:
    if document_count == 0:
        return []
    review_rows = [
        row
        for row in read_csv_rows(root / "07_review" / "review_log.csv")
        if row.get("doc_type") == doc_type and row.get("dimension_code") in dimensions
    ]
    issues = []
    for dimension in dimensions:
        dimension_rows = [row for row in review_rows if row.get("dimension_code") == dimension]
        rate = len(dimension_rows) / document_count
        if rate > 0.10:
            issues.append(
                {
                    "dimension": dimension,
                    "review_count": len(dimension_rows),
                    "review_rate": round(rate, 6),
                    "warning": "scoring reliability issue; do not lower weight before reviewing errors",
                }
            )
    return issues


def _recommendation(summary: dict[str, Any]) -> dict[str, Any]:
    pairwise = summary["pairwise_metrics"]
    min_spearman = min((item["spearman"] for item in pairwise), default=None)
    if min_spearman is None:
        return {
            "main_weight": "W2_main_recommended",
            "robustness_weights": ["W1_baseline", _third_scheme_name(summary["doc_type"])],
            "adopt_w2": False,
            "reason": "insufficient scored documents for correlation analysis",
            "adjustment_needed": "collect scored documents first",
        }
    if min_spearman >= 0.90:
        reason = "results are stable across W1/W2/W3"
        adopt_w2 = True
        adjustment = "adjust weights only; no dimension change indicated by weight sensitivity alone"
    elif min_spearman >= 0.80:
        reason = "results are basically acceptable but weight sensitivity should be disclosed"
        adopt_w2 = True
        adjustment = "use W2 as main weight and report robustness checks"
    else:
        reason = "results are weight sensitive and require manual inspection"
        adopt_w2 = False
        adjustment = "review samples and dimension definitions before changing weights"
    if summary["high_correlation_pairs"] or summary["low_variance_dimensions"]:
        adjustment += "; inspect possible redundant or low-variance dimensions before deleting or reweighting"
    if summary["scoring_reliability_issues"]:
        adjustment += "; review scoring reliability issues before lowering any dimension weight"
    return {
        "main_weight": "W2_main_recommended",
        "robustness_weights": ["W1_baseline", _third_scheme_name(summary["doc_type"])],
        "adopt_w2": adopt_w2,
        "min_spearman": min_spearman,
        "reason": reason,
        "adjustment_needed": adjustment,
    }


def _third_scheme_name(doc_type: str) -> str:
    return "W3_conservative_financial" if doc_type == "mda" else "W3_source_reliability"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Weight Sensitivity Analysis",
        "",
        "This module recalculates totals from existing bottom-level AR/N dimension scores only. It does not call qwen3:8b and does not rescore documents.",
        "",
    ]
    for key, title in [("mda", "MD&A"), ("news", "News")]:
        section = summary[key]
        lines.extend(_report_section(title, section))
    lines.extend(
        [
            "## Final Recommendation",
            "",
            f"- MD&A main weight: {summary['mda']['recommendation']['main_weight']}",
            f"- MD&A robustness weights: {', '.join(summary['mda']['recommendation']['robustness_weights'])}",
            f"- News main weight: {summary['news']['recommendation']['main_weight']}",
            f"- News robustness weights: {', '.join(summary['news']['recommendation']['robustness_weights'])}",
            "- Do not lower a dimension weight purely because of scoring errors; mark it as a scoring reliability issue first.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _report_section(title: str, section: dict[str, Any]) -> list[str]:
    recommendation = section["recommendation"]
    lines = [
        f"## {title}",
        "",
        f"- input: {section['input_path']}",
        f"- output: {section['output_path']}",
        f"- scored documents: {section['document_count']}",
        f"- skipped invalid rows: {section['skipped_invalid_rows']}",
        f"- recommend W2_main_recommended: {recommendation['adopt_w2']}",
        f"- stability judgement: {recommendation['reason']}",
        f"- adjustment guidance: {recommendation['adjustment_needed']}",
        "",
        "### Scheme Correlations",
        "",
        "| schemes | Pearson | Spearman | top 20% overlap | bottom 20% overlap | judgement |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in section["pairwise_metrics"]:
        lines.append(
            f"| {item['scheme_a']} vs {item['scheme_b']} | {item['pearson']:.3f} | {item['spearman']:.3f} | "
            f"{item['top_20_overlap']['ratio']:.3f} | {item['bottom_20_overlap']['ratio']:.3f} | {item['stability_judgement']} |"
        )
    if not section["pairwise_metrics"]:
        lines.append("| insufficient data |  |  |  |  |  |")
    lines.extend(
        [
            "",
            f"- documents with rank change > 10: {section['rank_change_gt_10_count']}",
            f"- documents with score change > 10 points: {section['score_change_gt_10_count']}",
            "",
            "### Most Weight-Sensitive Documents",
            "",
            "| document_id | max_rank_change | max_score_change |",
            "|---|---:|---:|",
        ]
    )
    for item in section["sensitive_documents"][:10]:
        lines.append(f"| {item['document_id']} | {item['max_rank_change']} | {item['max_score_change']:.3f} |")
    if not section["sensitive_documents"]:
        lines.append("| none | 0 | 0 |")
    lines.extend(_dimension_diagnostics(section))
    return lines


def _dimension_diagnostics(section: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "### Dimension Mean And Standard Deviation",
        "",
        "| dimension | mean | std | warning |",
        "|---|---:|---:|---|",
    ]
    low_variance = {item["dimension"] for item in section["low_variance_dimensions"]}
    for item in section["dimension_stats"]:
        warning = "low variance" if item["dimension"] in low_variance else ""
        lines.append(f"| {item['dimension']} | {item['mean']:.3f} | {item['std']:.3f} | {warning} |")
    lines.extend(
        [
            "",
            "### Possible Redundancy",
            "",
        ]
    )
    if section["high_correlation_pairs"]:
        for item in section["high_correlation_pairs"]:
            lines.append(
                f"- {item['dimension_a']} and {item['dimension_b']}: correlation={item['correlation']:.3f}, possible redundancy; do not auto-delete."
            )
    else:
        lines.append("- No dimension pair exceeded the 0.85 high-correlation warning threshold.")
    lines.extend(
        [
            "",
            "### Scoring Reliability Issues",
            "",
        ]
    )
    if section["scoring_reliability_issues"]:
        for item in section["scoring_reliability_issues"]:
            lines.append(
                f"- {item['dimension']}: review_rate={item['review_rate']:.3f}; scoring reliability issue, review errors before changing weights."
            )
    else:
        lines.append("- No high dimension-level scoring reliability issue flagged from review_log.csv.")
    lines.extend(["", "### Dimension Correlation Matrix", ""])
    matrix = section["dimension_correlation_matrix"]
    dimensions = list(matrix.keys())
    if not dimensions:
        lines.append("No scored dimensions available.")
        return lines
    lines.append("| dimension | " + " | ".join(dimensions) + " |")
    lines.append("|---" + "|---:" * len(dimensions) + "|")
    for left in dimensions:
        values = " | ".join(f"{matrix[left][right]:.3f}" for right in dimensions)
        lines.append(f"| {left} | {values} |")
    lines.append("")
    return lines


def _stability_judgement(spearman: float) -> str:
    if spearman >= 0.90:
        return "stable"
    if spearman >= 0.80:
        return "acceptable_with_sensitivity_disclosure"
    return "weight_sensitive_manual_review"


def _pearson(left: list[float], right: list[float], constant_result: float | None = None) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_den = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_den = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_den == 0 or right_den == 0:
        if constant_result is not None:
            return constant_result
        return 1.0 if left == right else 0.0
    return numerator / (left_den * right_den)


def _has_dimension_scores(row: dict[str, str], dimensions: list[str]) -> bool:
    return all(_is_number(row.get(dimension)) and 1 <= _safe_float(row.get(dimension)) <= 5 for dimension in dimensions)


def _has_pair(row: dict[str, str], left: str, right: str) -> bool:
    return _is_number(row.get(left)) and _is_number(row.get(right))


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _format_number(value: float) -> str:
    rounded = round(float(value), 6)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.6f}".rstrip("0").rstrip(".")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run weight calibration and sensitivity analysis on existing document scores.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--mda-input-path", type=Path)
    parser.add_argument("--news-input-path", type=Path)
    args = parser.parse_args()
    summary = run_weight_sensitivity(args.root, args.mda_input_path, args.news_input_path)
    print(f"Wrote weight sensitivity report: {summary['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
