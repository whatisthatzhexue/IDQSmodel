from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, load_json, read_csv_rows
from stability_common import ensure_stability_dirs, stability_root, write_markdown


def build_stability_report(root: Path = SCORING_ROOT, summary: dict[str, Any] | None = None) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    if summary is None:
        summary_path = base / "outputs" / "statistical_stability_summary.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
    manifest = summary.get("input_manifest") or _load_optional_json(base / "inputs" / "input_manifest.json", {})
    high_volatility = read_csv_rows(base / "outputs" / "high_volatility_cases.csv")
    final = _final_recommendation(summary, manifest, high_volatility)
    lines = [
        "# Statistical Stability Report",
        "",
        "## Executive Summary",
        f"- MD&A weight sensitivity: {_status(summary, 'mda_weight_sensitivity')}",
        f"- News weight sensitivity: {_status(summary, 'news_weight_sensitivity')}",
        f"- Repeatability: {_status(summary, 'repeatability')}",
        f"- Version stability: {_status(summary, 'version_comparison')}",
        f"- Structure comparison: {_status(summary, 'structure_comparison')}",
        f"- News aggregation: {_status(summary, 'news_aggregation')}",
        f"- High-volatility cases: {len(high_volatility)}",
        f"- Final candidate readiness: {final['final_candidate_readiness']}",
        "",
        "## Available Input Files",
    ]
    for item in manifest.get("available_inputs", []):
        lines.append(f"- {item.get('relative_path', item.get('path', ''))}: rows={item.get('row_count', 0)}")
    lines.extend(["", "## Missing Input Files"])
    missing = manifest.get("missing_inputs", [])
    if missing:
        for item in missing:
            lines.append(f"- {item.get('relative_path', item.get('path', ''))}")
    else:
        lines.append("- none recorded")
    lines.extend(["", "## MD&A Weight Sensitivity"])
    lines.extend(_weight_section(_weight_item(summary, "mda_weight_sensitivity", "mda")))
    lines.extend(["", "## News Weight Sensitivity"])
    lines.extend(_weight_section(_weight_item(summary, "news_weight_sensitivity", "news")))
    lines.extend(["", "## MD&A Repeatability"])
    lines.extend(_repeat_section(summary.get("repeatability", {}).get("mda", {})))
    lines.extend(["", "## News Repeatability"])
    lines.extend(_repeat_section(summary.get("repeatability", {}).get("news", {})))
    lines.extend(["", "## MD&A Version Stability"])
    lines.extend(_simple_section(summary.get("version_comparison", {}).get("mda", {}), ["status", "document_count", "moderate_or_major_count", "dimension_shift_count", "reason"]))
    lines.extend(["", "## News Aggregation Stability"])
    lines.extend(_simple_section(summary.get("news_aggregation", {}), ["status", "group_count", "sensitive_group_count", "low_news_count_group_count", "reason"]))
    lines.extend(["", "## Structure Comparison: Document-Level Vs Single-Dimension"])
    for doc_type in ["mda", "news"]:
        lines.append(f"### {doc_type.upper()}")
        lines.extend(_simple_section(summary.get("structure_comparison", {}).get(doc_type, {}), ["status", "document_count", "pearson", "spearman", "recommendation", "reason"]))
    lines.extend(["", "## High-Volatility Document List"])
    if high_volatility:
        for row in high_volatility[:30]:
            lines.append(f"- {row.get('doc_type', '')} {row.get('document_id', '')}: {row.get('issue_type', '')}, severity={row.get('severity', '')}, action={row.get('recommended_action', '')}")
    else:
        lines.append("- none")
    lines.extend(["", "## High-Risk Dimensions"])
    lines.extend(_high_risk_dimensions(summary))
    lines.extend(["", "## Final Recommendation"])
    for key, value in final.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Limitations",
            "- This module performs statistical stability analysis only.",
            "- It does not call qwen3:8b, does not rescore documents, and does not average MD&A with News.",
            "- Missing input files are reported as unavailable analyses instead of being imputed.",
        ]
    )
    report_path = base / "reports" / "statistical_stability_report.md"
    write_markdown(report_path, lines)
    return {"report_path": str(report_path), "final_recommendation": final}


def _status(summary: dict[str, Any], key: str) -> str:
    item = summary.get(key, {})
    if isinstance(item, dict):
        if "status" in item:
            return str(item["status"])
        statuses = [str(value.get("status")) for value in item.values() if isinstance(value, dict) and value.get("status")]
        return ",".join(statuses) if statuses else "unknown"
    return "unknown"


def _weight_section(item: dict[str, Any]) -> list[str]:
    lines = _simple_section(item, ["status", "input_path", "document_count"])
    recommendation = item.get("recommendation", {})
    lines.append(f"- W2_main_recommended recommendation: {recommendation.get('reason', '')}")
    lines.append(f"- adopt_w2: {recommendation.get('adopt_w2', False)}")
    lines.append(f"- low_variance_dimensions: {', '.join(dim.get('dimension', '') for dim in item.get('low_variance_dimensions', [])) or 'none'}")
    lines.append(f"- high_correlation_pairs: {len(item.get('high_correlation_pairs', []))}")
    return lines


def _repeat_section(item: dict[str, Any]) -> list[str]:
    return _simple_section(item, ["status", "document_count", "pearson", "spearman", "icc", "mean_absolute_difference", "reason"])


def _simple_section(item: dict[str, Any], keys: list[str]) -> list[str]:
    if not item:
        return ["- status: unavailable"]
    return [f"- {key}: {item.get(key, '')}" for key in keys]


def _high_risk_dimensions(summary: dict[str, Any]) -> list[str]:
    dims = []
    for item in [_weight_item(summary, "mda_weight_sensitivity", "mda"), _weight_item(summary, "news_weight_sensitivity", "news")]:
        for dim in item.get("low_variance_dimensions", []):
            dims.append(f"- {dim.get('dimension')}: low discrimination warning")
        for pair in item.get("high_correlation_pairs", []):
            dims.append(f"- {pair.get('dimension_a')} / {pair.get('dimension_b')}: possible redundancy")
    return dims or ["- none"]


def _final_recommendation(summary: dict[str, Any], manifest: dict[str, Any], high_volatility: list[dict[str, str]]) -> dict[str, str]:
    mda_weight = _weight_item(summary, "mda_weight_sensitivity", "mda")
    news_weight = _weight_item(summary, "news_weight_sensitivity", "news")
    news_missing = news_weight.get("status") in {"missing-news", "missing-input", "empty-input"} or summary.get("news_aggregation", {}).get("status") == "missing-news"
    blockers = []
    if high_volatility:
        blockers.append("high-volatility cases require review")
    if mda_weight.get("status") not in {"success", "skipped"}:
        blockers.append("MD&A stability input missing")
    if news_missing:
        blockers.append("News input missing; cross-validation cannot run fully")
    return {
        "current_main_weight_reasonable": str(mda_weight.get("recommendation", {}).get("adopt_w2", False) or news_weight.get("recommendation", {}).get("adopt_w2", False)),
        "recommend_W2_main_recommended": str(mda_weight.get("recommendation", {}).get("main_weight", "W2_main_recommended") == "W2_main_recommended"),
        "most_sensitive_dimensions": _sensitive_dimensions(summary),
        "manual_review_needed": str(len(high_volatility)),
        "final_candidate_readiness": "ready_after_review" if not blockers else "blocked",
        "blocking_reasons": "; ".join(blockers) if blockers else "none",
        "news_missing": str(news_missing),
        "cross_validation_can_run": str(not news_missing),
    }


def _sensitive_dimensions(summary: dict[str, Any]) -> str:
    dims = []
    for item in [_weight_item(summary, "mda_weight_sensitivity", "mda"), _weight_item(summary, "news_weight_sensitivity", "news")]:
        dims.extend(dim.get("dimension", "") for dim in item.get("low_variance_dimensions", []))
        for pair in item.get("high_correlation_pairs", []):
            dims.append(f"{pair.get('dimension_a')}/{pair.get('dimension_b')}")
    return ", ".join(dim for dim in dims if dim) or "none"


def _load_optional_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return load_json(path)


def _weight_item(summary: dict[str, Any], key: str, nested_key: str) -> dict[str, Any]:
    item = summary.get(key, {})
    if isinstance(item, dict) and item.get("status"):
        return item
    if isinstance(item, dict):
        nested = item.get(nested_key, {})
        return nested if isinstance(nested, dict) else {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build unified statistical stability report.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_stability_report(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
