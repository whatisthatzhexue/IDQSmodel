from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_stability_report import build_stability_report
from scoring_utils import SCORING_ROOT, save_json
from stability_common import (
    discover_inputs,
    ensure_stability_dirs,
    finalize_input_manifest,
    write_correlation_metrics,
    write_high_volatility,
)
from stability_news_aggregation import run_news_aggregation_stability
from stability_repeatability import run_repeatability_analysis
from stability_structure_comparison import run_structure_comparison
from stability_version_comparison import run_version_comparison
from stability_weight_sensitivity import run_weight_stability


def run_statistical_stability_analysis(
    root: Path = SCORING_ROOT,
    include_mda: bool = True,
    include_news: bool = True,
    include_repeatability: bool = True,
    include_version_comparison: bool = True,
    include_structure_comparison: bool = True,
    include_news_aggregation: bool = True,
    make_plots: bool = True,
) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    manifest = discover_inputs(root)
    summary: dict[str, Any] = {"input_manifest": manifest}
    high_volatility: list[dict[str, Any]] = []

    weight_summary = run_weight_stability(root=root, include_mda=include_mda, include_news=include_news, make_plots=make_plots, manifest=manifest)
    summary["mda_weight_sensitivity"] = weight_summary.get("mda", {})
    summary["news_weight_sensitivity"] = weight_summary.get("news", {})
    high_volatility.extend(weight_summary.get("high_volatility_cases", []))

    if include_repeatability:
        repeatability = run_repeatability_analysis(root=root, include_mda=include_mda, include_news=include_news, make_plots=make_plots, manifest=manifest)
    else:
        repeatability = {"mda": {"status": "skipped"}, "news": {"status": "skipped"}, "high_volatility_cases": []}
    summary["repeatability"] = repeatability
    high_volatility.extend(repeatability.get("high_volatility_cases", []))

    if include_version_comparison:
        version = run_version_comparison(root=root, include_mda=include_mda, make_plots=make_plots, manifest=manifest)
    else:
        version = {"mda": {"status": "skipped"}, "high_volatility_cases": []}
    summary["version_comparison"] = version
    high_volatility.extend(version.get("high_volatility_cases", []))

    if include_structure_comparison:
        structure = run_structure_comparison(root=root, include_mda=include_mda, include_news=include_news, manifest=manifest)
    else:
        structure = {"mda": {"status": "skipped"}, "news": {"status": "skipped"}, "high_volatility_cases": []}
    summary["structure_comparison"] = structure
    high_volatility.extend(structure.get("high_volatility_cases", []))

    if include_news and include_news_aggregation:
        news_aggregation = run_news_aggregation_stability(root=root, include_news=include_news, make_plots=make_plots, manifest=manifest)
    else:
        news_aggregation = {"status": "skipped", "high_volatility_cases": []}
    summary["news_aggregation"] = news_aggregation
    high_volatility.extend(news_aggregation.get("high_volatility_cases", []))

    high_volatility_path = write_high_volatility(root, high_volatility)
    summary["high_volatility_cases_path"] = str(high_volatility_path)
    summary["high_volatility_case_count"] = len(high_volatility)
    write_correlation_metrics(root, summary)
    manifest = finalize_input_manifest(root, manifest)
    summary["input_manifest"] = manifest
    save_json(base / "outputs" / "statistical_stability_summary.json", summary)
    report_result = build_stability_report(root, summary)
    summary["report_path"] = report_result["report_path"]
    summary["final_recommendation"] = report_result["final_recommendation"]
    save_json(base / "outputs" / "statistical_stability_summary.json", summary)
    return summary


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run statistical stability analysis without rescoring.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--include-mda", type=parse_bool, default=True)
    parser.add_argument("--include-news", type=parse_bool, default=True)
    parser.add_argument("--include-repeatability", type=parse_bool, default=True)
    parser.add_argument("--include-version-comparison", type=parse_bool, default=True)
    parser.add_argument("--include-structure-comparison", type=parse_bool, default=True)
    parser.add_argument("--include-news-aggregation", type=parse_bool, default=True)
    parser.add_argument("--make-plots", type=parse_bool, default=True)
    args = parser.parse_args()
    summary = run_statistical_stability_analysis(
        root=args.root,
        include_mda=args.include_mda,
        include_news=args.include_news,
        include_repeatability=args.include_repeatability,
        include_version_comparison=args.include_version_comparison,
        include_structure_comparison=args.include_structure_comparison,
        include_news_aggregation=args.include_news_aggregation,
        make_plots=args.make_plots,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
