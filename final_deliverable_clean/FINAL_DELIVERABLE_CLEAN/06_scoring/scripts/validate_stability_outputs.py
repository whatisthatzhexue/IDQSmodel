from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, load_json, save_json
from stability_common import sha256_file, stability_root, write_markdown
from stability_weight_sensitivity import MDA_WEIGHTS, NEWS_WEIGHTS


def validate_stability_outputs(root: Path = SCORING_ROOT) -> dict[str, Any]:
    base = stability_root(root)
    issues: list[str] = []
    csv_files = sorted((base / "outputs").glob("*.csv"))
    for path in csv_files:
        _validate_csv(path, issues)
    _validate_correlations(base / "outputs" / "correlation_metrics.json", issues)
    _validate_weights(issues)
    _validate_high_volatility(base / "outputs" / "high_volatility_cases.csv", issues)
    _validate_manifest(base / "inputs" / "input_manifest.json", issues)
    _validate_reports(base, issues)
    _validate_plots(base, issues)
    summary = {"passed": not issues, "issues": issues, "csv_count": len(csv_files)}
    save_json(base / "logs" / "validate_stability_outputs.json", summary)
    _write_report(base / "reports" / "validate_stability_outputs.md", summary)
    return summary


def _validate_csv(path: Path, issues: list[str]) -> None:
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as exc:  # noqa: BLE001 - validator should report all issues.
        issues.append(f"csv_unreadable:{path}:{exc}")
        return
    for row_idx, row in enumerate(rows, start=2):
        for field, value in row.items():
            if value == "":
                continue
            lower = field.lower()
            if lower.startswith("score_version"):
                continue
            if field.startswith("score_") or lower.endswith("_score") or lower in {"mean_score", "median_score", "trimmed_mean_score", "aggregation_score_range", "total_score_diff", "abs_total_score_diff", "max_score_change", "score_change_w1_w2", "score_change_w2_w3"}:
                _require_number(value, issues, path, row_idx, field)
            if field.startswith("rank_") or lower.endswith("_rank"):
                _require_int(value, issues, path, row_idx, field)


def _require_number(value: str, issues: list[str], path: Path, row_idx: int, field: str) -> None:
    try:
        float(value)
    except ValueError:
        issues.append(f"non_numeric_score:{path}:{row_idx}:{field}:{value}")


def _require_int(value: str, issues: list[str], path: Path, row_idx: int, field: str) -> None:
    try:
        int(float(value))
    except ValueError:
        issues.append(f"non_integer_rank:{path}:{row_idx}:{field}:{value}")


def _validate_correlations(path: Path, issues: list[str]) -> None:
    if not path.exists():
        issues.append("missing_correlation_metrics")
        return
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(f"correlation_metrics_unreadable:{exc}")
        return
    for idx, item in enumerate(metrics):
        value = item.get("value")
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            issues.append(f"correlation_not_numeric:{idx}:{value}")
            continue
        if number < -1 or number > 1:
            issues.append(f"correlation_out_of_bounds:{idx}:{number}")


def _validate_weights(issues: list[str]) -> None:
    for group_name, group in [("mda", MDA_WEIGHTS), ("news", NEWS_WEIGHTS)]:
        for scheme, weights in group.items():
            total = sum(float(value) for value in weights.values())
            if abs(total - 1.0) > 0.000001:
                issues.append(f"weight_sum_not_one:{group_name}:{scheme}:{total}")


def _validate_high_volatility(path: Path, issues: list[str]) -> None:
    if not path.exists():
        issues.append("missing_high_volatility_cases")
        return
    seen = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("document_id", ""), row.get("doc_type", ""), row.get("issue_type", ""), row.get("dimension_affected", ""))
            if key in seen:
                issues.append(f"duplicate_high_volatility_case:{key}")
            seen.add(key)


def _validate_manifest(path: Path, issues: list[str]) -> None:
    if not path.exists():
        issues.append("missing_input_manifest")
        return
    manifest = load_json(path)
    if "missing_inputs" not in manifest:
        issues.append("missing_input_record_absent")
    if "available_inputs" not in manifest:
        issues.append("available_input_record_absent")
    if manifest.get("input_files_unchanged") is False:
        issues.append("input_file_modified")
    for path_text, before in manifest.get("sha256_before", {}).items():
        file_path = Path(path_text)
        if file_path.exists() and sha256_file(file_path) != before:
            issues.append(f"input_file_checksum_changed:{path_text}")


def _validate_reports(base: Path, issues: list[str]) -> None:
    required = [
        base / "reports" / "statistical_stability_report.md",
        base / "reports" / "weight_sensitivity_report.md",
        base / "reports" / "repeatability_report.md",
        base / "reports" / "mda_version_stability_report.md",
        base / "reports" / "structure_stability_report.md",
        base / "reports" / "news_aggregation_stability_report.md",
    ]
    for path in required:
        if not path.exists():
            issues.append(f"missing_report:{path}")


def _validate_plots(base: Path, issues: list[str]) -> None:
    required = [
        "mda_weight_scheme_scatter_W1_W2.png",
        "mda_weight_scheme_scatter_W2_W3.png",
        "news_weight_scheme_scatter_W1_W2.png",
        "news_weight_scheme_scatter_W2_W3.png",
        "mda_rank_change_distribution.png",
        "news_rank_change_distribution.png",
        "repeatability_score_diff_distribution.png",
        "version_score_diff_distribution.png",
        "news_aggregation_range_distribution.png",
    ]
    skip_text = (base / "plots" / "plot_skips.md").read_text(encoding="utf-8") if (base / "plots" / "plot_skips.md").exists() else ""
    for filename in required:
        if not (base / "plots" / filename).exists() and filename not in skip_text:
            issues.append(f"missing_plot_or_skip:{filename}")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stability Output Validation",
        "",
        f"- passed: {summary['passed']}",
        f"- csv_count: {summary['csv_count']}",
        f"- issues: {len(summary['issues'])}",
        "",
    ]
    for issue in summary["issues"]:
        lines.append(f"- {issue}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate statistical stability outputs.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    summary = validate_stability_outputs(args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
