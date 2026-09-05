from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import (
    dimension_summary,
    ensure_paper_output_dirs,
    grade_distribution,
    load_json_safe,
    load_paper_status,
    score_summary,
    write_table,
)
from scoring_utils import SCORING_ROOT, read_csv_rows
from stability_common import safe_float, write_markdown


def build_paper_results_tables(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = load_paper_status(root)
    metadata, _ = load_json_safe(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json")
    mda_rows = _first_nonempty(root, ["FINAL_OUTPUT/mda_final_dataset.csv", "06_ratings/mda_document_scores_final_candidate.csv", "06_ratings/mda_document_scores.csv"])
    news_rows = _first_nonempty(root, ["FINAL_OUTPUT/news_final_dataset.csv", "06_ratings/news_v2_full/news_v2_full_document_scores.csv"])
    news_broad_rows = read_csv_rows(root / "FINAL_OUTPUT" / "news_broad_dataset.csv") or news_rows
    news_company_focused_rows = read_csv_rows(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv")
    news_suspect_rows = read_csv_rows(root / "08_reports" / "news_suspect_not_company_focused.csv")

    summary = score_summary(mda_rows)
    grades = grade_distribution(mda_rows)
    table_6 = [{**summary, "grade_A": grades["A"], "grade_B": grades["B"], "grade_C": grades["C"], "grade_D": grades["D"]}]
    write_table(root, "table_6_mda_score_summary", ["N", "mean", "median", "std", "min", "max", "p25", "p75", "grade_A", "grade_B", "grade_C", "grade_D"], table_6)

    if news_rows:
        news_summary = score_summary(news_rows)
        news_grades = grade_distribution(news_rows)
        table_7 = [{**news_summary, "grade_A": news_grades["A"], "grade_B": news_grades["B"], "grade_C": news_grades["C"], "grade_D": news_grades["D"]}]
        news_note = "News score summary uses the broad manually preserved sample; company-focused strict subset is reported in the final freeze summary."
    else:
        table_7 = [{"N": 0, "mean": "", "median": "", "std": "", "min": "", "max": "", "p25": "", "p75": "", "grade_A": 0, "grade_B": 0, "grade_C": 0, "grade_D": 0}]
        news_note = "Missing note: real News final scores are unavailable; this table is intentionally empty."
    write_table(root, "table_7_news_score_summary", ["N", "mean", "median", "std", "min", "max", "p25", "p75", "grade_A", "grade_B", "grade_C", "grade_D"], table_7, news_note)

    write_table(root, "table_8_mda_dimension_summary", ["dimension", "N", "mean", "median", "std", "min", "max"], dimension_summary(mda_rows, ["AR01", "AR02", "AR03", "AR04", "AR05"]))
    news_dim_note = "" if news_rows else "Missing note: News dimensions are unavailable because no real News final scores exist."
    write_table(root, "table_9_news_dimension_summary", ["dimension", "N", "mean", "median", "std", "min", "max"], dimension_summary(news_rows, ["N01", "N02", "N03", "N04", "N05"]), news_dim_note)

    table_10 = _stability_rows(root)
    write_table(root, "table_10_stability_summary", ["component", "status", "metric", "value", "notes"], table_10)
    table_11 = _freeze_rows(status, metadata)
    write_table(root, "table_11_final_freeze_summary", ["metric", "value", "notes"], table_11)
    _write_figures(root, mda_rows, news_rows)
    write_markdown(
        root / "PAPER_OUTPUT" / "03_results" / "results_summary.md",
        [
            "# Results Tables Summary",
            "",
            f"- MD&A final rows used: {len(mda_rows)}",
            f"- News broad final rows used: {len(news_broad_rows)}",
            f"- News company-focused strict subset rows: {len(news_company_focused_rows)}",
            f"- News suspect/manual-review rows outside strict subset: {len(news_suspect_rows)}",
            f"- paper_ready: {str(status['paper_ready']).lower()}",
            f"- freeze_allowed: {str(status['freeze_allowed']).lower()}",
            "- The broad News sample must not be described as fully clean company-specific News.",
            "- Final empirical conclusions are not allowed when paper_ready is false.",
        ],
    )
    return {"mda_rows": len(mda_rows), "news_rows": len(news_rows), "paper_ready": status["paper_ready"]}


def _first_nonempty(root: Path, rel_paths: list[str]) -> list[dict[str, str]]:
    for rel_path in rel_paths:
        rows = read_csv_rows(root / rel_path)
        if rows:
            return rows
    return []


def _stability_rows(root: Path) -> list[dict[str, Any]]:
    report = root / "11_stability_analysis" / "reports" / "statistical_stability_report.md"
    high_vol = root / "11_stability_analysis" / "outputs" / "high_volatility_cases.csv"
    rows = [
        {"component": "weight sensitivity", "status": "available" if report.exists() else "missing", "metric": "report", "value": str(report.exists()).lower(), "notes": "See statistical stability report"},
        {"component": "repeatability", "status": "available" if (root / "11_stability_analysis" / "outputs" / "mda_repeatability_comparison.csv").exists() else "missing-input", "metric": "repeat runs", "value": "", "notes": "Requires repeat-run scores"},
        {"component": "version stability", "status": "available" if (root / "11_stability_analysis" / "outputs" / "mda_version_comparison.csv").exists() else "missing-input", "metric": "v1-v2", "value": "", "notes": "Requires v1 and v2 scores"},
        {"component": "structure stability", "status": "available" if (root / "11_stability_analysis" / "outputs" / "mda_structure_comparison.csv").exists() else "missing-input", "metric": "document vs single-dimension", "value": "", "notes": "Requires structure comparison inputs"},
        {"component": "high-volatility cases", "status": "available" if high_vol.exists() else "missing", "metric": "row_count", "value": len(read_csv_rows(high_vol)), "notes": "Manual review list"},
    ]
    return rows


def _freeze_rows(status: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    excluded = metadata.get("excluded_documents_count", {}) if isinstance(metadata, dict) else {}
    news_meta = metadata.get("news", {}) if isinstance(metadata, dict) else {}
    robustness = news_meta.get("broad_vs_company_focused_robustness", {}) if isinstance(news_meta, dict) else {}
    cross_meta = metadata.get("cross_validation", {}) if isinstance(metadata, dict) else {}
    return [
        {"metric": "freeze_allowed", "value": status["freeze_allowed"], "notes": "Final freeze gate"},
        {"metric": "paper_ready", "value": status["paper_ready"], "notes": "Paper readiness gate"},
        {"metric": "blocking_reasons", "value": "; ".join(status["blocking_reasons"]), "notes": "Must be resolved before final empirical claims"},
        {"metric": "mda_final_dataset_size", "value": status["mda_final_dataset_size"], "notes": "Final output rows"},
        {"metric": "news_final_dataset_size", "value": status["news_final_dataset_size"], "notes": "Real scored News rows only"},
        {"metric": "news_broad_dataset_size", "value": robustness.get("broad_rows", news_meta.get("broad_count", "")), "notes": "Broad manually preserved real-News sample"},
        {"metric": "news_company_focused_dataset_size", "value": robustness.get("company_focused_rows", news_meta.get("company_focused_count", "")), "notes": "Strict company-focused subset for robustness"},
        {"metric": "news_suspect_or_manual_review_count", "value": robustness.get("suspect_or_manual_review_rows", news_meta.get("suspect_not_company_focused_count", "")), "notes": "Excluded from strict company-focused sample before validation"},
        {"metric": "excluded_mda_documents", "value": excluded.get("mda", ""), "notes": "Final freeze exclusions"},
        {"metric": "excluded_news_documents", "value": excluded.get("news", ""), "notes": "Final freeze exclusions"},
        {"metric": "synthetic_count", "value": status["synthetic_news_count"], "notes": "Excluded from empirical analysis"},
        {"metric": "claim_link_count", "value": status["claim_link_count"], "notes": "Cross-validation links"},
        {"metric": "cross_validation_pipeline_ready", "value": status.get("cross_validation_pipeline_ready", ""), "notes": "Pipeline ran and produced link artifacts"},
        {"metric": "cross_validation_empirical_ready", "value": status.get("cross_validation_empirical_ready", ""), "notes": "False when claim scoring is deterministic/mock or unresolved"},
        {"metric": "cross_validation_ready", "value": status.get("cross_validation_ready", ""), "notes": "Overall readiness gate"},
        {"metric": "mock_mode", "value": status.get("mock_mode", cross_meta.get("mock_mode", "")), "notes": "Claim-level scoring mode"},
        {"metric": "systemic_conflict_flag", "value": status.get("systemic_conflict_flag", cross_meta.get("systemic_conflict_flag", "")), "notes": "Unresolved claim-level conflict flag"},
    ]


def _write_figures(root: Path, mda_rows: list[dict[str, str]], news_rows: list[dict[str, str]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        write_markdown(root / "PAPER_OUTPUT" / "figures" / "figure_generation_missing.md", [f"# Figure Generation Missing", "", str(exc)])
        return
    fig_dir = root / "PAPER_OUTPUT" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _histogram(plt, [safe_float(row.get("total_score_100", ""), 0.0) for row in mda_rows], fig_dir / "figure_mda_score_distribution.png", "MD&A Score Distribution")
    means = dimension_summary(mda_rows, ["AR01", "AR02", "AR03", "AR04", "AR05"])
    _bar(plt, [row["dimension"] for row in means], [safe_float(row.get("mean", ""), 0.0) for row in means], fig_dir / "figure_mda_dimension_means.png", "MD&A Dimension Means")
    if news_rows:
        _histogram(plt, [safe_float(row.get("total_score_100", ""), 0.0) for row in news_rows], fig_dir / "figure_news_score_distribution.png", "News Score Distribution")
    weight_path = root / "11_stability_analysis" / "outputs" / "mda_weight_sensitivity_scores.csv"
    weight_rows = read_csv_rows(weight_path)
    if weight_rows and "score_W1_baseline" in weight_rows[0] and "score_W2_main_recommended" in weight_rows[0]:
        xs = [safe_float(row.get("score_W1_baseline", ""), 0.0) for row in weight_rows]
        ys = [safe_float(row.get("score_W2_main_recommended", ""), 0.0) for row in weight_rows]
        plt.figure(figsize=(5, 4))
        plt.scatter(xs, ys)
        plt.xlabel("W1 baseline")
        plt.ylabel("W2 main")
        plt.title("Weight Sensitivity Scatter")
        plt.tight_layout()
        plt.savefig(fig_dir / "figure_weight_sensitivity_scatter.png")
        plt.close()
    high_vol = read_csv_rows(root / "11_stability_analysis" / "outputs" / "high_volatility_cases.csv")
    if high_vol:
        severities: dict[str, int] = {}
        for row in high_vol:
            severities[row.get("severity", "unknown")] = severities.get(row.get("severity", "unknown"), 0) + 1
        _bar(plt, list(severities), list(severities.values()), fig_dir / "figure_high_volatility_cases.png", "High-volatility Cases")


def _histogram(plt: Any, values: list[float], path: Path, title: str) -> None:
    plt.figure(figsize=(5, 4))
    plt.hist(values or [0.0], bins=min(10, max(1, len(values) or 1)))
    plt.title(title)
    plt.xlabel("score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _bar(plt: Any, labels: list[str], values: list[float], path: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.bar(labels or ["missing"], values or [0.0])
    plt.title(title)
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper result tables and figures.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_results_tables(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
