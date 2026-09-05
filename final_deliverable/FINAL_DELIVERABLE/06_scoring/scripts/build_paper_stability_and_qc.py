from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import ensure_paper_output_dirs, load_json_safe, load_paper_status, write_dual_language_sections
from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


def build_paper_stability_and_qc(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = load_paper_status(root)
    metadata, _ = load_json_safe(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json")
    high_vol_path = root / "11_stability_analysis" / "outputs" / "high_volatility_cases.csv"
    high_vol_rows = read_csv_rows(high_vol_path)
    write_csv_rows(
        root / "PAPER_OUTPUT" / "05_quality_control" / "high_volatility_cases_for_review.csv",
        ["case_id", "document_id", "doc_type", "ticker", "company_name", "year_or_publish_date", "issue_type", "severity", "score_impact", "rank_impact", "dimension_affected", "recommended_action", "source_file"],
        high_vol_rows,
    )
    stability_available = (root / "11_stability_analysis" / "reports" / "statistical_stability_report.md").exists()
    news_robustness, _ = load_json_safe(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json")
    news_robustness_metrics = (
        news_robustness.get("news", {}).get("broad_vs_company_focused_robustness", {})
        if isinstance(news_robustness, dict)
        else {}
    )
    news_stability_status = _news_stability_status(root)
    review_count = len(read_csv_rows(root / "07_review" / "review_log.csv"))
    en_stability = [
        "# Stability Section",
        "",
        f"Weight sensitivity: {'available' if stability_available else 'missing; statistical stability report not found'}.",
        "Repeatability: reported when repeat-run inputs exist; otherwise marked missing-input.",
        "Version comparison: reported when v1 and v2 score inputs exist; otherwise marked missing-input.",
        "Structure comparison: reported when document-level and single-dimension score inputs exist; otherwise marked missing-input.",
        "News aggregation stability: reported only when real News scores exist.",
        f"News repeatability: {news_stability_status}.",
        f"News broad-vs-company-focused robustness: broad={news_robustness_metrics.get('broad_rows', '')}, strict={news_robustness_metrics.get('company_focused_rows', '')}, suspect/manual-review={news_robustness_metrics.get('suspect_or_manual_review_rows', '')}, rank_spearman={news_robustness_metrics.get('rank_spearman_shared_docs', '')}, top_20_overlap={news_robustness_metrics.get('top_20_overlap', '')}, bottom_20_overlap={news_robustness_metrics.get('bottom_20_overlap', '')}.",
        f"High-volatility cases: {len(high_vol_rows)} rows for review.",
        f"Final freeze result: freeze_allowed={status['freeze_allowed']}, paper_ready={status['paper_ready']}.",
    ]
    cn_stability = [
        "# 稳定性部分",
        "",
        f"权重敏感性：{'已生成' if stability_available else '缺失，未找到 statistical stability report'}。",
        "重复评分稳定性：只有 repeat-run 输入存在时才报告，否则标记 missing-input。",
        "版本稳定性：只有 v1/v2 输入存在时才报告，否则标记 missing-input。",
        "结构稳定性：只有 document-level 与 single-dimension 输入存在时才报告。",
        "News aggregation stability：仅在真实 News 分数存在时报告。",
        f"News repeatability：{news_stability_status}。",
        f"News broad-vs-company-focused robustness：broad={news_robustness_metrics.get('broad_rows', '')}，strict={news_robustness_metrics.get('company_focused_rows', '')}，suspect/manual-review={news_robustness_metrics.get('suspect_or_manual_review_rows', '')}，rank_spearman={news_robustness_metrics.get('rank_spearman_shared_docs', '')}，top_20_overlap={news_robustness_metrics.get('top_20_overlap', '')}，bottom_20_overlap={news_robustness_metrics.get('bottom_20_overlap', '')}。",
        f"High-volatility cases：{len(high_vol_rows)} 行需要复核。",
        f"Final freeze result：freeze_allowed={status['freeze_allowed']}，paper_ready={status['paper_ready']}。",
    ]
    en_qc = [
        "# Quality Control Section",
        "",
        "JSON/schema/evidence validation: final documents must pass parsing, schema validation, evidence checks, and score range checks.",
        "AR02 numeric checker: AR02 is controlled by Python numeric checks and review flags; it is not an external financial-statement audit.",
        f"Review log rows: {review_count}.",
        f"System-level error rate: {status['system_level_error_rate']}.",
        f"MD&A stability success rate: {status['mda_stability_success_rate']}.",
        f"Final blockers: {', '.join(status['blocking_reasons']) or 'none'}.",
        f"Final freeze metadata available: {str(bool(metadata)).lower()}.",
    ]
    cn_qc = [
        "# 质量控制部分",
        "",
        "JSON/schema/evidence validation：final 文档必须通过 parse、schema、evidence 与 score range 检查。",
        "AR02 numeric checker：AR02 由 Python 数值检查与 review flag 控制，不是财务报表审计。",
        f"Review log 行数：{review_count}。",
        f"System-level error rate：{status['system_level_error_rate']}。",
        f"MD&A stability success rate：{status['mda_stability_success_rate']}。",
        f"Final blockers：{', '.join(status['blocking_reasons']) or 'none'}。",
        f"Final freeze metadata available：{str(bool(metadata)).lower()}。",
    ]
    outputs = {}
    outputs.update(write_dual_language_sections(root, "04_stability", "stability_section_cn.md", "stability_section_en.md", cn_stability, en_stability))
    outputs.update(write_dual_language_sections(root, "05_quality_control", "quality_control_section_cn.md", "quality_control_section_en.md", cn_qc, en_qc))
    return outputs


def _news_stability_status(root: Path) -> str:
    report = root / "11_stability_analysis" / "reports" / "news_stability_report.md"
    if not report.exists():
        return "not_completed; news_stability_report.md missing"
    text = report.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("- repeatability_status:"):
            status = line.split(":", 1)[1].strip() or "not_completed"
            if status in {"not_completed", "insufficient_sample", "not_measured", "model_unavailable"}:
                return f"{status}; not used as a passed repeatability result"
            return status
    return "not_completed; repeatability_status not recorded"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper stability and quality-control sections.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_stability_and_qc(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
