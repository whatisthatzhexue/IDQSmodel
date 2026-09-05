from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from paper_utils import ensure_paper_output_dirs, load_json_safe, load_paper_status, real_news_rows, write_dual_language_sections, write_table
from scoring_utils import SCORING_ROOT, read_csv_rows


def build_paper_cross_validation_section(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = load_paper_status(root)
    real_news = real_news_rows(root)
    mda_claims = read_csv_rows(root / "09_cross_validation" / "mda_claims.csv")
    news_claims = read_csv_rows(root / "09_cross_validation" / "news_claims.csv")
    links = read_csv_rows(root / "09_cross_validation" / "claim_links.csv")
    final_metadata, _ = load_json_safe(root / "09_cross_validation" / "final_real_run" / "cross_validation_run_metadata.json")
    relation_counts = Counter(row.get("relation_type", "unknown") for row in links)
    source_counts = Counter(row.get("source_independence", "unknown") for row in links)
    mock_mode = bool(final_metadata.get("mock_mode")) or _report_indicates_mock(root)
    systemic_conflict = bool(final_metadata.get("systemic_conflict_flag"))
    pipeline_ready = bool(status.get("cross_validation_pipeline_ready")) or bool(links)
    empirical_ready = bool(status.get("cross_validation_empirical_ready")) and not mock_mode and not systemic_conflict
    summary_rows = [
        {"metric": "mda_claim_count", "value": len(mda_claims)},
        {"metric": "news_claim_count", "value": len(news_claims)},
        {"metric": "claim_link_count", "value": len(links)},
        {"metric": "strong_support", "value": sum(row.get("support_level", "").lower() == "strong" for row in links)},
        {"metric": "partial_support", "value": sum(row.get("support_level", "").lower() == "partial" for row in links)},
        {"metric": "contextual_support", "value": sum(row.get("support_level", "").lower() == "contextual" for row in links)},
        {"metric": "contradictions", "value": sum(row.get("contradiction_flag", "").lower() in {"1", "true", "yes"} for row in links)},
        {"metric": "possible_omissions", "value": sum(row.get("mda_omission_flag", "").lower() in {"1", "true", "yes"} for row in links)},
        {"metric": "same_source_repetitions", "value": sum(row.get("same_source_repetition_flag", "").lower() in {"1", "true", "yes"} for row in links)},
        {"metric": "mock_mode_or_deterministic_proxy", "value": mock_mode},
        {"metric": "systemic_conflict_flag", "value": systemic_conflict},
        {"metric": "source_independence_distribution", "value": dict(source_counts)},
    ]
    write_table(root, "table_12_cross_validation_summary", ["metric", "value"], summary_rows)
    if not real_news:
        interpretation = "Cross-validation cannot be interpreted as empirical evidence because real news data are not available."
    elif len(links) == 0:
        interpretation = "News scoring enters the document-level measurement model, but claim-level MD&A-News triangulation remains non-interpretable because no real claim links were produced."
    elif not empirical_ready:
        interpretation = "News scoring enters the document-level measurement model, but claim-level MD&A-News triangulation remains non-interpretable because claim scoring used deterministic proxy/mock mode or unresolved systemic conflict flags remain."
    else:
        interpretation = "Cross-validation has real News-backed links, but it remains auxiliary triangulation and should be interpreted separately from primary scores."
    en_lines = [
        "# Cross-validation Section",
        "",
        interpretation,
        f"Number of MD&A claims: {len(mda_claims)}.",
        f"Number of News claims: {len(news_claims)}.",
        f"Claim links: {len(links)}.",
        f"cross_validation_pipeline_ready: {str(pipeline_ready).lower()}.",
        f"cross_validation_empirical_ready: {str(empirical_ready).lower()}.",
        f"mock_mode_or_deterministic_proxy: {str(mock_mode).lower()}.",
        f"systemic_conflict_flag: {str(systemic_conflict).lower()}.",
        f"Relation distribution: {dict(relation_counts)}.",
        f"Source independence distribution: {dict(source_counts)}.",
        "Cross-validation is an auxiliary triangulation layer and does not alter MD&A or News scores.",
        "It should not be described as passed unless the readiness gate marks cross_validation_ready=true.",
    ]
    cn_lines = [
        "# Cross-validation 部分",
        "",
        interpretation,
        f"MD&A claims 数量：{len(mda_claims)}。",
        f"News claims 数量：{len(news_claims)}。",
        f"Claim links 数量：{len(links)}。",
        f"cross_validation_pipeline_ready：{str(pipeline_ready).lower()}。",
        f"cross_validation_empirical_ready：{str(empirical_ready).lower()}。",
        f"mock_mode_or_deterministic_proxy：{str(mock_mode).lower()}。",
        f"systemic_conflict_flag：{str(systemic_conflict).lower()}。",
        f"关系分布：{dict(relation_counts)}。",
        f"来源独立性分布：{dict(source_counts)}。",
        "Cross-validation 是辅助交叉论证层，不改变 MD&A 或 News 分数。",
        "除非 readiness gate 标记 cross_validation_ready=true，否则不得描述为已通过。",
    ]
    outputs = write_dual_language_sections(root, "06_cross_validation", "cross_validation_section_cn.md", "cross_validation_section_en.md", cn_lines, en_lines)
    outputs["cross_validation_ready"] = status["cross_validation_ready"]
    return outputs


def _report_indicates_mock(root: Path) -> bool:
    report = root / "09_cross_validation" / "cross_validation_report.md"
    if not report.exists():
        return False
    text = report.read_text(encoding="utf-8", errors="replace")
    return "mock mode: True" in text or "deterministic_claim_proxy" in text or "mock_mode=true" in text


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper cross-validation section.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_cross_validation_section(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
