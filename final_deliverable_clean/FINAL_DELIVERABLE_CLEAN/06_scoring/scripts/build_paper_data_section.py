from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import ensure_paper_output_dirs, load_paper_status, real_news_rows, registry_synthetic_count, write_dual_language_sections, write_table
from scoring_utils import SCORING_ROOT, read_csv_rows


def build_paper_data_section(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = load_paper_status(root)
    mda_registry = read_csv_rows(root / "01_registry" / "mda_registry.csv")
    news_registry = read_csv_rows(root / "01_registry" / "news_registry.csv")
    real_news = real_news_rows(root)
    synthetic_count = registry_synthetic_count(root)
    mda_final = read_csv_rows(root / "FINAL_OUTPUT" / "mda_final_dataset.csv")
    news_final = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    news_broad = read_csv_rows(root / "FINAL_OUTPUT" / "news_broad_dataset.csv") or news_final
    news_company_focused = read_csv_rows(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv")
    news_suspect = read_csv_rows(root / "08_reports" / "news_suspect_not_company_focused.csv")
    companies = sorted({row.get("ticker", "") for row in mda_registry if row.get("ticker")})
    years = sorted({row.get("report_year", "") for row in mda_registry if row.get("report_year")})

    sample_rows = [
        {"sample_step": "MD&A registry included", "count": sum(row.get("include_flag", "").lower() == "yes" for row in mda_registry), "notes": "MD&A source registry"},
        {"sample_step": "MD&A final dataset", "count": len(mda_final), "notes": "Final freeze output"},
        {"sample_step": "News registry included", "count": sum(row.get("include_flag", "").lower() == "yes" for row in news_registry), "notes": "Includes only registry rows"},
        {"sample_step": "Real News registry", "count": len(real_news), "notes": "Synthetic placeholders excluded"},
        {"sample_step": "Synthetic News placeholders", "count": synthetic_count, "notes": "Pipeline testing only"},
        {"sample_step": "News final dataset", "count": len(news_final), "notes": "Broad manually preserved real-News sample; not all rows are strict company-focused"},
        {"sample_step": "News broad final dataset", "count": len(news_broad), "notes": "Broad manually preserved real-News measurement sample"},
        {"sample_step": "News company-focused strict subset", "count": len(news_company_focused), "notes": "Title or lead paragraphs focus on the target company"},
        {"sample_step": "News suspect/manual-review outside strict subset", "count": len(news_suspect), "notes": "Market-roundup, list, sponsor, venue, sport, metro, lifestyle, or incidental-mention cases"},
    ]
    write_table(root, "table_1_sample_construction", ["sample_step", "count", "notes"], sample_rows)

    variable_rows = [
        {"variable": "document_id", "definition": "Stable document identifier for MD&A or News text", "source": "registry"},
        {"variable": "total_score_100", "definition": "Python-aggregated score on a 0-100 scale", "source": "final dataset"},
        {"variable": "AR01-AR05", "definition": "MD&A dimension scores on a 1-5 raw scale", "source": "MD&A scoring"},
        {"variable": "N01-N05", "definition": "News dimension scores on a 1-5 raw scale", "source": "News scoring"},
        {"variable": "evidence_locator", "definition": "Paragraph-level evidence pointer such as MDA_Pxxx or NEWS_Pxxx", "source": "ratings_long"},
        {"variable": "synthetic_flag", "definition": "Registry marker for placeholder News rows excluded from empirical analysis", "source": "news registry notes"},
    ]
    write_table(root, "table_2_variable_definitions", ["variable", "definition", "source"], variable_rows)

    missing_news_sentence = "News-based analyses and cross-validation are not available until real news data are supplied or scored."
    synthetic_sentence = "Synthetic news placeholders were used only for pipeline testing and excluded from final empirical analysis."
    en_lines = [
        "# Data Section",
        "",
        f"Company scope: {len(companies)} tickers are present in the MD&A registry.",
        f"Time range: {years[0] if years else 'missing'} to {years[-1] if years else 'missing'}.",
        "MD&A text source: extracted MD&A text files referenced by mda_registry.csv.",
        "News text source: real News registry rows and extracted NEWS_Pxxx text files when available.",
        "Text cleaning: document text is normalized into paragraph locators and scoring uses those locators for evidence checks.",
        "document_id rule: MD&A identifiers encode source document and year; News identifiers encode ticker, date, and source hash.",
        "Inclusion criteria: include_flag=yes, valid text path, valid schema, and evidence locator availability for final outputs.",
        "Exclusion criteria: missing evidence, schema invalid rows, severe scope violations, and synthetic News placeholders.",
        f"News broad sample: {len(news_broad)} rows; company-focused strict subset: {len(news_company_focused)} rows; suspect/manual-review rows: {len(news_suspect)}.",
        "The broad News sample must not be described as fully clean company-specific News; the strict subset is preferred for company-specific robustness claims.",
        synthetic_sentence if synthetic_count > 0 else "No synthetic news placeholders are included in the current real News registry.",
        missing_news_sentence if status["news_final_dataset_size"] == 0 else "Real News final scores are available for empirical tables.",
    ]
    cn_lines = [
        "# 数据部分",
        "",
        f"公司范围：MD&A registry 中包含 {len(companies)} 个 ticker。",
        f"时间范围：{years[0] if years else '缺失'} 至 {years[-1] if years else '缺失'}。",
        "MD&A 文本来源：mda_registry.csv 指向的 MD&A extracted text。",
        "News 文本来源：真实 News registry 与 NEWS_Pxxx 分段文本；若缺失则不进入实证分析。",
        "文本清洗规则：文本被标准化为段落定位符，评分 evidence 必须引用这些定位符。",
        "document_id 规则：MD&A 编码来源文档与年份；News 编码 ticker、日期与来源哈希。",
        "纳入标准：include_flag=yes、文本路径有效、schema 有效、final 中存在 evidence locator。",
        "排除标准：missing evidence、schema invalid、严重 scope violation 与 synthetic News placeholders。",
        f"News broad sample：{len(news_broad)} 行；company-focused strict subset：{len(news_company_focused)} 行；suspect/manual-review：{len(news_suspect)} 行。",
        "论文不得把 broad News sample 全部描述为 fully clean company-specific News；涉及 company-specific robustness 时优先使用 strict subset。",
        "Synthetic news placeholders 只用于 pipeline testing，并从最终实证样本中排除。" if synthetic_count > 0 else "当前真实 News registry 中没有 synthetic placeholders。",
        "真实 News 评分缺失时，News-based analyses and cross-validation are not available until real news data are supplied." if status["news_final_dataset_size"] == 0 else "真实 News final scores 可用于实证表格。",
    ]
    outputs = write_dual_language_sections(root, "02_data", "data_section_cn.md", "data_section_en.md", cn_lines, en_lines)
    outputs.update({"sample_table": str(root / "PAPER_OUTPUT" / "tables" / "table_1_sample_construction.csv")})
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper data section and sample tables.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_data_section(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
