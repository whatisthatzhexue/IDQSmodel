from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import ensure_paper_output_dirs, load_paper_status, write_dual_language_sections
from scoring_utils import SCORING_ROOT


def build_paper_limitations(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = load_paper_status(root)
    news_ready = bool(status.get("news_ready")) and int(status.get("news_final_dataset_size") or 0) > 0
    news_availability_en = (
        f"9. News availability limitation: real News scoring is available for {status['news_final_dataset_size']} non-synthetic final rows; remaining risks are relevance-filter false positives, low-confidence rows, and manual-review coverage."
        if news_ready
        else "9. News availability limitation: real News scoring must exist before News empirical analysis is possible."
    )
    news_availability_cn = (
        f"9. News availability limitation：当前已有 {status['news_final_dataset_size']} 条非 synthetic News final scoring；剩余风险是 relevance filter false positives、低置信度样本与人工复核覆盖。"
        if news_ready
        else "9. News availability limitation：真实 News 评分缺失时不能做 News 实证分析。"
    )
    en_lines = [
        "# Limitations",
        "",
        "1. MD&A-only limitation: the MD&A score evaluates MD&A disclosure quality and does not represent a full annual-report assessment.",
        "2. Local LLM limitation: qwen3:8b is a local 8B-scale model, not a human fact checker; the workflow depends on local model availability and JSON stability.",
        "3. AR02 internal consistency limitation: AR02 measures internal numeric consistency, calculation correctness, and traceability in the MD&A text; it is not an external financial-statement audit.",
        "4. News factual-accuracy limitation: News N01 can check internal consistency and verifiable cues in the article text, but it cannot prove that external facts are true.",
        "5. News model-risk limitation: qwen3:8b may misjudge English news, Malaysian company names, tickers, brand names, and multi-company articles; it may also produce JSON failures, evidence-locator mismatches, inflated scores, or ceiling effects.",
        "6. Context truncation limitation: long News inputs are capped at text[:11500]; truncation ratios are reported in 08_reports/news_context_truncation_audit.csv.",
        "7. Low-confidence News limitation: low-confidence target-company mentions are retained in the manual review queue and excluded from the main model unless manually validated.",
        "8. Reproducibility limitation: temperature=0 and seed=42 improve reproducibility, but they do not make the model unbiased or factually definitive.",
        news_availability_en,
        "10. Synthetic news exclusion: synthetic placeholders are pipeline-testing artifacts and are excluded from final empirical analysis.",
        "11. Cross-validation limitation: cross-validation is auxiliary triangulation and does not prove that MD&A claims are true.",
        "12. Small sample / stability threshold issue: samples that fail stability gates should not be used for final empirical claims.",
        "13. Human review requirement: low-confidence, numeric-review, and high-volatility cases require manual review.",
        "14. Generalizability limitation: results are limited to the sampled Malaysian F&B firms and the available text sources.",
    ]
    if not status["freeze_allowed"]:
        en_lines.append("The current package is not final empirical output because freeze gates are not fully satisfied.")
    if not status["paper_ready"]:
        en_lines.append("The outputs should be treated as pipeline validation materials rather than final empirical findings.")
    cn_lines = [
        "# 局限性",
        "",
        "1. MD&A-only limitation：MD&A 分数评价 MD&A disclosure quality，不代表完整年报质量。",
        "2. Local LLM limitation：qwen3:8b 是本地 8B 级模型，不是人工事实核查员；评分依赖本地模型可用性与 JSON 稳定性。",
        "3. AR02 internal consistency limitation：AR02 只衡量 MD&A 文本内部数值一致性、计算正确性和可追踪性，不是财务报表审计。",
        "4. News factual-accuracy limitation：News N01 只能检查文本内部一致性与可核验线索，不能证明外部事实一定真实。",
        "5. News model-risk limitation：qwen3:8b 可能误判英文新闻、马来西亚公司名、ticker、品牌名和多公司报道，也可能出现 JSON 输出失败、evidence locator 错配、评分偏高或 ceiling effect。",
        "6. Context truncation limitation：长新闻最多输入 text[:11500]，截断比例见 08_reports/news_context_truncation_audit.csv。",
        "7. Low-confidence News limitation：低置信度目标公司提及会保留在人工复核队列中，未经人工验证不进入主模型。",
        "8. Reproducibility limitation：temperature=0 与 seed=42 提高可复现性，但不等于模型无偏或事实完全正确。",
        news_availability_cn,
        "10. Synthetic news exclusion：synthetic placeholders 仅用于 pipeline testing，不进入最终实证分析。",
        "11. Cross-validation limitation：cross-validation 是辅助交叉论证，不能证明 MD&A 叙述为真。",
        "12. Small sample / stability threshold issue：未通过稳定性 gate 的样本不能用于最终实证结论。",
        "13. Human review requirement：低置信度、数值复核与高波动案例需要人工复核。",
        "14. Generalizability limitation：结果仅适用于样本中的马来西亚 F&B 公司和可用文本来源。",
    ]
    if not status["freeze_allowed"]:
        cn_lines.append("The current package is not final empirical output because freeze gates are not fully satisfied.")
    if not status["paper_ready"]:
        cn_lines.append("The outputs should be treated as pipeline validation materials rather than final empirical findings.")
    return write_dual_language_sections(root, "07_limitations", "limitations_cn.md", "limitations_en.md", cn_lines, en_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper limitations section.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_limitations(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
