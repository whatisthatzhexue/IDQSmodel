from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import ensure_paper_output_dirs, load_paper_status, write_dual_language_sections
from scoring_utils import SCORING_ROOT


def build_paper_methods_section(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = load_paper_status(root)
    qualifier_en = "Preliminary / not final" if not status["paper_ready"] else "Final"
    qualifier_cn = "初步材料 / 非最终实证结论" if not status["paper_ready"] else "最终材料"
    en_lines = [
        "# Methods Section",
        "",
        f"Status: {qualifier_en}.",
        "",
        "## 1. Research design",
        "This study constructs a reproducible text-scoring pipeline for Malaysian F&B firms. The scoring object is MD&A disclosure quality and news information quality, not a direct audit of firm truthfulness.",
        "",
        "## 2. Sample construction",
        "The sample is assembled from included MD&A registry records and real News registry records when available. Synthetic news placeholders are used only for pipeline testing and are excluded from final empirical analysis.",
        "",
        "## 3. MD&A scoring object definition",
        "The MD&A scoring object is the management discussion and analysis text. This is MD&A disclosure quality, not a full annual-report assessment.",
        "",
        "## 4. News scoring object definition",
        "News scoring evaluates news information quality and reliability at the document level.",
        "",
        "## 5. Scoring dimensions",
        "MD&A is scored on AR01-AR05 and News is scored on N01-N05. The dimension definitions and weights are fixed and are not changed by the paper output layer.",
        "",
        "## 6. Weighting scheme",
        "The main MD&A weights are AR01 0.15, AR02 0.25, AR03 0.30, AR04 0.20, AR05 0.10. The main News weights are N01 0.30, N02 0.25, N03 0.15, N04 0.20, N05 0.10.",
        "",
        "## 7. Local LLM scoring setup",
        "The scoring pipeline is designed for local Ollama with qwen3:8b using temperature=0, seed=42, num_ctx=4096, num_predict=192, think=false, stream=false, and JSON schema mode unless environment variables explicitly override these values and the runtime report records the actual values. If the local model is unavailable, the system writes blockers instead of fabricating scores.",
        "qwen3:8b is a local 8B-scale model, not a human fact checker. News N01 factual accuracy evaluates internal consistency and verifiable cues inside the article text; it cannot prove that external facts are true.",
        "",
        "## 8. Single-dimension scoring design",
        "The stable design uses one LLM call per dimension and a minimal JSON object containing score, evidence, and reason. Python performs validation and aggregation.",
        "",
        "## 9. AR02 numeric checker design",
        "AR02 evaluates internal numeric consistency, calculation correctness, and logical traceability in the MD&A text. AR02 is not an external financial-statement audit and does not verify whether company financial statements are true.",
        "",
        "## 10. Evidence locator and audit trail",
        "Each dimension score must include an evidence locator such as MDA_Pxxx or NEWS_Pxxx. Documents with critical missing evidence cannot enter final outputs.",
        "",
        "## 11. Quality control procedure",
        "Outputs pass JSON parsing, schema validation, evidence checks, score range checks, review logging, and final freeze gates.",
        "",
        "## 12. Statistical stability analysis",
        "The package records weight sensitivity, repeatability, version stability, structure stability, and high-volatility cases when inputs exist.",
        "",
        "## 13. Cross-validation design",
        "Cross-validation is an auxiliary triangulation layer. It compares MD&A and News claims for support, contradiction, omission, overstatement, and repetition. Cross-validation does not alter MD&A or News scores.",
        "",
        "## 14. Final freeze rule",
        "The final freeze gate requires MD&A stability, real News scoring, valid schema status, low system error rate, and real News-backed claim links. When gates fail, the package is not paper-ready.",
    ]
    cn_lines = [
        "# 研究方法部分",
        "",
        f"状态：{qualifier_cn}。",
        "",
        "## 1. 研究设计",
        "本研究构建一个可复现的马来西亚 F&B 企业文本评分流水线。评分对象是 MD&A disclosure quality 与 news information quality，而不是验证公司叙述真假。",
        "",
        "## 2. 样本构建",
        "样本来自纳入的 MD&A registry 与真实新闻 registry。Synthetic news placeholders 只用于 pipeline testing，不进入最终实证样本。",
        "",
        "## 3. MD&A 评分对象",
        "MD&A 评分对象是管理层讨论与分析文本。本研究评价 MD&A disclosure quality，不评价完整年报质量。",
        "",
        "## 4. News 评分对象",
        "News scoring 评价新闻文本的信息质量、来源透明度与可靠性。",
        "",
        "## 5. 评分维度",
        "MD&A 使用 AR01-AR05，News 使用 N01-N05。论文输出层不修改维度定义和权重。",
        "",
        "## 6. 权重方案",
        "MD&A 主权重为 AR01 0.15、AR02 0.25、AR03 0.30、AR04 0.20、AR05 0.10。News 主权重为 N01 0.30、N02 0.25、N03 0.15、N04 0.20、N05 0.10。",
        "",
        "## 7. 本地 LLM 设置",
        "评分设计基于本地 Ollama + qwen3:8b，默认 temperature=0、seed=42、num_ctx=4096、num_predict=192、think=false、stream=false，并使用 JSON schema mode；如果环境变量覆盖默认值，runtime report 会记录实际值。如果本地模型不可用，系统生成 blocker，而不是编造分数。",
        "qwen3:8b 是本地 8B 级模型，不是人工事实核查员。News N01 factual accuracy 只能评价文本内部一致性与可核验线索，不能证明外部事实一定真实。",
        "",
        "## 8. 单维评分设计",
        "稳定版本采用每次只评分一个维度的设计，每次只输出 score、evidence、reason 三字段 JSON，Python 负责验证与聚合。",
        "",
        "## 9. AR02 numeric checker",
        "AR02 只评价 MD&A 文本内部财务表达的数值一致性、计算正确性与逻辑可追踪性。AR02 不是财务报表审计，也不是对公司财报真假的验证。",
        "",
        "## 10. Evidence locator 与审计轨迹",
        "每个维度必须有 MDA_Pxxx 或 NEWS_Pxxx 证据定位。严重缺失 evidence 的文档不能进入 final。",
        "",
        "## 11. 质量控制",
        "系统执行 JSON parse、schema validation、evidence check、score range check、review log 与 final freeze gate。",
        "",
        "## 12. 统计稳定性",
        "当输入存在时，系统报告权重敏感性、重复评分稳定性、版本稳定性、结构稳定性与 high-volatility cases。",
        "",
        "## 13. Cross-validation 设计",
        "Cross-validation 是后置交叉论证层，只作为 auxiliary triangulation，不替代主评分，也不改变 MD&A 或 News 分数。",
        "",
        "## 14. Final freeze rule",
        "Final freeze 需要 MD&A stability、真实 News 评分、schema 有效、系统错误率低、以及真实 News-backed claim links。gate 不通过时不得写最终实证结论。",
    ]
    return write_dual_language_sections(root, "01_methods", "methods_section_cn.md", "methods_section_en.md", cn_lines, en_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper methods section.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_methods_section(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
