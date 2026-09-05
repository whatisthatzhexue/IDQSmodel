# 局限性

1. MD&A-only limitation：MD&A 分数评价 MD&A disclosure quality，不代表完整年报质量。
2. Local LLM limitation：qwen3:8b 是本地 8B 级模型，不是人工事实核查员；评分依赖本地模型可用性与 JSON 稳定性。
3. AR02 internal consistency limitation：AR02 只衡量 MD&A 文本内部数值一致性、计算正确性和可追踪性，不是财务报表审计。
4. News factual-accuracy limitation：News N01 只能检查文本内部一致性与可核验线索，不能证明外部事实一定真实。
5. News model-risk limitation：qwen3:8b 可能误判英文新闻、马来西亚公司名、ticker、品牌名和多公司报道，也可能出现 JSON 输出失败、evidence locator 错配、评分偏高或 ceiling effect。
6. Context truncation limitation：长新闻最多输入 text[:11500]，截断比例见 08_reports/news_context_truncation_audit.csv。
7. Low-confidence News limitation：低置信度目标公司提及会保留在人工复核队列中，未经人工验证不进入主模型。
8. Reproducibility limitation：temperature=0 与 seed=42 提高可复现性，但不等于模型无偏或事实完全正确。
9. News availability limitation：当前已有 125 条非 synthetic News final scoring；剩余风险是 relevance filter false positives、低置信度样本与人工复核覆盖。
10. Synthetic news exclusion：synthetic placeholders 仅用于 pipeline testing，不进入最终实证分析。
11. Cross-validation limitation：cross-validation 是辅助交叉论证，不能证明 MD&A 叙述为真。
12. Small sample / stability threshold issue：未通过稳定性 gate 的样本不能用于最终实证结论。
13. Human review requirement：低置信度、数值复核与高波动案例需要人工复核。
14. Generalizability limitation：结果仅适用于样本中的马来西亚 F&B 公司和可用文本来源。
The current package is not final empirical output because freeze gates are not fully satisfied.
The outputs should be treated as pipeline validation materials rather than final empirical findings.
