# 研究方法部分

状态：初步材料 / 非最终实证结论。

## 1. 研究设计
本研究构建一个可复现的马来西亚 F&B 企业文本评分流水线。评分对象是 MD&A disclosure quality 与 news information quality，而不是验证公司叙述真假。

## 2. 样本构建
样本来自纳入的 MD&A registry 与真实新闻 registry。Synthetic news placeholders 只用于 pipeline testing，不进入最终实证样本。

## 3. MD&A 评分对象
MD&A 评分对象是管理层讨论与分析文本。本研究评价 MD&A disclosure quality，不评价完整年报质量。

## 4. News 评分对象
News scoring 评价新闻文本的信息质量、来源透明度与可靠性。

## 5. 评分维度
MD&A 使用 AR01-AR05，News 使用 N01-N05。论文输出层不修改维度定义和权重。

## 6. 权重方案
MD&A 主权重为 AR01 0.15、AR02 0.25、AR03 0.30、AR04 0.20、AR05 0.10。News 主权重为 N01 0.30、N02 0.25、N03 0.15、N04 0.20、N05 0.10。

## 7. 本地 LLM 设置
评分设计基于本地 Ollama + qwen3:8b，默认 temperature=0、seed=42、num_ctx=4096、num_predict=192、think=false、stream=false，并使用 JSON schema mode；如果环境变量覆盖默认值，runtime report 会记录实际值。如果本地模型不可用，系统生成 blocker，而不是编造分数。
qwen3:8b 是本地 8B 级模型，不是人工事实核查员。News N01 factual accuracy 只能评价文本内部一致性与可核验线索，不能证明外部事实一定真实。

## 8. 单维评分设计
稳定版本采用每次只评分一个维度的设计，每次只输出 score、evidence、reason 三字段 JSON，Python 负责验证与聚合。

## 9. AR02 numeric checker
AR02 只评价 MD&A 文本内部财务表达的数值一致性、计算正确性与逻辑可追踪性。AR02 不是财务报表审计，也不是对公司财报真假的验证。

## 10. Evidence locator 与审计轨迹
每个维度必须有 MDA_Pxxx 或 NEWS_Pxxx 证据定位。严重缺失 evidence 的文档不能进入 final。

## 11. 质量控制
系统执行 JSON parse、schema validation、evidence check、score range check、review log 与 final freeze gate。

## 12. 统计稳定性
当输入存在时，系统报告权重敏感性、重复评分稳定性、版本稳定性、结构稳定性与 high-volatility cases。

## 13. Cross-validation 设计
Cross-validation 是后置交叉论证层，只作为 auxiliary triangulation，不替代主评分，也不改变 MD&A 或 News 分数。

## 14. Final freeze rule
Final freeze 需要 MD&A stability、真实 News 评分、schema 有效、系统错误率低、以及真实 News-backed claim links。gate 不通过时不得写最终实证结论。
