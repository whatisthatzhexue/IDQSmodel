# 数据部分

公司范围：MD&A registry 中包含 31 个 ticker。
时间范围：2020 至 2024。
MD&A 文本来源：mda_registry.csv 指向的 MD&A extracted text。
News 文本来源：真实 News registry 与 NEWS_Pxxx 分段文本；若缺失则不进入实证分析。
文本清洗规则：文本被标准化为段落定位符，评分 evidence 必须引用这些定位符。
document_id 规则：MD&A 编码来源文档与年份；News 编码 ticker、日期与来源哈希。
纳入标准：include_flag=yes、文本路径有效、schema 有效、final 中存在 evidence locator。
排除标准：missing evidence、schema invalid、严重 scope violation 与 synthetic News placeholders。
News broad sample：125 行；company-focused strict subset：69 行；suspect/manual-review：56 行。
论文不得把 broad News sample 全部描述为 fully clean company-specific News；涉及 company-specific robustness 时优先使用 strict subset。
当前真实 News registry 中没有 synthetic placeholders。
真实 News final scores 可用于实证表格。
