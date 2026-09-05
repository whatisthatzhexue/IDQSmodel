# 稳定性部分

权重敏感性：缺失，未找到 statistical stability report。
重复评分稳定性：只有 repeat-run 输入存在时才报告，否则标记 missing-input。
版本稳定性：只有 v1/v2 输入存在时才报告，否则标记 missing-input。
结构稳定性：只有 document-level 与 single-dimension 输入存在时才报告。
News aggregation stability：仅在真实 News 分数存在时报告。
News repeatability：not_completed; news_stability_report.md missing。
News broad-vs-company-focused robustness：broad=125，strict=0，suspect/manual-review=0，rank_spearman=0.0，top_20_overlap=0.0，bottom_20_overlap=0.0。
High-volatility cases：0 行需要复核。
Final freeze result：freeze_allowed=False，paper_ready=False。
