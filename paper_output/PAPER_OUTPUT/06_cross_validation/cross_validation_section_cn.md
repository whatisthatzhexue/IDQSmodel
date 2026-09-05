# Cross-validation 部分

News scoring enters the document-level measurement model, but claim-level MD&A-News triangulation remains non-interpretable because claim scoring used deterministic proxy/mock mode or unresolved systemic conflict flags remain.
MD&A claims 数量：2861。
News claims 数量：323。
Claim links 数量：316。
cross_validation_pipeline_ready：true。
cross_validation_empirical_ready：false。
mock_mode_or_deterministic_proxy：true。
systemic_conflict_flag：true。
关系分布：{'insufficient_evidence': 53, 'strong_support': 40, 'partial_support': 153, 'contradiction': 30, 'contextual_support': 22, 'same_source_repetition': 8, 'no_clear_relation': 10}。
来源独立性分布：{'unknown': 300, 'low': 10, 'medium': 6}。
Cross-validation 是辅助交叉论证层，不改变 MD&A 或 News 分数。
除非 readiness gate 标记 cross_validation_ready=true，否则不得描述为已通过。
