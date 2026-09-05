# MD&A News Cross-Validation Report

## Purpose
This module is a post-scoring validation layer. It does not match documents mechanically, average MD&A and news scores, or overwrite original scoring outputs. It evaluates support, qualification, contradiction, omission, overstatement, and repetition between MD&A and news claims.

## Method
claim extraction -> claim linking -> relation classification -> pair summary -> company-year summary.

## Bidirectional Logic
- MD&A -> news: whether news supports, contextualizes, qualifies, or challenges management explanations.
- News -> MD&A: whether MD&A supports, limits, or contradicts news claims.

## Run Status
- local model available: True
- mock mode: True
- cross_validation_pipeline_ready: True
- cross_validation_empirical_ready: False
- mda claims: 2861
- news claims: 323
- claim links: 316
- pair summaries: 149
- company-year summaries: 149

## Main Findings
- strong support cases: 40
- contextual support cases: 22
- contradictions: 30
- possible omissions: 0
- possible news overstatements: 0
- same-source repetitions: 8

## Empirical Readiness
cross_validation_pipeline_ready=true because claim extraction, linking, and reporting files were produced.
cross_validation_empirical_ready=false because claim scoring used deterministic_claim_proxy / mock_mode=true rather than qwen3:8b or a validated human rule pass.
News scoring enters the document-level measurement model, but claim-level MD&A-News triangulation remains non-interpretable because no real claim scoring pass was produced.

## Limitations
- News and MD&A are not absolute truth sources.
- News may rewrite company announcements or annual reports.
- MD&A may reflect management bias.
- Cross-validation is a review indicator, not a substitute for human judgment.
