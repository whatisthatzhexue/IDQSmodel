# Final Data Freeze Quality Report

## FINAL DATASET
- dataset_version: final_freeze_2026-06-16
- freeze_allowed: false
- mda_final_dataset_size: 149
- news_final_dataset_size: 125
- system_level_error_rate: 0.0
- numeric_review_flag_count: 0
- confirmed_numeric_failure_count: 0
- not_enough_numbers_count: 45
- no_numeric_issue_count: 104

## Research-Grade Reliability
- research-grade reliability: NO
- empirical analysis: NO
- blocking gates: mda_stability_success_rate, cross_validation_claim_links
- advisory warnings: none

## Exclusion And Reason Distribution
- confirmed_numeric_failure: 0
- cross-validation conflict: 0
- json failure: 0
- low confidence: 0
- missing evidence: 0
- not_enough_numbers: 45
- numeric_review_required: 0

## MD&A Vs News Consistency Summary
- claim_link_count: 0
- contradiction_count: 0
- systemic_conflict_flag: False
- News registry or News scores missing: NO
- News synthetic placeholder rows: 0
- real News registry rows: 125

## Stability Checks
- top_20_overlap_ratio: 0.0
- bottom_20_overlap_ratio: 0.0
- rank stability (Spearman): 0.0

## Bias And Dimension Risk
- systematic bias: systemic conflict not detected by freeze layer
- most unstable dimensions: AR02 / AR03 / N02 require review if corresponding evidence or News coverage is incomplete

## Final Recommendation
- use for paper empirical analysis: NO
- recommendation: do not publish as final until blocking gates or advisory warnings are resolved
- future model optimization: fix input coverage/stability first; do not change model or dimensions inside this freeze

## Limitations
- This layer does not call qwen3 and does not change AR/N dimension scores or weights.
- Cross-validation remains auxiliary and is not averaged into MD&A or News scores.
