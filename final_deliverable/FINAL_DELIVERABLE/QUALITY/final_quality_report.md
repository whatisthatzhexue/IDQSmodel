# Final Data Freeze Quality Report

## FINAL DATASET
- dataset_version: final_freeze_2026-06-16
- freeze_allowed: true
- mda_final_dataset_size: 149
- news_final_dataset_size: 125
- system_level_error_rate: 0.0
- numeric_review_flag_count: 124
- confirmed_numeric_failure_count: 0
- not_enough_numbers_count: 10
- no_numeric_issue_count: 15

## Research-Grade Reliability
- research-grade reliability: NO
- empirical analysis: NO
- blocking gates: none
- advisory warnings: cross_validation_systemic_conflict_unresolved

## Exclusion And Reason Distribution
- confirmed_numeric_failure: 0
- cross-validation conflict: 30
- json failure: 0
- low confidence: 0
- missing evidence: 0
- not_enough_numbers: 10
- numeric_review_required: 124

## MD&A Vs News Consistency Summary
- claim_link_count: 316
- contradiction_count: 30
- systemic_conflict_flag: True
- News registry or News scores missing: NO
- News synthetic placeholder rows: 0
- real News registry rows: 125

## Stability Checks
- top_20_overlap_ratio: 1.0
- bottom_20_overlap_ratio: 1.0
- rank stability (Spearman): 0.995772

## Bias And Dimension Risk
- systematic bias: systemic conflict detected by freeze layer
- most unstable dimensions: none

## Final Recommendation
- use for paper empirical analysis: NO
- recommendation: do not publish as final until blocking gates or advisory warnings are resolved
- future model optimization: fix input coverage/stability first; do not change model or dimensions inside this freeze

## Limitations
- This layer does not call qwen3 and does not change AR/N dimension scores or weights.
- Cross-validation remains auxiliary and is not averaged into MD&A or News scores.
