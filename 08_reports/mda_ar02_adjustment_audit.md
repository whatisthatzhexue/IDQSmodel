# MDA AR02 Adjustment Audit

- ratings_path: 06_ratings\mda_final_v2\mda_final_ratings_long.csv
- ar02_rows: 0
- adjustment_applied_rows: 0

The `raw_score` column is retained for backward compatibility. For AR02, `original_model_raw_score` records the score parsed from the raw qwen output, while `adjusted_raw_score` records the score used in final tables after numeric-checker postprocessing.
