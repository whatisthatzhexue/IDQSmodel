# MD&A Human Validation Instructions

- Requested sample size: 30
- Actual sample rows: 30
- Sampling mode: dimension-row sample
- Scope: MD&A-only model-generated candidate dataset.
- Do not edit the model score columns.
- Fill only the human_score_1_5, human_evidence_valid_y_n, human_dimension_fit_y_n, human_notes, and reviewer_id columns.
- Use human_score_1_5 only when the evidence supports an independent 1-5 judgment for the listed dimension.
- Mark human_evidence_valid_y_n as N if the locator or excerpt does not support the dimension.
- Pay extra attention to sample_reason values semantic_review and ar02_numeric_review.
- News scoring is outside this MD&A human-validation sample; use the News review queues for News-specific manual checks.
