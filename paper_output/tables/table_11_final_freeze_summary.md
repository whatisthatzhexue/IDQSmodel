| metric | value | notes |
| --- | --- | --- |
| freeze_allowed | False | Final freeze gate |
| paper_ready | False | Paper readiness gate |
| blocking_reasons | mda_stability_success_rate; cross_validation_not_interpretable; cross_validation_run_complete; reproducibility_package_incomplete; requirements_txt_exists; run_reproduce_exists; pytest_log_exists; self_check_log_exists; statistical_stability_validation_exists | Must be resolved before final empirical claims |
| mda_final_dataset_size | 149 | Final output rows |
| news_final_dataset_size | 125 | Real scored News rows only |
| news_broad_dataset_size | 125 | Broad manually preserved real-News sample |
| news_company_focused_dataset_size | 0 | Strict company-focused subset for robustness |
| news_suspect_or_manual_review_count | 0 | Excluded from strict company-focused sample before validation |
| excluded_mda_documents | 0 | Final freeze exclusions |
| excluded_news_documents | 0 | Final freeze exclusions |
| synthetic_count | 0 | Excluded from empirical analysis |
| claim_link_count | 0 | Cross-validation links |
| cross_validation_pipeline_ready | True | Pipeline ran and produced link artifacts |
| cross_validation_empirical_ready | False | False when claim scoring is deterministic/mock or unresolved |
| cross_validation_ready | False | Overall readiness gate |
| mock_mode | False | Claim-level scoring mode |
| systemic_conflict_flag | False | Unresolved claim-level conflict flag |
