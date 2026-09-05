# Filter News Registry Report

- audit_rows: 97306
- main_model_include_rows: 71
- broad_include_rows: 125
- strict_company_focus_rows: 69
- registry_selection_flag: broad_include_flag
- registry_rows: 125
- high_confidence_registry_rows: 60
- missing_raw_rows: 0
- missing_text_rows: 0
- synthetic_count: 0
- registry_output: 06_scoring/01_registry/news_registry.csv
- high_confidence_registry_output: 06_scoring/01_registry/news_registry_high_confidence.csv

## Registry Selection
- registry_selection_flag: broad_include_flag
- main_model_include_rows is the strict model-gate count from the audit, not necessarily the broad sample count.
- broad_include_rows is the broad manually-preserved News sample used to rebuild 01_registry/news_registry.csv.
