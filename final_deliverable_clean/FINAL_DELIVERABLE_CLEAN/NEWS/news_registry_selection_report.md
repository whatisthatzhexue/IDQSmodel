# News Registry Selection Report

- selected_registry: 01_registry/news_registry.csv
- selection_reason: fallback to canonical registry because rebuilt registry is empty or has no real include_flag=Yes rows
- rebuilt_registry_rows: 0
- rebuilt_real_include_rows: 0
- canonical_registry_rows: 125
- canonical_real_include_rows: 125

This guard prevents an empty rebuilt registry from freezing real News outputs to zero rows.
