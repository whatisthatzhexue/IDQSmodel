# Data Section

Company scope: 31 tickers are present in the MD&A registry.
Time range: 2020 to 2024.
MD&A text source: extracted MD&A text files referenced by mda_registry.csv.
News text source: real News registry rows and extracted NEWS_Pxxx text files when available.
Text cleaning: document text is normalized into paragraph locators and scoring uses those locators for evidence checks.
document_id rule: MD&A identifiers encode source document and year; News identifiers encode ticker, date, and source hash.
Inclusion criteria: include_flag=yes, valid text path, valid schema, and evidence locator availability for final outputs.
Exclusion criteria: missing evidence, schema invalid rows, severe scope violations, and synthetic News placeholders.
News broad sample: 125 rows; company-focused strict subset: 69 rows; suspect/manual-review rows: 56.
The broad News sample must not be described as fully clean company-specific News; the strict subset is preferred for company-specific robustness claims.
No synthetic news placeholders are included in the current real News registry.
Real News final scores are available for empirical tables.
