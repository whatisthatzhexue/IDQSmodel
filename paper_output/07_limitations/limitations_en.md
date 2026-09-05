# Limitations

1. MD&A-only limitation: the MD&A score evaluates MD&A disclosure quality and does not represent a full annual-report assessment.
2. Local LLM limitation: qwen3:8b is a local 8B-scale model, not a human fact checker; the workflow depends on local model availability and JSON stability.
3. AR02 internal consistency limitation: AR02 measures internal numeric consistency, calculation correctness, and traceability in the MD&A text; it is not an external financial-statement audit.
4. News factual-accuracy limitation: News N01 can check internal consistency and verifiable cues in the article text, but it cannot prove that external facts are true.
5. News model-risk limitation: qwen3:8b may misjudge English news, Malaysian company names, tickers, brand names, and multi-company articles; it may also produce JSON failures, evidence-locator mismatches, inflated scores, or ceiling effects.
6. Context truncation limitation: long News inputs are capped at text[:11500]; truncation ratios are reported in 08_reports/news_context_truncation_audit.csv.
7. Low-confidence News limitation: low-confidence target-company mentions are retained in the manual review queue and excluded from the main model unless manually validated.
8. Reproducibility limitation: temperature=0 and seed=42 improve reproducibility, but they do not make the model unbiased or factually definitive.
9. News availability limitation: real News scoring is available for 125 non-synthetic final rows; remaining risks are relevance-filter false positives, low-confidence rows, and manual-review coverage.
10. Synthetic news exclusion: synthetic placeholders are pipeline-testing artifacts and are excluded from final empirical analysis.
11. Cross-validation limitation: cross-validation is auxiliary triangulation and does not prove that MD&A claims are true.
12. Small sample / stability threshold issue: samples that fail stability gates should not be used for final empirical claims.
13. Human review requirement: low-confidence, numeric-review, and high-volatility cases require manual review.
14. Generalizability limitation: results are limited to the sampled Malaysian F&B firms and the available text sources.
The current package is not final empirical output because freeze gates are not fully satisfied.
The outputs should be treated as pipeline validation materials rather than final empirical findings.
