# qwen3:8b News Scoring Limitations

1. qwen3:8b is a local 8B-scale model, not a human fact checker.
2. News N01 factual accuracy checks internal consistency and verifiable cues in the article text; it cannot prove that external facts are true.
3. The model may misread English news, Malaysian company names, tickers, brand names, and multi-company articles.
4. JSON output failure, evidence-locator mismatch, high-score bias, and ceiling effects remain possible.
5. Long news can be truncated because the current prompt uses text[:11500]; see 08_reports/news_context_truncation_audit.csv.
6. Low-confidence News can enter the main model only with needs_manual_review=Yes; high-confidence samples should be used as a robustness sample.
7. temperature=0 and seed=42 improve reproducibility, but they do not remove model bias or guarantee factual truth.
