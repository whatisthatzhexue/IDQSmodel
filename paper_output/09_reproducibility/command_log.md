# Command Log

- `bash run_reproduce.sh`
- `SKIP_VENV=1 bash run_reproduce.sh`
- `python 06_scoring/scripts/audit_filter_news_relevance.py --mode manual_preserve`
- `python 06_scoring/scripts/build_news_registry_from_filter_news.py`
- `python 06_scoring/scripts/extract_filter_news_texts.py`
- `python 06_scoring/scripts/run_filter_news_final_scoring.py --model qwen3:8b`
- `python 06_scoring/scripts/backfill_qwen_runtime_metadata.py`
- `python 06_scoring/scripts/extract_mda_claims.py`
- `python 06_scoring/scripts/extract_news_claims.py`
- `python 06_scoring/scripts/run_claim_scoring.py --source-type news --mock`
- `python 06_scoring/scripts/run_claim_scoring.py --source-type mda --mock`
- `python 06_scoring/scripts/run_cross_validation.py --mode full --mock`
- `python 06_scoring/scripts/finalize_filter_news_cross_validation.py --skip-run`
- `python 06_scoring/scripts/self_check_pipeline.py --max-rounds 5`
- `python 06_scoring/scripts/validate_stability_outputs.py`
- `python 06_scoring/scripts/run_final_data_freeze.py`
- `python -m pytest 06_scoring/tests -q`
- `python 06_scoring/scripts/run_paper_readiness_check.py`
