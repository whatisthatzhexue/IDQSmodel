# Reproducibility Guide

Python version: 3.14.0.
Requirements: install from `requirements.txt`.
Setup command: `python3 -m venv .venv && . .venv/bin/activate && python -m pip install -r requirements.txt`.
Run command: `bash run_reproduce.sh`.
Local no-install run command: `SKIP_VENV=1 bash run_reproduce.sh`.
Expected outputs: `06_scoring/PAPER_OUTPUT/PAPER_READY_REPORT.md` and `06_scoring/PAPER_OUTPUT/PAPER_READY_STATUS.json`.
Validation commands: `python -m pytest 06_scoring/tests -q` and `python 06_scoring/scripts/self_check_pipeline.py --max-rounds 5`.
Known blockers: cross_validation_not_interpretable, systematic_matching_failure_detected.
News rerun command sequence: rerun the current Filter-news pipeline with `audit_filter_news_relevance.py --mode manual_preserve`, `build_news_registry_from_filter_news.py`, `extract_filter_news_texts.py`, `run_filter_news_final_scoring.py --model qwen3:8b`, then final freeze/readiness.
How to rerun final freeze: `python 06_scoring/scripts/run_final_data_freeze.py`.
How to interpret freeze_allowed=false: the system is engineering-ready but not cleared for final empirical conclusions.

Required commands:
- `bash run_reproduce.sh`
- `python 06_scoring/scripts/audit_filter_news_relevance.py --mode manual_preserve`
- `python 06_scoring/scripts/build_news_registry_from_filter_news.py`
- `python 06_scoring/scripts/extract_filter_news_texts.py`
- `python 06_scoring/scripts/run_filter_news_final_scoring.py --model qwen3:8b`
- `python 06_scoring/scripts/run_cross_validation.py --mode full --mock`
- `python 06_scoring/scripts/run_final_data_freeze.py`
- `python 06_scoring/scripts/run_paper_readiness_check.py`
