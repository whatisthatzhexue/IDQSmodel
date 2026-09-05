#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
# Default setup command: python3 -m venv .venv
if [ "${SKIP_VENV:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  PYTHON=python
else
  PYTHON="${PYTHON_BIN}"
fi

mkdir -p 06_scoring/PAPER_OUTPUT/09_reproducibility
LOG="06_scoring/PAPER_OUTPUT/09_reproducibility/run_reproduce.log"
: > "$LOG"

run_step() {
  echo "" | tee -a "$LOG"
  echo "## $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

run_step "$PYTHON" -m pytest 06_scoring/tests -q
run_step "$PYTHON" 06_scoring/scripts/audit_filter_news_relevance.py --mode manual_preserve
run_step "$PYTHON" 06_scoring/scripts/build_news_registry_from_filter_news.py
run_step "$PYTHON" 06_scoring/scripts/extract_filter_news_texts.py
run_step "$PYTHON" 06_scoring/scripts/run_filter_news_final_scoring.py --model qwen3:8b
run_step "$PYTHON" 06_scoring/scripts/backfill_qwen_runtime_metadata.py
run_step "$PYTHON" 06_scoring/scripts/extract_mda_claims.py
run_step "$PYTHON" 06_scoring/scripts/extract_news_claims.py
run_step "$PYTHON" 06_scoring/scripts/run_claim_scoring.py --source-type news --mock
run_step "$PYTHON" 06_scoring/scripts/run_claim_scoring.py --source-type mda --mock
run_step "$PYTHON" 06_scoring/scripts/run_cross_validation.py --mode full --mock
run_step "$PYTHON" 06_scoring/scripts/finalize_filter_news_cross_validation.py --skip-run
run_step "$PYTHON" 06_scoring/scripts/self_check_pipeline.py --max-rounds 5
run_step "$PYTHON" 06_scoring/scripts/validate_stability_outputs.py
run_step "$PYTHON" 06_scoring/scripts/run_final_data_freeze.py
run_step "$PYTHON" 06_scoring/scripts/build_paper_reproducibility_package.py
run_step "$PYTHON" 06_scoring/scripts/run_paper_readiness_check.py
