#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ "${SKIP_VENV:-0}" != "1" ]; then
  "${PYTHON_BIN}" -m venv .venv
  . .venv/bin/activate
  python -m pip install -r REPRODUCIBILITY/requirements.txt
  PYTHON=python
else
  PYTHON="${PYTHON_BIN}"
fi
"$PYTHON" 06_scoring/scripts/standalone_clean_reproduce.py
