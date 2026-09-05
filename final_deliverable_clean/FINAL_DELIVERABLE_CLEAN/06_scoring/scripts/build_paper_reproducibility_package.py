from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from paper_utils import ensure_paper_output_dirs, environment_summary, file_manifest, load_paper_status
from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from stability_common import write_markdown


REPRO_REQUIREMENTS = [
    "pytest",
    "matplotlib",
    "openpyxl",
]

EMPTY_CSV_HEADERS = ["relative_path", "row_count", "status", "category", "allowed_empty", "notes"]


def build_paper_reproducibility_package(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = load_paper_status(root)
    repo_root = root.parent
    requirements_path = repo_root / "requirements.txt"
    run_script_path = repo_root / "run_reproduce.sh"
    requirements_path.write_text("\n".join(REPRO_REQUIREMENTS) + "\n", encoding="utf-8")
    run_script_path.write_text(_run_reproduce_script(), encoding="utf-8")
    os.chmod(run_script_path, 0o755)
    env = environment_summary()
    save_json(root / "PAPER_OUTPUT" / "09_reproducibility" / "environment_summary.json", env)
    write_markdown(
        root / "PAPER_OUTPUT" / "09_reproducibility" / "environment_summary.md",
        [
            "# Environment Summary",
            "",
            f"- Python version: {env['python_version']}",
            f"- Platform: {env['platform']}",
            f"- Implementation: {env['implementation']}",
        ],
    )
    commands = [
        "bash run_reproduce.sh",
        "SKIP_VENV=1 bash run_reproduce.sh",
        "python 06_scoring/scripts/audit_filter_news_relevance.py --mode manual_preserve",
        "python 06_scoring/scripts/build_news_registry_from_filter_news.py",
        "python 06_scoring/scripts/extract_filter_news_texts.py",
        "python 06_scoring/scripts/run_filter_news_final_scoring.py --model qwen3:8b",
        "python 06_scoring/scripts/backfill_qwen_runtime_metadata.py",
        "python 06_scoring/scripts/extract_mda_claims.py",
        "python 06_scoring/scripts/extract_news_claims.py",
        "python 06_scoring/scripts/run_claim_scoring.py --source-type news --mock",
        "python 06_scoring/scripts/run_claim_scoring.py --source-type mda --mock",
        "python 06_scoring/scripts/run_cross_validation.py --mode full --mock",
        "python 06_scoring/scripts/finalize_filter_news_cross_validation.py --skip-run",
        "python 06_scoring/scripts/self_check_pipeline.py --max-rounds 5",
        "python 06_scoring/scripts/validate_stability_outputs.py",
        "python 06_scoring/scripts/run_final_data_freeze.py",
        "python -m pytest 06_scoring/tests -q",
        "python 06_scoring/scripts/run_paper_readiness_check.py",
    ]
    write_markdown(root / "PAPER_OUTPUT" / "09_reproducibility" / "command_log.md", ["# Command Log", "", *[f"- `{cmd}`" for cmd in commands]])
    if bool(status.get("news_ready")) and int(status.get("news_final_dataset_size") or 0) > 0:
        news_repro_line = (
            "News rerun command sequence: rerun the current Filter-news pipeline with "
            "`audit_filter_news_relevance.py --mode manual_preserve`, "
            "`build_news_registry_from_filter_news.py`, `extract_filter_news_texts.py`, "
            "`run_filter_news_final_scoring.py --model qwen3:8b`, then final freeze/readiness."
        )
    else:
        news_repro_line = (
            "How to replace missing News: provide `input_news/news_raw.csv` or `.xlsx` with ticker, "
            "company_name, publish_date, source_name, title, url, article_text; then rerun `bash run_reproduce.sh`."
        )
    guide_lines = [
        "# Reproducibility Guide",
        "",
        f"Python version: {env['python_version']}.",
        "Requirements: install from `requirements.txt`.",
        "Setup command: `python3 -m venv .venv && . .venv/bin/activate && python -m pip install -r requirements.txt`.",
        "Run command: `bash run_reproduce.sh`.",
        "Local no-install run command: `SKIP_VENV=1 bash run_reproduce.sh`.",
        "Expected outputs: `06_scoring/PAPER_OUTPUT/PAPER_READY_REPORT.md` and `06_scoring/PAPER_OUTPUT/PAPER_READY_STATUS.json`.",
        "Validation commands: `python -m pytest 06_scoring/tests -q` and `python 06_scoring/scripts/self_check_pipeline.py --max-rounds 5`.",
        f"Known blockers: {', '.join(status['blocking_reasons']) or 'none'}.",
        news_repro_line,
        "How to rerun final freeze: `python 06_scoring/scripts/run_final_data_freeze.py`.",
        "How to interpret freeze_allowed=false: the system is engineering-ready but not cleared for final empirical conclusions.",
        "",
        "Required commands:",
        "- `bash run_reproduce.sh`",
        "- `python 06_scoring/scripts/audit_filter_news_relevance.py --mode manual_preserve`",
        "- `python 06_scoring/scripts/build_news_registry_from_filter_news.py`",
        "- `python 06_scoring/scripts/extract_filter_news_texts.py`",
        "- `python 06_scoring/scripts/run_filter_news_final_scoring.py --model qwen3:8b`",
        "- `python 06_scoring/scripts/run_cross_validation.py --mode full --mock`",
        "- `python 06_scoring/scripts/run_final_data_freeze.py`",
        "- `python 06_scoring/scripts/run_paper_readiness_check.py`",
    ]
    write_markdown(root / "PAPER_OUTPUT" / "09_reproducibility" / "reproducibility_guide.md", guide_lines)
    empty_csv_audit_path = root / "PAPER_OUTPUT" / "09_reproducibility" / "empty_csv_audit.csv"
    empty_csv_rows = _empty_csv_audit_rows(root)
    write_csv_rows(empty_csv_audit_path, EMPTY_CSV_HEADERS, empty_csv_rows)
    _write_empty_csv_explanation(root, empty_csv_rows)
    file_manifest(root, root / "PAPER_OUTPUT" / "09_reproducibility" / "file_manifest.csv")
    _write_top_level_manifest(repo_root / "FINAL_PACKAGE_MANIFEST.md", status, env, empty_csv_rows)
    return {
        "requirements": str(requirements_path),
        "run_reproduce": str(run_script_path),
        "commands": commands,
        "empty_csv_audit_path": str(empty_csv_audit_path),
        "top_level_manifest": str(repo_root / "FINAL_PACKAGE_MANIFEST.md"),
    }


def _run_reproduce_script() -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            'cd "$(dirname "$0")"',
            "",
            'PYTHON_BIN="${PYTHON_BIN:-python3}"',
            "# Default setup command: python3 -m venv .venv",
            'if [ "${SKIP_VENV:-0}" != "1" ]; then',
            '  "${PYTHON_BIN}" -m venv .venv',
            "  . .venv/bin/activate",
            "  python -m pip install --upgrade pip",
            "  python -m pip install -r requirements.txt",
            "  PYTHON=python",
            "else",
            '  PYTHON="${PYTHON_BIN}"',
            "fi",
            "",
            "mkdir -p 06_scoring/PAPER_OUTPUT/09_reproducibility",
            'LOG="06_scoring/PAPER_OUTPUT/09_reproducibility/run_reproduce.log"',
            ': > "$LOG"',
            "",
            "run_step() {",
            '  echo "" | tee -a "$LOG"',
            '  echo "## $*" | tee -a "$LOG"',
            '  "$@" 2>&1 | tee -a "$LOG"',
            "}",
            "",
            'run_step "$PYTHON" -m pytest 06_scoring/tests -q',
            'run_step "$PYTHON" 06_scoring/scripts/audit_filter_news_relevance.py --mode manual_preserve',
            'run_step "$PYTHON" 06_scoring/scripts/build_news_registry_from_filter_news.py',
            'run_step "$PYTHON" 06_scoring/scripts/extract_filter_news_texts.py',
            'run_step "$PYTHON" 06_scoring/scripts/run_filter_news_final_scoring.py --model qwen3:8b',
            'run_step "$PYTHON" 06_scoring/scripts/backfill_qwen_runtime_metadata.py',
            'run_step "$PYTHON" 06_scoring/scripts/extract_mda_claims.py',
            'run_step "$PYTHON" 06_scoring/scripts/extract_news_claims.py',
            'run_step "$PYTHON" 06_scoring/scripts/run_claim_scoring.py --source-type news --mock',
            'run_step "$PYTHON" 06_scoring/scripts/run_claim_scoring.py --source-type mda --mock',
            'run_step "$PYTHON" 06_scoring/scripts/run_cross_validation.py --mode full --mock',
            'run_step "$PYTHON" 06_scoring/scripts/finalize_filter_news_cross_validation.py --skip-run',
            'run_step "$PYTHON" 06_scoring/scripts/self_check_pipeline.py --max-rounds 5',
            'run_step "$PYTHON" 06_scoring/scripts/validate_stability_outputs.py',
            'run_step "$PYTHON" 06_scoring/scripts/run_final_data_freeze.py',
            'run_step "$PYTHON" 06_scoring/scripts/build_paper_reproducibility_package.py',
            'run_step "$PYTHON" 06_scoring/scripts/run_paper_readiness_check.py',
            "",
        ]
    )


def _empty_csv_audit_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.csv")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in {
            "archive",
            "_self_correction",
            "FINAL_DELIVERABLE",
            "FINAL_DELIVERABLE_CLEAN",
            "FINAL_DELIVERABLE_WITH_NEWS",
            "PAPER_OUTPUT_FINAL_DELIVERABLE",
        }:
            continue
        if path.name.endswith("_failed_cases.csv") or path.name.endswith("_failed_or_excluded_cases.csv"):
            continue
        if "PAPER_OUTPUT/09_reproducibility/empty_csv_audit.csv" in str(path):
            continue
        if len(read_csv_rows(path)) > 0:
            continue
        category, allowed, notes = _classify_empty_csv(relative)
        rows.append(
            {
                "relative_path": str(relative),
                "row_count": 0,
                "status": "allowed_empty" if allowed else "problem_empty",
                "category": category,
                "allowed_empty": str(allowed).lower(),
                "notes": notes,
            }
        )
    return rows


def _classify_empty_csv(relative: Path) -> tuple[str, bool, str]:
    text = str(relative)
    required_nonempty = {
        "01_registry/news_registry.csv",
        "FINAL_OUTPUT/news_final_dataset.csv",
        "FINAL_OUTPUT/news_broad_dataset.csv",
        "FINAL_OUTPUT/news_company_focused_dataset.csv",
        "FINAL_OUTPUT/mda_final_dataset.csv",
        "06_ratings/news_final_full/news_final_document_scores.csv",
        "06_ratings/news_final_full/news_final_ratings_long.csv",
        "06_ratings/mda_final_v2/mda_final_document_scores.csv",
        "06_ratings/mda_final_v2/mda_final_ratings_long.csv",
    }
    if text in required_nonempty:
        return "required_final_output", False, "Required final output is empty and must be regenerated before review."
    if any(part in relative.parts for part in {"repeat_runs", "document_level", "single_dimension"}):
        return "optional_stability_input", True, "Optional repeatability/structure-comparison input may be empty when that robustness run was not executed."
    if relative.parts and relative.parts[0] in {"10_claim_extraction", "10_claim_mapping", "10_claim_scoring", "10_claims"}:
        return "optional_claim_pipeline_output", True, "Claim-stage scaffold or optional intermediate output; empirical readiness is governed by cross-validation metadata."
    return "optional_or_historical_output", True, "Allowed only if documented as optional, scaffold, or historical diagnostic; not used as a current final dataset input."


def _write_empty_csv_explanation(root: Path, rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    problem_rows = []
    for row in rows:
        category = str(row.get("category", "uncategorized"))
        counts[category] = counts.get(category, 0) + 1
        if str(row.get("allowed_empty", "")).lower() != "true":
            problem_rows.append(row)
    lines = [
        "# Empty CSV Explanation",
        "",
        f"- empty_csv_count: {len(rows)}",
        f"- problem_empty_count: {len(problem_rows)}",
        "",
        "## Allowed empty CSV categories",
    ]
    if counts:
        for category, count in sorted(counts.items()):
            lines.append(f"- {category}: {count}")
    else:
        lines.append("- none: 0")
    lines.extend(
        [
            "",
            "## Problem empty CSVs",
            "- none" if not problem_rows else "",
        ]
    )
    for row in problem_rows:
        lines.append(f"- {row.get('relative_path')}: {row.get('notes')}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Empty optional/historical/scaffold CSVs are not treated as evidence that current News or MD&A final outputs are missing.",
            "- Required final outputs are explicitly classified as problem_empty if they ever appear here.",
        ]
    )
    write_markdown(root / "PAPER_OUTPUT" / "09_reproducibility" / "empty_csv_explanation.md", lines)


def _write_top_level_manifest(path: Path, status: dict[str, Any], env: dict[str, Any], empty_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Final Package Manifest",
        "",
        f"- paper_ready: {str(status['paper_ready']).lower()}",
        f"- freeze_allowed: {str(status['freeze_allowed']).lower()}",
        f"- mda_ready: {str(status['mda_ready']).lower()}",
        f"- news_ready: {str(status['news_ready']).lower()}",
        f"- cross_validation_ready: {str(status['cross_validation_ready']).lower()}",
        f"- cross_validation_pipeline_ready: {str(status.get('cross_validation_pipeline_ready', False)).lower()}",
        f"- cross_validation_empirical_ready: {str(status.get('cross_validation_empirical_ready', False)).lower()}",
        f"- mock_mode: {str(status.get('mock_mode', False)).lower()}",
        f"- systemic_conflict_flag: {str(status.get('systemic_conflict_flag', False)).lower()}",
        f"- reproducibility_ready: {str(status['reproducibility_ready']).lower()}",
        f"- python_version: {env['python_version']}",
        f"- platform: {env['platform']}",
        f"- empty_csv_count: {len(empty_rows)}",
        f"- empty_csv_problem_count: {sum(str(row.get('allowed_empty', '')).lower() != 'true' for row in empty_rows)}",
        "",
        "## Reproducibility Entry Points",
        "- `requirements.txt`",
        "- `run_reproduce.sh`",
        "- `06_scoring/PAPER_OUTPUT/09_reproducibility/reproducibility_guide.md`",
        "- `06_scoring/PAPER_OUTPUT/09_reproducibility/empty_csv_audit.csv`",
        "- `06_scoring/PAPER_OUTPUT/09_reproducibility/empty_csv_explanation.md`",
    ]
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build reproducibility package for paper outputs.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_reproducibility_package(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
