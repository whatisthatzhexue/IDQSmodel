from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_paper_cross_validation_section import build_paper_cross_validation_section
from build_paper_data_section import build_paper_data_section
from build_paper_limitations import build_paper_limitations
from build_paper_methods_section import build_paper_methods_section
from build_paper_ready_report import build_paper_ready_report
from build_paper_reproducibility_package import build_paper_reproducibility_package
from build_paper_results_tables import build_paper_results_tables
from build_paper_scoring_framework import build_paper_scoring_framework
from build_paper_stability_and_qc import build_paper_stability_and_qc
from paper_utils import REQUIRED_PAPER_FILES, ensure_paper_output_dirs
from run_final_data_freeze import run_final_data_freeze
from run_paper_convergence_loop import run_paper_convergence_loop
from run_paper_readiness_check import run_paper_readiness_check
from scoring_utils import SCORING_ROOT, save_json


def run_paper_ready_pipeline(root: Path = SCORING_ROOT, max_convergence_rounds: int = 5, skip_tests: bool = False) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    steps: list[dict[str, Any]] = []
    steps.append({"step": "run_paper_readiness_check_initial", "result": run_paper_readiness_check(root)})
    steps.append({"step": "run_paper_convergence_loop", "result": run_paper_convergence_loop(root, max_convergence_rounds)})
    steps.append({"step": "run_final_data_freeze", "result": run_final_data_freeze(root)})
    steps.append({"step": "run_paper_readiness_check_after_freeze", "result": run_paper_readiness_check(root)})
    steps.append({"step": "build_paper_methods_section", "result": build_paper_methods_section(root)})
    steps.append({"step": "build_paper_data_section", "result": build_paper_data_section(root)})
    steps.append({"step": "build_paper_scoring_framework", "result": build_paper_scoring_framework(root)})
    steps.append({"step": "build_paper_results_tables", "result": build_paper_results_tables(root)})
    steps.append({"step": "build_paper_stability_and_qc", "result": build_paper_stability_and_qc(root)})
    steps.append({"step": "build_paper_cross_validation_section", "result": build_paper_cross_validation_section(root)})
    steps.append({"step": "build_paper_limitations", "result": build_paper_limitations(root)})
    steps.append({"step": "build_paper_reproducibility_package", "result": build_paper_reproducibility_package(root)})
    readiness_after_repro = run_paper_readiness_check(root)
    steps.append({"step": "run_paper_readiness_check_after_reproducibility", "result": readiness_after_repro})
    status = build_paper_ready_report(root)
    steps.append({"step": "build_paper_ready_report", "result": status})
    validation = validate_paper_outputs(root)
    steps.append({"step": "validate_paper_outputs", "result": validation})
    tests = {"skipped": True}
    if not skip_tests:
        tests = _run_tests(root)
        steps.append({"step": "run_tests", "result": tests})
    summary = {
        "paper_ready": status["paper_ready"],
        "freeze_allowed": status["freeze_allowed"],
        "mda_ready": status["mda_ready"],
        "news_ready": status["news_ready"],
        "cross_validation_ready": status["cross_validation_ready"],
        "reproducibility_ready": status["reproducibility_ready"],
        "blocking_reasons": status["blocking_reasons"],
        "advisory_warnings": status["advisory_warnings"],
        "recommended_next_actions": status["recommended_next_actions"],
        "can_write": _can_write(status),
        "cannot_write": _cannot_write(status),
        "validation": validation,
        "tests": tests,
        "steps": steps,
    }
    save_json(root / "PAPER_OUTPUT" / "00_status" / "paper_ready_pipeline_run.json", summary)
    return summary


def validate_paper_outputs(root: Path = SCORING_ROOT) -> dict[str, Any]:
    missing = [rel_path for rel_path in REQUIRED_PAPER_FILES if not (root / rel_path).exists()]
    status_path = root / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json"
    report_path = root / "PAPER_OUTPUT" / "PAPER_READY_REPORT.md"
    return {
        "passed": not missing and status_path.exists() and report_path.exists(),
        "missing_files": missing,
        "status_exists": status_path.exists(),
        "report_exists": report_path.exists(),
    }


def _run_tests(root: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "06_scoring/tests", "-q"]
    completed = subprocess.run(command, cwd=root.parent, text=True, capture_output=True)
    log_path = root / "PAPER_OUTPUT" / "09_reproducibility" / "pytest_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {"passed": completed.returncode == 0, "returncode": completed.returncode, "log_path": str(log_path)}


def _can_write(status: dict[str, Any]) -> list[str]:
    if status["paper_ready"]:
        return ["final empirical results", "methods", "scoring framework", "cross-validation interpretation", "reproducibility"]
    allowed = ["methodology", "scoring framework", "pipeline validation", "limitations", "reproducibility"]
    if status["mda_ready"] and not status["news_ready"]:
        allowed.append("MD&A-only preliminary analysis clearly labeled as MD&A-only")
    return allowed


def _cannot_write(status: dict[str, Any]) -> list[str]:
    blocked = []
    if not status["paper_ready"]:
        blocked.append("final empirical results")
    if not status["news_ready"]:
        blocked.extend(["real News empirical results", "substantive cross-validation conclusion"])
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the full paper-ready package.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--max-convergence-rounds", type=int, default=5)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    result = run_paper_ready_pipeline(args.root, args.max_convergence_rounds, args.skip_tests)
    printable = {key: result[key] for key in ["paper_ready", "freeze_allowed", "mda_ready", "news_ready", "cross_validation_ready", "reproducibility_ready", "blocking_reasons", "advisory_warnings", "recommended_next_actions", "can_write", "cannot_write"]}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    return 0 if result["validation"]["passed"] and (args.skip_tests or result["tests"].get("passed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
