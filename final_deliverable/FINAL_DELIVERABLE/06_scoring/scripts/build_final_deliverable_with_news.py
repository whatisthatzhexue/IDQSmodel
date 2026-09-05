from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_utils import load_json_safe
from scoring_utils import SCORING_ROOT, save_json
from stability_common import write_markdown


STATUS_FIELDS = [
    "mda_ready",
    "news_ready",
    "cross_validation_ready",
    "cross_validation_pipeline_ready",
    "cross_validation_empirical_ready",
    "mock_mode",
    "systemic_conflict_flag",
    "reproducibility_ready",
    "freeze_allowed",
    "paper_ready",
    "mda_final_dataset_size",
    "news_final_dataset_size",
    "news_company_focused_dataset_size",
    "synthetic_news_count",
    "eligible_company_year_pair_count",
    "claim_link_count",
    "system_level_error_rate",
    "blocking_reasons",
    "recommended_next_actions",
]


def build_final_deliverable_with_news(root: Path = SCORING_ROOT, test_results: str | None = None) -> dict[str, Any]:
    out = root / "FINAL_DELIVERABLE_WITH_NEWS"
    if out.exists():
        shutil.rmtree(out)
    for name in ["MDA", "NEWS", "CROSS_VALIDATION", "QUALITY", "REPRODUCIBILITY"]:
        (out / name).mkdir(parents=True, exist_ok=True)

    status = _final_status(root)
    save_json(out / "FINAL_STATUS.json", status)
    _write_final_report(out / "FINAL_REPORT.md", status)
    _write_manifest(out / "FINAL_PACKAGE_MANIFEST.md", status)

    _copy_or_empty(root / "FINAL_OUTPUT" / "mda_final_dataset.csv", out / "MDA" / "mda_final_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv", out / "MDA" / "mda_final_ratings_long.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "FINAL_OUTPUT" / "news_final_dataset.csv", out / "NEWS" / "news_final_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "FINAL_OUTPUT" / "news_broad_dataset.csv", out / "NEWS" / "news_broad_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv", out / "NEWS" / "news_company_focused_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", out / "NEWS" / "news_final_ratings_long.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "06_ratings" / "news_final_full" / "news_final_failed_cases.csv", out / "NEWS" / "news_final_failed_cases.csv", "document_id,status,reason\n")
    _copy_or_empty(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", out / "NEWS" / "news_model_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "01_registry" / "news_registry.csv", out / "NEWS" / "news_registry.csv", "document_id,include_flag\n")
    _copy_or_empty(root / "08_reports" / "filter_news_registry_report.md", out / "NEWS" / "filter_news_registry_report.md", "# News Registry Report\n")
    _copy_or_empty(root / "08_reports" / "filter_news_final_scoring_report.md", out / "NEWS" / "filter_news_final_scoring_report.md", "# News Scoring Report\n")
    _copy_or_empty(root / "08_reports" / "news_registry_selection_report.md", out / "NEWS" / "news_registry_selection_report.md", "# News Registry Selection Report\n")
    _copy_or_empty(root / "08_reports" / "news_manual_preserve_review_queue.csv", out / "NEWS" / "news_manual_preserve_review_queue.csv", "document_id,relevance_status\n")
    _copy_or_empty(root / "08_reports" / "news_possible_false_positives.csv", out / "NEWS" / "news_possible_false_positives.csv", "document_id,relevance_status\n")
    _copy_or_empty(root / "08_reports" / "news_wrong_company_candidates.csv", out / "NEWS" / "news_wrong_company_candidates.csv", "document_id,relevance_status\n")
    _copy_or_empty(root / "08_reports" / "news_company_focus_audit.csv", out / "NEWS" / "news_company_focus_audit.csv", "document_id,suspect_not_company_focused\n")
    _copy_or_empty(root / "08_reports" / "news_suspect_not_company_focused.csv", out / "NEWS" / "news_suspect_not_company_focused.csv", "document_id,suspect_not_company_focused\n")
    _copy_or_empty(root / "08_reports" / "news_company_focused_main_subset.csv", out / "NEWS" / "news_company_focused_main_subset.csv", "document_id,strict_main_sample_include\n")
    _copy_or_empty(root / "08_reports" / "news_broad_vs_company_focused_robustness.csv", out / "NEWS" / "news_broad_vs_company_focused_robustness.csv", "broad_rows,company_focused_rows\n")
    _copy_or_empty(root / "08_reports" / "news_broad_vs_company_focused_robustness.md", out / "NEWS" / "news_broad_vs_company_focused_robustness.md", "# News Broad Vs Company-Focused Robustness\n")
    _copy_or_empty(root / "08_reports" / "news_special_case_recheck.csv", out / "NEWS" / "news_special_case_recheck.csv", "original_document_id,current_filter_news_id\n")
    _copy_or_empty(root / "08_reports" / "news_special_case_recheck.md", out / "NEWS" / "news_special_case_recheck.md", "# News Special Case Recheck\n")
    _copy_or_empty(root / "08_reports" / "news_qwen_runtime_config.md", out / "NEWS" / "news_qwen_runtime_config.md", "# News Runtime Config\n")
    _copy_or_empty(root / "08_reports" / "qwen3_8b_limitations_news.md", out / "NEWS" / "qwen3_8b_limitations_news.md", "# qwen3:8b News Limitations\n")
    if not status.get("news_ready"):
        _copy_or_empty(root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md", out / "NEWS" / "real_news_missing_blocker.md", "# Real News Missing Blocker\n")

    _copy_or_empty(root / "09_cross_validation" / "final_real_run" / "eligible_pairs.csv", out / "CROSS_VALIDATION" / "eligible_pairs.csv", "ticker,year\n")
    _copy_or_empty(root / "09_cross_validation" / "final_real_run" / "claim_links.csv", out / "CROSS_VALIDATION" / "claim_links.csv", "claim_link_id\n")
    _copy_or_empty(root / "09_cross_validation" / "final_real_run" / "company_year_cross_validation_summary.csv", out / "CROSS_VALIDATION" / "company_year_cross_validation_summary.csv", "ticker,year\n")
    _copy_or_empty(root / "09_cross_validation" / "final_real_run" / "cross_validation_report.md", out / "CROSS_VALIDATION" / "cross_validation_report.md", "# Cross-validation Report\n\nNot paper-ready until interpretability blockers are resolved.\n")
    _copy_or_empty(root / "09_cross_validation" / "final_real_run" / "cross_validation_run_metadata.json", out / "CROSS_VALIDATION" / "cross_validation_run_metadata.json", "{}\n")

    _copy_or_empty(root / "FINAL_OUTPUT" / "final_quality_report.md", out / "QUALITY" / "final_quality_report.md", "# Final Quality Report\n")
    _copy_or_empty(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json", out / "QUALITY" / "final_freeze_metadata.json", "{}\n")
    _copy_or_empty(root / "07_review" / "review_log.csv", out / "QUALITY" / "review_log.csv", "document_id,review_reasons\n")

    _copy_or_empty(root.parent / "requirements.txt", out / "REPRODUCIBILITY" / "requirements.txt", "pytest\n")
    _copy_or_empty(root.parent / "run_reproduce.sh", out / "REPRODUCIBILITY" / "run_reproduce.sh", "#!/usr/bin/env bash\n")
    _write_environment_summary(out / "REPRODUCIBILITY" / "environment_summary.md")
    (out / "REPRODUCIBILITY" / "command_log.md").write_text(_command_log_text(status), encoding="utf-8")
    (out / "REPRODUCIBILITY" / "test_results.txt").write_text(test_results or "", encoding="utf-8")
    _write_hash_manifest(out / "REPRODUCIBILITY" / "file_hash_manifest.csv", out)

    return {"status": "success", "output_dir": str(out), "status_path": str(out / "FINAL_STATUS.json"), "paper_ready": status["paper_ready"]}


def _final_status(root: Path) -> dict[str, Any]:
    readiness, readiness_ok = load_json_safe(root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json")
    if not readiness_ok or not isinstance(readiness, dict):
        readiness = {}
    status = {field: readiness.get(field) for field in STATUS_FIELDS}
    defaults = {
        "mda_ready": False,
        "news_ready": False,
        "cross_validation_ready": False,
        "cross_validation_pipeline_ready": False,
        "cross_validation_empirical_ready": False,
        "mock_mode": False,
        "systemic_conflict_flag": False,
        "reproducibility_ready": False,
        "freeze_allowed": False,
        "paper_ready": False,
        "mda_final_dataset_size": _csv_row_count(root / "FINAL_OUTPUT" / "mda_final_dataset.csv"),
        "news_final_dataset_size": _csv_row_count(root / "FINAL_OUTPUT" / "news_final_dataset.csv"),
        "news_company_focused_dataset_size": _csv_row_count(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv"),
        "synthetic_news_count": 0,
        "eligible_company_year_pair_count": 0,
        "claim_link_count": 0,
        "system_level_error_rate": 1.0,
        "blocking_reasons": [],
        "recommended_next_actions": [],
    }
    for key, value in defaults.items():
        if status.get(key) is None:
            status[key] = value
    status["created_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    status["deliverable_type"] = "combined_with_news_blocker_aware"
    status["real_news_missing_blocker_exists"] = (
        (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md").exists()
        and not bool(status.get("news_ready"))
    )
    return status


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _write_final_report(path: Path, status: dict[str, Any]) -> None:
    lines = ["# Final Deliverable With News Report", ""]
    for field in STATUS_FIELDS:
        lines.append(f"- {field}: {status.get(field)}")
    if not status.get("paper_ready"):
        lines.extend(["", "## Blocking Reasons"])
        for blocker in status.get("blocking_reasons") or []:
            lines.append(f"- {blocker}")
    write_markdown(path, lines)


def _write_manifest(path: Path, status: dict[str, Any]) -> None:
    write_markdown(
        path,
        [
            "# Final Package Manifest",
            "",
            f"- paper_ready: {str(bool(status.get('paper_ready'))).lower()}",
            f"- freeze_allowed: {str(bool(status.get('freeze_allowed'))).lower()}",
            "- package_type: combined_with_news_blocker_aware",
        ],
    )


def _copy_or_empty(src: Path, dst: Path, empty_content: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text(empty_content, encoding="utf-8")


def _write_environment_summary(path: Path) -> None:
    completed = subprocess.run(["python3", "--version"], text=True, capture_output=True)
    write_markdown(path, ["# Environment Summary", "", f"- python: {(completed.stdout or completed.stderr).strip()}"])


def _command_log_text(status: dict[str, Any]) -> str:
    return "\n".join(
        [
            "- python3 scripts/audit_filter_news_relevance.py --mode manual_preserve",
            "- python3 scripts/build_news_registry_from_filter_news.py",
            "- python3 scripts/extract_filter_news_texts.py",
            "- python3 scripts/run_filter_news_final_scoring.py --model qwen3:8b",
            "- python3 scripts/run_final_data_freeze.py",
            "- python3 scripts/run_paper_readiness_check.py",
            f"- paper_ready: {str(bool(status.get('paper_ready'))).lower()}",
        ]
    )


def _write_hash_manifest(path: Path, base: Path) -> None:
    rows = []
    for file_path in sorted(item for item in base.rglob("*") if item.is_file()):
        if file_path == path:
            continue
        rows.append(
            {
                "relative_path": str(file_path.relative_to(base)),
                "sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
                "size_bytes": file_path.stat().st_size,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relative_path", "sha256", "size_bytes"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FINAL_DELIVERABLE_WITH_NEWS without fabricating missing News data.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--test-results", default="")
    args = parser.parse_args()
    print(json.dumps(build_final_deliverable_with_news(root=args.root, test_results=args.test_results), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
