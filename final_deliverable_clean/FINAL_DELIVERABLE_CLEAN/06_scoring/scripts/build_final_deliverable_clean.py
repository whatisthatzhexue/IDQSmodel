from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_utils import load_json_safe
from scoring_utils import SCORING_ROOT, save_json
from stability_common import write_markdown


STATUS_FIELDS = [
    "paper_ready",
    "freeze_allowed",
    "mda_structural_ready",
    "mda_semantic_ready",
    "mda_human_validation_ready",
    "mda_ready",
    "news_ready",
    "cross_validation_ready",
    "cross_validation_pipeline_ready",
    "cross_validation_empirical_ready",
    "mock_mode",
    "systemic_conflict_flag",
    "reproducibility_ready",
    "mda_final_dataset_size",
    "news_final_dataset_size",
    "news_company_focused_dataset_size",
    "synthetic_news_count",
    "claim_link_count",
    "eligible_company_year_pair_count",
    "blocking_reasons",
    "advisory_warnings",
    "recommended_next_actions",
]

FINAL_STATEMENTS = [
    "MD&A-only scoring is complete as a model-generated candidate dataset.",
    "Cross-validation is framework-ready but not empirically interpretable.",
    "Combined paper_ready remains false.",
    "MD&A-only results may be used only with caveat and recommended human validation.",
]


def build_final_deliverable_clean(root: Path = SCORING_ROOT) -> dict[str, Any]:
    out = root / "FINAL_DELIVERABLE_CLEAN"
    if out.exists():
        shutil.rmtree(out)
    for dirname in ["MDA", "NEWS", "CROSS_VALIDATION", "QUALITY", "REPRODUCIBILITY", "HUMAN_VALIDATION"]:
        (out / dirname).mkdir(parents=True, exist_ok=True)

    status = _final_status(root)
    save_json(out / "FINAL_STATUS.json", status)
    _write_final_report(out / "FINAL_REPORT.md", status)
    _write_manifest(out / "FINAL_PACKAGE_MANIFEST.md", status)

    _copy_mda(root, out)
    _copy_news(root, out, status)
    _copy_cross_validation(root, out, status)
    _copy_quality(root, out)
    _copy_reproducibility(root, out)
    _copy_human_validation(root, out)
    _copy_standalone_project(root, out, status)
    _write_root_reproduce_script(out)
    _sanitize_tree(out, root)
    _write_hash_manifest(out / "REPRODUCIBILITY" / "file_hash_manifest.csv", out)

    return {
        "status": "success",
        "output_dir": str(out),
        "paper_ready": bool(status["paper_ready"]),
        "mda_ready": bool(status["mda_ready"]),
        "news_ready": bool(status["news_ready"]),
        "cross_validation_ready": bool(status["cross_validation_ready"]),
        "reproducibility_ready": bool(status["reproducibility_ready"]),
    }


def _final_status(root: Path) -> dict[str, Any]:
    readiness, ok = load_json_safe(root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json")
    if not ok or not isinstance(readiness, dict):
        readiness = {}
    status = {field: readiness.get(field) for field in STATUS_FIELDS}
    defaults = {
        "paper_ready": False,
        "freeze_allowed": False,
        "mda_structural_ready": False,
        "mda_semantic_ready": False,
        "mda_human_validation_ready": False,
        "mda_ready": False,
        "news_ready": False,
        "cross_validation_ready": False,
        "cross_validation_pipeline_ready": False,
        "cross_validation_empirical_ready": False,
        "mock_mode": False,
        "systemic_conflict_flag": False,
        "reproducibility_ready": False,
        "mda_final_dataset_size": 0,
        "news_final_dataset_size": 0,
        "news_company_focused_dataset_size": _csv_row_count(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv"),
        "synthetic_news_count": 0,
        "claim_link_count": 0,
        "eligible_company_year_pair_count": 0,
        "blocking_reasons": [],
        "advisory_warnings": [],
        "recommended_next_actions": [],
    }
    for key, value in defaults.items():
        if status.get(key) is None:
            status[key] = value
    if int(status.get("news_final_dataset_size") or 0) == 0 or not status.get("news_ready"):
        status["claim_link_count"] = 0
        status["eligible_company_year_pair_count"] = 0
        status["cross_validation_pipeline_ready"] = True
        status["cross_validation_empirical_ready"] = False
        status["cross_validation_ready"] = False
    status.update(
        {
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "deliverable_type": "final_deliverable_clean_blocker_aware",
            "mda_only_candidate_complete": bool(status["mda_ready"] and int(status["mda_final_dataset_size"] or 0) > 0),
            "news_empirical_blocked": not bool(status["news_ready"]),
            "combined_paper_ready": bool(status["paper_ready"]),
            "real_news_missing_blocker_exists": (
                (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md").exists()
                and not bool(status["news_ready"])
            ),
        }
    )
    return status


def _copy_mda(root: Path, out: Path) -> None:
    _copy_or_empty(root / "FINAL_OUTPUT" / "mda_final_dataset.csv", out / "MDA" / "mda_final_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "06_ratings" / "mda_final_v2" / "mda_final_ratings_long.csv", out / "MDA" / "mda_final_ratings_long.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "06_ratings" / "mda_final_v2" / "mda_final_document_scores.csv", out / "MDA" / "mda_final_document_scores_v2.csv", "document_id\n")
    _copy_or_empty(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv", out / "MDA" / "mda_original_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv", out / "MDA" / "mda_original_ratings_long.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores_repaired.csv", out / "MDA" / "mda_repaired_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long_repaired.csv", out / "MDA" / "mda_repaired_ratings_long.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "08_reports" / "mda_semantic_evidence_audit.md", out / "MDA" / "mda_semantic_evidence_audit.md", "# MD&A Semantic Evidence Audit\n")
    _copy_or_empty(root / "08_reports" / "mda_semantic_evidence_audit.csv", out / "MDA" / "mda_semantic_evidence_audit.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "07_review" / "mda_semantic_evidence_review_pool.csv", out / "MDA" / "mda_semantic_evidence_review_pool.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "08_reports" / "mda_semantic_evidence_audit_v2.csv", out / "MDA" / "mda_semantic_evidence_audit_v2.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "08_reports" / "mda_semantic_evidence_audit_v2.md", out / "MDA" / "mda_semantic_evidence_audit_v2.md", "# MD&A Semantic Evidence Audit V2\n")
    _copy_or_empty(root / "07_review" / "mda_semantic_evidence_review_pool_v2.csv", out / "MDA" / "mda_semantic_evidence_review_pool_v2.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "08_reports" / "mda_semantic_repair_v2_actions.csv", out / "MDA" / "mda_semantic_repair_v2_actions.csv", "document_id,dimension_code,repair_status\n")
    _copy_or_empty(root / "08_reports" / "mda_semantic_repair_v2_report.md", out / "MDA" / "mda_semantic_repair_v2_report.md", "# MD&A Semantic Repair V2 Report\n")


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _copy_news(root: Path, out: Path, status: dict[str, Any]) -> None:
    _copy_or_empty(root / "FINAL_OUTPUT" / "news_final_dataset.csv", out / "NEWS" / "news_final_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "FINAL_OUTPUT" / "news_broad_dataset.csv", out / "NEWS" / "news_broad_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv", out / "NEWS" / "news_company_focused_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", out / "NEWS" / "news_final_ratings_long.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "06_ratings" / "news_final_full" / "news_final_failed_cases.csv", out / "NEWS" / "news_final_failed_cases.csv", "document_id,dimension_code\n")
    _copy_or_empty(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", out / "NEWS" / "news_model_document_scores.csv", "document_id\n")
    _copy_or_empty(root / "01_registry" / "news_registry.csv", out / "NEWS" / "news_registry.csv", "document_id,include_flag\n")
    _copy_or_empty(root / "01_registry" / "news_registry_high_confidence.csv", out / "NEWS" / "news_registry_high_confidence.csv", "document_id,include_flag\n")
    _copy_or_empty(root / "01_registry" / "company_alias_map.csv", out / "NEWS" / "company_alias_map.csv", "ticker,alias\n")
    _copy_or_empty(root / "08_reports" / "filter_news_relevance_audit.csv", out / "NEWS" / "filter_news_relevance_audit.csv", "document_id,relevance_status\n")
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
    _copy_or_empty(root / "08_reports" / "filter_news_registry_report.md", out / "NEWS" / "filter_news_registry_report.md", "# News Registry Report\n")
    _copy_or_empty(root / "08_reports" / "filter_news_final_scoring_report.md", out / "NEWS" / "filter_news_final_scoring_report.md", "# News Scoring Report\n")
    _copy_or_empty(root / "08_reports" / "news_registry_selection_report.md", out / "NEWS" / "news_registry_selection_report.md", "# News Registry Selection Report\n")
    _copy_or_empty(root / "08_reports" / "news_qwen_runtime_config.md", out / "NEWS" / "news_qwen_runtime_config.md", "# News Runtime Config\n")
    _copy_or_empty(root / "08_reports" / "qwen3_8b_limitations_news.md", out / "NEWS" / "qwen3_8b_limitations_news.md", "# qwen3:8b News Limitations\n")
    if not status.get("news_ready"):
        _copy_or_empty(root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md", out / "NEWS" / "real_news_missing_blocker.md", "# Real News Missing Blocker\n")


def _copy_cross_validation(root: Path, out: Path, status: dict[str, Any]) -> None:
    _copy_or_empty(root / "FINAL_OUTPUT" / "cross_validation_final_summary.csv", out / "CROSS_VALIDATION" / "cross_validation_final_summary.csv", "metric,value\n")
    if int(status.get("news_final_dataset_size") or 0) == 0:
        _copy_or_empty(root / "09_cross_validation" / "claim_links.csv", out / "CROSS_VALIDATION" / "stale_claim_links_inactive.csv", "link_id\n")
        (out / "CROSS_VALIDATION" / "claim_links.csv").write_text("link_id\n", encoding="utf-8")
        (out / "CROSS_VALIDATION" / "mda_news_cross_validation_pairs.csv").write_text("mda_document_id\n", encoding="utf-8")
    else:
        _copy_or_empty(root / "09_cross_validation" / "claim_links.csv", out / "CROSS_VALIDATION" / "claim_links.csv", "link_id\n")
        _copy_or_empty(root / "09_cross_validation" / "mda_news_cross_validation_pairs.csv", out / "CROSS_VALIDATION" / "mda_news_cross_validation_pairs.csv", "mda_document_id\n")
    _copy_or_empty(root / "09_cross_validation" / "cross_validation_report.md", out / "CROSS_VALIDATION" / "cross_validation_report.md", "# Cross-validation Report\n\nNot paper-ready until interpretability blockers are resolved.\n")


def _copy_quality(root: Path, out: Path) -> None:
    _copy_or_empty(root / "FINAL_OUTPUT" / "final_quality_report.md", out / "QUALITY" / "final_quality_report.md", "# Final Quality Report\n")
    _copy_or_empty(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json", out / "QUALITY" / "dataset_frozen_metadata.json", "{}\n")
    _copy_or_empty(root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json", out / "QUALITY" / "paper_readiness_check.json", "{}\n")
    _copy_or_empty(root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_report.md", out / "QUALITY" / "paper_readiness_report.md", "# Paper Readiness Report\n")
    _copy_or_empty(root / "PAPER_OUTPUT" / "09_reproducibility" / "empty_csv_audit.csv", out / "QUALITY" / "empty_csv_audit.csv", "relative_path,row_count,status,notes\n")


def _copy_reproducibility(root: Path, out: Path) -> None:
    _copy_or_empty(root.parent / "requirements.txt", out / "REPRODUCIBILITY" / "requirements.txt", "pytest\n")
    _copy_or_empty(root.parent / "run_reproduce.sh", out / "REPRODUCIBILITY" / "run_reproduce.sh", "#!/usr/bin/env bash\n")
    for name in ["environment_summary.json", "environment_summary.md", "command_log.md", "reproducibility_guide.md", "file_manifest.csv", "run_reproduce.log"]:
        _copy_or_empty(root / "PAPER_OUTPUT" / "09_reproducibility" / name, out / "REPRODUCIBILITY" / name, "\n")
    _copy_or_empty(root.parent / "FINAL_PACKAGE_MANIFEST.md", out / "REPRODUCIBILITY" / "top_level_final_package_manifest.md", "# Final Package Manifest\n")


def _copy_standalone_project(root: Path, out: Path, status: dict[str, Any]) -> None:
    packaged = out / "06_scoring"
    source_root = Path(__file__).resolve().parents[1]
    for dirname in ["scripts", "tests", "00_scorebook", "04_prompts"]:
        src = source_root / dirname
        dst = packaged / dirname
        if src.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*benchmark*", "*mock*"), dirs_exist_ok=True)
    copy_pairs = [
        (root / "MDA", packaged / "MDA"),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
    for rel in [
        "01_registry/mda_registry.csv",
        "01_registry/news_registry.csv",
        "01_registry/news_registry_high_confidence.csv",
        "06_ratings/mda_final_v2/mda_final_ratings_long.csv",
        "06_ratings/mda_final_v2/mda_final_document_scores.csv",
        "06_ratings/news_final_full/news_final_ratings_long.csv",
        "06_ratings/news_final_full/news_final_document_scores.csv",
        "08_reports/mda_semantic_evidence_audit_v2.csv",
        "08_reports/mda_semantic_repair_v2_actions.csv",
        "08_reports/filter_news_relevance_audit.csv",
        "08_reports/news_possible_false_positives.csv",
        "08_reports/news_wrong_company_candidates.csv",
        "PAPER_OUTPUT/00_status/paper_readiness_check.json",
        "FINAL_OUTPUT/dataset_frozen_metadata.json",
        "FINAL_OUTPUT/mda_final_dataset.csv",
        "FINAL_OUTPUT/news_final_dataset.csv",
        "FINAL_OUTPUT/news_broad_dataset.csv",
        "FINAL_OUTPUT/news_company_focused_dataset.csv",
    ]:
        _copy_or_empty(root / rel, packaged / rel, _empty_content_for(rel))
    save_json(packaged / "PAPER_OUTPUT" / "00_status" / "standalone_expected_status.json", status)
    _write_standalone_validator(packaged / "scripts" / "standalone_clean_reproduce.py")


def _write_root_reproduce_script(out: Path) -> None:
    path = out / "run_reproduce.sh"
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'cd "$(dirname "$0")"',
                'PYTHON_BIN="${PYTHON_BIN:-python3}"',
                'if [ "${SKIP_VENV:-0}" != "1" ]; then',
                '  "${PYTHON_BIN}" -m venv .venv',
                "  . .venv/bin/activate",
                "  python -m pip install -r REPRODUCIBILITY/requirements.txt",
                "  PYTHON=python",
                "else",
                '  PYTHON="${PYTHON_BIN}"',
                "fi",
                '"$PYTHON" 06_scoring/scripts/standalone_clean_reproduce.py',
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def _write_standalone_validator(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT.parent


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    status = json.loads((PKG / "FINAL_STATUS.json").read_text(encoding="utf-8"))
    mda_rows = rows(PKG / "MDA" / "mda_final_document_scores.csv")
    rating_rows = rows(PKG / "MDA" / "mda_final_ratings_long.csv")
    news_rows = rows(PKG / "NEWS" / "news_final_document_scores.csv")
    news_rating_rows = rows(PKG / "NEWS" / "news_final_ratings_long.csv")
    claim_links = rows(PKG / "CROSS_VALIDATION" / "claim_links.csv")
    if len(mda_rows) != int(status.get("mda_final_dataset_size", 0) or 0):
        raise SystemExit(f"mda row count mismatch: {len(mda_rows)}")
    if len(rating_rows) != len(mda_rows) * 5:
        raise SystemExit(f"ratings row count mismatch: {len(rating_rows)}")
    if len(news_rows) != int(status.get("news_final_dataset_size", 0) or 0):
        raise SystemExit(f"news row count mismatch: {len(news_rows)}")
    if len(news_rating_rows) != len(news_rows) * 5:
        raise SystemExit(f"news ratings row count mismatch: {len(news_rating_rows)}")
    if int(status.get("news_final_dataset_size", 0) or 0) == 0 and claim_links:
        raise SystemExit("claim_links must be empty when News is unavailable")
    bad_paths = []
    for path in PKG.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".txt", ".sh", ".py"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            blocked_user_path = "/Users/" + "zhaojiahao"
            if blocked_user_path in text:
                bad_paths.append(str(path.relative_to(PKG)))
    if bad_paths:
        raise SystemExit("absolute paths found: " + ", ".join(bad_paths[:5]))
    print(json.dumps({"standalone_reproduce": True, "mda_rows": len(mda_rows), "rating_rows": len(rating_rows), "news_rows": len(news_rows), "news_rating_rows": len(news_rating_rows), "claim_links": len(claim_links)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
""",
        encoding="utf-8",
    )


def _empty_content_for(rel: str) -> str:
    if rel.endswith(".json"):
        return "{}\n"
    if rel.endswith(".md"):
        return "# Missing\n"
    return "document_id\n"


def _copy_human_validation(root: Path, out: Path) -> None:
    src = root / "HUMAN_VALIDATION"
    dst = out / "HUMAN_VALIDATION"
    if not src.exists():
        return
    for path in sorted(item for item in src.rglob("*") if item.is_file()):
        target = dst / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _sanitize_tree(out: Path, root: Path) -> None:
    replacements = [
        (str(root) + "/", "06_scoring/"),
        (str(root.parent) + "/", ""),
        (str(root), "06_scoring"),
        (str(root.parent), "."),
        ("", ""),
        ("<LOCAL_USER>", "<LOCAL_USER>"),
    ]
    for path in sorted(item for item in out.rglob("*") if item.is_file()):
        if path.suffix.lower() not in {".csv", ".json", ".md", ".txt", ".sh", ".py"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = text
        for old, new in replacements:
            cleaned = cleaned.replace(old, new)
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8")


def _write_final_report(path: Path, status: dict[str, Any]) -> None:
    news_statement = (
        f"News scoring is ready with {status.get('news_final_dataset_size')} real non-synthetic final rows."
        if status.get("news_ready")
        else "News empirical analysis is blocked due to absence of real company-specific full-text News."
    )
    cross_statement = (
        f"Cross-validation has {status.get('claim_link_count')} claim links but is not paper-ready until interpretability blockers are resolved."
        if int(status.get("claim_link_count") or 0) > 0
        else "Cross-validation is framework-ready but not empirically interpretable."
    )
    final_statements = [FINAL_STATEMENTS[0], news_statement, cross_statement, *FINAL_STATEMENTS[2:]]
    lines = [
        "# Final Deliverable Clean Report",
        "",
        "## Final Statements",
        *[f"- {statement}" for statement in final_statements],
        "",
        "## Status",
    ]
    for field in STATUS_FIELDS:
        lines.append(f"- {field}: {status.get(field)}")
    if status.get("blocking_reasons"):
        lines.extend(["", "## Blocking Reasons"])
        for blocker in status["blocking_reasons"]:
            lines.append(f"- {blocker}")
    if status.get("recommended_next_actions"):
        lines.extend(["", "## Recommended Next Actions"])
        for action in status["recommended_next_actions"]:
            lines.append(f"- {action}")
    write_markdown(path, lines)


def _write_manifest(path: Path, status: dict[str, Any]) -> None:
    lines = [
        "# Final Package Manifest",
        "",
        "- package_type: final_deliverable_clean_blocker_aware",
        f"- paper_ready: {str(bool(status.get('paper_ready'))).lower()}",
        f"- freeze_allowed: {str(bool(status.get('freeze_allowed'))).lower()}",
        f"- mda_structural_ready: {str(bool(status.get('mda_structural_ready'))).lower()}",
        f"- mda_semantic_ready: {str(bool(status.get('mda_semantic_ready'))).lower()}",
        f"- mda_human_validation_ready: {str(bool(status.get('mda_human_validation_ready'))).lower()}",
        f"- mda_ready: {str(bool(status.get('mda_ready'))).lower()}",
        f"- news_ready: {str(bool(status.get('news_ready'))).lower()}",
        f"- cross_validation_ready: {str(bool(status.get('cross_validation_ready'))).lower()}",
        f"- cross_validation_pipeline_ready: {str(bool(status.get('cross_validation_pipeline_ready'))).lower()}",
        f"- cross_validation_empirical_ready: {str(bool(status.get('cross_validation_empirical_ready'))).lower()}",
        f"- reproducibility_ready: {str(bool(status.get('reproducibility_ready'))).lower()}",
        "",
        "## Sections",
        "- MDA",
        "- NEWS",
        "- CROSS_VALIDATION",
        "- QUALITY",
        "- REPRODUCIBILITY",
        "- HUMAN_VALIDATION",
    ]
    write_markdown(path, lines)


def _copy_or_empty(src: Path, dst: Path, empty_content: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text(empty_content, encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Build clean final deliverable with explicit readiness and blocker status.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_final_deliverable_clean(root=args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
