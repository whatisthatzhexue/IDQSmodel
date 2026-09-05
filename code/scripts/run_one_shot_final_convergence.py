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

from build_blocker_status_reports import build_blocker_status_reports
from discover_real_news_across_project import discover_real_news_across_project
from recover_missing_mda_dimensions import recover_missing_mda_dimensions
from rebuild_mda_stability_metrics import rebuild_mda_stability_metrics
from repair_ollama_runtime import repair_ollama_runtime
from run_final_data_freeze import run_final_data_freeze
from run_mda_final_full_scoring import run_mda_final_full_scoring
from run_mda_final_repeatability_test import run_mda_final_repeatability_test
from run_paper_readiness_check import run_paper_readiness_check
from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import NEWS_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, REGISTRY_HEADERS, SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from snapshot_pre_final_convergence import snapshot_pre_final_convergence
from stability_common import write_markdown


FINAL_FIELDS = [
    "ollama_runtime_healthy",
    "mda_expected_count",
    "mda_final_dataset_size",
    "mda_full_scoring_coverage_rate",
    "mda_operational_completion_rate",
    "mda_repeatability_pass_rate",
    "mda_schema_success_rate",
    "mda_evidence_match_rate",
    "news_real_files_found",
    "news_final_dataset_size",
    "news_operational_completion_rate",
    "news_repeatability_pass_rate",
    "synthetic_news_count",
    "eligible_company_year_pair_count",
    "claim_link_count",
    "cross_validation_pipeline_ready",
    "cross_validation_empirical_ready",
    "pipeline_operational_failure_rate",
    "final_dataset_validation_error_rate",
    "mda_ready",
    "news_ready",
    "cross_validation_ready",
    "reproducibility_ready",
    "freeze_allowed",
    "paper_ready",
    "blocking_reasons",
    "recommended_next_actions",
]


def run_one_shot_final_convergence(
    project_root: Path,
    model: str = "qwen3:8b",
    max_runtime_repair_rounds: int = 3,
    max_scoring_retries: int = 3,
    llm_workers: int = 1,
    cpu_workers: int = 6,
    resume: bool = True,
    run_validation: bool = False,
) -> dict[str, Any]:
    root = project_root / "06_scoring" if project_root.name != "06_scoring" else project_root
    project_root = root.parent
    command_log: list[dict[str, Any]] = []
    status: dict[str, Any] = {"created_at": _now(), "model": model, "project_root": str(project_root), "scoring_root": str(root)}
    checkpoint = root / "PAPER_OUTPUT" / "00_status" / "one_shot_final_convergence_checkpoint.json"

    try:
        status["snapshot"] = snapshot_pre_final_convergence(root=root)
    except FileExistsError:
        status["snapshot"] = {"status": "already_exists"}
    _checkpoint(checkpoint, status)

    metrics_before = rebuild_mda_stability_metrics(root)
    status["mda_metrics_before"] = metrics_before
    runtime = repair_ollama_runtime(root=root, model=model, max_repair_rounds=max_runtime_repair_rounds)
    status["ollama_runtime"] = runtime
    _checkpoint(checkpoint, status)

    recovery = recover_missing_mda_dimensions(root=root, model=model, runtime_status=runtime, max_scoring_retries=max_scoring_retries, llm_workers=llm_workers)
    status["mda_recovery"] = recovery
    recovery_ready = recovery.get("status") == "success" and int(recovery.get("missing_dimensions_still_failed") or 0) == 0
    if recovery_ready:
        repeatability = run_mda_final_repeatability_test(root=root, model=model, runtime_status=runtime, llm_workers=llm_workers)
    else:
        repeatability = {
            "status": "blocked",
            "reason": "mda_recovery_not_complete",
            "expected_documents": metrics_before.get("expected_stability_documents", 0),
            "operationally_complete_documents": 0,
            "operational_completion_rate": 0.0,
            "valid_paired_documents": 0,
            "stable_paired_documents": 0,
            "repeatability_pass_rate": None,
        }
        _write_blocked_repeatability(root, repeatability)
    status["mda_repeatability"] = repeatability
    metrics_after = rebuild_mda_stability_metrics(root)
    status["mda_metrics_after"] = metrics_after
    if recovery_ready:
        mda_full = run_mda_final_full_scoring(root=root, model=model, runtime_status=runtime, metrics=metrics_after, llm_workers=llm_workers)
    else:
        mda_full = {
            "status": "blocked",
            "reason": "mda_recovery_not_complete",
            "eligible_mda_count": metrics_after.get("eligible_mda_count", 0),
            "complete_mda_count": 0,
            "incomplete_mda_count": metrics_after.get("eligible_mda_count", 0),
            "excluded_mda_count": 0,
            "full_scoring_coverage_rate": 0.0,
        }
        _write_blocked_mda_full(root, mda_full)
    status["mda_full_scoring"] = mda_full
    metrics_after = rebuild_mda_stability_metrics(root)
    status["mda_metrics_after"] = metrics_after
    _checkpoint(checkpoint, status)

    discovery = discover_real_news_across_project(project_root=project_root, root=root)
    status["real_news_discovery"] = discovery
    news_summary = _run_news_if_possible(root, model, runtime, discovery, llm_workers)
    status["news_final_scoring"] = news_summary
    cross = _run_cross_validation_if_possible(root, status, command_log)
    status["cross_validation"] = cross

    freeze = run_final_data_freeze(root)
    paper = run_paper_readiness_check(root)
    status["final_freeze"] = freeze
    status["paper_readiness"] = paper
    build_blocker_status_reports(root=root, ollama_status={"ollama_available": runtime.get("runtime_healthy", False), "model_available": runtime.get("runtime_healthy", False), "errors": runtime.get("blocking_reasons", [])})
    final_status = _compose_final_status(root, status)
    if run_validation:
        final_status["validation_commands"] = _run_validation_commands(root, command_log)
    _build_deliverable(root, final_status, status, command_log)
    _checkpoint(checkpoint, {**status, "final_status": final_status})
    return final_status


def _run_news_if_possible(root: Path, model: str, runtime: dict[str, Any], discovery: dict[str, Any], llm_workers: int) -> dict[str, Any]:
    if not discovery.get("real_news_available"):
        return {"status": "blocked", "reason": "real_news_missing", "news_final_dataset_size": 0}
    if not runtime.get("runtime_healthy"):
        return {"status": "blocked", "reason": "ollama_runtime_unhealthy", "news_final_dataset_size": 0}
    registry_summary = _build_news_registry_from_standardized(root)
    if registry_summary["real_news_count"] <= 0:
        return {"status": "blocked", "reason": "no_standardized_news_rows", "news_final_dataset_size": 0}
    pilot_limit = min(10, registry_summary["real_news_count"])
    pilot = run_single_dimension_scoring(
        "news",
        root=root,
        output_name="news_final_pilot",
        ratings_prefix="news_final_pilot",
        model=model,
        llm_workers=llm_workers,
        resume=True,
        overwrite=False,
        limit=pilot_limit,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
    )
    pilot_rate = _rate(pilot.get("success_count", 0), pilot.get("selected_count", 0))
    if pilot_rate < 0.95:
        return {"status": "blocked", "reason": "news_pilot_operational_gate_failed", "pilot": pilot, "news_final_dataset_size": 0}
    full = run_single_dimension_scoring(
        "news",
        root=root,
        output_name="news_final_full",
        ratings_prefix="news_final",
        model=model,
        llm_workers=llm_workers,
        resume=True,
        overwrite=False,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
    )
    return {"status": "success", "registry": registry_summary, "pilot": pilot, "full": full, "news_final_dataset_size": full.get("success_count", 0)}


def _write_blocked_repeatability(root: Path, summary: dict[str, Any]) -> None:
    write_markdown(
        root / "PAPER_OUTPUT" / "00_status" / "mda_final_repeatability_report.md",
        [
            "# MD&A Final Repeatability Report",
            "",
            f"- status: {summary.get('status')}",
            f"- reason: {summary.get('reason')}",
            f"- expected_documents: {summary.get('expected_documents')}",
            f"- repeatability_pass_rate: {summary.get('repeatability_pass_rate')}",
            "",
            "Repeatability was not run because missing-dimension recovery did not produce complete valid outputs.",
        ],
    )


def _write_blocked_mda_full(root: Path, summary: dict[str, Any]) -> None:
    write_markdown(
        root / "PAPER_OUTPUT" / "00_status" / "mda_final_full_scoring_report.md",
        [
            "# MD&A Final Full Scoring Report",
            "",
            f"- status: {summary.get('status')}",
            f"- reason: {summary.get('reason')}",
            f"- eligible_mda_count: {summary.get('eligible_mda_count')}",
            f"- complete_mda_count: {summary.get('complete_mda_count')}",
            f"- full_scoring_coverage_rate: {summary.get('full_scoring_coverage_rate')}",
            "",
            "Full scoring was not run because MD&A recovery/repeatability gates did not pass.",
        ],
    )


def _build_news_registry_from_standardized(root: Path) -> dict[str, Any]:
    source = root / "input_news" / "real_news_raw" / "real_news_raw.csv"
    if not source.exists():
        return {"status": "blocked", "real_news_count": 0}
    rows = []
    rating_text_dir = root / "02_extracted_text" / "news"
    rating_text_dir.mkdir(parents=True, exist_ok=True)
    with source.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader, start=1):
            title = _clean(row.get("title", ""))
            article = _clean(row.get("article_text", ""))
            if not title or not article:
                continue
            doc_id = _news_doc_id(row, idx)
            text_path = rating_text_dir / f"{doc_id}.txt"
            text_path.write_text(_paragraphize_news(article), encoding="utf-8")
            rows.append(
                {
                    "document_id": doc_id,
                    "doc_type": "news",
                    "include_flag": "Yes",
                    "company_name": _clean(row.get("company_name", "")),
                    "ticker": _clean(row.get("ticker", "")),
                    "source_name": _clean(row.get("source_name", "")),
                    "publish_date": _clean(row.get("publish_date", "")),
                    "title": title,
                    "url": _clean(row.get("url", "")),
                    "text_path": f"02_extracted_text/news/{doc_id}.txt",
                    "notes": f"synthetic_flag=false; original_file_path={row.get('original_file_path', '')}; original_file_hash={row.get('original_file_hash', '')}",
                }
            )
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], rows)
    return {"status": "success" if rows else "blocked", "real_news_count": len(rows), "registry_path": str(root / "01_registry" / "news_registry.csv")}


def _run_cross_validation_if_possible(root: Path, status: dict[str, Any], command_log: list[dict[str, Any]]) -> dict[str, Any]:
    news_size = int(status.get("news_final_scoring", {}).get("news_final_dataset_size") or 0)
    mda_size = int(status.get("mda_full_scoring", {}).get("complete_mda_count") or 0)
    if news_size <= 0 or mda_size <= 0:
        return {"cross_validation_pipeline_ready": True, "cross_validation_empirical_ready": False, "reason": "missing current final MDA or News dataset", "claim_link_count": 0}
    result = _run_cmd(["python", "06_scoring/scripts/run_cross_validation.py", "--mode", "full"], cwd=root.parent, timeout=600)
    command_log.append(result)
    return {"cross_validation_pipeline_ready": result["returncode"] == 0, "cross_validation_empirical_ready": result["returncode"] == 0, "claim_link_count": _claim_link_count(root), "command": result}


def _compose_final_status(root: Path, status: dict[str, Any]) -> dict[str, Any]:
    paper = status.get("paper_readiness", {})
    metrics = status.get("mda_metrics_after", {})
    discovery = status.get("real_news_discovery", {})
    news = status.get("news_final_scoring", {})
    cross = status.get("cross_validation", {})
    freeze = status.get("final_freeze", {})
    runtime_healthy = bool(
        status.get("ollama_runtime", {}).get("runtime_healthy")
        or paper.get("mda_ready")
        or (root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv").exists()
    )
    metric_mda_ready = bool(
        runtime_healthy
        and metrics.get("operational_completion_rate", 0) >= 0.95
        and (metrics.get("repeatability_pass_rate") is not None and metrics.get("repeatability_pass_rate") >= 0.85)
        and metrics.get("schema_success_rate", 0) >= 0.98
        and metrics.get("evidence_match_rate", 0) >= 0.98
        and metrics.get("full_scoring_coverage_rate", 0) >= 0.95
        and metrics.get("unresolved_critical_numeric_conflicts", 1) == 0
    )
    mda_ready = bool(paper.get("mda_ready", metric_mda_ready))
    news_size = int(news.get("news_final_dataset_size") or paper.get("news_final_dataset_size") or len(read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")))
    synthetic_news_count = int(paper.get("synthetic_news_count", 0) or 0)
    news_ready = bool(
        news_size > 0
        and synthetic_news_count == 0
        and (paper.get("news_ready") is True or news.get("status") == "success")
    )
    cross_empirical = bool(paper.get("cross_validation_ready") or (cross.get("cross_validation_empirical_ready") and mda_ready and news_ready))
    pipeline_operational_failure_rate = 1 - float(metrics.get("operational_completion_rate") or 1)
    final_dataset_validation_error_rate = 0.0 if freeze.get("system_level_error_rate") == 0 else float(freeze.get("system_level_error_rate", 0) or 0)
    blockers = []
    if not runtime_healthy:
        blockers.append("ollama_runtime_unhealthy")
    if not mda_ready:
        blockers.append("mda_not_ready")
    if not discovery.get("real_news_available") and not news_ready:
        blockers.append("real_news_missing")
    if not news_ready:
        blockers.append("news_not_ready")
    if not cross_empirical:
        blockers.append("cross_validation_not_empirically_ready")
    if pipeline_operational_failure_rate >= 0.05:
        blockers.append("pipeline_operational_failure_rate")
    if final_dataset_validation_error_rate != 0:
        blockers.append("final_dataset_validation_error_rate")
    freeze_allowed = bool(paper.get("freeze_allowed", False))
    paper_ready = bool(paper.get("paper_ready", False))
    claim_link_count = int(paper.get("claim_link_count") or cross.get("claim_link_count") or _claim_link_count(root) or 0)
    return {
        "ollama_runtime_healthy": runtime_healthy,
        "mda_expected_count": metrics.get("expected_stability_documents", 0),
        "mda_final_dataset_size": int(paper.get("mda_final_dataset_size", 0) or 0),
        "mda_full_scoring_coverage_rate": metrics.get("full_scoring_coverage_rate", 0.0),
        "mda_operational_completion_rate": metrics.get("operational_completion_rate", 0.0),
        "mda_repeatability_pass_rate": metrics.get("repeatability_pass_rate"),
        "mda_schema_success_rate": metrics.get("schema_success_rate", 0.0),
        "mda_evidence_match_rate": metrics.get("evidence_match_rate", 0.0),
        "news_real_files_found": discovery.get("real_news_candidate_count", len(read_csv_rows(root / "01_registry" / "news_registry.csv"))),
        "news_final_dataset_size": news_size,
        "news_operational_completion_rate": _rate(news.get("full", {}).get("success_count", 0), news.get("full", {}).get("selected_count", 0)),
        "news_repeatability_pass_rate": None,
        "synthetic_news_count": synthetic_news_count,
        "eligible_company_year_pair_count": int(paper.get("eligible_company_year_pair_count") or 0),
        "claim_link_count": claim_link_count,
        "cross_validation_pipeline_ready": bool(paper.get("cross_validation_pipeline_ready", cross.get("cross_validation_pipeline_ready", True))),
        "cross_validation_empirical_ready": cross_empirical,
        "pipeline_operational_failure_rate": round(pipeline_operational_failure_rate, 6),
        "final_dataset_validation_error_rate": final_dataset_validation_error_rate,
        "legacy_system_level_error_rate": paper.get("system_level_error_rate", 0.0),
        "mda_ready": mda_ready,
        "news_ready": news_ready,
        "cross_validation_ready": cross_empirical,
        "reproducibility_ready": bool(paper.get("reproducibility_ready", True)),
        "freeze_allowed": freeze_allowed,
        "paper_ready": paper_ready,
        "blocking_reasons": sorted(set(blockers + paper.get("blocking_reasons", []))),
        "recommended_next_actions": _recommended_actions(blockers),
    }


def _recommended_actions(blockers: list[str]) -> list[str]:
    actions = []
    if "ollama_runtime_unhealthy" in blockers:
        actions.append("Start or repair local Ollama, ensure qwen3:8b answers /api/chat JSON schema tests, then rerun one-shot convergence.")
    if "real_news_missing" in blockers or "news_not_ready" in blockers:
        actions.append("Provide real news_raw.csv or news_raw.xlsx with ticker, company_name, publish_date, source_name, title, url, article_text.")
    if "mda_not_ready" in blockers:
        actions.append("Inspect MD&A final recovery JSON/schema failures, repair prompt/schema/evidence validation issues, then rerun recovery; do not lower gates or delete failed samples.")
    if "cross_validation_not_empirically_ready" in blockers:
        actions.append("Review cross-validation contradiction/systemic matching outputs before using final empirical conclusions.")
    return actions or ["No action required."]


def _build_deliverable(root: Path, final_status: dict[str, Any], status: dict[str, Any], command_log: list[dict[str, Any]]) -> None:
    out = root / "FINAL_DELIVERABLE"
    if out.exists():
        shutil.rmtree(out)
    for sub in ["MDA", "NEWS", "CROSS_VALIDATION", "QUALITY", "REPRODUCIBILITY"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    save_json(out / "FINAL_STATUS.json", final_status)
    save_json(out / "one_shot_internal_status.json", status)
    _write_final_report(out / "FINAL_REPORT.md", final_status)
    _write_manifest(out / "FINAL_PACKAGE_MANIFEST.md", final_status)
    _copy_if_exists(root / "FINAL_OUTPUT" / "mda_final_dataset.csv", out / "MDA" / "mda_final_document_scores.csv")
    _copy_if_exists(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv", out / "MDA" / "mda_final_ratings_long.csv")
    _copy_if_exists(root / "06_ratings" / "mda_final_full" / "mda_final_failed_cases.csv", out / "MDA" / "mda_failed_or_excluded_cases.csv")
    _copy_if_exists(root / "PAPER_OUTPUT" / "00_status" / "mda_stability_metrics_redefined.md", out / "MDA" / "mda_stability_report.md")
    _copy_if_exists(root / "FINAL_OUTPUT" / "news_final_dataset.csv", out / "NEWS" / "news_final_document_scores.csv")
    _copy_if_exists(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", out / "NEWS" / "news_final_ratings_long.csv")
    _copy_if_exists(root / "06_ratings" / "news_final_full" / "news_final_failed_cases.csv", out / "NEWS" / "news_failed_or_excluded_cases.csv")
    _copy_if_exists(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", out / "NEWS" / "news_model_document_scores.csv")
    _copy_if_exists(root / "01_registry" / "news_registry.csv", out / "NEWS" / "news_registry.csv")
    _copy_if_exists(root / "08_reports" / "filter_news_final_scoring_report.md", out / "NEWS" / "news_stability_report.md")
    _copy_if_exists(root / "08_reports" / "news_qwen_runtime_config.md", out / "NEWS" / "news_qwen_runtime_config.md")
    _copy_if_exists(root / "08_reports" / "qwen3_8b_limitations_news.md", out / "NEWS" / "qwen3_8b_limitations_news.md")
    _copy_if_exists(root / "08_reports" / "news_possible_false_positives.csv", out / "NEWS" / "news_possible_false_positives.csv")
    _copy_if_exists(root / "08_reports" / "news_wrong_company_candidates.csv", out / "NEWS" / "news_wrong_company_candidates.csv")
    _copy_if_exists(root / "08_reports" / "news_special_case_recheck.csv", out / "NEWS" / "news_special_case_recheck.csv")
    _copy_if_exists(root / "08_reports" / "news_special_case_recheck.md", out / "NEWS" / "news_special_case_recheck.md")
    _copy_if_exists(root / "09_cross_validation" / "final_real_run" / "eligible_pairs.csv", out / "CROSS_VALIDATION" / "eligible_pairs.csv")
    _copy_if_exists(root / "09_cross_validation" / "final_real_run" / "claim_links.csv", out / "CROSS_VALIDATION" / "claim_links.csv")
    _copy_if_exists(root / "FINAL_OUTPUT" / "cross_validation_final_summary.csv", out / "CROSS_VALIDATION" / "company_year_cross_validation_summary.csv")
    _copy_if_exists(root / "PAPER_OUTPUT" / "00_status" / "cross_validation_blocker_diagnosis.md", out / "CROSS_VALIDATION" / "cross_validation_report.md")
    _copy_if_exists(root / "FINAL_OUTPUT" / "final_quality_report.md", out / "QUALITY" / "final_quality_report.md")
    _copy_if_exists(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json", out / "QUALITY" / "final_freeze_metadata.json")
    _copy_if_exists(root / "07_review" / "review_log.csv", out / "QUALITY" / "review_log.csv")
    _copy_if_exists(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv", out / "QUALITY" / "numeric_check_summary.csv")
    _write_operational_failure_report(out / "QUALITY" / "operational_failure_report.md", final_status)
    _copy_if_exists(root.parent / "requirements.txt", out / "REPRODUCIBILITY" / "requirements.txt")
    _copy_if_exists(root.parent / "run_reproduce.sh", out / "REPRODUCIBILITY" / "run_reproduce.sh")
    _write_environment_summary(out / "REPRODUCIBILITY" / "environment_summary.md")
    save_json(out / "REPRODUCIBILITY" / "command_log.json", command_log)
    (out / "REPRODUCIBILITY" / "command_log.md").write_text("\n".join(f"- {item.get('cmd')}: rc={item.get('returncode')}" for item in command_log), encoding="utf-8")
    _write_hash_manifest(out / "REPRODUCIBILITY" / "file_hash_manifest.csv", out)


def _write_final_report(path: Path, final_status: dict[str, Any]) -> None:
    lines = ["# Final Convergence Report", ""]
    for field in FINAL_FIELDS:
        lines.append(f"- {field}: {final_status.get(field)}")
    if not final_status.get("paper_ready"):
        lines.extend(["", "## Blocking Reasons"])
        lines.extend(f"- {item}" for item in final_status.get("blocking_reasons", []))
    write_markdown(path, lines)


def _write_manifest(path: Path, final_status: dict[str, Any]) -> None:
    write_markdown(path, ["# Final Package Manifest", "", f"- paper_ready: {str(final_status.get('paper_ready')).lower()}", f"- freeze_allowed: {str(final_status.get('freeze_allowed')).lower()}", "- package_type: blocker-aware final deliverable"])


def _write_operational_failure_report(path: Path, status: dict[str, Any]) -> None:
    write_markdown(path, ["# Operational Failure Report", "", f"- pipeline_operational_failure_rate: {status.get('pipeline_operational_failure_rate')}", f"- final_dataset_validation_error_rate: {status.get('final_dataset_validation_error_rate')}", f"- blockers: {', '.join(status.get('blocking_reasons', []))}"])


def _write_environment_summary(path: Path) -> None:
    result = _run_cmd(["python", "--version"], cwd=path.parent, timeout=10)
    write_markdown(path, ["# Environment Summary", "", f"- python: {result.get('stdout') or result.get('stderr')}"])


def _write_hash_manifest(path: Path, base: Path) -> None:
    rows = []
    for file in sorted(item for item in base.rglob("*") if item.is_file()):
        if file == path:
            continue
        rows.append({"relative_path": str(file.relative_to(base)), "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "size_bytes": file.stat().st_size})
    write_csv_rows(path, ["relative_path", "sha256", "size_bytes"], rows)


def _run_validation_commands(root: Path, command_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands = [
        ["python", "-m", "pytest", "06_scoring/tests", "-q"],
        ["python", "06_scoring/scripts/self_check_pipeline.py", "--max-rounds", "5"],
        ["python", "06_scoring/scripts/validate_stability_outputs.py"],
        ["python", "06_scoring/scripts/run_final_data_freeze.py"],
        ["python", "06_scoring/scripts/run_paper_readiness_check.py"],
    ]
    results = []
    for cmd in commands:
        result = _run_cmd(cmd, cwd=root.parent, timeout=600)
        command_log.append(result)
        results.append({"cmd": " ".join(cmd), "returncode": result["returncode"]})
    return results


def _copy_if_exists(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        denominator_f = float(denominator)
        return round(float(numerator) / denominator_f, 6) if denominator_f else 0.0
    except (TypeError, ValueError):
        return 0.0


def _claim_link_count(root: Path) -> int:
    paths = [root / "09_cross_validation" / "final_real_run" / "claim_links.csv", root / "09_cross_validation" / "claim_links.csv"]
    for path in paths:
        if path.exists():
            return max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)
    return 0


def _paragraphize_news(article: str) -> str:
    paragraphs = [part.strip() for part in article.replace("\r", "\n").split("\n") if part.strip()]
    if not paragraphs:
        paragraphs = [article]
    return "\n\n".join(f"[NEWS_P{idx:03d}] {paragraph}" for idx, paragraph in enumerate(paragraphs, start=1))


def _news_doc_id(row: dict[str, str], idx: int) -> str:
    key = "|".join([row.get("ticker", ""), row.get("company_name", ""), row.get("publish_date", ""), row.get("title", ""), str(idx)])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8].upper()
    ticker = "".join(ch for ch in row.get("ticker", "NEWS").upper() if ch.isalnum()) or "NEWS"
    date = "".join(ch for ch in row.get("publish_date", "") if ch.isdigit())[:8] or f"{idx:08d}"
    return f"NEWS_{ticker}_{date}_{digest}"


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def _run_cmd(cmd: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        return {"cmd": " ".join(cmd), "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        return {"cmd": " ".join(cmd), "returncode": None, "stdout": exc.stdout or "", "stderr": "timeout"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-shot final convergence with honest blockers.")
    parser.add_argument("--project-root", type=Path, default=SCORING_ROOT.parent)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--max-runtime-repair-rounds", type=int, default=3)
    parser.add_argument("--max-scoring-retries", type=int, default=3)
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int, default=6)
    parser.add_argument("--resume", type=parse_bool, default=True)
    args = parser.parse_args()
    status = run_one_shot_final_convergence(args.project_root, args.model, args.max_runtime_repair_rounds, args.max_scoring_retries, args.llm_workers, args.cpu_workers, args.resume, run_validation=True)
    for field in FINAL_FIELDS:
        print(f"{field}: {status.get(field)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
