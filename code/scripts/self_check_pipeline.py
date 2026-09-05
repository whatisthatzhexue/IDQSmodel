from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from check_ollama import run_check
from benchmark_parallel_scoring import run_benchmark
from run_scoring import run_pipeline
from scoring_utils import (
    MDA_DOCUMENT_HEADERS,
    NEWS_DOCUMENT_HEADERS,
    REGISTRY_HEADERS,
    SCORING_ROOT,
    append_jsonl,
    ensure_directories,
    read_csv_rows,
    save_json,
    weight_sum_ok,
    write_csv_rows,
    write_default_registries,
)


def run_self_check(max_rounds: int = 10, root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_directories(root)
    write_default_registries(root)
    log_path = root / "_self_correction" / "self_correction_log.md"
    failures: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    if log_path.exists():
        log_path.unlink()

    checks = [
        ("Round 1", "static checks", lambda: _round_static(root)),
        ("Round 2", "unit tests", lambda: _round_pytest(root)),
        ("Round stability-1", "mock MD&A weight sensitivity", lambda: _round_stability_weight_mda(root)),
        ("Round stability-2", "mock News weight sensitivity", lambda: _round_stability_weight_news(root)),
        ("Round stability-3", "Pearson and Spearman correctness", lambda: _round_stability_correlations(root)),
        ("Round stability-4", "high-volatility case marking", lambda: _round_stability_high_volatility(root)),
        ("Round stability-5", "missing News data handling", lambda: _round_stability_missing_news(root)),
        ("Round stability-6", "statistical stability report generation", lambda: _round_stability_report(root)),
        ("Round stability-7", "statistical stability output validation", lambda: _round_stability_validation(root)),
        ("Round MDA-v2", "MD&A v2 cleaning audit and sample rescore smoke test", lambda: _round_mda_v2_module(root)),
        ("Round cross-validation", "MD&A news bidirectional cross-validation smoke test", lambda: _round_cross_validation(root)),
        ("Round stabilization", "MD&A v2 pre-full-rescore stabilization checks", lambda: _round_mda_v2_stabilization(root)),
        ("Round claim-1", "claim extraction correctness", lambda: _round_claim_extraction(root)),
        ("Round claim-2", "claim decomposition completeness", lambda: _round_claim_decomposition(root)),
        ("Round claim-3", "claim mapping correctness", lambda: _round_claim_mapping(root)),
        ("Round claim-4", "claim aggregation correctness", lambda: _round_claim_aggregation(root)),
        ("Round claim-5", "compare claim-level vs document-level scoring", lambda: _round_claim_comparison(root)),
        ("Round single-dimension", "MD&A/news single-dimension convergence smoke test", lambda: _round_single_dimension_convergence(root)),
        ("Round weight-sensitivity", "weight calibration and sensitivity analysis smoke test", lambda: _round_weight_sensitivity(root)),
        ("Round parallel", "mock parallel scoring llm_workers=2", lambda: _round_parallel_mock(root, 2)),
        ("Round parallel-resume", "resume overwrite and qwen benchmark checks", lambda: _round_parallel_resume_overwrite_qwen(root)),
    ][:max_rounds]

    for idx, (round_name, title, func) in enumerate(checks, start=1):
        try:
            result = func()
            passed = bool(result.get("passed"))
        except Exception as exc:  # noqa: BLE001 - self-check must log failures.
            result = {"passed": False, "error_summary": str(exc), "command": title, "files_changed": []}
            passed = False
        entry = {
            "round_number": idx,
            "round_name": round_name,
            "title": title,
            "command_run": result.get("command", title),
            "pass": passed,
            "error_summary": result.get("error_summary", ""),
            "files_changed": result.get("files_changed", []),
            "fix_applied": result.get("fix_applied", "none"),
            "next_action": result.get("next_action", "continue" if passed else "manual repair required"),
        }
        rounds.append(entry)
        _append_log(log_path, entry)
        append_jsonl(root / "_self_correction" / "fix_history.jsonl", entry)
        if not passed:
            failures.append(entry)
            break

    save_json(root / "_self_correction" / "validation_failures.json", failures)
    return {"rounds_run": len(rounds), "passed": len(failures) == 0, "failures": failures}


def _round_static(root: Path) -> dict[str, Any]:
    ensure_directories(root)
    modules = [
        "scoring_utils",
        "llm_clients",
        "mda_numeric_checker",
        "validate_model_scores",
        "run_scoring",
        "check_ollama",
        "repair_ollama_runtime",
        "freeze_mda_initial_scoring",
        "clean_mda_texts_v2",
        "audit_mda_clean_texts",
        "audit_mda_initial_scores",
        "select_mda_v2_rescore_sample",
        "run_mda_v2_sample_rescore",
        "run_mda_v2_full_rescore",
        "compare_mda_v1_v2_scores",
        "build_mda_rescore_decision_report",
        "run_cross_validation",
        "diagnose_mda_v2_sample_failures",
        "audit_ar02_numeric_discrepancies",
        "run_mda_v2_stability_batch",
        "snapshot_pre_final_convergence",
        "rebuild_mda_stability_metrics",
        "recover_missing_mda_dimensions",
        "run_mda_final_repeatability_test",
        "run_mda_final_full_scoring",
        "run_mda_v2_full_rescore",
        "json_stabilization",
        "diagnose_qwen_json_failures",
        "repair_existing_failed_outputs",
        "run_mda_dimensionwise_scoring",
        "aggregate_dimensionwise_mda_scores",
        "validate_dimensionwise_outputs",
        "claim_utils",
        "extract_mda_claims",
        "extract_news_claims",
        "run_claim_scoring",
        "aggregate_claim_scores",
        "compare_claim_vs_document_scoring",
        "generate_news_missing_next_steps",
        "run_single_dimension_scoring",
        "validate_json_outputs",
        "repair_json_pipeline",
        "scoring_stability_monitor",
        "run_weight_sensitivity",
        "stability_common",
        "stability_weight_sensitivity",
        "stability_repeatability",
        "stability_version_comparison",
        "stability_structure_comparison",
        "stability_news_aggregation",
        "build_stability_report",
        "validate_stability_outputs",
        "run_statistical_stability_analysis",
        "run_final_data_freeze",
        "news_pipeline_status_check",
        "check_real_news_availability",
        "discover_real_news_across_project",
        "build_news_minimal_pipeline",
        "build_news_registry",
        "extract_news_texts",
        "diagnose_mda_stability_failures",
        "diagnose_mda_stability_blocker",
        "repair_mda_stability_failures",
        "build_blocker_status_reports",
        "run_final_convergence_fix",
        "paper_utils",
        "run_paper_readiness_check",
        "run_paper_convergence_loop",
        "build_paper_methods_section",
        "build_paper_data_section",
        "build_paper_scoring_framework",
        "build_paper_results_tables",
        "build_paper_stability_and_qc",
        "build_paper_cross_validation_section",
        "build_paper_limitations",
        "build_paper_reproducibility_package",
        "build_paper_ready_report",
        "run_paper_ready_pipeline",
        "run_overnight_final_delivery",
        "run_one_shot_final_convergence",
    ]
    for module in modules:
        importlib.import_module(module)
    json.loads((root / "04_prompts" / "scoring_output_schema.json").read_text(encoding="utf-8"))
    for doc_type in ["mda", "news"]:
        if not weight_sum_ok(doc_type):
            return {"passed": False, "error_summary": f"{doc_type} weights do not sum to 1", "command": "static checks"}
        scorebook_path = root / "00_scorebook" / f"{doc_type}_scorebook.yaml"
        _validate_scorebook_text(scorebook_path, doc_type)
    return {
        "passed": True,
        "command": "import modules; parse JSON schema; validate scorebook weights",
        "files_changed": [],
        "next_action": "run unit tests",
    }


def claim_round_names() -> list[str]:
    return ["Round claim-1", "Round claim-2", "Round claim-3", "Round claim-4", "Round claim-5"]


def stability_round_names() -> list[str]:
    return [
        "Round stability-1",
        "Round stability-2",
        "Round stability-3",
        "Round stability-4",
        "Round stability-5",
        "Round stability-6",
        "Round stability-7",
    ]


def _round_pytest(root: Path) -> dict[str, Any]:
    repo_root = root.parent
    command = [sys.executable, "-m", "pytest", "06_scoring/tests", "-q"]
    completed = subprocess.run(command, cwd=repo_root, text=True, capture_output=True)
    output_path = root / "_self_correction" / "test_runs" / "round2_pytest.txt"
    output_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    return {
        "passed": completed.returncode == 0,
        "command": " ".join(command),
        "error_summary": "" if completed.returncode == 0 else (completed.stdout + completed.stderr)[-1000:],
        "files_changed": [str(output_path)],
        "next_action": "run synthetic smoke test",
    }


def _round_synthetic(root: Path) -> dict[str, Any]:
    mda_rows, news_rows, changed = _write_synthetic_inputs(root)
    mda_summary = run_pipeline("mda", "pilot", root=root, registry_rows_override=mda_rows, mock_mode=True, write_report=True)
    news_summary = run_pipeline("news", "pilot", root=root, registry_rows_override=news_rows, mock_mode=True, write_report=True)
    passed = mda_summary["success_count"] == 2 and news_summary["success_count"] == 2
    return {
        "passed": passed,
        "command": "build synthetic registry -> numeric checker -> mock scoring -> aggregate -> review_log -> summary report",
        "error_summary": "" if passed else f"mda={mda_summary}; news={news_summary}",
        "files_changed": changed
        + [
            str(root / "03_numeric_checks" / "mda_numeric_checks_summary.csv"),
            str(root / "06_ratings" / "mda_ratings_long.csv"),
            str(root / "06_ratings" / "news_ratings_long.csv"),
            str(root / "07_review" / "review_log.csv"),
            str(root / "08_reports" / "scoring_summary_report.md"),
        ],
        "next_action": "run local qwen JSON test if available",
    }


def _round_parallel_mock(root: Path, workers: int) -> dict[str, Any]:
    from parallel_scoring import run_parallel_pipeline

    tmp_root = _build_parallel_tmp_root(root)
    summary = run_parallel_pipeline("mda", "pilot", root=tmp_root, limit=2, llm_workers=workers, mock_mode=True, overwrite=True, resume=False)
    passed = summary["success_count"] == 2 and summary["failure_count"] == 0
    return {
        "passed": passed,
        "command": f"parallel mock scoring --llm-workers {workers}",
        "error_summary": "" if passed else str(summary),
        "files_changed": [
            str(tmp_root / "06_ratings" / "per_document" / "mda"),
            str(tmp_root / "08_reports" / "scoring_progress_log.jsonl"),
        ],
        "next_action": "run next parallel check",
    }


def _round_mda_v2_module(root: Path) -> dict[str, Any]:
    from audit_mda_clean_texts import audit_mda_clean_texts
    from audit_mda_initial_scores import audit_mda_initial_scores
    from build_mda_rescore_decision_report import build_mda_rescore_decision_report
    from clean_mda_texts_v2 import clean_mda_texts_v2
    from compare_mda_v1_v2_scores import compare_mda_v1_v2_scores
    from freeze_mda_initial_scoring import freeze_mda_initial_scoring
    from mda_numeric_checker import run_numeric_checks
    from run_mda_v2_sample_rescore import run_mda_v2_sample_rescore
    from select_mda_v2_rescore_sample import select_mda_v2_rescore_sample

    tmp_root = _build_mda_v2_tmp_root(root)
    v1_summary = run_pipeline("mda", "full", root=tmp_root, mock_mode=True, write_report=True)
    freeze_mda_initial_scoring(tmp_root)
    clean_mda_texts_v2(tmp_root)
    audit_mda_clean_texts(tmp_root)
    run_numeric_checks(
        tmp_root,
        input_dir=tmp_root / "02_extracted_text" / "mda_clean_v2",
        output_dir=tmp_root / "03_numeric_checks" / "mda_v2",
    )
    audit_mda_initial_scores(tmp_root)
    sample_summary = select_mda_v2_rescore_sample(tmp_root)
    v2_summary = run_mda_v2_sample_rescore(tmp_root, mock=True)
    comparison = compare_mda_v1_v2_scores(tmp_root)
    build_mda_rescore_decision_report(tmp_root)

    required = [
        tmp_root / "archive" / "v1_initial_mda_scoring" / "v1_metadata.json",
        tmp_root / "02_extracted_text" / "mda_clean_v2" / "MDA_V2_GOOD_2024.txt",
        tmp_root / "02_extracted_text" / "mda_clean_v2_mapping" / "MDA_V2_GOOD_2024_mapping.csv",
        tmp_root / "08_reports" / "mda_cleaning_report.md",
        tmp_root / "08_reports" / "mda_text_quality_audit.csv",
        tmp_root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv",
        tmp_root / "06_ratings" / "mda_v1_audit" / "mda_initial_score_audit.csv",
        tmp_root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_rescore_sample.csv",
        tmp_root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_document_scores_sample.csv",
        tmp_root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v1_v2_score_comparison.csv",
        tmp_root / "08_reports" / "mda_v1_v2_comparison_report.md",
        tmp_root / "08_reports" / "mda_rescore_decision_report.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    passed = (
        not missing
        and v1_summary["success_count"] == 2
        and sample_summary["selected_documents"] >= 1
        and v2_summary["success_count"] >= 1
        and comparison["compared_documents"] >= 1
    )
    return {
        "passed": passed,
        "command": "freeze -> clean v2 -> quality audit -> numeric v2 -> v1 audit -> sample mock rescore -> compare -> decision report",
        "error_summary": "" if passed else f"missing={missing}; v1={v1_summary}; sample={sample_summary}; v2={v2_summary}; comparison={comparison}",
        "files_changed": [str(path) for path in required if path.exists()],
        "next_action": "run parallel scoring checks",
    }


def _round_cross_validation(root: Path) -> dict[str, Any]:
    from run_cross_validation import run_cross_validation

    tmp_root = _build_cross_validation_tmp_root(root)
    summary = run_cross_validation("full", root=tmp_root, mock_mode=True)
    required = [
        tmp_root / "09_cross_validation" / "mda_claims.csv",
        tmp_root / "09_cross_validation" / "news_claims.csv",
        tmp_root / "09_cross_validation" / "claim_links.csv",
        tmp_root / "09_cross_validation" / "mda_news_cross_validation_pairs.csv",
        tmp_root / "09_cross_validation" / "company_year_cross_validation_summary.csv",
        tmp_root / "09_cross_validation" / "cross_validation_report.md",
    ]
    links = read_csv_rows(tmp_root / "09_cross_validation" / "claim_links.csv")
    by_news = {row.get("news_document_id", ""): row for row in links}
    expected = {
        "NEWS_CV_RAW_2024": {"contextual_support", "strong_support"},
        "NEWS_CV_GROWTH_2025": {"qualification", "possible_overstatement"},
        "NEWS_CV_SAFETY_2024": {"possible_omission"},
        "NEWS_CV_AR_2024": {"same_source_repetition"},
        "NEWS_CV_REV_2024": {"contradiction"},
    }
    relation_failures = [
        f"{doc_id}: got {by_news.get(doc_id, {}).get('relation_type')}"
        for doc_id, allowed in expected.items()
        if by_news.get(doc_id, {}).get("relation_type") not in allowed
    ]
    missing = [str(path) for path in required if not path.exists()]
    review_rows = read_csv_rows(tmp_root / "07_review" / "review_log.csv")
    review_reasons = {row.get("review_reason", "") for row in review_rows}
    required_reviews = {"cross_validation_contradiction", "cross_validation_possible_omission", "cross_validation_news_overstatement"}
    missing_reviews = sorted(required_reviews - review_reasons)
    passed = not missing and not relation_failures and not missing_reviews and summary["claim_link_count"] >= 5
    return {
        "passed": passed,
        "command": "mock claim extraction -> bidirectional link classification -> pair/company-year summaries",
        "error_summary": ""
        if passed
        else f"summary={summary}; missing={missing}; relation_failures={relation_failures}; missing_reviews={missing_reviews}",
        "files_changed": [str(path) for path in required if path.exists()],
        "fix_applied": "none",
        "next_action": "run parallel scoring checks",
    }


def _round_mda_v2_stabilization(root: Path) -> dict[str, Any]:
    from audit_ar02_numeric_discrepancies import audit_ar02_numeric_discrepancies
    from diagnose_mda_v2_sample_failures import diagnose_mda_v2_sample_failures
    from diagnose_qwen_json_failures import diagnose_qwen_json_failures
    from generate_news_missing_next_steps import generate_news_missing_next_steps
    from mda_numeric_checker import run_numeric_checks
    from repair_existing_failed_outputs import repair_existing_failed_outputs
    from run_mda_v2_full_rescore import run_mda_v2_full_rescore
    from run_mda_v2_stability_batch import run_mda_v2_stability_batch
    from validate_dimensionwise_outputs import validate_dimensionwise_outputs

    tmp_root = _build_stabilization_tmp_root(root)
    failure_summary = diagnose_mda_v2_sample_failures(tmp_root)
    qwen_diagnosis = diagnose_qwen_json_failures(tmp_root)
    repair_summary = repair_existing_failed_outputs(tmp_root)
    audit_summary = audit_ar02_numeric_discrepancies(tmp_root, sample_size=30)
    run_numeric_checks(
        tmp_root,
        input_dir=tmp_root / "02_extracted_text" / "mda_clean_v2",
        output_dir=tmp_root / "03_numeric_checks" / "mda_v2",
    )
    stability_summary = run_mda_v2_stability_batch(tmp_root, mock=True, llm_workers=1, cpu_workers=2, overwrite=True)
    full_summary = run_mda_v2_full_rescore(tmp_root, mock=True, llm_workers=1, cpu_workers=2, limit=1, overwrite=True)
    stability_validation = validate_dimensionwise_outputs(tmp_root, output_name="mda_v2_stability_batch", ratings_prefix="mda_v2_stability")
    news_report = generate_news_missing_next_steps(tmp_root)
    required = [
        tmp_root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_failure_cases.csv",
        tmp_root / "08_reports" / "mda_v2_sample_failure_diagnosis.md",
        tmp_root / "08_reports" / "qwen_json_failure_diagnosis.csv",
        tmp_root / "08_reports" / "qwen_json_failure_diagnosis.md",
        tmp_root / "08_reports" / "qwen_json_repair_attempts.csv",
        tmp_root / "03_numeric_checks" / "mda_v2_numeric_discrepancy_audit.csv",
        tmp_root / "08_reports" / "ar02_numeric_discrepancy_audit_report.md",
        tmp_root / "06_ratings" / "mda_v2_stability_batch",
        tmp_root / "08_reports" / "mda_v2_stability_batch_report.md",
        tmp_root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_validation_summary.json",
        tmp_root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv",
        tmp_root / "06_ratings" / "mda_v2_full" / "mda_v2_full_failed_cases.csv",
        tmp_root / "06_ratings" / "mda_document_scores_final_candidate.csv",
        news_report,
    ]
    missing = [str(path) for path in required if not path.exists()]
    original_scores = read_csv_rows(tmp_root / "06_ratings" / "mda_document_scores.csv")
    candidate_rows = read_csv_rows(tmp_root / "06_ratings" / "mda_document_scores_final_candidate.csv")
    passed = (
        not missing
        and failure_summary["failure_count"] >= 1
        and qwen_diagnosis["failure_count"] >= 1
        and repair_summary["parser_repaired_count"] >= 1
        and audit_summary["sample_count"] >= 1
        and full_summary["success_count"] >= 1
        and stability_summary.get("scoring_strategy") == "single_dimension"
        and full_summary.get("scoring_strategy") == "single_dimension"
        and stability_validation["is_valid"]
        and len(original_scores) == 1
        and candidate_rows
        and candidate_rows[0].get("score_version") == "v2_full"
        and stability_summary["recommendation_for_full_rescore"] in {"recommend_full_rescore", "continue_fixing_failures_do_not_full_rescore"}
    )
    return {
        "passed": passed,
        "command": "diagnose failures -> audit AR02 numeric discrepancies -> mock stability batch -> mock isolated full rescore",
        "error_summary": ""
        if passed
        else f"missing={missing}; failure={failure_summary}; qwen={qwen_diagnosis}; repair={repair_summary}; audit={audit_summary}; stability={stability_summary}; validation={stability_validation}; full={full_summary}",
        "files_changed": [str(path) for path in required if path.exists()],
        "fix_applied": "none",
        "next_action": "run parallel scoring checks",
    }


def _round_claim_extraction(root: Path) -> dict[str, Any]:
    from extract_mda_claims import extract_mda_claims
    from extract_news_claims import extract_news_claims

    tmp_root = _build_claim_tmp_root(root)
    mda_summary = extract_mda_claims(tmp_root)
    news_summary = extract_news_claims(tmp_root)
    required = [
        tmp_root / "10_claims" / "mda_claims" / "MDA_CLAIM_SELF_2024.json",
        tmp_root / "10_claims" / "news_claims" / "NEWS_CLAIM_SELF_2024.json",
        tmp_root / "10_claim_extraction" / "mda_claim_extraction_summary.json",
        tmp_root / "10_claim_extraction" / "news_claim_extraction_summary.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    passed = not missing and mda_summary["claim_count"] >= 5 and news_summary["claim_count"] >= 3
    return {
        "passed": passed,
        "command": "extract_mda_claims + extract_news_claims on synthetic claim fixture",
        "error_summary": "" if passed else f"missing={missing}; mda={mda_summary}; news={news_summary}",
        "files_changed": [str(path) for path in required if path.exists()],
        "fix_applied": "none",
        "next_action": "check claim decomposition completeness",
    }


def _round_claim_decomposition(root: Path) -> dict[str, Any]:
    from extract_mda_claims import extract_mda_claims

    tmp_root = _build_claim_tmp_root(root)
    extract_mda_claims(tmp_root)
    payload = json.loads((tmp_root / "10_claims" / "mda_claims" / "MDA_CLAIM_SELF_2024.json").read_text(encoding="utf-8"))
    claims = payload.get("claims", [])
    texts = [claim.get("claim_text", "").lower() for claim in claims]
    types = {claim.get("claim_type") for claim in claims}
    passed = (
        any("460" in text and "500" in text for text in texts)
        and any("export demand" in text for text in texts)
        and any("domestic sales" in text and claim.get("direction") == "decrease" for claim, text in zip(claims, texts))
        and {"financial_claim", "causal_claim", "cost_claim", "outlook_claim"}.issubset(types)
    )
    return {
        "passed": passed,
        "command": "verify one numeric change, each cause, cost, and outlook become separate claims",
        "error_summary": "" if passed else f"claims={claims}",
        "files_changed": [str(tmp_root / "10_claims" / "mda_claims" / "MDA_CLAIM_SELF_2024.json")],
        "fix_applied": "none",
        "next_action": "check claim to dimension mapping",
    }


def _round_claim_mapping(root: Path) -> dict[str, Any]:
    from claim_utils import map_claim_to_dimensions
    from extract_mda_claims import extract_mda_claims
    from run_claim_scoring import run_claim_scoring

    tmp_root = _build_claim_tmp_root(root)
    extract_mda_claims(tmp_root)
    scoring_summary = run_claim_scoring(tmp_root, source_type="mda", mock=True)
    payload = json.loads((tmp_root / "10_claims" / "mda_claims" / "MDA_CLAIM_SELF_2024.json").read_text(encoding="utf-8"))
    mapping = {claim["claim_type"]: map_claim_to_dimensions(claim) for claim in payload.get("claims", [])}
    passed = (
        "AR02" in mapping.get("financial_claim", [])
        and "AR03" in mapping.get("causal_claim", [])
        and "AR04" in mapping.get("outlook_claim", [])
        and scoring_summary["success_count"] == scoring_summary["claim_count"]
    )
    return {
        "passed": passed,
        "command": "map extracted claims to AR dimensions and mock score each claim",
        "error_summary": "" if passed else f"mapping={mapping}; scoring={scoring_summary}",
        "files_changed": [str(tmp_root / "10_claim_scoring" / "mda_claim_scores" / "MDA_CLAIM_SELF_2024.json")],
        "fix_applied": "none",
        "next_action": "check claim aggregation",
    }


def _round_claim_aggregation(root: Path) -> dict[str, Any]:
    from aggregate_claim_scores import aggregate_claim_scores
    from extract_mda_claims import extract_mda_claims
    from run_claim_scoring import run_claim_scoring

    tmp_root = _build_claim_tmp_root(root)
    extract_mda_claims(tmp_root)
    run_claim_scoring(tmp_root, source_type="mda", mock=True)
    summary = aggregate_claim_scores(tmp_root, source_type="mda")
    rows = read_csv_rows(tmp_root / "10_claim_mapping" / "mda_document_scores_claim_level.csv")
    passed = summary["document_count"] == 1 and rows and all(rows[0].get(code) for code in ["AR01", "AR02", "AR03", "AR04", "AR05"])
    return {
        "passed": passed,
        "command": "aggregate mda claim scores into weighted AR document scores",
        "error_summary": "" if passed else f"summary={summary}; rows={rows}",
        "files_changed": [str(tmp_root / "10_claim_mapping" / "mda_document_scores_claim_level.csv")],
        "fix_applied": "none",
        "next_action": "compare claim-level against document-level",
    }


def _round_claim_comparison(root: Path) -> dict[str, Any]:
    from aggregate_claim_scores import aggregate_claim_scores
    from compare_claim_vs_document_scoring import compare_claim_vs_document_scoring
    from extract_mda_claims import extract_mda_claims
    from run_claim_scoring import run_claim_scoring

    tmp_root = _build_claim_tmp_root(root)
    extract_mda_claims(tmp_root)
    run_claim_scoring(tmp_root, source_type="mda", mock=True)
    aggregate_claim_scores(tmp_root, source_type="mda")
    summary = compare_claim_vs_document_scoring(tmp_root)
    report = tmp_root / "08_reports" / "claim_vs_document_comparison.md"
    passed = report.exists() and "claim-level" in summary["recommended_primary_system"]
    return {
        "passed": passed,
        "command": "compare claim-level and document-level metrics",
        "error_summary": "" if passed else f"summary={summary}",
        "files_changed": [str(report)] if report.exists() else [],
        "fix_applied": "none",
        "next_action": "run parallel scoring checks",
    }


def _round_single_dimension_convergence(root: Path) -> dict[str, Any]:
    from run_single_dimension_scoring import run_single_dimension_scoring
    from scoring_stability_monitor import monitor_scoring_stability

    tmp_root = root / "_self_correction" / "single_dimension_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ensure_directories(tmp_root)
    mda_dir = tmp_root / "02_extracted_text" / "mda"
    news_dir = tmp_root / "02_extracted_text" / "news"
    mda_dir.mkdir(parents=True, exist_ok=True)
    news_dir.mkdir(parents=True, exist_ok=True)
    (mda_dir / "MDA_SINGLE_SELF_2024.txt").write_text(
        "[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.\n\n"
        "[MDA_P002]\nManagement explains demand, costs, risk controls, and cautious outlook.",
        encoding="utf-8",
    )
    (news_dir / "NEWS_SINGLE_SELF_2024.txt").write_text(
        "[TITLE]\nSingle Food reports better sales\n\n"
        "[NEWS_P001]\nAccording to the company statement, sales improved as demand increased.",
        encoding="utf-8",
    )
    write_csv_rows(
        tmp_root / "01_registry" / "mda_registry.csv",
        REGISTRY_HEADERS["mda"],
        [
            {
                "document_id": "MDA_SINGLE_SELF_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "7777",
                "ticker": "SINGLE",
                "company_name": "Single Food Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda/MDA_SINGLE_SELF_2024.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        tmp_root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_SINGLE_SELF_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Single Food Berhad",
                "ticker": "SINGLE",
                "source_name": "Synthetic Wire",
                "publish_date": "2024-06-16",
                "title": "Single Food reports better sales",
                "url": "",
                "text_path": "02_extracted_text/news/NEWS_SINGLE_SELF_2024.txt",
                "notes": "",
            }
        ],
    )
    mda_summary = run_single_dimension_scoring("mda", root=tmp_root, output_name="mda_single_self", ratings_prefix="mda_single_self", mock=True, overwrite=True)
    news_summary = run_single_dimension_scoring("news", root=tmp_root, output_name="news_single_self", ratings_prefix="news_single_self", mock=True, overwrite=True)
    mda_monitor = monitor_scoring_stability(tmp_root, "mda_single_self")
    news_monitor = monitor_scoring_stability(tmp_root, "news_single_self")
    required = [
        tmp_root / "06_ratings" / "mda_single_self" / "mda_single_self_document_scores.csv",
        tmp_root / "06_ratings" / "news_single_self" / "news_single_self_document_scores.csv",
        tmp_root / "08_reports" / "mda_single_self_stability_monitor.json",
        tmp_root / "08_reports" / "news_single_self_stability_monitor.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    passed = (
        not missing
        and mda_summary["dimension_task_count"] == 5
        and news_summary["dimension_task_count"] == 5
        and mda_monitor["passes_thresholds"]
        and news_monitor["passes_thresholds"]
    )
    return {
        "passed": passed,
        "command": "run mock single-dimension scoring for mda and news; monitor stability",
        "error_summary": "" if passed else f"missing={missing}; mda={mda_summary}; news={news_summary}; monitors={mda_monitor},{news_monitor}",
        "files_changed": [str(path) for path in required if path.exists()],
        "fix_applied": "none",
        "next_action": "run weight sensitivity checks",
    }


def _round_stability_weight_mda(root: Path) -> dict[str, Any]:
    from stability_weight_sensitivity import run_weight_stability

    tmp_root = _build_stability_tmp_root(root, include_news=False, include_optional=False)
    summary = run_weight_stability(root=tmp_root, include_mda=True, include_news=False, make_plots=False)
    output = tmp_root / "11_stability_analysis" / "outputs" / "mda_weight_sensitivity_scores.csv"
    passed = summary["mda"]["status"] == "success" and output.exists()
    return {
        "passed": passed,
        "command": "run mock MD&A weight sensitivity under 11_stability_analysis",
        "error_summary": "" if passed else str(summary),
        "files_changed": [str(output)] if output.exists() else [],
        "next_action": "run News weight sensitivity",
    }


def _round_stability_weight_news(root: Path) -> dict[str, Any]:
    from stability_weight_sensitivity import run_weight_stability

    tmp_root = _build_stability_tmp_root(root, include_news=True, include_optional=False)
    summary = run_weight_stability(root=tmp_root, include_mda=False, include_news=True, make_plots=False)
    output = tmp_root / "11_stability_analysis" / "outputs" / "news_weight_sensitivity_scores.csv"
    passed = summary["news"]["status"] == "success" and output.exists()
    return {
        "passed": passed,
        "command": "run mock News weight sensitivity under 11_stability_analysis",
        "error_summary": "" if passed else str(summary),
        "files_changed": [str(output)] if output.exists() else [],
        "next_action": "check correlation helpers",
    }


def _round_stability_correlations(root: Path) -> dict[str, Any]:
    from stability_weight_sensitivity import pearson_correlation, spearman_correlation

    passed = pearson_correlation([1, 2, 3], [1, 2, 3]) == 1.0 and pearson_correlation([1, 2, 3], [3, 2, 1]) == -1.0 and spearman_correlation([10, 20, 30], [1, 2, 3]) == 1.0
    return {
        "passed": passed,
        "command": "check Pearson and Spearman helper outputs",
        "error_summary": "" if passed else "correlation helper mismatch",
        "files_changed": [],
        "next_action": "check high-volatility marking",
    }


def _round_stability_high_volatility(root: Path) -> dict[str, Any]:
    from run_statistical_stability_analysis import run_statistical_stability_analysis

    tmp_root = _build_stability_tmp_root(root, include_news=True, include_optional=True)
    summary = run_statistical_stability_analysis(root=tmp_root, make_plots=False)
    high_rows = read_csv_rows(tmp_root / "11_stability_analysis" / "outputs" / "high_volatility_cases.csv")
    passed = summary["high_volatility_case_count"] >= 1 and any(row.get("issue_type") for row in high_rows)
    return {
        "passed": passed,
        "command": "run full mock stability analysis and inspect high-volatility cases",
        "error_summary": "" if passed else f"summary={summary}; high_rows={high_rows}",
        "files_changed": [str(tmp_root / "11_stability_analysis" / "outputs" / "high_volatility_cases.csv")],
        "next_action": "check missing News handling",
    }


def _round_stability_missing_news(root: Path) -> dict[str, Any]:
    from run_statistical_stability_analysis import run_statistical_stability_analysis

    tmp_root = _build_stability_tmp_root(root, include_news=False, include_optional=False)
    summary = run_statistical_stability_analysis(root=tmp_root, include_news=True, make_plots=False)
    passed = summary["news_weight_sensitivity"]["status"] == "missing-news" and summary["news_aggregation"]["status"] == "missing-news"
    return {
        "passed": passed,
        "command": "run stability analysis with missing News scores",
        "error_summary": "" if passed else str(summary),
        "files_changed": [str(tmp_root / "11_stability_analysis" / "reports" / "statistical_stability_report.md")],
        "next_action": "check report generation",
    }


def _round_stability_report(root: Path) -> dict[str, Any]:
    from run_statistical_stability_analysis import run_statistical_stability_analysis

    tmp_root = _build_stability_tmp_root(root, include_news=True, include_optional=True)
    summary = run_statistical_stability_analysis(root=tmp_root, make_plots=False)
    report_path = tmp_root / "11_stability_analysis" / "reports" / "statistical_stability_report.md"
    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    passed = report_path.exists() and "Executive Summary" in report_text and "Final Recommendation" in report_text
    return {
        "passed": passed,
        "command": "build unified statistical stability report",
        "error_summary": "" if passed else str(summary),
        "files_changed": [str(report_path)] if report_path.exists() else [],
        "next_action": "validate stability outputs",
    }


def _round_stability_validation(root: Path) -> dict[str, Any]:
    from run_statistical_stability_analysis import run_statistical_stability_analysis
    from validate_stability_outputs import validate_stability_outputs

    tmp_root = _build_stability_tmp_root(root, include_news=True, include_optional=True)
    run_statistical_stability_analysis(root=tmp_root, make_plots=False)
    validation = validate_stability_outputs(root=tmp_root)
    passed = validation["passed"]
    return {
        "passed": passed,
        "command": "validate statistical stability outputs",
        "error_summary": "" if passed else str(validation),
        "files_changed": [str(tmp_root / "11_stability_analysis" / "logs" / "validate_stability_outputs.json")],
        "next_action": "run MDA v2 checks",
    }


def _round_weight_sensitivity(root: Path) -> dict[str, Any]:
    from run_weight_sensitivity import run_weight_sensitivity

    tmp_root = root / "_self_correction" / "weight_sensitivity_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ensure_directories(tmp_root)
    write_csv_rows(
        tmp_root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            _weight_mda_row("MDA_WEIGHT_A", "WTA", "Weight Alpha Berhad", "2024", 5, 5, 3, 3, 2),
            _weight_mda_row("MDA_WEIGHT_B", "WTB", "Weight Beta Berhad", "2024", 3, 5, 5, 3, 2),
            _weight_mda_row("MDA_WEIGHT_C", "WTC", "Weight Cedar Berhad", "2024", 2, 3, 4, 3, 2),
        ],
    )
    write_csv_rows(
        tmp_root / "06_ratings" / "news_document_scores.csv",
        NEWS_DOCUMENT_HEADERS,
        [
            _weight_news_row("NEWS_WEIGHT_A", "WTA", "Weight Alpha Berhad", "2024-01-01", 5, 5, 3, 5, 2),
            _weight_news_row("NEWS_WEIGHT_B", "WTB", "Weight Beta Berhad", "2024-01-02", 3, 3, 3, 3, 2),
            _weight_news_row("NEWS_WEIGHT_C", "WTC", "Weight Cedar Berhad", "2024-01-03", 1, 1, 3, 1, 2),
        ],
    )
    summary = run_weight_sensitivity(tmp_root)
    required = [
        tmp_root / "08_reports" / "weight_sensitivity_report.md",
        tmp_root / "06_ratings" / "weight_sensitivity" / "mda_weight_sensitivity_scores.csv",
        tmp_root / "06_ratings" / "weight_sensitivity" / "news_weight_sensitivity_scores.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    passed = not missing and summary["mda"]["document_count"] == 3 and summary["news"]["document_count"] == 3
    return {
        "passed": passed,
        "command": "run weight sensitivity on synthetic document_scores",
        "error_summary": "" if passed else f"missing={missing}; summary={summary}",
        "files_changed": [str(path) for path in required if path.exists()],
        "fix_applied": "none",
        "next_action": "run parallel scoring checks",
    }


def _weight_mda_row(document_id: str, ticker: str, company: str, year: str, ar01: int, ar02: int, ar03: int, ar04: int, ar05: int) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update(
        {
            "document_id": document_id,
            "doc_type": "mda",
            "ticker": ticker,
            "company_name": company,
            "report_year": year,
            "dimension_count": "5",
            "AR01": str(ar01),
            "AR02": str(ar02),
            "AR03": str(ar03),
            "AR04": str(ar04),
            "AR05": str(ar05),
        }
    )
    return row


def _weight_news_row(document_id: str, ticker: str, company: str, publish_date: str, n01: int, n02: int, n03: int, n04: int, n05: int) -> dict[str, str]:
    row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
    row.update(
        {
            "document_id": document_id,
            "doc_type": "news",
            "ticker": ticker,
            "company_name": company,
            "publish_date": publish_date,
            "dimension_count": "5",
            "N01": str(n01),
            "N02": str(n02),
            "N03": str(n03),
            "N04": str(n04),
            "N05": str(n05),
        }
    )
    return row


def _round_parallel_resume_overwrite_qwen(root: Path) -> dict[str, Any]:
    from parallel_scoring import run_parallel_pipeline

    tmp_root = _build_parallel_tmp_root(root)
    first = run_parallel_pipeline("mda", "pilot", root=tmp_root, limit=2, llm_workers=2, mock_mode=True, overwrite=True, resume=False)
    second = run_parallel_pipeline("mda", "pilot", root=tmp_root, limit=2, llm_workers=2, mock_mode=True, overwrite=False, resume=True)
    third = run_parallel_pipeline("mda", "pilot", root=tmp_root, limit=2, llm_workers=2, mock_mode=True, overwrite=True, resume=True)
    check = run_check(root=root)
    qwen_summary = {"skipped": True}
    if check.get("ollama_available") and check.get("model_available"):
        qwen_summary = _run_qwen_benchmark_bounded(tmp_root, timeout_seconds=180)
    passed = first["success_count"] == 2 and second["skipped_count"] == 2 and third["success_count"] == 2
    return {
        "passed": passed,
        "command": "parallel resume overwrite checks; qwen workers 1/2 benchmark if available",
        "error_summary": "" if passed else f"first={first}; second={second}; third={third}; qwen={qwen_summary}",
        "files_changed": [
            str(tmp_root / "08_reports" / "parallel_benchmark_results.csv"),
            str(tmp_root / "08_reports" / "parallel_benchmark_report.md"),
            str(root / "08_reports" / "ollama_check_report.md"),
        ],
        "fix_applied": f"qwen_summary={qwen_summary}",
        "next_action": "complete",
    }


def _run_qwen_benchmark_bounded(root: Path, timeout_seconds: int) -> dict[str, Any]:
    queue: mp.Queue = mp.Queue()
    process = mp.Process(target=_qwen_benchmark_worker, args=(str(root), queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(10)
        return {
            "skipped": False,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "message": "live qwen benchmark exceeded self-check guardrail; mock/resume checks still passed",
        }
    if not queue.empty():
        return queue.get()
    return {
        "skipped": False,
        "error": f"live qwen benchmark exited without result; exitcode={process.exitcode}",
    }


def _qwen_benchmark_worker(root_text: str, queue: mp.Queue) -> None:
    try:
        rows = run_benchmark(
            "mda",
            sample_size=2,
            workers=[1, 2],
            root=Path(root_text),
            mock_mode=False,
            timeout_seconds=60,
        )
        queue.put({"skipped": False, "rows": rows})
    except Exception as exc:  # noqa: BLE001 - self-check records live benchmark diagnostics.
        queue.put({"skipped": False, "error": str(exc)})


def _round_qwen(root: Path) -> dict[str, Any]:
    check = run_check(root=root)
    if not check.get("ollama_available") or not check.get("model_available"):
        return {
            "passed": True,
            "command": "python 06_scoring/scripts/check_ollama.py",
            "error_summary": "Ollama or qwen3:8b unavailable; qwen JSON test skipped without fake scores.",
            "files_changed": [str(root / "08_reports" / "ollama_check_report.md")],
            "next_action": "run pilot/manual-template branch",
        }
    mda_rows, _, changed = _write_synthetic_inputs(root)
    summary = run_pipeline("mda", "pilot", limit=1, root=root, registry_rows_override=mda_rows[:1], mock_mode=False, write_report=True)
    passed = summary["success_count"] == 1
    return {
        "passed": passed,
        "command": "qwen3:8b short MD&A JSON scoring test",
        "error_summary": "" if passed else json.dumps(summary, ensure_ascii=False),
        "files_changed": changed + [str(root / "05_raw_model_outputs" / "mda")],
        "next_action": "run pilot tests",
    }


def _round_pilot(root: Path) -> dict[str, Any]:
    mda_summary = run_pipeline("mda", "pilot", root=root, mock_mode=False, write_report=True)
    news_summary = run_pipeline("news", "pilot", root=root, mock_mode=False, write_report=True)
    required = [
        root / "01_registry" / "mda_registry.csv",
        root / "01_registry" / "news_registry.csv",
        root / "03_numeric_checks" / "mda_numeric_checks_summary.csv",
        root / "06_ratings" / "mda_ratings_long.csv",
        root / "06_ratings" / "news_ratings_long.csv",
        root / "06_ratings" / "mda_document_scores.csv",
        root / "06_ratings" / "news_document_scores.csv",
        root / "07_review" / "review_log.csv",
        root / "08_reports" / "scoring_summary_report.md",
        root / "_self_correction" / "self_correction_log.md",
    ]
    missing = [str(path) for path in required if not path.exists()]
    passed = not missing
    return {
        "passed": passed,
        "command": "run_scoring.py --doc-type mda/news --mode pilot",
        "error_summary": "" if passed else "Missing files: " + ", ".join(missing),
        "files_changed": [str(path) for path in required if path.exists()],
        "next_action": "complete" if passed else "manual repair required",
        "fix_applied": f"mda_summary={mda_summary}; news_summary={news_summary}",
    }


def _write_synthetic_inputs(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    input_dir = root / "_self_correction" / "synthetic_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    samples = {
        "MDA_SYNTH_GOOD_2024.txt": (
            "[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million. "
            "Net profit increased by 10% from RM10 million to RM11 million.\n\n"
            "[MDA_P002]\nManagement attributed the improvement to higher sales volume, better product mix, and cost control. "
            "The MD&A discusses raw material risk and cautious outlook for consumer demand."
        ),
        "MDA_SYNTH_BAD_2024.txt": (
            "[MDA_P001]\nRevenue increased by 50% from RM100 million to RM120 million. "
            "Profit decreased by 10% from RM50 million to RM45 million.\n\n"
            "[MDA_P002]\nThe discussion provides limited explanation of operating drivers and no clear outlook."
        ),
        "NEWS_SYNTH_SOURCE_2024.txt": (
            "[TITLE]\nTest Food reports higher quarterly sales\n\n"
            "[LEAD]\nThe company said demand improved in export markets.\n\n"
            "[NEWS_P001]\nAccording to the company statement, revenue rose due to stronger sales volume and stable pricing. "
            "The article identifies the source and keeps the claims factual."
        ),
        "NEWS_SYNTH_CLICKBAIT_2024.txt": (
            "[TITLE]\nShocking food stock could explode soon\n\n"
            "[LEAD]\nInvestors are told to watch the company without clear evidence.\n\n"
            "[NEWS_P001]\nThe article gives promotional claims without named sources, dates, or verifiable figures."
        ),
    }
    changed = []
    for filename, text in samples.items():
        path = input_dir / filename
        path.write_text(text, encoding="utf-8")
        changed.append(str(path))

    mda_rows = [
        {
            "document_id": "MDA_SYNTH_GOOD_2024",
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "0001",
            "ticker": "SYNTHGOOD",
            "company_name": "Synthetic Good Food Berhad",
            "report_year": "2024",
            "extraction_status": "success",
            "text_path": str(input_dir / "MDA_SYNTH_GOOD_2024.txt"),
        },
        {
            "document_id": "MDA_SYNTH_BAD_2024",
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "0002",
            "ticker": "SYNTHBAD",
            "company_name": "Synthetic Bad Food Berhad",
            "report_year": "2024",
            "extraction_status": "success",
            "text_path": str(input_dir / "MDA_SYNTH_BAD_2024.txt"),
        },
    ]
    news_rows = [
        {
            "document_id": "NEWS_SYNTH_SOURCE_2024",
            "doc_type": "news",
            "include_flag": "Yes",
            "company_name": "Synthetic Good Food Berhad",
            "ticker": "SYNTHGOOD",
            "source_name": "Synthetic Wire",
            "publish_date": "2024-01-15",
            "title": "Test Food reports higher quarterly sales",
            "url": "",
            "text_path": str(input_dir / "NEWS_SYNTH_SOURCE_2024.txt"),
        },
        {
            "document_id": "NEWS_SYNTH_CLICKBAIT_2024",
            "doc_type": "news",
            "include_flag": "Yes",
            "company_name": "Synthetic Bad Food Berhad",
            "ticker": "SYNTHBAD",
            "source_name": "Synthetic Blog",
            "publish_date": "2024-01-16",
            "title": "Shocking food stock could explode soon",
            "url": "",
            "text_path": str(input_dir / "NEWS_SYNTH_CLICKBAIT_2024.txt"),
        },
    ]
    return mda_rows, news_rows, changed


def _build_cross_validation_tmp_root(root: Path) -> Path:
    tmp_root = root / "_self_correction" / "cross_validation_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ensure_directories(tmp_root)
    mda_dir = tmp_root / "02_extracted_text" / "mda"
    news_dir = tmp_root / "02_extracted_text" / "news"
    mda_dir.mkdir(parents=True, exist_ok=True)
    news_dir.mkdir(parents=True, exist_ok=True)
    mda_text = (
        "[MDA_P001]\nRevenue increased in 2024 as domestic demand improved.\n\n"
        "[MDA_P002]\nRaw material costs increased and pressured margins.\n\n"
        "[MDA_P003]\nThe company maintains a cautious outlook for 2025.\n\n"
        "[MDA_P004]\nThe group expanded capacity through a new production line."
    )
    (mda_dir / "MDA_CV_2024.txt").write_text(mda_text, encoding="utf-8")
    news_texts = {
        "NEWS_CV_RAW_2024": "Industry data showed F&B raw material costs increased in 2024, adding pressure to producers.",
        "NEWS_CV_GROWTH_2025": "Analysts said Cross Food expects explosive growth in 2025 after recent expansion.",
        "NEWS_CV_SAFETY_2024": "Regulators reported a food safety issue affecting Cross Food products in 2024.",
        "NEWS_CV_AR_2024": "According to the annual report, Cross Food said raw material costs increased during the year.",
        "NEWS_CV_REV_2024": "Market report said Cross Food revenue decreased in 2024 despite management saying revenue increased.",
    }
    for document_id, text in news_texts.items():
        (news_dir / f"{document_id}.txt").write_text(text, encoding="utf-8")
    mda_rows = [
        {
            "document_id": "MDA_CV_2024",
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "9998",
            "ticker": "CROSS",
            "company_name": "Cross Food Berhad",
            "report_year": "2024",
            "extraction_status": "success",
            "text_path": "02_extracted_text/mda/MDA_CV_2024.txt",
            "source_path": "",
            "source_file_sha256": "",
            "notes": "",
        }
    ]
    news_rows = []
    for idx, document_id in enumerate(news_texts, start=1):
        news_rows.append(
            {
                "document_id": document_id,
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Cross Food Berhad",
                "ticker": "CROSS",
                "source_name": "Synthetic News",
                "publish_date": f"2024-0{idx}-15",
                "title": document_id,
                "url": "",
                "text_path": f"02_extracted_text/news/{document_id}.txt",
                "notes": "",
            }
        )
    write_csv_rows(tmp_root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], mda_rows)
    write_csv_rows(tmp_root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], news_rows)
    write_csv_rows(
        tmp_root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            {
                "document_id": "MDA_CV_2024",
                "doc_type": "mda",
                "ticker": "CROSS",
                "company_name": "Cross Food Berhad",
                "report_year": "2024",
                "total_score_100": "75",
            }
        ],
    )
    write_csv_rows(
        tmp_root / "06_ratings" / "news_document_scores.csv",
        NEWS_DOCUMENT_HEADERS,
        [
            {
                "document_id": row["document_id"],
                "doc_type": "news",
                "ticker": "CROSS",
                "company_name": "Cross Food Berhad",
                "publish_date": row["publish_date"],
                "total_score_100": "65",
            }
            for row in news_rows
        ],
    )
    return tmp_root


def _build_stabilization_tmp_root(root: Path) -> Path:
    tmp_root = root / "_self_correction" / "mda_v2_stabilization_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ensure_directories(tmp_root)
    for rel in [
        "04_prompts/scoring_output_schema.json",
        "04_prompts/mda_scoring_prompt.txt",
    ]:
        target = tmp_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((root / rel).read_text(encoding="utf-8"), encoding="utf-8")

    text_dir = tmp_root / "02_extracted_text" / "mda_clean_v2"
    text_dir.mkdir(parents=True, exist_ok=True)
    texts = {
        "MDA_STABLE_GOOD_2024": (
            "[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.\n\n"
            "[MDA_P002]\nManagement discussed raw material costs, risk, and a cautious outlook."
        ),
        "MDA_STABLE_FAIL_2024": (
            "[MDA_P001]\nRevenue increased by 50% from RM100 million to RM120 million.\n\n"
            "[MDA_P002]\nThe discussion includes limited detail and requires manual review."
        ),
    }
    for doc_id, text in texts.items():
        (text_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

    rows = [
        {
            "document_id": doc_id,
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "9901" if "GOOD" in doc_id else "9902",
            "ticker": "STABLE",
            "company_name": "Stable Food Berhad",
            "report_year": "2024",
            "extraction_status": "success",
            "text_path": f"02_extracted_text/mda_clean_v2/{doc_id}.txt",
            "source_path": "",
            "source_file_sha256": "",
            "notes": "",
        }
        for doc_id in texts
    ]
    write_csv_rows(tmp_root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], rows)
    write_csv_rows(
        tmp_root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [{"document_id": "MDA_PILOT_ONLY", "doc_type": "mda", "total_score_100": "50"}],
    )
    write_csv_rows(tmp_root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], [])

    failed_dir = tmp_root / "06_ratings" / "mda_v2_sample_rescore" / "per_document"
    failed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_root / "05_raw_model_outputs" / "mda_v2_sample"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "MDA_STABLE_FAIL_2024_attempt_1.json"
    save_json(raw_path, {"content": "```json\n{\"document_id\":\"MDA_STABLE_FAIL_2024\"}\n```"})
    (failed_dir / "MDA_STABLE_FAIL_2024_parsed.json").write_text(
        json.dumps(
            {
                "document_id": "MDA_STABLE_FAIL_2024",
                "doc_type": "mda",
                "status": "failed",
                "result": {
                    "error_type": "scoring_failed",
                    "error_message": "json_parse_failed",
                    "parse_status": "parse_failed",
                    "schema_validation_status": "invalid",
                    "attempts_used": 3,
                    "raw_output_path": str(raw_path),
                },
            }
        ),
        encoding="utf-8",
    )
    numeric_dir = tmp_root / "03_numeric_checks" / "mda_v2"
    numeric_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        numeric_dir / "MDA_STABLE_FAIL_2024_numeric_checks.json",
        {
            "document_id": "MDA_STABLE_FAIL_2024",
            "checks": [
                {
                    "metric": "revenue",
                    "current_value": 120,
                    "previous_value": 100,
                    "stated_pct": 50,
                    "calculated_change_pct": 20,
                    "status": "material_discrepancy",
                    "source_text": "[MDA_P001] Revenue increased by 50% from RM100 million to RM120 million.",
                    "evidence_locator": "[MDA_P001]",
                }
            ],
        },
    )
    write_csv_rows(
        tmp_root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv",
        [
            "document_id",
            "num_financial_mentions",
            "num_calculable_changes",
            "num_consistent_checks",
            "num_minor_discrepancies",
            "num_material_discrepancies",
            "num_severe_direction_conflicts",
            "num_not_calculable",
            "has_revenue_discussion",
            "has_profit_discussion",
            "has_cost_discussion",
            "has_segment_discussion",
            "has_margin_discussion",
            "overall_numeric_consistency",
            "internal_check_notes",
        ],
        [
            {
                "document_id": "MDA_STABLE_FAIL_2024",
                "num_material_discrepancies": "1",
                "num_severe_direction_conflicts": "0",
                "overall_numeric_consistency": "weak",
            }
        ],
    )
    write_csv_rows(
        tmp_root / "08_reports" / "mda_text_quality_audit.csv",
        ["document_id", "needs_review"],
        [{"document_id": "MDA_STABLE_FAIL_2024", "needs_review": "true"}],
    )
    return tmp_root


def _build_claim_tmp_root(root: Path) -> Path:
    tmp_root = root / "_self_correction" / "claim_level_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ensure_directories(tmp_root)
    mda_dir = tmp_root / "02_extracted_text" / "mda_clean_v2"
    news_dir = tmp_root / "02_extracted_text" / "news"
    mda_dir.mkdir(parents=True, exist_ok=True)
    news_dir.mkdir(parents=True, exist_ok=True)
    (mda_dir / "MDA_CLAIM_SELF_2024.txt").write_text(
        "[MDA_P001]\nRevenue increased from 460 to 500 due to export demand and lower domestic sales.\n\n"
        "[MDA_P002]\nRaw material cost pressure increased and management expects a cautious outlook.",
        encoding="utf-8",
    )
    (news_dir / "NEWS_CLAIM_SELF_2024.txt").write_text(
        "[TITLE]\nSelf Food revenue rises as exports improve\n\n"
        "[NEWS_P001]\nAnalysts said Self Food revenue increased after stronger export demand, while domestic sales declined.",
        encoding="utf-8",
    )
    write_csv_rows(
        tmp_root / "01_registry" / "mda_registry.csv",
        REGISTRY_HEADERS["mda"],
        [
            {
                "document_id": "MDA_CLAIM_SELF_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "9909",
                "ticker": "SELF",
                "company_name": "Self Food Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda_clean_v2/MDA_CLAIM_SELF_2024.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        tmp_root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_CLAIM_SELF_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Self Food Berhad",
                "ticker": "SELF",
                "source_name": "Synthetic News",
                "publish_date": "2024-06-01",
                "title": "Self Food revenue rises as exports improve",
                "url": "",
                "text_path": "02_extracted_text/news/NEWS_CLAIM_SELF_2024.txt",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        tmp_root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [{"document_id": "MDA_CLAIM_SELF_2024", "doc_type": "mda", "AR02": "3", "AR02_std": "50", "total_score_100": "62.5"}],
    )
    return tmp_root


def _build_parallel_tmp_root(root: Path) -> Path:
    tmp_root = root / "_self_correction" / "parallel_tmp"
    ensure_directories(tmp_root)
    mda_rows, _, _ = _write_synthetic_inputs(tmp_root)
    write_csv_rows(tmp_root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], mda_rows)
    source_root = root
    for rel in [
        "04_prompts/scoring_output_schema.json",
        "04_prompts/mda_scoring_prompt.txt",
        "04_prompts/news_scoring_prompt.txt",
    ]:
        target = tmp_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((source_root / rel).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_root


def _build_stability_tmp_root(root: Path, *, include_news: bool, include_optional: bool) -> Path:
    tmp_root = root / "_self_correction" / "stability_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ensure_directories(tmp_root)
    mda_rows = [
        _self_mda_score("MDA_STAB_A", "STA", "Stability A Berhad", "2024", 5, 5, 2, 3, 3, 70),
        _self_mda_score("MDA_STAB_B", "STB", "Stability B Berhad", "2024", 2, 5, 5, 3, 3, 72),
        _self_mda_score("MDA_STAB_C", "STC", "Stability C Berhad", "2024", 3, 3, 3, 3, 3, 50),
        _self_mda_score("MDA_STAB_D", "STD", "Stability D Berhad", "2024", 1, 1, 2, 2, 3, 22),
    ]
    write_csv_rows(tmp_root / "06_ratings" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, mda_rows)
    if include_news:
        news_rows = [
            _self_news_score("NEWS_STAB_A1", "STA", "Stability A Berhad", "2024-01-01", "Self Wire", "A strong", 5, 5, 4, 5, 4, 90),
            _self_news_score("NEWS_STAB_A2", "STA", "Stability A Berhad", "2024-02-01", "Self Wire", "A weak", 2, 2, 3, 2, 4, 35),
            _self_news_score("NEWS_STAB_A3", "STA", "Stability A Berhad", "2024-03-01", "Self Wire", "A neutral", 3, 4, 3, 4, 4, 62),
            _self_news_score("NEWS_STAB_B1", "STB", "Stability B Berhad", "2024-01-15", "Self Wire", "B stable", 4, 5, 4, 4, 4, 78),
            _self_news_score("NEWS_STAB_B2", "STB", "Stability B Berhad", "2024-02-15", "Self Wire", "B stable 2", 4, 4, 4, 4, 4, 75),
        ]
        write_csv_rows(tmp_root / "06_ratings" / "news_document_scores.csv", NEWS_DOCUMENT_HEADERS, news_rows)
        registry_rows = []
        for row in news_rows:
            registry_row = {field: "" for field in REGISTRY_HEADERS["news"]}
            registry_row.update(
                {
                    "document_id": row["document_id"],
                    "doc_type": "news",
                    "include_flag": "Yes",
                    "company_name": row["company_name"],
                    "ticker": row["ticker"],
                    "source_name": row["source_name"],
                    "publish_date": row["publish_date"],
                    "title": row["title"],
                }
            )
            registry_rows.append(registry_row)
        write_csv_rows(tmp_root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], registry_rows)
    if include_optional:
        write_csv_rows(tmp_root / "06_ratings" / "repeat_runs" / "mda_repeat_run_1_document_scores.csv", MDA_DOCUMENT_HEADERS, [_self_mda_score("MDA_STAB_A", "STA", "Stability A Berhad", "2024", 5, 5, 2, 3, 3, 70)])
        write_csv_rows(tmp_root / "06_ratings" / "repeat_runs" / "mda_repeat_run_2_document_scores.csv", MDA_DOCUMENT_HEADERS, [_self_mda_score("MDA_STAB_A", "STA", "Stability A Berhad", "2024", 3, 2, 5, 4, 3, 47)])
        write_csv_rows(tmp_root / "archive" / "v1_initial_mda_scoring" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, [_self_mda_score("MDA_STAB_A", "STA", "Stability A Berhad", "2024", 5, 5, 2, 3, 3, 70)])
        write_csv_rows(tmp_root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv", MDA_DOCUMENT_HEADERS, [_self_mda_score("MDA_STAB_A", "STA", "Stability A Berhad", "2024", 3, 2, 5, 4, 3, 47)])
        write_csv_rows(tmp_root / "06_ratings" / "document_level" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, [_self_mda_score("MDA_STAB_A", "STA", "Stability A Berhad", "2024", 5, 5, 2, 3, 3, 70)])
        write_csv_rows(tmp_root / "06_ratings" / "single_dimension" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, [_self_mda_score("MDA_STAB_A", "STA", "Stability A Berhad", "2024", 4, 4, 3, 3, 3, 65)])
        if include_news:
            write_csv_rows(tmp_root / "06_ratings" / "repeat_runs" / "news_repeat_run_1_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_self_news_score("NEWS_STAB_A1", "STA", "Stability A Berhad", "2024-01-01", "Self Wire", "A strong", 5, 5, 4, 5, 4, 90)])
            write_csv_rows(tmp_root / "06_ratings" / "repeat_runs" / "news_repeat_run_2_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_self_news_score("NEWS_STAB_A1", "STA", "Stability A Berhad", "2024-01-01", "Self Wire", "A strong", 5, 4, 4, 4, 4, 82)])
            write_csv_rows(tmp_root / "06_ratings" / "document_level" / "news_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_self_news_score("NEWS_STAB_A1", "STA", "Stability A Berhad", "2024-01-01", "Self Wire", "A strong", 5, 5, 4, 5, 4, 90)])
            write_csv_rows(tmp_root / "06_ratings" / "single_dimension" / "news_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_self_news_score("NEWS_STAB_A1", "STA", "Stability A Berhad", "2024-01-01", "Self Wire", "A strong", 5, 4, 4, 4, 4, 82)])
    return tmp_root


def _self_mda_score(document_id: str, ticker: str, company: str, year: str, ar01: int, ar02: int, ar03: int, ar04: int, ar05: int, total: int) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "mda", "ticker": ticker, "company_name": company, "report_year": year, "dimension_count": "5", "total_score_100": str(total), "AR01": str(ar01), "AR02": str(ar02), "AR03": str(ar03), "AR04": str(ar04), "AR05": str(ar05)})
    return row


def _self_news_score(document_id: str, ticker: str, company: str, publish_date: str, source_name: str, title: str, n01: int, n02: int, n03: int, n04: int, n05: int, total: int) -> dict[str, str]:
    row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "news", "ticker": ticker, "company_name": company, "source_name": source_name, "publish_date": publish_date, "title": title, "dimension_count": "5", "total_score_100": str(total), "N01": str(n01), "N02": str(n02), "N03": str(n03), "N04": str(n04), "N05": str(n05)})
    return row


def _build_mda_v2_tmp_root(root: Path) -> Path:
    tmp_root = root / "_self_correction" / "mda_v2_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    ensure_directories(tmp_root)
    for rel in [
        "00_scorebook/mda_scorebook.yaml",
        "04_prompts/scoring_output_schema.json",
        "04_prompts/mda_scoring_prompt.txt",
    ]:
        target = tmp_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((root / rel).read_text(encoding="utf-8"), encoding="utf-8")

    text_dir = tmp_root / "02_extracted_text" / "mda"
    text_dir.mkdir(parents=True, exist_ok=True)
    texts = {
        "MDA_V2_GOOD_2024": (
            "ANNUAL REPORT 2024\n1\nBusiness Review\n"
            "[MDA_P001]\n"
            "Revenue increased by 20% from RM100 million to RM120 million due to higher sales volume and better product mix.\n\n"
            "Financial Review\n"
            "[MDA_P002]\n"
            "Net profit increased by 10% from RM10 million to RM11 million as cost control offset raw material pressure.\n\n"
            "Risk and Outlook\n"
            "[MDA_P003]\n"
            "Management discussed market demand, commodity cost risk, and a cautious outlook for the next financial year.\n"
        ),
        "MDA_V2_BAD_2024": (
            "2\nPerformance Review\n"
            "[MDA_P001]\n"
            "Revenue increased by 50% from RM100 million to RM120 million while profit decreased by 10% from RM50 million to RM45 million.\n\n"
            "Business Review\n"
            "[MDA_P002]\n"
            "The discussion gives limited explanation of operating drivers and repeats general statements about the market.\n\n"
            "Outlook\n"
            "[MDA_P003]\n"
            "Management mentioned demand uncertainty but gave little segment or cost detail.\n"
        ),
    }
    for doc_id, text in texts.items():
        (text_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")

    rows = [
        {
            "document_id": "MDA_V2_GOOD_2024",
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "0001",
            "ticker": "V2GOOD",
            "company_name": "V2 Good Food Berhad",
            "report_year": "2024",
            "extraction_status": "success",
            "text_path": "02_extracted_text/mda/MDA_V2_GOOD_2024.txt",
        },
        {
            "document_id": "MDA_V2_BAD_2024",
            "doc_type": "mda",
            "include_flag": "Yes",
            "stock_code": "0002",
            "ticker": "V2BAD",
            "company_name": "V2 Bad Food Berhad",
            "report_year": "2024",
            "extraction_status": "success",
            "text_path": "02_extracted_text/mda/MDA_V2_BAD_2024.txt",
        },
    ]
    write_csv_rows(tmp_root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], rows)
    return tmp_root


def _validate_scorebook_text(path: Path, doc_type: str) -> None:
    text = path.read_text(encoding="utf-8")
    required = ["scorebook_version:", "doc_type:", "scoring_basis:", "dimensions:", "weight:"]
    missing = [item for item in required if item not in text]
    if missing:
        raise ValueError(f"{path} missing {missing}")
    expected_codes = ["AR01", "AR02", "AR03", "AR04", "AR05"] if doc_type == "mda" else ["N01", "N02", "N03", "N04", "N05"]
    for code in expected_codes:
        if code not in text:
            raise ValueError(f"{path} missing dimension {code}")


def _append_log(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"## Round {entry['round_number']}: {entry['title']}\n\n")
        fh.write(f"- command run: `{entry['command_run']}`\n")
        fh.write(f"- pass/fail: {'pass' if entry['pass'] else 'fail'}\n")
        fh.write(f"- error summary: {entry['error_summary'] or 'none'}\n")
        fh.write(f"- files changed: {', '.join(entry['files_changed']) if entry['files_changed'] else 'none'}\n")
        fh.write(f"- fix applied: {entry['fix_applied']}\n")
        fh.write(f"- next action: {entry['next_action']}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scoring pipeline self-check rounds, including claim-level framework checks.")
    parser.add_argument("--max-rounds", type=int, default=10)
    args = parser.parse_args()
    summary = run_self_check(args.max_rounds)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
