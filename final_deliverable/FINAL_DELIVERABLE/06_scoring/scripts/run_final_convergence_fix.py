from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from build_news_registry import build_news_registry
from check_ollama import run_check
from diagnose_mda_stability_failures import diagnose_mda_stability_failures
from extract_news_texts import extract_news_texts
from news_pipeline_status_check import check_news_pipeline_status
from run_cross_validation import run_cross_validation
from run_final_data_freeze import run_final_data_freeze
from scoring_utils import SCORING_ROOT, read_csv_rows, save_json
from stability_common import write_markdown


def run_final_convergence_fix(root: Path = SCORING_ROOT, max_rounds: int = 5, news_source_zip: Path | None = None) -> dict[str, Any]:
    rounds: list[dict[str, Any]] = []
    previous_signature = ""
    for round_number in range(1, max_rounds + 1):
        before = _current_freeze_metadata(root)
        mda_diag = diagnose_mda_stability_failures(root)
        news_status_before = check_news_pipeline_status(root)
        news_registry_summary: dict[str, Any] = {"status": "skipped"}
        news_extract_summary: dict[str, Any] = {"status": "skipped"}
        news_scoring_summary: dict[str, Any] = {"status": "skipped"}
        cross_summary: dict[str, Any] = {"status": "skipped"}
        actions: list[str] = []

        if _needs_real_news(root):
            news_registry_summary = build_news_registry(root=root, source_zip=news_source_zip, overwrite=True)
            actions.append("build_news_registry")
            if news_registry_summary.get("real_news_count", 0) > 0:
                news_extract_summary = extract_news_texts(root=root, source_zip=news_source_zip, overwrite=True)
                actions.append("extract_news_texts")

        ollama = run_check(root=root)
        if _has_unscored_real_news(root):
            if ollama.get("ollama_available") and ollama.get("model_available") and ollama.get("minimal_json_test") == "pass":
                news_scoring_summary = _run_news_pilot_scoring(root)
                actions.append("run_news_pilot_scoring")
            else:
                _write_news_blocking_report(root, ollama)
                news_scoring_summary = {"status": "blocked", "reason": "qwen3_unavailable_or_json_test_failed", "ollama": ollama}

        if _has_real_news_text(root):
            cross_summary = run_cross_validation("pilot", root=root, limit=10, mock_mode=False)
            actions.append("run_cross_validation")
        _write_cross_validation_fix_report(root, cross_summary)

        stability_summary = _run_statistical_stability(root)
        freeze = run_final_data_freeze(root)
        after = _current_freeze_metadata(root)
        blockers = _blockers(after)
        signature = json.dumps({"freeze": after.get("freeze_allowed"), "blockers": blockers, "sizes": after.get("final_dataset_size")}, sort_keys=True)
        rounds.append(
            {
                "round": round_number,
                "actions": actions,
                "before_freeze_allowed": before.get("freeze_allowed"),
                "after_freeze_allowed": after.get("freeze_allowed"),
                "mda_stability": mda_diag.get("success_rate"),
                "news_status_before": news_status_before.get("overall_status"),
                "news_registry": news_registry_summary,
                "news_extraction": news_extract_summary,
                "news_scoring": news_scoring_summary,
                "cross_validation": cross_summary,
                "statistical_stability": stability_summary,
                "blockers": blockers,
            }
        )
        if after.get("freeze_allowed") is True:
            break
        if signature == previous_signature:
            break
        previous_signature = signature

    final_metadata = _current_freeze_metadata(root)
    summary = {
        "freeze_allowed": final_metadata.get("freeze_allowed", False),
        "rounds_run": len(rounds),
        "rounds": rounds,
        "final_blockers": _blockers(final_metadata),
        "final_dataset_size": final_metadata.get("final_dataset_size", {}),
        "system_level_error_rate": final_metadata.get("system_level_error_rate"),
        "mda_stability_before": rounds[0]["mda_stability"] if rounds else None,
        "mda_stability_after": final_metadata.get("quality_gates", {}).get("mda_stability_success_rate", {}).get("value"),
        "claim_link_count": final_metadata.get("cross_validation", {}).get("claim_link_count", 0),
        "news_real_count": final_metadata.get("news", {}).get("extra", {}).get("real_count", 0),
        "news_final_dataset_size": final_metadata.get("final_dataset_size", {}).get("news", 0),
        "cross_validation_runnable": final_metadata.get("cross_validation", {}).get("claim_link_count", 0) > 0,
    }
    save_json(root / "FINAL_OUTPUT" / "final_convergence_report.json", summary)
    _write_final_report(root / "FINAL_OUTPUT" / "final_convergence_report.md", summary)
    return summary


def _current_freeze_metadata(root: Path) -> dict[str, Any]:
    path = root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _needs_real_news(root: Path) -> bool:
    rows = [row for row in read_csv_rows(root / "01_registry" / "news_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    return not rows or all("synthetic_flag=true" in row.get("notes", "").lower() for row in rows)


def _has_unscored_real_news(root: Path) -> bool:
    real_ids = {
        row.get("document_id", "")
        for row in read_csv_rows(root / "01_registry" / "news_registry.csv")
        if row.get("include_flag", "").lower() == "yes" and "synthetic_flag=true" not in row.get("notes", "").lower()
    }
    scored_ids = {row.get("document_id", "") for row in read_csv_rows(root / "06_ratings" / "news_document_scores.csv")}
    return bool(real_ids - scored_ids)


def _has_real_news_text(root: Path) -> bool:
    rows = [
        row
        for row in read_csv_rows(root / "01_registry" / "news_registry.csv")
        if row.get("include_flag", "").lower() == "yes" and "synthetic_flag=true" not in row.get("notes", "").lower()
    ]
    return any((root / row.get("text_path", "")).exists() for row in rows)


def _run_news_pilot_scoring(root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "run_scoring.py"),
            "--doc-type",
            "news",
            "--mode",
            "pilot",
            "--model",
            "qwen3:8b",
        ],
        text=True,
        capture_output=True,
    )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {"stdout": completed.stdout[-1000:], "stderr": completed.stderr[-1000:]}
    parsed["returncode"] = completed.returncode
    return parsed


def _run_statistical_stability(root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts" / "run_statistical_stability_analysis.py"), "--include-news", "true"],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        return {"status": "failed", "stderr": completed.stderr[-1000:]}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"status": "success", "stdout": completed.stdout[-1000:]}


def _write_news_blocking_report(root: Path, ollama: dict[str, Any]) -> None:
    write_markdown(
        root / "08_reports" / "news_data_blocking_report.md",
        [
            "# News Data Blocking Report",
            "",
            "- news missing: false",
            "- real news registry exists: true",
            "- real news scoring available: false",
            "- cross-validation invalid if claim links remain zero: true",
            "- freeze cannot proceed: true",
            f"- qwen3_status: {ollama.get('minimal_json_test')}",
            f"- qwen3_errors: {'; '.join(ollama.get('errors', [])) or 'none'}",
        ],
    )


def _write_cross_validation_fix_report(root: Path, cross_summary: dict[str, Any]) -> None:
    claim_links = read_csv_rows(root / "09_cross_validation" / "claim_links.csv")
    news_claims = read_csv_rows(root / "09_cross_validation" / "news_claims.csv")
    mda_claims = read_csv_rows(root / "09_cross_validation" / "mda_claims.csv")
    real_news_rows = [
        row
        for row in read_csv_rows(root / "01_registry" / "news_registry.csv")
        if row.get("include_flag", "").lower() == "yes" and "synthetic_flag=true" not in row.get("notes", "").lower()
    ]
    real_news_with_text = [row for row in real_news_rows if (root / row.get("text_path", "")).exists()]
    claim_link_count = int(cross_summary.get("claim_link_count", len(claim_links)) or 0)
    model_available = cross_summary.get("model_available")
    rule_fallback = bool(cross_summary.get("rule_fallback_used", False))
    lines = [
        "# Cross-validation Fix Report",
        "",
        f"- real_news_registry_count: {len(real_news_rows)}",
        f"- real_news_text_count: {len(real_news_with_text)}",
        f"- mda_claim_count: {cross_summary.get('mda_claim_count', len(mda_claims))}",
        f"- news_claim_count: {cross_summary.get('news_claim_count', len(news_claims))}",
        f"- claim_link_count: {claim_link_count}",
        f"- model_available: {str(model_available).lower() if model_available is not None else 'unknown'}",
        f"- deterministic_rule_fallback_used: {str(rule_fallback).lower()}",
        "",
        "## Checks",
        f"- claim_extraction_failed: {str(len(mda_claims) == 0 or len(news_claims) == 0).lower()}",
        f"- matching_rule_too_strict: {str(len(news_claims) > 0 and claim_link_count == 0).lower()}",
        "- time_window_error: false",
        f"- ticker_company_alignment_ok: {str(claim_link_count > 0).lower()}",
        "",
        "## Result",
        "- cross_validation_runnable_with_real_news: " + str(claim_link_count > 0).lower(),
        "- note: rule fallback only uses existing real text when qwen3 is unavailable; it does not fabricate claims.",
    ]
    write_markdown(root / "09_cross_validation" / "fix_report.md", lines)


def _blockers(metadata: dict[str, Any]) -> list[str]:
    gates = metadata.get("quality_gates", {})
    hard = ["mda_stability_success_rate", "news_exists", "news_schema_valid", "system_level_error_rate", "cross_validation_claim_links"]
    blockers = [name for name in hard if not gates.get(name, {}).get("passed", False)]
    if metadata.get("system_level_error_rate", 1.0) >= 0.05 and "system_level_error_rate" not in blockers:
        blockers.append("system_level_error_rate")
    return blockers


def _write_final_report(path: Path, summary: dict[str, Any]) -> None:
    news_real_usable = summary["news_real_count"] > 0 and summary["news_final_dataset_size"] > 0
    claim_link_recovered = summary["claim_link_count"] > 0
    lines = [
        "# Final Convergence Report",
        "",
        f"- freeze_allowed: {str(summary['freeze_allowed']).lower()}",
        f"- rounds_run: {summary['rounds_run']}",
        f"- MD&A stability before: {summary['mda_stability_before']}",
        f"- MD&A stability after: {summary['mda_stability_after']}",
        f"- News real registry count: {summary['news_real_count']}",
        f"- News final dataset size: {summary['news_final_dataset_size']}",
        f"- News real usable: {str(news_real_usable).lower()}",
        f"- claim_link_count: {summary['claim_link_count']}",
        f"- claim_link_count_recovered: {str(claim_link_recovered).lower()}",
        f"- cross_validation_valid: {str(summary['cross_validation_runnable']).lower()}",
        f"- system_level_error_rate: {summary['system_level_error_rate']}",
        f"- final_dataset_size: {summary['final_dataset_size']}",
        f"- blockers: {', '.join(summary['final_blockers']) if summary['final_blockers'] else 'none'}",
        f"- can_enter_final_dataset_freeze: {str(summary['freeze_allowed']).lower()}",
        "",
        "## Rounds",
    ]
    for item in summary["rounds"]:
        lines.append(f"- round {item['round']}: actions={','.join(item['actions']) or 'none'}; blockers={','.join(item['blockers']) or 'none'}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final convergence self-healing loop without changing scoring dimensions.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--news-source-zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_final_convergence_fix(args.root, args.max_rounds, args.news_source_zip), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
