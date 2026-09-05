from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from repair_ollama_runtime import repair_ollama_runtime
from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import SCORING_ROOT, read_csv_rows
from stability_common import write_markdown


TARGET_FAILURES = {"missing_dimension", "json_parse_failure", "schema_validation_failure"}


def recover_missing_mda_dimensions(
    root: Path = SCORING_ROOT,
    model: str = "qwen3:8b",
    runtime_status: dict[str, Any] | None = None,
    max_scoring_retries: int = 3,
    llm_workers: int = 1,
) -> dict[str, Any]:
    cases = read_csv_rows(root / "PAPER_OUTPUT" / "00_status" / "mda_stability_blocker_cases.csv")
    target_doc_ids = sorted(
        {
            row.get("document_id", "")
            for row in cases
            if row.get("failure_reason") in TARGET_FAILURES or row.get("action_status") == "blocked_ollama_unavailable"
        }
    )
    expected_missing = sum(1 for row in cases if row.get("failure_reason") == "missing_dimension")
    existing = _existing_recovery_summary(root, expected_missing, len(target_doc_ids))
    if existing.get("should_stop"):
        _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_missing_dimensions_recovery_report.md", existing)
        return existing
    runtime_status = runtime_status or repair_ollama_runtime(root=root, model=model, max_repair_rounds=0)
    if not runtime_status.get("runtime_healthy"):
        summary = {
            "status": "blocked",
            "reason": "ollama_runtime_unhealthy",
            "missing_dimensions_expected": expected_missing,
            "missing_dimensions_recovered": 0,
            "missing_dimensions_still_failed": expected_missing,
            "documents_targeted": len(target_doc_ids),
            "documents_fully_recovered": 0,
            "schema_success_rate": 0.0,
            "evidence_match_rate": 0.0,
        }
        _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_missing_dimensions_recovery_report.md", summary)
        return summary
    if not target_doc_ids:
        summary = {
            "status": "skipped",
            "reason": "no missing stability dimensions",
            "missing_dimensions_expected": 0,
            "missing_dimensions_recovered": 0,
            "missing_dimensions_still_failed": 0,
            "documents_targeted": 0,
            "documents_fully_recovered": 0,
            "schema_success_rate": 0.0,
            "evidence_match_rate": 0.0,
        }
        _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_missing_dimensions_recovery_report.md", summary)
        return summary
    smoke_gate = _context_budget_smoke_gate(root)
    if not smoke_gate.get("passed"):
        summary = {
            "status": "blocked",
            "reason": "context_budget_smoke_test_not_passed",
            "missing_dimensions_expected": expected_missing,
            "missing_dimensions_recovered": 0,
            "missing_dimensions_still_failed": expected_missing,
            "documents_targeted": len(target_doc_ids),
            "documents_fully_recovered": 0,
            "schema_success_rate": 0.0,
            "evidence_match_rate": 0.0,
            "smoke_summary_path": str(smoke_gate.get("path", "")),
            "smoke_summary": smoke_gate.get("summary", {}),
        }
        _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_missing_dimensions_recovery_report.md", summary)
        return summary
    result = run_single_dimension_scoring(
        "mda",
        root=root,
        doc_ids=target_doc_ids,
        output_name="mda_context_repair",
        ratings_prefix="mda_context_repair",
        model=model,
        llm_workers=llm_workers,
        resume=True,
        overwrite=False,
        max_retries=max_scoring_retries,
        timeout_seconds=180,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
    )
    recovered = int(result.get("dimension_success_count", 0))
    summary = {
        "status": result.get("status", "completed"),
        "missing_dimensions_expected": expected_missing,
        "missing_dimensions_recovered": recovered,
        "missing_dimensions_still_failed": max(0, expected_missing - recovered),
        "documents_targeted": len(target_doc_ids),
        "documents_fully_recovered": int(result.get("success_count", 0)),
        "schema_success_rate": _rate(result.get("dimension_success_count", 0), result.get("dimension_task_count", 0)),
        "evidence_match_rate": result.get("json_validation_summary", {}).get("evidence_match_rate", 0.0),
        "summary": result,
    }
    _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_missing_dimensions_recovery_report.md", summary)
    return summary


def _context_budget_smoke_gate(root: Path) -> dict[str, Any]:
    path = root / "08_reports" / "context_budget_smoke_test_summary.json"
    if not path.exists():
        return {"passed": False, "path": str(path), "summary": {}}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"passed": False, "path": str(path), "summary": {}}
    passed = bool(summary.get("overall_status") == "passed" and summary.get("allow_recover_55_dimensions") is True)
    return {"passed": passed, "path": str(path), "summary": summary}


def _existing_recovery_summary(root: Path, expected_missing: int, documents_targeted: int) -> dict[str, Any]:
    per_dim = root / "06_ratings" / "mda_context_repair" / "per_dimension"
    if not per_dim.exists():
        return {"should_stop": False}
    records = []
    for path in per_dim.glob("*_parsed.json"):
        try:
            import json

            records.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            records.append({"status": "failed", "result": {"schema_validation_status": "invalid"}})
    if not records:
        return {"should_stop": False}
    success = sum(record.get("status") == "success" and record.get("result", {}).get("schema_validation_status") == "valid" for record in records)
    failed = len(records) - success
    if failed > 0:
        failure_reasons: dict[str, int] = {}
        sample_failures: list[dict[str, str]] = []
        for record in records:
            if record.get("status") == "success" and record.get("result", {}).get("schema_validation_status") == "valid":
                continue
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            reason = str(result.get("error_message") or result.get("error_type") or "unknown")
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            if len(sample_failures) < 5:
                sample_failures.append(
                    {
                        "document_id": str(record.get("document_id", "")),
                        "dimension_code": str(record.get("dimension_code", "")),
                        "error_type": str(result.get("error_type", "")),
                        "error_message": reason,
                    }
                )
        return {
            "should_stop": True,
            "status": "blocked",
            "reason": "existing_recovery_schema_or_json_failures",
            "missing_dimensions_expected": expected_missing,
            "missing_dimensions_recovered": success,
            "missing_dimensions_still_failed": max(0, expected_missing - success),
            "documents_targeted": documents_targeted,
            "documents_fully_recovered": 0,
            "schema_success_rate": round(success / len(records), 6),
            "evidence_match_rate": round(success / len(records), 6),
            "failure_reason_distribution": failure_reasons,
            "sample_failures": sample_failures,
        }
    if success >= expected_missing and expected_missing:
        return {
            "should_stop": True,
            "status": "success",
            "reason": "existing_recovery_outputs_complete",
            "missing_dimensions_expected": expected_missing,
            "missing_dimensions_recovered": success,
            "missing_dimensions_still_failed": 0,
            "documents_targeted": documents_targeted,
            "documents_fully_recovered": documents_targeted,
            "schema_success_rate": 1.0,
            "evidence_match_rate": 1.0,
        }
    return {"should_stop": False}


def _rate(numerator: Any, denominator: Any) -> float:
    try:
        denominator_f = float(denominator)
        return round(float(numerator) / denominator_f, 6) if denominator_f else 0.0
    except (TypeError, ValueError):
        return 0.0


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MD&A Missing Dimensions Recovery Report",
        "",
        f"- status: {summary.get('status')}",
        f"- reason: {summary.get('reason', '')}",
        f"- missing_dimensions_expected: {summary.get('missing_dimensions_expected')}",
        f"- missing_dimensions_recovered: {summary.get('missing_dimensions_recovered')}",
        f"- missing_dimensions_still_failed: {summary.get('missing_dimensions_still_failed')}",
        f"- documents_fully_recovered: {summary.get('documents_fully_recovered')}",
        f"- schema_success_rate: {summary.get('schema_success_rate')}",
        f"- evidence_match_rate: {summary.get('evidence_match_rate')}",
    ]
    if summary.get("failure_reason_distribution"):
        lines.extend(["", "## Failure Reason Distribution"])
        for reason, count in summary["failure_reason_distribution"].items():
            lines.append(f"- {reason}: {count}")
    if summary.get("sample_failures"):
        lines.extend(["", "## Sample Failures"])
        for item in summary["sample_failures"]:
            lines.append(f"- {item['document_id']} {item['dimension_code']}: {item['error_type']} | {item['error_message']}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover missing MD&A stability dimensions only when qwen3:8b runtime is healthy.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--max-scoring-retries", type=int, default=3)
    parser.add_argument("--llm-workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(recover_missing_mda_dimensions(args.root, args.model, None, args.max_scoring_retries, args.llm_workers), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
