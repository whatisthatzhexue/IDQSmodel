from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rebuild_mda_stability_metrics import rebuild_mda_stability_metrics
from repair_ollama_runtime import repair_ollama_runtime
from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows
from stability_common import write_markdown


FAILED_HEADERS = ["document_id", "ticker", "company_name", "report_year", "status", "reason", "recommended_action"]


def run_mda_final_full_scoring(
    root: Path = SCORING_ROOT,
    model: str = "qwen3:8b",
    runtime_status: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    llm_workers: int = 1,
) -> dict[str, Any]:
    eligible = [row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    runtime_status = runtime_status or repair_ollama_runtime(root=root, model=model, max_repair_rounds=0)
    metrics = metrics or rebuild_mda_stability_metrics(root)
    gate_passed = bool(
        runtime_status.get("runtime_healthy")
        and metrics.get("operational_completion_rate", 0) >= 0.95
        and (metrics.get("repeatability_pass_rate") is not None and metrics.get("repeatability_pass_rate") >= 0.85)
        and metrics.get("schema_success_rate", 0) >= 0.98
        and metrics.get("evidence_match_rate", 0) >= 0.98
    )
    if not gate_passed:
        reasons = []
        if not runtime_status.get("runtime_healthy"):
            reasons.append("ollama_runtime_unhealthy")
        if metrics.get("operational_completion_rate", 0) < 0.95:
            reasons.append("mda_operational_completion_rate_below_0.95")
        if metrics.get("repeatability_pass_rate") is None or metrics.get("repeatability_pass_rate", 0) < 0.85:
            reasons.append("mda_repeatability_pass_rate_below_0.85_or_missing")
        if metrics.get("schema_success_rate", 0) < 0.98:
            reasons.append("mda_schema_success_rate_below_0.98")
        if metrics.get("evidence_match_rate", 0) < 0.98:
            reasons.append("mda_evidence_match_rate_below_0.98")
        failed = [
            {
                "document_id": row.get("document_id", ""),
                "ticker": row.get("ticker", ""),
                "company_name": row.get("company_name", ""),
                "report_year": row.get("report_year", ""),
                "status": "not_scored",
                "reason": ";".join(reasons),
                "recommended_action": "repair runtime and pass MD&A stability gate before final full scoring",
            }
            for row in eligible
        ]
        out_dir = root / "06_ratings" / "mda_final_full"
        write_csv_rows(out_dir / "mda_final_failed_cases.csv", FAILED_HEADERS, failed)
        summary = {
            "status": "blocked",
            "reason": ";".join(reasons),
            "eligible_mda_count": len(eligible),
            "complete_mda_count": 0,
            "incomplete_mda_count": len(eligible),
            "excluded_mda_count": 0,
            "full_scoring_coverage_rate": 0.0,
        }
        _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_final_full_scoring_report.md", summary)
        return summary
    smoke_gate = _context_budget_smoke_gate(root)
    if not smoke_gate.get("passed"):
        out_dir = root / "06_ratings" / "mda_final_full"
        failed = [
            {
                "document_id": row.get("document_id", ""),
                "ticker": row.get("ticker", ""),
                "company_name": row.get("company_name", ""),
                "report_year": row.get("report_year", ""),
                "status": "not_scored",
                "reason": "context_budget_smoke_test_not_passed",
                "recommended_action": "run scripts/run_context_budget_smoke_test.py and pass all contract checks before final full scoring",
            }
            for row in eligible
        ]
        write_csv_rows(out_dir / "mda_final_failed_cases.csv", FAILED_HEADERS, failed)
        summary = {
            "status": "blocked",
            "reason": "context_budget_smoke_test_not_passed",
            "eligible_mda_count": len(eligible),
            "complete_mda_count": 0,
            "incomplete_mda_count": len(eligible),
            "excluded_mda_count": 0,
            "full_scoring_coverage_rate": 0.0,
            "smoke_summary_path": str(smoke_gate.get("path", "")),
        }
        _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_final_full_scoring_report.md", summary)
        return summary
    result = run_single_dimension_scoring(
        "mda",
        root=root,
        output_name="mda_final_full",
        ratings_prefix="mda_final",
        model=model,
        llm_workers=llm_workers,
        resume=True,
        overwrite=False,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
    )
    complete = int(result.get("success_count", 0))
    summary = {
        "status": "success",
        "eligible_mda_count": len(eligible),
        "complete_mda_count": complete,
        "incomplete_mda_count": max(0, len(eligible) - complete),
        "excluded_mda_count": 0,
        "full_scoring_coverage_rate": round(complete / len(eligible), 6) if eligible else 0.0,
        "summary": result,
    }
    _write_report(root / "PAPER_OUTPUT" / "00_status" / "mda_final_full_scoring_report.md", summary)
    return summary


def _context_budget_smoke_gate(root: Path) -> dict[str, Any]:
    path = root / "08_reports" / "context_budget_smoke_test_summary.json"
    if not path.exists():
        return {"passed": False, "path": str(path)}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"passed": False, "path": str(path)}
    return {
        "passed": bool(summary.get("overall_status") == "passed" and summary.get("allow_full_scoring") is True),
        "path": str(path),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# MD&A Final Full Scoring Report", ""]
    for key in ["status", "reason", "eligible_mda_count", "complete_mda_count", "incomplete_mda_count", "excluded_mda_count", "full_scoring_coverage_rate"]:
        if key in summary:
            lines.append(f"- {key}: {summary[key]}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final full MD&A single-dimension scoring only after gates pass.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--llm-workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_mda_final_full_scoring(args.root, args.model, None, None, args.llm_workers), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
