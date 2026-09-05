from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from repair_ollama_runtime import repair_ollama_runtime
from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows
from stability_common import safe_float, write_markdown


MDA_DIMS = ["AR01", "AR02", "AR03", "AR04", "AR05"]
CASE_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "report_year",
    "valid_pair",
    "stable_pair",
    "total_score_a",
    "total_score_b",
    "abs_total_score_diff",
    "max_dimension_diff",
    "dimension_diff_ge_2",
    "manual_review_required",
    "review_reasons",
]


def run_mda_final_repeatability_test(
    root: Path = SCORING_ROOT,
    model: str = "qwen3:8b",
    runtime_status: dict[str, Any] | None = None,
    llm_workers: int = 1,
) -> dict[str, Any]:
    selected_ids = [row.get("document_id", "") for row in read_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv") if row.get("document_id")]
    runtime_status = runtime_status or repair_ollama_runtime(root=root, model=model, max_repair_rounds=0)
    if not runtime_status.get("runtime_healthy"):
        summary = _blocked_summary(len(selected_ids), "ollama_runtime_unhealthy")
        _write_outputs(root, summary, [])
        return summary
    existing_cases, existing_summary = _compare_runs(root, selected_ids)
    if _complete_existing_pairs(existing_summary, len(selected_ids)):
        existing_summary["reused_existing_outputs"] = True
        _write_outputs(root, existing_summary, existing_cases)
        return existing_summary
    run_a = run_single_dimension_scoring(
        "mda",
        root=root,
        doc_ids=selected_ids,
        output_name="mda_repeatability_final/run_a",
        ratings_prefix="mda_repeatability_run_a",
        model=model,
        llm_workers=llm_workers,
        resume=True,
        overwrite=False,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
    )
    run_b = run_single_dimension_scoring(
        "mda",
        root=root,
        doc_ids=selected_ids,
        output_name="mda_repeatability_final/run_b",
        ratings_prefix="mda_repeatability_run_b",
        model=model,
        llm_workers=llm_workers,
        resume=False,
        overwrite=False,
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
    )
    cases, summary = _compare_runs(root, selected_ids)
    summary["run_a"] = run_a
    summary["run_b"] = run_b
    _write_outputs(root, summary, cases)
    return summary


def _complete_existing_pairs(summary: dict[str, Any], expected_documents: int) -> bool:
    return bool(expected_documents and summary.get("valid_paired_documents") == expected_documents)


def _compare_runs(root: Path, selected_ids: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_a = {row.get("document_id", ""): row for row in read_csv_rows(root / "06_ratings" / "mda_repeatability_final" / "run_a" / "mda_repeatability_run_a_document_scores.csv")}
    rows_b = {row.get("document_id", ""): row for row in read_csv_rows(root / "06_ratings" / "mda_repeatability_final" / "run_b" / "mda_repeatability_run_b_document_scores.csv")}
    cases = []
    valid = 0
    stable = 0
    exact = 0
    within_one = 0
    dim_total = 0
    abs_diffs: list[float] = []
    for doc_id in selected_ids:
        a = rows_a.get(doc_id, {})
        b = rows_b.get(doc_id, {})
        valid_pair = bool(a and b and all(a.get(code) and b.get(code) for code in MDA_DIMS))
        reasons = []
        if not valid_pair:
            reasons.append("missing_complete_pair")
        dim_diffs = [abs(safe_float(a.get(code)) - safe_float(b.get(code))) for code in MDA_DIMS] if valid_pair else []
        abs_total = abs(safe_float(a.get("total_score_100")) - safe_float(b.get("total_score_100"))) if valid_pair else 0.0
        max_dim = max(dim_diffs) if dim_diffs else 0.0
        dim_ge_2 = sum(diff >= 2 for diff in dim_diffs)
        if valid_pair:
            valid += 1
            dim_total += len(dim_diffs)
            exact += sum(diff == 0 for diff in dim_diffs)
            within_one += sum(diff <= 1 for diff in dim_diffs)
            abs_diffs.append(abs_total)
            if dim_ge_2:
                reasons.append("dimension_diff_ge_2")
            if abs_total > 5:
                reasons.append("total_score_abs_diff_gt_5")
        stable_pair = valid_pair and not reasons
        stable += int(stable_pair)
        cases.append(
            {
                "document_id": doc_id,
                "ticker": a.get("ticker") or b.get("ticker", ""),
                "company_name": a.get("company_name") or b.get("company_name", ""),
                "report_year": a.get("report_year") or b.get("report_year", ""),
                "valid_pair": str(valid_pair).lower(),
                "stable_pair": str(stable_pair).lower(),
                "total_score_a": a.get("total_score_100", ""),
                "total_score_b": b.get("total_score_100", ""),
                "abs_total_score_diff": round(abs_total, 6),
                "max_dimension_diff": round(max_dim, 6),
                "dimension_diff_ge_2": dim_ge_2,
                "manual_review_required": str(bool(reasons)).lower(),
                "review_reasons": ";".join(reasons),
            }
        )
    summary = {
        "status": "success",
        "expected_documents": len(selected_ids),
        "operationally_complete_documents": valid,
        "operational_completion_rate": round(valid / len(selected_ids), 6) if selected_ids else 0.0,
        "valid_paired_documents": valid,
        "stable_paired_documents": stable,
        "repeatability_pass_rate": round(stable / valid, 6) if valid else None,
        "dimension_exact_agreement": round(exact / dim_total, 6) if dim_total else None,
        "dimension_within_one_agreement": round(within_one / dim_total, 6) if dim_total else None,
        "mean_absolute_total_score_difference": round(statistics.mean(abs_diffs), 6) if abs_diffs else None,
        "max_total_score_difference": max(abs_diffs) if abs_diffs else None,
        "manual_review_required_count": sum(row["manual_review_required"] == "true" for row in cases),
    }
    return cases, summary


def _blocked_summary(expected: int, reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "reason": reason,
        "expected_documents": expected,
        "operationally_complete_documents": 0,
        "operational_completion_rate": 0.0,
        "valid_paired_documents": 0,
        "stable_paired_documents": 0,
        "repeatability_pass_rate": None,
        "dimension_exact_agreement": None,
        "dimension_within_one_agreement": None,
        "mean_absolute_total_score_difference": None,
        "max_total_score_difference": None,
        "manual_review_required_count": 0,
    }


def _write_outputs(root: Path, summary: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    out_dir = root / "PAPER_OUTPUT" / "00_status"
    write_csv_rows(out_dir / "mda_final_repeatability_cases.csv", CASE_HEADERS, cases)
    lines = ["# MD&A Final Repeatability Report", ""]
    for key in [
        "status",
        "reason",
        "expected_documents",
        "operationally_complete_documents",
        "operational_completion_rate",
        "valid_paired_documents",
        "stable_paired_documents",
        "repeatability_pass_rate",
        "dimension_exact_agreement",
        "dimension_within_one_agreement",
        "mean_absolute_total_score_difference",
        "max_total_score_difference",
        "manual_review_required_count",
    ]:
        if key in summary:
            lines.append(f"- {key}: {summary[key]}")
    write_markdown(out_dir / "mda_final_repeatability_report.md", lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final MD&A Run A/B repeatability test.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--llm-workers", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run_mda_final_repeatability_test(args.root, args.model, None, args.llm_workers), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
