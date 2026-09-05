from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_clients import OllamaClient
from run_single_dimension_scoring import SingleDimensionTask, score_one_dimension
from scoring_utils import (
    DEFAULT_ENV,
    SCORING_ROOT,
    dimensions_for,
    read_csv_rows,
    resolve_text_path,
    save_json,
    write_csv_rows,
)
from select_mda_dimension_context import select_document_dimension_contexts
from validate_prompt_context_budget import PromptBudgetConfig


RECORD_HEADERS = [
    "phase",
    "run_index",
    "document_id",
    "dimension_code",
    "status",
    "http_success",
    "inner_json_parse",
    "schema_success",
    "score_range_success",
    "evidence_match",
    "context_budget_pass",
    "error_type",
    "error_message",
    "raw_output_path",
    "parsed_output_path",
    "context_path",
]


def run_context_budget_smoke_test(
    *,
    root: Path = SCORING_ROOT,
    model: str = "qwen3:8b",
    base_url: str = DEFAULT_ENV["OLLAMA_BASE_URL"],
    repeat_count: int = 20,
    matrix_doc_count: int = 2,
    primary_dimension: str = "AR02",
    doc_ids: list[str] | None = None,
    mock: bool = False,
) -> dict[str, Any]:
    rows = _selected_rows(root, doc_ids, max(1, matrix_doc_count))
    if not rows:
        summary = {
            "overall_status": "blocked",
            "reason": "no_eligible_mda_documents",
            "allow_recover_55_dimensions": False,
            "allow_full_scoring": False,
            "records_path": "",
        }
        save_json(root / "08_reports" / "context_budget_smoke_test_summary.json", summary)
        return summary
    if not mock:
        status = OllamaClient(base_url=base_url, model=model, num_ctx=4096, num_predict=192).ensure_model_available()
        if not status.get("available") or not status.get("model_available"):
            summary = {
                "overall_status": "blocked",
                "reason": "ollama_or_qwen3_unavailable",
                "model_status": status,
                "allow_recover_55_dimensions": False,
                "allow_full_scoring": False,
                "records_path": "",
            }
            save_json(root / "08_reports" / "context_budget_smoke_test_summary.json", summary)
            return summary

    records: list[dict[str, Any]] = []
    primary_row = rows[0]
    for idx in range(1, repeat_count + 1):
        records.append(
            _run_smoke_call(
                root=root,
                row=primary_row,
                dimension_code=primary_dimension,
                phase="repeat_20",
                run_index=idx,
                output_name=f"mda_context_budget_smoke_test/repeat_{idx:02d}",
                model=model,
                base_url=base_url,
                mock=mock,
            )
        )
    matrix_rows = rows[:matrix_doc_count]
    run_index = 0
    for row in matrix_rows:
        for dimension in dimensions_for("mda"):
            run_index += 1
            records.append(
                _run_smoke_call(
                    root=root,
                    row=row,
                    dimension_code=dimension["code"],
                    phase="matrix_10",
                    run_index=run_index,
                    output_name="mda_context_budget_smoke_test/matrix_10",
                    model=model,
                    base_url=base_url,
                    mock=mock,
                )
            )

    out_dir = root / "08_reports"
    records_path = out_dir / "context_budget_smoke_test_records.csv"
    write_csv_rows(records_path, RECORD_HEADERS, records)
    summary = summarize_smoke_records(records)
    if mock:
        summary["overall_status"] = "mock_passed" if summary.get("overall_status") == "passed" else summary.get("overall_status")
        summary["allow_recover_55_dimensions"] = False
        summary["allow_full_scoring"] = False
        summary["reason"] = "mock_smoke_test_does_not_unlock_recovery_or_full_scoring"
    summary.update(
        {
            "repeat_count": repeat_count,
            "matrix_doc_count": len(matrix_rows),
            "records_path": str(records_path),
            "model_name": model if not mock else "mock_qwen_unavailable",
            "mock": mock,
        }
    )
    save_json(out_dir / "context_budget_smoke_test_summary.json", summary)
    return summary


def summarize_smoke_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [
        "http_success",
        "inner_json_parse",
        "schema_success",
        "score_range_success",
        "evidence_match",
        "context_budget_pass",
    ]
    total = len(records)
    summary: dict[str, Any] = {"total_calls": total}
    all_passed = total > 0
    for check in checks:
        passed = sum(_as_bool(row.get(check)) for row in records)
        summary[check] = f"{passed}/{total}"
        if passed != total:
            all_passed = False
    summary["overall_status"] = "passed" if all_passed else "blocked"
    summary["allow_recover_55_dimensions"] = all_passed
    summary["allow_full_scoring"] = all_passed
    return summary


def _run_smoke_call(
    *,
    root: Path,
    row: dict[str, str],
    dimension_code: str,
    phase: str,
    run_index: int,
    output_name: str,
    model: str,
    base_url: str,
    mock: bool,
) -> dict[str, Any]:
    doc_id = row.get("document_id", "")
    text_path = resolve_text_path(row, "mda", root)
    numeric_path = _numeric_path(root, doc_id)
    context_budget = _dimension_context_budget(4096)
    contexts = select_document_dimension_contexts(
        root=root,
        document_id=doc_id,
        text_path=text_path,
        numeric_check_path=numeric_path,
        context_budget=context_budget,
    )
    dimension = next(item for item in dimensions_for("mda") if item["code"] == dimension_code)
    context = contexts[dimension_code]
    task = SingleDimensionTask(
        document_id=doc_id,
        doc_type="mda",
        dimension_code=dimension_code,
        dimension_name=dimension["name"],
        dimension_definition=dimension,
        metadata=row,
        text_path=str(text_path),
        numeric_check_path=str(numeric_path) if numeric_path and numeric_path.exists() else "",
        output_name=output_name,
        root=str(root),
        model_name=model,
        base_url=base_url,
        temperature=0,
        seed=42,
        num_ctx=4096,
        timeout_seconds=180,
        max_retries=1,
        mock=mock,
        context_path=context.output_path,
        context_budget_passed=context.budget_passed,
    )
    result = score_one_dimension(task)
    record = _record_from_result(result, context.budget_passed)
    record.update({"phase": phase, "run_index": run_index})
    return record


def _record_from_result(result: Any, selector_budget_passed: bool) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    parsed_path = Path(result.parsed_output_path)
    if parsed_path.exists():
        try:
            parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            parsed = {}
    prompt_budget = parsed.get("prompt_budget", {}) if isinstance(parsed.get("prompt_budget"), dict) else {}
    post_call_budget = parsed.get("post_call_prompt_budget", {}) if isinstance(parsed.get("post_call_prompt_budget"), dict) else {}
    context_budget_pass = (
        selector_budget_passed
        and prompt_budget.get("budget_passed", True) is not False
        and post_call_budget.get("prompt_eval_passed", True) is not False
    )
    return {
        "phase": "",
        "run_index": "",
        "document_id": result.document_id,
        "dimension_code": result.dimension_code,
        "status": result.status,
        "http_success": bool(result.raw_output_path) and result.error_type != "request_failed",
        "inner_json_parse": result.parse_status == "parsed",
        "schema_success": result.schema_validation_status == "valid",
        "score_range_success": result.status == "success",
        "evidence_match": result.status == "success",
        "context_budget_pass": context_budget_pass,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "raw_output_path": result.raw_output_path,
        "parsed_output_path": result.parsed_output_path,
        "context_path": result.context_path,
    }


def _selected_rows(root: Path, doc_ids: list[str] | None, minimum_count: int) -> list[dict[str, str]]:
    rows = [row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    if doc_ids:
        wanted = set(doc_ids)
        rows = [row for row in rows if row.get("document_id") in wanted]
    existing = [row for row in rows if resolve_text_path(row, "mda", root).exists()]
    return existing[: max(minimum_count, 1)]


def _numeric_path(root: Path, doc_id: str) -> Path | None:
    for path in [
        root / "03_numeric_checks" / "mda_v2" / f"{doc_id}_numeric_checks.json",
        root / "03_numeric_checks" / "mda" / f"{doc_id}_numeric_checks.json",
    ]:
        if path.exists():
            return path
    return None


def _dimension_context_budget(num_ctx: int) -> int:
    config = PromptBudgetConfig(num_ctx=num_ctx)
    prompt_overhead_allowance = 1850
    return max(512, config.input_token_budget - prompt_overhead_allowance)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "passed", "success"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MD&A context-budget smoke test before recovery or full scoring.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--base-url", default=DEFAULT_ENV["OLLAMA_BASE_URL"])
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--matrix-doc-count", type=int, default=2)
    parser.add_argument("--primary-dimension", default="AR02")
    parser.add_argument("--doc-id", action="append", dest="doc_ids")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    summary = run_context_budget_smoke_test(
        root=args.root,
        model=args.model,
        base_url=args.base_url,
        repeat_count=args.repeat_count,
        matrix_doc_count=args.matrix_doc_count,
        primary_dimension=args.primary_dimension,
        doc_ids=args.doc_ids,
        mock=args.mock,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary.get("overall_status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
