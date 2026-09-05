from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from aggregate_dimensionwise_mda_scores import aggregate_dimensionwise_mda_scores
from json_stabilization import parse_model_json
from llm_clients import OllamaClient, OllamaClientError
from mda_numeric_checker import analyze_file
from run_scoring import mock_score_document, now_iso
from scoring_utils import (
    DEFAULT_ENV,
    SCORING_ROOT,
    append_jsonl,
    dimensions_for,
    ensure_directories,
    load_json,
    read_csv_rows,
    resolve_text_path,
    save_json,
)
from select_mda_dimension_context import select_document_dimension_contexts
from validate_prompt_context_budget import (
    PromptBudgetConfig,
    validate_post_call_prompt_eval,
    validate_prompt_context_budget,
)


class ContextBudgetRejected(RuntimeError):
    pass


@dataclass
class DimensionTask:
    document_id: str
    dimension_code: str
    dimension_name: str
    dimension_definition: dict[str, Any]
    metadata: dict[str, str]
    text_path: str
    numeric_check_path: str
    output_name: str
    root: str
    model_name: str
    base_url: str
    temperature: float
    seed: int
    num_ctx: int
    timeout_seconds: int
    max_retries: int
    mock: bool
    context_path: str = ""
    context_budget_passed: bool = True


@dataclass
class DimensionResult:
    document_id: str
    dimension_code: str
    status: str
    raw_output_path: str
    parsed_output_path: str
    parse_status: str
    schema_validation_status: str
    error_type: str
    error_message: str
    started_at: str
    finished_at: str
    duration_seconds: float
    attempts_used: int
    model_name: str
    numeric_check_path: str
    context_path: str = ""


def run_mda_dimensionwise_scoring(
    root: Path = SCORING_ROOT,
    doc_ids: list[str] | None = None,
    output_name: str = "mda_v2_dimensionwise",
    ratings_prefix: str = "mda_v2_dimensionwise",
    model: str = "qwen3:8b",
    llm_workers: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    mock: bool = False,
    max_retries: int = 2,
    timeout_seconds: int = 180,
    base_url: str = DEFAULT_ENV["OLLAMA_BASE_URL"],
    temperature: float = float(DEFAULT_ENV["OLLAMA_TEMPERATURE"]),
    seed: int = int(DEFAULT_ENV["OLLAMA_SEED"]),
    num_ctx: int = int(DEFAULT_ENV["OLLAMA_NUM_CTX"]),
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_directories(root)
    out_dir = root / "06_ratings" / output_name
    raw_dir = root / "05_raw_model_outputs" / output_name
    if overwrite:
        for path in [out_dir, raw_dir]:
            if path.exists():
                shutil.rmtree(path)
    (out_dir / "per_dimension").mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = [row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    if doc_ids is not None:
        wanted = set(doc_ids)
        registry_rows = [row for row in registry_rows if row.get("document_id") in wanted]
    if limit is not None:
        registry_rows = registry_rows[:limit]

    if not mock:
        client = OllamaClient(base_url=base_url, model=model, timeout=timeout_seconds, temperature=temperature, seed=seed, num_ctx=num_ctx)
        model_status = client.ensure_model_available()
        if not model_status.get("available") or not model_status.get("model_available"):
            summary = _unavailable_summary(output_name, ratings_prefix, registry_rows, model_status)
            save_json(out_dir / f"{ratings_prefix}_dimensionwise_summary.json", summary)
            return summary

    tasks, skipped = _build_dimension_tasks(
        root=root,
        rows=registry_rows,
        output_name=output_name,
        model=model,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        num_ctx=num_ctx,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        resume=resume,
        overwrite=overwrite,
        mock=mock,
    )
    progress_path = out_dir / f"{ratings_prefix}_dimensionwise_progress_log.jsonl"
    for row in skipped:
        append_jsonl(progress_path, row)

    results: list[DimensionResult] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, llm_workers)) as executor:
        futures = [executor.submit(score_one_dimension, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            append_jsonl(
                progress_path,
                {
                    "document_id": result.document_id,
                    "dimension_code": result.dimension_code,
                    "status": result.status,
                    "parse_status": result.parse_status,
                    "schema_validation_status": result.schema_validation_status,
                    "error_type": result.error_type or None,
                    "timestamp": result.finished_at,
                },
            )

    aggregate = aggregate_dimensionwise_mda_scores(
        root=root,
        output_name=output_name,
        ratings_prefix=ratings_prefix,
        doc_ids=[row.get("document_id", "") for row in registry_rows if row.get("document_id")],
        model_name=model if not mock else "mock_qwen_unavailable",
    )
    duration = round(time.perf_counter() - started, 3)
    summary = {
        **aggregate,
        "dimension_task_count": len(tasks),
        "dimension_success_count": sum(1 for item in results if item.status == "success"),
        "dimension_failure_count": sum(1 for item in results if item.status != "success"),
        "dimension_skipped_count": len(skipped),
        "json_parse_failures": sum(1 for item in results if item.parse_status != "parsed"),
        "schema_validation_failures": sum(1 for item in results if item.schema_validation_status != "valid"),
        "retry_count": sum(max(0, item.attempts_used - 1) for item in results),
        "llm_workers": llm_workers,
        "duration_seconds": duration,
        "model_name": model if not mock else "mock_qwen_unavailable",
        "llm_backend": "ollama" if not mock else "mock",
        "manual_template_only": False,
    }
    save_json(out_dir / f"{ratings_prefix}_dimensionwise_summary.json", summary)
    return summary


def score_one_dimension(task: DimensionTask) -> DimensionResult:
    root = Path(task.root)
    out_dir = root / "06_ratings" / task.output_name
    parsed_path = out_dir / "per_dimension" / f"{task.document_id}_{task.dimension_code}_parsed.json"
    started_clock = time.perf_counter()
    started_at = now_iso()
    raw_output_path = ""
    parse_status = "not_attempted"
    schema_status = "not_attempted"
    error_type = ""
    error_message = ""
    attempts_used = 0
    payload: dict[str, Any] | None = None
    budget_report: dict[str, Any] = {}
    post_call_budget_report: dict[str, Any] = {}
    try:
        text = Path(task.text_path).read_text(encoding="utf-8", errors="replace")
        context_text = _read_context_text(task, text)
        numeric_result = _load_optional_json(task.numeric_check_path)
        if task.mock:
            full_payload = mock_score_document("mda", task.document_id, text, numeric_result)
            dimension_payload = next(item for item in full_payload["dimension_scores"] if item["dimension_code"] == task.dimension_code)
            simple_payload = _simple_payload_from_dimension_payload(dimension_payload)
            validation_issues = _validate_simple_dimension_payload(simple_payload, task.dimension_code, text)
            payload = (
                _expand_simple_dimension_payload(simple_payload, task.dimension_definition, numeric_result, text)
                if not validation_issues
                else None
            )
            raw_output_path = str(
                _save_raw(
                    root,
                    task.output_name,
                    task.document_id,
                    task.dimension_code,
                    1,
                    {
                        "mock_mode": True,
                        "content": json.dumps(simple_payload, ensure_ascii=False),
                        "parsed_simple": simple_payload,
                    },
                )
            )
            attempts_used = 1
            parse_status = "parsed"
            validation_issues = validation_issues or _validate_dimension_payload(payload, task.dimension_definition, text)
            schema_status = "valid" if not validation_issues else "invalid"
            if validation_issues:
                error_type = "schema_validation_failed"
                error_message = ";".join(validation_issues)
                payload = None
        else:
            client = OllamaClient(
                base_url=task.base_url,
                model=task.model_name,
                timeout=task.timeout_seconds,
                temperature=task.temperature,
                seed=task.seed,
                num_ctx=task.num_ctx,
            )
            schema = _dimension_schema(task.dimension_definition)
            prompt, instructions_text, metadata_text = _dimension_prompt_parts(task, context_text, numeric_result)
            budget = validate_prompt_context_budget(
                instructions_text=instructions_text,
                schema_text=json.dumps(schema, ensure_ascii=False),
                selected_text=context_text,
                metadata_text=metadata_text,
                config=PromptBudgetConfig(num_ctx=task.num_ctx),
            )
            budget_report = budget.to_dict()
            if not budget.budget_passed or not task.context_budget_passed:
                error_type = "context_budget_failed"
                error_message = json.dumps(budget_report, ensure_ascii=False, sort_keys=True)
                payload = None
                attempts_used = 0
                raise ContextBudgetRejected(error_message)
            last_issues: list[str] = []
            for attempt in range(1, task.max_retries + 1):
                attempts_used = attempt
                try:
                    strict_prompt = prompt
                    if attempt > 1:
                        strict_prompt += "\n\nReturn exactly one JSON object. No markdown, no commentary, no extra keys."
                    result = client.score_document(strict_prompt, schema)
                    raw_output_path = str(
                        _save_raw(
                            root,
                            task.output_name,
                            task.document_id,
                            task.dimension_code,
                            attempt,
                            {
                                "model_name": task.model_name,
                                "base_url": task.base_url,
                                "temperature": task.temperature,
                                "seed": task.seed,
                                "num_ctx": task.num_ctx,
                                "attempt": attempt,
                                "started_at": started_at,
                                "finished_at": now_iso(),
                                "content": result.content,
                                "raw_response": result.raw_response,
                                "parsed_simple": result.parsed_json if isinstance(result.parsed_json, dict) else None,
                                "fallback_used": result.fallback_used,
                                "context_path": task.context_path,
                                "prompt_budget": budget_report,
                            },
                        )
                    )
                    post_call_budget = validate_post_call_prompt_eval(
                        result.raw_response,
                        config=PromptBudgetConfig(num_ctx=task.num_ctx),
                    )
                    post_call_budget_report = post_call_budget.to_dict()
                    if not post_call_budget.prompt_eval_passed:
                        error_type = "prompt_eval_too_high"
                        error_message = json.dumps(post_call_budget_report, ensure_ascii=False, sort_keys=True)
                        last_issues = [error_type]
                        payload = None
                        break
                    candidate, parse_status, schema_status, validation_issues = _parse_repair_validate_simple_payload(
                        client,
                        result.content,
                        schema,
                        task.dimension_code,
                        text,
                    )
                    schema_status = "valid" if not validation_issues else "invalid"
                    if isinstance(candidate, dict) and not validation_issues:
                        payload = _expand_simple_dimension_payload(candidate, task.dimension_definition, numeric_result, text)
                        validation_issues = _validate_dimension_payload(payload, task.dimension_definition, text)
                        schema_status = "valid" if not validation_issues else "invalid"
                    if isinstance(payload, dict) and not validation_issues:
                        break
                    last_issues = validation_issues or ["json_parse_failed"]
                except OllamaClientError as exc:
                    error_type = "request_failed"
                    error_message = str(exc)
                    last_issues = [str(exc)]
            if payload is None and not error_message:
                error_type = "manual_required"
                error_message = ";".join(last_issues)
        finished_at = now_iso()
        duration = round(time.perf_counter() - started_clock, 3)
        if payload is None:
            error_type = error_type or "manual_required"
            result = DimensionResult(
                task.document_id,
                task.dimension_code,
                "failed",
                raw_output_path,
                str(parsed_path),
                parse_status,
                schema_status,
                error_type,
                error_message,
                started_at,
                finished_at,
                duration,
                attempts_used,
                task.model_name if not task.mock else "mock_qwen_unavailable",
                task.numeric_check_path,
            )
            save_json(
                parsed_path,
                {
                    "document_id": task.document_id,
                    "doc_type": "mda",
                    "dimension_code": task.dimension_code,
                    "status": "failed",
                    "result": asdict(result),
                    "numeric_check_path": task.numeric_check_path,
                    "text_path": task.text_path,
                    "context_path": task.context_path,
                    "prompt_budget": budget_report,
                    "post_call_prompt_budget": post_call_budget_report,
                },
            )
            return result
        result = DimensionResult(
            task.document_id,
            task.dimension_code,
            "success",
            raw_output_path,
            str(parsed_path),
            "parsed",
            "valid",
            "",
            "",
            started_at,
            finished_at,
            duration,
            attempts_used,
            task.model_name if not task.mock else "mock_qwen_unavailable",
            task.numeric_check_path,
        )
        save_json(
            parsed_path,
            {
                "document_id": task.document_id,
                "doc_type": "mda",
                "dimension_code": task.dimension_code,
                "dimension_name": task.dimension_name,
                "status": "success",
                "payload": payload,
                "metadata": task.metadata,
                "text_path": task.text_path,
                "context_path": task.context_path,
                "numeric_check_path": task.numeric_check_path,
                "prompt_budget": budget_report,
                "post_call_prompt_budget": post_call_budget_report,
                "result": asdict(result),
            },
        )
        return result
    except Exception as exc:  # noqa: BLE001 - one dimension must not abort the batch.
        finished_at = now_iso()
        duration = round(time.perf_counter() - started_clock, 3)
        result = DimensionResult(
            task.document_id,
            task.dimension_code,
            "failed",
            raw_output_path,
            str(parsed_path),
            parse_status,
            schema_status,
            error_type or type(exc).__name__,
            error_message or str(exc),
            started_at,
            finished_at,
            duration,
            attempts_used,
            task.model_name if not task.mock else "mock_qwen_unavailable",
            task.numeric_check_path,
        )
        save_json(
            parsed_path,
            {
                "document_id": task.document_id,
                "doc_type": "mda",
                "dimension_code": task.dimension_code,
                "status": "failed",
                "result": asdict(result),
                "numeric_check_path": task.numeric_check_path,
                "text_path": task.text_path,
                "context_path": task.context_path,
                "prompt_budget": budget_report,
                "post_call_prompt_budget": post_call_budget_report,
            },
        )
        return result


def _build_dimension_tasks(
    *,
    root: Path,
    rows: list[dict[str, str]],
    output_name: str,
    model: str,
    base_url: str,
    temperature: float,
    seed: int,
    num_ctx: int,
    timeout_seconds: int,
    max_retries: int,
    resume: bool,
    overwrite: bool,
    mock: bool,
) -> tuple[list[DimensionTask], list[dict[str, Any]]]:
    tasks: list[DimensionTask] = []
    skipped: list[dict[str, Any]] = []
    dimensions = dimensions_for("mda")
    for row in rows:
        doc_id = row.get("document_id", "")
        if not doc_id:
            continue
        text_path = resolve_text_path(row, "mda", root)
        if not text_path.exists():
            continue
        numeric_path = _ensure_numeric_check(root, row, text_path)
        contexts = select_document_dimension_contexts(
            root=root,
            document_id=doc_id,
            text_path=text_path,
            numeric_check_path=numeric_path,
            context_budget=_dimension_context_budget(num_ctx),
        )
        for dimension in dimensions:
            code = dimension["code"]
            parsed_path = root / "06_ratings" / output_name / "per_dimension" / f"{doc_id}_{code}_parsed.json"
            if overwrite and parsed_path.exists():
                parsed_path.unlink()
            if resume and not overwrite and _dimension_completed(parsed_path):
                skipped.append(
                    {
                        "document_id": doc_id,
                        "dimension_code": code,
                        "status": "skipped",
                        "timestamp": now_iso(),
                    }
                )
                continue
            tasks.append(
                DimensionTask(
                    document_id=doc_id,
                    dimension_code=code,
                    dimension_name=dimension["name"],
                    dimension_definition=dimension,
                    metadata=row,
                    text_path=str(text_path),
                    numeric_check_path=str(numeric_path),
                    output_name=output_name,
                    root=str(root),
                    model_name=model,
                    base_url=base_url,
                    temperature=temperature,
                    seed=seed,
                    num_ctx=num_ctx,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                    mock=mock,
                    context_path=contexts[code].output_path,
                    context_budget_passed=contexts[code].budget_passed,
                )
            )
    return tasks, skipped


def _ensure_numeric_check(root: Path, row: dict[str, str], text_path: Path) -> Path:
    doc_id = row.get("document_id", "")
    candidates = [
        root / "03_numeric_checks" / "mda_v2" / f"{doc_id}_numeric_checks.json",
        root / "03_numeric_checks" / "mda" / f"{doc_id}_numeric_checks.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    output_path = root / "03_numeric_checks" / "mda" / f"{doc_id}_numeric_checks.json"
    result = analyze_file(text_path, document_id=doc_id)
    save_json(output_path, result)
    return output_path


def _dimension_context_budget(num_ctx: int) -> int:
    config = PromptBudgetConfig(num_ctx=num_ctx)
    prompt_overhead_allowance = 1850
    return max(512, config.input_token_budget - prompt_overhead_allowance)


def _read_context_text(task: DimensionTask, fallback_text: str) -> str:
    if task.context_path:
        path = Path(task.context_path)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return fallback_text


def _dimension_completed(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    return payload.get("status") == "success" and payload.get("result", {}).get("schema_validation_status") == "valid"


def _save_raw(root: Path, output_name: str, document_id: str, dimension_code: str, attempt: int, payload: dict[str, Any]) -> Path:
    path = root / "05_raw_model_outputs" / output_name / f"{document_id}_{dimension_code}_attempt_{attempt}.json"
    save_json(path, payload)
    return path


def _parse_repair_validate_simple_payload(
    client: OllamaClient,
    raw_text: str,
    schema: dict[str, Any],
    dimension_code: str,
    text: str,
) -> tuple[dict[str, Any] | None, str, str, list[str]]:
    candidate, parse_status, issues = _parse_validate_simple_payload(raw_text, dimension_code, text)
    if isinstance(candidate, dict) and not issues:
        return candidate, parse_status, "valid", []
    if raw_text.strip():
        repaired = client.repair_json(raw_text, schema)
        repaired_candidate, repaired_parse_status, repaired_issues = _parse_validate_simple_payload(
            repaired.content,
            dimension_code,
            text,
        )
        if isinstance(repaired_candidate, dict) and not repaired_issues:
            return repaired_candidate, repaired_parse_status, "valid", []
        candidate = repaired_candidate if isinstance(repaired_candidate, dict) else candidate
        parse_status = repaired_parse_status
        issues = repaired_issues or issues
    return (candidate if isinstance(candidate, dict) else None), parse_status, "invalid", issues or ["json_parse_failed"]


def _parse_validate_simple_payload(raw_text: str, dimension_code: str, text: str) -> tuple[dict[str, Any] | None, str, list[str]]:
    parsed = parse_model_json(raw_text)
    if not parsed.get("ok"):
        return None, "parse_failed", [str(parsed.get("failure_type") or "json_parse_failed")]
    candidate = parsed.get("payload")
    issues = _validate_simple_dimension_payload(candidate, dimension_code, text)
    return (candidate if isinstance(candidate, dict) else None), "parsed", issues


def _dimension_schema(dimension: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["score", "evidence", "reason"],
        "properties": {
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "evidence": {"type": "string", "pattern": "^\\[?MDA_P\\d{3}\\]?$"},
            "reason": {"type": "string", "minLength": 1, "maxLength": 240},
        },
        "additionalProperties": False,
    }


def _dimension_prompt(
    task: DimensionTask,
    selected_context_text: str,
    numeric_result: dict[str, Any] | None,
) -> str:
    prompt, _instructions, _metadata = _dimension_prompt_parts(task, selected_context_text, numeric_result)
    return prompt


def _dimension_prompt_parts(
    task: DimensionTask,
    selected_context_text: str,
    numeric_result: dict[str, Any] | None,
) -> tuple[str, str, str]:
    instructions = (
        "Score exactly ONE MD&A dimension for this research pipeline.\n"
        "Return exactly one JSON object and no markdown, arrays, nested objects, or extra text.\n"
        "Do not score or mention the other AR dimensions.\n"
        "Output only these fields: score, evidence, reason.\n"
        "score must be an integer from 1 to 5.\n"
        "evidence must be a single locator like MDA_P001.\n"
        "reason must be short, plain text, and under 240 characters.\n"
        "Use only the selected dimension-specific context below, not assumptions about omitted text.\n"
        "For AR02, numeric_check_summary is authoritative. LLM must not calculate percentages, growth rates, "
        "or totals. Do not perform arithmetic. Only explain the Python numeric_checker output in short text; "
        "Python will set the final AR02 score from numeric_check_summary.\n\n"
        "Valid response example: {\"score\": 3, \"evidence\": \"MDA_P001\", \"reason\": \"short reason\"}\n"
    )
    metadata = {
        "document_id": task.document_id,
        "registry": _compact_registry_metadata(task.metadata),
        "dimension_to_score": task.dimension_definition,
        "numeric_check_summary": _compact_numeric_summary(numeric_result),
    }
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    prompt = (
        instructions
        + "\nDOCUMENT_METADATA:\n"
        + metadata_text
        + "\n\nSELECTED_DIMENSION_CONTEXT:\n"
        + selected_context_text
    )
    return prompt, instructions, metadata_text


def _simple_payload_from_dimension_payload(payload: dict[str, Any]) -> dict[str, Any]:
    locator = _normalize_simple_evidence(str(payload.get("evidence_locator", ""))) or "MDA_P001"
    return {
        "score": _safe_score(payload.get("raw_score"), default=3),
        "evidence": locator,
        "reason": _short_text(str(payload.get("reason") or payload.get("comment_short") or "Evidence supports this score."), 240),
    }


def _compact_registry_metadata(metadata: dict[str, str]) -> dict[str, str]:
    allowed = ["document_id", "doc_type", "stock_code", "ticker", "company_name", "report_year"]
    return {key: str(metadata.get(key, "")) for key in allowed if metadata.get(key)}


def _compact_numeric_summary(numeric_result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(numeric_result, dict):
        return {}
    keys = [
        "document_id",
        "overall_numeric_consistency",
        "num_financial_mentions",
        "num_calculable_changes",
        "num_consistent_checks",
        "num_material_discrepancies",
        "num_minor_discrepancies",
        "num_severe_direction_conflicts",
        "has_revenue_discussion",
        "has_profit_discussion",
        "has_cost_discussion",
        "has_margin_discussion",
        "has_segment_discussion",
    ]
    return {key: numeric_result.get(key) for key in keys if key in numeric_result}


def _validate_simple_dimension_payload(payload: Any, dimension_code: str, text: str) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload_not_object"]
    allowed = {"score", "evidence", "reason"}
    issues: list[str] = []
    for key, value in payload.items():
        if key not in allowed:
            issues.append(f"unexpected_field_{key}")
        if isinstance(value, (dict, list)):
            issues.append(f"nested_value_{key}")
    for field in sorted(allowed):
        if field not in payload:
            issues.append(f"missing_field_{field}")
    try:
        payload["score"] = int(payload.get("score"))
    except (TypeError, ValueError):
        issues.append("score_not_integer")
    if payload.get("score") not in {1, 2, 3, 4, 5}:
        issues.append("score_out_of_range")
    evidence = _normalize_simple_evidence(str(payload.get("evidence", "")))
    payload["evidence"] = evidence
    if not evidence:
        issues.append("empty_evidence")
    elif f"[{evidence}]" not in text:
        issues.append("evidence_locator_not_found")
    reason = str(payload.get("reason", "")).strip()
    payload["reason"] = _short_text(reason, 240)
    if not reason:
        issues.append("empty_reason")
    if len(reason) > 240:
        issues.append("reason_too_long")
    if "calculate" in reason.lower() and re.search(r"\d+\s*[+\-*/]\s*\d+", reason):
        issues.append("reason_contains_arithmetic")
    return issues


def _expand_simple_dimension_payload(
    simple_payload: dict[str, Any],
    dimension: dict[str, Any],
    numeric_result: dict[str, Any] | None,
    text: str = "",
) -> dict[str, Any]:
    code = str(dimension.get("code", ""))
    score = _safe_score(simple_payload.get("score"), default=3)
    evidence = _normalize_simple_evidence(str(simple_payload.get("evidence", ""))) or "MDA_P001"
    reason = _short_text(str(simple_payload.get("reason", "")).strip() or "Evidence supports this score.", 240)
    if code == "AR02":
        score = _ar02_score_from_numeric(numeric_result)
        reason = _short_text(f"Python numeric checker: {reason}", 240)
    evidence_locator = f"[{evidence}]"
    evidence_text = _evidence_text_from_locator(text, evidence_locator) or f"{evidence_locator} {reason}"
    return {
        "dimension_code": code,
        "dimension_name": str(dimension.get("name", "")),
        "raw_score": score,
        "reason": reason,
        "evidence_locator": evidence_locator,
        "evidence_text_short": _short_text(evidence_text, 240),
        "confidence_level": "medium",
        "missing_type": "none",
        "comment_short": reason,
    }


def _ar02_score_from_numeric(numeric_result: dict[str, Any] | None) -> int:
    numeric_result = numeric_result or {}
    severe_conflicts = _safe_int(numeric_result.get("num_severe_direction_conflicts"))
    material = _safe_int(numeric_result.get("num_material_discrepancies"))
    consistency = str(numeric_result.get("overall_numeric_consistency", "")).strip().lower()
    if severe_conflicts > 0 or consistency in {"problematic", "severe", "failed"}:
        return 1
    if material > 0 or consistency in {"weak", "poor"}:
        return 2
    if consistency in {"strong", "clean", "good"}:
        return 4
    return 3


def _normalize_simple_evidence(locator: str) -> str:
    match = re.search(r"MDA_P\d{3}", locator)
    return match.group(0) if match else ""


def _safe_score(value: Any, default: int = 3) -> int:
    score = _safe_int(value, default)
    return score if score in {1, 2, 3, 4, 5} else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _short_text(value: str, limit: int) -> str:
    text = " ".join(str(value).split())
    return text[:limit].rstrip()


def _validate_dimension_payload(payload: Any, dimension: dict[str, Any], text: str) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload_not_object"]
    required = [
        "dimension_code",
        "dimension_name",
        "raw_score",
        "reason",
        "evidence_locator",
        "evidence_text_short",
        "confidence_level",
        "missing_type",
        "comment_short",
    ]
    issues = [f"missing_field_{field}" for field in required if field not in payload]
    if payload.get("dimension_code") != dimension["code"]:
        issues.append("dimension_code_mismatch")
    if str(payload.get("dimension_name", "")).strip() != dimension["name"]:
        payload["dimension_name"] = dimension["name"]
    try:
        payload["raw_score"] = int(payload.get("raw_score"))
    except (TypeError, ValueError):
        issues.append("raw_score_not_integer")
    if payload.get("raw_score") not in {1, 2, 3, 4, 5}:
        issues.append("raw_score_out_of_range")
    if payload.get("confidence_level") not in {"high", "medium", "low"}:
        issues.append("invalid_confidence_level")
    if payload.get("missing_type") not in {"none", "structural_missing", "operational_missing"}:
        issues.append("invalid_missing_type")
    locator = _normalize_locator(str(payload.get("evidence_locator", "")))
    payload["evidence_locator"] = locator
    if not locator:
        issues.append("empty_evidence_locator")
    elif not _locator_exists(locator, text):
        issues.append("evidence_locator_not_found")
    if not str(payload.get("evidence_text_short", "")).strip():
        issues.append("empty_evidence_text_short")
    return issues


def _normalize_locator(locator: str) -> str:
    tokens = re.findall(r"\[?MDA_P\d{3}\]?", locator)
    if not tokens:
        return locator.strip()
    return ", ".join(f"[{token.strip('[]')}]" for token in tokens)


def _locator_exists(locator: str, text: str) -> bool:
    tokens = re.findall(r"\[MDA_P\d{3}\]", locator)
    return bool(tokens) and all(token in text for token in tokens)


def _evidence_text_from_locator(text: str, locator: str) -> str:
    match = re.search(r"\[(MDA_P\d{3})\]", locator)
    if not match:
        return ""
    paragraph_id = match.group(1)
    marker = f"[{paragraph_id}]"
    start = text.find(marker)
    if start < 0:
        return ""
    next_match = re.search(r"\n\[MDA_P\d{3}\]\s*\n", text[start + len(marker) :])
    end = start + len(marker) + next_match.start() if next_match else len(text)
    paragraph = text[start:end].strip()
    paragraph = re.sub(r"(?m)^\[SECTION:\s*[^\]]+\]\s*$", "", paragraph).strip()
    return paragraph


def _load_optional_json(path_text: str) -> dict[str, Any] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:
        return None


def _unavailable_summary(output_name: str, ratings_prefix: str, rows: list[dict[str, str]], model_status: dict[str, Any]) -> dict[str, Any]:
    return {
        "doc_type": "mda",
        "scoring_strategy": "dimensionwise",
        "output_name": output_name,
        "ratings_prefix": ratings_prefix,
        "selected_count": len(rows),
        "success_count": 0,
        "failure_count": len(rows),
        "dimension_task_count": len(rows) * 5,
        "dimension_success_count": 0,
        "dimension_failure_count": len(rows) * 5,
        "json_parse_failures": 0,
        "schema_validation_failures": 0,
        "ollama_available": model_status.get("available", False),
        "model_available": model_status.get("model_available", False),
        "model_name": DEFAULT_ENV["OLLAMA_MODEL"],
        "llm_backend": "ollama",
        "manual_template_only": True,
        "message": model_status.get("error", "Local Ollama qwen3:8b unavailable."),
    }


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run dimension-wise MD&A scoring with local qwen3:8b.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--output-name", default="mda_v2_dimensionwise")
    parser.add_argument("--ratings-prefix", default="mda_v2_dimensionwise")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_mda_dimensionwise_scoring(
                root=args.root,
                output_name=args.output_name,
                ratings_prefix=args.ratings_prefix,
                model=args.model,
                llm_workers=args.llm_workers,
                resume=args.resume,
                overwrite=args.overwrite,
                mock=args.mock,
                limit=args.limit,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
