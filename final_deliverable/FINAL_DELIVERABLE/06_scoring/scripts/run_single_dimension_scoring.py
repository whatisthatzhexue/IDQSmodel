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

from aggregate_scores import aggregate_single_dimension_scores
from llm_clients import OllamaClient, OllamaClientError
from mda_numeric_checker import analyze_file
from repair_json_pipeline import repair_then_validate_json
from run_scoring import mock_score_document, now_iso
from scoring_utils import (
    DEFAULT_ENV,
    SCORING_ROOT,
    append_jsonl,
    dimensions_for,
    ensure_directories,
    load_json,
    read_csv_rows,
    registry_path,
    resolve_text_path,
    save_json,
)
from select_mda_dimension_context import select_document_dimension_contexts
from validate_prompt_context_budget import (
    PromptBudgetConfig,
    validate_post_call_prompt_eval,
    validate_prompt_context_budget,
)
from validate_json_outputs import normalize_simple_evidence, simple_dimension_schema, validate_json_outputs, validate_simple_dimension_json


@dataclass
class SingleDimensionTask:
    document_id: str
    doc_type: str
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
class SingleDimensionResult:
    document_id: str
    doc_type: str
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


def run_single_dimension_scoring(
    doc_type: str,
    root: Path = SCORING_ROOT,
    doc_ids: list[str] | None = None,
    output_name: str | None = None,
    ratings_prefix: str | None = None,
    model: str = "qwen3:8b",
    llm_workers: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    mock: bool = False,
    max_retries: int = 3,
    timeout_seconds: int = 180,
    base_url: str = DEFAULT_ENV["OLLAMA_BASE_URL"],
    temperature: float = float(DEFAULT_ENV["OLLAMA_TEMPERATURE"]),
    seed: int = int(DEFAULT_ENV["OLLAMA_SEED"]),
    num_ctx: int = int(DEFAULT_ENV["OLLAMA_NUM_CTX"]),
    limit: int | None = None,
    dimension_codes: list[str] | None = None,
) -> dict[str, Any]:
    if doc_type not in {"mda", "news"}:
        raise ValueError("doc_type must be mda or news")
    ensure_directories(root)
    output_name = output_name or f"{doc_type}_single_dimension"
    ratings_prefix = ratings_prefix or f"{doc_type}_single_dimension"
    out_dir = root / "06_ratings" / output_name
    raw_dir = root / "05_raw_model_outputs" / output_name
    if overwrite:
        for path in [out_dir, raw_dir]:
            if path.exists():
                shutil.rmtree(path)
    (out_dir / "per_dimension").mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = [row for row in read_csv_rows(registry_path(doc_type, root)) if row.get("include_flag", "").lower() == "yes"]
    if doc_ids is not None:
        wanted = set(doc_ids)
        registry_rows = [row for row in registry_rows if row.get("document_id") in wanted]
    if limit is not None:
        registry_rows = registry_rows[:limit]

    if not mock:
        client = OllamaClient(base_url=base_url, model=model, timeout=timeout_seconds, temperature=temperature, seed=seed, num_ctx=num_ctx)
        model_status = client.ensure_model_available()
        if not model_status.get("available") or not model_status.get("model_available"):
            summary = _unavailable_summary(doc_type, output_name, ratings_prefix, registry_rows, model_status)
            save_json(out_dir / f"{ratings_prefix}_single_dimension_summary.json", summary)
            return summary

    tasks, skipped = _build_tasks(
        doc_type=doc_type,
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
        dimension_codes=dimension_codes,
    )
    progress_path = out_dir / f"{ratings_prefix}_single_dimension_progress_log.jsonl"
    for row in skipped:
        append_jsonl(progress_path, row)

    results: list[SingleDimensionResult] = []
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
                    "doc_type": result.doc_type,
                    "dimension_code": result.dimension_code,
                    "status": result.status,
                    "parse_status": result.parse_status,
                    "schema_validation_status": result.schema_validation_status,
                    "error_type": result.error_type or None,
                    "timestamp": result.finished_at,
                },
            )

    aggregate = aggregate_single_dimension_scores(
        doc_type=doc_type,
        root=root,
        output_name=output_name,
        ratings_prefix=ratings_prefix,
        doc_ids=[row.get("document_id", "") for row in registry_rows if row.get("document_id")],
        model_name=model if not mock else "mock_qwen_unavailable",
    )
    validation = validate_json_outputs(root, output_name, doc_type)
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
        "json_validation_summary": validation,
    }
    save_json(out_dir / f"{ratings_prefix}_single_dimension_summary.json", summary)
    return summary


def score_one_dimension(task: SingleDimensionTask) -> SingleDimensionResult:
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
        text = _normalize_text(Path(task.text_path).read_text(encoding="utf-8", errors="replace"))
        scoring_text = _read_context_text(task, text) if task.doc_type == "mda" else text
        numeric_result = _load_optional_json(task.numeric_check_path)
        if task.mock:
            simple_payload = _mock_simple_payload(task, text, numeric_result)
            validation_issues = validate_simple_dimension_json(simple_payload, task.doc_type, task.dimension_code, text)
            payload = _expand_simple_payload(simple_payload, task, numeric_result, text) if not validation_issues else None
            raw_output_path = str(
                _save_raw(
                    root,
                    task.output_name,
                    task.document_id,
                    task.dimension_code,
                    1,
                    {"mock_mode": True, "content": json.dumps(simple_payload, ensure_ascii=False), "parsed_simple": simple_payload},
                )
            )
            attempts_used = 1
            parse_status = "parsed"
            schema_status = "valid" if not validation_issues else "invalid"
            if validation_issues:
                error_type = "manual_required"
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
            schema = simple_dimension_schema(task.doc_type)
            prompt, instructions_text, metadata_text = _single_dimension_prompt_parts(
                task.doc_type,
                task.document_id,
                task.metadata,
                task.dimension_definition,
                scoring_text,
                numeric_result,
            )
            if task.doc_type == "mda":
                budget = validate_prompt_context_budget(
                    instructions_text=instructions_text,
                    schema_text=json.dumps(schema, ensure_ascii=False),
                    selected_text=scoring_text,
                    metadata_text=metadata_text,
                    config=PromptBudgetConfig(num_ctx=task.num_ctx),
                )
                budget_report = budget.to_dict()
                if not budget.budget_passed or not task.context_budget_passed:
                    error_type = "context_budget_failed"
                    error_message = json.dumps(budget_report, ensure_ascii=False, sort_keys=True)
                    return _save_result(
                        task,
                        parsed_path,
                        None,
                        raw_output_path,
                        parse_status,
                        schema_status,
                        error_type,
                        error_message,
                        started_at,
                        started_clock,
                        attempts_used,
                        budget_report,
                        post_call_budget_report,
                    )
            last_issues: list[str] = []
            for attempt in range(1, task.max_retries + 1):
                attempts_used = attempt
                try:
                    strict_prompt = prompt
                    if attempt > 1:
                        strict_prompt += "\n\nRetry: return exactly one JSON object with only score, evidence, reason."
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
                    if task.doc_type == "mda":
                        post_call_budget = validate_post_call_prompt_eval(
                            result.raw_response,
                            config=PromptBudgetConfig(num_ctx=task.num_ctx),
                        )
                        post_call_budget_report = post_call_budget.to_dict()
                        if not post_call_budget.prompt_eval_passed:
                            error_type = "prompt_eval_too_high"
                            error_message = json.dumps(post_call_budget_report, ensure_ascii=False, sort_keys=True)
                            break
                    repaired = repair_then_validate_json(result.content, task.doc_type, task.dimension_code, text)
                    parse_status = repaired["parse_status"]
                    schema_status = repaired["schema_validation_status"]
                    if repaired["status"] == "valid":
                        payload = _expand_simple_payload(repaired["payload"], task, numeric_result, text)
                        break
                    last_issues = repaired["issues"]
                except OllamaClientError as exc:
                    error_type = "request_failed"
                    error_message = str(exc)
                    last_issues = [str(exc)]
            if payload is None and not error_message:
                error_type = "manual_required"
                error_message = ";".join(last_issues)
        return _save_result(
            task,
            parsed_path,
            payload,
            raw_output_path,
            parse_status,
            schema_status,
            error_type,
            error_message,
            started_at,
            started_clock,
            attempts_used,
            budget_report,
            post_call_budget_report,
        )
    except Exception as exc:  # noqa: BLE001 - one dimension must never abort a batch silently.
        return _save_result(
            task,
            parsed_path,
            None,
            raw_output_path,
            parse_status,
            schema_status,
            error_type or type(exc).__name__,
            error_message or str(exc),
            started_at,
            started_clock,
            attempts_used,
            budget_report,
            post_call_budget_report,
        )


def _single_dimension_prompt(
    doc_type: str,
    document_id: str,
    metadata: dict[str, str],
    dimension: dict[str, Any],
    text: str,
    numeric_result: dict[str, Any] | None = None,
) -> str:
    prompt, _instructions, _metadata = _single_dimension_prompt_parts(doc_type, document_id, metadata, dimension, text, numeric_result)
    return prompt


def _single_dimension_prompt_parts(
    doc_type: str,
    document_id: str,
    metadata: dict[str, str],
    dimension: dict[str, Any],
    text: str,
    numeric_result: dict[str, Any] | None = None,
) -> tuple[str, str, str]:
    locator_example = "MDA_P001" if doc_type == "mda" else "NEWS_P001"
    context = {
        "document_id": document_id,
        "doc_type": doc_type,
        "registry": _compact_registry_metadata(metadata),
        "dimension_to_score": dimension,
        "numeric_check_summary": _compact_numeric_summary(numeric_result),
    }
    ar02_rule = ""
    if doc_type == "mda" and dimension.get("code") == "AR02":
        ar02_rule = (
            "AR02 rule: numeric_check_summary is authoritative. LLM must not calculate growth rates, percentages, "
            "totals, or arithmetic correctness. Python numeric_checker controls the final score. Only explain whether "
            "the numeric_checker output is reasonably explained, logically consistent, or missing explanation.\n"
        )
    instructions = (
        f"Score exactly ONE {doc_type.upper()} dimension.\n"
        "Return exactly one JSON object. No markdown. No think block. No array. No nested JSON. No extra text.\n"
        "Do not score or mention other dimensions. Do not aggregate. Do not perform arithmetic.\n"
        "Allowed JSON fields only: score, evidence, reason.\n"
        "score must be integer 1-5. evidence must be one locator like "
        f"{locator_example}. reason must be short text under 240 characters.\n"
        "For MDA scoring, use only the selected dimension-specific context below when provided.\n"
        f"{ar02_rule}"
        f"Valid response example: {{\"score\": 3, \"evidence\": \"{locator_example}\", \"reason\": \"short reason\"}}\n"
    )
    metadata_text = json.dumps(context, ensure_ascii=False, indent=2)
    prompt = (
        instructions
        + "\nDOCUMENT_METADATA:\n"
        + metadata_text
        + "\n\nSELECTED_DIMENSION_CONTEXT:\n"
        + text
    )
    return prompt, instructions, metadata_text


def _build_tasks(
    *,
    doc_type: str,
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
    dimension_codes: list[str] | None,
) -> tuple[list[SingleDimensionTask], list[dict[str, Any]]]:
    tasks: list[SingleDimensionTask] = []
    skipped: list[dict[str, Any]] = []
    wanted_dimensions = {code.upper() for code in dimension_codes or []}
    for row in rows:
        doc_id = row.get("document_id", "")
        if not doc_id:
            continue
        text_path = resolve_text_path(row, doc_type, root)
        if not text_path.exists():
            continue
        numeric_path_text = str(_ensure_numeric_check(root, row, text_path)) if doc_type == "mda" else ""
        contexts = {}
        if doc_type == "mda":
            contexts = select_document_dimension_contexts(
                root=root,
                document_id=doc_id,
                text_path=text_path,
                numeric_check_path=Path(numeric_path_text) if numeric_path_text else None,
                context_budget=_dimension_context_budget(num_ctx),
            )
        for dimension in dimensions_for(doc_type):
            code = dimension["code"]
            if wanted_dimensions and code.upper() not in wanted_dimensions:
                continue
            parsed_path = root / "06_ratings" / output_name / "per_dimension" / f"{doc_id}_{code}_parsed.json"
            if overwrite and parsed_path.exists():
                parsed_path.unlink()
            if resume and not overwrite and _dimension_completed(parsed_path):
                skipped.append({"document_id": doc_id, "doc_type": doc_type, "dimension_code": code, "status": "skipped", "timestamp": now_iso()})
                continue
            tasks.append(
                SingleDimensionTask(
                    document_id=doc_id,
                    doc_type=doc_type,
                    dimension_code=code,
                    dimension_name=dimension["name"],
                    dimension_definition=dimension,
                    metadata=row,
                    text_path=str(text_path),
                    numeric_check_path=numeric_path_text,
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
                    context_path=contexts[code].output_path if doc_type == "mda" else "",
                    context_budget_passed=contexts[code].budget_passed if doc_type == "mda" else True,
                )
            )
    return tasks, skipped


def _compact_registry_metadata(metadata: dict[str, str]) -> dict[str, str]:
    allowed = [
        "document_id",
        "doc_type",
        "stock_code",
        "ticker",
        "company_name",
        "report_year",
        "source_name",
        "publish_date",
        "title",
        "url",
    ]
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


def _mock_simple_payload(task: SingleDimensionTask, text: str, numeric_result: dict[str, Any] | None) -> dict[str, Any]:
    full_payload = mock_score_document(task.doc_type, task.document_id, text, numeric_result)
    dimension_payload = next(item for item in full_payload["dimension_scores"] if item["dimension_code"] == task.dimension_code)
    evidence = normalize_simple_evidence(str(dimension_payload.get("evidence_locator", "")), task.doc_type) or _first_paragraph_locator(text, task.doc_type)
    return {
        "score": _safe_score(dimension_payload.get("raw_score")),
        "evidence": evidence,
        "reason": _short_text(str(dimension_payload.get("reason") or dimension_payload.get("comment_short") or "Evidence supports this score."), 240),
    }


def _expand_simple_payload(
    simple_payload: dict[str, Any],
    task: SingleDimensionTask,
    numeric_result: dict[str, Any] | None,
    text: str,
) -> dict[str, Any]:
    code = task.dimension_code
    score = _safe_score(simple_payload.get("score"))
    reason = _short_text(str(simple_payload.get("reason", "")).strip() or "Evidence supports this score.", 240)
    if task.doc_type == "mda" and code == "AR02":
        score = _ar02_score_from_numeric(numeric_result)
        reason = _short_text(f"Python numeric_checker: {reason}", 240)
    evidence = normalize_simple_evidence(str(simple_payload.get("evidence", "")), task.doc_type) or _first_paragraph_locator("", task.doc_type)
    evidence_locator = f"[{evidence}]"
    evidence_text = _evidence_text_from_locator(text, evidence_locator, task.doc_type) or f"{evidence_locator} {reason}"
    return {
        "dimension_code": code,
        "dimension_name": task.dimension_name,
        "raw_score": score,
        "reason": reason,
        "evidence_locator": evidence_locator,
        "evidence_text_short": _short_text(evidence_text, 240),
        "confidence_level": "medium",
        "missing_type": "none",
        "comment_short": reason,
    }


def _save_result(
    task: SingleDimensionTask,
    parsed_path: Path,
    payload: dict[str, Any] | None,
    raw_output_path: str,
    parse_status: str,
    schema_status: str,
    error_type: str,
    error_message: str,
    started_at: str,
    started_clock: float,
    attempts_used: int,
    prompt_budget: dict[str, Any] | None = None,
    post_call_prompt_budget: dict[str, Any] | None = None,
) -> SingleDimensionResult:
    finished_at = now_iso()
    duration = round(time.perf_counter() - started_clock, 3)
    status = "success" if payload is not None and not error_type else "failed"
    if status == "failed":
        error_type = error_type or "manual_required"
        schema_status = schema_status if schema_status != "not_attempted" else "invalid"
    result = SingleDimensionResult(
        task.document_id,
        task.doc_type,
        task.dimension_code,
        status,
        raw_output_path,
        str(parsed_path),
        "parsed" if status == "success" else parse_status,
        "valid" if status == "success" else schema_status,
        "" if status == "success" else error_type,
        "" if status == "success" else error_message,
        started_at,
        finished_at,
        duration,
        attempts_used,
        task.model_name if not task.mock else "mock_qwen_unavailable",
        task.numeric_check_path,
        task.context_path,
    )
    record = {
        "document_id": task.document_id,
        "doc_type": task.doc_type,
        "dimension_code": task.dimension_code,
        "dimension_name": task.dimension_name,
        "status": status,
        "metadata": task.metadata,
        "text_path": task.text_path,
        "context_path": task.context_path,
        "numeric_check_path": task.numeric_check_path,
        "prompt_budget": prompt_budget or {},
        "post_call_prompt_budget": post_call_prompt_budget or {},
        "result": asdict(result),
    }
    if payload is not None and status == "success":
        record["payload"] = payload
    save_json(parsed_path, record)
    return result


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


def _read_context_text(task: SingleDimensionTask, fallback_text: str) -> str:
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


def _first_paragraph_locator(text: str, doc_type: str) -> str:
    pattern = r"MDA_P\d{3}" if doc_type == "mda" else r"NEWS_P\d{3}"
    match = re.search(pattern, text)
    return match.group(0) if match else ("MDA_P001" if doc_type == "mda" else "NEWS_P001")


def _evidence_text_from_locator(text: str, locator: str, doc_type: str) -> str:
    prefix = "MDA" if doc_type == "mda" else "NEWS"
    match = re.search(rf"\[({prefix}_P\d{{3}})\]", locator)
    if not match:
        return ""
    paragraph_id = match.group(1)
    marker = f"[{paragraph_id}]"
    start = text.find(marker)
    if start < 0:
        return ""
    next_match = re.search(rf"\n\[{prefix}_P\d{{3}}\]\s*\n", text[start + len(marker) :])
    end = start + len(marker) + next_match.start() if next_match else len(text)
    paragraph = text[start:end].strip()
    paragraph = re.sub(r"(?m)^\[SECTION:\s*[^\]]+\]\s*$", "", paragraph).strip()
    return paragraph


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


def _normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


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


def _unavailable_summary(
    doc_type: str,
    output_name: str,
    ratings_prefix: str,
    rows: list[dict[str, str]],
    model_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "doc_type": doc_type,
        "scoring_strategy": "single_dimension",
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
    parser = argparse.ArgumentParser(description="Run low-error single-dimension MD&A/news scoring.")
    parser.add_argument("--doc-type", choices=["mda", "news"], required=True)
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--output-name")
    parser.add_argument("--ratings-prefix")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_single_dimension_scoring(
                args.doc_type,
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
