from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_clients import OllamaClient
from mda_numeric_checker import analyze_file
from run_scoring import (
    _build_document_score_row,
    _build_rating_rows,
    _review_reasons,
    _score_with_retries,
    _select_rows,
    build_prompt,
    handle_model_unavailable,
    mock_score_document,
    now_iso,
)
from scoring_utils import (
    DEFAULT_ENV,
    SCORING_ROOT,
    append_jsonl,
    ensure_directories,
    load_json,
    read_csv_rows,
    registry_path,
    resolve_text_path,
    save_json,
)
from validate_model_scores import normalize_scoring_payload, validate_scoring_payload


@dataclass
class ScoringTask:
    document_id: str
    doc_type: str
    text_path: str
    metadata: dict[str, str]
    numeric_check_path: str
    prompt_path: str
    schema_path: str
    output_dir: str
    model_name: str
    base_url: str
    temperature: float
    seed: int
    num_ctx: int
    attempt: int
    timeout_seconds: int
    max_retries: int
    mock_mode: bool = False


@dataclass
class ScoringResult:
    document_id: str
    doc_type: str
    success: bool
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
    needs_review: bool
    review_reasons: list[str]
    status: str
    worker_id: str


def cpu_worker_default() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def warn_if_too_many_workers(llm_workers: int) -> None:
    if llm_workers > 4:
        print(
            "Local qwen3:8b scoring may become slower or unstable with too many concurrent workers.",
            file=sys.stderr,
        )


def atomic_write_json(path: Path, payload: Any) -> None:
    save_json(path, payload)


def run_parallel_pipeline(
    doc_type: str,
    mode: str,
    *,
    root: Path = SCORING_ROOT,
    limit: int | None = None,
    cpu_workers: int | None = None,
    llm_workers: int = 1,
    batch_size: int = 1,
    resume: bool = True,
    overwrite: bool = False,
    max_retries: int = 3,
    timeout_seconds: int = 180,
    model: str = DEFAULT_ENV["OLLAMA_MODEL"],
    base_url: str = DEFAULT_ENV["OLLAMA_BASE_URL"],
    temperature: float = float(DEFAULT_ENV["OLLAMA_TEMPERATURE"]),
    seed: int = int(DEFAULT_ENV["OLLAMA_SEED"]),
    num_ctx: int = int(DEFAULT_ENV["OLLAMA_NUM_CTX"]),
    mock_mode: bool = False,
    aggregate: bool = True,
    include_manual_required: bool = False,
) -> dict[str, Any]:
    ensure_directories(root)
    if doc_type in {"mda", "news"} and not mock_mode:
        raise ValueError(f"{doc_type} single-call document-level scoring is disabled; use single-dimension scoring.")
    warn_if_too_many_workers(llm_workers)
    cpu_workers = cpu_workers or cpu_worker_default()
    registry_rows = read_csv_rows(registry_path(doc_type, root))
    selected = _select_rows(registry_rows, mode, limit)

    if not mock_mode:
        client = OllamaClient(base_url=base_url, model=model, timeout=timeout_seconds, temperature=temperature, seed=seed, num_ctx=num_ctx)
        model_status = client.ensure_model_available()
        if not model_status.get("available") or not model_status.get("model_available"):
            return handle_model_unavailable(
                doc_type,
                selected,
                root,
                "Local Ollama qwen3:8b is unavailable; generated manual scoring templates only.",
            )

    if doc_type == "mda":
        _run_numeric_checks_parallel(selected, root, cpu_workers)

    tasks, skipped = build_scoring_tasks(
        doc_type,
        selected,
        root=root,
        resume=resume,
        overwrite=overwrite,
        model=model,
        base_url=base_url,
        temperature=temperature,
        seed=seed,
        num_ctx=num_ctx,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        mock_mode=mock_mode,
        include_manual_required=include_manual_required,
    )
    progress_path = root / "08_reports" / "scoring_progress_log.jsonl"
    for row in skipped:
        append_jsonl(progress_path, row)

    started = time.perf_counter()
    results: list[ScoringResult] = []
    with ThreadPoolExecutor(max_workers=max(1, llm_workers)) as executor:
        futures = [executor.submit(score_one_document, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            append_jsonl(
                progress_path,
                {
                    "document_id": result.document_id,
                    "doc_type": result.doc_type,
                    "status": result.status,
                    "worker_id": result.worker_id,
                    "attempts_used": result.attempts_used,
                    "duration_seconds": result.duration_seconds,
                    "error_type": result.error_type or None,
                    "timestamp": result.finished_at,
                },
            )

    from aggregate_scores import aggregate_scores
    from build_review_log import build_review_log
    from build_summary_report import build_summary_report

    if aggregate:
        aggregate_scores(doc_type, root=root)
        build_review_log(doc_type, root=root)

    duration = round(time.perf_counter() - started, 3)
    summary = {
        "doc_type": doc_type,
        "mode": mode,
        "selected_count": len(selected),
        "task_count": len(tasks),
        "skipped_count": len(skipped),
        "success_count": sum(1 for item in results if item.success),
        "failure_count": sum(1 for item in results if not item.success),
        "json_parse_failures": sum(1 for item in results if item.parse_status != "parsed"),
        "schema_validation_failures": sum(1 for item in results if item.schema_validation_status != "valid"),
        "retry_count": sum(max(0, item.attempts_used - 1) for item in results),
        "llm_workers": llm_workers,
        "cpu_workers": cpu_workers,
        "batch_size": batch_size,
        "duration_seconds": duration,
        "docs_per_minute": round((len(results) / duration) * 60, 3) if duration else 0,
        "ollama_available": True,
        "model_available": True,
        "model_name": model if not mock_mode else "mock_qwen_unavailable",
        "llm_backend": "ollama" if not mock_mode else "mock",
        "manual_template_only": False,
        "message": "",
    }
    atomic_write_json(root / "08_reports" / f"{doc_type}_parallel_run_summary.json", summary)
    if aggregate:
        build_summary_report(root=root, summary_path=root / "08_reports" / f"{doc_type}_parallel_run_summary.json")
    return summary


def build_scoring_tasks(
    doc_type: str,
    rows: list[dict[str, str]],
    *,
    root: Path,
    resume: bool,
    overwrite: bool,
    model: str,
    base_url: str,
    temperature: float,
    seed: int,
    num_ctx: int,
    timeout_seconds: int,
    max_retries: int,
    mock_mode: bool,
    include_manual_required: bool,
) -> tuple[list[ScoringTask], list[dict[str, Any]]]:
    tasks: list[ScoringTask] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        document_id = row.get("document_id", "")
        if not document_id or document_id in seen:
            continue
        seen.add(document_id)
        parsed_path = root / "06_ratings" / "per_document" / doc_type / f"{document_id}_parsed.json"
        if overwrite and parsed_path.exists():
            parsed_path.unlink()
        if resume and not overwrite and _completed_ok(parsed_path):
            skipped.append(
                {
                    "document_id": document_id,
                    "doc_type": doc_type,
                    "status": "skipped",
                    "worker_id": "main",
                    "attempts_used": 0,
                    "duration_seconds": 0,
                    "error_type": None,
                    "timestamp": now_iso(),
                }
            )
            continue
        if resume and not overwrite and _manual_required(parsed_path) and not include_manual_required:
            skipped.append(
                {
                    "document_id": document_id,
                    "doc_type": doc_type,
                    "status": "skipped",
                    "worker_id": "main",
                    "attempts_used": 0,
                    "duration_seconds": 0,
                    "error_type": "manual_required",
                    "timestamp": now_iso(),
                }
            )
            continue
        text_path = resolve_text_path(row, doc_type, root)
        numeric_path = root / "03_numeric_checks" / "mda" / f"{document_id}_numeric_checks.json" if doc_type == "mda" else Path("")
        tasks.append(
            ScoringTask(
                document_id=document_id,
                doc_type=doc_type,
                text_path=str(text_path),
                metadata=row,
                numeric_check_path=str(numeric_path),
                prompt_path=str(root / "04_prompts" / f"{doc_type}_scoring_prompt.txt"),
                schema_path=str(root / "04_prompts" / "scoring_output_schema.json"),
                output_dir=str(root),
                model_name=model,
                base_url=base_url,
                temperature=temperature,
                seed=seed,
                num_ctx=num_ctx,
                attempt=1,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                mock_mode=mock_mode,
            )
        )
    return tasks, skipped


def score_one_document(task: ScoringTask) -> ScoringResult:
    root = Path(task.output_dir)
    started_clock = time.perf_counter()
    started_at = now_iso()
    worker_id = f"thread-{os.getpid()}"
    parsed_path = root / "06_ratings" / "per_document" / task.doc_type / f"{task.document_id}_parsed.json"
    raw_output_path = ""
    parse_status = "not_attempted"
    schema_status = "not_attempted"
    payload: dict[str, Any] | None = None
    error_type = ""
    error_message = ""
    attempts_used = 0
    review_rows: list[dict[str, Any]] = []
    numeric_result = None
    numeric_check_path = task.numeric_check_path
    try:
        text_path = Path(task.text_path)
        text = text_path.read_text(encoding="utf-8", errors="replace")
        schema = load_json(Path(task.schema_path))
        if task.doc_type == "mda" and numeric_check_path and Path(numeric_check_path).exists():
            numeric_result = load_json(Path(numeric_check_path))
        if task.mock_mode:
            payload = normalize_scoring_payload(
                mock_score_document(task.doc_type, task.document_id, text, numeric_result),
                task.doc_type,
                task.document_id,
            )
            raw_path = root / "05_raw_model_outputs" / task.doc_type / f"{task.document_id}_attempt_1.json"
            atomic_write_json(raw_path, {"mock_mode": True, "payload": payload})
            raw_output_path = str(raw_path)
            attempts_used = 1
            parse_status = "parsed"
            validation = validate_scoring_payload(payload, text, task.doc_type)
            schema_status = "valid" if validation.is_valid else "invalid"
            if not validation.is_valid:
                error_type = "schema_validation_failed"
                error_message = ";".join(validation.issues)
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
            prompt = build_prompt(task.doc_type, task.metadata, text, numeric_result, root)
            payload, raw_output_path, parse_status, schema_status, issues, attempts_used = _score_with_retries(
                client=client,
                prompt=prompt,
                schema=schema,
                doc_type=task.doc_type,
                document_id=task.document_id,
                text=text,
                root=root,
                max_retries=task.max_retries,
            )
            if payload is None:
                error_type = "scoring_failed"
                error_message = ";".join(issues)
        finished_at = now_iso()
        duration = round(time.perf_counter() - started_clock, 3)
        if payload is None:
            result = ScoringResult(
                task.document_id,
                task.doc_type,
                False,
                raw_output_path,
                str(parsed_path),
                parse_status,
                schema_status,
                error_type or "scoring_failed",
                error_message,
                started_at,
                finished_at,
                duration,
                attempts_used,
                True,
                [error_type or "scoring_failed"],
                "failed",
                worker_id,
            )
            atomic_write_json(parsed_path, {"document_id": task.document_id, "doc_type": task.doc_type, "status": "failed", "result": asdict(result)})
            return result
        doc_row = _build_document_score_row(payload, task.metadata, task.doc_type, task.model_name if not task.mock_mode else "mock_qwen_unavailable")
        rating_rows = _build_rating_rows(
            payload,
            task.doc_type,
            started_at,
            finished_at,
            task.model_name if not task.mock_mode else "mock_qwen_unavailable",
            raw_output_path,
            numeric_check_path,
            parse_status,
            schema_status,
        )
        review_rows = _review_reasons(
            payload,
            text,
            task.metadata,
            task.doc_type,
            float(doc_row["total_score_100"]),
            numeric_result,
            numeric_check_path,
            raw_output_path,
        )
        result = ScoringResult(
            task.document_id,
            task.doc_type,
            True,
            raw_output_path,
            str(parsed_path),
            parse_status,
            schema_status,
            "",
            "",
            started_at,
            finished_at,
            duration,
            attempts_used,
            bool(payload.get("needs_review")) or bool(review_rows),
            payload.get("review_reasons", []),
            "success",
            worker_id,
        )
        atomic_write_json(
            parsed_path,
            {
                "document_id": task.document_id,
                "doc_type": task.doc_type,
                "status": "success",
                "payload": payload,
                "metadata": task.metadata,
                "result": asdict(result),
                "ratings_rows": rating_rows,
                "document_score_row": doc_row,
                "review_rows": review_rows,
            },
        )
        return result
    except Exception as exc:  # noqa: BLE001 - worker must isolate document failure.
        finished_at = now_iso()
        duration = round(time.perf_counter() - started_clock, 3)
        result = ScoringResult(
            task.document_id,
            task.doc_type,
            False,
            raw_output_path,
            str(parsed_path),
            parse_status,
            schema_status,
            type(exc).__name__,
            str(exc),
            started_at,
            finished_at,
            duration,
            attempts_used,
            True,
            [type(exc).__name__],
            "failed",
            worker_id,
        )
        atomic_write_json(parsed_path, {"document_id": task.document_id, "doc_type": task.doc_type, "status": "failed", "result": asdict(result)})
        return result


def _completed_ok(parsed_path: Path) -> bool:
    if not parsed_path.exists():
        return False
    try:
        data = load_json(parsed_path)
    except Exception:
        return False
    return data.get("status") == "success" and data.get("result", {}).get("schema_validation_status") == "valid"


def _manual_required(parsed_path: Path) -> bool:
    if not parsed_path.exists():
        return False
    try:
        data = load_json(parsed_path)
    except Exception:
        return False
    result = data.get("result", {})
    return data.get("status") == "failed" or result.get("error_type") == "manual_required"


def _run_numeric_checks_parallel(rows: list[dict[str, str]], root: Path, cpu_workers: int) -> None:
    tasks = []
    for row in rows:
        document_id = row.get("document_id", "")
        text_path = resolve_text_path(row, "mda", root)
        if document_id and text_path.exists():
            tasks.append((document_id, str(text_path), str(root / "03_numeric_checks" / "mda" / f"{document_id}_numeric_checks.json")))
    if not tasks:
        return
    if cpu_workers <= 1:
        for task in tasks:
            _numeric_worker(task)
        return
    with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
        list(executor.map(_numeric_worker, tasks))


def _numeric_worker(task: tuple[str, str, str]) -> str:
    document_id, text_path, output_path = task
    result = analyze_file(Path(text_path), document_id=document_id)
    atomic_write_json(Path(output_path), result)
    return output_path
