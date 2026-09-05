from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_clients import OllamaClient, OllamaClientError
from mda_numeric_checker import analyze_text
from scoring_utils import (
    DEFAULT_ENV,
    FLAG_FIELDS,
    NEWS_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    REVIEW_LOG_HEADERS,
    SCORING_ROOT,
    SCOREBOOK_VERSION,
    dimensions_for,
    document_headers_for,
    document_total,
    ensure_directories,
    grade_label,
    load_json,
    purpose_track_for,
    read_csv_rows,
    registry_path,
    resolve_text_path,
    save_json,
    scoring_basis_for,
    standardize_score,
    weighted_score,
    weight_sum_ok,
    weights_for,
    write_csv_rows,
    write_default_registries,
)
from validate_model_scores import normalize_scoring_payload, validate_scoring_payload


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"1", "true", "yes", "y"}:
        return True
    if str(value).lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def handle_model_unavailable(
    doc_type: str,
    registry_rows: list[dict[str, str]],
    root: Path = SCORING_ROOT,
    reason: str = "Local Ollama qwen3:8b is unavailable",
) -> dict[str, Any]:
    ensure_directories(root)
    selected = _select_rows(registry_rows, "full", None)
    template_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for row in selected:
        for dimension in dimensions_for(doc_type):
            template_rows.append(
                {
                    "document_id": row.get("document_id", ""),
                    "doc_type": doc_type,
                    "dimension_code": dimension["code"],
                    "dimension_name": dimension["name"],
                    "manual_raw_score": "",
                    "evidence_locator": "",
                    "evidence_text_short": "",
                    "comment_short": "",
                    "manual_required_reason": reason,
                }
            )
        review_rows.append(
            _review_row(
                row.get("document_id", ""),
                doc_type,
                "manual_required",
                reason,
                "",
                "",
                "",
                "",
                "",
            )
        )
    manual_headers = [
        "document_id",
        "doc_type",
        "dimension_code",
        "dimension_name",
        "manual_raw_score",
        "evidence_locator",
        "evidence_text_short",
        "comment_short",
        "manual_required_reason",
    ]
    write_csv_rows(root / "07_review" / f"{doc_type}_manual_scoring_template.csv", manual_headers, template_rows)
    write_csv_rows(root / "06_ratings" / f"{doc_type}_ratings_long.csv", RATINGS_LONG_HEADERS, [])
    write_csv_rows(root / "06_ratings" / f"{doc_type}_document_scores.csv", document_headers_for(doc_type), [])
    _merge_review_log(root, review_rows)
    summary = {
        "doc_type": doc_type,
        "mode": "manual_template_only",
        "ollama_available": False,
        "model_available": False,
        "model_name": DEFAULT_ENV["OLLAMA_MODEL"],
        "llm_backend": "ollama",
        "selected_count": len(selected),
        "success_count": 0,
        "failure_count": len(selected),
        "json_parse_failures": 0,
        "schema_validation_failures": 0,
        "retry_count": 0,
        "manual_template_only": True,
        "message": reason,
    }
    write_summary_report(root, summary)
    return summary


def run_pipeline(
    doc_type: str,
    mode: str,
    limit: int | None = None,
    seed: int = 42,
    model: str = "qwen3:8b",
    base_url: str = "http://localhost:11434",
    overwrite: bool = False,
    resume: bool = True,
    max_retries: int = 3,
    temperature: float = 0,
    num_ctx: int = 8192,
    root: Path = SCORING_ROOT,
    registry_rows_override: list[dict[str, str]] | None = None,
    mock_mode: bool = False,
    write_report: bool = True,
) -> dict[str, Any]:
    ensure_directories(root)
    if doc_type in {"mda", "news"} and not mock_mode:
        raise ValueError(f"{doc_type} single-call document-level scoring is disabled; use single-dimension scoring.")
    write_default_registries(root)
    schema = load_json(root / "04_prompts" / "scoring_output_schema.json")
    registry_rows = registry_rows_override if registry_rows_override is not None else read_csv_rows(registry_path(doc_type, root))
    selected = _select_rows(registry_rows, mode, limit)
    if overwrite:
        _clear_doc_type_review_rows(root, doc_type)

    if not mock_mode:
        client = OllamaClient(base_url, model, timeout=DEFAULT_ENV["OLLAMA_TIMEOUT_SECONDS"], temperature=temperature, seed=seed, num_ctx=num_ctx)
        model_status = client.ensure_model_available()
        if not model_status.get("available") or not model_status.get("model_available"):
            return handle_model_unavailable(
                doc_type,
                selected,
                root,
                "Local Ollama qwen3:8b is unavailable; generated manual scoring templates only.",
            )
    else:
        client = None
        model_status = {"available": True, "model_available": True, "models": ["mock_qwen_unavailable"], "error": ""}

    ratings_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    numeric_summary_rows: list[dict[str, Any]] = []
    raw_attempt_count = 0
    json_parse_failures = 0
    schema_validation_failures = 0
    seen_document_ids: set[str] = set()

    for row in selected:
        document_id = row.get("document_id", "").strip()
        if not document_id:
            continue
        if document_id in seen_document_ids:
            review_rows.append(_review_row(document_id, doc_type, "risk_review", "duplicate_document_id", "", "", "", "", ""))
            continue
        seen_document_ids.add(document_id)

        text_path = resolve_text_path(row, doc_type, root)
        if not text_path.exists():
            review_rows.append(_review_row(document_id, doc_type, "manual_required", "missing_text_file", "", "", "", "", ""))
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 50:
            review_rows.append(_review_row(document_id, doc_type, "manual_required", "text_length_too_short", "", "", "", "", ""))
            continue

        numeric_result: dict[str, Any] | None = None
        numeric_check_path = ""
        if doc_type == "mda":
            numeric_result = analyze_text(text, document_id)
            numeric_path = root / "03_numeric_checks" / "mda" / f"{document_id}_numeric_checks.json"
            save_json(numeric_path, numeric_result)
            numeric_check_path = str(numeric_path)
            numeric_summary_rows.append(_numeric_summary_row(numeric_result))

        prompt = build_prompt(doc_type, row, text, numeric_result, root)
        started_at = now_iso()
        final_payload: dict[str, Any] | None = None
        final_raw_path = ""
        parse_status = "not_attempted"
        schema_status = "not_attempted"
        validation_issues: list[str] = []

        if mock_mode:
            final_payload = mock_score_document(doc_type, document_id, text, numeric_result)
            final_payload = normalize_scoring_payload(final_payload, doc_type, document_id)
            final_raw_path = str(_save_attempt(root, doc_type, document_id, 1, {"mock_mode": True, "payload": final_payload}))
            parse_status = "parsed"
            validation = validate_scoring_payload(final_payload, text, doc_type)
            schema_status = "valid" if validation.is_valid else "invalid"
            validation_issues = validation.issues
        else:
            final_payload, final_raw_path, parse_status, schema_status, validation_issues, attempts_used = _score_with_retries(
                client=client,
                prompt=prompt,
                schema=schema,
                doc_type=doc_type,
                document_id=document_id,
                text=text,
                root=root,
                max_retries=max_retries,
            )
            raw_attempt_count += attempts_used

        finished_at = now_iso()
        if final_payload is None:
            json_parse_failures += 1
            review_rows.append(
                _review_row(
                    document_id,
                    doc_type,
                    "json_parse_review",
                    "model_json_parse_failed",
                    "",
                    "",
                    "",
                    numeric_check_path,
                    final_raw_path,
                )
            )
            continue
        if schema_status != "valid":
            schema_validation_failures += 1
            review_rows.append(
                _review_row(
                    document_id,
                    doc_type,
                    "schema_validation_review",
                    ";".join(validation_issues),
                    "",
                    "",
                    "",
                    numeric_check_path,
                    final_raw_path,
                )
            )
            continue

        dim_scores = final_payload["dimension_scores"]
        doc_score_row = _build_document_score_row(final_payload, row, doc_type, model if not mock_mode else "mock_qwen_unavailable")
        document_rows.append(doc_score_row)
        ratings_rows.extend(
            _build_rating_rows(
                payload=final_payload,
                doc_type=doc_type,
                started_at=started_at,
                finished_at=finished_at,
                model_name=model if not mock_mode else "mock_qwen_unavailable",
                raw_output_path=final_raw_path,
                numeric_check_path=numeric_check_path,
                parse_status=parse_status,
                schema_status=schema_status,
            )
        )
        review_rows.extend(
            _review_reasons(
                payload=final_payload,
                text=text,
                row=row,
                doc_type=doc_type,
                total_score=float(doc_score_row["total_score_100"]),
                numeric_result=numeric_result,
                numeric_check_path=numeric_check_path,
                raw_output_path=final_raw_path,
            )
        )

    write_csv_rows(root / "06_ratings" / f"{doc_type}_ratings_long.csv", RATINGS_LONG_HEADERS, ratings_rows)
    write_csv_rows(root / "06_ratings" / f"{doc_type}_document_scores.csv", document_headers_for(doc_type), document_rows)
    _merge_review_log(root, review_rows)
    if doc_type == "mda":
        _write_numeric_summary(root, numeric_summary_rows)

    summary = {
        "doc_type": doc_type,
        "mode": mode,
        "ollama_available": model_status.get("available", False),
        "model_available": model_status.get("model_available", False),
        "model_name": model if not mock_mode else "mock_qwen_unavailable",
        "llm_backend": "ollama" if not mock_mode else "mock",
        "selected_count": len(selected),
        "success_count": len(document_rows),
        "failure_count": len(selected) - len(document_rows),
        "json_parse_failures": json_parse_failures,
        "schema_validation_failures": schema_validation_failures,
        "retry_count": max(0, raw_attempt_count - len(document_rows)),
        "manual_template_only": False,
        "message": "",
    }
    if write_report:
        write_summary_report(root, summary)
    return summary


def build_prompt(
    doc_type: str,
    row: dict[str, str],
    text: str,
    numeric_result: dict[str, Any] | None,
    root: Path = SCORING_ROOT,
) -> str:
    template_path = root / "04_prompts" / f"{doc_type}_scoring_prompt.txt"
    template = template_path.read_text(encoding="utf-8")
    context = json.dumps(
        {
            "document_id": row.get("document_id", ""),
            "registry": row,
            "scorebook_dimensions": dimensions_for(doc_type),
            "text": text,
        },
        ensure_ascii=False,
        indent=2,
    )
    numeric_summary = json.dumps(numeric_result or {}, ensure_ascii=False, indent=2)
    return template.format(document_context=context, numeric_check_summary=numeric_summary)


def mock_score_document(
    doc_type: str,
    document_id: str,
    text: str,
    numeric_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    first_locator = _first_locator(text, doc_type)
    evidence = _text_after_locator(text, first_locator)
    scores: dict[str, int] = {}
    missing: dict[str, str] = {}
    if doc_type == "mda":
        numeric_state = (numeric_result or {}).get("overall_numeric_consistency", "not_enough_numbers")
        ar02 = {"strong": 4, "moderate": 3, "weak": 2, "problematic": 1, "not_enough_numbers": 3}.get(numeric_state, 3)
        scores = {"AR01": 3, "AR02": ar02, "AR03": 3, "AR04": 3 if "risk" in text.lower() or "outlook" in text.lower() else 2, "AR05": 3}
        missing = {code: "none" for code in scores}
        if scores["AR04"] == 2:
            missing["AR04"] = "structural_missing"
    else:
        lowered = text.lower()
        scores = {
            "N01": 3,
            "N02": 4 if "source" in lowered or "according to" in lowered else 2,
            "N03": 2 if "shocking" in lowered or "must buy" in lowered else 3,
            "N04": 3,
            "N05": 2 if "shocking" in lowered else 3,
        }
        missing = {code: "none" for code in scores}
        if scores["N02"] == 2:
            missing["N02"] = "structural_missing"
    dimension_scores = []
    for dimension in dimensions_for(doc_type):
        code = dimension["code"]
        dimension_scores.append(
            {
                "dimension_code": code,
                "dimension_name": dimension["name"],
                "raw_score": scores[code],
                "reason": "Synthetic smoke-test score based on explicit text cues.",
                "evidence_locator": first_locator,
                "evidence_text_short": evidence[:180] or "Paragraph evidence is present in the input text.",
                "confidence_level": "medium",
                "missing_type": missing[code],
                "comment_short": "Smoke-test only; not a research score.",
            }
        )
    return {
        "document_id": document_id,
        "doc_type": doc_type,
        "scoring_basis": scoring_basis_for(doc_type),
        "scorebook_version": SCOREBOOK_VERSION,
        "evidence_limitation": "Synthetic self-check mock scoring; not used for real research scoring.",
        "dimension_scores": dimension_scores,
        "flags": {field: 0 for field in FLAG_FIELDS},
        "overall_comment": "Synthetic self-check output for pipeline verification.",
        "needs_review": True,
        "review_reasons": ["synthetic_mock_scoring"],
    }


def write_summary_report(root: Path, run_summary: dict[str, Any]) -> Path:
    report_path = root / "08_reports" / "scoring_summary_report.md"
    mda_registry = read_csv_rows(registry_path("mda", root))
    news_registry = read_csv_rows(registry_path("news", root))
    mda_scores = read_csv_rows(root / "06_ratings" / "mda_document_scores.csv")
    news_scores = read_csv_rows(root / "06_ratings" / "news_document_scores.csv")
    review_rows = read_csv_rows(root / "07_review" / "review_log.csv")
    current_review_rows = [row for row in review_rows if row.get("doc_type") == run_summary.get("doc_type")]
    numeric_rows = read_csv_rows(root / "03_numeric_checks" / "mda_numeric_checks_summary.csv")

    unavailable_line = ""
    if run_summary.get("manual_template_only"):
        unavailable_line = "\nLocal Ollama qwen3:8b is unavailable; generated manual scoring templates only.\n"

    body = f"""# Scoring Summary Report

## 1. Project Scope

This scoring module uses local Ollama qwen3:8b when available. MD&A scoring is not full annual-report scoring. AR02 is not a financial-statement audit; it only checks whether financial expressions in the provided MD&A text are internally consistent and logically traceable.
{unavailable_line}
## 2. Local Model Configuration

- OLLAMA_BASE_URL: {run_summary.get("base_url", DEFAULT_ENV["OLLAMA_BASE_URL"])}
- OLLAMA_MODEL: {run_summary.get("model_name", DEFAULT_ENV["OLLAMA_MODEL"])}
- temperature: {DEFAULT_ENV["OLLAMA_TEMPERATURE"]}
- seed: {DEFAULT_ENV["OLLAMA_SEED"]}
- num_ctx: {DEFAULT_ENV["OLLAMA_NUM_CTX"]}
- Ollama available: {run_summary.get("ollama_available")}
- qwen3:8b available: {run_summary.get("model_available")}
- cpu_workers: {run_summary.get("cpu_workers", "")}
- llm_workers: {run_summary.get("llm_workers", "")}
- batch_size: {run_summary.get("batch_size", "")}
- run duration seconds: {run_summary.get("duration_seconds", "")}
- docs per minute: {run_summary.get("docs_per_minute", "")}
- parallel recommendation: {_parallel_recommendation(run_summary)}

## 3. Input Materials

- MD&A registry rows: {len(mda_registry)}
- News registry rows: {len(news_registry)}
- MD&A include_flag=Yes rows: {_include_count(mda_registry)}
- News include_flag=Yes rows: {_include_count(news_registry)}
- Missing or unscored rows in latest run: {run_summary.get("failure_count", 0)}

## 4. Self-Correction Process

See `06_scoring/_self_correction/self_correction_log.md` for round-by-round details. Latest scoring command mode: {run_summary.get("mode", "")}.

## 5. Scoring Run Status

- doc_type: {run_summary.get("doc_type", "")}
- selected documents: {run_summary.get("selected_count", 0)}
- successfully scored documents: {run_summary.get("success_count", 0)}
- failed/manual-required documents: {run_summary.get("failure_count", 0)}
- JSON parse failures: {run_summary.get("json_parse_failures", 0)}
- schema validation failures: {run_summary.get("schema_validation_failures", 0)}
- retry count: {run_summary.get("retry_count", 0)}

## 6. Claim-Level Framework

{_claim_level_status_note(root)}

## 7. MD&A Results

{_mda_current_status_note(mda_scores, mda_registry, root)}

{_score_section(mda_scores, "AR")}

AR02 numeric check summary:
- material discrepancy count: {_sum_int(numeric_rows, "num_material_discrepancies")}
- severe direction conflict count: {_sum_int(numeric_rows, "num_severe_direction_conflicts")}

## 8. News Results

{_score_section(news_scores, "N")}

## 9. Red Flags

{_flag_section(mda_scores + news_scores)}

High-risk documents are listed in `06_scoring/07_review/review_log.csv`.

## 10. Review List

- manual review rows for latest doc_type: {len(current_review_rows)}
- review_log path: `06_scoring/07_review/review_log.csv`
- main review reasons: {_review_reason_summary(current_review_rows)}

## 11. Output Files

- `06_scoring/01_registry/mda_registry.csv`
- `06_scoring/01_registry/news_registry.csv`
- `06_scoring/03_numeric_checks/mda_numeric_checks_summary.csv`
- `06_scoring/06_ratings/mda_ratings_long.csv`
- `06_scoring/06_ratings/news_ratings_long.csv`
- `06_scoring/06_ratings/mda_document_scores.csv`
- `06_scoring/06_ratings/news_document_scores.csv`
- `06_scoring/07_review/review_log.csv`
- `06_scoring/08_reports/scoring_summary_report.md`
- `06_scoring/10_claims/`
- `06_scoring/10_claim_scoring/`
- `06_scoring/10_claim_mapping/`
- `06_scoring/08_reports/claim_vs_document_comparison.md`
- `06_scoring/_self_correction/self_correction_log.md`

## 12. Limitations

- qwen3:8b is a local 8B model, so scores should retain human review.
- MD&A scoring is based only on the provided MD&A text.
- These scores cannot determine whether complete annual-report financial statements are accurate.
- AR02 represents only internal consistency of financial expressions inside the MD&A text.
- News scoring without external sources can only evaluate internal text quality and source transparency.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body, encoding="utf-8")
    return report_path


def _mda_current_status_note(mda_scores: list[dict[str, str]], mda_registry: list[dict[str, str]], root: Path) -> str:
    include_count = _include_count(mda_registry)
    full_scores = root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv"
    if full_scores.exists():
        return (
            "Current full MD&A candidate is available at "
            "`06_scoring/06_ratings/mda_v2_full/mda_v2_full_document_scores.csv`. "
            "`mda_document_scores.csv` remains the original/pilot output and has not been overwritten.\n"
        )
    if len(mda_scores) < include_count:
        return (
            "Current `06_scoring/06_ratings/mda_document_scores.csv` is pilot output, not the full MD&A scoring result. "
            "Final MD&A result must wait for `06_scoring/06_ratings/mda_v2_full/mda_v2_full_document_scores.csv` and human confirmation.\n"
        )
    return (
        "`06_scoring/06_ratings/mda_document_scores.csv` row count matches the current MD&A registry include set, "
        "but v2_full should still be checked before treating it as final.\n"
    )


def _claim_level_status_note(root: Path) -> str:
    mda_claim_scores = root / "10_claim_mapping" / "mda_document_scores_claim_level.csv"
    news_claim_scores = root / "10_claim_mapping" / "news_document_scores_claim_level.csv"
    comparison_report = root / "08_reports" / "claim_vs_document_comparison.md"
    mda_count = len(read_csv_rows(mda_claim_scores))
    news_count = len(read_csv_rows(news_claim_scores))
    if mda_claim_scores.exists() or news_claim_scores.exists():
        return (
            "系统已从 document-level scoring 升级为 claim-level scoring framework. "
            "Document-level scoring is preserved as fallback, and hybrid comparison remains available.\n\n"
            f"- MD&A claim-level scores: {'available' if mda_claim_scores.exists() else 'not generated'} ({mda_count} rows)\n"
            f"- News claim-level scores: {'available' if news_claim_scores.exists() else 'not generated'} ({news_count} rows; current news registry may be empty)\n"
            f"- Claim/document comparison report: {'available' if comparison_report.exists() else 'not generated'}\n"
        )
    return (
        "Claim-level scoring framework is installed, but claim extraction/scoring outputs have not yet been generated in this root. "
        "Run `extract_mda_claims.py`, `run_claim_scoring.py`, `aggregate_claim_scores.py`, and `compare_claim_vs_document_scoring.py` to create comparison artifacts.\n"
    )


def _score_with_retries(
    client: OllamaClient | None,
    prompt: str,
    schema: dict[str, Any],
    doc_type: str,
    document_id: str,
    text: str,
    root: Path,
    max_retries: int,
) -> tuple[dict[str, Any] | None, str, str, str, list[str], int]:
    assert client is not None
    raw_content = ""
    attempts_used = 0
    last_raw_path = ""
    validation_issues: list[str] = []
    for attempt in range(1, max_retries + 1):
        attempts_used += 1
        started_at = now_iso()
        try:
            if attempt == 2 and raw_content:
                result = client.repair_json(raw_content, schema)
            else:
                strict_prompt = prompt
                if attempt >= 3:
                    strict_prompt += "\n\nReturn JSON only. No markdown. No explanation outside JSON."
                result = client.score_document(strict_prompt, schema)
            raw_content = result.content
            payload = result.parsed_json
            parse_status = "parsed" if payload is not None else "parse_failed"
            normalized_payload = normalize_scoring_payload(payload, doc_type, document_id) if isinstance(payload, dict) else payload
            validation = validate_scoring_payload(normalized_payload if normalized_payload is not None else raw_content, text, doc_type)
            schema_status = "valid" if validation.is_valid else "invalid"
            validation_issues = validation.issues
            last_raw_path = str(
                _save_attempt(
                    root,
                    doc_type,
                    document_id,
                    attempt,
                    {
                        "model_name": client.model,
                        "base_url": client.base_url,
                        "temperature": client.temperature,
                        "seed": client.seed,
                        "num_ctx": client.num_ctx,
                        "attempt": attempt,
                        "started_at": started_at,
                        "finished_at": now_iso(),
                        "parse_status": parse_status,
                        "schema_validation_status": schema_status,
                        "retry_reason": "" if attempt == 1 else "repair_or_strict_retry",
                        "fallback_used": result.fallback_used,
                        "raw_response": result.raw_response,
                        "content": raw_content,
                    },
                )
            )
            if isinstance(normalized_payload, dict) and validation.is_valid:
                return normalized_payload, last_raw_path, parse_status, schema_status, [], attempts_used
            if payload is not None and not validation.is_valid:
                critique_path = _save_named_raw(
                    root,
                    doc_type,
                    document_id,
                    f"{document_id}_critique_attempt_{attempt}.json",
                    {
                        "issues": validation.issues,
                        "instruction": "Critique pass only; do not alter scores here.",
                    },
                )
                last_raw_path = str(critique_path)
        except OllamaClientError as exc:
            last_raw_path = str(
                _save_attempt(
                    root,
                    doc_type,
                    document_id,
                    attempt,
                    {
                        "model_name": client.model,
                        "base_url": client.base_url,
                        "attempt": attempt,
                        "started_at": started_at,
                        "finished_at": now_iso(),
                        "parse_status": "request_failed",
                        "schema_validation_status": "not_attempted",
                        "retry_reason": str(exc),
                    },
                )
            )
            validation_issues = [str(exc)]
    return None, last_raw_path, "parse_failed", "invalid", validation_issues, attempts_used


def _save_attempt(root: Path, doc_type: str, document_id: str, attempt: int, payload: dict[str, Any]) -> Path:
    filename = f"{document_id}_attempt_{attempt}.json"
    return _save_named_raw(root, doc_type, document_id, filename, payload)


def _save_named_raw(root: Path, doc_type: str, document_id: str, filename: str, payload: dict[str, Any]) -> Path:
    path = root / "05_raw_model_outputs" / doc_type / filename
    save_json(path, payload)
    return path


def _select_rows(rows: list[dict[str, str]], mode: str, limit: int | None) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("include_flag", "").strip().lower() == "yes"]
    if mode == "pilot":
        selected = selected[: limit or 3]
    elif limit is not None:
        selected = selected[:limit]
    return selected


def _build_rating_rows(
    payload: dict[str, Any],
    doc_type: str,
    started_at: str,
    finished_at: str,
    model_name: str,
    raw_output_path: str,
    numeric_check_path: str,
    parse_status: str,
    schema_status: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    weights = weights_for(doc_type)
    for item in payload["dimension_scores"]:
        code = item["dimension_code"]
        raw = int(item["raw_score"])
        rows.append(
            {
                "rating_id": f"{payload['document_id']}_{code}",
                "document_id": payload["document_id"],
                "doc_type": doc_type,
                "scoring_basis": payload.get("scoring_basis", scoring_basis_for(doc_type)),
                "purpose_track": purpose_track_for(doc_type),
                "review_round": "initial",
                "reviewer_id": "local_qwen3_8b" if model_name != "mock_qwen_unavailable" else "synthetic_mock",
                "dimension_code": code,
                "dimension_name": item["dimension_name"],
                "raw_score": raw,
                "scale_min": 1,
                "scale_max": 5,
                "weight_value": weights[code],
                "std_score_100": standardize_score(raw),
                "weighted_score": weighted_score(raw, weights[code]),
                "evidence_locator": item["evidence_locator"],
                "evidence_text_short": item["evidence_text_short"],
                "confidence_level": item["confidence_level"],
                "comment_short": item["comment_short"],
                "missing_type": item["missing_type"],
                "started_at": started_at,
                "finished_at": finished_at,
                "scorebook_version": payload.get("scorebook_version", SCOREBOOK_VERSION),
                "model_name": model_name,
                "llm_backend": "ollama" if model_name != "mock_qwen_unavailable" else "mock",
                "raw_output_path": raw_output_path,
                "numeric_check_path": numeric_check_path,
                "parse_status": parse_status,
                "schema_validation_status": schema_status,
            }
        )
    return rows


def _build_document_score_row(
    payload: dict[str, Any],
    registry_row: dict[str, str],
    doc_type: str,
    model_name: str,
) -> dict[str, Any]:
    dim_scores = payload["dimension_scores"]
    raw_scores = {item["dimension_code"]: int(item["raw_score"]) for item in dim_scores}
    weights = weights_for(doc_type)
    total = document_total(raw_scores, weights)
    base = {
        "document_id": payload["document_id"],
        "doc_type": doc_type,
        "scoring_basis": payload.get("scoring_basis", scoring_basis_for(doc_type)),
        "dimension_count": len(dim_scores),
        "total_score_100": total,
        "grade_label": grade_label(total),
        "adjudication_flag": 1 if payload.get("needs_review") else 0,
        "needs_review": payload.get("needs_review", False),
        "review_reasons": ";".join(payload.get("review_reasons", [])),
        "final_version": "initial",
        "scorebook_version": payload.get("scorebook_version", SCOREBOOK_VERSION),
        "model_name": model_name,
        "llm_backend": "ollama" if model_name != "mock_qwen_unavailable" else "mock",
    }
    for field in FLAG_FIELDS:
        base[field] = payload.get("flags", {}).get(field, 0)
    for code, raw in raw_scores.items():
        base[code] = raw
        base[f"{code}_std"] = standardize_score(raw)
    if doc_type == "mda":
        base.update(
            {
                "stock_code": registry_row.get("stock_code", ""),
                "ticker": registry_row.get("ticker", ""),
                "company_name": registry_row.get("company_name", ""),
                "report_year": registry_row.get("report_year", ""),
            }
        )
    else:
        base.update(
            {
                "company_name": registry_row.get("company_name", ""),
                "ticker": registry_row.get("ticker", ""),
                "source_name": registry_row.get("source_name", ""),
                "publish_date": registry_row.get("publish_date", ""),
                "title": registry_row.get("title", ""),
                "url": registry_row.get("url", ""),
            }
        )
    return base


def _review_reasons(
    payload: dict[str, Any],
    text: str,
    row: dict[str, str],
    doc_type: str,
    total_score: float,
    numeric_result: dict[str, Any] | None,
    numeric_check_path: str,
    raw_output_path: str,
) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    doc_id = payload["document_id"]
    for item in payload["dimension_scores"]:
        reasons = []
        if item["raw_score"] in {1, 5}:
            reasons.append("extreme_raw_score")
        if item["confidence_level"] == "low":
            reasons.append("low_confidence")
        if item["missing_type"] != "none":
            reasons.append("missing_type_not_none")
        if not item.get("evidence_locator"):
            reasons.append("empty_evidence_locator")
        if not item.get("evidence_text_short"):
            reasons.append("empty_evidence_text_short")
        for reason in reasons:
            reviews.append(
                _review_row(
                    doc_id,
                    doc_type,
                    _review_type_for_reason(reason),
                    reason,
                    item["dimension_code"],
                    item["raw_score"],
                    item.get("evidence_locator", ""),
                    numeric_check_path,
                    raw_output_path,
                )
            )
    if numeric_result:
        if int(numeric_result.get("num_material_discrepancies", 0)) > 0:
            reviews.append(_review_row(doc_id, doc_type, "numeric_check_review", "material_discrepancy", "AR02", "", "", numeric_check_path, raw_output_path))
        if int(numeric_result.get("num_severe_direction_conflicts", 0)) > 0:
            reviews.append(_review_row(doc_id, doc_type, "numeric_check_review", "severe_direction_conflict", "AR02", "", "", numeric_check_path, raw_output_path))
    for field in ["flag_unverified_key_number", "flag_nonstandard_audit", "flag_disclosure_replacement"]:
        if payload.get("flags", {}).get(field) == 1:
            reviews.append(_review_row(doc_id, doc_type, "risk_review", field, "", "", "", numeric_check_path, raw_output_path))
    if total_score < 55:
        reviews.append(_review_row(doc_id, doc_type, "risk_review", "total_score_below_55", "", "", "", numeric_check_path, raw_output_path))
    if len(text.strip()) < 200:
        reviews.append(_review_row(doc_id, doc_type, "manual_required", "text_length_too_short", "", "", "", numeric_check_path, raw_output_path))
    if doc_type == "mda" and row.get("extraction_status", "success").lower() != "success":
        reviews.append(_review_row(doc_id, doc_type, "manual_required", "mda_extraction_status_not_success", "", "", "", numeric_check_path, raw_output_path))
    if doc_type == "news" and ("[TITLE]" not in text or "[NEWS_P" not in text):
        reviews.append(_review_row(doc_id, doc_type, "manual_required", "news_missing_title_or_body", "", "", "", numeric_check_path, raw_output_path))
    combined = json.dumps(payload, ensure_ascii=False).lower()
    if "audited financial statements match" in combined or "matches the audited financial statements" in combined:
        reviews.append(_review_row(doc_id, doc_type, "mda_scope_violation_review", "audit_match_claim_in_mda_output", "", "", "", numeric_check_path, raw_output_path))
    return reviews


def _review_type_for_reason(reason: str) -> str:
    mapping = {
        "low_confidence": "low_confidence_review",
        "missing_type_not_none": "missing_evidence_review",
        "empty_evidence_locator": "missing_evidence_review",
        "empty_evidence_text_short": "missing_evidence_review",
        "extreme_raw_score": "risk_review",
    }
    return mapping.get(reason, "risk_review")


def _review_row(
    document_id: str,
    doc_type: str,
    review_type: str,
    review_reason: str,
    dimension_code: str,
    raw_score: Any,
    evidence_locator: str,
    numeric_check_path: str,
    raw_output_path: str,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "doc_type": doc_type,
        "review_type": review_type,
        "review_reason": review_reason,
        "dimension_code": dimension_code,
        "raw_score": raw_score,
        "evidence_locator": evidence_locator,
        "numeric_check_path": numeric_check_path,
        "raw_output_path": raw_output_path,
        "created_at": now_iso(),
    }


def _merge_review_log(root: Path, new_rows: list[dict[str, Any]]) -> None:
    path = root / "07_review" / "review_log.csv"
    existing = read_csv_rows(path)
    write_csv_rows(path, REVIEW_LOG_HEADERS, existing + new_rows)


def _clear_doc_type_review_rows(root: Path, doc_type: str) -> None:
    path = root / "07_review" / "review_log.csv"
    existing = read_csv_rows(path)
    kept = [row for row in existing if row.get("doc_type") != doc_type]
    write_csv_rows(path, REVIEW_LOG_HEADERS, kept)


def _write_numeric_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    headers = [
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
    ]
    write_csv_rows(root / "03_numeric_checks" / "mda_numeric_checks_summary.csv", headers, rows)


def _numeric_summary_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {key: value for key, value in result.items() if key != "checks"}
    if isinstance(row.get("internal_check_notes"), list):
        row["internal_check_notes"] = " | ".join(row["internal_check_notes"])
    return row


def _first_locator(text: str, doc_type: str) -> str:
    if doc_type == "mda":
        match = re.search(r"\[MDA_P\d+\]", text)
        return match.group(0) if match else "[MDA_P001]"
    for pattern in [r"\[TITLE\]", r"\[LEAD\]", r"\[NEWS_P\d+\]"]:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return "[NEWS_P001]"


def _text_after_locator(text: str, locator: str) -> str:
    idx = text.find(locator)
    if idx < 0:
        return ""
    after = text[idx + len(locator) :].strip()
    return " ".join(after.split())[:240]


def _include_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if row.get("include_flag", "").strip().lower() == "yes")


def _score_section(rows: list[dict[str, str]], prefix: str) -> str:
    totals = [float(row["total_score_100"]) for row in rows if row.get("total_score_100")]
    if not totals:
        return "No scored documents yet."
    codes = [f"{prefix}{idx:02d}" for idx in range(1, 6)]
    lines = [
        f"- average score: {statistics.mean(totals):.2f}",
        f"- median score: {statistics.median(totals):.2f}",
        f"- lowest score: {min(totals):.2f}",
        f"- highest score: {max(totals):.2f}",
        f"- A/B/C/D distribution: {_grade_distribution(rows)}",
    ]
    for code in codes:
        values = [float(row[code]) for row in rows if row.get(code)]
        if values:
            lines.append(f"- {code} average raw score: {statistics.mean(values):.2f}")
    return "\n".join(lines)


def _grade_distribution(rows: list[dict[str, str]]) -> str:
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for row in rows:
        label = row.get("grade_label")
        if label in counts:
            counts[label] += 1
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _flag_section(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No scored documents yet."
    return "\n".join(f"- {field}: {_sum_int(rows, field)}" for field in FLAG_FIELDS)


def _sum_int(rows: list[dict[str, str]], field: str) -> int:
    total = 0
    for row in rows:
        try:
            total += int(float(row.get(field, 0) or 0))
        except ValueError:
            continue
    return total


def _review_reason_summary(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}
    for row in rows:
        reason = row.get("review_reason", "")
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "none"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{reason}={count}" for reason, count in ordered[:10])


def _parallel_recommendation(run_summary: dict[str, Any]) -> str:
    workers = int(run_summary.get("llm_workers") or 1)
    failure_count = int(run_summary.get("failure_count") or 0)
    if workers > 4:
        return "Reduce llm_workers; local qwen3:8b may be slower or unstable above 4 workers."
    if failure_count > 0:
        return "Keep or reduce llm_workers until failures are reviewed."
    if workers == 1:
        return "Benchmark llm_workers=2 before increasing concurrency."
    return "Current worker count is acceptable if the machine remains responsive."


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Qwen3:8b MD&A/news scoring.")
    parser.add_argument("--doc-type", choices=["mda", "news"], required=True)
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=int(DEFAULT_ENV["OLLAMA_SEED"]))
    parser.add_argument("--model", default=DEFAULT_ENV["OLLAMA_MODEL"])
    parser.add_argument("--base-url", default=DEFAULT_ENV["OLLAMA_BASE_URL"])
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--max-retries", type=int, default=int(DEFAULT_ENV["OLLAMA_MAX_RETRIES"]))
    parser.add_argument("--timeout-seconds", type=int, default=int(DEFAULT_ENV["OLLAMA_TIMEOUT_SECONDS"]))
    parser.add_argument("--temperature", type=float, default=float(DEFAULT_ENV["OLLAMA_TEMPERATURE"]))
    parser.add_argument("--num-ctx", type=int, default=int(DEFAULT_ENV["OLLAMA_NUM_CTX"]))
    parser.add_argument("--cpu-workers", type=int, default=int(os.environ.get("SCORING_CPU_WORKERS", max(1, (os.cpu_count() or 2) - 1))))
    parser.add_argument("--llm-workers", type=int, default=int(os.environ.get("SCORING_LLM_WORKERS", 1)))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--include-manual-required", type=parse_bool, default=False)
    parser.add_argument("--scoring-strategy", choices=["single-dimension", "single_dimension", "dimensionwise"], default="single-dimension")
    args = parser.parse_args()
    if not weight_sum_ok(args.doc_type):
        raise SystemExit(f"{args.doc_type} scorebook weights do not sum to 1")
    if args.doc_type in {"mda", "news"}:
        from run_single_dimension_scoring import run_single_dimension_scoring

        summary = run_single_dimension_scoring(
            args.doc_type,
            root=SCORING_ROOT,
            output_name="single_dimension",
            ratings_prefix=args.doc_type,
            model=args.model,
            llm_workers=args.llm_workers,
            resume=args.resume,
            overwrite=args.overwrite,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout_seconds,
            base_url=args.base_url,
            temperature=args.temperature,
            seed=args.seed,
            num_ctx=args.num_ctx,
            limit=args.limit if args.limit is not None else (3 if args.mode == "pilot" else None),
        )
        summary["mode"] = args.mode
        summary["cli_entrypoint"] = "run_scoring.py"
        summary["scoring_strategy"] = "single_dimension"
        summary["legacy_single_call_disabled"] = True
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0
    from parallel_scoring import run_parallel_pipeline

    summary = run_parallel_pipeline(
        doc_type=args.doc_type,
        mode=args.mode,
        root=SCORING_ROOT,
        limit=args.limit,
        cpu_workers=args.cpu_workers,
        llm_workers=args.llm_workers,
        batch_size=args.batch_size,
        resume=args.resume,
        overwrite=args.overwrite,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
        seed=args.seed,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        include_manual_required=args.include_manual_required,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
