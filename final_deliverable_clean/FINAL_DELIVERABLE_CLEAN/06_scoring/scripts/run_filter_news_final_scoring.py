from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from filter_news_common import FAILED_CASE_HEADERS, SCORING_ROOT, locator_text, normalize_space, text_locators, write_markdown_report
from llm_clients import OllamaClient, OllamaClientError
from scoring_utils import (
    DEFAULT_ENV,
    FLAG_FIELDS,
    NEWS_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    SCOREBOOK_VERSION,
    dimensions_for,
    grade_label,
    read_csv_rows,
    standardize_score,
    weighted_score,
    weights_for,
    write_csv_rows,
)


PROMPT_VERSION = "news_single_dimension_v3_strict_calibration"
MAX_NEWS_INPUT_CHARS = 11500

NEWS_SCORE_SCHEMA = {
    "type": "object",
    "required": ["score", "evidence", "reason"],
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "evidence": {"type": "string"},
        "reason": {"type": "string"},
    },
    "additionalProperties": False,
}

Scorer = Callable[[dict[str, Any], dict[str, str], str], dict[str, Any]]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_filter_news_final_scoring(
    root: Path = SCORING_ROOT,
    scorer: Scorer | None = None,
    model: str = DEFAULT_ENV["OLLAMA_MODEL"],
    base_url: str = DEFAULT_ENV["OLLAMA_BASE_URL"],
) -> dict[str, Any]:
    registry_rows = [row for row in read_csv_rows(root / "01_registry" / "news_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    output_dir = root / "06_ratings" / "news_final_full"
    doc_path = output_dir / "news_final_document_scores.csv"
    ratings_path = output_dir / "news_final_ratings_long.csv"
    failed_path = output_dir / "news_final_failed_cases.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = build_runtime_config(model=model, base_url=base_url)
    write_qwen_runtime_report(root, runtime_config, {"status": "starting", "news_registry_rows": len(registry_rows)})
    write_qwen_limitations_report(root)

    if not registry_rows:
        write_csv_rows(doc_path, NEWS_DOCUMENT_HEADERS, [])
        write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, [])
        write_csv_rows(failed_path, FAILED_CASE_HEADERS, [])
        summary = {
            "news_registry_rows": 0,
            "document_score_rows": 0,
            "ratings_long_rows": 0,
            "failed_cases": 0,
            "blocked": True,
            "blocker": "No valid real News registry rows. News scoring skipped without fabricated scores.",
        }
        write_scoring_report(root, summary)
        write_qwen_runtime_report(root, runtime_config, summary)
        write_context_truncation_audit(root, [])
        return summary

    if scorer is None:
        client = OllamaClient(
            base_url=base_url,
            model=model,
            timeout=DEFAULT_ENV["OLLAMA_TIMEOUT_SECONDS"],
            temperature=runtime_config["temperature"],
            seed=runtime_config["seed"],
            num_ctx=runtime_config["num_ctx"],
            num_predict=runtime_config["num_predict"],
        )
        runtime_config = client.runtime_config(think=False, stream=False)
        runtime_config.update(
            {
                "scorebook_version": SCOREBOOK_VERSION,
                "prompt_version": PROMPT_VERSION,
                "num_predict": int(runtime_config["num_predict"]),
            }
        )
        status = client.ensure_model_available()
        if not status.get("available") or not status.get("model_available"):
            write_csv_rows(doc_path, NEWS_DOCUMENT_HEADERS, [])
            write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, [])
            failed = [
                {
                    "document_id": row.get("document_id", ""),
                    "ticker": row.get("ticker", ""),
                    "company_name": row.get("company_name", ""),
                    "dimension_code": dimension["code"],
                    "failure_type": "model_unavailable",
                    "failure_reason": status.get("error") or "Local Ollama qwen3:8b unavailable.",
                    "raw_output_path": "",
                }
                for row in registry_rows
                for dimension in dimensions_for("news")
            ]
            write_csv_rows(failed_path, FAILED_CASE_HEADERS, failed)
            summary = {
                "news_registry_rows": len(registry_rows),
                "document_score_rows": 0,
                "ratings_long_rows": 0,
                "failed_cases": len(failed),
                "blocked": True,
                "blocker": "Local Ollama qwen3:8b unavailable. No fabricated News scores written.",
            }
            write_scoring_report(root, summary)
            write_qwen_runtime_report(root, runtime_config, summary | {"model_status": status})
            write_context_truncation_audit(root, [])
            return summary

        def scorer(dimension: dict[str, Any], row: dict[str, str], text: str) -> dict[str, Any]:
            prompt = build_dimension_prompt(dimension, row, text)
            return call_ollama_dimension(client, prompt)

    ratings: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    truncation_rows: list[dict[str, Any]] = []
    raw_dir = root / "05_raw_model_outputs" / "news_final_full"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dimensions = dimensions_for("news")
    weights = weights_for("news")

    for row in registry_rows:
        document_id = row.get("document_id", "")
        text_path = root / row.get("text_path", f"02_extracted_text/news_clean/{document_id}.txt")
        if not text_path.exists():
            for dimension in dimensions:
                failed.append(failed_case(row, dimension, "missing_text_file", str(text_path), ""))
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        truncation_rows.append(context_truncation_row(row, text))
        locators = text_locators(text)
        raw_scores: dict[str, float] = {}
        doc_rating_rows: list[dict[str, Any]] = []

        for dimension in dimensions:
            started = now_iso()
            raw_output_path = raw_dir / f"{document_id}_{dimension['code']}_raw.json"
            call_runtime = dict(runtime_config)
            call_runtime.update({"scoring_timestamp": started, "scorebook_version": SCOREBOOK_VERSION, "prompt_version": PROMPT_VERSION})
            try:
                payload = scorer(dimension, row, text)
            except Exception as exc:
                raw_output_path.write_text(
                    json.dumps({**call_runtime, "runtime_config": call_runtime, "error": str(exc)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                failed.append(failed_case(row, dimension, "model_call_failed", str(exc), str(raw_output_path)))
                continue
            raw_payload = {
                **call_runtime,
                "runtime_config": call_runtime,
                "scorebook_version": SCOREBOOK_VERSION,
                "prompt_version": PROMPT_VERSION,
                "scoring_timestamp": started,
                "document_id": document_id,
                "dimension_code": dimension["code"],
                "model_payload": payload,
            }
            raw_output_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else payload
            if not isinstance(parsed, dict):
                failed.append(failed_case(row, dimension, "json_parse_failed", "Payload is not a JSON object.", str(raw_output_path)))
                continue
            try:
                raw_score = int(parsed.get("score"))
            except Exception:
                failed.append(failed_case(row, dimension, "schema_validation_failed", "Missing integer score.", str(raw_output_path)))
                continue
            evidence = normalize_space(parsed.get("evidence")).strip("[]")
            reason = normalize_space(parsed.get("reason"))
            if raw_score < 1 or raw_score > 5 or not evidence or not reason:
                failed.append(failed_case(row, dimension, "schema_validation_failed", "Score/evidence/reason schema invalid.", str(raw_output_path)))
                continue
            if evidence not in locators:
                failed.append(failed_case(row, dimension, "evidence_match_failed", evidence, str(raw_output_path)))
                continue
            finished = now_iso()
            std = standardize_score(raw_score)
            weighted = weighted_score(raw_score, dimension["weight"])
            raw_scores[dimension["code"]] = raw_score
            doc_rating_rows.append(
                {
                    "rating_id": f"{document_id}_{dimension['code']}",
                    "document_id": document_id,
                    "doc_type": "news",
                    "scoring_basis": "news_quality",
                    "purpose_track": "news_general_quality",
                    "review_round": "filter_news_final",
                    "reviewer_id": "qwen3:8b",
                    "dimension_code": dimension["code"],
                    "dimension_name": dimension["name"],
                    "raw_score": raw_score,
                    "scale_min": 1,
                    "scale_max": 5,
                    "weight_value": dimension["weight"],
                    "std_score_100": std,
                    "weighted_score": weighted,
                    "evidence_locator": evidence,
                    "evidence_text_short": locator_text(text, evidence),
                    "confidence_level": "model",
                    "comment_short": reason,
                    "missing_type": "",
                    "started_at": started,
                    "finished_at": finished,
                    "scorebook_version": SCOREBOOK_VERSION,
                    "model_name": runtime_config["model_name"],
                    "llm_backend": runtime_config["llm_backend"],
                    "temperature": runtime_config["temperature"],
                    "seed": runtime_config["seed"],
                    "num_ctx": runtime_config["num_ctx"],
                    "num_predict": runtime_config["num_predict"],
                    "think": str(runtime_config["think"]).lower(),
                    "stream": str(runtime_config["stream"]).lower(),
                    "json_schema_mode": runtime_config["json_schema_mode"],
                    "prompt_version": PROMPT_VERSION,
                    "scoring_timestamp": started,
                    "raw_output_path": str(raw_output_path.relative_to(root)),
                    "numeric_check_path": "",
                    "parse_status": "parsed",
                    "schema_validation_status": "valid",
                }
            )

        ratings.extend(doc_rating_rows)
        if len(raw_scores) == len(dimensions):
            total = round(sum(weighted_score(score, weights[code]) for code, score in raw_scores.items()), 6)
            document = {
                "document_id": document_id,
                "doc_type": "news",
                "company_name": row.get("company_name", ""),
                "ticker": row.get("ticker", ""),
                "source_name": row.get("source_name", ""),
                "publish_date": row.get("publish_date", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "confidence_tier": row.get("confidence_tier", ""),
                "relevance_status": row.get("relevance_status", ""),
                "review_reason": row.get("review_reason", ""),
                "dimension_count": len(dimensions),
                "total_score_100": total,
                "grade_label": grade_label(total),
                "adjudication_flag": "No",
                "needs_review": row.get("needs_manual_review", "No") or "No",
                "review_reasons": row.get("review_reason", ""),
                "final_version": "filter_news_final",
                "scorebook_version": SCOREBOOK_VERSION,
                "model_name": runtime_config["model_name"],
                "llm_backend": runtime_config["llm_backend"],
                "temperature": runtime_config["temperature"],
                "seed": runtime_config["seed"],
                "num_ctx": runtime_config["num_ctx"],
                "num_predict": runtime_config["num_predict"],
                "think": str(runtime_config["think"]).lower(),
                "stream": str(runtime_config["stream"]).lower(),
                "json_schema_mode": runtime_config["json_schema_mode"],
                "prompt_version": PROMPT_VERSION,
                "scoring_timestamp": now_iso(),
            }
            for dimension in dimensions:
                code = dimension["code"]
                document[code] = raw_scores[code]
                document[f"{code}_std"] = standardize_score(raw_scores[code])
            for flag in FLAG_FIELDS:
                document[flag] = "0"
            documents.append(document)

    write_csv_rows(doc_path, NEWS_DOCUMENT_HEADERS, documents)
    write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, ratings)
    write_csv_rows(failed_path, FAILED_CASE_HEADERS, failed)
    write_context_truncation_audit(root, truncation_rows)
    summary = {
        "news_registry_rows": len(registry_rows),
        "document_score_rows": len(documents),
        "ratings_long_rows": len(ratings),
        "failed_cases": len(failed),
        "blocked": len(documents) == 0,
        "blocker": "" if documents else "No complete News document scores were produced.",
    }
    write_scoring_report(root, summary)
    write_qwen_runtime_report(root, runtime_config, summary)
    write_qwen_limitations_report(root)
    return summary


def call_ollama_dimension(client: OllamaClient, prompt: str) -> dict[str, Any]:
    try:
        result = client.chat_json(
            [
                {"role": "system", "content": "Return JSON only. Do not include markdown or explanations outside JSON."},
                {"role": "user", "content": prompt},
            ],
            NEWS_SCORE_SCHEMA,
            think=False,
        )
    except OllamaClientError as exc:
        raise RuntimeError(str(exc)) from exc
    return {"raw_response": result.raw_response, "content": result.content, "parsed": result.parsed_json}


def build_dimension_prompt(dimension: dict[str, Any], row: dict[str, str], text: str) -> str:
    metadata = {key: row.get(key, "") for key in ["document_id", "ticker", "company_name", "publish_date", "source_name", "title", "url", "confidence_tier", "needs_manual_review", "review_reason"]}
    criteria = {
        "N01": "Factual accuracy and internal consistency: check whether the article is internally coherent and uses factual claims carefully.",
        "N02": "Source transparency and verifiability: reward clear attribution, source names, and verifiable facts.",
        "N03": "Balance and objectivity: reward neutral wording and balanced treatment of claims.",
        "N04": "Relevance and information increment: reward company/business/F&B information that adds substance beyond generic market noise.",
        "N05": "Expression clarity and headline fit: reward clear prose and alignment between headline and article body.",
    }
    return (
        "Score one real News article for the specified dimension only.\n"
        "Return exactly JSON: {\"score\": 1-5, \"evidence\": \"NEWS_P001\", \"reason\": \"short reason\"}.\n"
        "Use one existing locator only: TITLE or NEWS_Pxxx. Do not invent evidence.\n\n"
        "Strict calibration anchors:\n"
        "- Score 5 is exceptional, not the default. Give 5 only when the dimension is strongly satisfied with concrete article evidence.\n"
        "- Score 4 means solid but not perfect. Use 4 for ordinary good reporting with some limits.\n"
        "- Score 3 means adequate/basic or mixed. Use 3 for CSR, sponsorship, event, metro, lifestyle, single-source, or low-information stories unless the dimension clearly deserves more.\n"
        "- Score 2 means weak, thin, mostly promotional, or poorly verifiable. Score 1 means unusable for that dimension.\n"
        "- N01 factual accuracy means internal consistency and verifiable cues only; it does not prove external facts are true.\n"
        "- N02 should not receive 5 for single-source stories without official announcement links, filings, named third-party sources, or independently checkable details.\n"
        "- N03 should not receive 5 for company-only claims, press-release style reporting, or one-sided promotional tone.\n"
        "- N04 for CSR, sponsorship, community, event, metro, or lifestyle stories should be lower than finance, operations, expansion, regulatory, product, management, market-performance, or substantive ESG news.\n"
        "- N05 headline clarity alone must not inflate scores for N01-N04 or the total quality impression.\n\n"
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"Dimension: {dimension['code']} - {dimension['name']} (weight {dimension['weight']})\n"
        f"Criteria: {criteria.get(dimension['code'], dimension['name'])}\n\n"
        "Article text:\n"
        + text[:MAX_NEWS_INPUT_CHARS]
    )


def failed_case(row: dict[str, str], dimension: dict[str, Any], failure_type: str, reason: str, raw_path: str) -> dict[str, str]:
    return {
        "document_id": row.get("document_id", ""),
        "ticker": row.get("ticker", ""),
        "company_name": row.get("company_name", ""),
        "dimension_code": dimension["code"],
        "failure_type": failure_type,
        "failure_reason": reason,
        "raw_output_path": raw_path,
    }


def write_scoring_report(root: Path, summary: dict[str, Any]) -> Path:
    extra = []
    if summary.get("blocked"):
        extra.extend(["## Blocker", str(summary.get("blocker", ""))])
    return write_markdown_report(
        root / "08_reports" / "filter_news_final_scoring_report.md",
        "Filter News Final Scoring Report",
        summary.items(),
        extra,
    )


def build_runtime_config(model: str, base_url: str) -> dict[str, Any]:
    def env_number(name: str, default: str, cast: Callable[[str], Any]) -> Any:
        value = os.environ.get(name, default)
        try:
            return cast(value)
        except (TypeError, ValueError):
            return cast(default)

    return {
        "model_name": model or os.environ.get("OLLAMA_MODEL", DEFAULT_ENV["OLLAMA_MODEL"]),
        "llm_backend": os.environ.get("SCORING_LLM_BACKEND", "ollama"),
        "base_url": base_url or os.environ.get("OLLAMA_BASE_URL", DEFAULT_ENV["OLLAMA_BASE_URL"]),
        "temperature": env_number("OLLAMA_TEMPERATURE", "0", float),
        "seed": env_number("OLLAMA_SEED", "42", int),
        "num_ctx": env_number("OLLAMA_NUM_CTX", "4096", int),
        "num_predict": env_number("OLLAMA_NUM_PREDICT", "192", int),
        "think": False,
        "stream": False,
        "json_schema_mode": "ollama_format_schema",
        "scorebook_version": SCOREBOOK_VERSION,
        "prompt_version": PROMPT_VERSION,
    }


def context_truncation_row(row: dict[str, str], text: str) -> dict[str, Any]:
    text_chars = len(text)
    truncated_chars = max(0, text_chars - MAX_NEWS_INPUT_CHARS)
    return {
        "document_id": row.get("document_id", ""),
        "ticker": row.get("ticker", ""),
        "text_chars": text_chars,
        "max_input_chars": MAX_NEWS_INPUT_CHARS,
        "input_chars_used": min(text_chars, MAX_NEWS_INPUT_CHARS),
        "truncated_flag": "1" if truncated_chars else "0",
        "truncated_chars": truncated_chars,
        "truncated_ratio": round(truncated_chars / text_chars, 6) if text_chars else 0,
    }


def write_context_truncation_audit(root: Path, rows: list[dict[str, Any]]) -> None:
    write_csv_rows(
        root / "08_reports" / "news_context_truncation_audit.csv",
        ["document_id", "ticker", "text_chars", "max_input_chars", "input_chars_used", "truncated_flag", "truncated_chars", "truncated_ratio"],
        rows,
    )


def write_qwen_runtime_report(root: Path, runtime_config: dict[str, Any], summary: dict[str, Any]) -> None:
    (root / "08_reports").mkdir(parents=True, exist_ok=True)
    lines = [
        "# News qwen3:8b Runtime Configuration",
        "",
        f"- model_name: {runtime_config.get('model_name')}",
        f"- llm_backend: {runtime_config.get('llm_backend')}",
        f"- base_url: {runtime_config.get('base_url')}",
        f"- temperature: {runtime_config.get('temperature')}",
        f"- seed: {runtime_config.get('seed')}",
        f"- num_ctx: {runtime_config.get('num_ctx')}",
        f"- num_predict: {runtime_config.get('num_predict')}",
        f"- think: {runtime_config.get('think')}",
        f"- stream: {runtime_config.get('stream')}",
        f"- JSON schema mode: {runtime_config.get('json_schema_mode')}",
        f"- scorebook_version: {runtime_config.get('scorebook_version')}",
        f"- prompt_version: {runtime_config.get('prompt_version')}",
        "",
        "## Scoring Summary",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    (root / "08_reports" / "news_qwen_runtime_config.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qwen_limitations_report(root: Path) -> None:
    (root / "08_reports").mkdir(parents=True, exist_ok=True)
    lines = [
        "# qwen3:8b News Scoring Limitations",
        "",
        "1. qwen3:8b is a local 8B-scale model, not a human fact checker.",
        "2. News N01 factual accuracy checks internal consistency and verifiable cues in the article text; it cannot prove that external facts are true.",
        "3. The model may misread English news, Malaysian company names, tickers, brand names, and multi-company articles.",
        "4. JSON output failure, evidence-locator mismatch, high-score bias, and ceiling effects remain possible.",
        f"5. Long news can be truncated because the current prompt uses text[:{MAX_NEWS_INPUT_CHARS}]; see 08_reports/news_context_truncation_audit.csv.",
        "6. Low-confidence News can enter the main model only with needs_manual_review=Yes; high-confidence samples should be used as a robustness sample.",
        "7. temperature=0 and seed=42 improve reproducibility, but they do not remove model bias or guarantee factual truth.",
    ]
    (root / "08_reports" / "qwen3_8b_limitations_news.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final single-dimension scoring for real Filter News articles.")
    parser.add_argument("--model", default=DEFAULT_ENV["OLLAMA_MODEL"])
    parser.add_argument("--base-url", default=DEFAULT_ENV["OLLAMA_BASE_URL"])
    args = parser.parse_args()
    summary = run_filter_news_final_scoring(model=args.model, base_url=args.base_url)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
