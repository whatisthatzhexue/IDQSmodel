from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from filter_news_common import SCORING_ROOT, locator_text, write_markdown_report
from llm_clients import OllamaClient
from run_filter_news_final_scoring import build_dimension_prompt, call_ollama_dimension
from scoring_utils import DEFAULT_ENV, dimensions_for
from scoring_utils import read_csv_rows


def run_filter_news_stability(root: Path = SCORING_ROOT, repeatability_sample_size: int = 5) -> dict[str, Any]:
    registry = [row for row in read_csv_rows(root / "01_registry" / "news_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    documents = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    ratings = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv")
    failed = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_failed_cases.csv")
    expected_ratings = len(registry) * 5
    operational_completion_rate = (len(ratings) / expected_ratings) if expected_ratings else 0.0
    schema_success_rate = (
        sum(1 for row in ratings if row.get("parse_status") == "parsed" and row.get("schema_validation_status") == "valid") / len(ratings)
        if ratings
        else 0.0
    )
    evidence_matches = 0
    for row in ratings:
        text_path = next((item.get("text_path", "") for item in registry if item.get("document_id") == row.get("document_id")), "")
        if text_path and locator_text((root / text_path).read_text(encoding="utf-8", errors="replace"), row.get("evidence_locator", "")):
            evidence_matches += 1
    evidence_match_rate = evidence_matches / len(ratings) if ratings else 0.0
    json_parse_failure_count = sum(1 for row in failed if row.get("failure_type") == "json_parse_failed")
    json_parse_failure_rate = json_parse_failure_count / expected_ratings if expected_ratings else 0.0
    if repeatability_sample_size <= 0:
        repeatability_status = "not_completed"
        repeatability_pass_rate: str | float = ""
    elif len(documents) < 10:
        repeatability_status = "insufficient_sample"
        repeatability_pass_rate = ""
    else:
        repeatability = run_repeatability_check(root, registry, ratings, repeatability_sample_size)
        repeatability_status = repeatability["status"]
        repeatability_pass_rate = repeatability["pass_rate"]
    synthetic_count = sum(1 for row in registry if row.get("synthetic_flag", "").lower() == "true")
    news_ready = (
        len(documents) > 0
        and synthetic_count == 0
        and operational_completion_rate >= 0.95
        and schema_success_rate >= 0.98
        and evidence_match_rate >= 0.98
    )
    summary = {
        "news_registry_rows": len(registry),
        "news_final_dataset_size": len(documents),
        "synthetic_count": synthetic_count,
        "operational_completion_rate": round(operational_completion_rate, 6),
        "schema_success_rate": round(schema_success_rate, 6),
        "evidence_match_rate": round(evidence_match_rate, 6),
        "json_parse_failure_rate": round(json_parse_failure_rate, 6),
        "repeatability_pass_rate": repeatability_pass_rate,
        "repeatability_status": repeatability_status,
        "failed_cases_count": len(failed),
        "news_ready": news_ready,
    }
    extra = []
    if repeatability_sample_size <= 0:
        extra.append("News repeatability was not rerun in this package; use not_completed rather than interpreting it as passed.")
    elif len(documents) < 10:
        extra.append("Repeatability is not a hard gate because valid real News sample size is below 10.")
    if not news_ready:
        extra.append("News ready gate is false under the final real-News criteria.")
    write_markdown_report(
        root / "11_stability_analysis" / "reports" / "news_stability_report.md",
        "News Stability Report",
        summary.items(),
        extra,
    )
    return summary


def run_repeatability_check(
    root: Path,
    registry: list[dict[str, str]],
    ratings: list[dict[str, str]],
    sample_size: int,
) -> dict[str, Any]:
    rating_index = {(row.get("document_id"), row.get("dimension_code")): row for row in ratings}
    sample = registry[:sample_size]
    if not sample:
        return {"status": "insufficient_sample", "pass_rate": ""}
    raw_dir = root / "11_stability_analysis" / "outputs" / "news_repeatability_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(
        base_url=DEFAULT_ENV["OLLAMA_BASE_URL"],
        model=DEFAULT_ENV["OLLAMA_MODEL"],
        timeout=DEFAULT_ENV["OLLAMA_TIMEOUT_SECONDS"],
        temperature=0,
        seed=42,
        num_ctx=4096,
        num_predict=192,
    )
    status = client.ensure_model_available()
    if not status.get("available") or not status.get("model_available"):
        return {"status": "model_unavailable", "pass_rate": ""}
    checks = 0
    passes = 0
    for row in sample:
        document_id = row.get("document_id", "")
        text_path = root / row.get("text_path", f"02_extracted_text/news_clean/{document_id}.txt")
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        for dimension in dimensions_for("news"):
            baseline = rating_index.get((document_id, dimension["code"]))
            if not baseline:
                continue
            checks += 1
            payload = call_ollama_dimension(client, build_dimension_prompt(dimension, row, text))
            (raw_dir / f"{document_id}_{dimension['code']}_repeat_raw.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            parsed = payload.get("parsed") if isinstance(payload.get("parsed"), dict) else payload
            if isinstance(parsed, dict) and str(parsed.get("score")) == str(baseline.get("raw_score")):
                passes += 1
    if checks == 0:
        return {"status": "not_measured", "pass_rate": ""}
    return {"status": "measured", "pass_rate": round(passes / checks, 6)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate final real-News scoring stability gates.")
    parser.add_argument("--repeatability-sample-size", type=int, default=5)
    args = parser.parse_args()
    summary = run_filter_news_stability(repeatability_sample_size=args.repeatability_sample_size)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
