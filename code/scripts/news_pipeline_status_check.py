from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, ensure_directories, read_csv_rows, save_json
from stability_common import write_markdown


def check_news_pipeline_status(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_directories(root)
    registry_path = root / "01_registry" / "news_registry.csv"
    text_dir = root / "02_extracted_text" / "news_clean"
    raw_dir = root / "05_raw_model_outputs" / "news_final_full"
    scores_path = root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv"
    ratings_path = root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv"
    registry_rows = read_csv_rows(registry_path)
    score_rows = read_csv_rows(scores_path)
    rating_rows = read_csv_rows(ratings_path)
    real_registry_rows = [
        row
        for row in registry_rows
        if row.get("include_flag", "").lower() == "yes" and "synthetic_flag=true" not in row.get("notes", "").lower()
    ]
    real_document_ids = {row.get("document_id", "") for row in real_registry_rows}
    real_score_rows = [row for row in score_rows if row.get("document_id", "") in real_document_ids]
    real_rating_rows = [row for row in rating_rows if row.get("document_id", "") in real_document_ids]
    text_files = sorted(text_dir.glob("*.txt")) if text_dir.exists() else []
    real_text_files = [
        root / row.get("text_path", "")
        for row in real_registry_rows
        if row.get("text_path") and (root / row.get("text_path", "")).exists()
    ]
    raw_files = sorted(path for path in raw_dir.glob("*") if path.is_file()) if raw_dir.exists() else []
    parse_total = len(real_rating_rows)
    parse_failures = sum(row.get("parse_status", "parsed").lower() not in {"", "parsed"} for row in real_rating_rows)
    schema_valid = sum(row.get("schema_validation_status", "valid").lower() in {"", "valid"} for row in real_rating_rows)
    summary = {
        "news_registry": {
            "path": str(registry_path),
            "exists": registry_path.exists(),
            "row_count": len(registry_rows),
            "included_count": sum(row.get("include_flag", "").lower() == "yes" for row in registry_rows),
            "real_included_count": len(real_registry_rows),
            "synthetic_count": sum("synthetic_flag=true" in row.get("notes", "").lower() for row in registry_rows),
        },
        "news_raw": {
            "path": str(raw_dir),
            "exists": raw_dir.exists(),
            "file_count": len(raw_files),
        },
        "news_text_extraction": {
            "path": str(text_dir),
            "exists": text_dir.exists(),
            "row_count": len(text_files),
            "real_text_count": len(real_text_files),
        },
        "news_scoring": {
            "document_scores_path": str(scores_path),
            "ratings_long_path": str(ratings_path),
            "document_score_count": len(score_rows),
            "rating_row_count": len(rating_rows),
            "real_document_score_count": len(real_score_rows),
            "real_rating_row_count": len(real_rating_rows),
            "json_parse_failure_rate": round(parse_failures / parse_total, 6) if parse_total else 1.0,
            "schema_validation_success_rate": round(schema_valid / parse_total, 6) if parse_total else 0.0,
        },
        "news_stability": {
            "can_run": len(real_score_rows) > 0 and len(real_rating_rows) > 0,
            "reason": "" if len(real_score_rows) > 0 and len(real_rating_rows) > 0 else "missing real news scores or ratings_long",
        },
    }
    summary["overall_status"] = _overall_status(summary)
    save_json(root / "08_reports" / "news_pipeline_status_report.json", summary)
    _write_report(root / "08_reports" / "news_pipeline_status_report.md", summary)
    return summary


def _overall_status(summary: dict[str, Any]) -> str:
    if summary["news_registry"]["included_count"] <= 0:
        return "missing_registry"
    if summary["news_registry"]["real_included_count"] <= 0:
        return "synthetic_only"
    if summary["news_text_extraction"]["row_count"] <= 0:
        return "missing_text_extraction"
    if summary["news_text_extraction"]["real_text_count"] <= 0:
        return "missing_real_text_extraction"
    if summary["news_scoring"]["real_document_score_count"] <= 0:
        return "missing_real_scoring"
    if not summary["news_stability"]["can_run"]:
        return "missing_stability_inputs"
    return "ready"


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# News Pipeline Status Report",
        "",
        f"- overall_status: {summary['overall_status']}",
        f"- news_registry rows: {summary['news_registry']['row_count']}",
        f"- news_registry included rows: {summary['news_registry']['included_count']}",
        f"- real included rows: {summary['news_registry']['real_included_count']}",
        f"- synthetic rows: {summary['news_registry']['synthetic_count']}",
        f"- raw file count: {summary['news_raw']['file_count']}",
        f"- extracted text count: {summary['news_text_extraction']['row_count']}",
        f"- real extracted text count: {summary['news_text_extraction']['real_text_count']}",
        f"- news document scores: {summary['news_scoring']['document_score_count']}",
        f"- news ratings rows: {summary['news_scoring']['rating_row_count']}",
        f"- real news document scores: {summary['news_scoring']['real_document_score_count']}",
        f"- real news ratings rows: {summary['news_scoring']['real_rating_row_count']}",
        f"- json_parse_failure_rate: {summary['news_scoring']['json_parse_failure_rate']}",
        f"- schema_validation_success_rate: {summary['news_scoring']['schema_validation_success_rate']}",
        f"- stability can run: {summary['news_stability']['can_run']}",
        f"- stability note: {summary['news_stability']['reason'] or 'none'}",
        "",
    ]
    if summary["overall_status"] != "ready":
        lines.append(
            "News pipeline is missing required final components; rerun audit_filter_news_relevance.py, "
            "build_news_registry_from_filter_news.py, extract_filter_news_texts.py, and "
            "run_filter_news_final_scoring.py before final freeze."
        )
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check News registry, text extraction, scoring, and stability readiness.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(check_news_pipeline_status(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
