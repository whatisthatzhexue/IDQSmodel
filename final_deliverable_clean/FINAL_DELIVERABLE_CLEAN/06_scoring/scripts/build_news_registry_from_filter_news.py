from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from filter_news_common import (
    FILTER_NEWS_REGISTRY_HEADERS,
    SCORING_ROOT,
    normalize_date,
    normalize_space,
    raw_row_key as build_raw_row_key,
    safe_id_part,
    sha256_text,
    short_hash,
    write_markdown_report,
)
from scoring_utils import read_csv_rows, write_csv_rows


def build_news_registry_from_filter_news(root: Path = SCORING_ROOT) -> dict[str, Any]:
    audit_path = root / "08_reports" / "filter_news_relevance_audit.csv"
    raw_path = root / "input_news" / "news_raw_from_filter_news.csv"
    output_path = root / "01_registry" / "news_registry.csv"
    high_confidence_path = root / "01_registry" / "news_registry_high_confidence.csv"
    audit_rows = read_csv_rows(audit_path)
    raw_rows = read_csv_rows(raw_path)
    raw_index = index_raw_rows(raw_rows)
    selection_flag = registry_selection_flag(audit_rows)
    registry_rows: list[dict[str, Any]] = []
    missing_text_count = 0
    missing_raw_count = 0

    for row in audit_rows:
        if normalize_space(row.get(selection_flag)).lower() != "yes":
            continue
        raw = raw_index.get(normalize_space(row.get("raw_row_key")), {})
        if not raw:
            missing_raw_count += 1
        article_text = normalize_space(raw.get("article_text"))
        if not article_text:
            missing_text_count += 1
            continue
        article_hash = sha256_text(article_text)
        ticker = safe_id_part(row.get("ticker", "UNKNOWN")).upper()
        publish_date = normalize_date(row.get("publish_date")) or "UNKNOWN_DATE"
        date_part = safe_id_part(publish_date)
        stable_hash = short_hash(article_hash + normalize_space(row.get("raw_row_key")), 10)
        document_id = f"NEWS_{ticker}_{date_part}_{stable_hash}"
        registry_rows.append(
            {
                "document_id": document_id,
                "doc_type": "news",
                "ticker": ticker,
                "company_name": row.get("company_name", ""),
                "publish_date": publish_date,
                "source_name": row.get("source_name", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "article_text_hash": article_hash,
                "text_length": len(article_text),
                "synthetic_flag": "false",
                "include_flag": "Yes",
                "relevance_status": row.get("relevance_status", ""),
                "confidence_tier": row.get("confidence_tier", ""),
                "needs_manual_review": row.get("needs_manual_review", ""),
                "review_reason": row.get("review_reason", ""),
                "shared_article_flag": row.get("shared_article_flag", "0"),
                "multi_company_article_flag": row.get("multi_company_article_flag", "0"),
                "original_file": row.get("original_file", ""),
                "original_file_hash": row.get("original_file_hash", ""),
                "raw_row_key": row.get("raw_row_key", ""),
                "text_path": f"02_extracted_text/news_clean/{document_id}.txt",
                "notes": f"source=Filter news real article text; synthetic_flag=false; registry_selection_flag={selection_flag}",
            }
        )

    high_confidence_rows = [row for row in registry_rows if row.get("confidence_tier") == "high"]
    write_csv_rows(output_path, FILTER_NEWS_REGISTRY_HEADERS, registry_rows)
    write_csv_rows(high_confidence_path, FILTER_NEWS_REGISTRY_HEADERS, high_confidence_rows)
    summary = {
        "audit_rows": len(audit_rows),
        "main_model_include_rows": sum(1 for row in audit_rows if normalize_space(row.get("main_model_include_flag")).lower() == "yes"),
        "broad_include_rows": sum(1 for row in audit_rows if normalize_space(row.get("broad_include_flag")).lower() == "yes"),
        "strict_company_focus_rows": sum(1 for row in audit_rows if normalize_space(row.get("strict_company_focus_flag")).lower() == "yes"),
        "registry_selection_flag": selection_flag,
        "registry_rows": len(registry_rows),
        "high_confidence_registry_rows": len(high_confidence_rows),
        "missing_raw_rows": missing_raw_count,
        "missing_text_rows": missing_text_count,
        "synthetic_count": 0,
        "registry_output": str(output_path),
        "high_confidence_registry_output": str(high_confidence_path),
    }
    extra = []
    if not registry_rows:
        extra.append("## Blocker")
        extra.append(f"No {selection_flag}=Yes real News rows survived registry construction.")
    else:
        extra.extend(
            [
                "## Registry Selection",
                f"- registry_selection_flag: {selection_flag}",
                "- main_model_include_rows is the strict model-gate count from the audit, not necessarily the broad sample count.",
                "- broad_include_rows is the broad manually-preserved News sample used to rebuild 01_registry/news_registry.csv.",
            ]
        )
    write_markdown_report(
        root / "08_reports" / "filter_news_registry_report.md",
        "Filter News Registry Report",
        summary.items(),
        extra,
    )
    return summary


def index_raw_rows(raw_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for idx, row in enumerate(raw_rows, start=1):
        index[build_raw_row_key(row, idx)] = row
    return index


def registry_selection_flag(audit_rows: list[dict[str, str]]) -> str:
    broad_count = sum(1 for row in audit_rows if normalize_space(row.get("broad_include_flag")).lower() == "yes")
    if broad_count > 0:
        return "broad_include_flag"
    return "main_model_include_flag"


def raw_key(row: dict[str, str]) -> str:
    return normalize_space(row.get("raw_row_key")) or build_raw_row_key(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real News registry from Filter News relevance audit.")
    parser.parse_args()
    summary = build_news_registry_from_filter_news()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
