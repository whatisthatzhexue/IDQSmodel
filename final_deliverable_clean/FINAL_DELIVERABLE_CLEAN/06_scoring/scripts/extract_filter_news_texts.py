from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_news_registry_from_filter_news import index_raw_rows
from filter_news_common import SCORING_ROOT, normalize_space, render_news_text, write_markdown_report
from scoring_utils import read_csv_rows, write_csv_rows


FAILED_EXTRACTION_HEADERS = ["document_id", "ticker", "raw_row_key", "failure_reason"]


def extract_filter_news_texts(root: Path = SCORING_ROOT) -> dict[str, Any]:
    registry_path = root / "01_registry" / "news_registry.csv"
    raw_path = root / "input_news" / "news_raw_from_filter_news.csv"
    output_dir = root / "02_extracted_text" / "news_clean"
    legacy_dir = root / "02_extracted_text" / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = read_csv_rows(raw_path)
    raw_index = index_raw_rows(raw_rows)
    registry_rows = [row for row in read_csv_rows(registry_path) if row.get("include_flag", "").lower() == "yes"]
    expected_names = {f"{row.get('document_id')}.txt" for row in registry_rows}
    archived_stale = archive_stale_news_clean_files(root, output_dir, expected_names)
    written = 0
    missing = 0
    locator_count = 0
    failed: list[dict[str, str]] = []

    for row in registry_rows:
        document_id = row.get("document_id", "")
        raw = raw_index.get(normalize_space(row.get("raw_row_key")), {})
        text = normalize_space(raw.get("article_text"))
        if not text:
            missing += 1
            failed.append(
                {
                    "document_id": document_id,
                    "ticker": row.get("ticker", ""),
                    "raw_row_key": row.get("raw_row_key", ""),
                    "failure_reason": "raw_row_key not found or article_text empty",
                }
            )
            continue
        rendered = render_news_text(
            row.get("title", ""),
            text,
            {
                "ticker": row.get("ticker", ""),
                "company_name": row.get("company_name", ""),
                "publish_date": row.get("publish_date", ""),
                "source_name": row.get("source_name", ""),
                "url": row.get("url", ""),
                "confidence_tier": row.get("confidence_tier", ""),
            },
        )
        path = root / row.get("text_path", f"02_extracted_text/news_clean/{document_id}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        legacy_path = legacy_dir / f"{document_id}.txt"
        if legacy_path != path:
            shutil.copy2(path, legacy_path)
        written += 1
        locator_count += rendered.count("[NEWS_P")

    write_csv_rows(root / "08_reports" / "news_text_extraction_failed_cases.csv", FAILED_EXTRACTION_HEADERS, failed)
    count_match = len(registry_rows) == len(list(output_dir.glob("*.txt"))) == written
    summary = {
        "registry_rows": len(registry_rows),
        "clean_text_files_written": written,
        "news_clean_txt_count": len(list(output_dir.glob("*.txt"))),
        "missing_raw_text": missing,
        "failed_cases": len(failed),
        "paragraph_locator_count": locator_count,
        "archived_stale_txt_files": archived_stale,
        "registry_text_count_match": count_match,
        "output_dir": str(output_dir),
        "rewrite_or_summary_used": False,
    }
    extra = []
    if not count_match:
        extra.extend(["## Failed Cases", "Registry rows and clean text files do not match; see news_text_extraction_failed_cases.csv."])
    for name in ["filter_news_cleaning_report.md", "news_text_extraction_report.md"]:
        write_markdown_report(root / "08_reports" / name, "Filter News Text Extraction Report", summary.items(), extra)
    return summary


def archive_stale_news_clean_files(root: Path, output_dir: Path, expected_names: set[str]) -> int:
    stale = [path for path in output_dir.glob("*.txt") if path.name not in expected_names]
    if not stale:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = root / "archive" / f"news_clean_pre_manual_preserve_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in stale:
        shutil.move(str(path), str(archive_dir / path.name))
    return len(stale)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract clean paragraph-located News text files from Filter News raw rows.")
    parser.parse_args()
    summary = extract_filter_news_texts()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
