from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from build_real_news_registry import REAL_NEWS_REGISTRY_HEADERS, _clean, _normalize_article_text
from scoring_utils import SCORING_ROOT, read_csv_rows
from stability_common import write_markdown


BOILERPLATE_LINES = {
    "advertisement",
    "advertisements",
    "cookie policy",
    "subscribe now",
    "read more",
    "click here",
    "all rights reserved",
}


def extract_real_news_texts(root: Path = SCORING_ROOT) -> dict[str, Any]:
    registry_rows = [
        row
        for row in read_csv_rows(root / "01_registry" / "news_registry.csv")
        if row.get("include_flag", "").strip().lower() == "yes" and row.get("synthetic_flag", "").strip().lower() == "false"
    ]
    cleaned = 0
    failed = 0
    lengths: list[int] = []
    paragraph_counts: list[int] = []
    short_news_count = 0
    out_dir = root / "02_extracted_text" / "news_clean"

    for row in registry_rows:
        source_row = _find_source_row(root, row)
        if not source_row:
            failed += 1
            continue
        title = row.get("title") or _clean(source_row.get("title", ""))
        article_text = _normalize_article_text(str(source_row.get("article_text", "") or ""))
        paragraphs = _clean_paragraphs(article_text)
        if not paragraphs:
            failed += 1
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{row['document_id']}.txt").write_text(_format_text(title, paragraphs), encoding="utf-8")
        cleaned += 1
        text_len = sum(len(paragraph) for paragraph in paragraphs)
        lengths.append(text_len)
        paragraph_counts.append(len(paragraphs))
        short_news_count += int(text_len < 200)

    summary = {
        "status": "success" if cleaned else "blocked",
        "cleaned_news_count": cleaned,
        "failed_cleaning_count": failed,
        "average_text_length": round(statistics.mean(lengths), 2) if lengths else 0.0,
        "short_news_count": short_news_count,
        "paragraph_count_distribution": _distribution(paragraph_counts),
    }
    _write_report(root, summary)
    return summary


def _find_source_row(root: Path, registry_row: dict[str, str]) -> dict[str, str] | None:
    source_path = root / registry_row.get("file_path", "")
    if not source_path.exists() or source_path.suffix.lower() != ".csv":
        return None
    with source_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        candidates = [dict(row) for row in reader]
    for row in candidates:
        if _row_matches(registry_row, row, strict_url=True):
            return row
    for row in candidates:
        if _row_matches(registry_row, row, strict_url=False):
            return row
    return None


def _row_matches(registry_row: dict[str, str], source_row: dict[str, str], strict_url: bool) -> bool:
    checks = [
        registry_row.get("ticker", "").upper() == _clean(source_row.get("ticker", "")).upper(),
        registry_row.get("title", "").lower() == _clean(source_row.get("title", "")).lower(),
        registry_row.get("source_name", "").lower() == _clean(source_row.get("source_name", "")).lower(),
    ]
    if strict_url:
        checks.append(registry_row.get("url", "") == _clean(source_row.get("url", "")))
    return all(checks)


def _clean_paragraphs(text: str) -> list[str]:
    raw_parts = re.split(r"\n\s*\n+", text.replace("\r", "\n"))
    paragraphs = []
    for part in raw_parts:
        lines = []
        for line in part.split("\n"):
            cleaned = _clean(line)
            if not cleaned:
                continue
            lowered = cleaned.lower().strip(" .:")
            if lowered in BOILERPLATE_LINES:
                continue
            if lowered.startswith("advertisement"):
                continue
            lines.append(cleaned)
        paragraph = " ".join(lines).strip()
        if paragraph:
            paragraphs.append(paragraph)
    if paragraphs:
        return paragraphs
    fallback = _clean(text)
    return [fallback] if fallback else []


def _format_text(title: str, paragraphs: list[str]) -> str:
    lines = ["[TITLE]", title.strip(), ""]
    for index, paragraph in enumerate(paragraphs, start=1):
        lines.extend([f"[NEWS_P{index:03d}]", paragraph, ""])
    return "\n".join(lines).strip() + "\n"


def _distribution(values: list[int]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for value in values:
        key = str(value)
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    lines = ["# News Text Cleaning Report", ""]
    for key in [
        "status",
        "cleaned_news_count",
        "failed_cleaning_count",
        "average_text_length",
        "short_news_count",
        "paragraph_count_distribution",
    ]:
        lines.append(f"- {key}: {summary.get(key)}")
    write_markdown(root / "08_reports" / "news_text_cleaning_report.md", lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract paragraph-addressable clean texts for strict real News registry rows.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(extract_real_news_texts(root=args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
