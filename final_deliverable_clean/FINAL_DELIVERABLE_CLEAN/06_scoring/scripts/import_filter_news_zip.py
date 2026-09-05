from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from filter_news_common import (
    FILTER_NEWS_CANDIDATES,
    RAW_NEWS_EXTRA_HEADERS,
    RAW_NEWS_HEADERS,
    SCORING_ROOT,
    SUPPORTED_NEWS_SUFFIXES,
    clean_article_text,
    infer_source_name,
    infer_ticker_from_path,
    load_company_lookup,
    normalize_date,
    normalize_space,
    read_table_file,
    sha256_file,
    write_markdown_report,
)
from scoring_utils import write_csv_rows


FIELD_ALIASES = {
    "ticker": ["ticker", "stock", "stock_code", "symbol"],
    "company_name": ["company", "company_name", "listed_company"],
    "publish_date": ["date", "publish_date", "published_at", "published_date", "publication_date"],
    "source_name": ["source", "source_name", "publisher", "publication"],
    "title": ["title", "headline", "summary"],
    "url": ["url", "link"],
    "article_text": ["article_text", "text", "body", "content", "full_text", "news_text"],
}


def find_filter_news_source(candidates: list[Path] | None = None) -> Path | None:
    for path in candidates or FILTER_NEWS_CANDIDATES:
        if path.exists():
            return path
    return None


def prepare_source(source: Path, root: Path = SCORING_ROOT) -> tuple[Path, Path]:
    extract_dir = root / "_tmp_filter_news_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            zf.extractall(extract_dir)
        return source, extract_dir
    return source, source


def import_filter_news(source: Path | None = None, root: Path = SCORING_ROOT) -> dict[str, Any]:
    source = source or find_filter_news_source()
    output_path = root / "input_news" / "news_raw_from_filter_news.csv"
    if source is None:
        write_csv_rows(output_path, RAW_NEWS_HEADERS + RAW_NEWS_EXTRA_HEADERS, [])
        report = {
            "filter_news_found": False,
            "source_path": "",
            "total_files_found": 0,
            "readable": 0,
            "unreadable": 0,
            "total_raw_rows": 0,
            "rows_with_article_text": 0,
            "rows_url_only": 0,
            "rows_missing_title": 0,
            "rows_missing_company": 0,
            "rows_missing_date": 0,
        }
        write_import_report(root, report)
        return report

    source_path, scan_root = prepare_source(source, root)
    company_lookup = load_company_lookup(root)
    files = sorted(path for path in scan_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_NEWS_SUFFIXES)
    rows: list[dict[str, Any]] = []
    unreadable: list[tuple[str, str]] = []
    readable = 0
    file_hashes: dict[Path, str] = {}
    for path in files:
        rel = path.relative_to(scan_root).as_posix()
        try:
            source_rows = read_table_file(path)
            readable += 1
            file_hashes[path] = sha256_file(path)
        except Exception as exc:
            unreadable.append((rel, str(exc)))
            continue
        for index, source_row in enumerate(source_rows, start=1):
            standardized = standardize_row(source_row, rel, company_lookup, file_hashes[path], index)
            rows.append(standardized)

    write_csv_rows(output_path, RAW_NEWS_HEADERS + RAW_NEWS_EXTRA_HEADERS, rows)
    report = {
        "filter_news_found": True,
        "source_path": str(source_path),
        "total_files_found": len(files),
        "readable": readable,
        "unreadable": len(unreadable),
        "total_raw_rows": len(rows),
        "rows_with_article_text": sum(1 for row in rows if row.get("article_text")),
        "rows_url_only": sum(1 for row in rows if row.get("import_status") == "url_only"),
        "rows_missing_title": sum(1 for row in rows if not row.get("title")),
        "rows_missing_company": sum(1 for row in rows if not row.get("company_name")),
        "rows_missing_date": sum(1 for row in rows if not row.get("publish_date")),
        "unreadable_files": unreadable[:50],
        "raw_output": str(output_path),
    }
    write_import_report(root, report)
    return report


def standardize_row(
    row: dict[str, Any],
    original_file: str,
    company_lookup: dict[str, dict[str, str]],
    original_file_hash: str,
    row_number: int,
) -> dict[str, Any]:
    normalized_keys = {normalize_space(key).lower(): value for key, value in row.items() if key is not None}

    def pick(target: str) -> str:
        for key in FIELD_ALIASES[target]:
            if key in normalized_keys:
                return normalize_space(normalized_keys[key])
        return ""

    url = pick("url")
    ticker = pick("ticker") or infer_ticker_from_path(original_file)
    ticker = ticker.upper().replace("&", "N")
    company_name = pick("company_name")
    if not company_name and ticker in company_lookup:
        company_name = company_lookup[ticker].get("company_name", "")
    title = pick("title")
    article_text = clean_article_text(pick("article_text"))
    source_name = pick("source_name") or infer_source_name(url, original_file)
    publish_date = normalize_date(pick("publish_date"))

    status = "article_text"
    if not article_text and url:
        status = "url_only"
    elif not article_text:
        status = "metadata_only"
    return {
        "ticker": ticker,
        "company_name": company_name,
        "publish_date": publish_date,
        "source_name": source_name,
        "title": title,
        "url": url,
        "article_text": article_text if status == "article_text" else "",
        "original_file": original_file,
        "original_file_hash": original_file_hash,
        "import_status": status,
        "raw_source_row": json.dumps({"row_number": row_number}, ensure_ascii=False),
    }


def write_import_report(root: Path, report: dict[str, Any]) -> Path:
    keys = [
        "filter_news_found",
        "source_path",
        "total_files_found",
        "readable",
        "unreadable",
        "total_raw_rows",
        "rows_with_article_text",
        "rows_url_only",
        "rows_missing_title",
        "rows_missing_company",
        "rows_missing_date",
        "raw_output",
    ]
    extra = []
    unreadable = report.get("unreadable_files") or []
    if unreadable:
        extra.append("## First unreadable files")
        extra.extend(f"- {path}: {reason}" for path, reason in unreadable[:20])
    return write_markdown_report(
        root / "08_reports" / "filter_news_import_report.md",
        "Filter News Import Report",
        [(key, report.get(key, "")) for key in keys],
        extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Desktop Filter news archive into standardized raw News CSV.")
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    report = import_filter_news(args.source)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
