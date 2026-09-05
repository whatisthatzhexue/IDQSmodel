from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any

from scoring_utils import REGISTRY_HEADERS, SCORING_ROOT, ensure_directories, read_csv_rows, write_csv_rows
from stability_common import write_markdown


DEFAULT_NEWS_ZIP_CANDIDATES = [
    Path(os.environ.get("NEWS_SOURCE_ZIP", "")) if os.environ.get("NEWS_SOURCE_ZIP") else None,
    Path("Desktop/Filter news(1).zip"),
    Path("Desktop/Filter news.zip"),
]

TICKER_ALIASES = {
    "F&N": "FN",
    "F N": "FN",
    "THREE-A RESOURCES BERHAD": "3A",
    "THREE A RESOURCES BERHAD": "3A",
    "MAG HOLDINGS BERHAD": "MAG",
}


def build_news_registry(
    root: Path = SCORING_ROOT,
    source_zip: Path | None = None,
    max_rows_per_ticker: int = 3,
    overwrite: bool = False,
) -> dict[str, Any]:
    ensure_directories(root)
    source = source_zip or _discover_source_zip()
    output_path = root / "01_registry" / "news_registry.csv"
    if not source or not source.exists():
        _write_blocking_report(root, "news missing", "No real news source zip was found.")
        return {"status": "blocked", "reason": "missing_real_news_source", "source_zip": str(source or ""), "real_news_count": 0}

    existing = read_csv_rows(output_path)
    if existing and not overwrite and any("synthetic_flag=false" in row.get("notes", "").lower() for row in existing):
        real_count = sum("synthetic_flag=false" in row.get("notes", "").lower() for row in existing)
        return {"status": "reused", "source_zip": str(source), "registry_path": str(output_path), "real_news_count": real_count}

    mda_rows = read_csv_rows(root / "01_registry" / "mda_registry.csv")
    ticker_order = _ticker_order(mda_rows)
    target_tickers = set(ticker_order)
    rows: list[dict[str, str]] = []
    per_ticker: dict[str, int] = {}
    with zipfile.ZipFile(source) as zf:
        members = sorted([name for name in zf.namelist() if name.lower().endswith(".csv") and "/new_news/" not in name.lower()], key=lambda name: _member_sort_key(name, ticker_order))
        if not members:
            members = sorted([name for name in zf.namelist() if name.lower().endswith(".csv")], key=lambda name: _member_sort_key(name, ticker_order))
        for member in members:
            ticker = _ticker_from_member(member)
            if target_tickers and ticker not in target_tickers:
                continue
            for source_row_index, source_row in _iter_csv_rows(zf, member):
                if per_ticker.get(ticker, 0) >= max_rows_per_ticker:
                    break
                title = _clean(source_row.get("title", ""))
                text = _clean(source_row.get("text", ""))
                publish_date = _date(source_row.get("published_date", ""))
                if not title or not text or not publish_date:
                    continue
                doc_id = _document_id(ticker, publish_date, source_row.get("content_id") or str(source_row_index), title)
                rows.append(
                    {
                        "document_id": doc_id,
                        "doc_type": "news",
                        "include_flag": "Yes",
                        "company_name": _company_for_ticker(mda_rows, ticker) or ticker,
                        "ticker": ticker,
                        "source_name": _clean(source_row.get("section") or source_row.get("category") or "real_news_csv"),
                        "publish_date": publish_date,
                        "title": title,
                        "url": _clean(source_row.get("url", "")),
                        "text_path": f"02_extracted_text/news/{doc_id}.txt",
                        "notes": f"synthetic_flag=false; source_zip={source}; zip_member={member}; content_id={_clean(source_row.get('content_id', ''))}; source_row_index={source_row_index}",
                    }
                )
                per_ticker[ticker] = per_ticker.get(ticker, 0) + 1
    rows = _dedupe_rows(rows)
    write_csv_rows(output_path, REGISTRY_HEADERS["news"], rows)
    summary = {
        "status": "success" if rows else "blocked",
        "source_zip": str(source),
        "registry_path": str(output_path),
        "real_news_count": len(rows),
        "ticker_count": len({row["ticker"] for row in rows}),
        "max_rows_per_ticker": max_rows_per_ticker,
    }
    if not rows:
        _write_blocking_report(root, "news missing", "Real news zip was found, but no usable rows matched MD&A tickers.")
    return summary


def _discover_source_zip() -> Path | None:
    for candidate in DEFAULT_NEWS_ZIP_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    return None


def _ticker_order(mda_rows: list[dict[str, str]]) -> list[str]:
    order: list[str] = []
    for row in mda_rows:
        ticker = _normalize_ticker(row.get("ticker", ""))
        if ticker and ticker not in order:
            order.append(ticker)
    return order


def _member_sort_key(member: str, order: list[str]) -> tuple[int, str]:
    ticker = _ticker_from_member(member)
    return (order.index(ticker) if ticker in order else 9999, member)


def _ticker_from_member(member: str) -> str:
    parts = [part for part in Path(member).parts if part not in {"/", ""}]
    folder = parts[-2] if len(parts) >= 2 else Path(member).stem.replace("_news", "")
    return _normalize_ticker(folder)


def _normalize_ticker(value: str) -> str:
    cleaned = " ".join(str(value or "").replace("_", " ").strip().upper().split())
    if cleaned in TICKER_ALIASES:
        return TICKER_ALIASES[cleaned]
    cleaned = cleaned.replace("&", "")
    return re.sub(r"[^A-Z0-9]", "", cleaned)


def _iter_csv_rows(zf: zipfile.ZipFile, member: str):
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.DictReader(text)
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def _document_id(ticker: str, publish_date: str, content_id: str, title: str) -> str:
    digest = hashlib.sha1(f"{ticker}|{publish_date}|{content_id}|{title}".encode("utf-8")).hexdigest()[:8].upper()
    return f"NEWS_{ticker}_{publish_date.replace('-', '')}_{digest}"


def _company_for_ticker(mda_rows: list[dict[str, str]], ticker: str) -> str:
    for row in mda_rows:
        if _normalize_ticker(row.get("ticker", "")) == ticker:
            return row.get("company_name", "")
    return ""


def _date(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    return match.group(0) if match else text[:10]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def _dedupe_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped = []
    seen = set()
    for row in rows:
        key = row["document_id"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _write_blocking_report(root: Path, title: str, reason: str) -> None:
    write_markdown(
        root / "08_reports" / "news_data_blocking_report.md",
        [
            "# News Data Blocking Report",
            "",
            f"- status: {title}",
            f"- reason: {reason}",
            "- cross-validation invalid: true",
            "- freeze cannot proceed: true",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build real News registry from local filtered-news CSV zip.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--max-rows-per-ticker", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_news_registry(args.root, args.source_zip, args.max_rows_per_ticker, args.overwrite), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
