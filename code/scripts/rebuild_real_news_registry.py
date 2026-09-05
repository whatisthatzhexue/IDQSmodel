from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_real_news_registry import _clean, _parse_date
from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows
from stability_common import write_markdown


ALIAS_HEADERS = ["ticker", "stock_code", "company_name", "alias", "alias_type", "match_strength"]
AUDIT_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "title",
    "source_name",
    "publish_date",
    "url",
    "text_length",
    "alias_hits",
    "strong_alias_hits",
    "weak_alias_hits",
    "relevance_status",
    "relevance_reason",
    "include_flag",
    "exclude_reason",
]
REBUILT_HEADERS = [
    "document_id",
    "doc_type",
    "ticker",
    "company_name",
    "publish_date",
    "source_name",
    "title",
    "url",
    "file_path",
    "article_text_hash",
    "text_length",
    "synthetic_flag",
    "include_flag",
    "exclude_reason",
    "created_at",
]
BUSINESS_TERMS = [
    "revenue",
    "profit",
    "sales",
    "share",
    "stock",
    "business",
    "product",
    "factory",
    "production",
    "market",
    "cost",
    "demand",
    "financial",
    "earnings",
    "management",
    "regulation",
    "risk",
    "brand",
    "food",
    "beverage",
    "milk",
    "brewery",
]


def rebuild_real_news_registry(root: Path = SCORING_ROOT) -> dict[str, Any]:
    aliases = _build_alias_map(root)
    write_csv_rows(root / "01_registry" / "company_alias_map.csv", ALIAS_HEADERS, aliases)
    candidates = _discover_candidate_rows(root)
    audit_rows = []
    registry_rows = []
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    seen_hashes: set[str] = set()
    seen_titles: set[tuple[str, str, str, str]] = set()

    for index, item in enumerate(candidates, start=1):
        audit = _audit_candidate(item, aliases)
        article_hash = hashlib.sha256(item["article_text"].encode("utf-8")).hexdigest()
        title_key = (audit["ticker"], audit["source_name"].lower(), audit["publish_date"], audit["title"].lower())
        if article_hash in seen_hashes or title_key in seen_titles:
            audit["relevance_status"] = "duplicate"
            audit["include_flag"] = "No"
            audit["exclude_reason"] = "duplicate"
        if audit["include_flag"] == "Yes":
            seen_hashes.add(article_hash)
            seen_titles.add(title_key)
        audit_rows.append(audit)
        if audit["include_flag"] == "Yes":
            registry_rows.append(
                {
                    "document_id": audit["document_id"],
                    "doc_type": "news",
                    "ticker": audit["ticker"],
                    "company_name": audit["company_name"],
                    "publish_date": audit["publish_date"],
                    "source_name": audit["source_name"],
                    "title": audit["title"],
                    "url": audit["url"],
                    "file_path": item["file_path"],
                    "article_text_hash": article_hash,
                    "text_length": audit["text_length"],
                    "synthetic_flag": "false",
                    "include_flag": "Yes",
                    "exclude_reason": "",
                    "created_at": created_at,
                }
            )

    write_csv_rows(root / "08_reports" / "news_relevance_audit.csv", AUDIT_HEADERS, audit_rows)
    write_csv_rows(root / "01_registry" / "news_registry_rebuilt.csv", REBUILT_HEADERS, registry_rows)
    summary = {
        "news_registry_total": len(audit_rows),
        "news_real_company_news_count": sum(1 for row in audit_rows if row["relevance_status"] == "real_company_news"),
        "news_included_count": len(registry_rows),
        "news_excluded_count": len(audit_rows) - len(registry_rows),
        "synthetic_count": 0,
    }
    _write_report(root, summary)
    if not registry_rows:
        _write_blocker(root)
    return summary


def _build_alias_map(root: Path) -> list[dict[str, str]]:
    aliases: list[dict[str, str]] = []
    for row in read_csv_rows(root / "01_registry" / "mda_registry.csv"):
        ticker = _clean(row.get("ticker", "")).upper()
        stock_code = _clean(row.get("stock_code", ""))
        company_name = _clean(row.get("company_name", ""))
        if not ticker and not company_name:
            continue
        for alias, alias_type, strength in _aliases_for_company(ticker, stock_code, company_name):
            aliases.append(
                {
                    "ticker": ticker,
                    "stock_code": stock_code,
                    "company_name": company_name,
                    "alias": alias,
                    "alias_type": alias_type,
                    "match_strength": strength,
                }
            )
    deduped = []
    seen = set()
    for row in aliases:
        key = (row["ticker"], row["alias"].lower())
        if key in seen or not row["alias"]:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _aliases_for_company(ticker: str, stock_code: str, company_name: str) -> list[tuple[str, str, str]]:
    aliases = []
    if len(company_name) >= 4:
        aliases.append((company_name, "legal_name", "strong"))
    stripped = re.sub(r"\b(berhad|bhd|holdings?|industries?|resources?)\b", "", company_name, flags=re.I)
    short = " ".join(stripped.split())
    if short and short.lower() != company_name.lower():
        aliases.append((short, "short_name", "medium"))
    if ticker:
        strength = "weak" if len(ticker) <= 3 or ticker in {"CI", "3A"} else "medium"
        aliases.append((ticker, "ticker", strength))
    if stock_code:
        aliases.append((stock_code, "stock_code", "weak"))
    known = {
        "DLADY": ["Dutch Lady", "Dutch Lady Milk"],
        "CARLSBG": ["Carlsberg Malaysia", "Carlsberg Brewery Malaysia"],
        "HEIM": ["Heineken Malaysia"],
        "3A": ["Three-A Resources", "Three-A"],
    }
    for alias in known.get(ticker, []):
        aliases.append((alias, "brand_name", "strong"))
    return aliases


def _discover_candidate_rows(root: Path) -> list[dict[str, str]]:
    candidates = []
    seen_paths: set[Path] = set()
    search_roots = [root / "input_news", root.parent / "01_raw_news", root.parent / "news", root.parent / "raw_news", root.parent / "data" / "news", root.parent / "04_source_links", root]
    for base in search_roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl", ".txt", ".html", ".htm"}:
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if _excluded_path(path):
                continue
            candidates.extend(_rows_from_file(root, path))
    return candidates


def _excluded_path(path: Path) -> bool:
    lowered = str(path).lower()
    excluded_terms = [
        "synthetic",
        "mock",
        "fixture",
        "benchmark",
        "__pycache__",
        ".pytest_cache",
        "final_deliverable",
        "final_output",
        "paper_output",
        "06_ratings",
        "08_reports",
        "09_cross_validation",
        "10_claim",
        "11_stability_analysis",
        "02_extracted_text/mda",
        "02_extracted_text/mda_clean",
        "/scripts/",
        "/tests/",
        "/docs/",
        "human_validation",
    ]
    return any(term in lowered for term in excluded_terms)


def _rows_from_file(root: Path, path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        try:
            with path.open(newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                fields = set(reader.fieldnames or [])
                if not {"title", "article_text"}.issubset(fields):
                    return []
                return [_candidate(root, path, dict(row)) for row in reader]
        except UnicodeDecodeError:
            return []
    if path.suffix.lower() in {".json", ".jsonl"}:
        rows = []
        for item in _json_items(path):
            if isinstance(item, dict) and (item.get("article_text") or item.get("text")):
                rows.append(_candidate(root, path, item))
        return rows
    if path.suffix.lower() in {".txt", ".html", ".htm"}:
        if not _news_context_path(path):
            return []
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(re.sub(r"\s+", "", text)) < 200:
            return []
        return [
            _candidate(
                root,
                path,
                {
                    "ticker": "",
                    "company_name": "",
                    "publish_date": "",
                    "source_name": path.parent.name,
                    "title": path.stem,
                    "url": "",
                    "article_text": text,
                    "_metadata_only_source": "true",
                },
            )
        ]
    return []


def _news_context_path(path: Path) -> bool:
    lowered = str(path).lower()
    return any(term in lowered for term in ["input_news", "raw_news", "/news/", "news_raw", "new_news"])


def _json_items(path: Path) -> list[Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    return loaded if isinstance(loaded, list) else [loaded]


def _candidate(root: Path, path: Path, row: dict[str, Any]) -> dict[str, str]:
    try:
        file_path = str(path.relative_to(root))
    except ValueError:
        file_path = str(path)
    return {
        "ticker": _clean(row.get("ticker", "")),
        "company_name": _clean(row.get("company_name", "")),
        "publish_date": _parse_date(_clean(row.get("publish_date", ""))) or _clean(row.get("publish_date", "")),
        "source_name": _clean(row.get("source_name", "")),
        "title": _clean(row.get("title", "")),
        "url": _clean(row.get("url", "")),
        "article_text": str(row.get("article_text") or row.get("text") or ""),
        "file_path": file_path,
        "_metadata_only_source": str(row.get("_metadata_only_source", "")),
    }


def _audit_candidate(item: dict[str, str], aliases: list[dict[str, str]]) -> dict[str, str]:
    title = item["title"]
    text = item["article_text"]
    combined = f"{title}\n{text}"
    compact_len = len(re.sub(r"\s+", "", text))
    hits = _alias_hits(combined, aliases)
    strong_hits = [hit for hit in hits if hit["match_strength"] == "strong"]
    weak_hits = [hit for hit in hits if hit["match_strength"] == "weak"]
    best = strong_hits[0] if strong_hits else (hits[0] if hits else {})
    publish_date = _parse_date(item["publish_date"]) or item["publish_date"]
    doc_id = _doc_id(best.get("ticker") or item["ticker"] or "NEWS", publish_date, title, text)
    status = "real_company_news"
    reason = "strong alias and business topic"
    include = "Yes"
    exclude_reason = ""
    if item.get("_metadata_only_source") == "true":
        status, reason, include, exclude_reason = "metadata_only", "pre-extracted text lacks trusted title/source/date metadata", "No", "metadata_only"
    elif compact_len < 200:
        status, reason, include, exclude_reason = "too_short", "article text shorter than 200 non-space characters", "No", "too_short"
    elif not title or not item["source_name"]:
        status, reason, include, exclude_reason = "metadata_only", "missing title or source_name", "No", "metadata_only"
    elif _url_only(text):
        status, reason, include, exclude_reason = "url_only", "article text is only URL-like", "No", "url_only"
    elif not strong_hits:
        status, reason, include, exclude_reason = "irrelevant_general_news", "no strong company alias hit; ticker-only/weak hits are insufficient", "No", "irrelevant_general_news"
    elif not _business_topic(combined):
        status, reason, include, exclude_reason = "irrelevant_general_news", "company alias hit but topic is not company/business relevant", "No", "irrelevant_general_news"
    return {
        "document_id": doc_id,
        "ticker": best.get("ticker") or item["ticker"],
        "company_name": best.get("company_name") or item["company_name"],
        "title": title,
        "source_name": item["source_name"],
        "publish_date": publish_date,
        "url": item["url"],
        "text_length": str(compact_len),
        "alias_hits": ";".join(hit["alias"] for hit in hits),
        "strong_alias_hits": ";".join(hit["alias"] for hit in strong_hits),
        "weak_alias_hits": ";".join(hit["alias"] for hit in weak_hits),
        "relevance_status": status,
        "relevance_reason": reason,
        "include_flag": include,
        "exclude_reason": exclude_reason,
    }


def _alias_hits(text: str, aliases: list[dict[str, str]]) -> list[dict[str, str]]:
    lowered = text.lower()
    hits = []
    for row in aliases:
        alias = row["alias"].strip()
        if not alias:
            continue
        if row["match_strength"] == "weak":
            pattern = r"\b" + re.escape(alias.lower()) + r"\b"
            matched = bool(re.search(pattern, lowered))
        else:
            matched = alias.lower() in lowered
        if matched:
            hits.append(row)
    hits.sort(key=lambda row: {"strong": 0, "medium": 1, "weak": 2}.get(row["match_strength"], 3))
    return hits


def _url_only(text: str) -> bool:
    lowered = text.strip().lower()
    return bool(re.fullmatch(r"https?://\S+", lowered))


def _business_topic(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in BUSINESS_TERMS)


def _doc_id(ticker: str, publish_date: str, title: str, text: str) -> str:
    date = re.sub(r"[^0-9]", "", publish_date or "")[:8] or "UNKNOWN"
    digest = hashlib.sha1(f"{ticker}|{publish_date}|{title}|{text[:200]}".encode("utf-8")).hexdigest()[:8].upper()
    safe_ticker = re.sub(r"[^A-Z0-9]", "", ticker.upper()) or "NEWS"
    return f"NEWS_{safe_ticker}_{date}_{digest}"


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    write_markdown(root / "08_reports" / "news_registry_rebuild_report.md", ["# News Registry Rebuild Report", "", *[f"- {k}: {v}" for k, v in summary.items()]])


def _write_blocker(root: Path) -> None:
    write_markdown(
        root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md",
        [
            "# Real News Missing Blocker",
            "",
            "MD&A final is complete and frozen.",
            "Combined MD&A + News paper-ready result is blocked because real company-specific full-text News data are missing.",
            "",
            "- synthetic_news_allowed: false",
            "- empirical_cross_validation_allowed: false",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild real News registry with alias-based company relevance audit.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(rebuild_real_news_registry(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
