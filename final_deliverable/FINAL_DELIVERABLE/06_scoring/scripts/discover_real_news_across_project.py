from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, write_csv_rows
from stability_common import write_markdown


SCAN_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".txt", ".md", ".html", ".htm", ".docx", ".pdf", ".zip"}
EXCLUDED_PARTS = {
    ".venv",
    "__pycache__",
    "tests",
    "_self_correction",
    "benchmark",
    "mock",
    "synthetic",
    "FINAL_OUTPUT",
    "PAPER_OUTPUT",
    "FINAL_DELIVERABLE",
    ".pytest_cache",
}

FIELD_SYNONYMS = {
    "ticker": {"ticker", "stock_code", "stock", "code"},
    "company_name": {"company", "company_name", "issuer", "firm"},
    "publish_date": {"publish_date", "published_date", "date", "publication_date"},
    "source_name": {"source", "source_name", "publisher", "section"},
    "title": {"title", "headline", "heading"},
    "url": {"url", "link", "source_url"},
    "article_text": {"article_text", "body", "content", "news_text", "full_text", "text"},
}

INVENTORY_HEADERS = [
    "file_path",
    "file_type",
    "category",
    "reason",
    "row_count",
    "matched_fields",
    "missing_fields",
    "sha256",
]

STANDARD_HEADERS = [
    "ticker",
    "company_name",
    "publish_date",
    "source_name",
    "title",
    "url",
    "article_text",
    "synthetic_flag",
    "original_file_path",
    "original_file_hash",
]


def discover_real_news_across_project(project_root: Path | None = None, root: Path = SCORING_ROOT) -> dict[str, Any]:
    project_root = project_root or root.parent
    inventory: list[dict[str, Any]] = []
    standardized: list[dict[str, Any]] = []
    for path in sorted(_candidate_files(project_root)):
        file_hash = _sha256(path)
        try:
            item, rows = _inspect_file(path, file_hash)
        except Exception as exc:  # noqa: BLE001 - inventory should retain unreadable cases.
            item, rows = _inventory_row(path, "unreadable", str(exc), "", {}, file_hash), []
        inventory.append(item)
        if item["category"] == "real_news_candidate":
            standardized.extend(rows)
    inventory_path = root / "PAPER_OUTPUT" / "00_status" / "real_news_discovery_inventory.csv"
    write_csv_rows(inventory_path, INVENTORY_HEADERS, inventory)
    out_dir = root / "input_news" / "real_news_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    standardized = _dedupe_standardized(standardized)
    if standardized:
        write_csv_rows(out_dir / "real_news_raw.csv", STANDARD_HEADERS, standardized)
    summary = {
        "project_root": str(project_root),
        "inventory_count": len(inventory),
        "real_news_candidate_count": sum(row["category"] == "real_news_candidate" for row in inventory),
        "standardized_news_rows": len(standardized),
        "source_link_only_count": sum(row["category"] == "source_link_only" for row in inventory),
        "synthetic_or_test_count": sum(row["category"] == "synthetic_or_test" for row in inventory),
        "inventory_path": str(inventory_path),
        "standardized_path": str(out_dir / "real_news_raw.csv") if standardized else "",
        "real_news_available": bool(standardized),
    }
    _write_report(root / "PAPER_OUTPUT" / "00_status" / "real_news_discovery_report.md", summary, inventory)
    return summary


def _candidate_files(project_root: Path) -> list[Path]:
    out = []
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        name_lower = path.name.lower()
        if "synthetic" in name_lower or "mock" in name_lower:
            continue
        out.append(path)
    return out


def _inspect_file(path: Path, file_hash: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _inspect_csv(path, file_hash)
    if suffix == ".zip":
        return _inspect_zip(path, file_hash)
    if suffix in {".txt", ".md", ".html", ".htm", ".json", ".jsonl"}:
        text = path.read_text(encoding="utf-8", errors="replace")[:5000].lower()
        if "synthetic_flag=true" in text or "placeholder" in text:
            return _inventory_row(path, "synthetic_or_test", "synthetic or placeholder marker", "", {}, file_hash), []
        if "http" in text and any(word in text for word in ["news", "article", "headline", "publisher"]):
            return _inventory_row(path, "source_link_only", "text file appears to contain news links but no structured article rows", "", {}, file_hash), []
        return _inventory_row(path, "not_news", "no structured news table detected", "", {}, file_hash), []
    return _inventory_row(path, "needs_manual_confirmation", "binary or spreadsheet file requires manual/openpyxl confirmation", "", {}, file_hash), []


def _inspect_csv(path: Path, file_hash: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        field_map = _field_map(fieldnames)
        missing = [key for key in FIELD_SYNONYMS if key not in field_map]
        rows = []
        for source_row in reader:
            if len(rows) >= 5000:
                break
            if missing:
                continue
            article = _clean(source_row.get(field_map["article_text"], ""))
            title = _clean(source_row.get(field_map["title"], ""))
            if not article or not title:
                continue
            rows.append(
                {
                    "ticker": _clean(source_row.get(field_map["ticker"], "")),
                    "company_name": _clean(source_row.get(field_map["company_name"], "")),
                    "publish_date": _clean(source_row.get(field_map["publish_date"], "")),
                    "source_name": _clean(source_row.get(field_map["source_name"], "")),
                    "title": title,
                    "url": _clean(source_row.get(field_map["url"], "")),
                    "article_text": article,
                    "synthetic_flag": "false",
                    "original_file_path": str(path),
                    "original_file_hash": file_hash,
                }
            )
    category = "real_news_candidate" if rows and not missing else ("source_link_only" if "url" in field_map and "article_text" not in field_map else "not_news")
    reason = "structured fields and article text found" if rows else ("missing required fields: " + ";".join(missing))
    return _inventory_row(path, category, reason, str(len(rows)), field_map, file_hash), rows


def _inspect_zip(path: Path, file_hash: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    reasons = []
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        for name in names[:200]:
            lower = name.lower()
            if "synthetic" in lower or "mock" in lower:
                continue
            if not lower.endswith(".csv"):
                if any(word in lower for word in ["news", "article"]):
                    reasons.append(f"candidate member: {name}")
                continue
            with zf.open(name) as raw:
                text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                reader = csv.DictReader(text)
                field_map = _field_map(reader.fieldnames or [])
                missing = [key for key in FIELD_SYNONYMS if key not in field_map]
                if missing:
                    reasons.append(f"{name} missing {','.join(missing)}")
                    continue
                for source_row in reader:
                    article = _clean(source_row.get(field_map["article_text"], ""))
                    title = _clean(source_row.get(field_map["title"], ""))
                    if not article or not title:
                        continue
                    rows.append(
                        {
                            "ticker": _clean(source_row.get(field_map["ticker"], "")),
                            "company_name": _clean(source_row.get(field_map["company_name"], "")),
                            "publish_date": _clean(source_row.get(field_map["publish_date"], "")),
                            "source_name": _clean(source_row.get(field_map["source_name"], "")),
                            "title": title,
                            "url": _clean(source_row.get(field_map["url"], "")),
                            "article_text": article,
                            "synthetic_flag": "false",
                            "original_file_path": f"{path}::{name}",
                            "original_file_hash": file_hash,
                        }
                    )
                    if len(rows) >= 5000:
                        break
    category = "real_news_candidate" if rows else ("source_link_only" if reasons else "not_news")
    reason = "zip contains structured news rows" if rows else ("; ".join(reasons[:5]) or "no news-like members")
    return _inventory_row(path, category, reason, str(len(rows)), {}, file_hash), rows


def _field_map(fieldnames: list[str]) -> dict[str, str]:
    normalized = {_norm(field): field for field in fieldnames}
    out: dict[str, str] = {}
    for canonical, synonyms in FIELD_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in normalized:
                out[canonical] = normalized[synonym]
                break
    return out


def _inventory_row(path: Path, category: str, reason: str, row_count: str, field_map: dict[str, str], file_hash: str) -> dict[str, Any]:
    missing = [key for key in FIELD_SYNONYMS if key not in field_map] if field_map else []
    return {
        "file_path": str(path),
        "file_type": path.suffix.lower(),
        "category": category,
        "reason": reason,
        "row_count": row_count,
        "matched_fields": ";".join(f"{key}={value}" for key, value in sorted(field_map.items())),
        "missing_fields": ";".join(missing),
        "sha256": file_hash,
    }


def _dedupe_standardized(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        key = (row["ticker"], row["company_name"], row["publish_date"], row["title"], row["article_text"][:120])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _write_report(path: Path, summary: dict[str, Any], inventory: list[dict[str, Any]]) -> None:
    lines = [
        "# Real News Discovery Report",
        "",
        f"- project_root: {summary['project_root']}",
        f"- inventory_count: {summary['inventory_count']}",
        f"- real_news_candidate_count: {summary['real_news_candidate_count']}",
        f"- standardized_news_rows: {summary['standardized_news_rows']}",
        f"- real_news_available: {str(summary['real_news_available']).lower()}",
        "",
        "## Candidate Files",
    ]
    candidates = [row for row in inventory if row["category"] == "real_news_candidate"]
    if candidates:
        lines.extend(f"- {row['file_path']} ({row['row_count']} rows)" for row in candidates[:50])
    else:
        lines.append("- none")
    write_markdown(path, lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _norm(value: str) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", "\n").split())


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover real News source files across the whole project.")
    parser.add_argument("--project-root", type=Path, default=SCORING_ROOT.parent)
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(discover_real_news_across_project(args.project_root, args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
