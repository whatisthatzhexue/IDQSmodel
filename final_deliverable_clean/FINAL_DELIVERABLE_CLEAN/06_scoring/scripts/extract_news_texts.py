from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from build_news_registry import _clean, _discover_source_zip
from scoring_utils import SCORING_ROOT, ensure_directories, read_csv_rows
from stability_common import write_markdown


def extract_news_texts(root: Path = SCORING_ROOT, source_zip: Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    ensure_directories(root)
    registry_rows = [
        row
        for row in read_csv_rows(root / "01_registry" / "news_registry.csv")
        if row.get("include_flag", "").strip().lower() in {"yes", "y", "true", "1"}
    ]
    source = source_zip or _discover_source_zip()
    if not source or not source.exists():
        _write_report(root, {"status": "blocked", "reason": "missing_real_news_source", "extracted_count": 0})
        return {"status": "blocked", "reason": "missing_real_news_source", "extracted_count": 0}

    extracted = 0
    missing = 0
    with zipfile.ZipFile(source) as zf:
        for row in registry_rows:
            if "synthetic_flag=true" in row.get("notes", "").lower():
                continue
            out_path = root / row.get("text_path", "")
            if out_path.exists() and not overwrite:
                extracted += 1
                continue
            source_row = _source_row_from_notes(zf, row.get("notes", ""))
            if not source_row:
                missing += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(_format_news_text(row, source_row), encoding="utf-8")
            extracted += 1
    summary = {"status": "success" if extracted else "blocked", "source_zip": str(source), "extracted_count": extracted, "missing_source_rows": missing}
    _write_report(root, summary)
    return summary


def _source_row_from_notes(zf: zipfile.ZipFile, notes: str) -> dict[str, str] | None:
    parsed = _parse_notes(notes)
    member = parsed.get("zip_member", "")
    content_id = parsed.get("content_id", "")
    source_row_index = parsed.get("source_row_index", "")
    if not member or member not in zf.namelist():
        return None
    with zf.open(member) as raw:
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.DictReader(text)
        for idx, row in enumerate(reader, start=1):
            if source_row_index and str(idx) == source_row_index:
                return row
            if content_id and str(row.get("content_id", "")) == content_id:
                return row
    return None


def _parse_notes(notes: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for part in str(notes or "").split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _format_news_text(registry_row: dict[str, str], source_row: dict[str, Any]) -> str:
    title = registry_row.get("title") or _clean(source_row.get("title", ""))
    summary = _clean(source_row.get("summary", ""))
    body = str(source_row.get("text", "") or "")
    paragraphs = _paragraphs(body)
    lines = ["[TITLE]", title, ""]
    if summary:
        lines.extend(["[LEAD]", summary, ""])
    for idx, paragraph in enumerate(paragraphs[:20], start=1):
        lines.extend([f"[NEWS_P{idx:03d}]", paragraph, ""])
    return "\n".join(lines).strip() + "\n"


def _paragraphs(text: str) -> list[str]:
    chunks = re.split(r"\n\s*\n+", text.replace("\r", "\n"))
    cleaned = [_clean(chunk) for chunk in chunks if _clean(chunk)]
    if cleaned:
        return cleaned
    fallback = _clean(text)
    return [fallback] if fallback else []


def _write_report(root: Path, summary: dict[str, Any]) -> None:
    write_markdown(
        root / "08_reports" / "news_text_extraction_report.md",
        ["# News Text Extraction Report", "", *[f"- {key}: {value}" for key, value in summary.items()]],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract real News texts from local filtered-news CSV zip.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extract_news_texts(args.root, args.source_zip, args.overwrite), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
