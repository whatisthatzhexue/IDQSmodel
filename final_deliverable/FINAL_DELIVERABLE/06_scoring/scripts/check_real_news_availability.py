from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, write_csv_rows
from stability_common import write_markdown


REQUIRED_NEWS_FIELDS = ["ticker", "company_name", "publish_date", "source_name", "title", "url", "article_text"]

FOUND_HEADERS = [
    "file_path",
    "scan_directory",
    "file_name",
    "extension",
    "size_bytes",
    "row_count",
    "required_fields_present",
    "missing_required_fields",
]

CANONICAL_FILE_NAMES = {"news_raw.xlsx", "news_raw.csv"}
EXTENSIONS = {".xlsx", ".csv", ".txt", ".docx", ".html", ".pdf"}


def check_real_news_availability(root: Path = SCORING_ROOT) -> dict[str, Any]:
    status_dir = root / "PAPER_OUTPUT" / "00_status"
    scan_dirs = _scan_dirs(root)
    rows: list[dict[str, Any]] = []
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for path in _iter_news_files(scan_dir):
            rows.append(_file_row(path, scan_dir))
    rows = _dedupe_rows(rows)
    write_csv_rows(status_dir / "real_news_files_found.csv", FOUND_HEADERS, rows)
    summary = {
        "real_news_available": bool(rows),
        "real_news_file_count": len(rows),
        "scan_directories": [str(path) for path in scan_dirs],
        "files_found_csv": str(status_dir / "real_news_files_found.csv"),
        "required_fields": REQUIRED_NEWS_FIELDS,
        "csv_or_xlsx_with_required_fields_count": sum(row["required_fields_present"] == "true" for row in rows),
    }
    _write_report(status_dir / "real_news_availability_report.md", summary, rows)
    blocker_path = status_dir / "real_news_missing_blocker.md"
    if not rows:
        _write_missing_blocker(blocker_path)
    elif blocker_path.exists():
        blocker_path.unlink()
    return summary


def _scan_dirs(root: Path) -> list[Path]:
    project_root = root.parent
    return [
        root / "input_news",
        project_root / "01_raw_news",
        project_root / "news",
        project_root / "raw_news",
        project_root / "data" / "news",
    ]


def _iter_news_files(scan_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in scan_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in CANONICAL_FILE_NAMES or suffix in EXTENSIONS:
            files.append(path)
    return sorted(files)


def _file_row(path: Path, scan_dir: Path) -> dict[str, Any]:
    row_count = ""
    missing = REQUIRED_NEWS_FIELDS
    if path.suffix.lower() == ".csv":
        row_count, missing = _inspect_csv(path)
    required_present = not missing if path.suffix.lower() in {".csv", ".xlsx"} else False
    return {
        "file_path": str(path),
        "scan_directory": str(scan_dir),
        "file_name": path.name,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "row_count": row_count,
        "required_fields_present": str(required_present).lower(),
        "missing_required_fields": ";".join(missing),
    }


def _inspect_csv(path: Path) -> tuple[str, list[str]]:
    try:
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            reader = csv.DictReader(fh)
            fields = {str(field or "").strip() for field in (reader.fieldnames or [])}
            count = sum(1 for _ in reader)
    except csv.Error:
        return "", REQUIRED_NEWS_FIELDS
    missing = [field for field in REQUIRED_NEWS_FIELDS if field not in fields]
    return str(count), missing


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for row in rows:
        key = row["file_path"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Real News Availability Report",
        "",
        f"- real_news_available: {str(summary['real_news_available']).lower()}",
        f"- real_news_file_count: {summary['real_news_file_count']}",
        f"- csv_or_xlsx_with_required_fields_count: {summary['csv_or_xlsx_with_required_fields_count']}",
        "",
        "## Scan Directories",
    ]
    lines.extend(f"- {item}" for item in summary["scan_directories"])
    lines.extend(["", "## Files Found"])
    if rows:
        for row in rows[:50]:
            lines.append(f"- {row['file_path']} | required_fields_present={row['required_fields_present']} | missing={row['missing_required_fields'] or 'none'}")
    else:
        lines.append("- none")
    write_markdown(path, lines)


def _write_missing_blocker(path: Path) -> None:
    write_markdown(
        path,
        [
            "# Real News Missing Blocker",
            "",
            "- news_ready remains false.",
            "- cross-validation cannot be interpreted empirically.",
            "- synthetic news is not used.",
            "- User must provide real news_raw.xlsx or news_raw.csv.",
            "- Required fields: ticker, company_name, publish_date, source_name, title, url, article_text.",
            "- Freeze remains blocked until real News final scores exist and pass schema validation.",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether real News source files are available for final freeze.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(check_real_news_availability(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
