from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from scoring_utils import REGISTRY_HEADERS, SCORING_ROOT, ensure_directories, write_csv_rows


DEFAULT_SOURCE_CSV = Path("Desktop/MDA/final_ready_mda.csv")


def build_document_id(row: dict[str, str]) -> str:
    stock_code = _clean_id_part(row.get("stock_code", ""))
    ticker = _clean_id_part(row.get("ticker", ""))
    report_year = _clean_id_part(row.get("report_year", ""))
    return f"MDA_{stock_code}_{ticker}_{report_year}"


def paragraphize_mda_text(text: str, max_chars: int = 1200) -> str:
    clean = _clean_text(text)
    blocks = _initial_blocks(clean)
    paragraphs: list[str] = []
    for block in blocks:
        paragraphs.extend(_split_large_block(block, max_chars))
    if not paragraphs and clean:
        paragraphs = [clean]
    return "\n\n".join(f"[MDA_P{idx:03d}]\n{paragraph}" for idx, paragraph in enumerate(paragraphs, start=1))


def prepare_mda_inputs(source_csv: Path = DEFAULT_SOURCE_CSV, root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_directories(root)
    source_csv = Path(source_csv)
    rows = _read_source_rows(source_csv)
    registry_rows: list[dict[str, Any]] = []
    included = 0
    skipped = 0
    missing_text = 0
    text_dir = root / "02_extracted_text" / "mda"
    text_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        document_id = build_document_id(row)
        include = _is_include_row(row)
        text = row.get("mda_text", "")
        if not text.strip():
            candidate = _text_file_candidate(source_csv.parent, row)
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8", errors="replace")
        if include and not text.strip():
            include = False
            missing_text += 1
        if include:
            numbered_text = paragraphize_mda_text(text)
            (text_dir / f"{document_id}.txt").write_text(numbered_text, encoding="utf-8")
            included += 1
        else:
            skipped += 1
        registry_rows.append(_registry_row(row, document_id, include))

    write_csv_rows(root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], registry_rows)
    report = {
        "source_csv": str(source_csv),
        "source_rows": len(rows),
        "registry_rows": len(registry_rows),
        "included_rows": included,
        "skipped_rows": skipped,
        "missing_text_rows": missing_text,
        "output_registry": str(root / "01_registry" / "mda_registry.csv"),
        "output_text_dir": str(text_dir),
    }
    _write_report(root, report)
    return report


def _read_source_rows(source_csv: Path) -> list[dict[str, str]]:
    with source_csv.open(newline="", encoding="utf-8-sig") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _registry_row(row: dict[str, str], document_id: str, include: bool) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "doc_type": "mda",
        "include_flag": "Yes" if include else "No",
        "stock_code": row.get("stock_code", ""),
        "ticker": row.get("ticker", ""),
        "company_name": row.get("company_name", ""),
        "report_year": row.get("report_year", ""),
        "extraction_status": "success" if include else row.get("codex_final_status", row.get("extractor_status", "")),
        "text_path": f"02_extracted_text/mda/{document_id}.txt" if include else "",
        "source_path": row.get("final_pdf_path") or row.get("source_files", ""),
        "source_file_sha256": row.get("file_sha256", ""),
        "notes": _notes(row),
    }


def _notes(row: dict[str, str]) -> str:
    items = [
        f"detected_heading={row.get('detected_heading', '')}",
        f"section_type_norm={row.get('section_type_norm', '')}",
        f"confidence={row.get('confidence', '')}",
        f"codex_final_status={row.get('codex_final_status', '')}",
        f"char_count={row.get('char_count', '')}",
    ]
    return "; ".join(item for item in items if not item.endswith("="))


def _is_include_row(row: dict[str, str]) -> bool:
    final_status = row.get("codex_final_status", "").strip().upper()
    audit_status = row.get("audit_status", "").strip().upper()
    extractor_status = row.get("extractor_status", "").strip().upper()
    return final_status == "CODEX_FINAL_PASS" and audit_status in {"AUDIT_PASS", ""} and extractor_status in {"EXTRACTED", ""}


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _initial_blocks(text: str) -> list[str]:
    blank_split = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blank_split) >= 3:
        return blank_split
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        starts_new = bool(current) and (_looks_like_heading(line) or sum(len(item) for item in current) > 800)
        if starts_new:
            blocks.append(" ".join(current).strip())
            current = []
        current.append(line)
    if current:
        blocks.append(" ".join(current).strip())
    return blocks


def _split_large_block(block: str, max_chars: int) -> list[str]:
    if len(block) <= max_chars:
        return [block]
    sentences = re.split(r"(?<=[.!?])\s+", block)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if current and len(current) + 1 + len(sentence) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current.strip())
    return chunks


def _looks_like_heading(line: str) -> bool:
    if len(line) > 80:
        return False
    letters = [char for char in line if char.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    heading_words = {"REVENUE", "OUTLOOK", "PROSPECTS", "DIVIDENDS", "FINANCIAL", "REVIEW", "PERFORMANCE", "RISK"}
    return uppercase_ratio > 0.75 or any(word in line.upper() for word in heading_words)


def _clean_id_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", str(value).strip())
    return cleaned.upper()


def _text_file_candidate(source_dir: Path, row: dict[str, str]) -> Path:
    filename = f"{_clean_id_part(row.get('stock_code', ''))}_{_clean_id_part(row.get('ticker', ''))}_{_clean_id_part(row.get('report_year', ''))}_mda.txt"
    return source_dir / "text_files" / filename


def _write_report(root: Path, report: dict[str, Any]) -> None:
    path = root / "08_reports" / "mda_input_preparation_report.md"
    lines = [
        "# MDA Input Preparation Report",
        "",
        f"- source_csv: {report['source_csv']}",
        f"- source_rows: {report['source_rows']}",
        f"- registry_rows: {report['registry_rows']}",
        f"- included_rows: {report['included_rows']}",
        f"- skipped_rows: {report['skipped_rows']}",
        f"- missing_text_rows: {report['missing_text_rows']}",
        f"- output_registry: {report['output_registry']}",
        f"- output_text_dir: {report['output_text_dir']}",
        "",
        "Only rows with `CODEX_FINAL_PASS` are marked `include_flag=Yes`.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare MD&A scoring inputs from final_ready_mda.csv.")
    parser.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    parser.add_argument("--root", default=str(SCORING_ROOT))
    args = parser.parse_args()
    report = prepare_mda_inputs(Path(args.source_csv), Path(args.root))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
