from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, write_csv_rows


FINANCIAL_TERMS = re.compile(r"\b(revenue|profit|cost|margin|risk|outlook|segment|market|sales|cash|rm|million|billion|sen|%)\b", re.I)
SECTION_TERMS = {
    "financial": "Financial Review",
    "performance": "Performance Review",
    "business": "Business Review",
    "operation": "Business Review",
    "risk": "Risk and Outlook",
    "outlook": "Risk and Outlook",
    "prospect": "Risk and Outlook",
    "market": "Market Review",
    "dividend": "Dividends",
}


def paragraph_ids(text: str) -> list[str]:
    return re.findall(r"\[(MDA_P\d{3})\]", text)


def clean_one_mda_text(document_id: str, raw_text: str) -> tuple[str, list[dict[str, str]]]:
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    raw_text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw_text)
    raw_lines = raw_text.splitlines()
    cleaned_lines: list[str] = []
    mapping: list[dict[str, str]] = []
    removed_idx = 0
    for idx, line in enumerate(raw_lines, start=1):
        original = line.rstrip()
        stripped = re.sub(r"\s+", " ", original).strip()
        if not stripped:
            cleaned_lines.append("")
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            removed_idx += 1
            mapping.append(_mapping_row(document_id, f"REMOVED_{removed_idx:03d}", "Unknown", f"L{idx}", stripped, "", "remove_page_number", "1", "isolated_page_number", "high"))
            continue
        if _looks_like_header_footer(stripped):
            removed_idx += 1
            mapping.append(_mapping_row(document_id, f"REMOVED_{removed_idx:03d}", "Unknown", f"L{idx}", stripped, "", "remove_header_footer", "1", "header_footer", "medium"))
            continue
        cleaned_lines.append(stripped)
    normalized = "\n".join(cleaned_lines)
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    blocks = _blocks(normalized)
    seen: set[str] = set()
    output_parts: list[str] = []
    para_idx = 0
    current_section = "Unknown"
    for raw_locator, block in blocks:
        clean = block.strip()
        if not clean:
            continue
        maybe_section = _section_for(clean)
        if maybe_section and len(clean) <= 120:
            current_section = maybe_section
            continue
        dedupe_key = re.sub(r"\W+", "", clean.lower())
        if dedupe_key in seen and not FINANCIAL_TERMS.search(clean):
            removed_idx += 1
            mapping.append(_mapping_row(document_id, f"REMOVED_{removed_idx:03d}", current_section, raw_locator, clean[:300], "", "remove_duplicate_paragraph", "1", "duplicate_nonfinancial_paragraph", "medium"))
            continue
        seen.add(dedupe_key)
        para_idx += 1
        para_id = f"MDA_P{para_idx:03d}"
        if not output_parts or not output_parts[-1].startswith(f"[SECTION: {current_section}]"):
            output_parts.append(f"[SECTION: {current_section}]")
        output_parts.append(f"[{para_id}]\n{clean}")
        mapping.append(_mapping_row(document_id, para_id, current_section, raw_locator, clean[:300], clean, "normalize_whitespace;renumber_paragraph", "0", "", "high"))
    return "\n\n".join(output_parts).strip() + "\n", mapping


def clean_mda_texts_v2(root: Path = SCORING_ROOT, input_dir: Path | None = None) -> dict[str, Any]:
    input_dir = input_dir or _choose_input_dir(root)
    raw_dir = root / "02_extracted_text" / "mda_raw"
    clean_dir = root / "02_extracted_text" / "mda_clean_v2"
    map_dir = root / "02_extracted_text" / "mda_clean_v2_mapping"
    for directory in [raw_dir, clean_dir, map_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(input_dir.glob("*.txt")):
        document_id = path.stem
        raw = path.read_text(encoding="utf-8", errors="replace")
        (raw_dir / path.name).write_text(raw, encoding="utf-8")
        cleaned, mapping = clean_one_mda_text(document_id, raw)
        (clean_dir / path.name).write_text(cleaned, encoding="utf-8")
        write_csv_rows(map_dir / f"{document_id}_mapping.csv", list(mapping[0].keys()) if mapping else _mapping_headers(), mapping)
        count += 1
    report = root / "08_reports" / "mda_cleaning_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"# MDA Cleaning Report\n\n- input_dir: {input_dir}\n- cleaned_documents: {count}\n- raw text copied: {raw_dir}\n- cleaned text: {clean_dir}\n- mapping: {map_dir}\n", encoding="utf-8")
    return {"input_dir": str(input_dir), "cleaned_documents": count}


def _choose_input_dir(root: Path) -> Path:
    for rel in ["02_extracted_text/mda", "02_extracted_text/mda_raw"]:
        path = root / rel
        if path.exists() and list(path.glob("*.txt")):
            return path
    return root / "02_extracted_text" / "mda"


def _blocks(text: str) -> list[tuple[str, str]]:
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    expanded_chunks: list[str] = []
    for chunk in chunks:
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if len(lines) > 1 and _section_for(lines[0]):
            expanded_chunks.append(lines[0])
            expanded_chunks.append(" ".join(lines[1:]))
        else:
            expanded_chunks.append(chunk)
    chunks = expanded_chunks
    if len(chunks) <= 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = []
        current: list[str] = []
        for line in lines:
            if current and (_section_for(line) or len(" ".join(current)) > 900):
                chunks.append(" ".join(current))
                current = []
            current.append(line)
        if current:
            chunks.append(" ".join(current))
    return [(f"B{idx:03d}", re.sub(r"\s+", " ", chunk).strip()) for idx, chunk in enumerate(chunks, start=1)]


def _section_for(text: str) -> str:
    upper = text.upper()
    if len(text) <= 120 and not re.search(r"[.!?]", text):
        for key, section in SECTION_TERMS.items():
            if key.upper() in upper:
                return section
    return ""


def _looks_like_header_footer(line: str) -> bool:
    if FINANCIAL_TERMS.search(line):
        return False
    upper = line.upper()
    return bool(re.search(r"ANNUAL REPORT \d{4}|CORPORATE GOVERNANCE|ADDITIONAL INFORMATION", upper)) and len(line) < 90


def _mapping_headers() -> list[str]:
    return ["document_id", "clean_paragraph_id", "clean_section", "raw_locator", "raw_text_snippet", "clean_text", "cleaning_actions", "is_removed", "removal_reason", "confidence_level"]


def _mapping_row(document_id: str, para_id: str, section: str, raw_locator: str, raw_snippet: str, clean_text: str, actions: str, is_removed: str, reason: str, confidence: str) -> dict[str, str]:
    return {
        "document_id": document_id,
        "clean_paragraph_id": para_id,
        "clean_section": section or "Unknown",
        "raw_locator": raw_locator,
        "raw_text_snippet": raw_snippet,
        "clean_text": clean_text,
        "cleaning_actions": actions,
        "is_removed": is_removed,
        "removal_reason": reason,
        "confidence_level": confidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir")
    args = parser.parse_args()
    print(clean_mda_texts_v2(input_dir=Path(args.input_dir) if args.input_dir else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
