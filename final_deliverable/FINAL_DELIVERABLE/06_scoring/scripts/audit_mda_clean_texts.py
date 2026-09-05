from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, write_csv_rows


HEADERS = [
    "document_id", "raw_word_count", "clean_word_count", "cleaning_loss_ratio", "paragraph_count", "section_count",
    "duplicate_ratio", "numeric_mention_count", "percentage_mention_count", "rm_mention_count",
    "has_revenue_terms", "has_profit_terms", "has_cost_terms", "has_margin_terms", "has_segment_terms",
    "has_risk_terms", "has_outlook_terms", "has_market_terms", "has_strategy_terms",
    "short_text_flag", "high_cleaning_loss_flag", "low_numeric_content_flag", "missing_risk_outlook_flag",
    "possible_extraction_issue_flag", "needs_review", "review_reasons",
]


def audit_one(document_id: str, raw_text: str, clean_text: str) -> dict[str, Any]:
    raw_words = _word_count(raw_text)
    clean_words = _word_count(clean_text)
    loss = round((raw_words - clean_words) / raw_words, 6) if raw_words else 0
    paragraphs = len(re.findall(r"\[MDA_P\d{3}\]", clean_text))
    sections = len(re.findall(r"\[SECTION:", clean_text))
    clean_paragraphs = re.split(r"\[MDA_P\d{3}\]", clean_text)[1:]
    duplicate_ratio = 0
    if clean_paragraphs:
        normalized = [re.sub(r"\W+", "", p.lower()) for p in clean_paragraphs if p.strip()]
        duplicate_ratio = round(1 - len(set(normalized)) / len(normalized), 6) if normalized else 0
    lowered = clean_text.lower()
    numeric_mentions = len(re.findall(r"\b\d+(?:\.\d+)?\b", clean_text))
    percentage_mentions = len(re.findall(r"\d+(?:\.\d+)?\s*%", clean_text))
    rm_mentions = len(re.findall(r"\bRM\s*\d", clean_text, flags=re.I))
    flags = {
        "short_text_flag": int(clean_words < 300),
        "high_cleaning_loss_flag": int(loss > 0.40),
        "low_numeric_content_flag": int(numeric_mentions == 0),
        "missing_risk_outlook_flag": int(not ("risk" in lowered or "outlook" in lowered or "prospect" in lowered)),
        "possible_extraction_issue_flag": int(paragraphs < 3),
    }
    reasons = [key for key, value in flags.items() if value]
    return {
        "document_id": document_id,
        "raw_word_count": raw_words,
        "clean_word_count": clean_words,
        "cleaning_loss_ratio": loss,
        "paragraph_count": paragraphs,
        "section_count": sections,
        "duplicate_ratio": duplicate_ratio,
        "numeric_mention_count": numeric_mentions,
        "percentage_mention_count": percentage_mentions,
        "rm_mention_count": rm_mentions,
        "has_revenue_terms": int("revenue" in lowered or "sales" in lowered),
        "has_profit_terms": int("profit" in lowered),
        "has_cost_terms": int("cost" in lowered or "expense" in lowered),
        "has_margin_terms": int("margin" in lowered),
        "has_segment_terms": int("segment" in lowered),
        "has_risk_terms": int("risk" in lowered),
        "has_outlook_terms": int("outlook" in lowered or "prospect" in lowered),
        "has_market_terms": int("market" in lowered),
        "has_strategy_terms": int("strategy" in lowered or "strategies" in lowered),
        **flags,
        "needs_review": int(bool(reasons)),
        "review_reasons": ";".join(reasons),
    }


def audit_mda_clean_texts(root: Path = SCORING_ROOT) -> dict[str, Any]:
    raw_dir = root / "02_extracted_text" / "mda_raw"
    clean_dir = root / "02_extracted_text" / "mda_clean_v2"
    rows = []
    for clean_path in sorted(clean_dir.glob("*.txt")):
        raw_path = raw_dir / clean_path.name
        rows.append(audit_one(clean_path.stem, raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else "", clean_path.read_text(encoding="utf-8", errors="replace")))
    out_csv = root / "08_reports" / "mda_text_quality_audit.csv"
    write_csv_rows(out_csv, HEADERS, rows)
    needs = sum(int(row["needs_review"]) for row in rows)
    (root / "08_reports" / "mda_text_quality_audit_report.md").write_text(f"# MDA Text Quality Audit Report\n\n- documents: {len(rows)}\n- needs_review_documents: {needs}\n", encoding="utf-8")
    return {"documents": len(rows), "needs_review_documents": needs}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def main() -> int:
    print(audit_mda_clean_texts())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
