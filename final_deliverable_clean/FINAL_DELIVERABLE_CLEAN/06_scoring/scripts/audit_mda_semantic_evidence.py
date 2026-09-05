from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows
from stability_common import safe_float, write_markdown


AUDIT_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "report_year",
    "dimension_code",
    "raw_score",
    "evidence_locator",
    "evidence_exists",
    "evidence_text_short",
    "evidence_semantic_status",
    "evidence_issue_type",
    "score_evidence_alignment",
    "needs_review",
    "recommended_action",
]

NOISE_TERMS = [
    "registered office",
    "principal bankers",
    "board of directors",
    "corporate information",
    "notice of annual general meeting",
    "proxy form",
]

KEYWORDS = {
    "AR01": ["operation", "business", "financial", "market", "risk", "outlook", "prospect", "revenue", "profit", "segment", "performance", "management discussion"],
    "AR02": ["revenue", "sales", "profit", "loss", "cost", "margin", "cash flow", "rm", "%", "segment", "dividend", "finance cost", "million", "sen"],
    "AR03": ["due to", "because", "driven by", "mainly attributable", "demand", "sales volume", "market", "export", "cost increase", "product", "segment", "growth", "decline"],
    "AR04": ["risk", "challenge", "uncertain", "outlook", "future", "prospect", "raw material", "supply chain", "foreign exchange", "competition", "regulation", "food safety", "halal"],
    "AR05": ["section", "table", "summary", "paragraph", "discussion", "analysis", "review", "structured", "segment", "outlook", "risk", "financial"],
}


def audit_mda_semantic_evidence(root: Path = SCORING_ROOT) -> dict[str, Any]:
    scores_path = _find_existing(root, ["06_ratings/mda_final_full/mda_final_document_scores.csv"])
    ratings_path = _find_existing(root, ["06_ratings/mda_final_full/mda_final_ratings_long.csv"])
    score_rows = {row.get("document_id", ""): row for row in read_csv_rows(scores_path)}
    rating_rows = read_csv_rows(ratings_path)
    text_cache: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []

    for row in rating_rows:
        doc_id = row.get("document_id", "")
        score_row = score_rows.get(doc_id, {})
        full_text = text_cache.setdefault(doc_id, _load_text(root, doc_id, score_row))
        locator = row.get("evidence_locator", "")
        evidence_text = _extract_evidence_text(locator, full_text)
        evidence_exists = bool(evidence_text.strip())
        classification = classify_evidence(row.get("dimension_code", ""), evidence_text, row.get("comment_short", ""))
        if not evidence_exists:
            classification = {
                "evidence_semantic_status": "invalid",
                "evidence_issue_type": "evidence_not_in_text",
                "score_evidence_alignment": "not_checkable",
                "recommended_action": "check_text_cleaning",
            }
        audit_row = {
            "document_id": doc_id,
            "ticker": score_row.get("ticker", ""),
            "company_name": score_row.get("company_name", ""),
            "report_year": score_row.get("report_year", ""),
            "dimension_code": row.get("dimension_code", ""),
            "raw_score": row.get("raw_score", ""),
            "evidence_locator": locator,
            "evidence_exists": str(evidence_exists).lower(),
            "evidence_text_short": _short(evidence_text or row.get("evidence_text_short", "")),
            "needs_review": str(classification["evidence_semantic_status"] in {"weak", "invalid", "not_checkable"}).lower(),
            **classification,
        }
        audit_rows.append(audit_row)

    write_csv_rows(root / "08_reports" / "mda_semantic_evidence_audit.csv", AUDIT_HEADERS, audit_rows)
    review_rows = [row for row in audit_rows if row["needs_review"] == "true"]
    write_csv_rows(root / "07_review" / "mda_semantic_evidence_review_pool.csv", AUDIT_HEADERS, review_rows)
    summary = _summary(audit_rows)
    _write_report(root / "08_reports" / "mda_semantic_evidence_audit.md", summary)
    return summary


def classify_evidence(dimension_code: str, evidence_text: str, reason: str = "") -> dict[str, str]:
    text = " ".join([evidence_text or "", reason or ""]).lower()
    evidence_lower = (evidence_text or "").lower()
    if not evidence_text.strip():
        return _classification("not_checkable", "evidence_not_in_text", "not_checkable", "check_text_cleaning")
    if any(term in evidence_lower for term in NOISE_TERMS):
        return _classification("invalid", "non_mda_noise", "misaligned", "check_text_cleaning")

    hits = _keyword_hits(dimension_code, text)
    if dimension_code == "AR02":
        if not hits and not re.search(r"\d+(?:\.\d+)?\s*(?:%|million|sen|rm)", text):
            return _classification("invalid", "missing_financial_content", "misaligned", "rerun_dimension_only")
        if "numeric" in text or "not calculable" in text:
            return _classification("acceptable", "numeric_review_required", "partial", "manual_review")
    if dimension_code == "AR03" and not hits:
        return _classification("weak", "missing_causal_explanation", "partial", "manual_review")
    if dimension_code == "AR04" and not hits:
        return _classification("weak", "missing_risk_or_outlook", "partial", "manual_review")
    if dimension_code == "AR05" and not hits:
        return _classification("weak", "missing_structure_basis", "partial", "manual_review")
    if dimension_code == "AR01" and not hits:
        return _classification("weak", "too_generic", "partial", "manual_review")

    status = "strong" if hits >= 2 else "acceptable"
    return _classification(status, "none", "aligned", "accept")


def _classification(status: str, issue: str, alignment: str, action: str) -> dict[str, str]:
    return {
        "evidence_semantic_status": status,
        "evidence_issue_type": issue,
        "score_evidence_alignment": alignment,
        "recommended_action": action,
    }


def _keyword_hits(dimension_code: str, text: str) -> int:
    return sum(1 for keyword in KEYWORDS.get(dimension_code, []) if keyword in text)


def _find_existing(root: Path, candidates: list[str]) -> Path:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    raise FileNotFoundError(candidates[0])


def _load_text(root: Path, doc_id: str, score_row: dict[str, str]) -> str:
    candidates = [
        root / "02_extracted_text" / "mda_clean_v2" / f"{doc_id}.txt",
        root / "02_extracted_text" / "mda" / f"{doc_id}.txt",
    ]
    registry = {row.get("document_id", ""): row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv")}
    reg_row = registry.get(doc_id, {})
    if reg_row.get("text_path"):
        candidates.insert(0, root / reg_row["text_path"])
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _extract_evidence_text(locator: str, full_text: str) -> str:
    ids = re.findall(r"MDA_P\d+(?:_\d+)?", locator or "")
    if not ids or not full_text:
        return ""
    paragraphs = _paragraph_map(full_text)
    chunks = [paragraphs.get(eid, "") for eid in ids]
    return " ".join(chunk for chunk in chunks if chunk).strip()


def _paragraph_map(full_text: str) -> dict[str, str]:
    matches = list(re.finditer(r"\[(MDA_P\d+(?:_\d+)?)\]", full_text))
    out: dict[str, str] = {}
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        out[match.group(1)] = " ".join(full_text[start:end].split())
    return out


def _short(text: str, limit: int = 280) -> str:
    cleaned = " ".join(str(text or "").split())
    return cleaned[:limit]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(row["evidence_semantic_status"] for row in rows)
    by_dimension: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    problem_docs: Counter[str] = Counter()
    for row in rows:
        by_dimension[row["dimension_code"]][row["evidence_semantic_status"]] += 1
        if row["needs_review"] == "true":
            problem_docs[row["document_id"]] += 1
    summary = {
        "total_dimension_rows": len(rows),
        "strong_count": status_counts.get("strong", 0),
        "acceptable_count": status_counts.get("acceptable", 0),
        "weak_count": status_counts.get("weak", 0),
        "invalid_count": status_counts.get("invalid", 0),
        "needs_review_count": sum(1 for row in rows if row["needs_review"] == "true"),
        "needs_rerun_count": sum(1 for row in rows if row["recommended_action"] == "rerun_dimension_only"),
        "by_dimension_summary": {dim: dict(counts) for dim, counts in by_dimension.items()},
        "top_problem_documents": [{"document_id": doc_id, "issue_count": count} for doc_id, count in problem_docs.most_common(20)],
    }
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# MD&A Semantic Evidence Audit", ""]
    for key in ["total_dimension_rows", "strong_count", "acceptable_count", "weak_count", "invalid_count", "needs_review_count", "needs_rerun_count"]:
        lines.append(f"- {key}: {summary[key]}")
    lines.extend(["", "## By Dimension", ""])
    for dim, counts in summary["by_dimension_summary"].items():
        lines.append(f"- {dim}: {json.dumps(counts, sort_keys=True)}")
    lines.extend(["", "## Top Problem Documents", ""])
    for item in summary["top_problem_documents"]:
        lines.append(f"- {item['document_id']}: {item['issue_count']}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit semantic support of MD&A evidence locators.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(audit_mda_semantic_evidence(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
