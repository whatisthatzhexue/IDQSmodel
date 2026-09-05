from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from validate_prompt_context_budget import estimate_tokens


MANIFEST_HEADERS = [
    "document_id",
    "dimension_code",
    "source_paragraph_count",
    "selected_paragraph_count",
    "selected_paragraph_ids",
    "selection_reason",
    "input_char_count",
    "estimated_token_count",
    "context_budget",
    "budget_passed",
    "fallback_used",
]

DIMENSION_CODES = ["AR01", "AR02", "AR03", "AR04", "AR05"]

AR02_KEYWORDS = [
    "revenue",
    "sales",
    "profit",
    "cost",
    "margin",
    "cash flow",
    "cashflow",
    "segment",
    "rm",
    "%",
    "dividend",
    "pbt",
    "pat",
    "earnings",
]
AR03_KEYWORDS = [
    "revenue",
    "sales",
    "profit",
    "cost",
    "margin",
    "volume",
    "price",
    "product",
    "segment",
    "region",
    "demand",
    "raw material",
    "fx",
    "export",
    "capacity",
    "because",
    "due to",
    "driven by",
    "attributable",
    "resulting from",
    "mainly attributable",
]
AR04_KEYWORDS = [
    "risk",
    "challenge",
    "uncertainty",
    "outlook",
    "prospect",
    "future",
    "raw material",
    "food safety",
    "supply chain",
    "fx",
    "competition",
    "regulation",
    "halal",
    "mitigation",
    "response",
    "inflation",
]
COVERAGE_KEYWORDS = {
    "financial": ["revenue", "profit", "sales", "margin", "cash", "dividend", "rm", "%"],
    "business": ["product", "segment", "market", "customer", "export", "domestic", "operations"],
    "drivers": ["because", "due to", "driven by", "attributable", "resulting from", "mainly"],
    "risk": ["risk", "challenge", "uncertainty", "supply chain", "inflation", "competition"],
    "outlook": ["outlook", "prospect", "future", "going forward", "priorities"],
}


@dataclass(frozen=True)
class Paragraph:
    paragraph_id: str
    text: str
    index: int
    section_heading: str = "Unknown"


@dataclass(frozen=True)
class DimensionContext:
    document_id: str
    dimension_code: str
    context_text: str
    selected_paragraph_ids: list[str]
    source_paragraph_count: int
    selected_paragraph_count: int
    selection_reason: str
    input_char_count: int
    estimated_token_count: int
    context_budget: int
    budget_passed: bool
    fallback_used: bool
    output_path: str

    def manifest_row(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "dimension_code": self.dimension_code,
            "source_paragraph_count": self.source_paragraph_count,
            "selected_paragraph_count": self.selected_paragraph_count,
            "selected_paragraph_ids": "|".join(self.selected_paragraph_ids),
            "selection_reason": self.selection_reason,
            "input_char_count": self.input_char_count,
            "estimated_token_count": self.estimated_token_count,
            "context_budget": self.context_budget,
            "budget_passed": self.budget_passed,
            "fallback_used": self.fallback_used,
        }


def select_document_dimension_contexts(
    *,
    root: Path = SCORING_ROOT,
    document_id: str,
    text_path: Path,
    numeric_check_path: Path | None = None,
    context_budget: int = 2600,
) -> dict[str, DimensionContext]:
    text = text_path.read_text(encoding="utf-8", errors="replace")
    paragraphs = parse_mda_paragraphs(text)
    numeric_summary = _load_numeric_summary(numeric_check_path)
    contexts = {
        code: _build_context(root, document_id, code, paragraphs, numeric_summary, context_budget)
        for code in DIMENSION_CODES
    }
    _write_manifest(root, list(contexts.values()))
    return contexts


def parse_mda_paragraphs(text: str) -> list[Paragraph]:
    section_spans = _section_spans(text)
    marker_re = re.compile(r"(?m)^\[(MDA_P\d{3})\]\s*$")
    matches = list(marker_re.finditer(text))
    if not matches:
        stripped = text.strip()
        return [Paragraph("MDA_P001", stripped, 0, "Unknown")] if stripped else []
    paragraphs: list[Paragraph] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        body = "\n".join(line for line in body.splitlines() if not line.strip().startswith("[SECTION:"))
        paragraph_id = match.group(1)
        if body:
            paragraphs.append(
                Paragraph(
                    paragraph_id=paragraph_id,
                    text=body.strip(),
                    index=len(paragraphs),
                    section_heading=_section_for_offset(section_spans, match.start()),
                )
            )
    return paragraphs


def _build_context(
    root: Path,
    document_id: str,
    dimension_code: str,
    paragraphs: list[Paragraph],
    numeric_summary: dict[str, Any],
    context_budget: int,
) -> DimensionContext:
    scored = _scored_paragraphs(dimension_code, paragraphs, numeric_summary)
    selected, fallback_used = _fit_budget(
        document_id=document_id,
        dimension_code=dimension_code,
        paragraphs=paragraphs,
        scored=scored,
        numeric_summary=numeric_summary,
        context_budget=context_budget,
    )
    context_text = _format_context(document_id, dimension_code, paragraphs, selected, numeric_summary)
    estimated = estimate_tokens(context_text)
    budget_passed = estimated <= context_budget
    if not budget_passed and selected:
        selected, fallback_used = _shrink_selected(document_id, dimension_code, paragraphs, selected, numeric_summary, context_budget)
        context_text = _format_context(document_id, dimension_code, paragraphs, selected, numeric_summary)
        estimated = estimate_tokens(context_text)
        budget_passed = estimated <= context_budget
    output_dir = root / "02_extracted_text" / "mda_dimension_context"
    output_path = output_dir / f"{document_id}_{dimension_code}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(context_text, encoding="utf-8")
    return DimensionContext(
        document_id=document_id,
        dimension_code=dimension_code,
        context_text=context_text,
        selected_paragraph_ids=[paragraphs[idx].paragraph_id for idx in selected],
        source_paragraph_count=len(paragraphs),
        selected_paragraph_count=len(selected),
        selection_reason=_selection_reason(dimension_code),
        input_char_count=len(context_text),
        estimated_token_count=estimated,
        context_budget=context_budget,
        budget_passed=budget_passed,
        fallback_used=fallback_used,
        output_path=str(output_path),
    )


def _scored_paragraphs(
    dimension_code: str,
    paragraphs: list[Paragraph],
    numeric_summary: dict[str, Any],
) -> dict[int, int]:
    if not paragraphs:
        return {}
    if dimension_code == "AR01":
        return _coverage_selection(paragraphs)
    if dimension_code == "AR02":
        return _keyword_selection_with_neighbors(paragraphs, AR02_KEYWORDS, numeric_summary)
    if dimension_code == "AR03":
        return _keyword_selection(paragraphs, AR03_KEYWORDS, causal_bonus=True)
    if dimension_code == "AR04":
        return _keyword_selection(paragraphs, AR04_KEYWORDS)
    return _structure_selection(paragraphs)


def _coverage_selection(paragraphs: list[Paragraph]) -> dict[int, int]:
    selected: dict[int, int] = {}
    for category, keywords in COVERAGE_KEYWORDS.items():
        ranked = sorted(
            ((idx, _keyword_count(par.text, keywords)) for idx, par in enumerate(paragraphs)),
            key=lambda item: item[1],
            reverse=True,
        )
        for idx, score in ranked[:2]:
            if score > 0:
                selected[idx] = max(selected.get(idx, 0), 80 + score)
    for idx in [0, len(paragraphs) // 2, len(paragraphs) - 1]:
        if 0 <= idx < len(paragraphs):
            selected[idx] = max(selected.get(idx, 0), 20)
    return selected


def _keyword_selection_with_neighbors(
    paragraphs: list[Paragraph],
    keywords: list[str],
    numeric_summary: dict[str, Any],
) -> dict[int, int]:
    selected: dict[int, int] = {}
    evidence_ids = _numeric_evidence_ids(numeric_summary)
    for idx, paragraph in enumerate(paragraphs):
        score = _keyword_count(paragraph.text, keywords)
        if paragraph.paragraph_id in evidence_ids:
            score += 50
        if score <= 0:
            continue
        selected[idx] = max(selected.get(idx, 0), 100 + score)
        for neighbor in [idx - 1, idx + 1]:
            if 0 <= neighbor < len(paragraphs):
                selected[neighbor] = max(selected.get(neighbor, 0), 30)
    return selected


def _keyword_selection(paragraphs: list[Paragraph], keywords: list[str], *, causal_bonus: bool = False) -> dict[int, int]:
    selected: dict[int, int] = {}
    causal_terms = ["because", "due to", "driven by", "resulting from", "mainly attributable", "attributable"]
    for idx, paragraph in enumerate(paragraphs):
        score = _keyword_count(paragraph.text, keywords)
        if causal_bonus:
            score += 20 * _keyword_count(paragraph.text, causal_terms)
        if score > 0:
            selected[idx] = 80 + score
    return selected


def _structure_selection(paragraphs: list[Paragraph]) -> dict[int, int]:
    selected: dict[int, int] = {}
    if not paragraphs:
        return selected
    sections_seen: set[str] = set()
    for idx, paragraph in enumerate(paragraphs):
        if paragraph.section_heading not in sections_seen:
            selected[idx] = 60
            sections_seen.add(paragraph.section_heading)
    for idx in [0, len(paragraphs) // 4, len(paragraphs) // 2, (3 * len(paragraphs)) // 4, len(paragraphs) - 1]:
        selected[idx] = max(selected.get(idx, 0), 40)
    return selected


def _fit_budget(
    *,
    document_id: str,
    dimension_code: str,
    paragraphs: list[Paragraph],
    scored: dict[int, int],
    numeric_summary: dict[str, Any],
    context_budget: int,
) -> tuple[list[int], bool]:
    if not scored and paragraphs:
        scored = {0: 10}
    selected = [idx for idx, _score in sorted(scored.items(), key=lambda item: (-item[1], item[0]))]
    selected = sorted(selected)
    if estimate_tokens(_format_context(document_id, dimension_code, paragraphs, selected, numeric_summary)) <= context_budget:
        return selected, False
    return _shrink_selected(document_id, dimension_code, paragraphs, selected, numeric_summary, context_budget)


def _shrink_selected(
    document_id: str,
    dimension_code: str,
    paragraphs: list[Paragraph],
    selected: list[int],
    numeric_summary: dict[str, Any],
    context_budget: int,
) -> tuple[list[int], bool]:
    kept = list(selected)
    while len(kept) > 1 and estimate_tokens(_format_context(document_id, dimension_code, paragraphs, kept, numeric_summary)) > context_budget:
        kept.pop()
    return kept, kept != selected


def _format_context(
    document_id: str,
    dimension_code: str,
    paragraphs: list[Paragraph],
    selected_indices: list[int],
    numeric_summary: dict[str, Any],
) -> str:
    selected = [paragraphs[idx] for idx in selected_indices if 0 <= idx < len(paragraphs)]
    feature_payload = _feature_payload(paragraphs)
    parts = [
        f"document_id: {document_id}",
        f"dimension_code: {dimension_code}",
        "context_scope: dimension_specific_selected_paragraphs_only",
        "source_features_json:",
        json.dumps(feature_payload, ensure_ascii=False, sort_keys=True),
    ]
    if dimension_code == "AR02":
        parts.extend(["numeric_check_summary:", json.dumps(_compact_numeric_summary(numeric_summary), ensure_ascii=False, sort_keys=True)])
    if dimension_code == "AR05":
        parts.extend(["structure_features_json:", json.dumps(_structure_features(paragraphs), ensure_ascii=False, sort_keys=True)])
    parts.append("selected_paragraphs:")
    for paragraph in selected:
        parts.append(f"[{paragraph.paragraph_id}]")
        parts.append(paragraph.text)
    return "\n".join(parts).strip() + "\n"


def _feature_payload(paragraphs: list[Paragraph]) -> dict[str, Any]:
    sections = [item for item in dict.fromkeys(par.section_heading for par in paragraphs) if item]
    coverage = {
        category: any(_keyword_count(par.text, keywords) > 0 for par in paragraphs)
        for category, keywords in COVERAGE_KEYWORDS.items()
    }
    return {
        "section_headings": sections[:30],
        "section_count": len(sections),
        "paragraph_count": len(paragraphs),
        "coverage_map": coverage,
        "has_financial_or_business_or_risk_or_outlook": any(coverage.values()),
    }


def _structure_features(paragraphs: list[Paragraph]) -> dict[str, Any]:
    lengths = [len(par.text) for par in paragraphs]
    sentence_counts = [max(1, len(re.findall(r"[.!?。！？]", par.text))) for par in paragraphs]
    avg_sentence_length = 0.0
    if paragraphs:
        avg_sentence_length = round(sum(len(par.text) / count for par, count in zip(paragraphs, sentence_counts)) / len(paragraphs), 3)
    duplicate_ratio = 0.0
    if paragraphs:
        normalized = [" ".join(par.text.lower().split())[:180] for par in paragraphs]
        duplicate_ratio = round(1 - (len(set(normalized)) / len(normalized)), 6)
    return {
        "paragraph_count": len(paragraphs),
        "section_count": len({par.section_heading for par in paragraphs}),
        "average_sentence_length_chars": avg_sentence_length,
        "paragraph_length_distribution": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "mean": round(sum(lengths) / len(lengths), 3) if lengths else 0,
        },
        "repetition_ratio": duplicate_ratio,
    }


def _compact_numeric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        return {}
    keys = [
        "document_id",
        "overall_numeric_consistency",
        "num_financial_mentions",
        "num_calculable_changes",
        "num_consistent_checks",
        "num_material_discrepancies",
        "num_minor_discrepancies",
        "num_severe_direction_conflicts",
        "has_revenue_discussion",
        "has_profit_discussion",
        "has_cost_discussion",
        "has_margin_discussion",
        "has_segment_discussion",
    ]
    result = {key: payload.get(key) for key in keys if key in payload}
    material_checks = [
        {
            "status": check.get("status"),
            "evidence_locator": check.get("evidence_locator"),
            "source_text": check.get("source_text"),
            "note": check.get("note"),
        }
        for check in payload.get("checks", [])
        if str(check.get("status", "")).lower() in {"material_discrepancy", "severe_direction_conflict"}
    ]
    if material_checks:
        result["audited_or_material_discrepancies"] = material_checks[:20]
    return result


def _load_numeric_summary(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_manifest(root: Path, contexts: list[DimensionContext]) -> None:
    path = root / "02_extracted_text" / "mda_dimension_context" / "context_selection_manifest.csv"
    existing = [
        row
        for row in read_csv_rows(path)
        if (row.get("document_id"), row.get("dimension_code"))
        not in {(ctx.document_id, ctx.dimension_code) for ctx in contexts}
    ]
    rows = existing + [ctx.manifest_row() for ctx in contexts]
    write_csv_rows(path, MANIFEST_HEADERS, rows)


def _section_spans(text: str) -> list[tuple[int, str]]:
    spans = []
    for match in re.finditer(r"(?m)^\[SECTION:\s*([^\]]+)\]\s*$", text):
        spans.append((match.start(), match.group(1).strip() or "Unknown"))
    return spans


def _section_for_offset(spans: list[tuple[int, str]], offset: int) -> str:
    current = "Unknown"
    for section_offset, heading in spans:
        if section_offset <= offset:
            current = heading
        else:
            break
    return current


def _selection_reason(dimension_code: str) -> str:
    return {
        "AR01": "coverage_map_and_representative_sections",
        "AR02": "financial_numeric_keywords_with_neighbor_context_and_numeric_summary",
        "AR03": "operating_driver_keywords_with_causal_priority",
        "AR04": "risk_outlook_challenge_keywords",
        "AR05": "python_structure_features_and_representative_paragraphs",
    }[dimension_code]


def _keyword_count(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    count = 0
    for keyword in keywords:
        key = keyword.lower()
        if key == "%":
            count += 1 if "%" in lowered else 0
        elif key == "rm":
            count += 1 if re.search(r"\brm\b|\brm(?=\d)|\brm\s", lowered) else 0
        elif re.fullmatch(r"[a-z0-9]+", key):
            count += 1 if re.search(rf"\b{re.escape(key)}\b", lowered) else 0
        else:
            count += 1 if key in lowered else 0
    return count


def _numeric_evidence_ids(payload: dict[str, Any]) -> set[str]:
    evidence_ids: set[str] = set()
    for check in payload.get("checks", []):
        locator = str(check.get("evidence_locator") or check.get("source_text") or "")
        evidence_ids.update(re.findall(r"MDA_P\d{3}", locator))
    return evidence_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dimension-specific MD&A context files.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--numeric-check-path", type=Path)
    parser.add_argument("--context-budget", type=int, default=2600)
    args = parser.parse_args()
    contexts = select_document_dimension_contexts(
        root=args.root,
        document_id=args.document_id,
        text_path=args.text_path,
        numeric_check_path=args.numeric_check_path,
        context_budget=args.context_budget,
    )
    print(json.dumps({code: context.manifest_row() for code, context in contexts.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
