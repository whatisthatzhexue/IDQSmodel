from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, save_json, write_csv_rows


METRIC_TERMS = [
    "revenue",
    "sales",
    "turnover",
    "profit before tax",
    "profit after tax",
    "net profit",
    "gross profit",
    "profit",
    "gross margin",
    "operating profit",
    "ebitda",
    "ebit",
    "cost of sales",
    "raw material cost",
    "distribution cost",
    "administrative expenses",
    "cash flow",
    "operating cash flow",
    "borrowings",
    "debt",
    "assets",
    "liabilities",
    "dividend",
    "segment revenue",
    "segment profit",
]

INCREASE_WORDS = {"increased", "rose", "improved", "grew", "higher"}
DECREASE_WORDS = {"decreased", "declined", "fell", "deteriorated", "reduced", "lower"}
CHANGE_WORDS = sorted(INCREASE_WORDS | DECREASE_WORDS, key=len, reverse=True)
CHANGE_NOUNS = ["increase", "decrease", "decline", "reduction", "growth"]

AMOUNT_RE = r"(?:RM\s*)?(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(RM'000|RM000|million|billion|sen|%)?"
PERCENT_RE = r"(-?\d+(?:\.\d+)?)\s*(?:%|per cent|percent|percentage points?)"
METRIC_RE = r"(" + "|".join(re.escape(term) for term in sorted(METRIC_TERMS, key=len, reverse=True)) + r")"
CHANGE_RE = r"(" + "|".join(CHANGE_WORDS) + r")"


def calculate_change_pct(current_value: float, previous_value: float) -> float | dict[str, str]:
    if previous_value == 0:
        return {"status": "not_calculable_previous_zero"}
    return round((current_value - previous_value) / abs(previous_value) * 100, 6)


def classify_percentage_delta(stated_pct: float, calculated_pct: float) -> str:
    diff = abs(stated_pct - calculated_pct)
    if diff <= 1:
        return "consistent"
    if diff <= 3:
        return "minor_discrepancy"
    return "material_discrepancy"


def detect_direction_conflict(text: str, calculated_pct: float) -> bool:
    lowered = text.lower()
    says_increase = any(word in lowered for word in INCREASE_WORDS)
    says_decrease = any(word in lowered for word in DECREASE_WORDS)
    if says_increase and calculated_pct < 0:
        return True
    if says_decrease and calculated_pct > 0:
        return True
    return False


def parse_amount(value: str, unit: str | None) -> float:
    number = float(value.replace(",", ""))
    normalized = (unit or "").lower()
    if normalized == "billion":
        return number * 1000
    if normalized in {"rm'000", "rm000"}:
        return number / 1000
    if normalized == "sen":
        return number / 100
    return number


def analyze_file(path: Path, document_id: str | None = None) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return analyze_text(text, document_id=document_id or path.stem)


def analyze_text(text: str, document_id: str) -> dict[str, Any]:
    checks = _extract_change_checks(text)
    counts = {
        "consistent": 0,
        "minor_discrepancy": 0,
        "material_discrepancy": 0,
        "severe_direction_conflict": 0,
        "not_calculable": 0,
    }
    notes: list[str] = []
    for check in checks:
        status = check["status"]
        if status == "severe_direction_conflict":
            counts["severe_direction_conflict"] += 1
        elif status == "not_calculable_previous_zero":
            counts["not_calculable"] += 1
        elif status == "special_case_negative_base":
            counts["not_calculable"] += 1
        elif status == "needs_manual_review":
            counts["not_calculable"] += 1
        else:
            counts[status] += 1
        notes.append(check["note"])

    lowered = text.lower()
    result = {
        "document_id": document_id,
        "num_financial_mentions": _count_financial_mentions(lowered),
        "num_calculable_changes": sum(
            1
            for item in checks
            if item["status"]
            in {"consistent", "minor_discrepancy", "material_discrepancy", "severe_direction_conflict"}
        ),
        "num_consistent_checks": counts["consistent"],
        "num_minor_discrepancies": counts["minor_discrepancy"],
        "num_material_discrepancies": counts["material_discrepancy"],
        "num_severe_direction_conflicts": counts["severe_direction_conflict"],
        "num_not_calculable": counts["not_calculable"],
        "has_revenue_discussion": _has_any(lowered, ["revenue", "sales", "turnover"]),
        "has_profit_discussion": _has_any(lowered, ["profit", "ebit", "ebitda"]),
        "has_cost_discussion": _has_any(lowered, ["cost", "expense", "raw material"]),
        "has_segment_discussion": _has_any(lowered, ["segment"]),
        "has_margin_discussion": _has_any(lowered, ["margin"]),
        "overall_numeric_consistency": _overall_consistency(counts, checks),
        "internal_check_notes": notes,
        "checks": checks,
    }
    return result


def _extract_change_checks(text: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    normalized = _normalize_numeric_text_for_patterns(" ".join(text.split()))
    patterns = [
        (
            "direction_by_from_to",
            re.compile(
            rf"{METRIC_RE}[^.]*?{CHANGE_RE}[^.]*?by\s+{PERCENT_RE}[^.]*?from\s+{AMOUNT_RE}\s+to\s+{AMOUNT_RE}",
            re.IGNORECASE,
            ),
        ),
        (
            "from_to_direction_by",
            re.compile(
            rf"{METRIC_RE}[^.]*?from\s+{AMOUNT_RE}\s+to\s+{AMOUNT_RE}[^.]*?{CHANGE_RE}[^.]*?by\s+{PERCENT_RE}",
            re.IGNORECASE,
            ),
        ),
        (
            "percent_change_in_metric_as_compared",
            re.compile(
                rf"{PERCENT_RE}\s+({'|'.join(CHANGE_NOUNS)})\s+in\s+{METRIC_RE}[^.]*?of\s+{AMOUNT_RE}[^.]*?(?:as compared with|compared with|compared to|as compared to)\s+{AMOUNT_RE}",
                re.IGNORECASE,
            ),
        ),
    ]
    seen_spans: set[tuple[int, int]] = set()
    for pattern_name, pattern in patterns:
        for match in pattern.finditer(normalized):
            if match.span() in seen_spans:
                continue
            seen_spans.add(match.span())
            groups = match.groups()
            if pattern_name == "direction_by_from_to":
                metric, direction_word, stated_pct, prev_value, prev_unit, curr_value, curr_unit = groups
            elif pattern_name == "from_to_direction_by":
                metric, prev_value, prev_unit, curr_value, curr_unit, direction_word, stated_pct = groups
            else:
                stated_pct, direction_word, metric, curr_value, curr_unit, prev_value, prev_unit = groups
            previous = parse_amount(prev_value, prev_unit)
            current = parse_amount(curr_value, curr_unit)
            stated = float(stated_pct)
            snippet = match.group(0)
            year_count = len(set(re.findall(r"\b20\d{2}\b", snippet)))
            if year_count >= 3:
                calculated: float | dict[str, str] = {"status": "needs_manual_review"}
                status = "needs_manual_review"
                note = f"{metric}: multiple years in one numeric expression; manual review required."
            elif "percentage point" in snippet.lower():
                calculated = round(current - previous, 6)
                comparable_stated = -abs(stated) if direction_word.lower() in DECREASE_WORDS else abs(stated)
                status = classify_percentage_delta(comparable_stated, calculated)
                note = f"{metric}: stated {comparable_stated:.2f} percentage points, calculated {calculated:.2f} percentage points, status {status}."
            else:
                calculated = calculate_change_pct(current, previous)
            if isinstance(calculated, dict) and calculated.get("status") == "needs_manual_review":
                pass
            elif isinstance(calculated, dict):
                status = calculated["status"]
                note = f"{metric}: not calculable because previous value is zero."
            elif previous < 0:
                status = "special_case_negative_base"
                note = f"{metric}: negative base or profit/loss swing; ordinary growth rate is not forced."
            elif detect_direction_conflict(f"{direction_word} {snippet}", calculated):
                status = "severe_direction_conflict"
                note = f"{metric}: direction word conflicts with calculated change {calculated:.2f}%."
            else:
                comparable_stated = -abs(stated) if direction_word.lower() in DECREASE_WORDS else abs(stated)
                status = classify_percentage_delta(comparable_stated, calculated)
                note = f"{metric}: stated {comparable_stated:.2f}%, calculated {calculated:.2f}%, status {status}."
            locator = _locator_for_snippet(text, snippet)
            checks.append(
                {
                    "metric": metric.lower(),
                    "direction_word": direction_word.lower(),
                    "stated_pct": stated,
                    "previous_value": previous,
                    "current_value": current,
                    "calculated_change_pct": calculated,
                    "status": status,
                    "source_text": snippet[:300],
                    "evidence_locator": locator,
                    "note": note,
                }
            )
    checks.extend(_extract_sign_swing_checks(text))
    checks.extend(_extract_margin_checks(text))
    checks.extend(_extract_segment_sum_checks(text))
    return checks


def _extract_sign_swing_checks(text: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    normalized = _normalize_numeric_text_for_patterns(" ".join(text.split()))
    pattern = re.compile(rf"{METRIC_RE}[^.]*?{CHANGE_RE}[^.]*?from\s+{AMOUNT_RE}\s+to\s+{AMOUNT_RE}", re.IGNORECASE)
    for match in pattern.finditer(normalized):
        snippet = match.group(0)
        if re.search(r"\bby\s+" + PERCENT_RE, snippet, re.IGNORECASE):
            continue
        metric, direction_word, prev_value, prev_unit, curr_value, curr_unit = match.groups()
        previous = parse_amount(prev_value, prev_unit)
        current = parse_amount(curr_value, curr_unit)
        if previous < 0 or previous < 0 < current or previous > 0 > current:
            checks.append(
                {
                    "metric": metric.lower(),
                    "direction_word": direction_word.lower(),
                    "stated_pct": "",
                    "previous_value": previous,
                    "current_value": current,
                    "calculated_change_pct": {"status": "special_case_negative_base"},
                    "status": "special_case_negative_base",
                    "source_text": snippet[:300],
                    "evidence_locator": _locator_for_snippet(text, snippet),
                    "note": f"{metric}: negative base or profit/loss swing; ordinary growth rate is not forced.",
                }
            )
    return checks


def _extract_margin_checks(text: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for locator, paragraph in _paragraph_chunks(text):
        normalized = _normalize_numeric_text_for_patterns(" ".join(paragraph.split()))
        lowered = normalized.lower()
        if "margin" not in lowered or "profit" not in lowered or "revenue" not in lowered:
            continue
        margin_match = _percent_near_term(normalized, "margin")
        if not margin_match:
            continue
        profit = _amount_near_terms(normalized, ["gross profit", "operating profit", "net profit", "profit"])
        revenue = _amount_near_terms(normalized, ["revenue", "sales", "turnover"])
        if profit is None or revenue is None:
            continue
        stated = float(margin_match.group(1))
        if revenue == 0:
            status = "not_calculable_previous_zero"
            calculated: float | dict[str, str] = {"status": status}
            note = "margin: not calculable because revenue is zero."
        else:
            calculated = round(profit / abs(revenue) * 100, 6)
            status = classify_percentage_delta(stated, calculated)
            note = f"margin: stated {stated:.2f}%, calculated {calculated:.2f}% from profit/revenue, status {status}."
        checks.append(
            {
                "metric": "margin",
                "direction_word": "",
                "stated_pct": stated,
                "previous_value": revenue,
                "current_value": profit,
                "calculated_change_pct": calculated,
                "status": status,
                "source_text": normalized[:300],
                "evidence_locator": locator,
                "note": note,
            }
        )
    return checks


def _extract_segment_sum_checks(text: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for locator, paragraph in _paragraph_chunks(text):
        normalized = _normalize_numeric_text_for_patterns(" ".join(paragraph.split()))
        lowered = normalized.lower()
        if "segment" not in lowered or "total" not in lowered:
            continue
        amounts = _amount_mentions(normalized)
        if len(amounts) < 3:
            continue
        total_candidates = [item for item in amounts if "total" in normalized[max(0, item["start"] - 50) : item["end"] + 50].lower()]
        if not total_candidates:
            continue
        total_item = total_candidates[-1]
        component_values = [item["value"] for item in amounts if item["end"] <= total_item["start"]]
        if len(component_values) < 2:
            continue
        component_sum = round(sum(component_values), 6)
        total = total_item["value"]
        if total == 0:
            status = "not_calculable_previous_zero"
            calculated: float | dict[str, str] = {"status": status}
            note = "segment revenue: not calculable because stated total is zero."
        else:
            diff_pct = round((component_sum - total) / abs(total) * 100, 6)
            calculated = component_sum
            year_count = len(set(re.findall(r"\b20\d{2}\b", normalized)))
            if len(amounts) > 8 or year_count >= 3 or abs(diff_pct) > 1000:
                status = "needs_manual_review"
                note = (
                    f"segment revenue: possible table extraction or multi-year pairing issue; "
                    f"components sum {component_sum:.2f}, stated total {total:.2f}, difference {diff_pct:.2f}%."
                )
            else:
                status = classify_percentage_delta(0, diff_pct)
                note = f"segment revenue: components sum {component_sum:.2f}, stated total {total:.2f}, difference {diff_pct:.2f}%, status {status}."
        checks.append(
            {
                "metric": "segment revenue",
                "direction_word": "",
                "stated_pct": 0,
                "previous_value": total,
                "current_value": component_sum,
                "calculated_change_pct": calculated,
                "status": status,
                "source_text": normalized[:300],
                "evidence_locator": locator,
                "note": note,
            }
        )
    return checks


def _paragraph_chunks(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"\[MDA_P\d{3}\]", text))
    if matches:
        chunks = []
        for idx, match in enumerate(matches):
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            chunks.append((match.group(0), text[match.end() : end]))
        return chunks
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]
    return [("", sentence) for sentence in sentences]


def _locator_for_snippet(text: str, snippet: str) -> str:
    normalized_snippet = re.sub(r"\s+", " ", snippet).strip()
    if not normalized_snippet:
        return ""
    probe = normalized_snippet[:100]
    for locator, chunk in _paragraph_chunks(text):
        if probe in re.sub(r"\s+", " ", chunk).strip():
            return locator
    return ""


def _percent_near_term(text: str, term: str) -> re.Match[str] | None:
    for match in re.finditer(PERCENT_RE, text, re.IGNORECASE):
        window = text[max(0, match.start() - 80) : match.end() + 80].lower()
        if term in window:
            return match
    return None


def _amount_near_terms(text: str, terms: list[str]) -> float | None:
    lowered = text.lower()
    amounts = _amount_mentions(text)
    for term in terms:
        for term_match in re.finditer(re.escape(term), lowered):
            after = [amount for amount in amounts if amount["start"] >= term_match.end() and amount["start"] - term_match.end() <= 100]
            if after:
                return after[0]["value"]
    for amount in amounts:
        window = lowered[max(0, amount["start"] - 80) : amount["end"] + 80]
        if any(term in window for term in terms):
            return amount["value"]
    return None


def _amount_mentions(text: str) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    normalized = _normalize_numeric_text_for_patterns(text)
    for match in re.finditer(AMOUNT_RE, normalized, re.IGNORECASE):
        unit = match.group(2) or ""
        if unit == "%":
            continue
        mentions.append({"value": parse_amount(match.group(1), unit), "start": match.start(), "end": match.end()})
    return mentions


def _normalize_numeric_text_for_patterns(text: str) -> str:
    normalized = re.sub(r"\(\s*RM\s*([0-9][0-9,]*(?:\.\d+)?)\s*(million|billion)?\s*\)", r"-\1 \2", text, flags=re.IGNORECASE)
    normalized = re.sub(r"\(\s*([0-9][0-9,]*(?:\.\d+)?)\s*\)", r"-\1", normalized)
    normalized = re.sub(r"\bRM\s*'?\s*000\s+(-?\d[\d,]*(?:\.\d+)?)", r"\1 RM'000", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bRM000\s+(-?\d[\d,]*(?:\.\d+)?)", r"\1 RM000", normalized, flags=re.IGNORECASE)
    return normalized


def _count_financial_mentions(lowered: str) -> int:
    mentions = 0
    for term in METRIC_TERMS:
        mentions += len(re.findall(rf"\b{re.escape(term)}\b", lowered))
    mentions += len(re.findall(r"\bRM\s*\d", lowered))
    mentions += len(re.findall(r"\d+(?:\.\d+)?\s*%", lowered))
    return mentions


def _has_any(lowered: str, terms: list[str]) -> bool:
    return any(term in lowered for term in terms)


def _overall_consistency(counts: dict[str, int], checks: list[dict[str, Any]]) -> str:
    if not checks:
        return "not_enough_numbers"
    if counts["severe_direction_conflict"] > 0:
        return "problematic"
    if counts["material_discrepancy"] > 0:
        return "weak"
    if counts["minor_discrepancy"] > 0:
        return "moderate"
    if counts["consistent"] >= 2:
        return "strong"
    return "moderate"


def run_numeric_checks(root: Path = SCORING_ROOT, input_dir: Path | None = None, output_dir: Path | None = None) -> list[dict[str, Any]]:
    input_dir = input_dir or root / "02_extracted_text" / "mda"
    output_dir = output_dir or root / "03_numeric_checks" / "mda"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.txt")):
        result = analyze_file(path, document_id=path.stem)
        save_json(output_dir / f"{path.stem}_numeric_checks.json", result)
        summary_rows.append(_summary_row(result))
    headers = [
        "document_id",
        "num_financial_mentions",
        "num_calculable_changes",
        "num_consistent_checks",
        "num_minor_discrepancies",
        "num_material_discrepancies",
        "num_severe_direction_conflicts",
        "num_not_calculable",
        "has_revenue_discussion",
        "has_profit_discussion",
        "has_cost_discussion",
        "has_segment_discussion",
        "has_margin_discussion",
        "overall_numeric_consistency",
        "internal_check_notes",
    ]
    summary_name = f"{output_dir.name}_numeric_checks_summary.csv" if output_dir.name != "mda" else "mda_numeric_checks_summary.csv"
    summary_path = output_dir.parent / summary_name
    write_csv_rows(summary_path, headers, summary_rows)
    return summary_rows


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {key: result.get(key, "") for key in result if key != "checks"}
    if isinstance(row.get("internal_check_notes"), list):
        row["internal_check_notes"] = " | ".join(row["internal_check_notes"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic MD&A numeric checks.")
    parser.add_argument("--root", default=str(SCORING_ROOT))
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    rows = run_numeric_checks(Path(args.root), Path(args.input_dir) if args.input_dir else None, Path(args.output_dir) if args.output_dir else None)
    print(f"Wrote numeric checks for {len(rows)} MD&A documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
