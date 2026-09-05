from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from scoring_utils import dimensions_for, grade_label, standardize_score, document_total, weights_for


MDA_CLAIM_TYPES = {
    "financial_claim",
    "causal_claim",
    "cost_claim",
    "segment_claim",
    "risk_claim",
    "outlook_claim",
    "strategy_claim",
}

NEWS_CLAIM_TYPES = {
    "event_claim",
    "financial_event_claim",
    "market_claim",
    "risk_claim",
    "analyst_claim",
    "regulatory_claim",
}

CLAIM_FIELDS = [
    "claim_id",
    "document_id",
    "source_type",
    "claim_type",
    "topic",
    "claim_text",
    "value",
    "unit",
    "direction",
    "metric_name",
    "evidence_locator",
    "confidence",
]

CLAIM_SCORE_FIELDS = [
    "claim_id",
    "document_id",
    "source_type",
    "dimension",
    "score",
    "reason",
    "evidence_locator",
    "confidence",
    "model_name",
    "llm_backend",
    "parse_status",
    "schema_validation_status",
    "raw_output_path",
]

CLAIM_DOCUMENT_HEADERS = [
    "document_id",
    "source_type",
    "scoring_level",
    "dimension_count",
    "claim_count",
    "scored_claim_count",
    "total_score_100",
    "grade_label",
    "AR01",
    "AR02",
    "AR03",
    "AR04",
    "AR05",
    "N01",
    "N02",
    "N03",
    "N04",
    "N05",
    "AR01_std",
    "AR02_std",
    "AR03_std",
    "AR04_std",
    "AR05_std",
    "N01_std",
    "N02_std",
    "N03_std",
    "N04_std",
    "N05_std",
    "aggregation_method",
    "model_name",
    "llm_backend",
]

MDA_DIMENSION_RULES = {
    "financial_claim": ["AR02"],
    "causal_claim": ["AR03"],
    "cost_claim": ["AR03"],
    "segment_claim": ["AR03"],
    "risk_claim": ["AR04"],
    "outlook_claim": ["AR04"],
    "strategy_claim": ["AR03"],
}

NEWS_DIMENSION_RULES = {
    "event_claim": ["N01"],
    "financial_event_claim": ["N01"],
    "market_claim": ["N04"],
    "risk_claim": ["N01"],
    "analyst_claim": ["N02"],
    "regulatory_claim": ["N01", "N02"],
}


def extract_claims_from_text(source_type: str, document_id: str, text: str, metadata: dict[str, str] | None = None) -> list[dict[str, Any]]:
    metadata = metadata or {}
    claims: list[dict[str, Any]] = []
    for locator, body in iter_located_blocks(source_type, text):
        for sentence in split_sentences(body):
            claims.extend(_claims_from_sentence(source_type, sentence, locator, document_id, metadata))
    normalized: list[dict[str, Any]] = []
    for idx, claim in enumerate(claims, start=1):
        normalized.append(normalize_claim(source_type, document_id, claim, idx))
    return normalized


def normalize_claim(source_type: str, document_id: str, claim: dict[str, Any], index: int) -> dict[str, Any]:
    claim_type = claim.get("claim_type", "")
    allowed = MDA_CLAIM_TYPES if source_type == "mda" else NEWS_CLAIM_TYPES
    if claim_type not in allowed:
        claim_type = "financial_claim" if source_type == "mda" else "event_claim"
    return {
        "claim_id": f"{document_id}_CL{index:04d}",
        "document_id": document_id,
        "source_type": source_type,
        "claim_type": claim_type,
        "topic": claim.get("topic", "other") or "other",
        "claim_text": clean_text(claim.get("claim_text", "")),
        "value": claim.get("value", ""),
        "unit": claim.get("unit", ""),
        "direction": normalize_direction(claim.get("direction", "neutral")),
        "metric_name": claim.get("metric_name", ""),
        "evidence_locator": claim.get("evidence_locator", ""),
        "confidence": normalize_confidence(claim.get("confidence", "medium")),
    }


def iter_located_blocks(source_type: str, text: str) -> list[tuple[str, str]]:
    pattern = r"\[(MDA_P\d{3}|NEWS_P\d{3}|TITLE|LEAD)\]"
    matches = list(re.finditer(pattern, text))
    if not matches:
        default = "[MDA_P001]" if source_type == "mda" else "[NEWS_P001]"
        return [(default, text)]
    blocks: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        locator = f"[{match.group(1)}]"
        body = text[start:end].strip()
        if body:
            blocks.append((locator, body))
    return blocks


def split_sentences(text: str) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [clean_text(item) for item in candidates if clean_text(item)]


def map_claim_to_dimensions(claim: dict[str, Any]) -> list[str]:
    source_type = claim.get("source_type", "mda")
    claim_type = claim.get("claim_type", "")
    rules = MDA_DIMENSION_RULES if source_type == "mda" else NEWS_DIMENSION_RULES
    return rules.get(claim_type, ["AR03"] if source_type == "mda" else ["N04"])


def score_claim_with_rules(claim: dict[str, Any], dimension: str | None = None) -> dict[str, Any]:
    dimension = dimension or map_claim_to_dimensions(claim)[0]
    claim_type = claim.get("claim_type", "")
    text = claim.get("claim_text", "")
    score = 3
    reasons = []
    if claim.get("evidence_locator"):
        score += 1
        reasons.append("has locator")
    if claim.get("confidence") == "high":
        score += 1
        reasons.append("high confidence")
    if claim_type in {"financial_claim", "financial_event_claim"} and claim.get("value") and claim.get("direction") in {"increase", "decrease"}:
        score += 1
        reasons.append("structured financial direction")
    if claim_type in {"causal_claim", "cost_claim", "segment_claim", "market_claim", "analyst_claim"} and len(text.split()) >= 3:
        score += 1
        reasons.append("specific claim text")
    if claim_type in {"risk_claim", "outlook_claim", "regulatory_claim"}:
        score += 1
        reasons.append("risk/outlook coverage")
    if claim.get("confidence") == "low":
        score -= 1
        reasons.append("low confidence")
    return {
        "claim_id": claim["claim_id"],
        "document_id": claim["document_id"],
        "source_type": claim["source_type"],
        "dimension": dimension,
        "score": max(1, min(5, score)),
        "reason": "; ".join(reasons) or "Rule-based claim-level smoke score.",
        "evidence_locator": claim.get("evidence_locator", ""),
        "confidence": claim.get("confidence", "medium"),
    }


def aggregate_document_claim_scores(
    source_type: str,
    document_id: str,
    claims: list[dict[str, Any]],
    claim_scores: list[dict[str, Any]],
    model_name: str,
    llm_backend: str,
) -> dict[str, Any]:
    prefix = "AR" if source_type == "mda" else "N"
    dimension_codes = [item["code"] for item in dimensions_for(source_type)]
    score_by_claim = {row["claim_id"]: row for row in claim_scores}
    scores_by_dimension: dict[str, list[int]] = defaultdict(list)
    for claim in claims:
        mapped = map_claim_to_dimensions(claim)
        score_row = score_by_claim.get(claim["claim_id"])
        if not score_row:
            continue
        for dimension in mapped:
            if dimension.startswith(prefix):
                scores_by_dimension[dimension].append(int(score_row["score"]))
    raw_scores = {}
    for code in dimension_codes:
        if code in {"AR01", "AR05"}:
            raw_scores[code] = _proxy_dimension_score(code, claims)
        elif code in {"N03", "N05"}:
            raw_scores[code] = _proxy_dimension_score(code, claims)
        else:
            raw_scores[code] = _average_score(scores_by_dimension.get(code, []))
    total = document_total(raw_scores, weights_for(source_type))
    row = {
        "document_id": document_id,
        "source_type": source_type,
        "scoring_level": "claim",
        "dimension_count": len(dimension_codes),
        "claim_count": len(claims),
        "scored_claim_count": len(claim_scores),
        "total_score_100": total,
        "grade_label": grade_label(total),
        "aggregation_method": "claim_type_mapping_weighted_average",
        "model_name": model_name,
        "llm_backend": llm_backend,
    }
    for code in [f"AR{i:02d}" for i in range(1, 6)] + [f"N{i:02d}" for i in range(1, 6)]:
        row[code] = ""
        row[f"{code}_std"] = ""
    for code, raw in raw_scores.items():
        row[code] = raw
        row[f"{code}_std"] = standardize_score(raw)
    return row


def validate_claim_schema(claim: dict[str, Any]) -> list[str]:
    issues = [f"missing_{field}" for field in CLAIM_FIELDS if field not in claim]
    if claim.get("direction") not in {"increase", "decrease", "neutral"}:
        issues.append("invalid_direction")
    if claim.get("confidence") not in {"high", "medium", "low"}:
        issues.append("invalid_confidence")
    if not claim.get("evidence_locator"):
        issues.append("empty_evidence_locator")
    if not claim.get("claim_text"):
        issues.append("empty_claim_text")
    return issues


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip(" ;,")


def normalize_direction(value: Any) -> str:
    text = str(value or "").lower()
    if text in {"increase", "increased", "up", "higher", "rise", "rises", "positive", "optimistic"}:
        return "increase"
    if text in {"decrease", "decreased", "down", "lower", "decline", "declined", "negative", "cautious"}:
        return "decrease"
    return "neutral"


def normalize_confidence(value: Any) -> str:
    text = str(value or "").lower()
    return text if text in {"high", "medium", "low"} else "medium"


def _claims_from_sentence(source_type: str, sentence: str, locator: str, document_id: str, metadata: dict[str, str]) -> list[dict[str, Any]]:
    lower = sentence.lower()
    claims: list[dict[str, Any]] = []
    if source_type == "mda":
        claims.extend(_mda_claims_from_sentence(sentence, lower, locator))
    else:
        claims.extend(_news_claims_from_sentence(sentence, lower, locator, metadata))
    return claims


def _mda_claims_from_sentence(sentence: str, lower: str, locator: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    financial = _financial_claim(sentence, lower, locator, source_type="mda")
    if financial:
        claims.append(financial)
    if any(term in lower for term in ["due to", "because", "driven by", "as ", "offset by"]):
        claims.extend(_causal_claims(sentence, lower, locator))
    if any(term in lower for term in ["raw material", "cost", "packaging", "logistics"]):
        claims.append(_base_claim("cost_claim", "cost", sentence, locator, _direction_from_text(lower), "cost"))
    if any(term in lower for term in ["segment", "export", "domestic"]):
        claims.append(_base_claim("segment_claim", _topic_from_text(lower), sentence, locator, _direction_from_text(lower), "segment"))
    if any(term in lower for term in ["risk", "uncertain", "pressure", "competition", "shortage"]):
        claims.append(_base_claim("risk_claim", _topic_from_text(lower), sentence, locator, "neutral", "risk"))
    if any(term in lower for term in ["outlook", "expect", "forecast", "cautious"]):
        claims.append(_base_claim("outlook_claim", "outlook", sentence, locator, _direction_from_text(lower), "outlook"))
    if any(term in lower for term in ["strategy", "expansion", "capacity", "new product"]):
        claims.append(_base_claim("strategy_claim", "strategy", sentence, locator, "neutral", "strategy"))
    return _dedupe_claims(claims)


def _news_claims_from_sentence(sentence: str, lower: str, locator: str, metadata: dict[str, str]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    financial = _financial_claim(sentence, lower, locator, source_type="news")
    if financial:
        financial["claim_type"] = "financial_event_claim"
        claims.append(financial)
    if "analyst" in lower or "analysts said" in lower:
        claims.append(_base_claim("analyst_claim", _topic_from_text(lower), sentence, locator, _direction_from_text(lower), "analyst_view"))
    if any(term in lower for term in ["market", "industry", "demand", "export", "domestic"]):
        claims.append(_base_claim("market_claim", _topic_from_text(lower), sentence, locator, _direction_from_text(lower), "market"))
    if any(term in lower for term in ["risk", "pressure", "safety", "shortage"]):
        claims.append(_base_claim("risk_claim", _topic_from_text(lower), sentence, locator, "neutral", "risk"))
    if any(term in lower for term in ["regulator", "regulatory", "bursa"]):
        claims.append(_base_claim("regulatory_claim", "regulatory", sentence, locator, "neutral", "regulatory"))
    if not claims and sentence:
        claims.append(_base_claim("event_claim", _topic_from_text(lower), sentence, locator, _direction_from_text(lower), metadata.get("title", "")))
    return _dedupe_claims(claims)


def _financial_claim(sentence: str, lower: str, locator: str, source_type: str) -> dict[str, Any] | None:
    metric = _metric_from_text(lower)
    has_numeric = bool(re.search(r"\d", sentence))
    has_direction = _direction_from_text(lower) != "neutral"
    if not metric or not (has_numeric or has_direction):
        return None
    claim_type = "financial_claim" if source_type == "mda" else "financial_event_claim"
    claim_text = _strip_causal_tail(sentence) or sentence
    return _base_claim(claim_type, metric, claim_text, locator, _direction_from_text(lower), metric, value=_value_from_text(sentence))


def _causal_claims(sentence: str, lower: str, locator: str) -> list[dict[str, Any]]:
    parts = re.split(r"\bdue to\b|\bbecause\b|\bdriven by\b|\boffset by\b|\bas\b", sentence, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) < 2:
        return []
    cause_text = parts[1]
    fragments = [clean_text(item) for item in re.split(r"\band\b|,", cause_text) if clean_text(item)]
    claims = []
    for fragment in fragments:
        direction = _direction_from_text(fragment.lower())
        claims.append(_base_claim("causal_claim", _topic_from_text(fragment.lower()), fragment, locator, direction, "cause"))
    return claims


def _base_claim(
    claim_type: str,
    topic: str,
    claim_text: str,
    locator: str,
    direction: str,
    metric_name: str,
    *,
    value: str = "",
    unit: str = "",
) -> dict[str, Any]:
    return {
        "claim_type": claim_type,
        "topic": topic or "other",
        "claim_text": clean_text(claim_text),
        "value": value,
        "unit": unit,
        "direction": direction,
        "metric_name": metric_name,
        "evidence_locator": locator,
        "confidence": "high" if locator else "medium",
    }


def _direction_from_text(lower: str) -> str:
    if any(term in lower for term in ["increase", "increased", "higher", "rise", "rises", "improved", "stronger", "growth"]):
        return "increase"
    if any(term in lower for term in ["decrease", "decreased", "decline", "declined", "lower", "fell", "reduced", "cautious"]):
        return "decrease"
    return "neutral"


def _metric_from_text(lower: str) -> str:
    for metric in ["revenue", "profit", "sales", "margin", "cost", "cash flow"]:
        if metric in lower:
            return metric.replace(" ", "_")
    return ""


def _topic_from_text(lower: str) -> str:
    if "export" in lower:
        return "export"
    if "domestic" in lower:
        return "domestic_market"
    if "raw material" in lower or "cost" in lower:
        return "cost"
    if "revenue" in lower:
        return "revenue"
    if "profit" in lower:
        return "profit"
    if "sales" in lower:
        return "sales"
    if "outlook" in lower or "expect" in lower:
        return "outlook"
    if "risk" in lower or "pressure" in lower:
        return "risk"
    if "regulator" in lower:
        return "regulatory"
    return "other"


def _value_from_text(text: str) -> str:
    numbers = re.findall(r"(?:RM\s*)?\d+(?:,\d{3})*(?:\.\d+)?%?", text, flags=re.IGNORECASE)
    if len(numbers) >= 2:
        return f"{numbers[0]} -> {numbers[1]}"
    if numbers:
        return numbers[0]
    return ""


def _strip_causal_tail(sentence: str) -> str:
    return clean_text(re.split(r"\bdue to\b|\bbecause\b|\bdriven by\b|\boffset by\b|\bas\b", sentence, maxsplit=1, flags=re.IGNORECASE)[0])


def _dedupe_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for claim in claims:
        key = (claim.get("claim_type"), claim.get("topic"), claim.get("claim_text"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(claim)
    return deduped


def _average_score(scores: list[int]) -> int:
    if not scores:
        return 3
    return max(1, min(5, round(sum(scores) / len(scores))))


def _proxy_dimension_score(code: str, claims: list[dict[str, Any]]) -> int:
    if not claims:
        return 1
    locators = {claim.get("evidence_locator", "") for claim in claims if claim.get("evidence_locator")}
    types = {claim.get("claim_type", "") for claim in claims}
    if code in {"AR01", "N05"}:
        return max(1, min(5, 2 + len(types) // 2))
    if code in {"AR05", "N03"}:
        return max(1, min(5, 2 + len(locators)))
    return 3
