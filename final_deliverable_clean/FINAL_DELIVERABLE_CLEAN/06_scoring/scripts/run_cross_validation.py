from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from check_ollama import run_check
from llm_clients import OllamaClient, OllamaClientError
from run_scoring import _review_row
from scoring_utils import (
    DEFAULT_ENV,
    REVIEW_LOG_HEADERS,
    SCORING_ROOT,
    ensure_directories,
    read_csv_rows,
    registry_path,
    resolve_text_path,
    save_json,
    write_csv_rows,
    write_default_registries,
)


CROSS_DIR_NAME = "09_cross_validation"

CLAIM_TYPES = {
    "performance",
    "financial_result",
    "cost_pressure",
    "margin",
    "cash_flow",
    "segment_performance",
    "market_condition",
    "risk",
    "outlook",
    "strategy",
    "expansion",
    "regulation",
    "supply_chain",
    "product",
    "other",
}

TOPICS = {
    "revenue",
    "profit",
    "cost",
    "margin",
    "raw_material",
    "export",
    "domestic_market",
    "consumer_demand",
    "competition",
    "food_safety",
    "supply_chain",
    "foreign_exchange",
    "regulation",
    "halal_certification",
    "capacity_expansion",
    "new_product",
    "dividend",
    "cash_flow",
    "other",
}

MDA_CLAIM_HEADERS = [
    "claim_id",
    "document_id",
    "doc_type",
    "company_name",
    "ticker",
    "report_year",
    "claim_type",
    "topic",
    "metric_name",
    "direction",
    "value",
    "unit",
    "period",
    "claim_text",
    "evidence_locator",
    "confidence_level",
]

NEWS_CLAIM_HEADERS = [
    "claim_id",
    "document_id",
    "doc_type",
    "company_name",
    "ticker",
    "publish_date",
    "source_name",
    "title",
    "claim_type",
    "topic",
    "metric_name",
    "direction",
    "value",
    "unit",
    "period",
    "claim_text",
    "evidence_locator",
    "source_attribution",
    "source_independence",
    "confidence_level",
]

CLAIM_LINK_HEADERS = [
    "link_id",
    "mda_claim_id",
    "news_claim_id",
    "mda_document_id",
    "news_document_id",
    "ticker",
    "company_name",
    "report_year",
    "publish_date",
    "topic",
    "claim_type",
    "relation_type",
    "direction",
    "support_level",
    "source_independence",
    "time_alignment",
    "evidence_summary",
    "contradiction_flag",
    "mda_omission_flag",
    "news_overstatement_flag",
    "same_source_repetition_flag",
    "confidence_level",
    "review_required",
]

PAIR_HEADERS = [
    "mda_document_id",
    "ticker",
    "company_name",
    "report_year",
    "num_related_news",
    "num_claim_links",
    "num_strong_support",
    "num_partial_support",
    "num_contextual_support",
    "num_contradictions",
    "num_possible_omissions",
    "num_news_overstatements",
    "num_same_source_repetitions",
    "avg_source_independence_score",
    "mda_supported_by_news_score",
    "news_supported_by_mda_score",
    "cross_validation_score",
    "cross_validation_label",
    "review_required",
    "review_reasons",
]

COMPANY_YEAR_HEADERS = [
    "ticker",
    "company_name",
    "report_year",
    "mda_document_id",
    "mda_total_score_100",
    "news_avg_score_100",
    "num_news",
    "cross_validation_score",
    "mda_supported_by_news_score",
    "news_supported_by_mda_score",
    "contradiction_count",
    "possible_omission_count",
    "news_overstatement_count",
    "same_source_repetition_count",
    "independent_support_count",
    "review_required",
    "review_reasons",
]

CLAIMS_SCHEMA = {
    "type": "object",
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_type": {"type": "string"},
                    "topic": {"type": "string"},
                    "metric_name": {"type": "string"},
                    "direction": {"type": "string"},
                    "value": {"type": "string"},
                    "unit": {"type": "string"},
                    "period": {"type": "string"},
                    "claim_text": {"type": "string"},
                    "evidence_locator": {"type": "string"},
                    "source_attribution": {"type": "string"},
                    "source_independence": {"type": "string"},
                    "confidence_level": {"type": "string"},
                },
                "required": ["claim_type", "topic", "direction", "claim_text", "evidence_locator", "confidence_level"],
            },
        }
    },
}


def run_cross_validation(
    mode: str,
    *,
    root: Path = SCORING_ROOT,
    model: str = DEFAULT_ENV["OLLAMA_MODEL"],
    base_url: str = DEFAULT_ENV["OLLAMA_BASE_URL"],
    timeout_seconds: int = 180,
    limit: int | None = None,
    mock_mode: bool = False,
) -> dict[str, Any]:
    ensure_directories(root)
    write_default_registries(root)
    cross_dir = root / CROSS_DIR_NAME
    raw_dir = cross_dir / "raw_model_outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)

    mda_rows = _select_rows(read_csv_rows(registry_path("mda", root)), mode, limit)
    news_rows = _select_rows(read_csv_rows(registry_path("news", root)), mode, limit)
    review_rows: list[dict[str, Any]] = []
    notes: list[str] = []
    model_available = True
    rule_fallback_used = False

    if not mock_mode:
        if not mda_rows or not news_rows:
            notes.append("Cross-validation skipped because MD&A or news registry has no included rows.")
            model_available = True
            mda_claims: list[dict[str, Any]] = []
            news_claims: list[dict[str, Any]] = []
        else:
            check = run_check(base_url=base_url, model=model, root=root)
            model_available = bool(check.get("ollama_available") and check.get("model_available"))
            if not model_available:
                if _has_real_news(news_rows):
                    notes.append("local model unavailable; deterministic rule extraction used on real News text without fabricated claims.")
                    mda_claims = _extract_claims_for_rows("mda", mda_rows, root, raw_dir, None, review_rows, mock_mode=True)
                    news_claims = _extract_claims_for_rows("news", news_rows, root, raw_dir, None, review_rows, mock_mode=True)
                    rule_fallback_used = True
                else:
                    notes.append("local model unavailable and no real News registry rows; empty templates generated without fabricated claim links.")
                    mda_claims = []
                    news_claims = []
            else:
                client = OllamaClient(base_url=base_url, model=model, timeout=timeout_seconds)
                mda_claims = _extract_claims_for_rows("mda", mda_rows, root, raw_dir, client, review_rows, mock_mode=False)
                news_claims = _extract_claims_for_rows("news", news_rows, root, raw_dir, client, review_rows, mock_mode=False)
    else:
        mda_claims = _extract_claims_for_rows("mda", mda_rows, root, raw_dir, None, review_rows, mock_mode=True)
        news_claims = _extract_claims_for_rows("news", news_rows, root, raw_dir, None, review_rows, mock_mode=True)

    links = [] if (not mock_mode and not model_available and not rule_fallback_used) else link_claims(mda_claims, news_claims, mda_rows)
    links.extend(_validate_links_for_review(links, review_rows))
    pairs = build_pair_summary(mda_rows, news_rows, links)
    company_year = build_company_year_summary(root, mda_rows, news_rows, pairs, links)

    write_csv_rows(cross_dir / "mda_claims.csv", MDA_CLAIM_HEADERS, mda_claims)
    write_csv_rows(cross_dir / "news_claims.csv", NEWS_CLAIM_HEADERS, news_claims)
    write_csv_rows(cross_dir / "claim_links.csv", CLAIM_LINK_HEADERS, links)
    write_csv_rows(cross_dir / "mda_news_cross_validation_pairs.csv", PAIR_HEADERS, pairs)
    write_csv_rows(cross_dir / "company_year_cross_validation_summary.csv", COMPANY_YEAR_HEADERS, company_year)
    _merge_cross_review_rows(root, review_rows)
    report_path = write_report(root, mda_claims, news_claims, links, pairs, company_year, notes, model_available, mock_mode)
    return {
        "mode": mode,
        "mda_claim_count": len(mda_claims),
        "news_claim_count": len(news_claims),
        "claim_link_count": len(links),
        "pair_count": len(pairs),
        "company_year_count": len(company_year),
        "review_count": len(review_rows),
        "model_available": model_available,
        "mock_mode": mock_mode,
        "rule_fallback_used": rule_fallback_used,
        "report_path": str(report_path),
    }


def _select_rows(rows: list[dict[str, str]], mode: str, limit: int | None) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("include_flag", "").strip().lower() in {"yes", "y", "true", "1"}]
    if mode == "pilot":
        selected = selected[: limit or 5]
    elif limit is not None:
        selected = selected[:limit]
    return selected


def _has_real_news(rows: list[dict[str, str]]) -> bool:
    return any("synthetic_flag=true" not in row.get("notes", "").lower() for row in rows)


def _extract_claims_for_rows(
    doc_type: str,
    rows: list[dict[str, str]],
    root: Path,
    raw_dir: Path,
    client: OllamaClient | None,
    review_rows: list[dict[str, Any]],
    *,
    mock_mode: bool,
) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for row in rows:
        document_id = row.get("document_id", "").strip()
        text_path = resolve_text_path(row, doc_type, root)
        if not document_id or not text_path.exists():
            review_rows.append(_cross_review_row(row, "manual_required", "cross_validation_missing_text", "", ""))
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        if mock_mode:
            raw_payload = {"mock_mode": True, "claims": _mock_claim_payload(doc_type, row, text)}
            save_json(raw_dir / f"{document_id}_claims_raw.json", raw_payload)
            parsed_claims = raw_payload["claims"]
        else:
            parsed_claims = _extract_claims_with_llm(doc_type, row, text, raw_dir, client, review_rows)
        for idx, claim in enumerate(parsed_claims, start=1):
            normalized = _normalize_claim(doc_type, row, claim, idx)
            claims.append(normalized)
    return claims


def _extract_claims_with_llm(
    doc_type: str,
    row: dict[str, str],
    text: str,
    raw_dir: Path,
    client: OllamaClient | None,
    review_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assert client is not None
    document_id = row.get("document_id", "")
    prompt = _claim_prompt(doc_type, row, text)
    raw_path = raw_dir / f"{document_id}_claims_raw.json"
    try:
        result = client.chat_json(
            [
                {"role": "system", "content": "Extract claims for research cross-validation. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            CLAIMS_SCHEMA,
            think=False,
        )
        save_json(raw_path, {"raw_response": result.raw_response, "content": result.content, "parsed": result.parsed_json})
    except OllamaClientError as exc:
        save_json(raw_path, {"error": str(exc), "document_id": document_id})
        review_rows.append(_cross_review_row(row, "model_output_review", "cross_validation_model_call_failed", "", str(raw_path)))
        return []
    parsed = result.parsed_json
    if not isinstance(parsed, dict) or not isinstance(parsed.get("claims"), list):
        review_rows.append(_cross_review_row(row, "json_parse_review", "cross_validation_json_parse_failed", "", str(raw_path)))
        return []
    valid_claims = []
    for claim in parsed["claims"]:
        if isinstance(claim, dict) and claim.get("claim_text") and claim.get("evidence_locator"):
            valid_claims.append(claim)
        else:
            review_rows.append(_cross_review_row(row, "schema_validation_review", "cross_validation_schema_validation_failed", "", str(raw_path)))
    return valid_claims


def _claim_prompt(doc_type: str, row: dict[str, str], text: str) -> str:
    metadata = json.dumps(row, ensure_ascii=False)
    source_note = (
        "For news, include source_attribution and source_independence: high, medium, low, or unknown."
        if doc_type == "news"
        else "For MD&A, focus only on the MD&A narrative and do not audit external financial statements."
    )
    return (
        f"Document type: {doc_type}\nMetadata: {metadata}\n{source_note}\n"
        f"Allowed claim_type values: {sorted(CLAIM_TYPES)}\nAllowed topic values: {sorted(TOPICS)}\n"
        "Extract concise claims relevant to MD&A/news triangulation. Use paragraph IDs or sentence positions as evidence_locator.\n"
        "Return JSON object with key claims only.\n\nTEXT:\n"
        + text[:12000]
    )


def _mock_claim_payload(doc_type: str, row: dict[str, str], text: str) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []
    for idx, sentence in enumerate(_sentences(text), start=1):
        lower = sentence.lower()
        locator = _locator(sentence, idx)
        base = {
            "value": "",
            "unit": "",
            "period": row.get("report_year", "") if doc_type == "mda" else _year_from_date(row.get("publish_date", "")),
            "claim_text": sentence.strip(),
            "evidence_locator": locator,
            "confidence_level": "high",
        }
        if "annual report" in lower or "bursa" in lower or "company announcement" in lower or "company said" in lower:
            source_attribution = "company_or_filing"
            source_independence = "low"
        elif "regulator" in lower or "industry data" in lower or "market report" in lower or "analyst" in lower:
            source_attribution = "third_party"
            source_independence = "high" if "regulator" in lower or "industry data" in lower else "medium"
        elif "sources said" in lower:
            source_attribution = "unnamed_sources"
            source_independence = "unknown"
        else:
            source_attribution = ""
            source_independence = "unknown" if doc_type == "news" else ""

        if "raw material" in lower or "packaging" in lower or "logistics" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "cost_pressure",
                    "topic": "raw_material",
                    "metric_name": "raw_material_cost",
                    "direction": _direction(lower, default="increased"),
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
        elif "revenue" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "financial_result",
                    "topic": "revenue",
                    "metric_name": "revenue",
                    "direction": _direction(lower),
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
        elif "profit" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "financial_result",
                    "topic": "profit",
                    "metric_name": "profit",
                    "direction": _direction(lower),
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
        elif "margin" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "margin",
                    "topic": "margin",
                    "metric_name": "margin",
                    "direction": _direction(lower, default="decreased" if "pressure" in lower else "unknown"),
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
        elif "cautious outlook" in lower or "caution" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "outlook",
                    "topic": "consumer_demand",
                    "metric_name": "outlook",
                    "direction": "cautious",
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
        elif "explosive growth" in lower or "strong growth" in lower or "optimistic" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "outlook",
                    "topic": "consumer_demand",
                    "metric_name": "outlook",
                    "direction": "optimistic",
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
        elif "capacity" in lower or "production line" in lower or "expanded" in lower or "expansion" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "expansion",
                    "topic": "capacity_expansion",
                    "metric_name": "capacity",
                    "direction": "increased",
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
        elif "food safety" in lower:
            claims.append(
                {
                    **base,
                    "claim_type": "risk",
                    "topic": "food_safety",
                    "metric_name": "food_safety_issue",
                    "direction": "negative",
                    "source_attribution": source_attribution,
                    "source_independence": source_independence,
                }
            )
    return claims


def _normalize_claim(doc_type: str, row: dict[str, str], claim: dict[str, Any], index: int) -> dict[str, Any]:
    document_id = row.get("document_id", "")
    claim_type = claim.get("claim_type", "other") if claim.get("claim_type") in CLAIM_TYPES else "other"
    topic = claim.get("topic", "other") if claim.get("topic") in TOPICS else "other"
    normalized = {
        "claim_id": f"{document_id}_C{index:03d}",
        "document_id": document_id,
        "doc_type": doc_type,
        "company_name": row.get("company_name", ""),
        "ticker": row.get("ticker", ""),
        "claim_type": claim_type,
        "topic": topic,
        "metric_name": claim.get("metric_name", ""),
        "direction": claim.get("direction", "unknown"),
        "value": claim.get("value", ""),
        "unit": claim.get("unit", ""),
        "period": claim.get("period", ""),
        "claim_text": claim.get("claim_text", ""),
        "evidence_locator": claim.get("evidence_locator", ""),
        "confidence_level": claim.get("confidence_level", "medium"),
    }
    if doc_type == "mda":
        normalized["report_year"] = row.get("report_year", "")
    else:
        normalized.update(
            {
                "publish_date": row.get("publish_date", ""),
                "source_name": row.get("source_name", ""),
                "title": row.get("title", ""),
                "source_attribution": claim.get("source_attribution", ""),
                "source_independence": _normalize_source_independence(claim.get("source_independence", "unknown")),
            }
        )
    return normalized


def link_claims(
    mda_claims: list[dict[str, Any]],
    news_claims: list[dict[str, Any]],
    mda_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    mda_by_doc = {row.get("document_id", ""): row for row in mda_rows}
    claims_by_doc = defaultdict(list)
    for claim in mda_claims:
        claims_by_doc[claim["document_id"]].append(claim)
    links: list[dict[str, Any]] = []
    link_index = 1
    for news_claim in news_claims:
        mda_doc = _best_mda_document_for_news(news_claim, mda_rows)
        if not mda_doc:
            continue
        candidates = [
            claim
            for claim in claims_by_doc.get(mda_doc.get("document_id", ""), [])
            if _topic_close(claim.get("topic", ""), news_claim.get("topic", ""))
        ]
        if not candidates:
            candidates = [
                claim
                for claim in claims_by_doc.get(mda_doc.get("document_id", ""), [])
                if _claim_type_close(claim.get("claim_type", ""), news_claim.get("claim_type", ""))
            ]
        best_mda_claim = _choose_best_mda_claim(news_claim, candidates)
        link = classify_relation(link_index, best_mda_claim, news_claim, mda_doc)
        if link:
            links.append(link)
            link_index += 1
    for link in links:
        if link["mda_document_id"] not in mda_by_doc:
            link["review_required"] = "true"
    return links


def classify_relation(
    link_index: int,
    mda_claim: dict[str, Any] | None,
    news_claim: dict[str, Any],
    mda_doc: dict[str, str],
) -> dict[str, Any] | None:
    topic = news_claim.get("topic", "other")
    claim_type = news_claim.get("claim_type", "other")
    source_independence = _normalize_source_independence(news_claim.get("source_independence", "unknown"))
    time_alignment = _time_alignment(mda_doc.get("report_year", ""), news_claim.get("publish_date", ""))
    relation_type = "no_clear_relation"
    direction = "no_direction"
    support_level = "none"
    contradiction = 0
    omission = 0
    overstatement = 0
    repetition = 0
    confidence = "medium"

    if mda_claim is None:
        if claim_type in {"risk", "regulation"} or topic in {"food_safety", "regulation"}:
            relation_type = "possible_omission"
            direction = "news_contextualizes_mda"
            support_level = "unknown"
            omission = 1
            confidence = "high"
        else:
            relation_type = "insufficient_evidence"
            support_level = "unknown"
    else:
        same_source = source_independence == "low" and _company_source(news_claim)
        if same_source:
            relation_type = "same_source_repetition"
            direction = "bidirectional_support"
            support_level = "moderate"
            repetition = 1
            confidence = "high"
        elif mda_claim.get("direction") == "cautious" and news_claim.get("direction") == "optimistic":
            relation_type = "qualification"
            direction = "mda_qualifies_news"
            support_level = "weak"
            overstatement = 1
            confidence = "high"
        elif _directions_conflict(mda_claim.get("direction", ""), news_claim.get("direction", "")):
            relation_type = "contradiction"
            direction = "contradiction"
            support_level = "conflicting"
            contradiction = 1
            confidence = "high"
        elif claim_type == "expansion" and mda_claim.get("claim_type") == "expansion":
            relation_type = "strong_support"
            direction = "mda_supports_news"
            support_level = "strong"
            confidence = "high"
        elif topic in {"raw_material", "supply_chain", "foreign_exchange", "consumer_demand", "competition"}:
            relation_type = "contextual_support"
            direction = "news_supports_mda"
            support_level = "moderate"
        elif _topic_close(mda_claim.get("topic", ""), topic) and mda_claim.get("direction") == news_claim.get("direction"):
            relation_type = "strong_support" if source_independence in {"medium", "high"} else "partial_support"
            direction = "bidirectional_support"
            support_level = "strong" if relation_type == "strong_support" else "moderate"
        elif _topic_close(mda_claim.get("topic", ""), topic):
            relation_type = "partial_support"
            direction = "bidirectional_support"
            support_level = "moderate"
        else:
            relation_type = "no_clear_relation"
            support_level = "none"

    if source_independence == "unknown" and support_level == "strong":
        support_level = "moderate"
    if time_alignment == "outside_window" and support_level == "strong":
        support_level = "moderate"
        relation_type = "partial_support"
    review_required = any([contradiction, omission, overstatement, repetition]) or relation_type in {"insufficient_evidence"}
    evidence_summary = _evidence_summary(mda_claim, news_claim)
    return {
        "link_id": f"CVL_{link_index:05d}",
        "mda_claim_id": mda_claim.get("claim_id", "") if mda_claim else "",
        "news_claim_id": news_claim.get("claim_id", ""),
        "mda_document_id": mda_doc.get("document_id", ""),
        "news_document_id": news_claim.get("document_id", ""),
        "ticker": news_claim.get("ticker") or mda_doc.get("ticker", ""),
        "company_name": news_claim.get("company_name") or mda_doc.get("company_name", ""),
        "report_year": mda_doc.get("report_year", ""),
        "publish_date": news_claim.get("publish_date", ""),
        "topic": topic,
        "claim_type": claim_type,
        "relation_type": relation_type,
        "direction": direction,
        "support_level": support_level,
        "source_independence": source_independence,
        "time_alignment": time_alignment,
        "evidence_summary": evidence_summary,
        "contradiction_flag": contradiction,
        "mda_omission_flag": omission,
        "news_overstatement_flag": overstatement,
        "same_source_repetition_flag": repetition,
        "confidence_level": confidence,
        "review_required": "true" if review_required else "false",
    }


def build_pair_summary(
    mda_rows: list[dict[str, str]],
    news_rows: list[dict[str, str]],
    links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    news_by_id = {row.get("document_id", ""): row for row in news_rows}
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        by_doc[link.get("mda_document_id", "")].append(link)
    rows = []
    for mda in mda_rows:
        doc_id = mda.get("document_id", "")
        doc_links = by_doc.get(doc_id, [])
        related_news = {link.get("news_document_id", "") for link in doc_links if link.get("news_document_id") in news_by_id}
        counts = _relation_counts(doc_links)
        avg_independence = _avg_independence(doc_links)
        mda_supported = _direction_score(doc_links, "news_supports_mda")
        news_supported = _direction_score(doc_links, "mda_supports_news")
        cv_score, label, reasons = _cross_validation_score(counts, avg_independence, mda_supported, news_supported)
        rows.append(
            {
                "mda_document_id": doc_id,
                "ticker": mda.get("ticker", ""),
                "company_name": mda.get("company_name", ""),
                "report_year": mda.get("report_year", ""),
                "num_related_news": len(related_news),
                "num_claim_links": len(doc_links),
                "num_strong_support": counts["strong_support"],
                "num_partial_support": counts["partial_support"],
                "num_contextual_support": counts["contextual_support"],
                "num_contradictions": counts["contradiction"],
                "num_possible_omissions": counts["possible_omission"],
                "num_news_overstatements": counts["news_overstatement"],
                "num_same_source_repetitions": counts["same_source_repetition"],
                "avg_source_independence_score": round(avg_independence, 3),
                "mda_supported_by_news_score": mda_supported,
                "news_supported_by_mda_score": news_supported,
                "cross_validation_score": cv_score,
                "cross_validation_label": label,
                "review_required": "true" if reasons else "false",
                "review_reasons": ";".join(reasons),
            }
        )
    return rows


def build_company_year_summary(
    root: Path,
    mda_rows: list[dict[str, str]],
    news_rows: list[dict[str, str]],
    pair_rows: list[dict[str, Any]],
    links: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mda_scores = {row.get("document_id", ""): row for row in read_csv_rows(root / "06_ratings" / "mda_document_scores.csv")}
    news_scores = {row.get("document_id", ""): row for row in read_csv_rows(root / "06_ratings" / "news_document_scores.csv")}
    news_by_ticker = defaultdict(list)
    for row in news_rows:
        news_by_ticker[(row.get("ticker", ""), _year_from_date(row.get("publish_date", "")))].append(row)
    links_by_doc = defaultdict(list)
    for link in links:
        links_by_doc[link.get("mda_document_id", "")].append(link)
    pair_by_doc = {row.get("mda_document_id", ""): row for row in pair_rows}
    rows = []
    for mda in mda_rows:
        doc_id = mda.get("document_id", "")
        year = mda.get("report_year", "")
        ticker = mda.get("ticker", "")
        related_news = news_by_ticker.get((ticker, year), [])
        news_avg = _avg_score([news_scores.get(row.get("document_id", ""), {}).get("total_score_100", "") for row in related_news])
        doc_links = links_by_doc.get(doc_id, [])
        pair = pair_by_doc.get(doc_id, {})
        rows.append(
            {
                "ticker": ticker,
                "company_name": mda.get("company_name", ""),
                "report_year": year,
                "mda_document_id": doc_id,
                "mda_total_score_100": mda_scores.get(doc_id, {}).get("total_score_100", ""),
                "news_avg_score_100": news_avg,
                "num_news": len(related_news),
                "cross_validation_score": pair.get("cross_validation_score", ""),
                "mda_supported_by_news_score": pair.get("mda_supported_by_news_score", ""),
                "news_supported_by_mda_score": pair.get("news_supported_by_mda_score", ""),
                "contradiction_count": sum(int(link.get("contradiction_flag", 0)) for link in doc_links),
                "possible_omission_count": sum(int(link.get("mda_omission_flag", 0)) for link in doc_links),
                "news_overstatement_count": sum(int(link.get("news_overstatement_flag", 0)) for link in doc_links),
                "same_source_repetition_count": sum(int(link.get("same_source_repetition_flag", 0)) for link in doc_links),
                "independent_support_count": sum(
                    1
                    for link in doc_links
                    if link.get("relation_type") in {"strong_support", "contextual_support", "partial_support"}
                    and link.get("source_independence") in {"medium", "high"}
                ),
                "review_required": pair.get("review_required", "false"),
                "review_reasons": pair.get("review_reasons", ""),
            }
        )
    return rows


def write_report(
    root: Path,
    mda_claims: list[dict[str, Any]],
    news_claims: list[dict[str, Any]],
    links: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    company_year: list[dict[str, Any]],
    notes: list[str],
    model_available: bool,
    mock_mode: bool,
) -> Path:
    counts = _relation_counts(links)
    lines = [
        "# MD&A News Cross-Validation Report",
        "",
        "## Purpose",
        "This module is a post-scoring validation layer. It does not match documents mechanically, average MD&A and news scores, or overwrite original scoring outputs. It evaluates support, qualification, contradiction, omission, overstatement, and repetition between MD&A and news claims.",
        "",
        "## Method",
        "claim extraction -> claim linking -> relation classification -> pair summary -> company-year summary.",
        "",
        "## Bidirectional Logic",
        "- MD&A -> news: whether news supports, contextualizes, qualifies, or challenges management explanations.",
        "- News -> MD&A: whether MD&A supports, limits, or contradicts news claims.",
        "",
        "## Run Status",
        f"- local model available: {model_available}",
        f"- mock mode: {mock_mode}",
        f"- cross_validation_pipeline_ready: {len(mda_claims) > 0 and len(news_claims) > 0}",
        f"- cross_validation_empirical_ready: {not mock_mode and len(links) > 0}",
        f"- mda claims: {len(mda_claims)}",
        f"- news claims: {len(news_claims)}",
        f"- claim links: {len(links)}",
        f"- pair summaries: {len(pairs)}",
        f"- company-year summaries: {len(company_year)}",
        "",
        "## Main Findings",
        f"- strong support cases: {counts['strong_support']}",
        f"- contextual support cases: {counts['contextual_support']}",
        f"- contradictions: {counts['contradiction']}",
        f"- possible omissions: {counts['possible_omission']}",
        f"- possible news overstatements: {counts['news_overstatement']}",
        f"- same-source repetitions: {counts['same_source_repetition']}",
        "",
    ]
    if notes:
        lines.append("## Notes")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    if not model_available and not mock_mode:
        lines.extend(["## Local Model Unavailable", "local model unavailable; empty templates generated and no claim links were fabricated.", ""])
    if mock_mode:
        lines.extend(
            [
                "## Empirical Readiness",
                "cross_validation_pipeline_ready=true because claim extraction, linking, and reporting files were produced.",
                "cross_validation_empirical_ready=false because claim scoring used deterministic_claim_proxy / mock_mode=true rather than qwen3:8b or a validated human rule pass.",
                "News scoring enters the document-level measurement model, but claim-level MD&A-News triangulation remains non-interpretable because no real claim scoring pass was produced.",
                "",
            ]
        )
    lines.extend(
        [
            "## Limitations",
            "- News and MD&A are not absolute truth sources.",
            "- News may rewrite company announcements or annual reports.",
            "- MD&A may reflect management bias.",
            "- Cross-validation is a review indicator, not a substitute for human judgment.",
            "",
        ]
    )
    path = root / CROSS_DIR_NAME / "cross_validation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _validate_links_for_review(links: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for link in links:
        reasons = []
        if int(link.get("contradiction_flag", 0)):
            reasons.append("cross_validation_contradiction")
        if int(link.get("mda_omission_flag", 0)):
            reasons.append("cross_validation_possible_omission")
        if int(link.get("news_overstatement_flag", 0)):
            reasons.append("cross_validation_news_overstatement")
        if int(link.get("same_source_repetition_flag", 0)) and link.get("support_level") == "strong":
            reasons.append("cross_validation_same_source_marked_strong")
        if link.get("source_independence") == "unknown" and link.get("support_level") == "strong":
            reasons.append("cross_validation_unknown_source_marked_strong")
        if link.get("relation_type") == "strong_support" and not link.get("evidence_summary"):
            reasons.append("cross_validation_strong_support_missing_evidence")
        if not link.get("mda_claim_id") and not int(link.get("mda_omission_flag", 0)):
            reasons.append("cross_validation_missing_mda_evidence_locator")
        if not link.get("news_claim_id"):
            reasons.append("cross_validation_missing_news_evidence_locator")
        if link.get("time_alignment") == "outside_window" and link.get("support_level") == "strong":
            reasons.append("cross_validation_outside_window_marked_strong")
        if reasons:
            link["review_required"] = "true"
            for reason in reasons:
                review_rows.append(
                    _review_row(
                        link.get("mda_document_id", "") or link.get("news_document_id", ""),
                        "cross_validation",
                        "cross_validation_review",
                        reason,
                        "",
                        "",
                        link.get("mda_claim_id") or link.get("news_claim_id", ""),
                        "",
                        "",
                    )
                )
    return []


def _merge_cross_review_rows(root: Path, new_rows: list[dict[str, Any]]) -> None:
    path = root / "07_review" / "review_log.csv"
    existing = [row for row in read_csv_rows(path) if row.get("doc_type") != "cross_validation"]
    write_csv_rows(path, REVIEW_LOG_HEADERS, existing + new_rows)


def _cross_review_row(
    row: dict[str, str],
    review_type: str,
    reason: str,
    evidence_locator: str,
    raw_output_path: str,
) -> dict[str, Any]:
    return _review_row(
        row.get("document_id", ""),
        "cross_validation",
        review_type,
        reason,
        "",
        "",
        evidence_locator,
        "",
        raw_output_path,
    )


def _relation_counts(links: list[dict[str, Any]]) -> dict[str, int]:
    counts = defaultdict(int)
    for link in links:
        counts[link.get("relation_type", "")] += 1
        if int(link.get("news_overstatement_flag", 0)):
            counts["news_overstatement"] += 1
    return counts


def _cross_validation_score(
    counts: dict[str, int],
    avg_independence: float,
    mda_supported: int,
    news_supported: int,
) -> tuple[int, str, list[str]]:
    reasons = []
    if counts["contradiction"] or counts["possible_omission"]:
        if counts["contradiction"]:
            reasons.append("contradiction")
        if counts["possible_omission"]:
            reasons.append("possible_omission")
        return 1, "D = contradiction / omission risk", reasons
    if counts["news_overstatement"]:
        return 2, "D = contradiction / omission risk", ["news_overstatement"]
    support_total = counts["strong_support"] + counts["contextual_support"] + counts["partial_support"]
    if support_total >= 3 and avg_independence >= 0.6 and min(mda_supported, news_supported) >= 3:
        return 5, "A = strong triangulation", []
    if support_total >= 2:
        return 4, "B = moderate triangulation", []
    if support_total >= 1:
        return 3, "C = weak or neutral triangulation", []
    return 3, "C = weak or neutral triangulation", []


def _direction_score(links: list[dict[str, Any]], direction_key: str) -> int:
    score = 1
    for link in links:
        direction = link.get("direction", "")
        relation = link.get("relation_type", "")
        if direction_key == "news_supports_mda" and direction in {"news_supports_mda", "bidirectional_support"}:
            score = max(score, 4 if relation in {"strong_support", "contextual_support"} else 3)
        if direction_key == "mda_supports_news" and direction in {"mda_supports_news", "bidirectional_support"}:
            score = max(score, 4 if relation == "strong_support" else 3)
    return score


def _avg_independence(links: list[dict[str, Any]]) -> float:
    if not links:
        return 0.0
    values = {"high": 1.0, "medium": 0.67, "low": 0.33, "unknown": 0.0}
    return sum(values.get(link.get("source_independence", "unknown"), 0.0) for link in links) / len(links)


def _avg_score(values: list[str]) -> str:
    nums = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return "" if not nums else str(round(sum(nums) / len(nums), 3))


def _best_mda_document_for_news(news_claim: dict[str, Any], mda_rows: list[dict[str, str]]) -> dict[str, str] | None:
    ticker = news_claim.get("ticker", "")
    company = news_claim.get("company_name", "").lower()
    publish_year = _year_from_date(news_claim.get("publish_date", ""))
    candidates = []
    for row in mda_rows:
        same_company = (ticker and row.get("ticker") == ticker) or (company and row.get("company_name", "").lower() == company)
        if not same_company:
            continue
        alignment = _time_alignment(row.get("report_year", ""), news_claim.get("publish_date", ""))
        score = 2 if alignment in {"same_period", "post_mda"} else 1
        if publish_year and row.get("report_year") == publish_year:
            score += 1
        candidates.append((score, row))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0], reverse=True)[0][1]


def _choose_best_mda_claim(news_claim: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    ranked = []
    for claim in candidates:
        score = 0
        if claim.get("topic") == news_claim.get("topic"):
            score += 3
        if claim.get("claim_type") == news_claim.get("claim_type"):
            score += 2
        if claim.get("metric_name") and claim.get("metric_name") == news_claim.get("metric_name"):
            score += 2
        if _directions_conflict(claim.get("direction", ""), news_claim.get("direction", "")):
            score += 4
        if claim.get("direction") == "cautious" and news_claim.get("direction") == "optimistic":
            score += 4
        ranked.append((score, claim))
    return sorted(ranked, key=lambda item: item[0], reverse=True)[0][1]


def _topic_close(left: str, right: str) -> bool:
    if left == right:
        return True
    groups = [
        {"raw_material", "cost", "margin", "supply_chain"},
        {"consumer_demand", "domestic_market", "competition"},
        {"capacity_expansion", "new_product", "strategy"},
        {"food_safety", "regulation"},
    ]
    return any(left in group and right in group for group in groups)


def _claim_type_close(left: str, right: str) -> bool:
    if left == right:
        return True
    return {left, right} <= {"performance", "financial_result", "margin", "cost_pressure", "market_condition"}


def _directions_conflict(left: str, right: str) -> bool:
    left_norm = left.lower()
    right_norm = right.lower()
    positive = {"increased", "increase", "positive", "improved", "optimistic"}
    negative = {"decreased", "decrease", "negative", "declined", "cautious"}
    return (left_norm in positive and right_norm in negative) or (left_norm in negative and right_norm in positive)


def _direction(text: str, default: str = "unknown") -> str:
    positive_match = re.search(r"\b(increased|increase|improved|higher|rose|growth|expanded)\b", text)
    negative_match = re.search(r"\b(decreased|decrease|declined|lower|fell|drop|reduced)\b", text)
    if positive_match and negative_match:
        return "increased" if positive_match.start() < negative_match.start() else "decreased"
    if positive_match:
        return "increased"
    if negative_match:
        return "decreased"
    return default


def _company_source(news_claim: dict[str, Any]) -> bool:
    text = " ".join(
        [
            news_claim.get("source_attribution", ""),
            news_claim.get("claim_text", ""),
            news_claim.get("title", ""),
        ]
    ).lower()
    return any(token in text for token in ["annual report", "bursa", "company announcement", "company_or_filing", "company said"])


def _time_alignment(report_year: str, publish_date: str) -> str:
    if not report_year or not publish_date:
        return "unknown"
    try:
        year = int(report_year)
        published = date.fromisoformat(publish_date[:10])
    except ValueError:
        return "unknown"
    start = date(year, 1, 1)
    end = date(year + 1, 6, 30)
    if published < start:
        return "pre_mda"
    if start <= published <= date(year, 12, 31):
        return "same_period"
    if date(year + 1, 1, 1) <= published <= end:
        return "post_mda"
    return "outside_window"


def _evidence_summary(mda_claim: dict[str, Any] | None, news_claim: dict[str, Any]) -> str:
    news_text = news_claim.get("claim_text", "")
    if not mda_claim:
        return f"News claim: {news_text[:220]}"
    return f"MD&A: {mda_claim.get('claim_text', '')[:160]} | News: {news_text[:160]}"


def _normalize_source_independence(value: Any) -> str:
    normalized = str(value or "unknown").strip().lower()
    return normalized if normalized in {"high", "medium", "low", "unknown"} else "unknown"


def _sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _locator(sentence: str, idx: int) -> str:
    match = re.search(r"\[(MDA_P\d+)\]", sentence)
    if match:
        return match.group(1)
    return f"S{idx:03d}"


def _year_from_date(value: str) -> str:
    match = re.match(r"(\d{4})", value or "")
    return match.group(1) if match else ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MD&A and news bidirectional cross-validation.")
    parser.add_argument("--mode", choices=["pilot", "full"], required=True)
    parser.add_argument("--model", default=DEFAULT_ENV["OLLAMA_MODEL"])
    parser.add_argument("--base-url", default=DEFAULT_ENV["OLLAMA_BASE_URL"])
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    summary = run_cross_validation(
        args.mode,
        model=args.model,
        base_url=args.base_url,
        timeout_seconds=args.timeout_seconds,
        limit=args.limit,
        mock_mode=args.mock,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
