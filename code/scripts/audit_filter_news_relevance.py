from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from filter_news_common import (
    RELEVANCE_HEADERS,
    SCORING_ROOT,
    has_business_context,
    has_company_event_context,
    is_generic_market_title,
    likely_search_result,
    normalize_space,
    raw_article_hash,
    raw_row_key,
    short_hash,
    write_company_alias_map,
    write_markdown_report,
)
from scoring_utils import read_csv_rows, write_csv_rows


FILTER_VERSION = "manual_preserve_v1"

COMPANY_FOCUS_ALIAS_TYPES = {"legal_name", "brand_name", "common_name", "short_name"}

COMPANY_FOCUS_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "title",
    "source_name",
    "publish_date",
    "url",
    "relevance_status",
    "confidence_tier",
    "main_model_include_flag",
    "company_focus_status",
    "suspect_not_company_focused",
    "company_focus_reason",
    "strict_main_sample_include",
    "title_focus_alias_hits",
    "lead_focus_alias_hits",
    "filter_mode",
    "filter_version",
]


def load_aliases(root: Path = SCORING_ROOT) -> dict[str, list[dict[str, str]]]:
    write_company_alias_map(root)
    aliases: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv_rows(root / "01_registry" / "company_alias_map.csv"):
        ticker = normalize_space(row.get("ticker")).upper()
        if ticker:
            aliases[ticker].append(row)
    return aliases


def audit_filter_news_relevance(root: Path = SCORING_ROOT, mode: str = "manual_preserve") -> dict[str, Any]:
    if mode not in {"strict", "manual_preserve"}:
        raise ValueError("mode must be strict or manual_preserve")
    raw_path = root / "input_news" / "news_raw_from_filter_news.csv"
    output_path = root / "08_reports" / "filter_news_relevance_audit.csv"
    previous_audit_rows = read_csv_rows(output_path)
    aliases = load_aliases(root)
    strong_alias_rows = flatten_strong_aliases(aliases)
    raw_rows = read_csv_rows(raw_path)
    body_tickers = article_hash_tickers(raw_rows)

    strict_rows = classify_rows(raw_rows, aliases, strong_alias_rows, body_tickers, mode="strict")
    manual_rows = classify_rows(raw_rows, aliases, strong_alias_rows, body_tickers, mode="manual_preserve")
    audit_rows = manual_rows if mode == "manual_preserve" else strict_rows
    annotate_sample_flags(root, audit_rows)

    write_csv_rows(output_path, RELEVANCE_HEADERS, audit_rows)
    write_auxiliary_reports(root, raw_rows, previous_audit_rows, strict_rows, manual_rows, audit_rows, mode, aliases)
    summary = summarize_rows(audit_rows, raw_rows, mode)
    write_main_report(root, summary, strict_rows, manual_rows)
    return summary


def article_hash_tickers(raw_rows: list[dict[str, str]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in raw_rows:
        ticker = normalize_space(row.get("ticker")).upper()
        title = normalize_space(row.get("title"))
        text = normalize_space(row.get("article_text"))
        if ticker and text:
            mapping[raw_article_hash(title, text)].add(ticker)
    return mapping


def classify_rows(
    raw_rows: list[dict[str, str]],
    aliases: dict[str, list[dict[str, str]]],
    strong_alias_rows: list[dict[str, str]],
    body_tickers: dict[str, set[str]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_global: set[str] = set()
    seen_by_ticker: dict[str, set[str]] = defaultdict(set)
    for idx, row in enumerate(raw_rows, start=1):
        classified = classify_row(
            row,
            idx,
            aliases,
            strong_alias_rows,
            body_tickers,
            seen_global,
            seen_by_ticker,
            mode=mode,
        )
        rows.append(classified)
    return rows


def classify_row(
    row: dict[str, str],
    index: int,
    aliases: dict[str, list[dict[str, str]]],
    strong_alias_rows: list[dict[str, str]],
    body_tickers: dict[str, set[str]],
    seen_global: set[str],
    seen_by_ticker: dict[str, set[str]],
    *,
    mode: str,
) -> dict[str, Any]:
    ticker = normalize_space(row.get("ticker")).upper()
    company_name = normalize_space(row.get("company_name"))
    title = normalize_space(row.get("title"))
    source_name = normalize_space(row.get("source_name"))
    publish_date = normalize_space(row.get("publish_date"))
    url = normalize_space(row.get("url"))
    text = normalize_space(row.get("article_text"))
    text_length = len(text)
    row_key = raw_row_key(row, index)
    article_hash = raw_article_hash(title, text)
    document_id = f"NEWS_CANDIDATE_{index:06d}_{short_hash(row_key, 8)}"
    combined = f"{title}\n{text}"
    lead_text = first_two_paragraphs(row.get("article_text", ""))

    hits = alias_hits(combined, ticker, aliases)
    title_hits = alias_hits(title, ticker, aliases)
    lead_hits = alias_hits(lead_text, ticker, aliases)
    strong = [hit for hit in hits if hit["match_strength"] == "strong"]
    medium = [hit for hit in hits if hit["match_strength"] == "medium"]
    weak = [hit for hit in hits if hit["match_strength"] == "weak"]
    strong_title = [hit for hit in title_hits if hit["match_strength"] == "strong"]
    medium_title = [hit for hit in title_hits if hit["match_strength"] == "medium"]
    weak_title = [hit for hit in title_hits if hit["match_strength"] == "weak"]
    focus_title_hits = company_focus_alias_hits(title, ticker, aliases)
    focus_lead_hits = company_focus_alias_hits(lead_text, ticker, aliases)
    other_strong = other_strong_alias_hits(combined, ticker, strong_alias_rows)
    target_hit = bool(strong or medium or weak)
    multi_company = bool(target_hit and other_strong)
    shared_article = len(body_tickers.get(article_hash, set())) > 1
    focus_context_text = f"{title}\n{lead_text}"
    substantive = has_company_focus_context(focus_context_text) or has_substantive_company_topic(focus_context_text)
    soft_story = is_soft_csr_lifestyle_story(combined)
    venue_story = is_venue_not_company_story(ticker, combined)
    known_wrong_company = is_known_wrong_company_story(ticker, combined)
    focus_blocker = disqualifying_company_focus_reason(ticker, title, lead_text, text)
    title_company_focus = bool(focus_title_hits) and substantive and not focus_blocker and not soft_story
    lead_company_focus = bool(focus_lead_hits) and has_company_focus_context(focus_context_text) and not focus_blocker and not soft_story

    status = "no_target_alias_industry_news"
    reason = ""
    include = "No"
    main_include = "No"
    high_include = "No"
    confidence = "none"
    needs_review = "No"
    review_reason = ""
    exclude_reason = ""

    if not text and url:
        status, reason, exclude_reason = "unusable_url_only", "URL exists but no article body text was supplied.", "url_only"
    elif not text:
        status, reason, exclude_reason = "unusable_missing_text", "No article body text was supplied.", "missing_text"
    elif text.lower().startswith("http") and text_length < 200:
        status, reason, exclude_reason = "unusable_url_only", "Article text field contains only a URL-like value.", "url_only"
    elif not (title or source_name or publish_date):
        status, reason, exclude_reason = "unusable_metadata_only", "Title, source name, and publish date are all missing.", "missing_required_metadata"
    elif likely_search_result(combined):
        status, reason, exclude_reason = "unusable_search_result", "Text looks like a search-result or listing page.", "search_result_only"
    elif text_length < 200:
        status, reason, exclude_reason = "unusable_too_short", "Article body text is below 200 characters.", "too_short"
    else:
        duplicate = article_hash in seen_global if mode == "strict" else article_hash in seen_by_ticker[ticker]
        if duplicate:
            status, reason, exclude_reason = "duplicate", "Duplicate article body within the applicable duplicate scope.", "duplicate_body"
        elif known_wrong_company:
            status, reason, exclude_reason = "wrong_company", "Known non-target company/entity appears for this ticker.", "wrong_company"
            needs_review = "Yes"
            review_reason = "wrong_company"
        elif venue_story:
            status = "no_target_alias_industry_news"
            reason = "Target words refer to a venue/event name rather than the listed company."
            include = "Manual"
            needs_review = "Yes"
            review_reason = "venue_not_company_news"
            exclude_reason = "not_company_news"
        elif other_strong and not target_hit:
            status, reason, exclude_reason = "wrong_company", "Another target company's strong alias appears, but the row ticker's target alias does not.", "wrong_company"
            needs_review = "Yes"
            review_reason = "wrong_company"
        else:
            generic_title = is_generic_market_title(title)
            weak_title_context = bool(weak_title) and (has_company_event_context(focus_context_text) or has_business_context(focus_context_text))
            if generic_title and not target_hit:
                status = "no_target_alias_industry_news"
                reason = "Generic market or industry title without target-company alias."
                include = "Manual"
                needs_review = "Yes"
                review_reason = "market_or_industry_story_without_target_alias"
                exclude_reason = "no_target_alias"
            elif generic_title and target_hit:
                status = "low_confidence_company_news"
                reason = "Generic market title, but the body/title contains a target-company alias."
                include = "Manual"
                confidence = "low"
                needs_review = "Yes"
                review_reason = "generic_market_title_with_target_alias"
            elif title_company_focus:
                status = "high_confidence_company_news"
                reason = "Target-company legal, brand, or common alias appears in the title with substantive company-news context."
                include = "Yes"
                main_include = "Yes"
                high_include = "Yes"
                confidence = "high"
            elif lead_company_focus:
                status = "high_confidence_company_news"
                reason = "Target-company legal, brand, or common alias appears in the first two paragraphs with substantive company-news context."
                include = "Yes"
                main_include = "Yes"
                high_include = "Yes"
                confidence = "high"
            elif soft_story and (strong or medium):
                status = "low_confidence_company_news"
                reason = "CSR, sponsorship, metro, lifestyle, or event story with target-company participation but limited firm-level information."
                include = "Manual"
                confidence = "low"
                needs_review = "Yes"
                review_reason = "csr_sponsorship_lifestyle_review"
            elif focus_blocker and (strong or medium or weak):
                status = "low_confidence_company_news"
                reason = f"Target-company alias appears only in a context requiring manual review: {focus_blocker}."
                include = "Manual"
                confidence = "low"
                needs_review = "Yes"
                review_reason = focus_blocker
            elif (medium_title or focus_lead_hits) and has_company_focus_context(focus_context_text):
                status = "medium_confidence_company_news"
                reason = "Target-company alias appears in title or first two paragraphs with company-relevant context."
                include = "Yes"
                main_include = "Yes"
                confidence = "medium"
            elif weak_title_context:
                status = "low_confidence_company_news"
                reason = "Ticker-only weak alias appears in title with business or F&B context."
                include = "Manual"
                confidence = "low"
                needs_review = "Yes"
                review_reason = "ticker_only_title_context"
            elif strong or medium:
                status = "low_confidence_company_news"
                reason = "Target-company alias appears, but company-specific relevance is weak."
                include = "Manual"
                confidence = "low"
                needs_review = "Yes"
                review_reason = "weak_company_specific_context"
            elif has_business_context(combined):
                status = "no_target_alias_industry_news"
                reason = "F&B/business context exists, but no target-company alias is present."
                include = "Manual"
                needs_review = "Yes"
                review_reason = "manual_review_no_target_alias"
                exclude_reason = "no_target_alias"
            else:
                status = "no_target_alias_industry_news"
                reason = "No target-company alias and no clear company-specific business event."
                exclude_reason = "irrelevant_general_news"

        seen_global.add(article_hash)
        seen_by_ticker[ticker].add(article_hash)

    if mode == "strict":
        if include == "Manual":
            include = "No"
        main_include = "Yes" if include == "Yes" and status in {"high_confidence_company_news", "medium_confidence_company_news"} else "No"
        if status == "low_confidence_company_news":
            include = "No"
            main_include = "No"
            exclude_reason = exclude_reason or "manual_review_required"

    return {
        "document_id": document_id,
        "ticker": ticker,
        "company_name": company_name,
        "title": title,
        "source_name": source_name,
        "publish_date": publish_date,
        "url": url,
        "text_length": text_length,
        "alias_hits": "|".join(hit["alias"] for hit in hits),
        "strong_alias_hits": "|".join(hit["alias"] for hit in strong),
        "medium_alias_hits": "|".join(hit["alias"] for hit in medium),
        "weak_alias_hits": "|".join(hit["alias"] for hit in weak),
        "relevance_status": status,
        "relevance_reason": reason,
        "include_flag": include,
        "exclude_reason": exclude_reason,
        "confidence_tier": confidence,
        "main_model_include_flag": main_include,
        "broad_include_flag": "Yes" if main_include == "Yes" else "No",
        "strict_company_focus_flag": "Yes" if main_include == "Yes" else "No",
        "high_confidence_include_flag": high_include,
        "needs_manual_review": needs_review,
        "review_reason": review_reason,
        "shared_article_flag": "1" if shared_article else "0",
        "multi_company_article_flag": "1" if multi_company else "0",
        "raw_article_hash": article_hash,
        "raw_row_key": row_key,
        "filter_mode": mode,
        "filter_version": FILTER_VERSION,
        "original_file": row.get("original_file", ""),
        "original_file_hash": row.get("original_file_hash", ""),
    }


def alias_hits(text: str, ticker: str, aliases: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    if ticker not in aliases:
        return []
    return [hit for hit in aliases[ticker] if alias_in_text(hit["alias"], text)]


def flatten_strong_aliases(aliases: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for alias_rows in aliases.values():
        for row in alias_rows:
            if row.get("match_strength") != "strong":
                continue
            alias = normalize_space(row.get("alias"))
            if len(alias) <= 4:
                continue
            new_row = dict(row)
            new_row["_alias_lower"] = alias.lower()
            rows.append(new_row)
    return rows


def other_strong_alias_hits(text: str, ticker: str, strong_alias_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    lower = text.lower()
    hits: list[dict[str, str]] = []
    for row in strong_alias_rows:
        if row.get("ticker") == ticker:
            continue
        alias_lower = row.get("_alias_lower", "")
        if alias_lower and alias_lower in lower:
            hits.append(row)
            if len(hits) >= 5:
                break
    return hits


def alias_in_text(alias: str, text: str) -> bool:
    alias_clean = normalize_space(alias)
    if not alias_clean:
        return False
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(alias_clean)}(?![A-Za-z0-9])", text, flags=re.I))


def alias_hit_count(text: str, alias: str) -> int:
    alias_clean = normalize_space(alias)
    if not alias_clean:
        return 0
    return len(re.findall(rf"(?<![A-Za-z0-9]){re.escape(alias_clean)}(?![A-Za-z0-9])", text, flags=re.I))


def legal_or_brand_alias_count(text: str, hits: list[dict[str, str]]) -> int:
    return sum(
        alias_hit_count(text, hit.get("alias", ""))
        for hit in hits
        if hit.get("alias_type") in {"legal_name", "brand_name"}
    )


def first_two_paragraphs(text: str) -> str:
    raw = str(text or "").replace("\r", "\n")
    news_paragraphs = [
        normalize_space(match.group(1))
        for match in re.finditer(r"\[NEWS_P\d{3}\]\s*(.*?)(?=\n\[NEWS_P\d{3}\]|\Z)", raw, flags=re.S)
        if normalize_space(match.group(1))
    ]
    if news_paragraphs:
        return normalize_space(" ".join(news_paragraphs[:2]))
    paragraphs = []
    for part in re.split(r"\n\s*\n|(?=\n?\[NEWS_P\d{3}\])", raw):
        cleaned = normalize_space(re.sub(r"^\s*\[NEWS_P\d{3}\]\s*", "", part))
        if cleaned:
            paragraphs.append(cleaned)
    if len(paragraphs) < 2:
        normalized = normalize_space(raw)
        paragraphs = [item.strip() for item in re.split(r"(?<=[.!?])\s+", normalized) if item.strip()]
    return normalize_space(" ".join(paragraphs[:2])) or normalize_space(raw)[:1200]


def company_focus_alias_hits(text: str, ticker: str, aliases: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    return [
        hit
        for hit in alias_hits(text, ticker, aliases)
        if hit.get("match_strength") != "weak" and hit.get("alias_type") in COMPANY_FOCUS_ALIAS_TYPES
    ]


def has_company_focus_context(text: str) -> bool:
    lower = text.lower()
    keywords = [
        "acquisition",
        "analyst",
        "board",
        "business",
        "capacity",
        "commodity",
        "contract",
        "cost",
        "dividend",
        "earnings",
        "expansion",
        "export",
        "factory",
        "finance",
        "financial",
        "growth",
        "investment",
        "joint venture",
        "management",
        "manufacturing",
        "margin",
        "market analysis",
        "market performance",
        "market share",
        "merger",
        "net profit",
        "outlook",
        "plant",
        "product",
        "profit",
        "quarter",
        "raw material",
        "regulation",
        "regulatory",
        "revenue",
        "sales",
        "share price",
        "supply chain",
        "supply",
        "takeover",
        "tax",
    ]
    return any(keyword in lower for keyword in keywords)


def has_substantive_company_topic(text: str) -> bool:
    lower = text.lower()
    keywords = [
        "annual result",
        "board",
        "capacity",
        "commodity",
        "cost",
        "dividend",
        "earnings",
        "esg",
        "expansion",
        "export",
        "factory",
        "financial",
        "growth",
        "halal",
        "launch",
        "management",
        "margin",
        "market share",
        "net profit",
        "outlook",
        "plant",
        "product",
        "profit",
        "quarter",
        "raw material",
        "regulatory",
        "revenue",
        "sales",
        "supply",
        "sustainability",
    ]
    return any(keyword in lower for keyword in keywords)


def is_soft_csr_lifestyle_story(text: str) -> bool:
    lower = text.lower()
    if any(keyword in lower for keyword in ["home repairs", "lifestyle", "metro", "stadium", "football", "sponsorship", "sponsor"]):
        return True
    soft = any(
        keyword in lower
        for keyword in [
            "csr",
            "community",
            "charity",
            "sponsor",
            "sponsorship",
            "stadium",
            "football",
            "concert",
            "festival",
            "lifestyle",
            "metro",
            "home repairs",
            "volunteer",
            "participant",
        ]
    )
    substantive = has_substantive_company_topic(text)
    return soft and not substantive


def is_venue_not_company_story(ticker: str, text: str) -> bool:
    lower = text.lower()
    return ticker == "AJI" and "ajinomoto stadium" in lower


def is_known_wrong_company_story(ticker: str, text: str) -> bool:
    lower = text.lower()
    return ticker == "PPB" and "perdana petroleum" in lower


def disqualifying_company_focus_reason(ticker: str, title: str, lead_text: str, text: str) -> str:
    combined = f"{title}\n{lead_text}\n{text}"
    title_lower = title.lower()
    lead_lower = lead_text.lower()
    early_lower = f"{title_lower}\n{lead_lower}"
    if is_venue_not_company_story(ticker, combined):
        return "venue_not_company_news"
    if any(marker in early_lower for marker in ["client list", "clients include", "client roster", "among its clients"]):
        return "client_list_only"
    if any(marker in early_lower for marker in ["sponsor list", "sponsors include", "sponsored by", "sponsorship", "supported by"]):
        return "sponsor_list_or_csr_only"
    if any(marker in early_lower for marker in ["index constituent", "constituent list", "constituents include"]):
        return "index_constituent_list_only"
    if is_generic_market_title(title) or any(
        marker in title_lower or marker in lead_lower
        for marker in ["stock market roundup", "market roundup", "gainers and losers", "trading ideas", "fbm klci"]
    ):
        return "stock_market_roundup_list_only"
    if any(marker in early_lower for marker in ["lifestyle", "sport", "sports", "metro", "community article", "school programme", "football"]):
        if not has_company_focus_context(f"{title}\n{lead_text}"):
            return "lifestyle_sport_metro_late_mention"
    return "" 


def annotate_sample_flags(root: Path, audit_rows: list[dict[str, Any]]) -> None:
    registry_rows = [
        row
        for row in read_csv_rows(root / "01_registry" / "news_registry.csv")
        if normalize_space(row.get("include_flag")).lower() == "yes"
        and normalize_space(row.get("synthetic_flag")).lower() not in {"1", "true", "yes"}
        and "synthetic_flag=true" not in normalize_space(row.get("notes")).lower()
    ]
    registry_by_raw_key = {row.get("raw_row_key", ""): row for row in registry_rows if row.get("raw_row_key")}
    broad_keys = set(registry_by_raw_key)

    focus_rows = read_csv_rows(root / "08_reports" / "news_company_focus_audit.csv")
    strict_doc_ids = {
        row.get("document_id", "")
        for row in focus_rows
        if normalize_space(row.get("strict_main_sample_include")).lower() == "yes"
        and normalize_space(row.get("suspect_not_company_focused")).lower() != "yes"
    }

    for row in audit_rows:
        raw_key = row.get("raw_row_key", "")
        if broad_keys:
            broad = raw_key in broad_keys
        else:
            broad = normalize_space(row.get("main_model_include_flag")).lower() == "yes"
        registry_doc_id = registry_by_raw_key.get(raw_key, {}).get("document_id", "")
        if strict_doc_ids and registry_doc_id:
            strict = registry_doc_id in strict_doc_ids
        else:
            strict = broad and normalize_space(row.get("main_model_include_flag")).lower() == "yes"
        row["broad_include_flag"] = "Yes" if broad else "No"
        row["strict_company_focus_flag"] = "Yes" if strict else "No"


def summarize_rows(rows: list[dict[str, Any]], raw_rows: list[dict[str, str]], mode: str) -> dict[str, Any]:
    counts = Counter(row["relevance_status"] for row in rows)
    return {
        "filter_mode": mode,
        "filter_version": FILTER_VERSION,
        "total_raw_news": len(raw_rows),
        "included_yes": sum(1 for row in rows if row["include_flag"] == "Yes"),
        "manual_candidates": sum(1 for row in rows if row["include_flag"] == "Manual"),
        "main_model_include_yes": sum(1 for row in rows if row["main_model_include_flag"] == "Yes"),
        "broad_include_yes": sum(1 for row in rows if row.get("broad_include_flag") == "Yes"),
        "strict_company_focus_yes": sum(1 for row in rows if row.get("strict_company_focus_flag") == "Yes"),
        "high_confidence_include_yes": sum(1 for row in rows if row["high_confidence_include_flag"] == "Yes"),
        "needs_manual_review_count": sum(1 for row in rows if row["needs_manual_review"] == "Yes"),
        "ticker_coverage": len({row.get("ticker", "") for row in rows if row.get("main_model_include_flag") == "Yes"}),
        "status_distribution": dict(sorted(counts.items())),
        "audit_output": "08_reports/filter_news_relevance_audit.csv",
    }


def write_auxiliary_reports(
    root: Path,
    raw_rows: list[dict[str, str]],
    previous_audit_rows: list[dict[str, str]],
    strict_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    mode: str,
    aliases: dict[str, list[dict[str, str]]],
) -> None:
    write_diagnostic_reports(root, raw_rows, previous_audit_rows, strict_rows, manual_rows)
    write_before_after_summary(root, strict_rows, manual_rows)
    inclusions = [row for row in audit_rows if row.get("main_model_include_flag") == "Yes"][:50]
    exclusions = [row for row in audit_rows if row.get("main_model_include_flag") != "Yes"][:50]
    queue = [row for row in audit_rows if row.get("needs_manual_review") == "Yes" or row.get("include_flag") == "Manual"]
    possible_false = [
        row
        for row in audit_rows
        if row.get("review_reason") in {"venue_not_company_news", "csr_sponsorship_lifestyle_review", "weak_company_specific_context", "ticker_only_title_context"}
    ]
    wrong_company = [row for row in audit_rows if row.get("relevance_status") == "wrong_company" or row.get("review_reason") == "wrong_company"]
    write_csv_rows(root / "08_reports" / "news_refilter_sample_inclusions.csv", RELEVANCE_HEADERS, inclusions)
    write_csv_rows(root / "08_reports" / "news_refilter_sample_exclusions.csv", RELEVANCE_HEADERS, exclusions)
    write_csv_rows(root / "08_reports" / "news_manual_preserve_review_queue.csv", RELEVANCE_HEADERS, queue)
    write_csv_rows(root / "08_reports" / "news_possible_false_positives.csv", RELEVANCE_HEADERS, possible_false)
    write_csv_rows(root / "08_reports" / "news_wrong_company_candidates.csv", RELEVANCE_HEADERS, wrong_company)
    write_company_focus_reports(root, raw_rows, audit_rows, aliases)


def write_company_focus_reports(root: Path, raw_rows: list[dict[str, str]], audit_rows: list[dict[str, Any]], aliases: dict[str, list[dict[str, str]]]) -> None:
    final_rows = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    registry_by_id = {row.get("document_id", ""): row for row in read_csv_rows(root / "01_registry" / "news_registry.csv")}
    audit_by_raw_key = {row.get("raw_row_key", ""): row for row in audit_rows if row.get("raw_row_key")}
    audit_by_doc_id = {row.get("document_id", ""): row for row in audit_rows if row.get("document_id")}
    raw_by_key = {raw_row_key(row, idx): row for idx, row in enumerate(raw_rows, start=1)}
    focus_rows: list[dict[str, str]] = []

    if final_rows:
        source_rows: list[tuple[dict[str, str], dict[str, str], str, bool]] = []
        for final in final_rows:
            doc_id = final.get("document_id", "")
            registry = registry_by_id.get(doc_id, {})
            audit = audit_by_raw_key.get(registry.get("raw_row_key", "")) or audit_by_doc_id.get(doc_id, {})
            raw = raw_by_key.get(registry.get("raw_row_key", "")) or raw_by_key.get(audit.get("raw_row_key", "")) or {}
            context = {**raw, **registry, **audit}
            text = ""
            text_path = registry.get("text_path", "")
            if text_path:
                path = root / text_path
                if path.exists():
                    text = path.read_text(encoding="utf-8", errors="replace")
            source_rows.append((final, context, text, True))
    else:
        source_rows = []
        for row in audit_rows:
            raw = raw_by_key.get(row.get("raw_row_key", "")) or {}
            source_rows.append((row, {**raw, **row}, "", False))

    for final_or_audit, context_row, text_override, is_final_sample in source_rows:
        ticker = normalize_space(final_or_audit.get("ticker") or context_row.get("ticker")).upper()
        title = normalize_space(final_or_audit.get("title") or context_row.get("title"))
        article_text = text_override or normalize_space(context_row.get("article_text", ""))
        if not article_text:
            article_text = normalize_space(context_row.get("relevance_reason", ""))
        focus = assess_company_focus(ticker, title, article_text, aliases)
        main_include = (
            normalize_space(final_or_audit.get("final_include_flag"))
            if is_final_sample
            else normalize_space(context_row.get("main_model_include_flag") or final_or_audit.get("final_include_flag"))
        )
        strict_include = "Yes" if focus["suspect_not_company_focused"] == "No" and main_include.lower() in {"yes", "true"} else "No"
        focus_rows.append(
            {
                "document_id": normalize_space(final_or_audit.get("document_id") or context_row.get("document_id")),
                "ticker": ticker,
                "company_name": normalize_space(final_or_audit.get("company_name") or context_row.get("company_name")),
                "title": title,
                "source_name": normalize_space(final_or_audit.get("source_name") or context_row.get("source_name")),
                "publish_date": normalize_space(final_or_audit.get("publish_date") or context_row.get("publish_date")),
                "url": normalize_space(final_or_audit.get("url") or context_row.get("url")),
                "relevance_status": normalize_space(context_row.get("relevance_status")),
                "confidence_tier": normalize_space(context_row.get("confidence_tier")),
                "main_model_include_flag": "Yes" if main_include.lower() in {"yes", "true"} else "No",
                "company_focus_status": focus["company_focus_status"],
                "suspect_not_company_focused": focus["suspect_not_company_focused"],
                "company_focus_reason": focus["company_focus_reason"],
                "strict_main_sample_include": strict_include,
                "title_focus_alias_hits": "|".join(hit.get("alias", "") for hit in focus["title_focus_alias_hits"]),
                "lead_focus_alias_hits": "|".join(hit.get("alias", "") for hit in focus["lead_focus_alias_hits"]),
                "filter_mode": normalize_space(context_row.get("filter_mode")) or "manual_preserve",
                "filter_version": FILTER_VERSION,
            }
        )

    suspect_rows = [row for row in focus_rows if row.get("suspect_not_company_focused") == "Yes"]
    strict_subset = [row for row in focus_rows if row.get("strict_main_sample_include") == "Yes"]
    write_csv_rows(root / "08_reports" / "news_company_focus_audit.csv", COMPANY_FOCUS_HEADERS, focus_rows)
    write_csv_rows(root / "08_reports" / "news_suspect_not_company_focused.csv", COMPANY_FOCUS_HEADERS, suspect_rows)
    write_csv_rows(root / "08_reports" / "news_company_focused_main_subset.csv", COMPANY_FOCUS_HEADERS, strict_subset)


def assess_company_focus(ticker: str, title: str, text: str, aliases: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    lead_text = first_two_paragraphs(text)
    focus_context_text = f"{title}\n{lead_text}"
    title_hits = company_focus_alias_hits(title, ticker, aliases)
    lead_hits = company_focus_alias_hits(lead_text, ticker, aliases)
    blocker = disqualifying_company_focus_reason(ticker, title, lead_text, text)
    context = has_company_focus_context(focus_context_text) or has_substantive_company_topic(focus_context_text)
    soft_story = is_soft_csr_lifestyle_story(f"{title}\n{text}")
    if blocker:
        status = "suspect_not_company_focused"
        reason = blocker
        suspect = "Yes"
    elif title_hits and context and not soft_story:
        status = "company_focused"
        reason = "title_alias_with_company_operating_context"
        suspect = "No"
    elif lead_hits and has_company_focus_context(focus_context_text) and not soft_story:
        status = "company_focused"
        reason = "lead_alias_with_company_operating_context"
        suspect = "No"
    else:
        status = "suspect_not_company_focused"
        if not (title_hits or lead_hits):
            reason = "no_title_or_lead_company_alias"
        elif soft_story:
            reason = "soft_csr_lifestyle_or_event_story"
        else:
            reason = "insufficient_company_operating_context"
        suspect = "Yes"
    return {
        "company_focus_status": status,
        "suspect_not_company_focused": suspect,
        "company_focus_reason": reason,
        "title_focus_alias_hits": title_hits,
        "lead_focus_alias_hits": lead_hits,
    }


def write_diagnostic_reports(
    root: Path,
    raw_rows: list[dict[str, str]],
    previous_audit_rows: list[dict[str, str]],
    strict_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
) -> None:
    current_registry = read_csv_rows(root / "01_registry" / "news_registry.csv")
    rebuilt_registry = read_csv_rows(root / "01_registry" / "news_registry_rebuilt.csv")
    news_clean_count = len(list((root / "02_extracted_text" / "news_clean").glob("*.txt")))
    scored_rows = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    final_rows = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    previous_counts = Counter(row.get("relevance_status", "") for row in previous_audit_rows)
    strict_counts = Counter(row.get("relevance_status", "") for row in strict_rows)
    manual_counts = Counter(row.get("relevance_status", "") for row in manual_rows)
    counts_rows = [
        {"metric": "input_news_rows", "value": len(raw_rows)},
        {"metric": "previous_audit_rows", "value": len(previous_audit_rows)},
        {"metric": "previous_include_flag_yes", "value": sum(row.get("include_flag") == "Yes" for row in previous_audit_rows)},
        {"metric": "strict_main_model_include_yes", "value": sum(row.get("main_model_include_flag") == "Yes" for row in strict_rows)},
        {"metric": "manual_preserve_main_model_include_yes", "value": sum(row.get("main_model_include_flag") == "Yes" for row in manual_rows)},
        {"metric": "manual_preserve_broad_include_yes", "value": sum(row.get("broad_include_flag") == "Yes" for row in manual_rows)},
        {"metric": "manual_preserve_strict_company_focus_yes", "value": sum(row.get("strict_company_focus_flag") == "Yes" for row in manual_rows)},
        {"metric": "manual_preserve_manual_candidates", "value": sum(row.get("include_flag") == "Manual" for row in manual_rows)},
        {"metric": "strict_ticker_coverage", "value": len({row.get("ticker", "") for row in strict_rows if row.get("main_model_include_flag") == "Yes"})},
        {"metric": "manual_preserve_ticker_coverage", "value": len({row.get("ticker", "") for row in manual_rows if row.get("main_model_include_flag") == "Yes"})},
        {"metric": "news_registry_rows", "value": len(current_registry)},
        {"metric": "news_registry_rebuilt_rows", "value": len(rebuilt_registry)},
        {"metric": "news_clean_txt_count", "value": news_clean_count},
        {"metric": "news_final_document_scores_rows", "value": len(scored_rows)},
        {"metric": "final_output_news_final_dataset_rows", "value": len(final_rows)},
    ]
    for status, count in sorted(strict_counts.items()):
        counts_rows.append({"metric": f"strict_status:{status}", "value": count})
    for status, count in sorted(manual_counts.items()):
        counts_rows.append({"metric": f"manual_preserve_status:{status}", "value": count})
    write_csv_rows(root / "08_reports" / "news_refilter_diagnostic_counts.csv", ["metric", "value"], counts_rows)

    empty_rebuilt_risk = len(rebuilt_registry) == 0 and len(current_registry) > 0
    previous_manual_to_no = sum(
        1
        for row in previous_audit_rows
        if row.get("relevance_status") in {"needs_manual_check", "industry_relevant_news"} and row.get("include_flag") == "No"
    )
    lines = [
        "# News Refilter Diagnostic Report",
        "",
        "## Current Row Counts",
        f"- input_news/news_raw_from_filter_news.csv rows: {len(raw_rows)}",
        f"- previous strict audit include_flag=Yes rows: {sum(row.get('include_flag') == 'Yes' for row in previous_audit_rows)}",
        f"- previous relevance_status distribution: {dict(sorted(previous_counts.items()))}",
        f"- previous ticker coverage: {len({row.get('ticker', '') for row in previous_audit_rows if row.get('include_flag') == 'Yes'})}",
        f"- 01_registry/news_registry.csv rows: {len(current_registry)}",
        f"- 01_registry/news_registry_rebuilt.csv rows: {len(rebuilt_registry)}",
        f"- 02_extracted_text/news_clean/*.txt count: {news_clean_count}",
        f"- 06_ratings/news_final_full/news_final_document_scores.csv rows: {len(scored_rows)}",
        f"- FINAL_OUTPUT/news_final_dataset.csv rows: {len(final_rows)}",
        "",
        "## Diagnosis",
        f"- News was filtered most heavily at strict audit / registry construction: strict main-model rows={sum(row.get('main_model_include_flag') == 'Yes' for row in strict_rows)}, manual_preserve main-model rows={sum(row.get('main_model_include_flag') == 'Yes' for row in manual_rows)}, broad include rows={sum(row.get('broad_include_flag') == 'Yes' for row in manual_rows)}.",
        f"- Empty rebuilt registry override risk: {'yes' if empty_rebuilt_risk else 'no'}.",
        "- Global duplicate risk: strict mode uses global body-hash de-duplication, while manual_preserve only de-duplicates within the same ticker and marks cross-ticker shared articles.",
        f"- Manual / needs_manual_check / industry_relevant_news rows converted to No in the previous audit: {previous_manual_to_no}.",
    ]
    (root / "08_reports" / "news_refilter_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_before_after_summary(root: Path, strict_rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]]) -> None:
    rows = []
    for mode, data in [("strict", strict_rows), ("manual_preserve", manual_rows)]:
        counts = Counter(row["relevance_status"] for row in data)
        rows.append(
            {
                "filter_mode": mode,
                "total_rows": len(data),
                "include_flag_yes": sum(row.get("include_flag") == "Yes" for row in data),
                "include_flag_manual": sum(row.get("include_flag") == "Manual" for row in data),
                "main_model_include_yes": sum(row.get("main_model_include_flag") == "Yes" for row in data),
                "broad_include_yes": sum(row.get("broad_include_flag") == "Yes" for row in data),
                "strict_company_focus_yes": sum(row.get("strict_company_focus_flag") == "Yes" for row in data),
                "high_confidence_include_yes": sum(row.get("high_confidence_include_flag") == "Yes" for row in data),
                "needs_manual_review_yes": sum(row.get("needs_manual_review") == "Yes" for row in data),
                "ticker_coverage": len({row.get("ticker", "") for row in data if row.get("main_model_include_flag") == "Yes"}),
                "duplicate_rows": counts.get("duplicate", 0),
                "no_target_alias_industry_news": counts.get("no_target_alias_industry_news", 0),
            }
        )
    write_csv_rows(
        root / "08_reports" / "news_refilter_before_after_summary.csv",
        [
            "filter_mode",
            "total_rows",
            "include_flag_yes",
            "include_flag_manual",
            "main_model_include_yes",
            "broad_include_yes",
            "strict_company_focus_yes",
            "high_confidence_include_yes",
            "needs_manual_review_yes",
            "ticker_coverage",
            "duplicate_rows",
            "no_target_alias_industry_news",
        ],
        rows,
    )


def write_main_report(root: Path, summary: dict[str, Any], strict_rows: list[dict[str, Any]], manual_rows: list[dict[str, Any]]) -> None:
    extra = [
        "",
        "## Strict vs Manual Preserve",
        f"- strict main_model_include_flag=Yes: {sum(row.get('main_model_include_flag') == 'Yes' for row in strict_rows)}",
        f"- manual_preserve main_model_include_flag=Yes: {sum(row.get('main_model_include_flag') == 'Yes' for row in manual_rows)}",
        f"- manual_preserve broad_include_flag=Yes: {sum(row.get('broad_include_flag') == 'Yes' for row in manual_rows)}",
        f"- manual_preserve strict_company_focus_flag=Yes: {sum(row.get('strict_company_focus_flag') == 'Yes' for row in manual_rows)}",
        "- manual_preserve retains low-confidence target-company mentions in the manual review queue and excludes them from the main model unless manually validated.",
        "- The broad sample is tracked separately with broad_include_flag=Yes; the strict company-focused subset is tracked with strict_company_focus_flag=Yes.",
        "- no_target_alias_industry_news rows remain outside the main model unless manually validated later.",
    ]
    write_markdown_report(
        root / "08_reports" / "filter_news_relevance_audit_report.md",
        "Filter News Relevance Audit Report",
        summary.items(),
        extra,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit imported Filter news rows for company News relevance.")
    parser.add_argument("--mode", choices=["strict", "manual_preserve"], default="manual_preserve")
    args = parser.parse_args()
    summary = audit_filter_news_relevance(mode=args.mode)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
