from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from scoring_utils import SCORING_ROOT, read_csv_rows, short_hash, write_csv_rows


RAW_NEWS_HEADERS = [
    "ticker",
    "company_name",
    "publish_date",
    "source_name",
    "title",
    "url",
    "article_text",
    "original_file",
    "original_file_hash",
]

RAW_NEWS_EXTRA_HEADERS = [
    "import_status",
    "raw_source_row",
]

RELEVANCE_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "title",
    "source_name",
    "publish_date",
    "url",
    "text_length",
    "alias_hits",
    "strong_alias_hits",
    "medium_alias_hits",
    "weak_alias_hits",
    "relevance_status",
    "relevance_reason",
    "include_flag",
    "exclude_reason",
    "confidence_tier",
    "main_model_include_flag",
    "broad_include_flag",
    "strict_company_focus_flag",
    "high_confidence_include_flag",
    "needs_manual_review",
    "review_reason",
    "shared_article_flag",
    "multi_company_article_flag",
    "raw_article_hash",
    "raw_row_key",
    "filter_mode",
    "filter_version",
    "original_file",
    "original_file_hash",
]

FILTER_NEWS_REGISTRY_HEADERS = [
    "document_id",
    "doc_type",
    "ticker",
    "company_name",
    "publish_date",
    "source_name",
    "title",
    "url",
    "article_text_hash",
    "text_length",
    "synthetic_flag",
    "include_flag",
    "relevance_status",
    "confidence_tier",
    "needs_manual_review",
    "review_reason",
    "shared_article_flag",
    "multi_company_article_flag",
    "original_file",
    "original_file_hash",
    "raw_row_key",
    "text_path",
    "notes",
]

FAILED_CASE_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "dimension_code",
    "failure_type",
    "failure_reason",
    "raw_output_path",
]

FILTER_NEWS_CANDIDATES = [
    Path("Desktop/Filter news（1）.zip"),
    Path("Desktop/Filter news (1).zip"),
    Path("Desktop/Filter news(1).zip"),
    Path("Desktop/Filter news（1）"),
    Path("Desktop/Filter news (1)"),
    Path("Desktop/Filter news(1)"),
]

SUPPORTED_NEWS_SUFFIXES = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".txt", ".html", ".htm", ".docx", ".pdf"}

BUSINESS_KEYWORDS = {
    "acquisition",
    "analyst",
    "annual",
    "award",
    "berhad",
    "bursa",
    "business",
    "capacity",
    "cash",
    "ceo",
    "chairman",
    "commodity",
    "company",
    "consumer",
    "cost",
    "dividend",
    "earnings",
    "esg",
    "export",
    "factory",
    "finance",
    "financial",
    "food",
    "forecast",
    "growth",
    "halal",
    "inflation",
    "ingredient",
    "launch",
    "loss",
    "management",
    "manufacturing",
    "margin",
    "market",
    "net profit",
    "outlook",
    "performance",
    "plant",
    "price",
    "product",
    "profit",
    "raw material",
    "revenue",
    "risk",
    "sales",
    "shares",
    "stock",
    "supply",
    "sustainability",
    "tax",
    "trade",
}

COMPANY_EVENT_KEYWORDS = {
    "acquisition",
    "appoint",
    "award",
    "board",
    "brand",
    "capacity",
    "ceo",
    "chairman",
    "contract",
    "cost",
    "dividend",
    "earnings",
    "esg",
    "expansion",
    "export",
    "factory",
    "financial result",
    "food safety",
    "growth",
    "halal",
    "launch",
    "management",
    "margin",
    "net profit",
    "outlook",
    "plant",
    "product",
    "profit",
    "quarter",
    "raw material",
    "recall",
    "revenue",
    "sales",
    "supply",
    "sustainability",
}

GENERIC_MARKET_TITLE_PATTERNS = [
    r"\bfbm klci\b",
    r"\bbursa malaysia\b",
    r"\bbursa malaysia\b.*\b(red|lower|higher|gain|decline|cautious|trade|close|opens?)\b",
    r"\bringgit\b",
    r"\bwall street\b",
    r"\bregional sentiment\b",
    r"\bmarket breadth\b",
    r"\bcounters\b",
    r"\bindices\b",
    r"\bindex\b",
    r"\bsea of red\b",
    r"\bbargain hunting\b",
    r"\btrading ideas\b",
    r"\btake profit\b",
]

SEARCH_RESULT_MARKERS = {
    "search results",
    "related searches",
    "top stories",
    "showing results",
    "people also ask",
    "google news",
    "bing news",
    "yahoo news",
}

COOKIE_AD_PATTERNS = [
    r"(?i)^advertisement$",
    r"(?i)^related articles?$",
    r"(?i)^subscribe now$",
    r"(?i)^sign up for.*newsletter",
    r"(?i)^for more stories",
    r"(?i)^follow us on",
    r"(?i)^copyright\b",
    r"(?i)^all rights reserved",
    r"(?i)^cookie policy",
    r"(?i)^privacy policy",
    r"(?i)^terms of use",
    r"(?i)^click here",
]

CURATED_ALIASES: dict[str, list[tuple[str, str, str]]] = {
    "CARLSBG": [
        ("Carlsberg Brewery Malaysia Berhad", "legal_name", "strong"),
        ("Carlsberg Malaysia", "brand_name", "strong"),
        ("Carlsberg Brewery", "brand_name", "strong"),
        ("Carlsberg", "brand_name", "strong"),
    ],
    "HEIM": [
        ("Heineken Malaysia Berhad", "legal_name", "strong"),
        ("Heineken Malaysia", "brand_name", "strong"),
        ("HEIM", "ticker", "weak"),
    ],
    "DLADY": [
        ("Dutch Lady Milk Industries Berhad", "legal_name", "strong"),
        ("Dutch Lady", "brand_name", "strong"),
        ("DLMI", "short_name", "medium"),
    ],
    "AJI": [
        ("Ajinomoto Malaysia Berhad", "legal_name", "strong"),
        ("Ajinomoto Malaysia", "brand_name", "strong"),
        ("Ajinomoto", "brand_name", "strong"),
    ],
    "FN": [
        ("Fraser & Neave Holdings Bhd", "legal_name", "strong"),
        ("Fraser & Neave", "brand_name", "strong"),
        ("F&NHB", "short_name", "medium"),
        ("F&N", "brand_name", "medium"),
    ],
    "NESTLE": [
        ("Nestle Malaysia Berhad", "legal_name", "strong"),
        ("Nestle Malaysia", "brand_name", "strong"),
        ("Nestle", "brand_name", "strong"),
        ("Nestlé Malaysia Berhad", "legal_name", "strong"),
        ("Nestlé Malaysia", "brand_name", "strong"),
        ("Nestlé", "brand_name", "strong"),
    ],
    "QL": [("QL Resources Berhad", "legal_name", "strong"), ("QL Resources", "brand_name", "strong")],
    "HUPSENG": [("Hup Seng Industries Berhad", "legal_name", "strong"), ("Hup Seng", "brand_name", "strong")],
    "PWROOT": [("Power Root Berhad", "legal_name", "strong"), ("Power Root", "brand_name", "strong")],
    "OFI": [
        ("Oriental Food Industries Holdings Berhad", "legal_name", "strong"),
        ("Oriental Food", "brand_name", "strong"),
        ("OFI", "ticker", "weak"),
    ],
    "FFB": [("Farm Fresh Berhad", "legal_name", "strong"), ("Farm Fresh", "brand_name", "strong")],
    "SDS": [("SDS Group Berhad", "legal_name", "strong"), ("SDS", "ticker", "weak")],
    "KAWAN": [("Kawan Food Berhad", "legal_name", "strong"), ("Kawan Food", "brand_name", "strong")],
    "REX": [("Rex Industry Berhad", "legal_name", "strong"), ("Rex", "brand_name", "medium")],
    "COCOLND": [("Cocoaland Holdings Berhad", "legal_name", "strong"), ("Cocoaland", "brand_name", "strong")],
    "APOLLO": [
        ("Apollo Food Holdings Berhad", "legal_name", "strong"),
        ("Apollo Food", "brand_name", "strong"),
        ("Apollo", "brand_name", "weak"),
    ],
    "ORGABIO": [("Orgabio Holdings Berhad", "legal_name", "strong"), ("Orgabio", "brand_name", "strong")],
    "SPRITZER": [
        ("Spritzer Bhd", "legal_name", "strong"),
        ("Spritzer Berhad", "legal_name", "strong"),
        ("Spritzer", "brand_name", "strong"),
    ],
    "KOTRA": [("Kotra Industries Berhad", "legal_name", "strong"), ("Kotra", "brand_name", "strong")],
    "KHEESAN": [("Khee San Berhad", "legal_name", "strong"), ("Khee San", "brand_name", "strong")],
    "LEONGHUP": [("Leong Hup International Berhad", "legal_name", "strong"), ("Leong Hup", "brand_name", "strong")],
    "CAB": [("CAB Cakaran Corporation Berhad", "legal_name", "strong"), ("CAB Cakaran", "brand_name", "strong")],
    "GCB": [("Guan Chong Berhad", "legal_name", "strong"), ("Guan Chong", "brand_name", "strong")],
    "MSM": [
        ("MSM Malaysia Holdings Berhad", "legal_name", "strong"),
        ("MSM Malaysia", "brand_name", "strong"),
        ("MSM", "ticker", "weak"),
    ],
    "PPB": [
        ("PPB Group Berhad", "legal_name", "strong"),
        ("PPB Group", "brand_name", "strong"),
        ("PPB", "ticker", "weak"),
    ],
    "MFLOUR": [("Malayan Flour Mills Berhad", "legal_name", "strong"), ("Malayan Flour Mills", "brand_name", "strong")],
}


def sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8", errors="ignore")).hexdigest()


def raw_article_hash(title: str, text: str) -> str:
    normalized = normalize_space(re.sub(r"\W+", " ", (text or "").lower()))
    return sha256_text(normalized)


def raw_row_key(row: dict[str, Any], index: int | str = "") -> str:
    source_row = normalize_space(row.get("raw_source_row")) or str(index or "")
    original_file = normalize_space(row.get("original_file"))
    title = normalize_space(row.get("title")).lower()
    url = normalize_space(row.get("url")).lower()
    fingerprint = short_hash("|".join([original_file, source_row, title, url]), 12)
    return f"{safe_id_part(original_file) or 'raw'}:{source_row or 'unknown'}:{fingerprint}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\ufeff", " ")).strip()


def normalize_date(value: Any) -> str:
    text = normalize_space(value)
    if not text:
        return ""
    text = text.replace("/", "-")
    candidates = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d %B %Y",
        "%B %d, %Y",
    ]
    for fmt in candidates:
        try:
            return datetime.strptime(text[:19], fmt).date().isoformat()
        except ValueError:
            pass
    match = re.search(r"(20\d{2}|19\d{2})[-_/. ]([01]?\d)[-_/. ]([0-3]?\d)", text)
    if match:
        year, month, day = match.groups()
        try:
            return datetime(int(year), int(month), int(day)).date().isoformat()
        except ValueError:
            return text[:10]
    match = re.search(r"(20\d{2}|19\d{2})", text)
    return match.group(1) if match else text[:10]


def safe_id_part(value: str) -> str:
    part = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_")
    return part or "UNKNOWN"


def infer_ticker_from_path(path_text: str) -> str:
    parts = [part for part in Path(path_text).parts if part and part not in {"/", "."}]
    for part in reversed(parts):
        stem = Path(part).stem
        match = re.match(r"([A-Za-z0-9&.-]{2,30})(?:_news)?$", stem)
        if match and not stem.lower().startswith("filter news"):
            return match.group(1).replace("&", "N").upper()
    return ""


def infer_source_name(url: str, original_file: str = "") -> str:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url or "")
    if match:
        domain = match.group(1).lower()
        if "thestar.com.my" in domain:
            return "The Star"
        return domain
    if original_file:
        return Path(original_file).parts[0] if Path(original_file).parts else ""
    return ""


def is_bad_company_name(name: str, ticker: str = "") -> bool:
    cleaned = normalize_space(name)
    lower = cleaned.lower()
    ticker_upper = normalize_space(ticker).upper()
    if not cleaned:
        return True
    if len(cleaned) <= 5:
        return True
    if ticker_upper and cleaned.upper() == ticker_upper:
        return True
    return any(
        marker in lower
        for marker in [
            "annual report",
            "annual review",
            " ar report",
            "ar report",
            "part 1",
            "part i",
        ]
    )


def load_company_lookup(root: Path = SCORING_ROOT) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(root / "01_registry" / "mda_registry.csv"):
        ticker = normalize_space(row.get("ticker")).upper()
        if not ticker:
            continue
        current = lookup.setdefault(
            ticker,
            {
                "ticker": ticker,
                "stock_code": normalize_space(row.get("stock_code")),
                "company_name": "",
            },
        )
        name = normalize_space(row.get("company_name"))
        if name and (not current.get("company_name") or is_bad_company_name(current.get("company_name", ""), ticker)):
            current["company_name"] = name
    for ticker, aliases in CURATED_ALIASES.items():
        entry = lookup.setdefault(ticker, {"ticker": ticker, "stock_code": "", "company_name": aliases[0][0]})
        if is_bad_company_name(entry.get("company_name", ""), ticker):
            entry["company_name"] = aliases[0][0]
    return lookup


def build_company_alias_map(root: Path = SCORING_ROOT) -> list[dict[str, str]]:
    lookup = load_company_lookup(root)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(ticker: str, stock_code: str, company_name: str, alias: str, alias_type: str, strength: str) -> None:
        alias_clean = normalize_space(alias)
        if not ticker or not alias_clean:
            return
        key = (ticker.upper(), alias_clean.lower())
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "ticker": ticker.upper(),
                "stock_code": stock_code,
                "company_name": company_name,
                "alias": alias_clean,
                "alias_type": alias_type,
                "match_strength": strength,
            }
        )

    for ticker, info in sorted(lookup.items()):
        company = info.get("company_name", "")
        stock = info.get("stock_code", "")
        if company and not is_bad_company_name(company, ticker):
            add(ticker, stock, company, company, "legal_name", "strong")
            simplified = re.sub(r"\b(Berhad|Bhd|Holdings?|Corporation|Industries|Group|Malaysia)\b", "", company, flags=re.I)
            simplified = normalize_space(simplified)
            if simplified and simplified.lower() != company.lower() and len(simplified) > 4:
                add(ticker, stock, company, simplified, "short_name", "medium")
        add(ticker, stock, company, ticker, "ticker", "weak")
        if stock:
            add(ticker, stock, company, stock, "stock_code", "weak")
        for alias, alias_type, strength in CURATED_ALIASES.get(ticker, []):
            add(ticker, stock, company or alias, alias, alias_type, strength)
    return rows


def write_company_alias_map(root: Path = SCORING_ROOT) -> Path:
    rows = build_company_alias_map(root)
    path = root / "01_registry" / "company_alias_map.csv"
    write_csv_rows(path, ["ticker", "stock_code", "company_name", "alias", "alias_type", "match_strength"], rows)
    return path


def clean_article_text(text: str) -> str:
    value = html.unescape(text or "")
    value = re.sub(r"<script[\s\S]*?</script>", " ", value, flags=re.I)
    value = re.sub(r"<style[\s\S]*?</style>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    kept: list[str] = []
    for raw_line in value.splitlines():
        line = normalize_space(raw_line)
        if not line:
            kept.append("")
            continue
        if any(re.search(pattern, line) for pattern in COOKIE_AD_PATTERNS):
            continue
        kept.append(line)
    value = "\n".join(kept)
    value = re.sub(r"\n{3,}", "\n\n", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def paragraphize_news(title: str, text: str) -> str:
    cleaned = clean_article_text(text)
    paragraphs = [normalize_space(item) for item in re.split(r"\n\s*\n+", cleaned) if normalize_space(item)]
    if len(paragraphs) <= 1:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            sentence = normalize_space(sentence)
            if not sentence:
                continue
            current.append(sentence)
            current_len += len(sentence)
            if current_len >= 450:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
        if current:
            chunks.append(" ".join(current))
        paragraphs = chunks or ([cleaned] if cleaned else [])
    lines = ["[TITLE]", normalize_space(title), ""]
    for idx, paragraph in enumerate(paragraphs, start=1):
        lines.append(f"[NEWS_P{idx:03d}] {paragraph}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_news_text(title: str, text: str, metadata: dict[str, Any] | None = None) -> str:
    base = paragraphize_news(title, text)
    metadata = metadata or {}
    meta_lines = ["[METADATA]"]
    for key in ["ticker", "company_name", "publish_date", "source_name", "url", "confidence_tier"]:
        meta_lines.append(f"{key}: {normalize_space(metadata.get(key))}")
    if "[TITLE]" not in base:
        return "\n".join(["[TITLE]", normalize_space(title), "", *meta_lines, "", base]).strip() + "\n"
    title_block, _, rest = base.partition("\n\n")
    return "\n\n".join([title_block, "\n".join(meta_lines), rest]).strip() + "\n"


def text_locators(text: str) -> dict[str, str]:
    locators: dict[str, str] = {}
    if "[TITLE]" in text:
        title_match = re.search(r"\[TITLE\]\s*\n(.+)", text)
        locators["TITLE"] = title_match.group(1).strip() if title_match else ""
    for match in re.finditer(r"\[(NEWS_P\d{3})\]\s*(.*?)(?=\n\[NEWS_P\d{3}\]|\Z)", text, flags=re.S):
        locators[match.group(1)] = normalize_space(match.group(2))
    return locators


def locator_text(text: str, locator: str, max_len: int = 220) -> str:
    key = normalize_space(locator).strip("[]")
    locators = text_locators(text)
    return locators.get(key, "")[:max_len]


def has_business_context(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in BUSINESS_KEYWORDS)


def has_company_event_context(text: str) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in COMPANY_EVENT_KEYWORDS)


def is_generic_market_title(title: str) -> bool:
    lower = title.lower()
    return any(re.search(pattern, lower) for pattern in GENERIC_MARKET_TITLE_PATTERNS)


def likely_search_result(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in SEARCH_RESULT_MARKERS)


def write_markdown_report(path: Path, title: str, rows: Iterable[tuple[str, Any]], extra_lines: list[str] | None = None) -> Path:
    lines = [f"# {title}", ""]
    for key, value in rows:
        lines.append(f"- {key}: {value}")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def copy_without_metadata(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_jsonish_file(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(payload if isinstance(payload, dict) else {"value": payload})
        return rows
    payload = json.loads(text)
    if isinstance(payload, list):
        return [item if isinstance(item, dict) else {"value": item} for item in payload]
    if isinstance(payload, dict):
        for key in ["articles", "items", "rows", "data", "news", "results"]:
            if isinstance(payload.get(key), list):
                return [item if isinstance(item, dict) else {"value": item} for item in payload[key]]
        return [payload]
    return []


def read_table_file(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if suffix in {".json", ".jsonl"}:
        return read_jsonish_file(path)
    if suffix in {".txt", ".html", ".htm"}:
        return [{"title": path.stem, "text": path.read_text(encoding="utf-8-sig", errors="replace")}]
    if suffix == ".docx":
        with zipfile.ZipFile(path) as zf:
            xml_text = zf.read("word/document.xml").decode("utf-8", errors="replace")
        return [{"title": path.stem, "text": re.sub(r"<[^>]+>", " ", xml_text)}]
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except Exception as exc:
                raise RuntimeError("pypdf/PyPDF2 unavailable") from exc
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return [{"title": path.stem, "text": text}]
    if suffix in {".xlsx", ".xls"}:
        try:
            from openpyxl import load_workbook  # type: ignore
        except Exception as exc:
            raise RuntimeError("openpyxl unavailable") from exc
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [normalize_space(cell) for cell in rows[0]]
        return [dict(zip(headers, row)) for row in rows[1:]]
    raise RuntimeError(f"unsupported suffix: {suffix}")
