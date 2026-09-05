from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCORING_ROOT = Path(__file__).resolve().parents[1]
SCOREBOOK_VERSION = "2026-06-15"

DEFAULT_ENV = {
    "SCORING_LLM_BACKEND": "ollama",
    "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
    "OLLAMA_MODEL": "qwen3:8b",
    "OLLAMA_TIMEOUT_SECONDS": "180",
    "OLLAMA_TEMPERATURE": "0",
    "OLLAMA_SEED": "42",
    "OLLAMA_NUM_CTX": "4096",
    "OLLAMA_NUM_PREDICT": "192",
    "OLLAMA_MAX_RETRIES": "3",
}

DIRECTORIES = [
    "00_scorebook",
    "01_registry",
    "02_extracted_text/mda",
    "02_extracted_text/news",
    "03_numeric_checks/mda",
    "04_prompts",
    "05_raw_model_outputs/mda",
    "05_raw_model_outputs/news",
    "06_ratings",
    "06_ratings/per_document/mda",
    "06_ratings/per_document/news",
    "07_review",
    "08_reports",
    "10_claims/mda_claims",
    "10_claims/news_claims",
    "10_claim_extraction",
    "10_claim_scoring/mda_claim_scores",
    "10_claim_scoring/news_claim_scores",
    "10_claim_scoring/raw_model_outputs/mda",
    "10_claim_scoring/raw_model_outputs/news",
    "10_claim_mapping",
    "11_stability_analysis/inputs",
    "11_stability_analysis/outputs",
    "11_stability_analysis/reports",
    "11_stability_analysis/plots",
    "11_stability_analysis/logs",
    "FINAL_OUTPUT",
    "scripts",
    "tests",
    "_self_correction",
    "_self_correction/test_runs",
    "_self_correction/synthetic_inputs",
]

FLAG_FIELDS = [
    "flag_unverified_key_number",
    "flag_excessive_promotional_tone",
    "flag_nonstandard_audit",
    "flag_disclosure_replacement",
    "flag_retraction_or_correction_needed",
]

DIMENSIONS = {
    "mda": [
        {"code": "AR01", "name": "MD&A content completeness", "weight": 0.15},
        {"code": "AR02", "name": "MD&A internal numeric consistency and traceability", "weight": 0.25},
        {"code": "AR03", "name": "Operating analysis specificity", "weight": 0.30},
        {"code": "AR04", "name": "MD&A internal risk and forward-looking disclosure", "weight": 0.20},
        {"code": "AR05", "name": "MD&A readability and structural clarity", "weight": 0.10},
    ],
    "news": [
        {"code": "N01", "name": "Factual accuracy and internal consistency", "weight": 0.30},
        {"code": "N02", "name": "Source transparency and verifiability", "weight": 0.25},
        {"code": "N03", "name": "Balance and objectivity", "weight": 0.15},
        {"code": "N04", "name": "Relevance and information increment", "weight": 0.20},
        {"code": "N05", "name": "Expression clarity and headline fit", "weight": 0.10},
    ],
}

REGISTRY_HEADERS = {
    "mda": [
        "document_id",
        "doc_type",
        "include_flag",
        "stock_code",
        "ticker",
        "company_name",
        "report_year",
        "extraction_status",
        "text_path",
        "source_path",
        "source_file_sha256",
        "notes",
    ],
    "news": [
        "document_id",
        "doc_type",
        "include_flag",
        "company_name",
        "ticker",
        "source_name",
        "publish_date",
        "title",
        "url",
        "text_path",
        "notes",
    ],
}

RATINGS_LONG_HEADERS = [
    "rating_id",
    "document_id",
    "doc_type",
    "scoring_basis",
    "purpose_track",
    "review_round",
    "reviewer_id",
    "dimension_code",
    "dimension_name",
    "raw_score",
    "original_model_raw_score",
    "adjusted_raw_score",
    "adjustment_applied",
    "adjustment_rule",
    "adjustment_reason",
    "numeric_checker_result",
    "scale_min",
    "scale_max",
    "weight_value",
    "std_score_100",
    "weighted_score",
    "evidence_locator",
    "evidence_text_short",
    "confidence_level",
    "comment_short",
    "missing_type",
    "started_at",
    "finished_at",
    "scorebook_version",
    "model_name",
    "llm_backend",
    "temperature",
    "seed",
    "num_ctx",
    "num_predict",
    "think",
    "stream",
    "json_schema_mode",
    "prompt_version",
    "scoring_timestamp",
    "raw_output_path",
    "numeric_check_path",
    "parse_status",
    "schema_validation_status",
]

MDA_DOCUMENT_HEADERS = [
    "document_id",
    "doc_type",
    "scoring_basis",
    "stock_code",
    "ticker",
    "company_name",
    "report_year",
    "dimension_count",
    "total_score_100",
    "grade_label",
    "AR01",
    "AR02",
    "AR03",
    "AR04",
    "AR05",
    "AR01_std",
    "AR02_std",
    "AR03_std",
    "AR04_std",
    "AR05_std",
    *FLAG_FIELDS,
    "adjudication_flag",
    "needs_review",
    "review_reasons",
    "final_version",
    "scorebook_version",
    "model_name",
    "llm_backend",
    "temperature",
    "seed",
    "num_ctx",
    "num_predict",
    "think",
    "stream",
    "json_schema_mode",
    "prompt_version",
    "scoring_timestamp",
]

NEWS_DOCUMENT_HEADERS = [
    "document_id",
    "doc_type",
    "company_name",
    "ticker",
    "source_name",
    "publish_date",
    "title",
    "url",
    "confidence_tier",
    "relevance_status",
    "review_reason",
    "dimension_count",
    "total_score_100",
    "grade_label",
    "N01",
    "N02",
    "N03",
    "N04",
    "N05",
    "N01_std",
    "N02_std",
    "N03_std",
    "N04_std",
    "N05_std",
    *FLAG_FIELDS,
    "adjudication_flag",
    "needs_review",
    "review_reasons",
    "final_version",
    "scorebook_version",
    "model_name",
    "llm_backend",
    "temperature",
    "seed",
    "num_ctx",
    "num_predict",
    "think",
    "stream",
    "json_schema_mode",
    "prompt_version",
    "scoring_timestamp",
]

REVIEW_LOG_HEADERS = [
    "document_id",
    "doc_type",
    "review_type",
    "review_reason",
    "dimension_code",
    "raw_score",
    "evidence_locator",
    "numeric_check_path",
    "raw_output_path",
    "created_at",
]


def env_value(name: str) -> str:
    return os.environ.get(name, DEFAULT_ENV[name])


def ensure_directories(root: Path = SCORING_ROOT) -> None:
    for item in DIRECTORIES:
        (root / item).mkdir(parents=True, exist_ok=True)


def standardize_score(raw_score: int | float) -> float:
    raw = float(raw_score)
    if raw < 1 or raw > 5:
        raise ValueError(f"raw_score must be between 1 and 5, got {raw_score}")
    return round(100 * (raw - 1) / 4, 6)


def weighted_score(raw_score: int | float, weight_value: int | float) -> float:
    return round(standardize_score(raw_score) * float(weight_value), 6)


def document_total(raw_scores: dict[str, int | float], weights: dict[str, int | float]) -> float:
    total = 0.0
    for code, raw_score in raw_scores.items():
        total += weighted_score(raw_score, weights[code])
    return round(total, 6)


def grade_label(total_score_100: int | float | str) -> str:
    total = float(total_score_100)
    if total >= 85:
        return "A"
    if total >= 70:
        return "B"
    if total >= 55:
        return "C"
    return "D"


def dimensions_for(doc_type: str) -> list[dict[str, Any]]:
    if doc_type not in DIMENSIONS:
        raise ValueError("doc_type must be mda or news")
    return [dict(item) for item in DIMENSIONS[doc_type]]


def weights_for(doc_type: str) -> dict[str, float]:
    return {item["code"]: float(item["weight"]) for item in dimensions_for(doc_type)}


def scoring_basis_for(doc_type: str) -> str:
    return "mda_disclosure_quality" if doc_type == "mda" else "news_quality"


def purpose_track_for(doc_type: str) -> str:
    return "mda_general_quality" if doc_type == "mda" else "news_general_quality"


def document_headers_for(doc_type: str) -> list[str]:
    return MDA_DOCUMENT_HEADERS if doc_type == "mda" else NEWS_DOCUMENT_HEADERS


def registry_path(doc_type: str, root: Path = SCORING_ROOT) -> Path:
    return root / "01_registry" / f"{doc_type}_registry.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv_rows(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def short_hash(text: str, length: int = 10) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def resolve_text_path(row: dict[str, str], doc_type: str, root: Path = SCORING_ROOT) -> Path:
    path_text = row.get("text_path", "").strip()
    if path_text:
        path = Path(path_text)
        return path if path.is_absolute() else root / path
    return root / "02_extracted_text" / doc_type / f"{row['document_id']}.txt"


def write_default_registries(root: Path = SCORING_ROOT) -> None:
    ensure_directories(root)
    for doc_type, headers in REGISTRY_HEADERS.items():
        path = registry_path(doc_type, root)
        if not path.exists():
            write_csv_rows(path, headers, [])


def weight_sum_ok(doc_type: str) -> bool:
    return abs(sum(weights_for(doc_type).values()) - 1.0) < 0.000001
