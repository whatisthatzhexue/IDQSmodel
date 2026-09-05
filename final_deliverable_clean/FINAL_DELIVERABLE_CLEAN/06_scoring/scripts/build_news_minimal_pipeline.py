from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from news_pipeline_status_check import check_news_pipeline_status
from scoring_utils import (
    NEWS_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    REGISTRY_HEADERS,
    SCORING_ROOT,
    dimensions_for,
    ensure_directories,
    grade_label,
    read_csv_rows,
    standardize_score,
    weighted_score,
    weights_for,
    write_csv_rows,
)


PLACEHOLDER_NEWS = [
    {
        "document_id": "NEWS_PLACEHOLDER_SOURCE_2024",
        "ticker": "NEWSMIN",
        "company_name": "Minimal News Placeholder Berhad",
        "source_name": "Synthetic Placeholder Wire",
        "publish_date": "2024-06-01",
        "title": "Minimal News Placeholder reports operating update",
        "text": "[NEWS_P001]\nAccording to the company statement, the group reported stable demand and continued cost monitoring in 2024.\n\n[NEWS_P002]\nThe article identifies the company statement as its source and does not provide independent market confirmation.\n",
        "scores": {"N01": 3, "N02": 3, "N03": 3, "N04": 3, "N05": 3},
    },
    {
        "document_id": "NEWS_PLACEHOLDER_MARKET_2024",
        "ticker": "NEWSMIN",
        "company_name": "Minimal News Placeholder Berhad",
        "source_name": "Synthetic Market Note",
        "publish_date": "2024-07-01",
        "title": "Market note discusses food input cost pressure",
        "text": "[NEWS_P001]\nA market note said Malaysian food producers continued to face input cost pressure during 2024.\n\n[NEWS_P002]\nThe note gives industry context but does not directly verify individual company financial statements.\n",
        "scores": {"N01": 3, "N02": 3, "N03": 3, "N04": 4, "N05": 3},
    },
]


def build_news_minimal_pipeline(root: Path = SCORING_ROOT, overwrite: bool = False) -> dict[str, Any]:
    ensure_directories(root)
    before = check_news_pipeline_status(root)
    registry_path = root / "01_registry" / "news_registry.csv"
    scores_path = root / "06_ratings" / "news_document_scores.csv"
    ratings_path = root / "06_ratings" / "news_ratings_long.csv"
    text_dir = root / "02_extracted_text" / "news"
    text_dir.mkdir(parents=True, exist_ok=True)

    registry_rows = read_csv_rows(registry_path)
    text_files = sorted(text_dir.glob("*.txt"))
    created_synthetic = False

    if registry_rows and not overwrite:
        selected_rows = registry_rows
    elif text_files:
        selected_rows = _registry_from_text_files(text_files)
    else:
        selected_rows = _write_placeholder_texts(text_dir)
        created_synthetic = True

    write_csv_rows(registry_path, REGISTRY_HEADERS["news"], selected_rows)
    document_rows, rating_rows = _score_news_rows(root, selected_rows)
    write_csv_rows(scores_path, NEWS_DOCUMENT_HEADERS, document_rows)
    write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, rating_rows)
    after = check_news_pipeline_status(root)
    summary = {
        "created_synthetic_placeholder": created_synthetic,
        "registry_path": str(registry_path),
        "text_dir": str(text_dir),
        "news_document_scores_path": str(scores_path),
        "news_ratings_long_path": str(ratings_path),
        "registry_count": len(selected_rows),
        "document_score_count": len(document_rows),
        "rating_row_count": len(rating_rows),
        "before_status": before["overall_status"],
        "after_status": after["overall_status"],
        "synthetic_flag_count": sum("synthetic_flag=true" in row.get("notes", "").lower() for row in selected_rows),
    }
    return summary


def _write_placeholder_texts(text_dir: Path) -> list[dict[str, str]]:
    rows = []
    for item in PLACEHOLDER_NEWS:
        path = text_dir / f"{item['document_id']}.txt"
        path.write_text(item["text"], encoding="utf-8")
        rows.append(
            {
                "document_id": item["document_id"],
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": item["company_name"],
                "ticker": item["ticker"],
                "source_name": item["source_name"],
                "publish_date": item["publish_date"],
                "title": item["title"],
                "url": "",
                "text_path": f"02_extracted_text/news/{item['document_id']}.txt",
                "notes": "synthetic_flag=true; placeholder_minimal_news_pipeline=true",
            }
        )
    return rows


def _registry_from_text_files(text_files: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in text_files:
        doc_id = path.stem
        rows.append(
            {
                "document_id": doc_id,
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Unknown News Company",
                "ticker": "UNKNOWN",
                "source_name": "Existing Local Text",
                "publish_date": "2024-01-01",
                "title": doc_id,
                "url": "",
                "text_path": f"02_extracted_text/news/{path.name}",
                "notes": "synthetic_flag=false; built_from_existing_text=true",
            }
        )
    return rows


def _score_news_rows(root: Path, registry_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    weights = weights_for("news")
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    document_rows = []
    rating_rows = []
    placeholder_scores = {item["document_id"]: item["scores"] for item in PLACEHOLDER_NEWS}
    for row in registry_rows:
        doc_id = row.get("document_id", "")
        scores = placeholder_scores.get(doc_id) or _heuristic_scores((root / row.get("text_path", "")).read_text(encoding="utf-8", errors="replace") if row.get("text_path") else "")
        total = round(sum(weighted_score(scores[code], weights[code]) for code in weights), 6)
        doc_row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
        doc_row.update(
            {
                "document_id": doc_id,
                "doc_type": "news",
                "company_name": row.get("company_name", ""),
                "ticker": row.get("ticker", ""),
                "source_name": row.get("source_name", ""),
                "publish_date": row.get("publish_date", ""),
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "dimension_count": "5",
                "total_score_100": total,
                "grade_label": grade_label(total),
                "adjudication_flag": "0",
                "needs_review": "True" if "synthetic_flag=true" in row.get("notes", "").lower() else "False",
                "review_reasons": "synthetic_placeholder_news_pipeline" if "synthetic_flag=true" in row.get("notes", "").lower() else "",
                "final_version": "news_minimal_pilot",
                "scorebook_version": "2026-06-15",
                "model_name": "minimal_python_news_pipeline",
                "llm_backend": "python_no_llm",
            }
        )
        for code, score in scores.items():
            doc_row[code] = score
            doc_row[f"{code}_std"] = standardize_score(score)
        document_rows.append(doc_row)
        for idx, dimension in enumerate(dimensions_for("news"), start=1):
            code = dimension["code"]
            score = scores[code]
            rating_row = {field: "" for field in RATINGS_LONG_HEADERS}
            rating_row.update(
                {
                    "rating_id": f"{doc_id}_{code}",
                    "document_id": doc_id,
                    "doc_type": "news",
                    "scoring_basis": "news_quality",
                    "purpose_track": "news_general_quality",
                    "review_round": "pilot",
                    "reviewer_id": "minimal_python_news_pipeline",
                    "dimension_code": code,
                    "dimension_name": dimension["name"],
                    "raw_score": score,
                    "scale_min": "1",
                    "scale_max": "5",
                    "weight_value": weights[code],
                    "std_score_100": standardize_score(score),
                    "weighted_score": weighted_score(score, weights[code]),
                    "evidence_locator": f"[NEWS_P{idx:03d}]" if idx <= 2 else "[NEWS_P001]",
                    "evidence_text_short": "Minimal News pipeline evidence locator present.",
                    "confidence_level": "medium",
                    "comment_short": "Minimal pilot News score generated without qwen; review before research use.",
                    "missing_type": "none",
                    "started_at": now,
                    "finished_at": now,
                    "scorebook_version": "2026-06-15",
                    "model_name": "minimal_python_news_pipeline",
                    "llm_backend": "python_no_llm",
                    "raw_output_path": "",
                    "numeric_check_path": "",
                    "parse_status": "parsed",
                    "schema_validation_status": "valid",
                }
            )
            rating_rows.append(rating_row)
    return document_rows, rating_rows


def _heuristic_scores(text: str) -> dict[str, int]:
    lowered = text.lower()
    return {
        "N01": 3,
        "N02": 4 if "according to" in lowered or "source" in lowered else 3,
        "N03": 3,
        "N04": 4 if "market" in lowered or "industry" in lowered else 3,
        "N05": 3,
    }


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a minimal independent News pipeline if News inputs are missing.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()
    print(json.dumps(build_news_minimal_pipeline(args.root, args.overwrite), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
