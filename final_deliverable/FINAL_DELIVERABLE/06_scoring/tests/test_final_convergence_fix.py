from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from build_news_registry import build_news_registry
from diagnose_mda_stability_failures import diagnose_mda_stability_failures
from extract_news_texts import extract_news_texts
from run_cross_validation import run_cross_validation
from scoring_utils import MDA_DOCUMENT_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows


def test_build_news_registry_and_extract_texts_from_real_zip(tmp_path):
    root = tmp_path / "06_scoring"
    zip_path = tmp_path / "real_news.zip"
    _write_news_zip(zip_path)
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        REGISTRY_HEADERS["mda"],
        [
            {
                "document_id": "MDA_2658_AJI_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "2658",
                "ticker": "AJI",
                "company_name": "Ajinomoto",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda/MDA_2658_AJI_2024.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        ],
    )

    registry_summary = build_news_registry(root=root, source_zip=zip_path, max_rows_per_ticker=2, overwrite=True)
    extract_summary = extract_news_texts(root=root, source_zip=zip_path, overwrite=True)

    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    assert registry_summary["real_news_count"] == 2
    assert extract_summary["extracted_count"] == 2
    assert all("synthetic_flag=false" in row["notes"] for row in registry_rows)
    assert (root / registry_rows[0]["text_path"]).read_text(encoding="utf-8").startswith("[TITLE]")
    assert "[NEWS_P001]" in (root / registry_rows[0]["text_path"]).read_text(encoding="utf-8")


def test_diagnose_mda_stability_failures_breaks_down_failure_rates(tmp_path):
    root = tmp_path / "06_scoring"
    selection = [{"document_id": "MDA_OK"}, {"document_id": "MDA_FAIL"}]
    write_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv", ["document_id"], selection)
    write_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_score("MDA_OK")])
    _write_per_doc(root, "MDA_OK", "success", "parsed", "valid")
    _write_per_doc(root, "MDA_FAIL", "failed", "parse_failed", "invalid")

    summary = diagnose_mda_stability_failures(root=root)

    assert summary["success_rate"] == 0.5
    assert summary["json_failure_rate"] == 0.5
    assert summary["schema_failure_rate"] == 0.5
    assert summary["top_root_causes"][0]["reason"] == "json_parse_failed"
    assert (root / "08_reports" / "mda_stability_failure_root_cause.md").exists()


def test_cross_validation_uses_rule_fallback_for_real_news_when_model_unavailable(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        REGISTRY_HEADERS["mda"],
        [
            {
                "document_id": "MDA_AJI_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "2658",
                "ticker": "AJI",
                "company_name": "Ajinomoto",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda/MDA_AJI_2024.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_AJI_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Ajinomoto",
                "ticker": "AJI",
                "source_name": "real_news_csv",
                "publish_date": "2024-05-01",
                "title": "Ajinomoto reports revenue growth",
                "url": "https://example.com",
                "text_path": "02_extracted_text/news/NEWS_AJI_2024.txt",
                "notes": "synthetic_flag=false",
            }
        ],
    )
    (root / "02_extracted_text" / "mda").mkdir(parents=True)
    (root / "02_extracted_text" / "news").mkdir(parents=True)
    (root / "02_extracted_text" / "mda" / "MDA_AJI_2024.txt").write_text("[MDA_P001]\nRevenue increased during 2024 due to stronger demand.", encoding="utf-8")
    (root / "02_extracted_text" / "news" / "NEWS_AJI_2024.txt").write_text("[NEWS_P001]\nAjinomoto said revenue increased during 2024 due to stronger demand.", encoding="utf-8")

    summary = run_cross_validation("pilot", root=root, base_url="http://127.0.0.1:9", timeout_seconds=1, limit=1, mock_mode=False)

    assert summary["model_available"] is False
    assert summary["rule_fallback_used"] is True
    assert summary["claim_link_count"] > 0


def _write_news_zip(path: Path) -> None:
    rows = [
        {
            "content_id": "1",
            "title": "Ajinomoto reports revenue growth",
            "text": "Ajinomoto reported revenue growth and higher raw material costs in Malaysia.",
            "section": "business",
            "category": "market",
            "content_tier": "",
            "content_length": "80",
            "authors": "Reporter",
            "published_date": "2024-05-01",
            "keywords": "Ajinomoto revenue",
            "summary": "Revenue growth",
            "url": "https://example.com/aji-1",
            "top_image": "",
        },
        {
            "content_id": "2",
            "title": "Ajinomoto sees cost pressure",
            "text": "The company said packaging and logistics costs increased during the year.",
            "section": "business",
            "category": "market",
            "content_tier": "",
            "content_length": "70",
            "authors": "Reporter",
            "published_date": "2024-06-01",
            "keywords": "Ajinomoto cost",
            "summary": "Cost pressure",
            "url": "https://example.com/aji-2",
            "top_image": "",
        },
    ]
    fieldnames = list(rows[0])
    csv_path = path.parent / "AJI_news.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with zipfile.ZipFile(path, "w") as zf:
        zf.write(csv_path, "New_news/AJI/AJI_news.csv")


def _mda_score(document_id: str) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "mda", "ticker": "AJI", "company_name": "Ajinomoto", "report_year": "2024", "dimension_count": "5", "total_score_100": "75"})
    return row


def _write_per_doc(root: Path, document_id: str, status: str, parse_status: str, schema_status: str) -> None:
    path = root / "06_ratings" / "mda_v2_stability_batch" / "per_document" / f"{document_id}_parsed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "{"
        f'"document_id":"{document_id}",'
        f'"status":"{status}",'
        f'"result":{{"parse_status":"{parse_status}","schema_validation_status":"{schema_status}","error_type":"scoring_failed","error_message":"json_parse_failed"}}'
        "}",
        encoding="utf-8",
    )
