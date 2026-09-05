from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from audit_filter_news_relevance import audit_filter_news_relevance
from build_news_registry_from_filter_news import build_news_registry_from_filter_news
from extract_filter_news_texts import extract_filter_news_texts
from import_filter_news_zip import import_filter_news
from run_filter_news_final_scoring import run_filter_news_final_scoring
from score_package_checks import check_mda
from scoring_utils import NEWS_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, write_csv_rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def seed_mda_registry(root: Path) -> None:
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        ["document_id", "doc_type", "include_flag", "stock_code", "ticker", "company_name", "report_year", "extraction_status", "text_path", "source_path", "source_file_sha256", "notes"],
        [
            {
                "document_id": "MDA_2836_CARLSBG_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "2836",
                "ticker": "CARLSBG",
                "company_name": "Carlsberg Brewery Malaysia Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda/MDA_2836_CARLSBG_2024.txt",
                "source_path": "source.pdf",
                "source_file_sha256": "abc",
                "notes": "",
            }
        ],
    )


def test_import_audit_registry_extract_and_score_real_filter_news(tmp_path: Path) -> None:
    root = tmp_path
    seed_mda_registry(root)
    news_text = (
        "Carlsberg Malaysia reported stronger business performance and revenue growth in its brewery operations. "
        "The company said product demand improved, management highlighted supply discipline, and analysts noted "
        "that market share, profitability, pricing and operating costs remained important for the F&B group. "
        "The article includes company-specific business facts and source context for investors."
    )
    archive = tmp_path / "Filter news(1).zip"
    csv_content = (
        "title,text,published_date,url\n"
        f"Carlsberg Malaysia posts stronger revenue,\"{news_text}\",2025-06-26 00:00:00,https://www.thestar.com.my/business/carlsberg\n"
        "Generic ringgit story,\"KUALA LUMPUR: The ringgit moved higher on central bank expectations.\",2025-01-01,https://example.com/ringgit\n"
    )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("New_news/CARLSBG/CARLSBG_news.csv", csv_content)

    import_summary = import_filter_news(archive, root)
    assert import_summary["filter_news_found"] is True
    assert import_summary["total_raw_rows"] == 2
    relevance = audit_filter_news_relevance(root)
    assert relevance["included_yes"] == 1
    registry = build_news_registry_from_filter_news(root)
    assert registry["registry_rows"] == 1
    extraction = extract_filter_news_texts(root)
    assert extraction["clean_text_files_written"] == 1

    def fake_scorer(dimension, row, text):
        assert "[NEWS_P001]" in text
        return {"score": 4, "evidence": "NEWS_P001", "reason": f"{dimension['code']} evidence is sufficient"}

    scoring = run_filter_news_final_scoring(root, scorer=fake_scorer)
    assert scoring["document_score_rows"] == 1
    assert scoring["ratings_long_rows"] == 5


def test_scoring_writes_empty_outputs_when_no_registry_rows(tmp_path: Path) -> None:
    (tmp_path / "01_registry").mkdir(parents=True)
    write_csv_rows(tmp_path / "01_registry" / "news_registry.csv", ["document_id", "include_flag"], [])
    summary = run_filter_news_final_scoring(tmp_path, scorer=lambda dimension, row, text: {})
    assert summary["blocked"] is True
    assert (tmp_path / "06_ratings" / "news_final_full" / "news_final_document_scores.csv").exists()
    assert (tmp_path / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv").exists()


def test_score_package_check_recomputes_mda_totals(tmp_path: Path) -> None:
    results = tmp_path / "results"
    rows = []
    ratings = []
    for idx in range(149):
        doc_id = f"MDA_TEST_{idx:03d}"
        row = {
            "document_id": doc_id,
            "AR01": "3",
            "AR02": "3",
            "AR03": "3",
            "AR04": "3",
            "AR05": "3",
            "total_score_100": "50.0",
        }
        rows.append(row)
        for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
            ratings.append({"document_id": doc_id, "dimension_code": code})
    write_csv(results / "MDA" / "mda_final_document_scores.csv", rows)
    write_csv(results / "MDA" / "mda_final_ratings_long.csv", ratings)
    check_mda(results)
