import csv
from pathlib import Path

from build_real_news_registry import REAL_NEWS_REGISTRY_HEADERS, build_real_news_registry
from extract_real_news_texts import extract_real_news_texts
from rebuild_real_news_registry import rebuild_real_news_registry
from scoring_utils import read_csv_rows, write_csv_rows


def _write_raw_news(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["ticker", "company_name", "publish_date", "source_name", "title", "url", "article_text"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _long_article(prefix: str = "Alpha reported revenue growth.") -> str:
    return prefix + " " + ("The company said demand improved and margins stabilized in Malaysia. " * 6)


def test_build_real_news_registry_blocks_when_input_missing(tmp_path):
    root = tmp_path / "06_scoring"

    summary = build_real_news_registry(root=root)

    assert summary["status"] == "blocked"
    assert summary["included_news_rows"] == 0
    assert summary["synthetic_count"] == 0
    assert (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md").exists()
    blocker = (root / "PAPER_OUTPUT" / "00_status" / "real_news_missing_blocker.md").read_text(encoding="utf-8")
    assert "MD&A final is complete and frozen." in blocker
    assert "real full-text News data are missing" in blocker


def test_build_real_news_registry_includes_only_valid_full_text_and_marks_duplicates(tmp_path):
    root = tmp_path / "06_scoring"
    raw = root / "input_news" / "news_raw.csv"
    duplicate_text = _long_article()
    _write_raw_news(
        raw,
        [
            {
                "ticker": "AAA",
                "company_name": "Alpha Food",
                "publish_date": "2024-01-02",
                "source_name": "Reuters",
                "title": "Alpha expands",
                "url": "http://a.example",
                "article_text": duplicate_text,
            },
            {
                "ticker": "AAA",
                "company_name": "Alpha Food",
                "publish_date": "2024-01-02",
                "source_name": "Reuters",
                "title": "Alpha expands",
                "url": "http://dup.example",
                "article_text": duplicate_text,
            },
            {
                "ticker": "BBB",
                "company_name": "Beta Food",
                "publish_date": "bad-date",
                "source_name": "Reuters",
                "title": "Beta link",
                "url": "http://b.example",
                "article_text": "http://b.example",
            },
        ],
    )

    summary = build_real_news_registry(root=root)
    rows = read_csv_rows(root / "01_registry" / "news_registry.csv")

    assert summary["status"] == "success"
    assert summary["total_news_rows"] == 3
    assert summary["included_news_rows"] == 1
    assert summary["excluded_news_rows"] == 2
    assert summary["duplicate_count"] == 1
    assert summary["invalid_date_count"] == 1
    assert summary["short_text_count"] == 1
    assert rows[0]["document_id"].startswith("NEWS_AAA_20240102_")
    assert rows[0]["synthetic_flag"] == "false"
    assert rows[0]["include_flag"] == "Yes"
    assert rows[1]["exclude_reason"] == "duplicate"
    assert "invalid_publish_date" in rows[2]["exclude_reason"]


def test_extract_real_news_texts_writes_title_and_numbered_paragraphs(tmp_path):
    root = tmp_path / "06_scoring"
    raw = root / "input_news" / "news_raw.csv"
    article = "Alpha reported revenue growth.\n\nAdvertisement\n\nThe company said margins improved by 12% in 2024."
    _write_raw_news(
        raw,
        [
            {
                "ticker": "AAA",
                "company_name": "Alpha Food",
                "publish_date": "2024-01-02",
                "source_name": "Reuters",
                "title": "Alpha expands",
                "url": "http://a.example",
                "article_text": article,
            }
        ],
    )
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REAL_NEWS_REGISTRY_HEADERS,
        [
            {
                "document_id": "NEWS_AAA_20240102_ABCD1234",
                "doc_type": "news",
                "ticker": "AAA",
                "company_name": "Alpha Food",
                "publish_date": "2024-01-02",
                "source_name": "Reuters",
                "title": "Alpha expands",
                "url": "http://a.example",
                "file_path": "input_news/news_raw.csv",
                "article_text_hash": "hash",
                "text_length": str(len(article)),
                "synthetic_flag": "false",
                "include_flag": "Yes",
                "exclude_reason": "",
                "created_at": "2026-06-22T00:00:00+00:00",
            }
        ],
    )

    summary = extract_real_news_texts(root=root)
    text = (root / "02_extracted_text" / "news_clean" / "NEWS_AAA_20240102_ABCD1234.txt").read_text(encoding="utf-8")

    assert summary["status"] == "success"
    assert summary["cleaned_news_count"] == 1
    assert summary["failed_cleaning_count"] == 0
    assert "[TITLE]\nAlpha expands" in text
    assert "[NEWS_P001]\nAlpha reported revenue growth." in text
    assert "[NEWS_P002]\nThe company said margins improved by 12% in 2024." in text
    assert "Advertisement" not in text


def test_rebuild_real_news_registry_uses_alias_relevance_not_ticker_only(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        ["document_id", "doc_type", "include_flag", "stock_code", "ticker", "company_name", "report_year", "text_path"],
        [
            {
                "document_id": "MDA_3A_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "0012",
                "ticker": "3A",
                "company_name": "Three-A Resources Berhad",
                "report_year": "2024",
                "text_path": "02_extracted_text/mda/MDA_3A_2024.txt",
            },
            {
                "document_id": "MDA_DLADY_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "3026",
                "ticker": "DLADY",
                "company_name": "Dutch Lady Milk Industries Berhad",
                "report_year": "2024",
                "text_path": "02_extracted_text/mda/MDA_DLADY_2024.txt",
            },
        ],
    )
    source = root / "input_news" / "mixed_news.csv"
    _write_raw_news(
        source,
        [
            {
                "ticker": "3A",
                "company_name": "Three-A Resources Berhad",
                "publish_date": "2024-01-02",
                "source_name": "Newswire",
                "title": "3A road project opens",
                "url": "http://example.com/3a-road",
                "article_text": _long_article("The 3A route opened after heavy rain. This article is about a public road project."),
            },
            {
                "ticker": "DLADY",
                "company_name": "Dutch Lady Milk Industries Berhad",
                "publish_date": "2024-03-04",
                "source_name": "Business Wire",
                "title": "Dutch Lady expands yoghurt output",
                "url": "http://example.com/dlady",
                "article_text": _long_article("Dutch Lady Milk Industries Berhad expanded production and reported stronger dairy demand."),
            },
        ],
    )

    summary = rebuild_real_news_registry(root=root)
    audit_rows = read_csv_rows(root / "08_reports" / "news_relevance_audit.csv")
    included = read_csv_rows(root / "01_registry" / "news_registry_rebuilt.csv")

    assert summary["news_registry_total"] == 2
    assert summary["news_real_company_news_count"] == 1
    assert summary["news_included_count"] == 1
    assert any(row["relevance_status"] == "irrelevant_general_news" for row in audit_rows)
    assert included[0]["ticker"] == "DLADY"
    assert included[0]["include_flag"] == "Yes"


def test_rebuild_real_news_registry_does_not_treat_mda_text_as_news(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        ["document_id", "doc_type", "include_flag", "stock_code", "ticker", "company_name", "report_year", "text_path"],
        [
            {
                "document_id": "MDA_3A_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "0012",
                "ticker": "3A",
                "company_name": "Three-A Resources Berhad",
                "report_year": "2024",
                "text_path": "02_extracted_text/mda/MDA_3A_2024.txt",
            }
        ],
    )
    mda_text = root / "02_extracted_text" / "mda" / "MDA_3A_2024.txt"
    mda_text.parent.mkdir(parents=True)
    mda_text.write_text(
        "[MDA_P001]\nThree-A Resources Berhad reported revenue and profit growth in its management discussion.\n"
        * 20,
        encoding="utf-8",
    )

    summary = rebuild_real_news_registry(root=root)

    assert summary["news_registry_total"] == 0
    assert summary["news_included_count"] == 0


def test_rebuild_real_news_registry_excludes_preextracted_news_text_without_metadata(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        ["document_id", "doc_type", "include_flag", "stock_code", "ticker", "company_name", "report_year", "text_path"],
        [
            {
                "document_id": "MDA_DLADY_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "3026",
                "ticker": "DLADY",
                "company_name": "Dutch Lady Milk Industries Berhad",
                "report_year": "2024",
                "text_path": "02_extracted_text/mda/MDA_DLADY_2024.txt",
            }
        ],
    )
    news_text = root / "02_extracted_text" / "news" / "NEWS_DLADY_20240101_ABCD1234.txt"
    news_text.parent.mkdir(parents=True)
    news_text.write_text("Dutch Lady Milk Industries Berhad reported stronger dairy demand. " * 20, encoding="utf-8")

    summary = rebuild_real_news_registry(root=root)
    audit_rows = read_csv_rows(root / "08_reports" / "news_relevance_audit.csv")

    assert summary["news_registry_total"] == 1
    assert summary["news_included_count"] == 0
    assert audit_rows[0]["relevance_status"] == "metadata_only"
