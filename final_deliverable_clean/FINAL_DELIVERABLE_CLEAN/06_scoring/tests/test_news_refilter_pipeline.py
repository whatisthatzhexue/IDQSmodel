from __future__ import annotations

import json
from pathlib import Path

from audit_filter_news_relevance import audit_filter_news_relevance
from build_news_registry_from_filter_news import build_news_registry_from_filter_news
from extract_filter_news_texts import extract_filter_news_texts
from filter_news_common import RELEVANCE_HEADERS, raw_row_key
from paper_utils import news_registry_path, read_news_registry_rows
from run_filter_news_final_scoring import run_filter_news_final_scoring
from run_final_data_freeze import run_final_data_freeze
from scoring_utils import NEWS_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, read_csv_rows, write_csv_rows


def _seed_mda_registry(root: Path) -> None:
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        [
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
        [
            {
                "document_id": "MDA_5102_GCB_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "5102",
                "ticker": "GCB",
                "company_name": "Guan Chong Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda/MDA_5102_GCB_2024.txt",
                "source_path": "source.pdf",
                "source_file_sha256": "abc",
                "notes": "",
            },
            {
                "document_id": "MDA_5202_MSM_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "5202",
                "ticker": "MSM",
                "company_name": "MSM Malaysia Holdings Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda/MDA_5202_MSM_2024.txt",
                "source_path": "source.pdf",
                "source_file_sha256": "def",
                "notes": "",
            },
        ],
    )


def _long_text(prefix: str) -> str:
    return prefix + " " + (
        "The company reported revenue growth, margin pressure, product demand, factory capacity, "
        "supply costs, exports, management outlook, and F&B market risks in Malaysia. "
        * 4
    )


def _seed_raw_filter_news(root: Path) -> None:
    shared_text = _long_text(
        "Guan Chong Berhad and MSM Malaysia Holdings Berhad discussed cocoa and sugar supply conditions."
    )
    industry_text = _long_text(
        "Malaysia F&B manufacturers reported demand recovery and higher raw material costs."
    )
    write_csv_rows(
        root / "input_news" / "news_raw_from_filter_news.csv",
        [
            "ticker",
            "company_name",
            "publish_date",
            "source_name",
            "title",
            "url",
            "article_text",
            "original_file",
            "original_file_hash",
            "import_status",
            "raw_source_row",
        ],
        [
            {
                "ticker": "GCB",
                "company_name": "Guan Chong Berhad",
                "publish_date": "2024-03-04",
                "source_name": "The Star",
                "title": "Guan Chong and MSM discuss supply outlook",
                "url": "https://example.com/shared-gcb",
                "article_text": shared_text,
                "original_file": "Filter news/GCB/news.csv",
                "original_file_hash": "hash-gcb",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "MSM",
                "company_name": "MSM Malaysia Holdings Berhad",
                "publish_date": "2024-03-04",
                "source_name": "The Star",
                "title": "Guan Chong and MSM discuss supply outlook",
                "url": "https://example.com/shared-msm",
                "article_text": shared_text,
                "original_file": "Filter news/MSM/news.csv",
                "original_file_hash": "hash-msm",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "GCB",
                "company_name": "Guan Chong Berhad",
                "publish_date": "2024-03-04",
                "source_name": "The Star",
                "title": "Guan Chong and MSM discuss supply outlook duplicate",
                "url": "https://example.com/shared-gcb-dup",
                "article_text": shared_text,
                "original_file": "Filter news/GCB/news.csv",
                "original_file_hash": "hash-gcb",
                "import_status": "imported",
                "raw_source_row": "2",
            },
            {
                "ticker": "GCB",
                "company_name": "Guan Chong Berhad",
                "publish_date": "2024-05-01",
                "source_name": "The Star",
                "title": "F&B sector demand recovery broadens",
                "url": "https://example.com/industry",
                "article_text": industry_text,
                "original_file": "Filter news/GCB/news.csv",
                "original_file_hash": "hash-gcb",
                "import_status": "imported",
                "raw_source_row": "3",
            },
        ],
    )


def _write_raw_filter_news(root: Path, rows: list[dict[str, str]]) -> None:
    write_csv_rows(
        root / "input_news" / "news_raw_from_filter_news.csv",
        [
            "ticker",
            "company_name",
            "publish_date",
            "source_name",
            "title",
            "url",
            "article_text",
            "original_file",
            "original_file_hash",
            "import_status",
            "raw_source_row",
        ],
        rows,
    )


def test_empty_rebuilt_registry_does_not_override_nonempty_news_registry(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        ["document_id", "include_flag", "synthetic_flag", "title", "source_name", "publish_date"],
        [
            {
                "document_id": "NEWS_GCB_2024_03_04_REAL",
                "include_flag": "Yes",
                "synthetic_flag": "false",
                "title": "Guan Chong result",
                "source_name": "The Star",
                "publish_date": "2024-03-04",
            }
        ],
    )
    write_csv_rows(root / "01_registry" / "news_registry_rebuilt.csv", ["document_id", "include_flag"], [])

    assert news_registry_path(root).name == "news_registry.csv"
    assert [row["document_id"] for row in read_news_registry_rows(root)] == ["NEWS_GCB_2024_03_04_REAL"]
    report = root / "08_reports" / "news_registry_selection_report.md"
    assert report.exists()
    assert "fallback" in report.read_text(encoding="utf-8").lower()


def test_manual_preserve_mode_keeps_manual_candidates_and_dedupes_per_ticker(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    _seed_mda_registry(root)
    _seed_raw_filter_news(root)

    summary = audit_filter_news_relevance(root, mode="manual_preserve")
    rows = read_csv_rows(root / "08_reports" / "filter_news_relevance_audit.csv")

    assert summary["filter_mode"] == "manual_preserve"
    industry = next(row for row in rows if row["relevance_status"] == "no_target_alias_industry_news")
    assert industry["include_flag"] == "Manual"
    assert industry["main_model_include_flag"] == "No"
    included = [row for row in rows if row["main_model_include_flag"] == "Yes"]
    assert {row["ticker"] for row in included} == {"GCB", "MSM"}
    assert all(row["shared_article_flag"] == "1" for row in included)
    duplicate = next(row for row in rows if row["relevance_status"] == "duplicate")
    assert duplicate["ticker"] == "GCB"
    assert duplicate["main_model_include_flag"] == "No"


def test_registry_builder_uses_broad_include_flag_not_only_strict_main_model(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    raw_rows = [
        {
            "ticker": "GCB",
            "company_name": "Guan Chong Berhad",
            "publish_date": "2024-03-04",
            "source_name": "The Star",
            "title": "Guan Chong reports stronger cocoa demand",
            "url": "https://example.com/gcb",
            "article_text": _long_text("Guan Chong Berhad reported stronger cocoa demand and factory capacity growth."),
            "original_file": "Filter news/GCB/news.csv",
            "original_file_hash": "hash-gcb",
            "import_status": "imported",
            "raw_source_row": "1",
        },
        {
            "ticker": "MSM",
            "company_name": "MSM Malaysia Holdings Berhad",
            "publish_date": "2024-05-05",
            "source_name": "The Star",
            "title": "MSM appoints new acting group CEO",
            "url": "https://example.com/msm",
            "article_text": _long_text("MSM Malaysia Holdings Berhad appointed a new acting group CEO and discussed management outlook."),
            "original_file": "Filter news/MSM/news.csv",
            "original_file_hash": "hash-msm",
            "import_status": "imported",
            "raw_source_row": "1",
        },
        {
            "ticker": "AJI",
            "company_name": "Ajinomoto Malaysia Berhad",
            "publish_date": "2024-06-01",
            "source_name": "Sports Daily",
            "title": "Ajinomoto Stadium hosts football final",
            "url": "https://example.com/aji-stadium",
            "article_text": _long_text("Ajinomoto Stadium hosted a football final without Ajinomoto Malaysia company operations."),
            "original_file": "Filter news/AJI/news.csv",
            "original_file_hash": "hash-aji",
            "import_status": "imported",
            "raw_source_row": "1",
        },
    ]
    _write_raw_filter_news(root, raw_rows)
    audit_rows = []
    for idx, raw in enumerate(raw_rows, start=1):
        row = {field: "" for field in [*RELEVANCE_HEADERS, "broad_include_flag", "strict_company_focus_flag"]}
        row.update(
            {
                "document_id": f"NEWS_CANDIDATE_{idx:06d}",
                "ticker": raw["ticker"],
                "company_name": raw["company_name"],
                "title": raw["title"],
                "source_name": raw["source_name"],
                "publish_date": raw["publish_date"],
                "url": raw["url"],
                "text_length": str(len(raw["article_text"])),
                "raw_row_key": raw_row_key(raw, idx),
                "filter_mode": "manual_preserve",
                "filter_version": "manual_preserve_v1",
            }
        )
        if raw["ticker"] == "GCB":
            row.update(
                {
                    "relevance_status": "high_confidence_company_news",
                    "include_flag": "Yes",
                    "main_model_include_flag": "Yes",
                    "broad_include_flag": "Yes",
                    "strict_company_focus_flag": "Yes",
                    "confidence_tier": "high",
                }
            )
        elif raw["ticker"] == "MSM":
            row.update(
                {
                    "relevance_status": "low_confidence_company_news",
                    "include_flag": "Manual",
                    "main_model_include_flag": "No",
                    "broad_include_flag": "Yes",
                    "strict_company_focus_flag": "No",
                    "confidence_tier": "low",
                    "needs_manual_review": "Yes",
                    "review_reason": "manual_preserve_validated_broad_sample",
                }
            )
        else:
            row.update(
                {
                    "relevance_status": "no_target_alias_industry_news",
                    "include_flag": "Manual",
                    "main_model_include_flag": "No",
                    "broad_include_flag": "No",
                    "strict_company_focus_flag": "No",
                    "confidence_tier": "none",
                    "needs_manual_review": "Yes",
                    "review_reason": "venue_not_company_news",
                }
            )
        audit_rows.append(row)
    write_csv_rows(root / "08_reports" / "filter_news_relevance_audit.csv", [*RELEVANCE_HEADERS, "broad_include_flag", "strict_company_focus_flag"], audit_rows)

    summary = build_news_registry_from_filter_news(root)

    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    assert summary["registry_selection_flag"] == "broad_include_flag"
    assert summary["main_model_include_rows"] == 1
    assert summary["broad_include_rows"] == 2
    assert len(registry_rows) == 2
    assert {row["ticker"] for row in registry_rows} == {"GCB", "MSM"}


def test_manual_preserve_routes_stadium_perdana_and_csr_false_positives_to_review(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    _seed_mda_registry(root)
    _write_raw_filter_news(
        root,
        [
            {
                "ticker": "AJI",
                "company_name": "Ajinomoto Malaysia Berhad",
                "publish_date": "2024-01-10",
                "source_name": "Sports Daily",
                "title": "Ajinomoto Stadium hosts community football final",
                "url": "https://example.com/ajinomoto-stadium",
                "article_text": _long_text(
                    "Ajinomoto Stadium hosted a football final with concerts, food stalls, and visitors, but the story was about the venue and event logistics rather than Ajinomoto Malaysia operations."
                ),
                "original_file": "Filter news/AJI/news.csv",
                "original_file_hash": "hash-aji",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "PPB",
                "company_name": "PPB Group Berhad",
                "publish_date": "2024-02-15",
                "source_name": "The Star",
                "title": "Perdana Petroleum secures new work order",
                "url": "https://example.com/perdana-petroleum",
                "article_text": _long_text(
                    "Perdana Petroleum Berhad secured a new work order for offshore support vessels. The report discusses oil and gas operations and Perdana Petroleum management, not PPB Group Berhad."
                ),
                "original_file": "Filter news/PPB/news.csv",
                "original_file_hash": "hash-ppb",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "NESTLE",
                "company_name": "Nestle Malaysia Berhad",
                "publish_date": "2024-04-23",
                "source_name": "The Star",
                "title": "Home repairs for Orang Asli",
                "url": "https://example.com/home-repairs",
                "article_text": _long_text(
                    "The scope of home repairs included volunteer work. Nestle Malaysia was named once as a CSR participant in a lifestyle and community article, with no financial, product, expansion, market-performance, regulatory, or operational analysis."
                ),
                "original_file": "Filter news/NESTLE/news.csv",
                "original_file_hash": "hash-nestle",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "PPB",
                "company_name": "PPB Group Berhad",
                "publish_date": "2024-05-20",
                "source_name": "Business Wire",
                "title": "PPB Group reports flour and grains margin update",
                "url": "https://example.com/ppb-group",
                "article_text": _long_text(
                    "PPB Group Berhad reported flour and grains revenue, margins, management outlook, product demand, and commodity cost pressure in Malaysia."
                ),
                "original_file": "Filter news/PPB/news.csv",
                "original_file_hash": "hash-ppb",
                "import_status": "imported",
                "raw_source_row": "2",
            },
        ],
    )

    audit_filter_news_relevance(root, mode="manual_preserve")
    rows = read_csv_rows(root / "08_reports" / "filter_news_relevance_audit.csv")
    by_title = {row["title"]: row for row in rows}

    stadium = by_title["Ajinomoto Stadium hosts community football final"]
    assert stadium["main_model_include_flag"] == "No"
    assert stadium["needs_manual_review"] == "Yes"
    assert "venue" in stadium["review_reason"]

    perdana = by_title["Perdana Petroleum secures new work order"]
    assert perdana["relevance_status"] == "wrong_company"
    assert perdana["main_model_include_flag"] == "No"

    csr = by_title["Home repairs for Orang Asli"]
    assert csr["confidence_tier"] in {"low", "medium", "none"}
    assert csr["high_confidence_include_flag"] == "No"
    assert csr["needs_manual_review"] == "Yes"

    ppb = by_title["PPB Group reports flour and grains margin update"]
    assert ppb["confidence_tier"] == "high"
    assert ppb["main_model_include_flag"] == "Yes"

    assert read_csv_rows(root / "08_reports" / "news_possible_false_positives.csv")
    assert read_csv_rows(root / "08_reports" / "news_wrong_company_candidates.csv")


def test_company_focus_gate_writes_suspect_queue_and_strict_subset(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    _seed_mda_registry(root)
    _write_raw_filter_news(
        root,
        [
            {
                "ticker": "AJI",
                "company_name": "Ajinomoto Malaysia Berhad",
                "publish_date": "2024-03-01",
                "source_name": "Sports Daily",
                "title": "Tokyo football final thrills packed crowd",
                "url": "https://example.com/stadium-only",
                "article_text": _long_text(
                    "The match was played at Ajinomoto Stadium before thousands of fans. The report covered sport, venue logistics, supporters, and community entertainment with no Ajinomoto Malaysia business operations."
                ),
                "original_file": "Filter news/AJI/news.csv",
                "original_file_hash": "hash-aji",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "AJI",
                "company_name": "Ajinomoto Malaysia Berhad",
                "publish_date": "2024-03-02",
                "source_name": "Agency News",
                "title": "Creative agency wins regional marketing account",
                "url": "https://example.com/client-list",
                "article_text": _long_text(
                    "The article is about an advertising agency. Its client list includes Ajinomoto, several banks, and retailers, but gives no Ajinomoto Malaysia financial, product, regulatory, or operational update."
                ),
                "original_file": "Filter news/AJI/news.csv",
                "original_file_hash": "hash-aji",
                "import_status": "imported",
                "raw_source_row": "2",
            },
            {
                "ticker": "MSM",
                "company_name": "MSM Malaysia Holdings Berhad",
                "publish_date": "2024-03-03",
                "source_name": "The Star",
                "title": "FBM KLCI ends higher on selective buying",
                "url": "https://example.com/market-roundup",
                "article_text": _long_text(
                    "The stock market roundup listed gainers and losers including MSM Malaysia, PPB Group, Spritzer, banks, and utilities, without company-specific analysis of MSM operations."
                ),
                "original_file": "Filter news/MSM/news.csv",
                "original_file_hash": "hash-msm",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "NESTLE",
                "company_name": "Nestle Malaysia Berhad",
                "publish_date": "2024-03-04",
                "source_name": "Metro",
                "title": "School programme draws community support",
                "url": "https://example.com/sponsor-list",
                "article_text": _long_text(
                    "The community article described a school event and sponsor list naming Nestle Malaysia among many participants. It did not focus on Nestle Malaysia products, investment, finance, regulation, supply chain, or market performance."
                ),
                "original_file": "Filter news/NESTLE/news.csv",
                "original_file_hash": "hash-nestle",
                "import_status": "imported",
                "raw_source_row": "1",
            },
            {
                "ticker": "NESTLE",
                "company_name": "Nestle Malaysia Berhad",
                "publish_date": "2024-03-05",
                "source_name": "The Star",
                "title": "Nestle Malaysia expands product capacity",
                "url": "https://example.com/nestle-capacity",
                "article_text": _long_text(
                    "Nestle Malaysia Berhad said it expanded product capacity, invested in manufacturing lines, managed raw material costs, and expects stronger revenue from new product demand."
                ),
                "original_file": "Filter news/NESTLE/news.csv",
                "original_file_hash": "hash-nestle",
                "import_status": "imported",
                "raw_source_row": "2",
            },
        ],
    )

    audit_filter_news_relevance(root, mode="manual_preserve")

    focus_rows = read_csv_rows(root / "08_reports" / "news_company_focus_audit.csv")
    suspect_rows = read_csv_rows(root / "08_reports" / "news_suspect_not_company_focused.csv")
    subset_rows = read_csv_rows(root / "08_reports" / "news_company_focused_main_subset.csv")
    by_title = {row["title"]: row for row in focus_rows}

    assert by_title["Tokyo football final thrills packed crowd"]["suspect_not_company_focused"] == "Yes"
    assert by_title["Creative agency wins regional marketing account"]["suspect_not_company_focused"] == "Yes"
    assert by_title["FBM KLCI ends higher on selective buying"]["suspect_not_company_focused"] == "Yes"
    assert by_title["School programme draws community support"]["suspect_not_company_focused"] == "Yes"
    assert by_title["Nestle Malaysia expands product capacity"]["strict_main_sample_include"] == "Yes"
    assert {row["title"] for row in suspect_rows} >= {
        "Tokyo football final thrills packed crowd",
        "Creative agency wins regional marketing account",
        "FBM KLCI ends higher on selective buying",
        "School programme draws community support",
    }
    assert {row["title"] for row in subset_rows} == {"Nestle Malaysia expands product capacity"}


def test_registry_extraction_scoring_freeze_and_runtime_outputs(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    _seed_mda_registry(root)
    _seed_raw_filter_news(root)
    audit_filter_news_relevance(root, mode="manual_preserve")

    registry_summary = build_news_registry_from_filter_news(root)
    extraction_summary = extract_filter_news_texts(root)
    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    assert registry_summary["registry_rows"] == 2
    assert extraction_summary["clean_text_files_written"] == len(registry_rows)
    assert len(list((root / "02_extracted_text" / "news_clean").glob("*.txt"))) == len(registry_rows)

    def fake_scorer(dimension, row, text):
        assert "[METADATA]" in text
        assert "[NEWS_P001]" in text
        return {"score": 4, "evidence": "NEWS_P001", "reason": f"{dimension['code']} evidence is sufficient"}

    scoring = run_filter_news_final_scoring(root, scorer=fake_scorer, model="qwen3:8b")
    assert scoring["document_score_rows"] == len(registry_rows)
    assert scoring["ratings_long_rows"] == len(registry_rows) * 5

    rating_rows = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv")
    assert len(rating_rows) == scoring["document_score_rows"] * 5
    for field in ["model_name", "temperature", "seed", "num_ctx", "num_predict"]:
        assert rating_rows[0][field] != ""

    raw_payload = json.loads(next((root / "05_raw_model_outputs" / "news_final_full").glob("*_N01_raw.json")).read_text())
    runtime = raw_payload["runtime_config"]
    assert runtime["model_name"] == "qwen3:8b"
    assert runtime["llm_backend"] == "ollama"
    assert runtime["temperature"] == 0
    assert runtime["seed"] == 42
    assert runtime["num_ctx"] == 4096
    assert runtime["num_predict"] == 192
    assert runtime["think"] is False
    assert runtime["stream"] is False
    assert runtime["json_schema_mode"]
    assert runtime["scorebook_version"]
    assert runtime["prompt_version"]

    freeze = run_final_data_freeze(root)
    final_news = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert freeze["final_dataset_size"]["news"] == len(registry_rows)
    assert len(final_news) == len(registry_rows)
    assert metadata["final_dataset_size"]["news"] == len(registry_rows)
    assert metadata["news"]["extra"]["synthetic_count"] == 0
    assert all(row.get("synthetic_flag", "false") == "false" for row in registry_rows)
    assert (root / "08_reports" / "qwen3_8b_limitations_news.md").exists()
    assert (root / "08_reports" / "news_context_truncation_audit.csv").exists()
