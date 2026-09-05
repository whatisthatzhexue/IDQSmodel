import csv
from pathlib import Path

from run_weight_sensitivity import run_weight_sensitivity
from scoring_utils import MDA_DOCUMENT_HEADERS, NEWS_DOCUMENT_HEADERS, read_csv_rows, write_csv_rows


def _write_scores(root: Path) -> None:
    write_csv_rows(
        root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            _mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 5, 5, 3, 3, 2),
            _mda_row("MDA_B", "BBB", "Beta Foods", "2024", 3, 5, 5, 3, 2),
            _mda_row("MDA_C", "CCC", "Cedar Foods", "2024", 2, 3, 4, 3, 2),
            _mda_row("MDA_D", "DDD", "Delta Foods", "2024", 1, 2, 2, 3, 2),
            _mda_row("MDA_E", "EEE", "Echo Foods", "2024", 4, 1, 1, 3, 2),
        ],
    )
    write_csv_rows(
        root / "06_ratings" / "news_document_scores.csv",
        NEWS_DOCUMENT_HEADERS,
        [
            _news_row("NEWS_A", "AAA", "Alpha Foods", "2024-01-01", 5, 5, 3, 5, 2),
            _news_row("NEWS_B", "BBB", "Beta Foods", "2024-01-02", 4, 5, 3, 4, 2),
            _news_row("NEWS_C", "CCC", "Cedar Foods", "2024-01-03", 3, 3, 3, 3, 2),
            _news_row("NEWS_D", "DDD", "Delta Foods", "2024-01-04", 2, 2, 3, 2, 2),
            _news_row("NEWS_E", "EEE", "Echo Foods", "2024-01-05", 1, 1, 3, 1, 2),
        ],
    )


def test_weight_sensitivity_recalculates_scores_ranks_and_report(tmp_path):
    root = tmp_path / "06_scoring"
    _write_scores(root)

    summary = run_weight_sensitivity(root)

    mda_output = root / "06_ratings" / "weight_sensitivity" / "mda_weight_sensitivity_scores.csv"
    news_output = root / "06_ratings" / "weight_sensitivity" / "news_weight_sensitivity_scores.csv"
    report = root / "08_reports" / "weight_sensitivity_report.md"
    assert mda_output.exists()
    assert news_output.exists()
    assert report.exists()
    assert summary["mda"]["document_count"] == 5
    assert summary["news"]["document_count"] == 5

    mda_rows = read_csv_rows(mda_output)
    rows_by_id = {row["document_id"]: row for row in mda_rows}
    first = rows_by_id["MDA_A"]
    assert list(first.keys()) == [
        "document_id",
        "ticker",
        "company_name",
        "report_year",
        "AR01",
        "AR02",
        "AR03",
        "AR04",
        "AR05",
        "score_W1_baseline",
        "score_W2_main_recommended",
        "score_W3_conservative_financial",
        "rank_W1",
        "rank_W2",
        "rank_W3",
        "rank_change_W1_W2",
        "rank_change_W2_W3",
        "max_rank_change",
    ]
    assert float(first["score_W2_main_recommended"]) == 67.5
    assert float(rows_by_id["MDA_B"]["score_W2_main_recommended"]) == 75.0
    assert int(rows_by_id["MDA_B"]["rank_W2"]) == 1
    assert int(rows_by_id["MDA_D"]["max_rank_change"]) == 1

    text = report.read_text(encoding="utf-8")
    assert "# Weight Sensitivity Analysis" in text
    assert "W2_main_recommended" in text
    assert "possible redundancy" in text
    assert "low variance" in text
    assert "Most Weight-Sensitive Documents" in text
    assert summary["mda"]["recommendation"]["main_weight"] == "W2_main_recommended"


def test_weight_sensitivity_prefers_v2_full_when_available(tmp_path):
    root = tmp_path / "06_scoring"
    _write_scores(root)
    v2_dir = root / "06_ratings" / "mda_v2_full"
    write_csv_rows(
        v2_dir / "mda_v2_full_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [_mda_row("MDA_V2", "V22", "V2 Foods", "2025", 5, 5, 5, 5, 5)],
    )

    summary = run_weight_sensitivity(root)

    rows = read_csv_rows(root / "06_ratings" / "weight_sensitivity" / "mda_weight_sensitivity_scores.csv")
    assert [row["document_id"] for row in rows] == ["MDA_V2"]
    assert summary["mda"]["input_path"].endswith("mda_v2_full_document_scores.csv")


def test_weight_sensitivity_handles_empty_news_scores(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(root / "06_ratings" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 3, 3, 3, 3, 3)])
    write_csv_rows(root / "06_ratings" / "news_document_scores.csv", NEWS_DOCUMENT_HEADERS, [])

    summary = run_weight_sensitivity(root)

    news_output = root / "06_ratings" / "weight_sensitivity" / "news_weight_sensitivity_scores.csv"
    with news_output.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows == []
    assert summary["news"]["document_count"] == 0
    assert summary["news"]["recommendation"]["reason"] == "insufficient scored documents for correlation analysis"
    assert summary["news"]["low_variance_dimensions"] == []


def _mda_row(document_id: str, ticker: str, company: str, year: str, ar01: int, ar02: int, ar03: int, ar04: int, ar05: int) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update(
        {
            "document_id": document_id,
            "doc_type": "mda",
            "ticker": ticker,
            "company_name": company,
            "report_year": year,
            "dimension_count": "5",
            "AR01": str(ar01),
            "AR02": str(ar02),
            "AR03": str(ar03),
            "AR04": str(ar04),
            "AR05": str(ar05),
        }
    )
    return row


def _news_row(document_id: str, ticker: str, company: str, publish_date: str, n01: int, n02: int, n03: int, n04: int, n05: int) -> dict[str, str]:
    row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
    row.update(
        {
            "document_id": document_id,
            "doc_type": "news",
            "ticker": ticker,
            "company_name": company,
            "publish_date": publish_date,
            "dimension_count": "5",
            "N01": str(n01),
            "N02": str(n02),
            "N03": str(n03),
            "N04": str(n04),
            "N05": str(n05),
        }
    )
    return row
