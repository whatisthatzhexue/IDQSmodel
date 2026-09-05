from __future__ import annotations

import json
from pathlib import Path

from run_statistical_stability_analysis import run_statistical_stability_analysis
from scoring_utils import MDA_DOCUMENT_HEADERS, NEWS_DOCUMENT_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows
from stability_weight_sensitivity import pearson_correlation, spearman_correlation, run_weight_stability
from validate_stability_outputs import validate_stability_outputs


def test_statistical_stability_analysis_runs_without_rewriting_inputs(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_scores(root)
    _write_news_scores(root)
    _write_repeat_runs(root)
    _write_version_inputs(root)
    _write_structure_inputs(root)
    _write_news_registry(root)
    before = (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8")

    summary = run_statistical_stability_analysis(root=root, make_plots=False)

    assert (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8") == before
    for subdir in ["inputs", "outputs", "reports", "plots", "logs"]:
        assert (root / "11_stability_analysis" / subdir).is_dir()
    assert summary["mda_weight_sensitivity"]["status"] == "success"
    assert summary["news_weight_sensitivity"]["status"] == "success"
    assert summary["repeatability"]["mda"]["status"] == "success"
    assert summary["version_comparison"]["mda"]["status"] == "success"
    assert summary["structure_comparison"]["mda"]["status"] == "success"
    assert summary["news_aggregation"]["status"] == "success"
    assert (root / "11_stability_analysis" / "reports" / "statistical_stability_report.md").exists()

    high_volatility = read_csv_rows(root / "11_stability_analysis" / "outputs" / "high_volatility_cases.csv")
    assert any(row["issue_type"] == "version_comparison_abs_diff_gt_10" for row in high_volatility)
    assert any(row["issue_type"] == "news_aggregation_score_range_gt_10" for row in high_volatility)

    validation = validate_stability_outputs(root=root)
    assert validation["passed"], validation


def test_weight_stability_outputs_expected_fields_and_flags_sensitive_cases(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_scores(root)

    summary = run_weight_stability(root=root, include_mda=True, include_news=False, make_plots=False)

    assert summary["mda"]["document_count"] == 4
    output_rows = read_csv_rows(root / "11_stability_analysis" / "outputs" / "mda_weight_sensitivity_scores.csv")
    assert list(output_rows[0].keys()) == [
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
        "score_change_W1_W2",
        "score_change_W2_W3",
        "max_score_change",
    ]
    assert "recommendation" in summary["mda"]
    assert any(item["dimension"] == "AR05" for item in summary["mda"]["low_variance_dimensions"])


def test_correlation_helpers_are_statistically_bounded():
    assert pearson_correlation([1, 2, 3], [1, 2, 3]) == 1.0
    assert pearson_correlation([1, 2, 3], [3, 2, 1]) == -1.0
    assert spearman_correlation([10, 20, 30], [1, 2, 3]) == 1.0
    assert -1.0 <= spearman_correlation([10, 20, 30], [2, 1, 3]) <= 1.0


def test_missing_news_inputs_generate_missing_news_report(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_scores(root)

    summary = run_statistical_stability_analysis(root=root, include_news=True, make_plots=False)

    assert summary["news_weight_sensitivity"]["status"] == "missing-news"
    assert summary["news_aggregation"]["status"] == "missing-news"
    report = (root / "11_stability_analysis" / "reports" / "statistical_stability_report.md").read_text(encoding="utf-8")
    assert "missing-news" in report
    assert read_csv_rows(root / "11_stability_analysis" / "outputs" / "news_weight_sensitivity_scores.csv") == []
    validation = validate_stability_outputs(root=root)
    assert validation["passed"], validation


def test_statistical_stability_prefers_final_news_dataset_over_legacy_placeholder(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_scores(root)
    write_csv_rows(
        root / "06_ratings" / "news_document_scores.csv",
        NEWS_DOCUMENT_HEADERS,
        [_news_row("NEWS_PLACEHOLDER_SOURCE_2024", "NEWSMIN", "Minimal News Placeholder Berhad", "2024-06-01", "Synthetic Wire", "Placeholder note", 3, 3, 3, 3, 3, 50)],
    )
    write_csv_rows(
        root / "FINAL_OUTPUT" / "news_final_dataset.csv",
        NEWS_DOCUMENT_HEADERS,
        [
            _news_row("NEWS_REAL_A", "AAA", "Alpha Foods", "2024-03-01", "Business Source", "Alpha Foods expands capacity", 5, 4, 4, 5, 4, 86),
            _news_row("NEWS_REAL_B", "BBB", "Beta Foods", "2024-04-01", "Business Source", "Beta Foods reports margins", 4, 4, 4, 4, 4, 75),
        ],
    )
    _write_news_registry(root)

    summary = run_statistical_stability_analysis(root=root, include_mda=False, include_news=True, include_repeatability=False, include_version_comparison=False, include_structure_comparison=False, make_plots=False)

    assert summary["news_weight_sensitivity"]["input_path"].endswith("FINAL_OUTPUT/news_final_dataset.csv")
    output_rows = read_csv_rows(root / "11_stability_analysis" / "outputs" / "news_weight_sensitivity_scores.csv")
    assert {row["document_id"] for row in output_rows} == {"NEWS_REAL_A", "NEWS_REAL_B"}
    assert "NEWS_PLACEHOLDER_SOURCE_2024" not in {row["document_id"] for row in output_rows}


def test_high_volatility_issue_types_distinguish_repeat_and_version_dimension_shifts(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_scores(root)
    _write_news_scores(root)
    _write_repeat_runs(root)
    _write_version_inputs(root)
    _write_structure_inputs(root)
    _write_news_registry(root)
    write_csv_rows(root / "06_ratings" / "repeat_runs" / "mda_repeat_run_2_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 4, 2, 2, 3, 3, 47)])

    run_statistical_stability_analysis(root=root, make_plots=False)

    high_volatility = read_csv_rows(root / "11_stability_analysis" / "outputs" / "high_volatility_cases.csv")
    ar02_shift_issue_types = {
        row["issue_type"]
        for row in high_volatility
        if row["document_id"] == "MDA_A" and row["doc_type"] == "mda" and row["dimension_affected"] == "AR02"
    }
    assert "repeat_run_max_dimension_diff_ge_2" in ar02_shift_issue_types
    assert "version_comparison_max_dimension_diff_ge_2" in ar02_shift_issue_types
    validation = validate_stability_outputs(root=root)
    assert validation["passed"], validation


def test_validate_stability_outputs_rejects_corrupt_correlation(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_scores(root)
    run_statistical_stability_analysis(root=root, include_news=False, make_plots=False)
    metrics_path = root / "11_stability_analysis" / "outputs" / "correlation_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics.append({"analysis": "bad", "metric": "pearson", "value": 1.5})
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

    validation = validate_stability_outputs(root=root)

    assert not validation["passed"]
    assert any("correlation_out_of_bounds" in issue for issue in validation["issues"])


def _write_mda_scores(root: Path) -> None:
    write_csv_rows(
        root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            _mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 5, 5, 2, 3, 3, 70),
            _mda_row("MDA_B", "BBB", "Beta Foods", "2024", 2, 5, 5, 3, 3, 72),
            _mda_row("MDA_C", "CCC", "Cedar Foods", "2024", 3, 3, 3, 3, 3, 50),
            _mda_row("MDA_D", "DDD", "Delta Foods", "2024", 1, 1, 2, 2, 3, 22),
        ],
    )


def _write_news_scores(root: Path) -> None:
    rows = [
        _news_row("NEWS_A1", "AAA", "Alpha Foods", "2024-01-01", "Source A", "Alpha strong", 5, 5, 4, 5, 4, 90),
        _news_row("NEWS_A2", "AAA", "Alpha Foods", "2024-02-01", "Source B", "Alpha weak", 2, 2, 3, 2, 4, 35),
        _news_row("NEWS_A3", "AAA", "Alpha Foods", "2024-03-01", "Source C", "Alpha neutral", 3, 4, 3, 4, 4, 62),
        _news_row("NEWS_B1", "BBB", "Beta Foods", "2024-01-15", "Source A", "Beta stable", 4, 5, 4, 4, 4, 78),
        _news_row("NEWS_B2", "BBB", "Beta Foods", "2024-02-15", "Source B", "Beta stable 2", 4, 4, 4, 4, 4, 75),
    ]
    write_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv", NEWS_DOCUMENT_HEADERS, rows)
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, rows)


def _write_news_registry(root: Path) -> None:
    rows = []
    source_rows = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv") or read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    for row in source_rows:
        registry_row = {field: "" for field in REGISTRY_HEADERS["news"]}
        registry_row.update(
            {
                "document_id": row["document_id"],
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": row["company_name"],
                "ticker": row["ticker"],
                "source_name": row["source_name"],
                "publish_date": row["publish_date"],
                "title": row["title"],
            }
        )
        rows.append(registry_row)
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], rows)


def _write_repeat_runs(root: Path) -> None:
    write_csv_rows(root / "06_ratings" / "repeat_runs" / "mda_repeat_run_1_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 5, 5, 2, 3, 3, 70)])
    write_csv_rows(root / "06_ratings" / "repeat_runs" / "mda_repeat_run_2_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 4, 4, 2, 3, 3, 62)])
    write_csv_rows(root / "06_ratings" / "repeat_runs" / "news_repeat_run_1_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_news_row("NEWS_A1", "AAA", "Alpha Foods", "2024-01-01", "Source A", "Alpha strong", 5, 5, 4, 5, 4, 90)])
    write_csv_rows(root / "06_ratings" / "repeat_runs" / "news_repeat_run_2_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_news_row("NEWS_A1", "AAA", "Alpha Foods", "2024-01-01", "Source A", "Alpha strong", 5, 4, 4, 4, 4, 82)])


def _write_version_inputs(root: Path) -> None:
    write_csv_rows(root / "archive" / "v1_initial_mda_scoring" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 5, 5, 2, 3, 3, 70)])
    write_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 3, 2, 5, 4, 3, 47)])


def _write_structure_inputs(root: Path) -> None:
    write_csv_rows(root / "06_ratings" / "document_level" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 5, 5, 2, 3, 3, 70)])
    write_csv_rows(root / "06_ratings" / "single_dimension" / "mda_document_scores.csv", MDA_DOCUMENT_HEADERS, [_mda_row("MDA_A", "AAA", "Alpha Foods", "2024", 4, 4, 3, 3, 3, 65)])
    write_csv_rows(root / "06_ratings" / "document_level" / "news_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_news_row("NEWS_A1", "AAA", "Alpha Foods", "2024-01-01", "Source A", "Alpha strong", 5, 5, 4, 5, 4, 90)])
    write_csv_rows(root / "06_ratings" / "single_dimension" / "news_document_scores.csv", NEWS_DOCUMENT_HEADERS, [_news_row("NEWS_A1", "AAA", "Alpha Foods", "2024-01-01", "Source A", "Alpha strong", 5, 4, 4, 4, 4, 82)])


def _mda_row(document_id: str, ticker: str, company: str, year: str, ar01: int, ar02: int, ar03: int, ar04: int, ar05: int, total: int) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "mda", "ticker": ticker, "company_name": company, "report_year": year, "dimension_count": "5", "total_score_100": str(total), "AR01": str(ar01), "AR02": str(ar02), "AR03": str(ar03), "AR04": str(ar04), "AR05": str(ar05)})
    return row


def _news_row(document_id: str, ticker: str, company: str, publish_date: str, source_name: str, title: str, n01: int, n02: int, n03: int, n04: int, n05: int, total: int) -> dict[str, str]:
    row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "news", "ticker": ticker, "company_name": company, "source_name": source_name, "publish_date": publish_date, "title": title, "dimension_count": "5", "total_score_100": str(total), "N01": str(n01), "N02": str(n02), "N03": str(n03), "N04": str(n04), "N05": str(n05)})
    return row
