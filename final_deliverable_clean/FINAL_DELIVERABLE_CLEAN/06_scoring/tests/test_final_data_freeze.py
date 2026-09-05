from __future__ import annotations

import json
from pathlib import Path

from rebuild_real_news_registry import REBUILT_HEADERS
from run_final_data_freeze import run_final_data_freeze
from scoring_utils import MDA_DOCUMENT_HEADERS, NEWS_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows


def test_final_data_freeze_builds_research_grade_outputs_when_all_hard_gates_pass(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)

    summary = run_final_data_freeze(root=root)

    final_dir = root / "FINAL_OUTPUT"
    assert summary["freeze_allowed"] is True
    assert (final_dir / "mda_final_dataset.csv").exists()
    assert (final_dir / "news_final_dataset.csv").exists()
    assert (final_dir / "cross_validation_final_summary.csv").exists()
    assert (final_dir / "final_quality_report.md").exists()
    assert (final_dir / "dataset_frozen_metadata.json").exists()

    mda_rows = read_csv_rows(final_dir / "mda_final_dataset.csv")
    news_rows = read_csv_rows(final_dir / "news_final_dataset.csv")
    assert [row["document_id"] for row in mda_rows] == ["MDA_GOOD_2024"]
    assert mda_rows[0]["low_confidence_flag"] == "true"
    assert [row["document_id"] for row in news_rows] == ["NEWS_GOOD_2024"]

    metadata = json.loads((final_dir / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert metadata["freeze_allowed"] is True
    assert metadata["mda"]["excluded_count"] == 0
    assert metadata["news"]["excluded_count"] == 0
    assert metadata["quality_gates"]["mda_stability_success_rate"]["passed"] is True
    assert metadata["quality_gates"]["system_level_error_rate"]["passed"] is True
    assert metadata["quality_gates"]["cross_validation_claim_links"]["passed"] is True
    assert metadata["quality_gates"]["news_exists"]["passed"] is True
    assert metadata["quality_gates"]["news_registry_real"]["passed"] is True
    assert metadata["quality_gates"]["cross_validation_no_systemic_conflict"]["passed"] is True

    report = (final_dir / "final_quality_report.md").read_text(encoding="utf-8")
    assert "research-grade reliability: YES" in report
    assert "empirical analysis: YES" in report
    assert "FINAL DATASET" in report


def test_final_data_freeze_filters_invalid_but_blocks_high_system_error_rate(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    _append_invalid_documents(root)

    summary = run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert summary["freeze_allowed"] is False
    assert metadata["quality_gates"]["system_level_error_rate"]["passed"] is False
    assert metadata["mda"]["excluded_count"] == 1
    assert metadata["news"]["excluded_count"] == 1


def test_final_data_freeze_blocks_when_news_missing_or_stability_below_threshold(tmp_path):
    root = tmp_path / "06_scoring"
    _write_blocked_fixture(root)

    summary = run_final_data_freeze(root=root)

    final_dir = root / "FINAL_OUTPUT"
    metadata = json.loads((final_dir / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert summary["freeze_allowed"] is False
    assert metadata["freeze_allowed"] is False
    assert metadata["quality_gates"]["mda_stability_success_rate"]["passed"] is False
    assert metadata["quality_gates"]["news_exists"]["passed"] is False
    assert metadata["final_dataset_size"]["mda"] == 1
    assert metadata["final_dataset_size"]["news"] == 0
    assert (final_dir / "mda_final_dataset.csv").exists()
    assert (final_dir / "news_final_dataset.csv").exists()
    report = (final_dir / "final_quality_report.md").read_text(encoding="utf-8")
    assert "research-grade reliability: NO" in report
    assert "News registry or News scores missing" in report


def test_final_data_freeze_uses_redefined_mda_stability_metrics_when_available(tmp_path):
    root = tmp_path / "06_scoring"
    _write_blocked_fixture(root)
    (root / "PAPER_OUTPUT" / "00_status").mkdir(parents=True)
    (root / "PAPER_OUTPUT" / "00_status" / "mda_stability_metrics_redefined.json").write_text(
        json.dumps(
            {
                "operational_completion_rate": 1.0,
                "operationally_complete_documents": 2,
                "expected_stability_documents": 2,
                "legacy_mda_stability_success_rate": 0.5,
                "operational_metrics_source": "mda_stability_ratings_long_complete",
            }
        ),
        encoding="utf-8",
    )

    run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert metadata["quality_gates"]["mda_stability_success_rate"]["passed"] is True
    assert metadata["quality_gates"]["mda_stability_success_rate"]["value"] == 1.0
    assert metadata["mda"]["extra"]["legacy_success_rate"] == 0.5
    assert metadata["mda"]["extra"]["source"] == "mda_stability_metrics_redefined"


def test_final_data_freeze_blocks_when_real_news_has_no_claim_links(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    _write_cross_validation(root, contradiction_count=0, support_count=0)

    summary = run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert summary["freeze_allowed"] is False
    assert metadata["quality_gates"]["cross_validation_claim_links"]["passed"] is False


def test_final_data_freeze_detects_systemic_cross_validation_conflict(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    _write_cross_validation(root, contradiction_count=4, support_count=0)

    summary = run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert summary["freeze_allowed"] is True
    assert metadata["freeze_allowed"] is True
    assert metadata["quality_gates"]["cross_validation_no_systemic_conflict"]["passed"] is False
    cross_rows = read_csv_rows(root / "FINAL_OUTPUT" / "cross_validation_final_summary.csv")
    assert cross_rows[0]["systemic_conflict_flag"] == "true"
    report = (root / "FINAL_OUTPUT" / "final_quality_report.md").read_text(encoding="utf-8")
    assert "advisory warnings: cross_validation_systemic_conflict_unresolved" in report
    assert "research-grade reliability: NO" in report
    assert "systemic conflict detected by freeze layer" in report
    assert "no systemic conflict detected by freeze layer" not in report


def test_final_data_freeze_writes_company_focused_news_dataset_from_focus_audit(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    news_scores = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    news_scores.append(_news_doc("NEWS_MARKET_ROUNDUP_2024", "AAA", "Alpha Foods", "2024-03-02", "Market Daily", "FBM KLCI ends higher", 70))
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, news_scores)
    news_ratings = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv")
    news_ratings.extend(_rating_rows("news", "NEWS_MARKET_ROUNDUP_2024", ["N01", "N02", "N03", "N04", "N05"]))
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", RATINGS_LONG_HEADERS, news_ratings)
    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    registry_rows.append(
        {
            "document_id": "NEWS_MARKET_ROUNDUP_2024",
            "doc_type": "news",
            "include_flag": "Yes",
            "company_name": "Alpha Foods",
            "ticker": "AAA",
            "source_name": "Market Daily",
            "publish_date": "2024-03-02",
            "title": "FBM KLCI ends higher",
            "notes": "synthetic_flag=false",
        }
    )
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], registry_rows)
    write_csv_rows(
        root / "08_reports" / "news_company_focus_audit.csv",
        ["document_id", "strict_main_sample_include", "suspect_not_company_focused", "company_focus_reason"],
        [
            {
                "document_id": "NEWS_GOOD_2024",
                "strict_main_sample_include": "Yes",
                "suspect_not_company_focused": "No",
                "company_focus_reason": "title_or_lead_company_operating_focus",
            },
            {
                "document_id": "NEWS_MARKET_ROUNDUP_2024",
                "strict_main_sample_include": "No",
                "suspect_not_company_focused": "Yes",
                "company_focus_reason": "market_roundup_incidental_mention",
            },
        ],
    )

    run_final_data_freeze(root=root)

    broad = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    focused = read_csv_rows(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert {row["document_id"] for row in broad} == {"NEWS_GOOD_2024", "NEWS_MARKET_ROUNDUP_2024"}
    assert [row["document_id"] for row in focused] == ["NEWS_GOOD_2024"]
    assert focused[0]["company_focus_reason"] == "title_or_lead_company_operating_focus"
    assert metadata["final_dataset_size"]["news_company_focused"] == 1
    assert metadata["news"]["company_focused_count"] == 1
    assert metadata["news"]["suspect_not_company_focused_count"] == 1


def test_final_data_freeze_writes_broad_dataset_and_focus_robustness_report(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    news_scores = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    news_scores.append(_news_doc("NEWS_MARKET_ROUNDUP_2024", "AAA", "Alpha Foods", "2024-03-02", "Market Daily", "FBM KLCI ends higher", 70))
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, news_scores)
    news_ratings = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv")
    news_ratings.extend(_rating_rows("news", "NEWS_MARKET_ROUNDUP_2024", ["N01", "N02", "N03", "N04", "N05"]))
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", RATINGS_LONG_HEADERS, news_ratings)
    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    registry_rows.append(
        {
            "document_id": "NEWS_MARKET_ROUNDUP_2024",
            "doc_type": "news",
            "include_flag": "Yes",
            "company_name": "Alpha Foods",
            "ticker": "AAA",
            "source_name": "Market Daily",
            "publish_date": "2024-03-02",
            "title": "FBM KLCI ends higher",
            "notes": "synthetic_flag=false",
        }
    )
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], registry_rows)
    write_csv_rows(
        root / "08_reports" / "news_company_focus_audit.csv",
        ["document_id", "strict_main_sample_include", "suspect_not_company_focused", "company_focus_reason"],
        [
            {
                "document_id": "NEWS_GOOD_2024",
                "strict_main_sample_include": "Yes",
                "suspect_not_company_focused": "No",
                "company_focus_reason": "title_or_lead_company_operating_focus",
            },
            {
                "document_id": "NEWS_MARKET_ROUNDUP_2024",
                "strict_main_sample_include": "No",
                "suspect_not_company_focused": "Yes",
                "company_focus_reason": "market_roundup_incidental_mention",
            },
        ],
    )

    run_final_data_freeze(root=root)

    broad = read_csv_rows(root / "FINAL_OUTPUT" / "news_broad_dataset.csv")
    final_news = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    robustness = read_csv_rows(root / "08_reports" / "news_broad_vs_company_focused_robustness.csv")
    report = (root / "08_reports" / "news_broad_vs_company_focused_robustness.md").read_text(encoding="utf-8")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert [row["document_id"] for row in broad] == [row["document_id"] for row in final_news]
    assert robustness[0]["broad_rows"] == "2"
    assert robustness[0]["company_focused_rows"] == "1"
    assert robustness[0]["suspect_or_manual_review_rows"] == "1"
    assert "News broad sample" in report
    assert metadata["final_outputs"]["news_broad_dataset"].endswith("news_broad_dataset.csv")
    assert metadata["news"]["broad_count"] == 2


def test_final_news_dataset_preserves_relevance_and_model_confidence_separately(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    news_scores = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    news_scores[0]["confidence_tier"] = "medium"
    news_scores[0]["relevance_status"] = "model_medium_confidence"
    news_scores[0]["needs_review"] = "No"
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, news_scores)
    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    registry_rows[0]["confidence_tier"] = "low"
    registry_rows[0]["relevance_status"] = "low_confidence_company_news"
    registry_rows[0]["needs_manual_review"] = "Yes"
    registry_rows[0]["review_reason"] = "weak_company_specific_context"
    registry_headers = [
        *REGISTRY_HEADERS["news"],
        "synthetic_flag",
        "relevance_status",
        "confidence_tier",
        "needs_manual_review",
        "review_reason",
    ]
    write_csv_rows(root / "01_registry" / "news_registry.csv", registry_headers, registry_rows)
    write_csv_rows(
        root / "08_reports" / "news_company_focus_audit.csv",
        [
            "document_id",
            "relevance_status",
            "confidence_tier",
            "company_focus_status",
            "suspect_not_company_focused",
            "company_focus_reason",
            "strict_main_sample_include",
        ],
        [
            {
                "document_id": "NEWS_GOOD_2024",
                "relevance_status": "low_confidence_company_news",
                "confidence_tier": "low",
                "company_focus_status": "suspect_not_company_focused",
                "suspect_not_company_focused": "Yes",
                "company_focus_reason": "market_roundup_incidental_mention",
                "strict_main_sample_include": "No",
            }
        ],
    )

    run_final_data_freeze(root=root)

    final_news = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    assert final_news[0]["confidence_tier"] == "medium"
    assert final_news[0]["model_confidence_tier"] == "medium"
    assert final_news[0]["relevance_confidence_tier"] == "low"
    assert final_news[0]["relevance_status_original"] == "low_confidence_company_news"
    assert final_news[0]["relevance_needs_manual_review"] == "Yes"
    assert final_news[0]["relevance_review_reason"] == "weak_company_specific_context"
    assert final_news[0]["strict_company_focus_flag"] == "No"
    assert final_news[0]["suspect_not_company_focused"] == "Yes"
    assert final_news[0]["company_focus_reason"] == "market_roundup_incidental_mention"
    assert final_news[0]["low_confidence_flag"] == "false"


def test_final_data_freeze_falls_back_when_rebuilt_news_registry_is_empty(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    write_csv_rows(root / "01_registry" / "news_registry_rebuilt.csv", REBUILT_HEADERS, [])

    summary = run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert summary["freeze_allowed"] is True
    assert metadata["final_dataset_size"]["news"] > 0
    assert metadata["quality_gates"]["news_exists"]["passed"] is True
    assert metadata["news"]["extra"]["registry_path"].endswith("news_registry.csv")
    assert metadata["cross_validation"]["claim_link_count"] > 0
    assert metadata["quality_gates"]["cross_validation_claim_links"]["passed"] is True
    assert read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv") != []
    report = (root / "08_reports" / "news_registry_selection_report.md").read_text(encoding="utf-8")
    assert "fallback" in report.lower()


def test_final_data_freeze_scope_word_in_news_text_is_not_scope_violation(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    news_ratings = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv")
    for row in news_ratings:
        if row["dimension_code"] == "N04":
            row["comment_short"] = "The article mentions the scope of home repairs but remains target-company news."
            row["evidence_text_short"] = "The scope of home repairs was described in a CSR paragraph."
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", RATINGS_LONG_HEADERS, news_ratings)

    run_final_data_freeze(root=root)

    final_news = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert [row["document_id"] for row in final_news] == ["NEWS_GOOD_2024"]
    assert metadata["news"]["excluded_count"] == 0


def test_final_data_freeze_scope_of_work_comment_is_not_scope_violation(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    news_ratings = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv")
    for row in news_ratings:
        if row["dimension_code"] == "N03":
            row["comment_short"] = "The source describes the scope of work for a factory services contract."
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", RATINGS_LONG_HEADERS, news_ratings)

    run_final_data_freeze(root=root)

    final_news = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert [row["document_id"] for row in final_news] == ["NEWS_GOOD_2024"]
    assert metadata["news"]["excluded_count"] == 0


def test_final_data_freeze_review_reasons_scope_violation_excludes_news(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    news_scores = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    news_scores[0]["review_reasons"] = "scope_violation"
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, news_scores)

    run_final_data_freeze(root=root)

    final_news = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert final_news == []
    assert metadata["news"]["excluded_count"] == 1
    assert metadata["news"]["excluded_documents"][0]["reasons"] == ["scope violation"]


def test_final_data_freeze_excludes_structured_wrong_company_news(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    news_scores = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    news_scores[0]["review_reasons"] = "wrong_company"
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, news_scores)

    run_final_data_freeze(root=root)

    final_news = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert final_news == []
    assert metadata["news"]["excluded_count"] == 1
    assert metadata["news"]["excluded_documents"][0]["reasons"] == ["scope violation"]


def test_final_data_freeze_labels_ar02_numeric_review_without_failure_wording(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    ratings_path = root / "06_ratings" / "mda_v2_full" / "mda_v2_full_ratings_long.csv"
    ratings = read_csv_rows(ratings_path)
    for row in ratings:
        if row["dimension_code"] == "AR02":
            row["numeric_check_path"] = "03_numeric_checks/mda/MDA_GOOD_2024_AR02.json"
    write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, ratings)

    run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert metadata["mda"]["numeric_review_flag_count"] == 0
    assert metadata["mda"]["confirmed_numeric_failure_count"] == 0
    assert metadata["mda"]["no_numeric_issue_count"] == 1
    assert metadata["reason_distribution"]["numeric_review_required"] == 0
    assert metadata["reason_distribution"]["confirmed_numeric_failure"] == 0
    assert "numeric failure" not in metadata["reason_distribution"]
    report = (root / "FINAL_OUTPUT" / "final_quality_report.md").read_text(encoding="utf-8")
    assert "numeric_review_flag_count: 0" in report
    assert "confirmed_numeric_failure_count: 0" in report
    assert "numeric failure" not in report


def test_final_data_freeze_counts_explicit_ar02_numeric_review_reason(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    ratings_path = root / "06_ratings" / "mda_v2_full" / "mda_v2_full_ratings_long.csv"
    ratings = read_csv_rows(ratings_path)
    for row in ratings:
        if row["dimension_code"] == "AR02":
            row["comment_short"] = "numeric_review_required: figures need manual checking"
    write_csv_rows(ratings_path, RATINGS_LONG_HEADERS, ratings)

    run_final_data_freeze(root=root)

    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert metadata["mda"]["numeric_review_flag_count"] == 1
    assert metadata["reason_distribution"]["numeric_review_required"] == 1


def test_final_data_freeze_prefers_repaired_mda_candidate_when_available(tmp_path):
    root = tmp_path / "06_scoring"
    _write_success_fixture(root)
    write_csv_rows(
        root / "06_ratings" / "mda_final_full" / "mda_final_document_scores_repaired.csv",
        MDA_DOCUMENT_HEADERS,
        [_mda_doc("MDA_REPAIRED_2024", "AAA", "Alpha Foods", "2024", 88)],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long_repaired.csv",
        RATINGS_LONG_HEADERS,
        _rating_rows("mda", "MDA_REPAIRED_2024", ["AR01", "AR02", "AR03", "AR04", "AR05"]),
    )

    run_final_data_freeze(root=root)

    mda_rows = read_csv_rows(root / "FINAL_OUTPUT" / "mda_final_dataset.csv")
    metadata = json.loads((root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").read_text(encoding="utf-8"))
    assert [row["document_id"] for row in mda_rows] == ["MDA_REPAIRED_2024"]
    assert mda_rows[0]["total_score_100"] == "88"
    assert metadata["mda"]["selected_input_label"] == "mda_final_full_repaired"


def _write_success_fixture(root: Path) -> None:
    write_csv_rows(
        root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            _mda_doc("MDA_GOOD_2024", "AAA", "Alpha Foods", "2024", 80),
        ],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_v2_full" / "mda_v2_full_ratings_long.csv",
        RATINGS_LONG_HEADERS,
        _rating_rows("mda", "MDA_GOOD_2024", ["AR01", "AR02", "AR03", "AR04", "AR05"], low_confidence_dimension="AR02"),
    )
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv",
        ["document_id"],
        [{"document_id": "MDA_GOOD_2024"}],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [_mda_doc("MDA_GOOD_2024", "AAA", "Alpha Foods", "2024", 80)],
    )
    write_csv_rows(
        root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv",
        NEWS_DOCUMENT_HEADERS,
        [_news_doc("NEWS_GOOD_2024", "AAA", "Alpha Foods", "2024-03-01", "Business Wire", "Alpha good", 76)],
    )
    write_csv_rows(
        root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv",
        RATINGS_LONG_HEADERS,
        _rating_rows("news", "NEWS_GOOD_2024", ["N01", "N02", "N03", "N04", "N05"]),
    )
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_GOOD_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Alpha Foods",
                "ticker": "AAA",
                "source_name": "Business Wire",
                "publish_date": "2024-03-01",
                "title": "Alpha good",
            }
        ],
    )
    _write_cross_validation(root, contradiction_count=0, support_count=3)


def _append_invalid_documents(root: Path) -> None:
    mda_scores = read_csv_rows(root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv")
    mda_scores.append(_mda_doc("MDA_BAD_2024", "BBB", "Beta Foods", "2024", 50))
    write_csv_rows(root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv", MDA_DOCUMENT_HEADERS, mda_scores)
    mda_ratings = read_csv_rows(root / "06_ratings" / "mda_v2_full" / "mda_v2_full_ratings_long.csv")
    mda_ratings.extend(_rating_rows("mda", "MDA_BAD_2024", ["AR01", "AR02", "AR03", "AR04", "AR05"], missing_evidence_dimension="AR03"))
    write_csv_rows(root / "06_ratings" / "mda_v2_full" / "mda_v2_full_ratings_long.csv", RATINGS_LONG_HEADERS, mda_ratings)

    news_scores = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    news_scores.append(_news_doc("NEWS_BAD_2024", "BBB", "Beta Foods", "2024-04-01", "Business Wire", "Beta bad", 40))
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, news_scores)
    news_ratings = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv")
    news_ratings.extend(_rating_rows("news", "NEWS_BAD_2024", ["N01", "N02", "N03", "N04", "N05"], schema_invalid_dimension="N02"))
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", RATINGS_LONG_HEADERS, news_ratings)
    registry_rows = read_csv_rows(root / "01_registry" / "news_registry.csv")
    registry_rows.append(
        {
            "document_id": "NEWS_BAD_2024",
            "doc_type": "news",
            "include_flag": "Yes",
            "company_name": "Beta Foods",
            "ticker": "BBB",
            "source_name": "Business Wire",
            "publish_date": "2024-04-01",
            "title": "Beta bad",
            "notes": "synthetic_flag=false",
        }
    )
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], registry_rows)


def _write_blocked_fixture(root: Path) -> None:
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv",
        ["document_id"],
        [{"document_id": "MDA_ONLY_2024"}, {"document_id": "MDA_FAIL_2024"}],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [_mda_doc("MDA_ONLY_2024", "AAA", "Alpha Foods", "2024", 70)],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_ratings_long.csv",
        RATINGS_LONG_HEADERS,
        _rating_rows("mda", "MDA_ONLY_2024", ["AR01", "AR02", "AR03", "AR04", "AR05"]),
    )
    write_csv_rows(root / "01_registry" / "news_registry.csv", REGISTRY_HEADERS["news"], [])
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv", NEWS_DOCUMENT_HEADERS, [])
    write_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv", RATINGS_LONG_HEADERS, [])
    _write_cross_validation(root, contradiction_count=0, support_count=0)


def _write_cross_validation(root: Path, contradiction_count: int, support_count: int) -> None:
    link_rows = []
    for idx in range(contradiction_count):
        link_rows.append(_claim_link(f"L_CON_{idx}", contradiction="1"))
    for idx in range(support_count):
        link_rows.append(_claim_link(f"L_SUP_{idx}", contradiction="0"))
    write_csv_rows(
        root / "09_cross_validation" / "claim_links.csv",
        [
            "link_id",
            "mda_claim_id",
            "news_claim_id",
            "mda_document_id",
            "news_document_id",
            "ticker",
            "company_name",
            "report_year",
            "publish_date",
            "topic",
            "claim_type",
            "relation_type",
            "direction",
            "support_level",
            "source_independence",
            "time_alignment",
            "evidence_summary",
            "contradiction_flag",
            "mda_omission_flag",
            "news_overstatement_flag",
            "same_source_repetition_flag",
            "confidence_level",
            "review_required",
        ],
        link_rows,
    )
    write_csv_rows(
        root / "09_cross_validation" / "mda_news_cross_validation_pairs.csv",
        [
            "mda_document_id",
            "ticker",
            "company_name",
            "report_year",
            "num_related_news",
            "num_claim_links",
            "num_contradictions",
            "num_possible_omissions",
            "num_news_overstatements",
            "cross_validation_score",
            "cross_validation_label",
            "review_required",
            "review_reasons",
        ],
        [
            {
                "mda_document_id": "MDA_GOOD_2024",
                "ticker": "AAA",
                "company_name": "Alpha Foods",
                "report_year": "2024",
                "num_related_news": str(support_count + contradiction_count),
                "num_claim_links": str(support_count + contradiction_count),
                "num_contradictions": str(contradiction_count),
                "num_possible_omissions": "0",
                "num_news_overstatements": "0",
                "cross_validation_score": "4" if contradiction_count == 0 else "1",
                "cross_validation_label": "supported" if contradiction_count == 0 else "conflict",
                "review_required": "false" if contradiction_count == 0 else "true",
                "review_reasons": "",
            }
        ],
    )


def _mda_doc(document_id: str, ticker: str, company: str, year: str, total: int) -> dict[str, str]:
    row = {field: "" for field in MDA_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "mda", "ticker": ticker, "company_name": company, "report_year": year, "dimension_count": "5", "total_score_100": str(total), "AR01": "4", "AR02": "4", "AR03": "4", "AR04": "4", "AR05": "4"})
    return row


def _news_doc(document_id: str, ticker: str, company: str, publish_date: str, source_name: str, title: str, total: int) -> dict[str, str]:
    row = {field: "" for field in NEWS_DOCUMENT_HEADERS}
    row.update({"document_id": document_id, "doc_type": "news", "ticker": ticker, "company_name": company, "source_name": source_name, "publish_date": publish_date, "title": title, "dimension_count": "5", "total_score_100": str(total), "N01": "4", "N02": "4", "N03": "4", "N04": "4", "N05": "4"})
    return row


def _rating_rows(doc_type: str, document_id: str, dimensions: list[str], missing_evidence_dimension: str = "", schema_invalid_dimension: str = "", low_confidence_dimension: str = "") -> list[dict[str, str]]:
    rows = []
    prefix = "MDA" if doc_type == "mda" else "NEWS"
    for idx, code in enumerate(dimensions, start=1):
        row = {field: "" for field in RATINGS_LONG_HEADERS}
        row.update(
            {
                "rating_id": f"{document_id}_{code}",
                "document_id": document_id,
                "doc_type": doc_type,
                "dimension_code": code,
                "raw_score": "4",
                "std_score_100": "75",
                "weighted_score": "15",
                "evidence_locator": "" if code == missing_evidence_dimension else f"[{prefix}_P{idx:03d}]",
                "confidence_level": "low" if code == low_confidence_dimension else "high",
                "parse_status": "parsed",
                "schema_validation_status": "invalid" if code == schema_invalid_dimension else "valid",
            }
        )
        rows.append(row)
    return rows


def _claim_link(link_id: str, contradiction: str) -> dict[str, str]:
    return {
        "link_id": link_id,
        "mda_claim_id": f"M_{link_id}",
        "news_claim_id": f"N_{link_id}",
        "mda_document_id": "MDA_GOOD_2024",
        "news_document_id": "NEWS_GOOD_2024",
        "ticker": "AAA",
        "company_name": "Alpha Foods",
        "report_year": "2024",
        "publish_date": "2024-03-01",
        "topic": "revenue",
        "claim_type": "performance",
        "relation_type": "contradiction" if contradiction == "1" else "strong_support",
        "direction": "news_supports_mda",
        "support_level": "none" if contradiction == "1" else "strong",
        "source_independence": "high",
        "time_alignment": "aligned",
        "evidence_summary": "",
        "contradiction_flag": contradiction,
        "mda_omission_flag": "0",
        "news_overstatement_flag": "0",
        "same_source_repetition_flag": "0",
        "confidence_level": "high",
        "review_required": "true" if contradiction == "1" else "false",
    }
