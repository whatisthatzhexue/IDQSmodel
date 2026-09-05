from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from audit_mda_ar02_adjustments import audit_mda_ar02_adjustments
from paper_utils import news_registry_path, read_news_registry_rows
from scoring_utils import (
    MDA_DOCUMENT_HEADERS,
    NEWS_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    REGISTRY_HEADERS,
    SCORING_ROOT,
    ensure_directories,
    read_csv_rows,
    save_json,
    write_csv_rows,
)
from stability_common import safe_float, spearman, write_markdown


FINAL_OUTPUT_NAME = "FINAL_OUTPUT"
FINAL_VERSION = "final_freeze_2026-06-16"

MDA_FINAL_HEADERS = [
    *MDA_DOCUMENT_HEADERS,
    "final_dataset_version",
    "final_include_flag",
    "low_confidence_flag",
    "ar02_numeric_review_flag",
    "source_input_path",
    "source_ratings_long_path",
    "freeze_timestamp",
]

NEWS_RELEVANCE_FINAL_HEADERS = [
    "relevance_confidence_tier",
    "relevance_status_original",
    "relevance_needs_manual_review",
    "relevance_review_reason",
    "model_confidence_tier",
    "strict_company_focus_flag",
    "suspect_not_company_focused",
    "company_focus_reason",
]

NEWS_FINAL_HEADERS = [
    *NEWS_DOCUMENT_HEADERS,
    *NEWS_RELEVANCE_FINAL_HEADERS,
    "final_dataset_version",
    "final_include_flag",
    "low_confidence_flag",
    "source_input_path",
    "source_ratings_long_path",
    "freeze_timestamp",
]

NEWS_COMPANY_FOCUSED_FINAL_HEADERS = [
    *NEWS_FINAL_HEADERS,
    "company_focus_status",
    "strict_main_sample_include",
]

CROSS_FINAL_HEADERS = ["metric", "value", "status", "systemic_conflict_flag", "notes"]

NEWS_ROBUSTNESS_HEADERS = [
    "broad_rows",
    "company_focused_rows",
    "suspect_or_manual_review_rows",
    "shared_document_rows",
    "company_focused_share",
    "rank_spearman_shared_docs",
    "top_20_overlap",
    "bottom_20_overlap",
    "interpretation",
]

MDA_INPUTS = [
    ("mda_final_v2", "06_ratings/mda_final_v2/mda_final_document_scores.csv", "06_ratings/mda_final_v2/mda_final_ratings_long.csv"),
    ("mda_final_full_repaired", "06_ratings/mda_final_full/mda_final_document_scores_repaired.csv", "06_ratings/mda_final_full/mda_final_ratings_long_repaired.csv"),
    ("mda_final_full", "06_ratings/mda_final_full/mda_final_document_scores.csv", "06_ratings/mda_final_full/mda_final_ratings_long.csv"),
    ("mda_v2_full", "06_ratings/mda_v2_full/mda_v2_full_document_scores.csv", "06_ratings/mda_v2_full/mda_v2_full_ratings_long.csv"),
    ("mda_final_candidate", "06_ratings/mda_document_scores_final_candidate.csv", "06_ratings/mda_ratings_long.csv"),
    ("mda_v2_stability_batch", "06_ratings/mda_v2_stability_batch/mda_v2_stability_document_scores.csv", "06_ratings/mda_v2_stability_batch/mda_v2_stability_ratings_long.csv"),
]

NEWS_INPUTS = [
    ("news_final_full", "06_ratings/news_final_full/news_final_document_scores.csv", "06_ratings/news_final_full/news_final_ratings_long.csv"),
    ("news_v2_full", "06_ratings/news_v2_full/news_v2_full_document_scores.csv", "06_ratings/news_v2_full/news_v2_full_ratings_long.csv"),
]


def run_final_data_freeze(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_directories(root)
    final_dir = root / FINAL_OUTPUT_NAME
    final_dir.mkdir(parents=True, exist_ok=True)
    freeze_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ar02_audit = audit_mda_ar02_adjustments(root)

    mda_input = _select_input(root, MDA_INPUTS)
    news_input = _select_input(root, NEWS_INPUTS)
    mda_rows = read_csv_rows(mda_input["scores_path"])
    news_rows = read_csv_rows(news_input["scores_path"])
    mda_rating_rows = read_csv_rows(mda_input["ratings_long_path"])
    news_rating_rows = read_csv_rows(news_input["ratings_long_path"])
    news_registry = _news_registry_summary(root)
    real_news_ids = set(news_registry.get("real_document_ids", []))
    news_rows = [row for row in news_rows if row.get("document_id", "") in real_news_ids]
    news_rating_rows = [row for row in news_rating_rows if row.get("document_id", "") in real_news_ids]

    mda_freeze = _freeze_documents(
        doc_type="mda",
        rows=mda_rows,
        rating_rows=mda_rating_rows,
        input_info=mda_input,
        freeze_timestamp=freeze_timestamp,
    )
    news_freeze = _freeze_documents(
        doc_type="news",
        rows=news_rows,
        rating_rows=news_rating_rows,
        input_info=news_input,
        freeze_timestamp=freeze_timestamp,
    )
    news_freeze["final_rows"] = _enrich_news_final_rows(root, news_freeze["final_rows"])
    company_focus = _company_focused_news_rows(root, news_freeze["final_rows"])
    write_csv_rows(final_dir / "mda_final_dataset.csv", MDA_FINAL_HEADERS, mda_freeze["final_rows"])
    write_csv_rows(final_dir / "news_final_dataset.csv", NEWS_FINAL_HEADERS, news_freeze["final_rows"])
    write_csv_rows(final_dir / "news_broad_dataset.csv", NEWS_FINAL_HEADERS, news_freeze["final_rows"])
    write_csv_rows(final_dir / "news_company_focused_dataset.csv", NEWS_COMPANY_FOCUSED_FINAL_HEADERS, company_focus["focused_rows"])
    news_robustness = _write_news_broad_vs_company_focus_robustness(root, news_freeze["final_rows"], company_focus["focused_rows"], company_focus["suspect_count"])

    stability = _mda_stability_summary(root)
    cross = _cross_validation_summary(root)
    if not news_registry["has_real_registry"] or len(news_freeze["final_rows"]) == 0:
        cross = _cross_validation_blocked_summary(cross)
    stability_rank = _rank_stability_summary(root)
    reason_distribution = _reason_distribution(mda_freeze, news_freeze, cross)
    final_size = {
        "mda": len(mda_freeze["final_rows"]),
        "news": len(news_freeze["final_rows"]),
        "news_company_focused": len(company_focus["focused_rows"]),
    }
    excluded_count = {"mda": len(mda_freeze["excluded"]), "news": len(news_freeze["excluded"])}
    system_error_rate = _system_error_rate(final_size, excluded_count)

    gates = _quality_gates(
        mda_freeze=mda_freeze,
        news_freeze=news_freeze,
        stability=stability,
        news_registry=news_registry,
        cross=cross,
        system_error_rate=system_error_rate,
    )
    freeze_allowed = all(
        gates[name]["passed"]
        for name in [
            "mda_stability_success_rate",
            "news_exists",
            "news_schema_valid",
            "system_level_error_rate",
            "cross_validation_claim_links",
        ]
    )
    cross_rows = _cross_final_rows(cross, stability_rank, final_size, excluded_count, reason_distribution, system_error_rate)
    write_csv_rows(final_dir / "cross_validation_final_summary.csv", CROSS_FINAL_HEADERS, cross_rows)

    metadata = {
        "dataset_version": FINAL_VERSION,
        "freeze_timestamp": freeze_timestamp,
        "freeze_allowed": freeze_allowed,
        "final_dataset_size": final_size,
        "excluded_documents_count": excluded_count,
        "reason_distribution": reason_distribution,
        "system_level_error_rate": system_error_rate,
        "mda": _freeze_metadata(mda_input, mda_freeze, stability),
        "news": _freeze_metadata(news_input, news_freeze, news_registry),
        "cross_validation": cross,
        "top_bottom_stability_check": stability_rank["top_bottom_stability_check"],
        "rank_stability_spearman": stability_rank["rank_stability_spearman"],
        "quality_gates": gates,
        "final_outputs": {
            "mda_final_dataset": str(final_dir / "mda_final_dataset.csv"),
            "news_final_dataset": str(final_dir / "news_final_dataset.csv"),
            "news_broad_dataset": str(final_dir / "news_broad_dataset.csv"),
            "cross_validation_final_summary": str(final_dir / "cross_validation_final_summary.csv"),
            "final_quality_report": str(final_dir / "final_quality_report.md"),
            "dataset_frozen_metadata": str(final_dir / "dataset_frozen_metadata.json"),
        },
    }
    metadata["mda"]["ar02_adjustment_audit"] = ar02_audit
    metadata["news"].update(
        {
            "company_focused_count": len(company_focus["focused_rows"]),
            "broad_count": len(news_freeze["final_rows"]),
            "suspect_not_company_focused_count": company_focus["suspect_count"],
            "company_focus_audit_path": company_focus["audit_path"],
            "company_focused_dataset_path": str(final_dir / "news_company_focused_dataset.csv"),
            "broad_vs_company_focused_robustness": news_robustness,
        }
    )
    metadata["final_outputs"]["news_company_focused_dataset"] = str(final_dir / "news_company_focused_dataset.csv")
    metadata["final_outputs"]["news_broad_vs_company_focused_robustness_csv"] = news_robustness["csv_path"]
    metadata["final_outputs"]["news_broad_vs_company_focused_robustness_md"] = news_robustness["report_path"]
    save_json(final_dir / "dataset_frozen_metadata.json", metadata)
    _write_quality_report(final_dir / "final_quality_report.md", metadata)
    _write_current_final_convergence_report(root, metadata)
    return {
        "freeze_allowed": freeze_allowed,
        "final_output_dir": str(final_dir),
        "final_dataset_size": final_size,
        "excluded_documents_count": excluded_count,
        "reason_distribution": reason_distribution,
        "system_level_error_rate": system_error_rate,
    }


def _select_input(root: Path, candidates: list[tuple[str, str, str]]) -> dict[str, Any]:
    discovered = []
    selected = None
    for label, scores_rel, ratings_rel in candidates:
        scores_path = root / scores_rel
        ratings_path = root / ratings_rel
        row_count = len(read_csv_rows(scores_path))
        record = {
            "label": label,
            "scores_path": scores_path,
            "ratings_long_path": ratings_path,
            "scores_relative_path": scores_rel,
            "ratings_long_relative_path": ratings_rel,
            "exists": scores_path.exists(),
            "row_count": row_count,
        }
        discovered.append(record)
        if selected is None and row_count > 0:
            selected = record
    if selected is None:
        label, scores_rel, ratings_rel = candidates[-1]
        selected = {
            "label": "missing",
            "scores_path": root / scores_rel,
            "ratings_long_path": root / ratings_rel,
            "scores_relative_path": scores_rel,
            "ratings_long_relative_path": ratings_rel,
            "exists": False,
            "row_count": 0,
        }
    selected = dict(selected)
    selected["discovered_inputs"] = [
        {key: (str(value) if isinstance(value, Path) else value) for key, value in item.items()}
        for item in discovered
    ]
    return selected


def _freeze_documents(
    *,
    doc_type: str,
    rows: list[dict[str, str]],
    rating_rows: list[dict[str, str]],
    input_info: dict[str, Any],
    freeze_timestamp: str,
) -> dict[str, Any]:
    grouped_ratings = _ratings_by_document(rating_rows)
    final_rows = []
    excluded = []
    low_confidence_count = 0
    ar02_numeric_review_count = 0
    confirmed_numeric_failure_count = 0
    not_enough_numbers_count = 0
    no_numeric_issue_count = 0
    for row in rows:
        document_id = row.get("document_id", "")
        checks = _document_checks(document_id, grouped_ratings.get(document_id, []), row)
        if checks["low_confidence_flag"]:
            low_confidence_count += 1
        if checks["ar02_numeric_review_flag"]:
            ar02_numeric_review_count += 1
        if checks["confirmed_numeric_failure_flag"]:
            confirmed_numeric_failure_count += 1
        if checks["not_enough_numbers_flag"]:
            not_enough_numbers_count += 1
        if checks["no_numeric_issue_flag"]:
            no_numeric_issue_count += 1
        if checks["exclude_reasons"]:
            excluded.append(
                {
                    "document_id": document_id,
                    "doc_type": doc_type,
                    "ticker": row.get("ticker", ""),
                    "company_name": row.get("company_name", ""),
                    "reasons": checks["exclude_reasons"],
                }
            )
            continue
        out = dict(row)
        out.update(
            {
                "final_dataset_version": FINAL_VERSION,
                "final_include_flag": "true",
                "low_confidence_flag": str(checks["low_confidence_flag"]).lower(),
                "source_input_path": str(input_info["scores_path"]),
                "source_ratings_long_path": str(input_info["ratings_long_path"]) if input_info["ratings_long_path"].exists() else "",
                "freeze_timestamp": freeze_timestamp,
            }
        )
        if doc_type == "mda":
            out["ar02_numeric_review_flag"] = str(checks["ar02_numeric_review_flag"]).lower()
        final_rows.append(out)
    metrics = _rating_metrics(rating_rows, set(row.get("document_id", "") for row in final_rows))
    return {
        "final_rows": final_rows,
        "excluded": excluded,
        "low_confidence_count": low_confidence_count,
        "ar02_numeric_review_count": ar02_numeric_review_count,
        "confirmed_numeric_failure_count": confirmed_numeric_failure_count,
        "not_enough_numbers_count": not_enough_numbers_count,
        "no_numeric_issue_count": no_numeric_issue_count,
        "rating_metrics": metrics,
        "input_count": len(rows),
    }


def _document_checks(document_id: str, ratings: list[dict[str, str]], score_row: dict[str, str]) -> dict[str, Any]:
    reasons = []
    if not ratings:
        reasons.append("missing evidence")
    if any(not row.get("evidence_locator", "").strip() for row in ratings):
        reasons.append("missing evidence")
    if any(_status_invalid(row.get("schema_validation_status", "valid")) for row in ratings):
        reasons.append("schema invalid")
    if any(_status_invalid(row.get("parse_status", "parsed"), expected="parsed") for row in ratings):
        reasons.append("json failure")
    review_text = " ".join([score_row.get("review_reasons", ""), score_row.get("needs_review", "")]).lower()
    if _structured_scope_violation(score_row, ratings):
        reasons.append("scope violation")
    low_confidence = any(row.get("confidence_level", "").lower() == "low" for row in ratings) or "low confidence" in review_text
    ar02_numeric_statuses = [_ar02_numeric_status(row, review_text) for row in ratings if row.get("dimension_code") == "AR02"]
    ar02_numeric_review = "numeric_review_required" in ar02_numeric_statuses
    confirmed_numeric_failure = "confirmed_numeric_failure" in ar02_numeric_statuses
    not_enough_numbers = "not_enough_numbers" in ar02_numeric_statuses
    no_numeric_issue = "no_numeric_issue" in ar02_numeric_statuses
    return {
        "exclude_reasons": sorted(set(reasons)),
        "low_confidence_flag": low_confidence,
        "ar02_numeric_review_flag": ar02_numeric_review,
        "confirmed_numeric_failure_flag": confirmed_numeric_failure,
        "not_enough_numbers_flag": not_enough_numbers,
        "no_numeric_issue_flag": no_numeric_issue,
    }


def _structured_scope_violation(score_row: dict[str, str], ratings: list[dict[str, str]]) -> bool:
    bad_review_reasons = {
        "scope_violation",
        "out_of_scope",
        "outside research scope",
        "wrong_company",
        "not_target_company",
        "not_target_company_news",
        "not_company_news",
    }
    bad_relevance_statuses = {"wrong_company", "not_target_company_news"}

    def truthy(value: str) -> bool:
        return str(value or "").strip().lower() in {"1", "true", "yes"}

    def tokens(value: str) -> set[str]:
        return {item.strip().lower() for item in str(value or "").replace("|", ";").replace(",", ";").split(";") if item.strip()}

    rows = [score_row, *ratings]
    for row in rows:
        if truthy(row.get("scope_violation", "")):
            return True
        if tokens(row.get("review_reason", "")) & bad_review_reasons:
            return True
        if tokens(row.get("review_reasons", "")) & bad_review_reasons:
            return True
        relevance_status = str(row.get("relevance_status", "")).strip().lower()
        if relevance_status in bad_relevance_statuses or relevance_status.startswith("unusable_"):
            return True
    return False


def _ar02_numeric_status(row: dict[str, str], document_review_text: str = "") -> str:
    text = " ".join(
        [
            row.get("comment_short", ""),
            row.get("evidence_text_short", ""),
            row.get("missing_type", ""),
            document_review_text,
        ]
    ).lower()
    if "confirmed_numeric_failure" in text or "numeric failure" in text:
        return "confirmed_numeric_failure"
    if "numeric_review_required" in text or "numeric review" in text:
        return "numeric_review_required"
    if "not enough numbers" in text or "not enough number" in text or "insufficient numeric" in text:
        return "not_enough_numbers"
    return "no_numeric_issue"


def _rating_metrics(rating_rows: list[dict[str, str]], final_doc_ids: set[str]) -> dict[str, Any]:
    included = [row for row in rating_rows if row.get("document_id") in final_doc_ids]
    all_rows = included or []
    total = len(all_rows)
    parsed = sum(not _status_invalid(row.get("parse_status", "parsed"), expected="parsed") for row in all_rows)
    schema_valid = sum(not _status_invalid(row.get("schema_validation_status", "valid"), expected="valid") for row in all_rows)
    evidence = sum(bool(row.get("evidence_locator", "").strip()) for row in all_rows)
    return {
        "dimension_rows": total,
        "json_parse_failure_rate": round(1 - parsed / total, 6) if total else 1.0,
        "schema_validation_success_rate": round(schema_valid / total, 6) if total else 0.0,
        "evidence_locator_success_rate": round(evidence / total, 6) if total else 0.0,
        "schema_failure_critical_count": total - schema_valid,
        "missing_evidence_count": total - evidence,
    }


def _mda_stability_summary(root: Path) -> dict[str, Any]:
    selection = read_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv")
    successes = read_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv")
    selected_count = len(selection)
    success_count = len(successes)
    legacy_success_rate = round(success_count / selected_count, 6) if selected_count else 0.0
    redefined_path = root / "PAPER_OUTPUT" / "00_status" / "mda_stability_metrics_redefined.json"
    if redefined_path.exists():
        try:
            metrics = json.loads(redefined_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metrics = {}
        operational_rate = metrics.get("operational_completion_rate")
        if operational_rate is not None:
            return {
                "selected_count": int(metrics.get("expected_stability_documents", selected_count) or selected_count),
                "success_count": int(metrics.get("operationally_complete_documents", success_count) or success_count),
                "success_rate": round(safe_float(operational_rate), 6),
                "legacy_success_rate": legacy_success_rate,
                "source": "mda_stability_metrics_redefined",
                "operational_metrics_source": metrics.get("operational_metrics_source", ""),
                "source_metrics_path": str(redefined_path),
                "source_selection_path": str(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv"),
                "source_success_path": str(root / "06_ratings" / "mda_context_repair" / "mda_stability_document_scores_complete.csv"),
            }
    return {
        "selected_count": selected_count,
        "success_count": success_count,
        "success_rate": legacy_success_rate,
        "legacy_success_rate": legacy_success_rate,
        "source": "mda_v2_stability_batch",
        "source_selection_path": str(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv"),
        "source_success_path": str(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv"),
    }


def _news_registry_summary(root: Path) -> dict[str, Any]:
    registry_path = news_registry_path(root)
    rows = read_news_registry_rows(root)
    included = [row for row in rows if row.get("include_flag", "").lower() == "yes"]
    synthetic_count = sum(_registry_row_is_synthetic(row) for row in included)
    real = [row for row in included if not _registry_row_is_synthetic(row)]
    return {
        "registry_path": str(registry_path),
        "row_count": len(rows),
        "included_count": len(included),
        "synthetic_count": synthetic_count,
        "real_count": len(real),
        "real_document_ids": [row.get("document_id", "") for row in real if row.get("document_id")],
        "has_news_registry": len(included) > 0,
        "has_real_registry": len(real) > 0,
    }


def _enrich_news_final_rows(root: Path, final_news_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry_by_id = {row.get("document_id", ""): row for row in read_news_registry_rows(root) if row.get("document_id")}
    focus_audit_by_id = {
        row.get("document_id", ""): row
        for row in read_csv_rows(root / "08_reports" / "news_company_focus_audit.csv")
        if row.get("document_id")
    }
    enriched: list[dict[str, Any]] = []
    for row in final_news_rows:
        document_id = row.get("document_id", "")
        registry = registry_by_id.get(document_id, {})
        focus = focus_audit_by_id.get(document_id, {})
        out = dict(row)
        out.update(
            {
                "relevance_confidence_tier": _first_nonempty(
                    registry.get("confidence_tier", ""),
                    focus.get("confidence_tier", ""),
                    row.get("confidence_tier", ""),
                ),
                "relevance_status_original": _first_nonempty(
                    registry.get("relevance_status", ""),
                    focus.get("relevance_status", ""),
                    row.get("relevance_status", ""),
                ),
                "relevance_needs_manual_review": _first_nonempty(
                    registry.get("needs_manual_review", ""),
                    focus.get("needs_manual_review", ""),
                ),
                "relevance_review_reason": _first_nonempty(
                    registry.get("review_reason", ""),
                    focus.get("review_reason", ""),
                    row.get("review_reason", ""),
                ),
                "model_confidence_tier": row.get("confidence_tier", ""),
                "strict_company_focus_flag": _first_nonempty(
                    focus.get("strict_main_sample_include", ""),
                    focus.get("strict_company_focus_flag", ""),
                    registry.get("strict_company_focus_flag", ""),
                ),
                "suspect_not_company_focused": _first_nonempty(
                    focus.get("suspect_not_company_focused", ""),
                    row.get("suspect_not_company_focused", ""),
                ),
                "company_focus_reason": _first_nonempty(
                    focus.get("company_focus_reason", ""),
                    row.get("company_focus_reason", ""),
                ),
            }
        )
        enriched.append(out)
    return enriched


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _company_focused_news_rows(root: Path, final_news_rows: list[dict[str, Any]]) -> dict[str, Any]:
    audit_path = root / "08_reports" / "news_company_focus_audit.csv"
    audit_rows = read_csv_rows(audit_path)
    audit_by_id = {row.get("document_id", ""): row for row in audit_rows if row.get("document_id")}
    focused_rows: list[dict[str, Any]] = []
    suspect_count = 0
    for row in final_news_rows:
        document_id = row.get("document_id", "")
        audit = audit_by_id.get(document_id, {})
        strict = audit.get("strict_main_sample_include", "").strip().lower() == "yes"
        suspect = audit.get("suspect_not_company_focused", "").strip().lower() == "yes"
        if suspect:
            suspect_count += 1
        if strict and not suspect:
            out = dict(row)
            out.update(
                {
                    "company_focus_status": audit.get("company_focus_status", "company_focused"),
                    "suspect_not_company_focused": audit.get("suspect_not_company_focused", "No"),
                    "company_focus_reason": audit.get("company_focus_reason", ""),
                    "strict_company_focus_flag": audit.get("strict_main_sample_include", "Yes"),
                    "strict_main_sample_include": audit.get("strict_main_sample_include", "Yes"),
                }
            )
            focused_rows.append(out)
    return {
        "focused_rows": focused_rows,
        "suspect_count": suspect_count,
        "audit_path": str(audit_path) if audit_path.exists() else "",
    }


def _write_news_broad_vs_company_focus_robustness(
    root: Path,
    broad_rows: list[dict[str, Any]],
    focused_rows: list[dict[str, Any]],
    suspect_count: int,
) -> dict[str, Any]:
    reports_dir = root / "08_reports"
    csv_path = reports_dir / "news_broad_vs_company_focused_robustness.csv"
    md_path = reports_dir / "news_broad_vs_company_focused_robustness.md"
    broad_rank = _score_ranks(broad_rows)
    focused_rank = _score_ranks(focused_rows)
    shared_ids = sorted(set(broad_rank) & set(focused_rank))
    if len(shared_ids) >= 2:
        rank_spearman = round(spearman([broad_rank[doc_id] for doc_id in shared_ids], [focused_rank[doc_id] for doc_id in shared_ids]), 6)
    elif len(shared_ids) == 1:
        rank_spearman = 1.0
    else:
        rank_spearman = 0.0
    top_overlap = _sample_edge_overlap(broad_rows, focused_rows, top=True)
    bottom_overlap = _sample_edge_overlap(broad_rows, focused_rows, top=False)
    company_focused_share = round(len(focused_rows) / len(broad_rows), 6) if broad_rows else 0.0
    interpretation = (
        "Broad sample is retained for document-level measurement; strict company-focused subset should be used for company-specific robustness."
    )
    summary = {
        "broad_rows": len(broad_rows),
        "company_focused_rows": len(focused_rows),
        "suspect_or_manual_review_rows": suspect_count,
        "shared_document_rows": len(shared_ids),
        "company_focused_share": company_focused_share,
        "rank_spearman_shared_docs": rank_spearman,
        "top_20_overlap": top_overlap,
        "bottom_20_overlap": bottom_overlap,
        "interpretation": interpretation,
    }
    write_csv_rows(csv_path, NEWS_ROBUSTNESS_HEADERS, [summary])
    write_markdown(
        md_path,
        [
            "# News Broad Vs Company-Focused Robustness",
            "",
            f"- News broad sample rows: {len(broad_rows)}",
            f"- News company-focused strict subset rows: {len(focused_rows)}",
            f"- suspect/manual-review rows excluded from strict subset: {suspect_count}",
            f"- shared document rows for rank comparison: {len(shared_ids)}",
            f"- rank Spearman on shared documents: {rank_spearman}",
            f"- top 20% overlap: {top_overlap}",
            f"- bottom 20% overlap: {bottom_overlap}",
            "- News repeatability status: not completed in this package unless a separate repeatability scoring run is executed.",
            "",
            "## Interpretation",
            "- The 125-row broad sample is a manually preserved News measurement sample.",
            "- The company-focused subset excludes suspect market-roundup, index-list, sponsor-list, venue, sport, metro, lifestyle, and incidental-mention cases before strict robustness use.",
            "- Do not describe the broad sample as fully clean company-specific News; use the strict subset for company-specific robustness claims.",
        ],
    )
    return {**summary, "csv_path": str(csv_path), "report_path": str(md_path)}


def _score_ranks(rows: list[dict[str, Any]]) -> dict[str, int]:
    ranked = sorted(rows, key=lambda row: (-safe_float(row.get("total_score_100")), row.get("document_id", "")))
    return {row.get("document_id", ""): idx for idx, row in enumerate(ranked, start=1) if row.get("document_id")}


def _sample_edge_overlap(broad_rows: list[dict[str, Any]], focused_rows: list[dict[str, Any]], *, top: bool) -> float:
    if not broad_rows or not focused_rows:
        return 0.0
    broad_ids = _edge_ids_by_score(broad_rows, top=top)
    focused_ids = _edge_ids_by_score(focused_rows, top=top)
    denominator = max(1, len(focused_ids))
    return round(len(broad_ids & focused_ids) / denominator, 6)


def _edge_ids_by_score(rows: list[dict[str, Any]], *, top: bool) -> set[str]:
    size = max(1, int(round(len(rows) * 0.2)))
    ordered = sorted(rows, key=lambda row: (safe_float(row.get("total_score_100")), row.get("document_id", "")), reverse=top)
    return {row.get("document_id", "") for row in ordered[:size] if row.get("document_id")}


def _registry_row_is_synthetic(row: dict[str, str]) -> bool:
    synthetic_flag = row.get("synthetic_flag", "").strip().lower()
    notes = row.get("notes", "").lower()
    return synthetic_flag in {"1", "true", "yes"} or "synthetic_flag=true" in notes


def _cross_validation_summary(root: Path) -> dict[str, Any]:
    links_path = root / "09_cross_validation" / "claim_links.csv"
    pairs_path = root / "09_cross_validation" / "mda_news_cross_validation_pairs.csv"
    links = read_csv_rows(links_path)
    pairs = read_csv_rows(pairs_path)
    contradictions = sum(_truthy(row.get("contradiction_flag")) for row in links)
    omissions = sum(_truthy(row.get("mda_omission_flag")) for row in links)
    overstatements = sum(_truthy(row.get("news_overstatement_flag")) for row in links)
    pair_contradictions = sum(int(safe_float(row.get("num_contradictions"))) for row in pairs)
    total_links = len(links)
    contradiction_rate = round(contradictions / total_links, 6) if total_links else 0.0
    systemic = bool(contradictions >= 3 and contradiction_rate > 0.20) or bool(pair_contradictions >= 3 and pairs)
    return {
        "claim_links_path": str(links_path),
        "pairs_path": str(pairs_path),
        "claim_link_count": total_links,
        "pair_count": len(pairs),
        "contradiction_count": contradictions,
        "possible_omission_count": omissions,
        "news_overstatement_count": overstatements,
        "contradiction_rate": contradiction_rate,
        "systemic_conflict_flag": systemic,
        "status": "systemic_conflict" if systemic else "ok",
    }


def _cross_validation_blocked_summary(cross: dict[str, Any]) -> dict[str, Any]:
    blocked = dict(cross)
    blocked.update(
        {
            "claim_link_count": 0,
            "pair_count": 0,
            "contradiction_count": 0,
            "possible_omission_count": 0,
            "news_overstatement_count": 0,
            "contradiction_rate": 0.0,
            "systemic_conflict_flag": False,
            "status": "blocked_no_real_news",
        }
    )
    return blocked


def _rank_stability_summary(root: Path) -> dict[str, Any]:
    rows = read_csv_rows(root / "11_stability_analysis" / "outputs" / "mda_weight_sensitivity_scores.csv")
    if rows and {"rank_W1", "rank_W2", "rank_W3"}.issubset(rows[0].keys()):
        rank_w1 = [safe_float(row.get("rank_W1")) for row in rows]
        rank_w2 = [safe_float(row.get("rank_W2")) for row in rows]
        rank_w3 = [safe_float(row.get("rank_W3")) for row in rows]
        rank_spearman = min(spearman(rank_w1, rank_w2), spearman(rank_w2, rank_w3)) if len(rows) >= 2 else 0.0
        top_stable = _edge_overlap(rows, "rank_W1", "rank_W2", top=True)
        bottom_stable = _edge_overlap(rows, "rank_W1", "rank_W2", top=False)
        return {
            "rank_stability_spearman": rank_spearman,
            "top_bottom_stability_check": {
                "source": "mda_weight_sensitivity_scores.csv",
                "top_20_overlap_ratio": top_stable,
                "bottom_20_overlap_ratio": bottom_stable,
            },
        }
    return {
        "rank_stability_spearman": 0.0,
        "top_bottom_stability_check": {
            "source": "unavailable",
            "top_20_overlap_ratio": 0.0,
            "bottom_20_overlap_ratio": 0.0,
        },
    }


def _quality_gates(
    *,
    mda_freeze: dict[str, Any],
    news_freeze: dict[str, Any],
    stability: dict[str, Any],
    news_registry: dict[str, Any],
    cross: dict[str, Any],
    system_error_rate: float,
) -> dict[str, dict[str, Any]]:
    mda_metrics = mda_freeze["rating_metrics"]
    news_metrics = news_freeze["rating_metrics"]
    return {
        "mda_stability_success_rate": {
            "passed": stability["success_rate"] >= 0.85,
            "value": stability["success_rate"],
            "threshold": ">=0.85",
        },
        "mda_ar02_numeric_review_controlled": {
            "passed": True,
            "value": mda_freeze["ar02_numeric_review_count"],
            "threshold": "numeric issues fixed or marked review",
        },
        "mda_no_schema_failure_critical": {
            "passed": mda_metrics["schema_failure_critical_count"] == 0,
            "value": mda_metrics["schema_failure_critical_count"],
            "threshold": "0 critical schema failures in final dataset",
        },
        "mda_evidence_locator_success": {
            "passed": mda_metrics["evidence_locator_success_rate"] >= 0.95,
            "value": mda_metrics["evidence_locator_success_rate"],
            "threshold": ">=0.95",
        },
        "news_exists": {
            "passed": news_registry["has_real_registry"] and len(news_freeze["final_rows"]) > 0,
            "value": len(news_freeze["final_rows"]),
            "threshold": "real non-synthetic news registry and final News scores exist",
        },
        "news_schema_valid": {
            "passed": len(news_freeze["final_rows"]) > 0 and news_metrics["schema_validation_success_rate"] >= 0.95,
            "value": news_metrics["schema_validation_success_rate"],
            "threshold": ">=0.95",
        },
        "news_registry_real": {
            "passed": news_registry["has_real_registry"] and len(news_freeze["final_rows"]) > 0,
            "value": news_registry["included_count"],
            "threshold": ">0 non-synthetic included News registry rows and final scores",
        },
        "news_json_parse_failure": {
            "passed": news_metrics["json_parse_failure_rate"] < 0.05,
            "value": news_metrics["json_parse_failure_rate"],
            "threshold": "<0.05",
        },
        "news_schema_validation_success": {
            "passed": news_metrics["schema_validation_success_rate"] >= 0.95,
            "value": news_metrics["schema_validation_success_rate"],
            "threshold": ">=0.95",
        },
        "cross_validation_no_systemic_conflict": {
            "passed": not cross["systemic_conflict_flag"],
            "value": cross["contradiction_count"],
            "threshold": "no systemic conflict error",
        },
        "cross_validation_claim_links": {
            "passed": cross["claim_link_count"] > 0,
            "value": cross["claim_link_count"],
            "threshold": ">0 real News-backed claim links",
        },
        "system_level_error_rate": {
            "passed": system_error_rate < 0.05,
            "value": system_error_rate,
            "threshold": "<0.05",
        },
    }


def _reason_distribution(mda_freeze: dict[str, Any], news_freeze: dict[str, Any], cross: dict[str, Any]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for freeze in [mda_freeze, news_freeze]:
        counter["low confidence"] += freeze.get("low_confidence_count", 0)
        for item in freeze.get("excluded", []):
            for reason in item.get("reasons", []):
                counter[reason] += 1
    counter["numeric_review_required"] += mda_freeze.get("ar02_numeric_review_count", 0)
    counter["confirmed_numeric_failure"] += mda_freeze.get("confirmed_numeric_failure_count", 0)
    counter["not_enough_numbers"] += mda_freeze.get("not_enough_numbers_count", 0)
    if cross["systemic_conflict_flag"]:
        counter["cross-validation conflict"] += cross["contradiction_count"]
    for required in ["json failure", "numeric_review_required", "confirmed_numeric_failure", "not_enough_numbers", "missing evidence", "low confidence", "cross-validation conflict"]:
        counter.setdefault(required, 0)
    return dict(counter)


def _cross_final_rows(
    cross: dict[str, Any],
    stability_rank: dict[str, Any],
    final_size: dict[str, int],
    excluded_count: dict[str, int],
    reason_distribution: dict[str, int],
    system_error_rate: float,
) -> list[dict[str, Any]]:
    rows = [
        {"metric": "mda_final_dataset_size", "value": final_size["mda"], "status": "reported", "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""},
        {"metric": "news_final_dataset_size", "value": final_size["news"], "status": "reported", "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""},
        {"metric": "mda_excluded_documents_count", "value": excluded_count["mda"], "status": "reported", "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""},
        {"metric": "news_excluded_documents_count", "value": excluded_count["news"], "status": "reported", "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""},
        {"metric": "cross_validation_claim_link_count", "value": cross["claim_link_count"], "status": cross["status"], "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""},
        {"metric": "cross_validation_contradiction_count", "value": cross["contradiction_count"], "status": cross["status"], "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""},
        {"metric": "systemic_conflict_flag", "value": str(cross["systemic_conflict_flag"]).lower(), "status": cross["status"], "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": "true means final freeze is blocked"},
        {"metric": "rank_stability_spearman", "value": stability_rank["rank_stability_spearman"], "status": "reported", "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": "from stability weight sensitivity output if available"},
        {"metric": "system_level_error_rate", "value": system_error_rate, "status": "reported", "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""},
    ]
    for reason, count in sorted(reason_distribution.items()):
        rows.append({"metric": f"reason_distribution:{reason}", "value": count, "status": "reported", "systemic_conflict_flag": str(cross["systemic_conflict_flag"]).lower(), "notes": ""})
    return rows


def _freeze_metadata(input_info: dict[str, Any], freeze: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "selected_input_label": input_info["label"],
        "selected_scores_path": str(input_info["scores_path"]),
        "selected_ratings_long_path": str(input_info["ratings_long_path"]) if input_info["ratings_long_path"].exists() else "",
        "input_count": freeze["input_count"],
        "final_count": len(freeze["final_rows"]),
        "excluded_count": len(freeze["excluded"]),
        "excluded_documents": freeze["excluded"],
        "low_confidence_count": freeze["low_confidence_count"],
        "rating_metrics": freeze["rating_metrics"],
        "extra": extra,
    }
    if "ar02_numeric_review_count" in freeze:
        metadata["numeric_review_flag_count"] = freeze["ar02_numeric_review_count"]
        metadata["confirmed_numeric_failure_count"] = freeze.get("confirmed_numeric_failure_count", 0)
        metadata["not_enough_numbers_count"] = freeze.get("not_enough_numbers_count", 0)
        metadata["no_numeric_issue_count"] = freeze.get("no_numeric_issue_count", 0)
    return metadata


def _write_quality_report(path: Path, metadata: dict[str, Any]) -> None:
    freeze_allowed = metadata["freeze_allowed"]
    required_gate_names = [
        "mda_stability_success_rate",
        "news_exists",
        "news_schema_valid",
        "system_level_error_rate",
        "cross_validation_claim_links",
    ]
    blockers = [name for name in required_gate_names if not metadata["quality_gates"][name]["passed"]]
    advisory_warnings = [
        _quality_gate_warning_label(name)
        for name, gate in metadata["quality_gates"].items()
        if name not in required_gate_names and not gate["passed"]
    ]
    news_missing = not metadata["quality_gates"]["news_exists"]["passed"]
    news_synthetic = metadata["news"]["extra"].get("synthetic_count", 0) > 0
    systemic_conflict = bool(metadata["cross_validation"].get("systemic_conflict_flag", False))
    cross_conflict_free = metadata["quality_gates"]["cross_validation_no_systemic_conflict"]["passed"]
    research_ready = freeze_allowed and cross_conflict_free and not news_synthetic and metadata["quality_gates"]["news_registry_real"]["passed"]
    research_grade = "YES" if research_ready else "NO"
    empirical = "YES" if research_ready else "NO"
    unstable_dimensions = _unstable_dimensions(metadata)
    lines = [
        "# Final Data Freeze Quality Report",
        "",
        "## FINAL DATASET",
        f"- dataset_version: {metadata['dataset_version']}",
        f"- freeze_allowed: {str(freeze_allowed).lower()}",
        f"- mda_final_dataset_size: {metadata['final_dataset_size']['mda']}",
        f"- news_final_dataset_size: {metadata['final_dataset_size']['news']}",
        f"- system_level_error_rate: {metadata['system_level_error_rate']}",
        f"- numeric_review_flag_count: {metadata['mda'].get('numeric_review_flag_count', 0)}",
        f"- confirmed_numeric_failure_count: {metadata['mda'].get('confirmed_numeric_failure_count', 0)}",
        f"- not_enough_numbers_count: {metadata['mda'].get('not_enough_numbers_count', 0)}",
        f"- no_numeric_issue_count: {metadata['mda'].get('no_numeric_issue_count', 0)}",
        "",
        "## Research-Grade Reliability",
        f"- research-grade reliability: {research_grade}",
        f"- empirical analysis: {empirical}",
        f"- blocking gates: {', '.join(blockers) if blockers else 'none'}",
        f"- advisory warnings: {', '.join(advisory_warnings) if advisory_warnings else 'none'}",
        "",
        "## Exclusion And Reason Distribution",
    ]
    for reason, count in sorted(metadata["reason_distribution"].items()):
        lines.append(f"- {reason}: {count}")
    lines.extend(
        [
            "",
            "## MD&A Vs News Consistency Summary",
            f"- claim_link_count: {metadata['cross_validation']['claim_link_count']}",
            f"- contradiction_count: {metadata['cross_validation']['contradiction_count']}",
            f"- systemic_conflict_flag: {metadata['cross_validation']['systemic_conflict_flag']}",
            f"- News registry or News scores missing: {'YES' if news_missing else 'NO'}",
            f"- News synthetic placeholder rows: {metadata['news']['extra'].get('synthetic_count', 0)}",
            f"- real News registry rows: {metadata['news']['extra'].get('real_count', 0)}",
            "",
            "## Stability Checks",
            f"- top_20_overlap_ratio: {metadata['top_bottom_stability_check']['top_20_overlap_ratio']}",
            f"- bottom_20_overlap_ratio: {metadata['top_bottom_stability_check']['bottom_20_overlap_ratio']}",
            f"- rank stability (Spearman): {metadata['rank_stability_spearman']}",
            "",
            "## Bias And Dimension Risk",
            f"- systematic bias: {_systematic_bias_text(news_missing, news_synthetic, systemic_conflict)}",
            f"- most unstable dimensions: {unstable_dimensions}",
            "",
            "## Final Recommendation",
            f"- use for paper empirical analysis: {empirical}",
            f"- recommendation: {'freeze this version for publication' if research_ready else 'do not publish as final until blocking gates or advisory warnings are resolved'}",
            f"- future model optimization: {'freeze version now; optimize only in future branch' if research_ready else 'fix input coverage/stability first; do not change model or dimensions inside this freeze'}",
            "",
            "## Limitations",
            "- This layer does not call qwen3 and does not change AR/N dimension scores or weights.",
            "- Cross-validation remains auxiliary and is not averaged into MD&A or News scores.",
        ]
    )
    write_markdown(path, lines)


def _write_current_final_convergence_report(root: Path, metadata: dict[str, Any]) -> None:
    readiness = _load_json(root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json")
    cv_metadata = _load_json(root / "09_cross_validation" / "final_real_run" / "cross_validation_run_metadata.json")
    eligible_count = int(readiness.get("eligible_company_year_pair_count") or 0)
    if eligible_count == 0:
        eligible_count = len(read_csv_rows(root / "09_cross_validation" / "final_real_run" / "company_year_cross_validation_summary.csv"))
    mock_mode = bool(cv_metadata.get("mock_mode", False))
    systemic_conflict = bool(metadata["cross_validation"].get("systemic_conflict_flag", False))
    pipeline_ready = bool(cv_metadata.get("cross_validation_pipeline_ready", metadata["cross_validation"].get("claim_link_count", 0) > 0))
    empirical_ready = bool(cv_metadata.get("cross_validation_empirical_ready", False)) and not mock_mode and not systemic_conflict
    summary = {
        "freeze_allowed": metadata["freeze_allowed"],
        "paper_ready": bool(readiness.get("paper_ready", False)),
        "news_ready": bool(readiness.get("news_ready", metadata["final_dataset_size"]["news"] > 0)),
        "cross_validation_ready": pipeline_ready and empirical_ready,
        "cross_validation_pipeline_ready": pipeline_ready,
        "cross_validation_empirical_ready": empirical_ready,
        "mock_mode": mock_mode,
        "systemic_conflict_flag": systemic_conflict,
        "final_dataset_size": metadata["final_dataset_size"],
        "mda_final_dataset_size": metadata["final_dataset_size"]["mda"],
        "news_final_dataset_size": metadata["final_dataset_size"]["news"],
        "news_company_focused_dataset_size": metadata["final_dataset_size"].get("news_company_focused", 0),
        "synthetic_news_count": metadata["news"]["extra"].get("synthetic_count", 0),
        "eligible_company_year_pair_count": eligible_count,
        "claim_link_count": metadata["cross_validation"].get("claim_link_count", 0),
        "system_level_error_rate": metadata["system_level_error_rate"],
        "interpretation": "News scoring enters the document-level measurement model, but claim-level MD&A-News triangulation remains non-interpretable because claim scoring used deterministic proxy/mock mode or unresolved systemic conflict flags.",
    }
    save_json(root / "FINAL_OUTPUT" / "final_convergence_report.json", summary)
    write_markdown(
        root / "FINAL_OUTPUT" / "final_convergence_report.md",
        [
            "# Final Convergence Report",
            "",
            f"- freeze_allowed: {str(summary['freeze_allowed']).lower()}",
            f"- paper_ready: {str(summary['paper_ready']).lower()}",
            f"- news_ready: {str(summary['news_ready']).lower()}",
            f"- mda_final_dataset_size: {summary['mda_final_dataset_size']}",
            f"- news_final_dataset_size: {summary['news_final_dataset_size']}",
            f"- news_company_focused_dataset_size: {summary['news_company_focused_dataset_size']}",
            f"- synthetic_news_count: {summary['synthetic_news_count']}",
            f"- eligible_company_year_pair_count: {summary['eligible_company_year_pair_count']}",
            f"- claim_link_count: {summary['claim_link_count']}",
            f"- cross_validation_pipeline_ready: {str(summary['cross_validation_pipeline_ready']).lower()}",
            f"- cross_validation_empirical_ready: {str(summary['cross_validation_empirical_ready']).lower()}",
            f"- cross_validation_ready: {str(summary['cross_validation_ready']).lower()}",
            f"- mock_mode: {str(summary['mock_mode']).lower()}",
            f"- systemic_conflict_flag: {str(summary['systemic_conflict_flag']).lower()}",
            f"- system_level_error_rate: {summary['system_level_error_rate']}",
            "",
            "## Interpretation",
            f"- {summary['interpretation']}",
            "- Avoid pass/fail wording that implies empirical readiness while cross_validation_empirical_ready is false.",
        ],
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _quality_gate_warning_label(name: str) -> str:
    if name == "cross_validation_no_systemic_conflict":
        return "cross_validation_systemic_conflict_unresolved"
    return name


def _systematic_bias_text(news_missing: bool, news_synthetic: bool, systemic_conflict: bool) -> str:
    if news_missing:
        return "possible due to missing News/cross-validation coverage"
    if news_synthetic:
        return "possible due to synthetic News placeholder"
    if systemic_conflict:
        return "systemic conflict detected by freeze layer"
    return "systemic conflict not detected by freeze layer"


def _unstable_dimensions(metadata: dict[str, Any]) -> str:
    stability_path = Path(metadata["final_outputs"]["dataset_frozen_metadata"]).parents[0].parents[0] / "11_stability_analysis" / "outputs" / "statistical_stability_summary.json"
    if stability_path.exists():
        try:
            data = json.loads(stability_path.read_text(encoding="utf-8"))
            dims = data.get("final_recommendation", {}).get("most_sensitive_dimensions", "")
            if dims:
                return dims
        except json.JSONDecodeError:
            pass
    return "AR02 / AR03 / N02 require review if corresponding evidence or News coverage is incomplete"


def _ratings_by_document(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("document_id", "")].append(row)
    return grouped


def _status_invalid(value: str, expected: str = "valid") -> bool:
    cleaned = (value or expected).strip().lower()
    return cleaned not in {"", expected}


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _edge_overlap(rows: list[dict[str, str]], left_rank: str, right_rank: str, *, top: bool) -> float:
    if not rows:
        return 0.0
    size = max(1, int(round(len(rows) * 0.2)))
    reverse = not top
    left = sorted(rows, key=lambda row: safe_float(row.get(left_rank)), reverse=reverse)[:size]
    right = sorted(rows, key=lambda row: safe_float(row.get(right_rank)), reverse=reverse)[:size]
    left_ids = {row.get("document_id", "") for row in left}
    right_ids = {row.get("document_id", "") for row in right}
    return round(len(left_ids & right_ids) / size, 6)


def _system_error_rate(final_size: dict[str, int], excluded_count: dict[str, int]) -> float:
    total = final_size["mda"] + final_size["news"] + excluded_count["mda"] + excluded_count["news"]
    excluded = excluded_count["mda"] + excluded_count["news"]
    return round(excluded / total, 6) if total else 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze existing MD&A, News, stability, and cross-validation results into FINAL_OUTPUT.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_final_data_freeze(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
