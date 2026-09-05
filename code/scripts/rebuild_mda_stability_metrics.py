from __future__ import annotations

import argparse
import json
import statistics
import re
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json
from stability_common import safe_float, write_markdown


MDA_DIMS = ["AR01", "AR02", "AR03", "AR04", "AR05"]


def rebuild_mda_stability_metrics(root: Path = SCORING_ROOT) -> dict[str, Any]:
    batch_dir = root / "06_ratings" / "mda_v2_stability_batch"
    selected = [row.get("document_id", "") for row in read_csv_rows(batch_dir / "mda_v2_stability_batch_selection.csv") if row.get("document_id")]
    expected_count = len(selected)
    text_by_doc = _text_by_doc(root)
    legacy_stats = _stability_output_stats(
        root,
        selected,
        read_csv_rows(batch_dir / "mda_v2_stability_ratings_long.csv"),
        text_by_doc,
    )
    operational_source = "mda_v2_stability_batch"
    operational_stats = legacy_stats
    complete_path = root / "06_ratings" / "mda_context_repair" / "mda_stability_ratings_long_complete.csv"
    complete_ratings = read_csv_rows(complete_path)
    if complete_ratings:
        operational_source = "mda_stability_ratings_long_complete"
        operational_stats = _stability_output_stats(root, selected, complete_ratings, text_by_doc)
    legacy_rate = legacy_stats["completion_rate"]
    paired = _paired_repeatability(root)
    full = _full_scoring_coverage(root)
    critical_numeric_conflicts = _critical_numeric_conflicts(root, selected)
    metrics = {
        "expected_stability_documents": expected_count,
        "operationally_complete_documents": operational_stats["complete_count"],
        "operational_completion_rate": operational_stats["completion_rate"],
        "operational_metrics_source": operational_source,
        "valid_paired_documents": paired["valid_paired_documents"],
        "stable_paired_documents": paired["stable_paired_documents"],
        "repeatability_pass_rate": paired["repeatability_pass_rate"],
        "schema_success_rate": operational_stats["schema_success_rate"],
        "evidence_match_rate": operational_stats["evidence_match_rate"],
        "json_parse_success_rate": operational_stats["json_parse_success_rate"],
        "critical_numeric_conflict_rate": round(critical_numeric_conflicts / expected_count, 6) if expected_count else 0.0,
        "unresolved_critical_numeric_conflicts": critical_numeric_conflicts,
        "full_scoring_coverage_rate": full["full_scoring_coverage_rate"],
        "eligible_mda_count": full["eligible_mda_count"],
        "complete_mda_count": full["complete_mda_count"],
        "legacy_mda_stability_success_rate": legacy_rate,
        "legacy_metric_note": "legacy_mda_stability_success_rate mixes operational execution failure, structured-output failure, and true score repeatability; it is retained for traceability but is not the final statistical stability definition.",
        "mda_ready": bool(
            operational_stats["completion_rate"] >= 0.95
            and (paired["repeatability_pass_rate"] is not None and paired["repeatability_pass_rate"] >= 0.85)
            and operational_stats["schema_success_rate"] >= 0.98
            and operational_stats["evidence_match_rate"] >= 0.98
            and critical_numeric_conflicts == 0
            and full["full_scoring_coverage_rate"] >= 0.95
        ),
        "repeatability_details": paired,
    }
    out_dir = root / "PAPER_OUTPUT" / "00_status"
    save_json(out_dir / "mda_stability_metrics_redefined.json", metrics)
    _write_report(out_dir / "mda_stability_metrics_redefined.md", metrics)
    return metrics


def _group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("document_id", ""), []).append(row)
    return grouped


def _stability_output_stats(
    root: Path,
    selected: list[str],
    ratings: list[dict[str, str]],
    text_by_doc: dict[str, str],
) -> dict[str, Any]:
    expected_count = len(selected)
    ratings_by_doc = _group_rows(ratings)
    complete_docs = []
    dimension_records = 0
    parsed_records = 0
    schema_valid_records = 0
    evidence_valid_records = 0
    for doc_id in selected:
        rows = ratings_by_doc.get(doc_id, [])
        by_dim = {row.get("dimension_code", ""): row for row in rows}
        doc_complete = True
        for code in MDA_DIMS:
            row = by_dim.get(code)
            if not row:
                doc_complete = False
                continue
            dimension_records += 1
            parse_ok = row.get("parse_status", "parsed").lower() in {"", "parsed"}
            schema_ok = row.get("schema_validation_status", "valid").lower() in {"", "valid"}
            evidence_ok = _evidence_exists(row.get("evidence_locator", ""), text_by_doc.get(doc_id, "") or _direct_text(root, doc_id))
            parsed_records += int(parse_ok)
            schema_valid_records += int(schema_ok)
            evidence_valid_records += int(evidence_ok)
            if not (parse_ok and schema_ok and evidence_ok):
                doc_complete = False
        if doc_complete:
            complete_docs.append(doc_id)
    return {
        "complete_count": len(complete_docs),
        "completion_rate": round(len(complete_docs) / expected_count, 6) if expected_count else 0.0,
        "dimension_records": dimension_records,
        "json_parse_success_rate": round(parsed_records / dimension_records, 6) if dimension_records else 0.0,
        "schema_success_rate": round(schema_valid_records / dimension_records, 6) if dimension_records else 0.0,
        "evidence_match_rate": round(evidence_valid_records / dimension_records, 6) if dimension_records else 0.0,
    }


def _text_by_doc(root: Path) -> dict[str, str]:
    out = {}
    for row in read_csv_rows(root / "01_registry" / "mda_registry.csv"):
        doc_id = row.get("document_id", "")
        paths = []
        if row.get("text_path"):
            paths.append(root / row["text_path"])
        paths.extend(
            [
                root / "02_extracted_text" / "mda_clean_v2" / f"{doc_id}.txt",
                root / "02_extracted_text" / "mda" / f"{doc_id}.txt",
            ]
        )
        for path in paths:
            if path.exists():
                out[doc_id] = path.read_text(encoding="utf-8", errors="replace")
                break
    return out


def _direct_text(root: Path, doc_id: str) -> str:
    for path in [
        root / "02_extracted_text" / "mda_clean_v2" / f"{doc_id}.txt",
        root / "02_extracted_text" / "mda" / f"{doc_id}.txt",
    ]:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _evidence_exists(evidence: str, text: str) -> bool:
    locators = re.findall(r"MDA_P\d{3}", evidence or "")
    if not locators or not text:
        return False
    return all(locator in text for locator in locators)


def _paired_repeatability(root: Path) -> dict[str, Any]:
    run_a = root / "06_ratings" / "mda_repeatability_final" / "run_a" / "mda_repeatability_run_a_document_scores.csv"
    run_b = root / "06_ratings" / "mda_repeatability_final" / "run_b" / "mda_repeatability_run_b_document_scores.csv"
    if not run_a.exists() or not run_b.exists():
        return {
            "valid_paired_documents": 0,
            "stable_paired_documents": 0,
            "repeatability_pass_rate": None,
            "reason": "final repeatability Run A/Run B outputs are missing",
            "dimension_exact_agreement": None,
            "dimension_within_one_agreement": None,
            "mean_absolute_total_score_difference": None,
            "max_total_score_difference": None,
        }
    left = {row.get("document_id", ""): row for row in read_csv_rows(run_a)}
    right = {row.get("document_id", ""): row for row in read_csv_rows(run_b)}
    valid = 0
    stable = 0
    exact = 0
    within_one = 0
    dim_total = 0
    abs_total_diffs: list[float] = []
    for doc_id in sorted(set(left) & set(right)):
        lrow = left[doc_id]
        rrow = right[doc_id]
        if not all(lrow.get(code) and rrow.get(code) for code in MDA_DIMS):
            continue
        valid += 1
        dim_diffs = [abs(safe_float(lrow.get(code)) - safe_float(rrow.get(code))) for code in MDA_DIMS]
        total_diff = abs(safe_float(lrow.get("total_score_100")) - safe_float(rrow.get("total_score_100")))
        abs_total_diffs.append(total_diff)
        dim_total += len(dim_diffs)
        exact += sum(diff == 0 for diff in dim_diffs)
        within_one += sum(diff <= 1 for diff in dim_diffs)
        if max(dim_diffs) < 2 and total_diff <= 5:
            stable += 1
    return {
        "valid_paired_documents": valid,
        "stable_paired_documents": stable,
        "repeatability_pass_rate": round(stable / valid, 6) if valid else None,
        "reason": "" if valid else "no valid paired final repeatability documents",
        "dimension_exact_agreement": round(exact / dim_total, 6) if dim_total else None,
        "dimension_within_one_agreement": round(within_one / dim_total, 6) if dim_total else None,
        "mean_absolute_total_score_difference": round(statistics.mean(abs_total_diffs), 6) if abs_total_diffs else None,
        "max_total_score_difference": max(abs_total_diffs) if abs_total_diffs else None,
    }


def _full_scoring_coverage(root: Path) -> dict[str, Any]:
    eligible = [
        row
        for row in read_csv_rows(root / "01_registry" / "mda_registry.csv")
        if row.get("include_flag", "").lower() == "yes"
    ]
    full_paths = [
        root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv",
        root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv",
    ]
    rows: list[dict[str, str]] = []
    for path in full_paths:
        if path.exists():
            rows = read_csv_rows(path)
            break
    eligible_ids = {row.get("document_id", "") for row in eligible}
    complete_ids = {row.get("document_id", "") for row in rows if row.get("document_id", "") in eligible_ids and all(row.get(code) for code in MDA_DIMS)}
    return {
        "eligible_mda_count": len(eligible),
        "complete_mda_count": len(complete_ids),
        "full_scoring_coverage_rate": round(len(complete_ids) / len(eligible), 6) if eligible else 0.0,
    }


def _critical_numeric_conflicts(root: Path, selected: list[str]) -> int:
    selected_set = set(selected)
    rows = read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv") + read_csv_rows(root / "03_numeric_checks" / "mda_numeric_checks_summary.csv")
    conflicts = 0
    seen = set()
    for row in rows:
        doc_id = row.get("document_id", "")
        if doc_id in seen or (selected_set and doc_id not in selected_set):
            continue
        seen.add(doc_id)
        severe = safe_float(row.get("num_severe_direction_conflicts"))
        unresolved = str(row.get("unresolved_critical_numeric_conflict", "")).lower() in {"1", "true", "yes"}
        if severe > 0 or unresolved:
            conflicts += 1
    return conflicts


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# MD&A Stability Metrics Redefined",
        "",
        f"- expected_stability_documents: {metrics['expected_stability_documents']}",
        f"- operationally_complete_documents: {metrics['operationally_complete_documents']}",
        f"- operational_completion_rate: {metrics['operational_completion_rate']}",
        f"- operational_metrics_source: {metrics['operational_metrics_source']}",
        f"- legacy_mda_stability_success_rate: {metrics['legacy_mda_stability_success_rate']}",
        f"- valid_paired_documents: {metrics['valid_paired_documents']}",
        f"- stable_paired_documents: {metrics['stable_paired_documents']}",
        f"- repeatability_pass_rate: {metrics['repeatability_pass_rate']}",
        f"- schema_success_rate: {metrics['schema_success_rate']}",
        f"- evidence_match_rate: {metrics['evidence_match_rate']}",
        f"- critical_numeric_conflict_rate: {metrics['critical_numeric_conflict_rate']}",
        f"- full_scoring_coverage_rate: {metrics['full_scoring_coverage_rate']}",
        f"- mda_ready: {str(metrics['mda_ready']).lower()}",
        "",
        "## Legacy Metric Note",
        metrics["legacy_metric_note"],
    ]
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild MD&A stability metrics with operational and repeatability rates separated.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(rebuild_mda_stability_metrics(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
