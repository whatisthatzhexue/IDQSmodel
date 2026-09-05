from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from stability_common import write_markdown


DIMENSION_CODES = ["AR01", "AR02", "AR03", "AR04", "AR05"]

CASE_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "report_year",
    "dimension_code",
    "stability_case_type",
    "failure_reason",
    "source_file",
    "raw_output_path",
    "parsed_output_path",
    "evidence_locator",
    "evidence_exists",
    "score_current",
    "score_comparison",
    "score_diff",
    "needs_rescore",
    "recommended_action",
]

FAILURE_REASONS = [
    "json_parse_failure",
    "schema_validation_failure",
    "evidence_locator_failure",
    "missing_dimension",
    "low_confidence",
    "AR02_numeric_discrepancy",
    "score_difference_too_large",
    "repeatability_failure",
    "version_comparison_failure",
    "structure_comparison_failure",
    "ollama_http_error",
    "ollama_502_error",
    "text_too_long",
    "text_extraction_issue",
    "missing_output",
    "unknown",
]

RECOMMENDED_ACTIONS = {
    "json_parse_failure": "repair_json_only",
    "schema_validation_failure": "repair_json_only",
    "evidence_locator_failure": "fix_evidence_mapping",
    "missing_dimension": "rerun_failed_dimension_only",
    "low_confidence": "manual_review",
    "AR02_numeric_discrepancy": "fix_numeric_checker",
    "score_difference_too_large": "rerun_failed_dimension_only",
    "repeatability_failure": "rerun_failed_dimension_only",
    "version_comparison_failure": "exclude_from_comparison_as_not_comparable",
    "structure_comparison_failure": "exclude_from_comparison_as_not_comparable",
    "ollama_http_error": "rerun_failed_dimension_only",
    "ollama_502_error": "rerun_failed_dimension_only",
    "text_too_long": "rerun_failed_dimension_only",
    "text_extraction_issue": "manual_review",
    "missing_output": "rerun_document_all_dimensions",
    "unknown": "manual_review",
}


def diagnose_mda_stability_blocker(root: Path = SCORING_ROOT) -> dict[str, Any]:
    batch_dir = root / "06_ratings" / "mda_v2_stability_batch"
    out_dir = root / "PAPER_OUTPUT" / "00_status"
    selection_rows = read_csv_rows(batch_dir / "mda_v2_stability_batch_selection.csv")
    score_rows = read_csv_rows(batch_dir / "mda_v2_stability_document_scores.csv")
    rating_rows = read_csv_rows(batch_dir / "mda_v2_stability_ratings_long.csv")
    review_rows = read_csv_rows(batch_dir / "mda_v2_stability_review_log.csv") + read_csv_rows(root / "07_review" / "review_log.csv")
    numeric_rows = _numeric_rows(root)
    metadata = _metadata_by_doc(root, selection_rows, score_rows)
    text_cache: dict[str, str] = {}

    selected_ids = [row.get("document_id", "") for row in selection_rows if row.get("document_id")]
    scored_ids = {row.get("document_id", "") for row in score_rows if row.get("document_id")}
    score_by_doc = {row.get("document_id", ""): row for row in score_rows if row.get("document_id")}
    ratings_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rating_rows:
        if row.get("document_id"):
            ratings_by_doc[row["document_id"]].append(row)
    review_by_doc: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in review_rows:
        if row.get("document_id"):
            review_by_doc[row["document_id"]].append(row)

    cases: list[dict[str, Any]] = []
    for document_id in selected_ids:
        doc_meta = metadata.get(document_id, {})
        per_document_path = batch_dir / "per_document" / f"{document_id}_parsed.json"
        per_doc = _load_per_document(per_document_path)
        doc_ratings = ratings_by_doc.get(document_id, [])
        is_success = document_id in scored_ids
        text = _text_for_doc(root, doc_meta, document_id, text_cache)

        if not is_success:
            cases.extend(_failed_document_cases(document_id, doc_meta, per_doc, per_document_path))
            present = {row.get("dimension_code", "") for row in doc_ratings}
            for missing in sorted(set(DIMENSION_CODES) - present):
                cases.append(
                    _case(
                        document_id=document_id,
                        meta=doc_meta,
                        dimension_code=missing,
                        stability_case_type="stability_batch",
                        failure_reason="missing_dimension",
                        source_file=str(batch_dir / "mda_v2_stability_ratings_long.csv"),
                        raw_output_path=_result_field(per_doc, "raw_output_path"),
                        parsed_output_path=_result_field(per_doc, "parsed_output_path") or str(per_document_path),
                        needs_rescore=True,
                    )
                )
        if is_success and not doc_ratings:
            cases.append(
                _case(
                    document_id=document_id,
                    meta=doc_meta,
                    dimension_code="ALL",
                    stability_case_type="stability_batch",
                    failure_reason="missing_output",
                    source_file=str(batch_dir / "mda_v2_stability_ratings_long.csv"),
                    parsed_output_path=str(per_document_path),
                    needs_rescore=True,
                )
            )
        cases.extend(_rating_cases(document_id, doc_meta, is_success, doc_ratings, text, score_by_doc.get(document_id, {}), batch_dir, include_missing=is_success))

        numeric = numeric_rows.get(document_id, {})
        if _numeric_has_issue(numeric):
            cases.append(
                _case(
                    document_id=document_id,
                    meta=doc_meta,
                    dimension_code="AR02",
                    stability_case_type="numeric_checker",
                    failure_reason="AR02_numeric_discrepancy",
                    source_file=numeric.get("numeric_check_path", "") or "03_numeric_checks",
                    parsed_output_path=str(per_document_path),
                    evidence_locator=_ar02_evidence(doc_ratings),
                    evidence_exists=_evidence_exists(_ar02_evidence(doc_ratings), text),
                    score_current=score_by_doc.get(document_id, {}).get("AR02", ""),
                    needs_rescore=False,
                )
            )

        for review in review_by_doc.get(document_id, []):
            review_text = (review.get("review_reason", "") + " " + review.get("review_type", "")).lower()
            if "score difference" in review_text:
                dim = review.get("dimension_code", "") or "ALL"
                cases.append(
                    _case(
                        document_id=document_id,
                        meta=doc_meta,
                        dimension_code=dim,
                        stability_case_type="review_log",
                        failure_reason="score_difference_too_large",
                        source_file=str(batch_dir / "mda_v2_stability_review_log.csv"),
                        raw_output_path=review.get("raw_output_path", ""),
                        evidence_locator=review.get("evidence_locator", ""),
                        evidence_exists=_evidence_exists(review.get("evidence_locator", ""), text),
                        score_current=review.get("raw_score", ""),
                        needs_rescore=True,
                    )
                )

    cases.extend(_stability_component_cases(root, metadata))
    distribution = Counter(case["failure_reason"] for case in cases)
    for reason in FAILURE_REASONS:
        distribution.setdefault(reason, 0)
    doc_counter = Counter(case["document_id"] for case in cases if case["document_id"])
    dim_counter = Counter(case["dimension_code"] or "UNKNOWN" for case in cases if case["dimension_code"])
    failed_documents = sorted(set(selected_ids) - scored_ids)
    selected_count = len(selected_ids)
    success_count = len(scored_ids & set(selected_ids))
    success_rate = round(success_count / selected_count, 6) if selected_count else 0.0
    summary = {
        "selected_count": selected_count,
        "success_count": success_count,
        "failure_count": max(0, selected_count - success_count),
        "success_rate": success_rate,
        "success_fraction": f"{success_count}/{selected_count}" if selected_count else "0/0",
        "is_16_over_27": success_count == 16 and selected_count == 27,
        "failed_document_ids": failed_documents,
        "failure_type_distribution": dict(distribution),
        "top_unstable_documents": [{"document_id": doc, "case_count": count} for doc, count in doc_counter.most_common(10)],
        "top_unstable_dimensions": [{"dimension_code": dim, "case_count": count} for dim, count in dim_counter.most_common(10)],
        "case_count": len(cases),
        "repairable_case_count": sum(1 for case in cases if case["recommended_action"] in {"repair_json_only", "fix_evidence_mapping", "fix_numeric_checker"}),
        "rescore_case_count": sum(1 for case in cases if case["recommended_action"] in {"rerun_single_dimension", "rerun_failed_dimension_only", "rerun_document_all_dimensions"}),
        "manual_review_case_count": sum(1 for case in cases if case["recommended_action"] == "manual_review"),
        "stability_success_rate_explanation": "numerator is selected MD&A stability-batch documents present in mda_v2_stability_document_scores; denominator is mda_v2_stability_batch_selection selected documents",
    }
    write_csv_rows(out_dir / "mda_stability_blocker_cases.csv", CASE_HEADERS, cases)
    save_json(out_dir / "mda_stability_blocker_diagnosis.json", summary)
    _write_report(out_dir / "mda_stability_blocker_diagnosis.md", summary)
    return summary


def _metadata_by_doc(root: Path, selection_rows: list[dict[str, str]], score_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(root / "01_registry" / "mda_registry.csv") + score_rows + selection_rows:
        doc_id = row.get("document_id", "")
        if not doc_id:
            continue
        merged = metadata.setdefault(doc_id, {})
        for key in ["ticker", "company_name", "report_year", "text_path", "stock_code"]:
            if row.get(key) and not merged.get(key):
                merged[key] = row[key]
    return metadata


def _numeric_rows(root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for rel_path in [
        "03_numeric_checks/mda_v2_numeric_checks_summary.csv",
        "03_numeric_checks/mda_numeric_checks_summary.csv",
    ]:
        for row in read_csv_rows(root / rel_path):
            doc_id = row.get("document_id", "")
            if doc_id and doc_id not in rows:
                row = dict(row)
                row["numeric_check_path"] = str(root / rel_path)
                rows[doc_id] = row
    return rows


def _load_per_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "failed",
            "result": {
                "error_type": "missing_output",
                "error_message": "missing per-document parsed output",
                "parsed_output_path": str(path),
            },
        }
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "result": {
                "parse_status": "parse_failed",
                "schema_validation_status": "invalid",
                "error_type": "json_parse",
                "parsed_output_path": str(path),
            },
        }
    return loaded if isinstance(loaded, dict) else {"status": "failed", "result": {"error_type": "wrong_root_type", "parsed_output_path": str(path)}}


def _failed_document_cases(document_id: str, meta: dict[str, str], per_doc: dict[str, Any], per_document_path: Path) -> list[dict[str, Any]]:
    result = per_doc.get("result", {}) if isinstance(per_doc.get("result"), dict) else {}
    text = " ".join(
        str(result.get(field, ""))
        for field in ["error_type", "error_message", "parse_status", "schema_validation_status", "status"]
    ).lower()
    reasons: list[str] = []
    if "502" in text:
        reasons.append("ollama_502_error")
    elif "http" in text or "ollama" in text:
        reasons.append("ollama_http_error")
    if "too long" in text or "context" in text or "num_ctx" in text:
        reasons.append("text_too_long")
    if "extract" in text:
        reasons.append("text_extraction_issue")
    if "json" in text or result.get("parse_status") == "parse_failed":
        reasons.append("json_parse_failure")
    if "schema" in text or result.get("schema_validation_status") == "invalid":
        reasons.append("schema_validation_failure")
    if "dimension_count" in text or "dimension_codes" in text:
        reasons.append("structure_comparison_failure")
    if not reasons:
        reasons.append("missing_output" if "missing" in text or not _result_field(per_doc, "raw_output_path") else "unknown")
    return [
        _case(
            document_id=document_id,
            meta=meta,
            dimension_code="ALL",
            stability_case_type="stability_batch",
            failure_reason=reason,
            source_file=str(per_document_path),
            raw_output_path=_result_field(per_doc, "raw_output_path"),
            parsed_output_path=_result_field(per_doc, "parsed_output_path") or str(per_document_path),
            needs_rescore=RECOMMENDED_ACTIONS[reason] != "repair_json_only",
        )
        for reason in _dedupe(reasons)
    ]


def _rating_cases(
    document_id: str,
    meta: dict[str, str],
    is_success: bool,
    ratings: list[dict[str, str]],
    text: str,
    score_row: dict[str, str],
    batch_dir: Path,
    *,
    include_missing: bool,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    present = {row.get("dimension_code", "") for row in ratings}
    if include_missing:
        for missing in sorted(set(DIMENSION_CODES) - present):
            cases.append(
                _case(
                    document_id=document_id,
                    meta=meta,
                    dimension_code=missing,
                    stability_case_type="stability_batch",
                    failure_reason="missing_dimension",
                    source_file=str(batch_dir / "mda_v2_stability_ratings_long.csv"),
                    score_current=score_row.get(missing, ""),
                    needs_rescore=True,
                )
            )
    for row in ratings:
        dimension = row.get("dimension_code", "")
        evidence = row.get("evidence_locator", "")
        exists = _evidence_exists(evidence, text)
        base = {
            "document_id": document_id,
            "meta": meta,
            "dimension_code": dimension,
            "stability_case_type": "stability_batch",
            "source_file": str(batch_dir / "mda_v2_stability_ratings_long.csv"),
            "raw_output_path": row.get("raw_output_path", ""),
            "parsed_output_path": "",
            "evidence_locator": evidence,
            "evidence_exists": exists,
            "score_current": row.get("raw_score", ""),
        }
        if not evidence.strip() or exists is False:
            cases.append(_case(**base, failure_reason="evidence_locator_failure", needs_rescore=False))
        if row.get("schema_validation_status", "valid").lower() not in {"", "valid"}:
            cases.append(_case(**base, failure_reason="schema_validation_failure", needs_rescore=False))
        if row.get("parse_status", "parsed").lower() not in {"", "parsed"}:
            cases.append(_case(**base, failure_reason="json_parse_failure", needs_rescore=False))
        if row.get("confidence_level", "").lower() == "low":
            cases.append(_case(**base, failure_reason="low_confidence", needs_rescore=False))
    return cases


def _stability_component_cases(root: Path, metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    specs = [
        (
            root / "11_stability_analysis" / "outputs" / "mda_repeatability_comparison.csv",
            "repeatability_failure",
            "repeatability",
            "run1_total_score",
            "run2_total_score",
            "abs_total_score_diff",
        ),
        (
            root / "11_stability_analysis" / "outputs" / "mda_version_comparison.csv",
            "version_comparison_failure",
            "version_comparison",
            "total_score_a",
            "total_score_b",
            "abs_total_score_diff",
        ),
        (
            root / "11_stability_analysis" / "outputs" / "mda_structure_comparison.csv",
            "structure_comparison_failure",
            "structure_comparison",
            "score_document_level",
            "score_single_dimension",
            "abs_total_score_diff",
        ),
    ]
    for path, reason, case_type, current_field, comparison_field, diff_field in specs:
        for row in read_csv_rows(path):
            diff = _abs_float(row.get(diff_field, ""))
            status_text = " ".join(row.values()).lower()
            if diff <= 10 and not any(token in status_text for token in ["failed", "unstable", "high_volatility", "major"]):
                continue
            doc_id = row.get("document_id", "")
            meta = {**metadata.get(doc_id, {}), **row}
            cases.append(
                _case(
                    document_id=doc_id,
                    meta=meta,
                    dimension_code="ALL",
                    stability_case_type=case_type,
                    failure_reason=reason,
                    source_file=str(path),
                    score_current=row.get(current_field, ""),
                    score_comparison=row.get(comparison_field, ""),
                    score_diff=row.get(diff_field, ""),
                    needs_rescore=False,
                )
            )
    return cases


def _case(
    *,
    document_id: str,
    meta: dict[str, str],
    dimension_code: str,
    stability_case_type: str,
    failure_reason: str,
    source_file: str = "",
    raw_output_path: str = "",
    parsed_output_path: str = "",
    evidence_locator: str = "",
    evidence_exists: bool | str = "",
    score_current: Any = "",
    score_comparison: Any = "",
    score_diff: Any = "",
    needs_rescore: bool = False,
) -> dict[str, Any]:
    if failure_reason not in FAILURE_REASONS:
        failure_reason = "unknown"
    if evidence_exists == "":
        evidence_value = ""
    else:
        evidence_value = str(bool(evidence_exists)).lower()
    return {
        "document_id": document_id,
        "ticker": meta.get("ticker", ""),
        "company_name": meta.get("company_name", ""),
        "report_year": meta.get("report_year", ""),
        "dimension_code": dimension_code or "ALL",
        "stability_case_type": stability_case_type,
        "failure_reason": failure_reason,
        "source_file": source_file,
        "raw_output_path": raw_output_path,
        "parsed_output_path": parsed_output_path,
        "evidence_locator": evidence_locator,
        "evidence_exists": evidence_value,
        "score_current": score_current,
        "score_comparison": score_comparison,
        "score_diff": score_diff,
        "needs_rescore": str(bool(needs_rescore)).lower(),
        "recommended_action": RECOMMENDED_ACTIONS[failure_reason],
    }


def _text_for_doc(root: Path, meta: dict[str, str], document_id: str, cache: dict[str, str]) -> str:
    if document_id in cache:
        return cache[document_id]
    candidates = []
    if meta.get("text_path"):
        candidates.append(root / meta["text_path"])
    candidates.append(root / "02_extracted_text" / "mda" / f"{document_id}.txt")
    text = ""
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            break
    cache[document_id] = text
    return text


def _evidence_exists(evidence_locator: str, text: str) -> bool | str:
    locators = re.findall(r"MDA_P\d{3}", evidence_locator or "")
    if not locators:
        return "" if not evidence_locator else False
    if not text:
        return False
    return all(f"[{locator}]" in text or locator in text for locator in locators)


def _ar02_evidence(rows: list[dict[str, str]]) -> str:
    for row in rows:
        if row.get("dimension_code") == "AR02":
            return row.get("evidence_locator", "")
    return ""


def _result_field(per_doc: dict[str, Any], field: str) -> str:
    result = per_doc.get("result", {}) if isinstance(per_doc.get("result"), dict) else {}
    return str(result.get(field, "") or per_doc.get(field, "") or "")


def _numeric_has_issue(row: dict[str, str]) -> bool:
    try:
        material = int(float(row.get("num_material_discrepancies") or 0))
        severe = int(float(row.get("num_severe_direction_conflicts") or 0))
    except ValueError:
        return False
    return material > 0 or severe > 0


def _abs_float(value: str) -> float:
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MD&A Stability Blocker Diagnosis",
        "",
        "## Success Rate Source",
        f"- numerator_success_count: {summary['success_count']}",
        f"- denominator_selected_count: {summary['selected_count']}",
        f"- success_fraction: {summary['success_fraction']}",
        f"- equals_16_over_27: {str(summary['is_16_over_27']).lower()}",
        f"- mda_stability_success_rate: {summary['success_rate']}",
        f"- definition: {summary['stability_success_rate_explanation']}",
        "",
        "## Failed Documents",
    ]
    if summary["failed_document_ids"]:
        lines.extend(f"- {doc_id}" for doc_id in summary["failed_document_ids"])
    else:
        lines.append("- none")
    lines.extend(["", "## Failure Reason Distribution"])
    for reason in FAILURE_REASONS:
        lines.append(f"- {reason}: {summary['failure_type_distribution'].get(reason, 0)}")
    lines.extend(["", "## Top Failed Dimensions"])
    for item in summary["top_unstable_dimensions"]:
        lines.append(f"- {item['dimension_code']}: {item['case_count']} cases")
    lines.extend(["", "## Top Failed Documents"])
    for item in summary["top_unstable_documents"]:
        lines.append(f"- {item['document_id']}: {item['case_count']} cases")
    lines.extend(
        [
            "",
            "## Repairability",
            f"- parser_or_mapping_repairable_case_count: {summary['repairable_case_count']}",
            f"- rescore_case_count: {summary['rescore_case_count']}",
            f"- manual_review_case_count: {summary['manual_review_case_count']}",
            "",
            "## Guardrails",
            "- The 0.85 freeze threshold was not lowered.",
            "- Failed samples were not deleted from the denominator.",
            "- Cases CSV records actions without overwriting v1/v2/pilot/sample outputs.",
        ]
    )
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the MD&A stability blocker with strict failure classes.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(diagnose_mda_stability_blocker(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
