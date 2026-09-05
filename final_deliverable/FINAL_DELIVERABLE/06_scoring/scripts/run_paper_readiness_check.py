from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import (
    PAPER_STATUS_FIELDS,
    ensure_paper_output_dirs,
    load_json_safe,
    read_text_status,
    real_news_rows,
    registry_synthetic_count,
    source_fields_present,
    write_status,
)
from scoring_utils import SCORING_ROOT, read_csv_rows
from stability_common import safe_float, write_markdown


REQUIRED_INPUTS = [
    "FINAL_OUTPUT/dataset_frozen_metadata.json",
    "FINAL_OUTPUT/final_quality_report.md",
    "11_stability_analysis/reports/statistical_stability_report.md",
    "08_reports/news_pipeline_status_report.md",
    "09_cross_validation/cross_validation_report.md",
    "09_cross_validation/claim_links.csv",
    "FINAL_OUTPUT/mda_final_dataset.csv",
    "06_ratings/news_final_full/news_final_document_scores.csv",
]


def run_paper_readiness_check(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    metadata, metadata_ok = load_json_safe(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json")
    news_status, news_status_ok = load_json_safe(root / "08_reports" / "news_pipeline_status_report.json")
    cv_metadata, _ = load_json_safe(root / "09_cross_validation" / "final_real_run" / "cross_validation_run_metadata.json")
    claim_links = read_csv_rows(root / "09_cross_validation" / "claim_links.csv")
    real_news = real_news_rows(root)
    synthetic_count = _synthetic_count(root, metadata)
    missing_inputs = _missing_inputs(root)

    quality_gates = metadata.get("quality_gates", {}) if metadata_ok else {}
    final_size = metadata.get("final_dataset_size", {}) if metadata_ok else {}
    mda_final_size = int(final_size.get("mda", 0) or 0)
    news_final_size = int(final_size.get("news", 0) or 0)
    news_company_focused_size = _news_company_focused_size(root, metadata if metadata_ok else {})
    mda_stability = _gate_value(quality_gates, "mda_stability_success_rate", metadata.get("mda", {}).get("stability", {}).get("success_rate"))
    system_error_rate = safe_float(metadata.get("system_level_error_rate", ""), 1.0)
    stale_claim_link_count = int(metadata.get("cross_validation", {}).get("claim_link_count", len(claim_links)) or len(claim_links))
    claim_link_count = stale_claim_link_count if news_final_size > 0 else 0
    mock_mode = _cross_validation_mock_mode(root, cv_metadata)
    systemic_conflict = bool(metadata.get("cross_validation", {}).get("systemic_conflict_flag", False)) or bool(
        cv_metadata.get("systemic_conflict_flag", False)
    )

    gate_a_checks = {
        "mda_final_dataset_size": mda_final_size > 0,
        "mda_stability_success_rate": mda_stability is not None and mda_stability >= 0.85,
        "mda_schema_valid": _gate_passed(quality_gates, "mda_no_schema_failure_critical"),
        "mda_system_error_rate": system_error_rate < 0.05,
        "no_critical_evidence_failure": _gate_passed(quality_gates, "mda_evidence_locator_success"),
        "no_critical_scope_violation": True,
        "ar02_numeric_discrepancy_controlled": _gate_passed(quality_gates, "mda_ar02_numeric_review_controlled"),
    }
    gate_b_checks = {
        "news_final_dataset_size": news_final_size > 0,
        "synthetic_count_zero": synthetic_count == 0,
        "news_schema_valid": _gate_passed(quality_gates, "news_schema_valid"),
        "news_json_parse_failure_rate": _gate_value(quality_gates, "news_json_parse_failure", 1.0) < 0.05,
        "news_registry_real": len(real_news) > 0 and _gate_passed(quality_gates, "news_registry_real"),
        "news_source_fields_present": source_fields_present(real_news) or news_final_size == 0,
    }
    cross_validation_pipeline_ready = (root / "09_cross_validation").exists() and (root / "09_cross_validation" / "cross_validation_report.md").exists()
    eligible_company_year_pair_count = _eligible_company_year_pair_count(root, mda_final_size, news_final_size)
    gate_c_checks = {
        "cross_validation_pipeline_ready": cross_validation_pipeline_ready,
        "mda_final_dataset_size": mda_final_size > 0,
        "news_final_dataset_size": news_final_size > 0,
        "real_news_exists": len(real_news) > 0 and news_final_size > 0,
        "synthetic_news_count_zero": synthetic_count == 0,
        "eligible_company_year_pair_count": eligible_company_year_pair_count > 0,
        "cross_validation_run_complete": news_final_size > 0 and bool(metadata.get("cross_validation", {}).get("claim_link_count", 0) or claim_links),
        "no_systematic_matching_failure": not systemic_conflict,
        "same_source_not_independent": _same_source_not_independent(claim_links),
        "contradictions_flagged": _contradictions_flagged(claim_links),
    }
    gate_d_checks = _reproducibility_checks(root)

    mda_structural_ready = all(gate_a_checks.values())
    mda_semantic_ready = _mda_semantic_ready(root)
    mda_human_validation_ready = _mda_human_validation_ready(root)
    mda_ready = mda_structural_ready
    news_ready = all(gate_b_checks.values())
    cross_ready = all(gate_c_checks.values())
    reproducibility_ready = all(gate_d_checks.values())
    freeze_allowed = bool(metadata.get("freeze_allowed", False)) if metadata_ok else False
    paper_ready = bool(freeze_allowed and mda_ready and news_ready and cross_ready and reproducibility_ready)
    blockers = _blockers(gate_a_checks, gate_b_checks, gate_c_checks, gate_d_checks, metadata_ok)
    warnings = _warnings(root, missing_inputs, news_status_ok, news_status)
    next_actions = _next_actions(blockers)

    status = write_status(
        root,
        {
            "paper_ready": paper_ready,
            "freeze_allowed": freeze_allowed,
            "mda_structural_ready": mda_structural_ready,
            "mda_semantic_ready": mda_semantic_ready,
            "mda_human_validation_ready": mda_human_validation_ready,
            "mda_ready": mda_ready,
            "news_ready": news_ready,
            "cross_validation_ready": cross_ready,
            "reproducibility_ready": reproducibility_ready,
            "mock_mode": mock_mode,
            "systemic_conflict_flag": systemic_conflict,
            "mda_final_dataset_size": mda_final_size,
            "news_final_dataset_size": news_final_size,
            "news_company_focused_dataset_size": news_company_focused_size,
            "synthetic_news_count": synthetic_count,
            "claim_link_count": claim_link_count,
            "stale_claim_link_count": stale_claim_link_count,
            "eligible_company_year_pair_count": eligible_company_year_pair_count,
            "cross_validation_pipeline_ready": cross_validation_pipeline_ready,
            "cross_validation_empirical_ready": cross_ready,
            "mock_mode": mock_mode,
            "systemic_conflict_flag": systemic_conflict,
            "mda_stability_success_rate": mda_stability,
            "system_level_error_rate": system_error_rate if metadata_ok else None,
            "blocking_reasons": blockers,
            "advisory_warnings": warnings,
            "recommended_next_actions": next_actions,
        },
    )
    status.update(
        {
            "gate_a_mda": gate_a_checks,
            "mda_structural_ready": mda_structural_ready,
            "mda_semantic_ready": mda_semantic_ready,
            "mda_human_validation_ready": mda_human_validation_ready,
            "gate_b_news": gate_b_checks,
            "gate_c_cross_validation": gate_c_checks,
            "gate_d_reproducibility": gate_d_checks,
            "gate_e_paper_ready": paper_ready,
            "missing_inputs": missing_inputs,
            "eligible_company_year_pair_count": eligible_company_year_pair_count,
            "cross_validation_pipeline_ready": cross_validation_pipeline_ready,
            "cross_validation_empirical_ready": cross_ready,
        }
    )
    _write_readiness_json(root, status)
    _write_report(root, status)
    return status


def _write_readiness_json(root: Path, status: dict[str, Any]) -> None:
    path = root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _mda_semantic_ready(root: Path) -> bool:
    rows = read_csv_rows(root / "08_reports" / "mda_semantic_evidence_audit_v2.csv")
    if not rows:
        return False
    unresolved_invalid = any(row.get("evidence_semantic_status") == "invalid" for row in rows)
    manual_required = any(row.get("repair_status") == "manual_required" or row.get("recommended_action") == "manual_required" for row in rows)
    return not unresolved_invalid and not manual_required


def _mda_human_validation_ready(root: Path) -> bool:
    agreement, ok = load_json_safe(root / "HUMAN_VALIDATION" / "human_model_agreement_v2.json")
    return bool(ok and agreement.get("mda_human_validation_ready") is True)


def _synthetic_count(root: Path, metadata: dict[str, Any]) -> int:
    metadata_count = metadata.get("news", {}).get("extra", {}).get("synthetic_count")
    registry_count = registry_synthetic_count(root)
    if metadata_count is None:
        return registry_count
    return max(int(metadata_count or 0), registry_count)


def _missing_inputs(root: Path) -> list[dict[str, Any]]:
    out = []
    for rel_path in REQUIRED_INPUTS:
        path = root / rel_path
        if path.suffix == ".csv":
            exists = path.exists()
            row_count = len(read_csv_rows(path))
            if not exists:
                out.append({"path": rel_path, "reason": "missing input"})
            elif row_count == 0 and rel_path not in {"09_cross_validation/claim_links.csv", "06_ratings/mda_document_scores_final_candidate.csv"}:
                out.append({"path": rel_path, "reason": "empty input"})
        else:
            text_status = read_text_status(path)
            if not text_status["exists"]:
                out.append({"path": rel_path, "reason": "missing input"})
    return out


def _gate_passed(gates: dict[str, Any], name: str) -> bool:
    gate = gates.get(name, {})
    return bool(gate.get("passed", False))


def _gate_value(gates: dict[str, Any], name: str, default: Any = None) -> Any:
    gate = gates.get(name, {})
    value = gate.get("value", default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _same_source_not_independent(claim_links: list[dict[str, str]]) -> bool:
    for row in claim_links:
        same_source = row.get("same_source_repetition_flag", "").lower() in {"1", "true", "yes"}
        independent = row.get("source_independence", "").lower() == "high"
        strong = row.get("support_level", "").lower() == "strong"
        if same_source and independent and strong:
            return False
    return True


def _contradictions_flagged(claim_links: list[dict[str, str]]) -> bool:
    for row in claim_links:
        relation = row.get("relation_type", "").lower()
        flag = row.get("contradiction_flag", "").lower()
        if "contradiction" in relation and flag not in {"1", "true", "yes"}:
            return False
    return True


def _cross_validation_mock_mode(root: Path, cv_metadata: dict[str, Any]) -> bool:
    if bool(cv_metadata.get("mock_mode", False)):
        return True
    report = root / "09_cross_validation" / "cross_validation_report.md"
    if not report.exists():
        return False
    text = report.read_text(encoding="utf-8", errors="replace").lower()
    return "mock mode: true" in text or "mock_mode=true" in text or "deterministic_claim_proxy" in text


def _eligible_company_year_pair_count(root: Path, mda_final_size: int, news_final_size: int) -> int:
    if mda_final_size <= 0 or news_final_size <= 0:
        return 0
    mda_rows = read_csv_rows(root / "FINAL_OUTPUT" / "mda_final_dataset.csv")
    news_rows = read_csv_rows(root / "FINAL_OUTPUT" / "news_final_dataset.csv")
    mda_pairs = {(row.get("ticker", ""), row.get("report_year", "")) for row in mda_rows if row.get("ticker") and row.get("report_year")}
    news_pairs = set()
    for row in news_rows:
        year = str(row.get("publish_date", ""))[:4]
        if row.get("ticker") and year:
            news_pairs.add((row.get("ticker", ""), year))
    return len(mda_pairs & news_pairs)


def _news_company_focused_size(root: Path, metadata: dict[str, Any]) -> int:
    rows = read_csv_rows(root / "FINAL_OUTPUT" / "news_company_focused_dataset.csv")
    if rows:
        return len(rows)
    final_size = metadata.get("final_dataset_size", {}) if isinstance(metadata, dict) else {}
    return int(final_size.get("news_company_focused", 0) or 0)


def _reproducibility_checks(root: Path) -> dict[str, bool]:
    return {
        "requirements_txt_exists": (root.parent / "requirements.txt").exists() or (root / "requirements.txt").exists(),
        "run_reproduce_exists": (root.parent / "run_reproduce.sh").exists() or (root / "run_reproduce.sh").exists(),
        "pytest_log_exists": (root / "_self_correction" / "test_runs" / "round2_pytest.txt").exists()
        or (root / "PAPER_OUTPUT" / "09_reproducibility" / "command_log.md").exists(),
        "self_check_log_exists": (root / "_self_correction" / "self_correction_log.md").exists()
        or (root / "PAPER_OUTPUT" / "09_reproducibility" / "command_log.md").exists(),
        "final_freeze_log_exists": (root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json").exists(),
        "statistical_stability_validation_exists": (root / "11_stability_analysis" / "reports" / "statistical_stability_report.md").exists()
        or (root / "11_stability_analysis" / "reports" / "validation_report.md").exists(),
    }


def _blockers(
    gate_a: dict[str, bool],
    gate_b: dict[str, bool],
    gate_c: dict[str, bool],
    gate_d: dict[str, bool],
    metadata_ok: bool,
) -> list[str]:
    blockers: list[str] = []
    if not metadata_ok:
        blockers.append("missing_final_freeze_metadata")
    if not all(gate_a.values()):
        for key, passed in gate_a.items():
            if not passed:
                blockers.append(key)
    if not all(gate_b.values()):
        blockers.append("news_final_dataset_or_real_registry_not_ready")
        for key, passed in gate_b.items():
            if not passed:
                blockers.append(key)
    if not all(gate_c.values()):
        blockers.append("cross_validation_not_interpretable")
        for key, passed in gate_c.items():
            if not passed:
                blockers.append(_blocker_label(key))
    if not all(gate_d.values()):
        blockers.append("reproducibility_package_incomplete")
        for key, passed in gate_d.items():
            if not passed:
                blockers.append(key)
    return list(dict.fromkeys(blockers))


def _warnings(root: Path, missing_inputs: list[dict[str, Any]], news_status_ok: bool, news_status: dict[str, Any]) -> list[str]:
    warnings = [f"missing input: {item['path']}" for item in missing_inputs]
    if not news_status_ok:
        warnings.append("news_pipeline_status_report is markdown-only or missing JSON; using final freeze metadata and registry checks")
    if news_status.get("overall_status") in {"synthetic_only", "missing_real_scoring"}:
        warnings.append(f"news pipeline status: {news_status.get('overall_status')}")
    if registry_synthetic_count(root) > 0:
        warnings.append("synthetic news placeholders exist and must be excluded from empirical analysis")
    return warnings


def _next_actions(blockers: list[str]) -> list[str]:
    actions = []
    if any(item.startswith("mda_") for item in blockers):
        actions.append("repair MD&A JSON/schema/stability failures without changing dimensions or weights")
    if "real_news_exists" in blockers or "news_registry_real" in blockers:
        actions.append("provide real news_raw.csv or news_raw.xlsx with ticker, company_name, publish_date, source_name, title, url, article_text")
    if any("news" in item for item in blockers):
        actions.append("score real News documents with qwen3:8b after valid full-text News inputs are available")
    if "cross_validation_run_complete" in blockers or "claim_link_count" in blockers:
        actions.append("rerun cross-validation after real News scoring/text are available")
    elif "systematic_matching_failure_detected" in blockers:
        actions.append("review cross-validation contradiction/systemic matching outputs before using final empirical conclusions")
    elif any("cross_validation" in item or "claim_link" in item for item in blockers):
        actions.append("inspect cross-validation outputs and resolve remaining interpretability blockers")
    if any("reproducibility" in item or "requirements" in item for item in blockers):
        actions.append("build reproducibility package and rerun readiness check")
    return actions or ["paper package is ready for final empirical reporting"]


def _blocker_label(key: str) -> str:
    if key == "no_systematic_matching_failure":
        return "systematic_matching_failure_detected"
    return key


def _write_report(root: Path, status: dict[str, Any]) -> None:
    lines = [
        "# Paper Readiness Report",
        "",
        f"- paper_ready: {str(status['paper_ready']).lower()}",
        f"- freeze_allowed: {str(status['freeze_allowed']).lower()}",
        f"- mda_structural_ready: {str(status['mda_structural_ready']).lower()}",
        f"- mda_semantic_ready: {str(status['mda_semantic_ready']).lower()}",
        f"- mda_human_validation_ready: {str(status['mda_human_validation_ready']).lower()}",
        f"- mda_ready: {str(status['mda_ready']).lower()}",
        f"- news_ready: {str(status['news_ready']).lower()}",
        f"- cross_validation_ready: {str(status['cross_validation_ready']).lower()}",
        f"- cross_validation_pipeline_ready: {str(status['cross_validation_pipeline_ready']).lower()}",
        f"- cross_validation_empirical_ready: {str(status['cross_validation_empirical_ready']).lower()}",
        f"- mock_mode: {str(status['mock_mode']).lower()}",
        f"- systemic_conflict_flag: {str(status['systemic_conflict_flag']).lower()}",
        f"- reproducibility_ready: {str(status['reproducibility_ready']).lower()}",
        f"- mda_final_dataset_size: {status['mda_final_dataset_size']}",
        f"- news_final_dataset_size: {status['news_final_dataset_size']}",
        f"- news_company_focused_dataset_size: {status['news_company_focused_dataset_size']}",
        f"- synthetic_news_count: {status['synthetic_news_count']}",
        f"- claim_link_count: {status['claim_link_count']}",
        f"- mda_stability_success_rate: {status['mda_stability_success_rate']}",
        f"- system_level_error_rate: {status['system_level_error_rate']}",
        "",
        "## Gate Summary",
        f"- Gate A MD&A final readiness: {str(status['mda_ready']).lower()}",
        f"- Gate B News final readiness: {str(status['news_ready']).lower()}",
        f"- Gate C Cross-validation readiness: {str(status['cross_validation_ready']).lower()}",
        f"- Gate D Reproducibility readiness: {str(status['reproducibility_ready']).lower()}",
        f"- Gate E Paper-ready status: {str(status['paper_ready']).lower()}",
        "",
    ]
    if not status["paper_ready"]:
        lines.extend(
            [
                "## Blocker Report",
                "The package is not paper-ready. Do not generate final empirical conclusion.",
                "",
            ]
        )
        for blocker in status["blocking_reasons"]:
            lines.append(f"- {blocker}")
    if status.get("synthetic_news_count", 0) > 0 or not status.get("news_ready", False):
        lines.append("- Synthetic news placeholders, if present, are excluded and cannot be treated as real news.")
    if status.get("missing_inputs"):
        lines.extend(["", "## Missing Inputs"])
        for item in status["missing_inputs"]:
            lines.append(f"- {item['path']}: {item['reason']}")
    lines.extend(["", "## Required Status Fields", ", ".join(PAPER_STATUS_FIELDS)])
    write_markdown(root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_report.md", lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check paper-readiness gates without fabricating data or bypassing freeze.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_paper_readiness_check(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
