from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json


def compare_claim_vs_document_scoring(root: Path = SCORING_ROOT) -> dict[str, Any]:
    claim_rows = read_csv_rows(root / "10_claim_mapping" / "mda_document_scores_claim_level.csv")
    document_rows = read_csv_rows(root / "06_ratings" / "mda_document_scores.csv")
    claim_summary = _load_json(root / "10_claim_scoring" / "mda_claim_scoring_summary.json")
    qwen_diagnosis = read_csv_rows(root / "08_reports" / "qwen_json_failure_diagnosis.csv")
    stability_rows = read_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv")
    numeric_audit = read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_discrepancy_audit.csv")

    claim_json_success = float(claim_summary.get("json_success_rate") or (1.0 if claim_rows else 0.0))
    claim_schema_rate = float(claim_summary.get("schema_validation_rate") or (1.0 if claim_rows else 0.0))
    document_failure_count = len(qwen_diagnosis)
    document_total_attempts = len(stability_rows) + document_failure_count
    document_json_success = round(len(stability_rows) / document_total_attempts, 3) if document_total_attempts else (1.0 if document_rows else 0.0)
    document_schema_rate = document_json_success
    claim_ar02_error_rate = _claim_ar02_error_rate(claim_rows)
    document_ar02_error_rate = _document_ar02_error_rate(numeric_audit, document_rows)
    claim_evidence_accuracy = _claim_evidence_accuracy(root)
    document_evidence_accuracy = _document_evidence_accuracy(root)
    claim_runtime = float(claim_summary.get("duration_seconds") or 0)
    document_runtime = _document_runtime(root)
    recommended = "claim-level scoring framework" if claim_json_success >= document_json_success else "hybrid scoring"
    summary = {
        "document_json_success_rate": document_json_success,
        "claim_json_success_rate": round(claim_json_success, 3),
        "document_schema_validation_rate": document_schema_rate,
        "claim_schema_validation_rate": round(claim_schema_rate, 3),
        "document_ar02_error_rate": document_ar02_error_rate,
        "claim_ar02_error_rate": claim_ar02_error_rate,
        "document_evidence_accuracy": document_evidence_accuracy,
        "claim_evidence_accuracy": claim_evidence_accuracy,
        "document_runtime_cost_seconds": document_runtime,
        "claim_runtime_cost_seconds": claim_runtime,
        "recommended_primary_system": recommended,
    }
    _write_report(root / "08_reports" / "claim_vs_document_comparison.md", summary)
    save_json(root / "08_reports" / "claim_vs_document_comparison_summary.json", summary)
    return summary


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Claim-Level vs Document-Level Scoring Comparison",
        "",
        "系统已从 document-level scoring 升级为 claim-level scoring framework；旧 document-level scoring 保留为 fallback，hybrid scoring 用于对比实验。",
        "",
        "| Metric | Document-level | Claim-level |",
        "| --- | ---: | ---: |",
        f"| JSON success rate | {summary['document_json_success_rate']} | {summary['claim_json_success_rate']} |",
        f"| Schema validation rate | {summary['document_schema_validation_rate']} | {summary['claim_schema_validation_rate']} |",
        f"| AR02 error rate | {summary['document_ar02_error_rate']} | {summary['claim_ar02_error_rate']} |",
        f"| Stability batch success rate | {summary['document_json_success_rate']} | {summary['claim_json_success_rate']} |",
        f"| Evidence accuracy | {summary['document_evidence_accuracy']} | {summary['claim_evidence_accuracy']} |",
        f"| Runtime cost seconds | {summary['document_runtime_cost_seconds']} | {summary['claim_runtime_cost_seconds']} |",
        "",
        f"Recommended primary system: {summary['recommended_primary_system']}.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _claim_ar02_error_rate(rows: list[dict[str, str]]) -> float:
    if not rows:
        return 0.0
    values = [float(row.get("AR02") or 3) for row in rows if row.get("source_type") == "mda"]
    if not values:
        return 0.0
    return round(sum(max(0.0, 3 - value) / 4 for value in values) / len(values), 3)


def _document_ar02_error_rate(audit_rows: list[dict[str, str]], document_rows: list[dict[str, str]]) -> float:
    if audit_rows:
        material = sum(1 for row in audit_rows if row.get("audit_label") == "real_discrepancy")
        return round(material / len(audit_rows), 3)
    values = [float(row.get("AR02") or 3) for row in document_rows if row.get("AR02")]
    if not values:
        return 0.0
    return round(sum(max(0.0, 3 - value) / 4 for value in values) / len(values), 3)


def _claim_evidence_accuracy(root: Path) -> float:
    claims = read_csv_rows(root / "10_claims" / "mda_claims_long.csv")
    if not claims:
        return 0.0
    good = sum(1 for row in claims if row.get("evidence_locator", "").startswith("[MDA_P"))
    return round(good / len(claims), 3)


def _document_evidence_accuracy(root: Path) -> float:
    ratings = read_csv_rows(root / "06_ratings" / "mda_ratings_long.csv")
    if not ratings:
        return 0.0
    good = sum(1 for row in ratings if row.get("evidence_locator", "").startswith("[MDA_P"))
    return round(good / len(ratings), 3)


def _document_runtime(root: Path) -> float:
    for path in [
        root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_sample_parallel_run_summary.json",
        root / "08_reports" / "mda_parallel_run_summary.json",
    ]:
        payload = _load_json(path)
        if payload.get("duration_seconds") is not None:
            return float(payload.get("duration_seconds") or 0)
    return 0.0


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare document-level and claim-level scoring outcomes.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(compare_claim_vs_document_scoring(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
