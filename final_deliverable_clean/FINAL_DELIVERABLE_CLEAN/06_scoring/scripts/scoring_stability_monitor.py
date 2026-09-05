from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, save_json


THRESHOLDS = {
    "json_success_rate": 0.98,
    "schema_success_rate": 0.98,
    "evidence_match_rate": 0.98,
    "system_failure_rate": 0.05,
}


def monitor_scoring_stability(root: Path = SCORING_ROOT, output_name: str = "single_dimension") -> dict[str, Any]:
    records = _load_dimension_records(root / "06_ratings" / output_name / "per_dimension")
    total = len(records)
    parsed = sum(_result(record).get("parse_status") == "parsed" for record in records)
    schema_valid = sum(_result(record).get("schema_validation_status") == "valid" for record in records)
    evidence_found = sum(_evidence_found(record) for record in records)
    failures = sum(record.get("status") != "success" for record in records)
    summary = {
        "output_name": output_name,
        "dimension_record_count": total,
        "json_success_rate": round(parsed / total, 6) if total else 0.0,
        "schema_success_rate": round(schema_valid / total, 6) if total else 0.0,
        "evidence_match_rate": round(evidence_found / total, 6) if total else 0.0,
        "system_failure_rate": round(failures / total, 6) if total else 0.0,
        "manual_required_count": sum(_result(record).get("error_type") == "manual_required" or record.get("status") != "success" for record in records),
        "violations": [],
    }
    violations = []
    for metric in ["json_success_rate", "schema_success_rate", "evidence_match_rate"]:
        if summary[metric] < THRESHOLDS[metric]:
            violations.append(f"{metric}_below_{THRESHOLDS[metric]}")
    if summary["system_failure_rate"] >= THRESHOLDS["system_failure_rate"]:
        violations.append("system_failure_rate_at_or_above_0.05")
    summary["violations"] = violations
    summary["passes_thresholds"] = not violations
    report_dir = root / "08_reports"
    save_json(report_dir / f"{output_name}_stability_monitor.json", summary)
    _write_report(report_dir / f"{output_name}_stability_monitor.md", summary)
    return summary


def _load_dimension_records(per_dimension_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(per_dimension_dir.glob("*_parsed.json")):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            records.append({"status": "failed", "result": {"parse_status": "parse_failed", "schema_validation_status": "invalid", "error_type": "json_decode_error"}})
    return records


def _result(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    return result if isinstance(result, dict) else {}


def _evidence_found(record: dict[str, Any]) -> bool:
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
    locator = str(payload.get("evidence_locator", ""))
    return bool(locator and record.get("status") == "success")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Scoring Stability Monitor",
        "",
        f"- output_name: {summary['output_name']}",
        f"- dimension_record_count: {summary['dimension_record_count']}",
        f"- json_success_rate: {summary['json_success_rate']}",
        f"- schema_success_rate: {summary['schema_success_rate']}",
        f"- evidence_match_rate: {summary['evidence_match_rate']}",
        f"- system_failure_rate: {summary['system_failure_rate']}",
        f"- manual_required_count: {summary['manual_required_count']}",
        f"- passes_thresholds: {summary['passes_thresholds']}",
        f"- violations: {', '.join(summary['violations']) if summary['violations'] else 'none'}",
        "",
        "If thresholds are not met, reduce output complexity before changing model capability.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor JSON/schema/evidence/system stability for single-dimension scoring outputs.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()
    print(json.dumps(monitor_scoring_stability(args.root, args.output_name), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
