from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from diagnose_qwen_json_failures import RUN_SPECS, _load_failed_records, _read_raw_model_text, _resolve_raw_path
from json_stabilization import is_parser_repairable, parse_model_json
from scoring_utils import SCORING_ROOT, save_json, write_csv_rows


REPAIR_HEADERS = [
    "document_id",
    "source_run",
    "raw_output_path",
    "repair_status",
    "failure_type",
    "recommended_fix",
    "repaired_json_path",
    "note",
]


def repair_existing_failed_outputs(root: Path = SCORING_ROOT) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    repaired_count = 0
    for spec in RUN_SPECS:
        source_run = spec["source_run"]
        raw_dir = root / spec["raw_dir"]
        ratings_dir = root / spec["ratings_dir"]
        for record_path, record in _load_failed_records(ratings_dir):
            document_id = str(record.get("document_id") or record_path.name.removesuffix("_parsed.json"))
            result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
            raw_path = _resolve_raw_path(root, raw_dir, document_id, result.get("raw_output_path", ""))
            parsed = parse_model_json(_read_raw_model_text(raw_path))
            repair_status = "not_repairable"
            repaired_json_path = ""
            note = "requires dimensionwise rescore or manual review"
            if is_parser_repairable(parsed):
                repaired_json_path = str(root / "08_reports" / "repaired_qwen_json" / source_run / f"{document_id}.json")
                save_json(Path(repaired_json_path), parsed["payload"])
                repair_status = "parser_repaired"
                repaired_count += 1
                note = "parser recovered a JSON object; schema still requires separate validation"
            rows.append(
                {
                    "document_id": document_id,
                    "source_run": source_run,
                    "raw_output_path": str(raw_path),
                    "repair_status": repair_status,
                    "failure_type": parsed.get("failure_type", "unknown"),
                    "recommended_fix": parsed.get("recommended_fix", "manual_required"),
                    "repaired_json_path": repaired_json_path,
                    "note": note,
                }
            )
    write_csv_rows(root / "08_reports" / "qwen_json_repair_attempts.csv", REPAIR_HEADERS, rows)
    summary = {
        "attempted_count": len(rows),
        "parser_repaired_count": repaired_count,
        "not_repairable_count": len(rows) - repaired_count,
        "output_csv": "06_scoring/08_reports/qwen_json_repair_attempts.csv",
        "repaired_json_dir": "06_scoring/08_reports/repaired_qwen_json",
    }
    save_json(root / "08_reports" / "qwen_json_repair_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair parser-recoverable qwen3:8b JSON failures into isolated report artifacts.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(repair_existing_failed_outputs(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
