from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from json_stabilization import parse_model_json
from scoring_utils import SCORING_ROOT
from validate_json_outputs import validate_simple_dimension_json


def repair_then_validate_json(raw_text: str, doc_type: str, dimension_code: str, text: str) -> dict[str, Any]:
    parsed = parse_model_json(raw_text or "")
    parse_status = "parsed" if parsed.get("ok") else "parse_failed"
    candidate = parsed.get("payload") if parsed.get("ok") else None
    issues = validate_simple_dimension_json(candidate, doc_type, dimension_code, text) if isinstance(candidate, dict) else ["json_parse_failed"]
    if isinstance(candidate, dict) and not issues:
        return {
            "status": "valid",
            "payload": candidate,
            "parse_status": parse_status,
            "schema_validation_status": "valid",
            "issues": [],
            "failure_type": parsed.get("failure_type", ""),
            "recommended_fix": parsed.get("recommended_fix", ""),
        }
    return {
        "status": "manual_required",
        "payload": None,
        "parse_status": parse_status,
        "schema_validation_status": "invalid",
        "issues": sorted(set(issues)),
        "failure_type": parsed.get("failure_type", "json_parse_failed"),
        "recommended_fix": parsed.get("recommended_fix", "rescore_dimensionwise"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair and validate one raw simple dimension JSON output.")
    parser.add_argument("raw_output_path", type=Path)
    parser.add_argument("text_path", type=Path)
    parser.add_argument("--doc-type", choices=["mda", "news"], required=True)
    parser.add_argument("--dimension-code", required=True)
    args = parser.parse_args()
    raw_text = args.raw_output_path.read_text(encoding="utf-8", errors="replace")
    text = args.text_path.read_text(encoding="utf-8", errors="replace")
    result = repair_then_validate_json(raw_text, args.doc_type, args.dimension_code, text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "valid" else 1


if __name__ == "__main__":
    raise SystemExit(main())
