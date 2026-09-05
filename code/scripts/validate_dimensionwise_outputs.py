from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, dimensions_for, read_csv_rows, save_json, weights_for


def validate_dimensionwise_outputs(
    root: Path = SCORING_ROOT,
    output_name: str = "mda_v2_dimensionwise",
    ratings_prefix: str = "mda_v2_dimensionwise",
) -> dict[str, Any]:
    out_dir = root / "06_ratings" / output_name
    document_scores = read_csv_rows(out_dir / f"{ratings_prefix}_document_scores.csv")
    ratings_long = read_csv_rows(out_dir / f"{ratings_prefix}_ratings_long.csv")
    failed_cases = read_csv_rows(out_dir / f"{ratings_prefix}_failed_cases.csv")
    expected_codes = [item["code"] for item in dimensions_for("mda")]
    issues: list[str] = []

    seen_doc_ids: set[str] = set()
    for row in document_scores:
        doc_id = row.get("document_id", "")
        if not doc_id:
            issues.append("blank_document_id")
            continue
        if doc_id in seen_doc_ids:
            issues.append(f"duplicate_document_id:{doc_id}")
        seen_doc_ids.add(doc_id)
        if int(float(row.get("dimension_count") or 0)) != 5:
            issues.append(f"dimension_count_not_5:{doc_id}")
        missing_codes = [code for code in expected_codes if not row.get(code)]
        if missing_codes:
            issues.append(f"missing_document_score_dimensions:{doc_id}:{','.join(missing_codes)}")
        expected_total = _expected_total(row)
        observed_total = float(row.get("total_score_100") or -1)
        if abs(observed_total - expected_total) > 0.000001:
            issues.append(f"total_score_formula_mismatch:{doc_id}")

    ratings_by_doc: dict[str, set[str]] = {}
    for row in ratings_long:
        ratings_by_doc.setdefault(row.get("document_id", ""), set()).add(row.get("dimension_code", ""))
    for doc_id in seen_doc_ids:
        if ratings_by_doc.get(doc_id, set()) != set(expected_codes):
            issues.append(f"ratings_long_dimension_set_mismatch:{doc_id}")

    per_doc_success = sorted((out_dir / "per_document").glob("*_parsed.json"))
    if document_scores and len(per_doc_success) < len(document_scores):
        issues.append("missing_per_document_records")

    summary = {
        "is_valid": not issues,
        "output_name": output_name,
        "ratings_prefix": ratings_prefix,
        "document_score_count": len(document_scores),
        "ratings_long_count": len(ratings_long),
        "failed_case_count": len(failed_cases),
        "issues": issues,
    }
    save_json(out_dir / f"{ratings_prefix}_validation_summary.json", summary)
    return summary


def _expected_total(row: dict[str, str]) -> float:
    weights = weights_for("mda")
    total = 0.0
    for code, weight in weights.items():
        raw = float(row.get(code) or 0)
        total += (100 * (raw - 1) / 4) * weight
    return round(total, 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate dimension-wise MD&A aggregate outputs.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--output-name", default="mda_v2_dimensionwise")
    parser.add_argument("--ratings-prefix", default="mda_v2_dimensionwise")
    args = parser.parse_args()
    summary = validate_dimensionwise_outputs(args.root, args.output_name, args.ratings_prefix)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
