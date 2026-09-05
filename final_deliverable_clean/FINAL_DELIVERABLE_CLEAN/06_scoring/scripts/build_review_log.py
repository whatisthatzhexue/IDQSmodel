from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from run_scoring import _review_row
from scoring_utils import REVIEW_LOG_HEADERS, SCORING_ROOT, load_json, read_csv_rows, write_csv_rows


def build_review_log(doc_type: str, root: Path = SCORING_ROOT) -> dict[str, Any]:
    per_doc_dir = root / "06_ratings" / "per_document" / doc_type
    new_rows: list[dict[str, Any]] = []
    for path in sorted(per_doc_dir.glob("*_parsed.json")):
        record = load_json(path)
        if record.get("status") == "success":
            new_rows.extend(record.get("review_rows", []))
            continue
        result = record.get("result", {})
        new_rows.append(
            _review_row(
                record.get("document_id", ""),
                doc_type,
                "manual_required",
                result.get("error_type", "scoring_failed"),
                "",
                "",
                "",
                "",
                result.get("raw_output_path", ""),
            )
        )
    path = root / "07_review" / "review_log.csv"
    existing = [row for row in read_csv_rows(path) if row.get("doc_type") != doc_type]
    write_csv_rows(path, REVIEW_LOG_HEADERS, existing + new_rows)
    return {"doc_type": doc_type, "review_count": len(new_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build review_log.csv from per-document scoring records.")
    parser.add_argument("--doc-type", choices=["mda", "news"], default="mda")
    args = parser.parse_args()
    print(build_review_log(args.doc_type))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
