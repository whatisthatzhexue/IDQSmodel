from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from claim_utils import CLAIM_FIELDS, extract_claims_from_text, validate_claim_schema
from scoring_utils import SCORING_ROOT, ensure_directories, read_csv_rows, registry_path, resolve_text_path, save_json, write_csv_rows


def extract_mda_claims(root: Path = SCORING_ROOT, limit: int | None = None) -> dict[str, Any]:
    ensure_directories(root)
    output_dir = root / "10_claims" / "mda_claims"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_csv_rows(registry_path("mda", root)) if row.get("include_flag", "").strip().lower() in {"yes", "y", "true", "1"}]
    if limit is not None:
        rows = rows[:limit]
    all_claim_rows: list[dict[str, Any]] = []
    issue_count = 0
    for row in rows:
        document_id = row.get("document_id", "")
        text_path = root / "02_extracted_text" / "mda_clean_v2" / f"{document_id}.txt"
        if not text_path.exists():
            text_path = resolve_text_path(row, "mda", root)
        if not document_id or not text_path.exists():
            save_json(output_dir / f"{document_id or 'missing_document'}_claims_error.json", {"document_id": document_id, "claims": [], "error": "missing_text"})
            issue_count += 1
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")
        claims = extract_claims_from_text("mda", document_id, text, row)
        for claim in claims:
            issues = validate_claim_schema(claim)
            claim["schema_issues"] = issues
            issue_count += 1 if issues else 0
        payload = {
            "document_id": document_id,
            "source_type": "mda",
            "text_path": str(text_path),
            "claim_count": len(claims),
            "claims": claims,
        }
        save_json(output_dir / f"{document_id}.json", payload)
        all_claim_rows.extend(claims)
    write_csv_rows(root / "10_claims" / "mda_claims_long.csv", CLAIM_FIELDS, all_claim_rows)
    summary = {
        "source_type": "mda",
        "document_count": len(rows),
        "claim_count": len(all_claim_rows),
        "schema_issue_count": issue_count,
        "output_dir": str(output_dir),
    }
    save_json(root / "10_claim_extraction" / "mda_claim_extraction_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract claim-level MD&A statements from cleaned v2 MD&A text.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(extract_mda_claims(args.root, args.limit), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
