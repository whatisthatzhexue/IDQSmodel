from __future__ import annotations

import argparse
from pathlib import Path

from compare_mda_v1_v2_scores import full_rescore_decision
from scoring_utils import SCORING_ROOT, load_json, read_csv_rows


def build_mda_rescore_decision_report(root: Path = SCORING_ROOT) -> Path:
    archive_meta = root / "archive" / "v1_initial_mda_scoring" / "v1_metadata.json"
    frozen = archive_meta.exists()
    meta = load_json(archive_meta) if frozen else {}
    quality = read_csv_rows(root / "08_reports" / "mda_text_quality_audit.csv")
    audit = read_csv_rows(root / "06_ratings" / "mda_v1_audit" / "mda_initial_score_audit.csv")
    numeric = read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv")
    comparison = read_csv_rows(root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v1_v2_score_comparison.csv")
    sample = read_csv_rows(root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_rescore_sample.csv")
    sample_scores = read_csv_rows(root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_document_scores_sample.csv")
    sample_review = read_csv_rows(root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_sample_review_log.csv")
    decision = full_rescore_decision(comparison)
    quality_issues = sum(int(row.get("needs_review", 0) or 0) for row in quality)
    audit_issues = len({row["document_id"] for row in audit if row.get("needs_review") == "1"})
    material_numeric = sum(int(float(row.get("num_material_discrepancies", 0) or 0)) for row in numeric)
    severe_numeric = sum(int(float(row.get("num_severe_direction_conflicts", 0) or 0)) for row in numeric)
    parsed_counts, failed_ids = _sample_terminal_status(root)
    selected_sample_count = sum(1 for row in sample if row.get("selected") in {"1", "True", "true"})
    overlap_note = ""
    if len(comparison) < min(int(meta.get("num_documents", 0) or 0), len(sample_scores)):
        overlap_note = (
            "Only documents present in both frozen v1 document scores and successful v2 sample scores can be compared. "
            "The current frozen v1 document_scores file contains limited rows."
        )
    path = root / "08_reports" / "mda_rescore_decision_report.md"
    next_command = "python 06_scoring/scripts/run_mda_v2_full_rescore.py --model qwen3:8b --llm-workers 1 --resume"
    failed_preview = ", ".join(failed_ids[:20]) if failed_ids else "none"
    review_preview = ", ".join(sorted({row.get("document_id", "") for row in sample_review if row.get("document_id")})[:20]) or "none"
    body = f"""# MDA Rescore Decision Report

1. v1 initial scoring frozen: {frozen}
2. frozen documents: {meta.get('num_documents', 0)}
3. cleaned MD&A documents: {len(quality)}
4. documents with text quality issues: {quality_issues}
5. documents with initial-score evidence/scope issues: {audit_issues}
6. AR02 numeric material discrepancies: {material_numeric}
7. AR02 severe direction conflicts: {severe_numeric}
8. v1/v2 compared sample documents: {len(comparison)}
9. full rescore recommended: {decision['full_rescore_recommended']}
10. decision reason: {decision['reason']}

## Sample Rescore Status

- selected v2 sample documents: {selected_sample_count}
- terminal parsed sample records: {sum(parsed_counts.values())}
- successful v2 sample score documents: {len(sample_scores)}
- failed/manual-required v2 sample documents: {parsed_counts.get('failed', 0)}
- sample review rows: {len(sample_review)}
- failed sample preview: {failed_preview}

## Interpretation

{overlap_note or 'The v1/v2 comparison has enough overlap for the current frozen v1 inputs.'}

If full rescore is recommended, run cleaned-text v2 full scoring into a separate v2_full output, never over v1.

Suggested command if needed:

```bash
{next_command}
```

Before any full rescore, manually inspect failed sample outputs and prompt/schema stability. Sample records needing attention include: {review_preview}.

If full rescore is not recommended, keep v1 initial scoring and manually review flagged cases in `06_scoring/07_review/review_log.csv` and `06_scoring/06_ratings/mda_v2_sample_rescore/mda_v2_sample_review_log.csv`.
"""
    path.write_text(body, encoding="utf-8")
    return path


def _sample_terminal_status(root: Path) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    failed_ids: list[str] = []
    parsed_dir = root / "06_ratings" / "mda_v2_sample_rescore" / "per_document"
    for path in sorted(parsed_dir.glob("*_parsed.json")):
        try:
            payload = load_json(path)
        except Exception:  # noqa: BLE001 - report should be robust to one bad trace file.
            status = "unreadable"
        else:
            status = payload.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status != "success":
            failed_ids.append(path.name.removesuffix("_parsed.json"))
    return counts, failed_ids


def main() -> int:
    print(build_mda_rescore_decision_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
