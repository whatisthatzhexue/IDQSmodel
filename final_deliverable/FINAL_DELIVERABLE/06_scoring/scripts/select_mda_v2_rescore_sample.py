from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


HEADERS = ["document_id", "selected", "selection_reasons"]


def select_mda_v2_rescore_sample(root: Path = SCORING_ROOT, seed: int = 42) -> dict[str, Any]:
    quality = {row["document_id"]: row for row in read_csv_rows(root / "08_reports" / "mda_text_quality_audit.csv")}
    audit_rows = read_csv_rows(root / "06_ratings" / "mda_v1_audit" / "mda_initial_score_audit.csv")
    numeric = {row["document_id"]: row for row in read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv")}
    scores = {row["document_id"]: row for row in read_csv_rows(root / "archive" / "v1_initial_mda_scoring" / "mda_document_scores.csv")}
    ratings = read_csv_rows(root / "archive" / "v1_initial_mda_scoring" / "mda_ratings_long.csv")
    doc_ids = sorted(set(quality) | set(scores) | {row.get("document_id", "") for row in audit_rows})
    selected: dict[str, set[str]] = {doc: set() for doc in doc_ids if doc}
    for doc in selected:
        q = quality.get(doc, {})
        n = numeric.get(doc, {})
        s = scores.get(doc, {})
        if q.get("needs_review") == "1":
            selected[doc].add("text_quality_needs_review")
        if int(float(q.get("clean_word_count", 999999) or 999999)) < 300:
            selected[doc].add("clean_word_count_lt_300")
        if float(q.get("cleaning_loss_ratio", 0) or 0) > 0.40:
            selected[doc].add("cleaning_loss_ratio_gt_0_40")
        if int(float(n.get("num_material_discrepancies", 0) or 0)) > 0:
            selected[doc].add("numeric_material_discrepancy")
        if int(float(n.get("num_severe_direction_conflicts", 0) or 0)) > 0:
            selected[doc].add("numeric_severe_direction_conflict")
        if float(s.get("total_score_100", 100) or 100) < 55:
            selected[doc].add("total_score_lt_55")
    for row in audit_rows:
        if row.get("needs_review") == "1":
            selected.setdefault(row["document_id"], set()).add("initial_score_audit_needs_review")
        if row.get("evidence_valid_in_clean_v2") == "0":
            selected.setdefault(row["document_id"], set()).add("evidence_locator_invalid")
    for row in ratings:
        doc = row.get("document_id", "")
        if row.get("raw_score") in {"1", "5"}:
            selected.setdefault(doc, set()).add("extreme_dimension_score")
        if row.get("confidence_level", "").lower() == "low":
            selected.setdefault(doc, set()).add("low_confidence")
    rng = random.Random(seed)
    random_docs = rng.sample(doc_ids, min(len(doc_ids), max(5, int(len(doc_ids) * 0.10)))) if doc_ids else []
    for doc in random_docs:
        selected.setdefault(doc, set()).add("random_10pct")
    rows = [
        {"document_id": doc, "selected": int(bool(reasons)), "selection_reasons": ";".join(sorted(reasons))}
        for doc, reasons in sorted(selected.items())
        if reasons
    ]
    out = root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_rescore_sample.csv"
    write_csv_rows(out, HEADERS, rows)
    return {"selected_documents": len(rows)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(select_mda_v2_rescore_sample(seed=args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
