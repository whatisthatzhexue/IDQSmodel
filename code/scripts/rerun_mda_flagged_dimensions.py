from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import (
    MDA_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    SCORING_ROOT,
    grade_label,
    read_csv_rows,
    standardize_score,
    weighted_score,
    weights_for,
    write_csv_rows,
)
from stability_common import safe_float, write_markdown


def rerun_mda_flagged_dimensions(root: Path = SCORING_ROOT, model: str = "qwen3:8b", llm_workers: int = 1, execute: bool = True) -> dict[str, Any]:
    audit_rows = read_csv_rows(root / "08_reports" / "mda_semantic_evidence_audit.csv")
    flagged = [
        row
        for row in audit_rows
        if row.get("evidence_semantic_status") == "invalid" or row.get("recommended_action") == "rerun_dimension_only"
    ]
    flagged_pairs = sorted({(row["document_id"], row["dimension_code"]) for row in flagged if row.get("document_id") and row.get("dimension_code")})
    out_name = "mda_semantic_repair"
    if execute and flagged_pairs:
        by_dim: dict[str, list[str]] = defaultdict(list)
        for doc_id, code in flagged_pairs:
            by_dim[code].append(doc_id)
        for code, doc_ids in sorted(by_dim.items()):
            run_single_dimension_scoring(
                "mda",
                root=root,
                doc_ids=sorted(set(doc_ids)),
                output_name=out_name,
                ratings_prefix="mda_semantic_repair",
                model=model,
                llm_workers=llm_workers,
                resume=True,
                overwrite=False,
                dimension_codes=[code],
            )

    repaired = _merge_repaired_outputs(root, flagged_pairs, out_name)
    summary = {
        "flagged_dimension_count": len(flagged_pairs),
        "repaired_dimension_count": repaired["repaired_dimension_count"],
        "unrepaired_dimension_count": len(flagged_pairs) - repaired["repaired_dimension_count"],
        "document_scores_repaired_path": str(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores_repaired.csv"),
        "ratings_long_repaired_path": str(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long_repaired.csv"),
        "execute": execute,
    }
    _write_report(root / "08_reports" / "mda_semantic_repair_report.md", summary)
    return summary


def _merge_repaired_outputs(root: Path, flagged_pairs: list[tuple[str, str]], output_name: str) -> dict[str, int]:
    original_rows = read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv")
    repaired_payloads = _load_repaired_payloads(root, flagged_pairs, output_name)
    repaired_count = 0
    merged_rows = []
    for row in original_rows:
        key = (row.get("document_id", ""), row.get("dimension_code", ""))
        payload = repaired_payloads.get(key)
        if payload:
            row = _row_from_payload(row, payload)
            repaired_count += 1
        merged_rows.append(row)
    write_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long_repaired.csv", RATINGS_LONG_HEADERS, merged_rows)
    document_rows = _document_scores(root, merged_rows)
    write_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores_repaired.csv", MDA_DOCUMENT_HEADERS, document_rows)
    return {"repaired_dimension_count": repaired_count}


def _load_repaired_payloads(root: Path, flagged_pairs: list[tuple[str, str]], output_name: str) -> dict[tuple[str, str], dict[str, Any]]:
    payloads = {}
    for doc_id, code in flagged_pairs:
        path = root / "06_ratings" / output_name / "per_dimension" / f"{doc_id}_{code}_parsed.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        payload = data.get("payload") or {}
        if payload.get("raw_score") in {"", None}:
            continue
        payloads[(doc_id, code)] = {**payload, "_raw_output_path": data.get("raw_output_path", ""), "_numeric_check_path": data.get("numeric_check_path", "")}
    return payloads


def _row_from_payload(original: dict[str, str], payload: dict[str, Any]) -> dict[str, str]:
    row = dict(original)
    score = int(float(payload.get("raw_score")))
    weight = safe_float(row.get("weight_value")) or weights_for("mda").get(row.get("dimension_code", ""), 0)
    row.update(
        {
            "raw_score": str(score),
            "std_score_100": str(standardize_score(score)),
            "weighted_score": str(weighted_score(score, weight)),
            "evidence_locator": str(payload.get("evidence_locator", "")),
            "evidence_text_short": str(payload.get("evidence_text_short", "")),
            "confidence_level": str(payload.get("confidence_level", "medium")),
            "comment_short": str(payload.get("comment_short") or payload.get("reason") or ""),
            "raw_output_path": str(payload.get("_raw_output_path", "")),
            "numeric_check_path": str(payload.get("_numeric_check_path", "")),
            "parse_status": "parsed",
            "schema_validation_status": "valid",
            "review_round": "semantic_repair",
        }
    )
    return row


def _document_scores(root: Path, rating_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    originals = {row.get("document_id", ""): row for row in read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv")}
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rating_rows:
        grouped[row.get("document_id", "")][row.get("dimension_code", "")] = row
    out = []
    weights = weights_for("mda")
    for doc_id, dim_rows in grouped.items():
        base = dict(originals.get(doc_id, {}))
        total = 0.0
        complete = True
        for code, weight in weights.items():
            dim_row = dim_rows.get(code)
            if not dim_row:
                complete = False
                continue
            score = safe_float(dim_row.get("raw_score"))
            base[code] = str(int(score)) if score.is_integer() else str(score)
            base[f"{code}_std"] = str(standardize_score(score))
            total += weighted_score(score, weight)
        base["dimension_count"] = str(len(dim_rows))
        base["total_score_100"] = str(round(total, 6)) if complete else ""
        base["grade_label"] = grade_label(total) if complete else ""
        base["final_version"] = "mda_semantic_repaired_candidate"
        out.append(base)
    return sorted(out, key=lambda row: row.get("document_id", ""))


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# MD&A Semantic Repair Report", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rerun only MD&A document-dimensions flagged by semantic evidence audit.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rerun_mda_flagged_dimensions(args.root, args.model, args.llm_workers, not args.no_execute), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
