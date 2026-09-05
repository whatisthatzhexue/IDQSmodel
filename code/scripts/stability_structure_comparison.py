from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, write_csv_rows
from stability_common import (
    discover_inputs,
    ensure_stability_dirs,
    fmt,
    pearson,
    safe_float,
    safe_int,
    spearman,
    stability_root,
    top_bottom_overlap,
    write_empty_csv,
    write_markdown,
)


STRUCTURE_HEADERS = [
    "document_id",
    "ticker",
    "company_name",
    "score_document_level",
    "score_single_dimension",
    "total_score_diff",
    "abs_total_score_diff",
    "max_dimension_diff",
    "rank_document_level",
    "rank_single_dimension",
    "rank_change",
    "structure_stability_flag",
    "review_required",
    "review_reasons",
]


def run_structure_comparison(root: Path = SCORING_ROOT, include_mda: bool = True, include_news: bool = True, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    manifest = manifest or discover_inputs(root)
    selected = manifest.get("selected_inputs", {})
    summary: dict[str, Any] = {}
    high_volatility: list[dict[str, Any]] = []
    for doc_type, dimensions, enabled in [
        ("mda", ["AR01", "AR02", "AR03", "AR04", "AR05"], include_mda),
        ("news", ["N01", "N02", "N03", "N04", "N05"], include_news),
    ]:
        if not enabled:
            summary[doc_type] = {"status": "skipped", "reason": f"include_{doc_type} false"}
            continue
        doc_path = Path(selected.get(f"{doc_type}_document_level") or "")
        single_path = Path(selected.get(f"{doc_type}_single_dimension") or "")
        item, cases = _analyze(root, doc_type, doc_path, single_path, dimensions)
        summary[doc_type] = item
        high_volatility.extend(cases)
    summary["high_volatility_cases"] = high_volatility
    _write_report(base / "reports" / "structure_stability_report.md", summary)
    save_json(base / "outputs" / "structure_comparison_summary.json", summary)
    return summary


def _analyze(root: Path, doc_type: str, doc_path: Path, single_path: Path, dimensions: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output_path = stability_root(root) / "outputs" / f"{doc_type}_structure_comparison.csv"
    if not str(doc_path) or not str(single_path) or not doc_path.is_file() or not single_path.is_file():
        write_empty_csv(output_path, STRUCTURE_HEADERS)
        return {
            "status": "missing-input",
            "doc_type": doc_type,
            "document_level_path": str(doc_path) if str(doc_path) != "." else "",
            "single_dimension_path": str(single_path) if str(single_path) != "." else "",
            "output_path": str(output_path),
            "document_count": 0,
            "reason": "document-level or single-dimension comparison input missing.",
            "pearson": 0.0,
            "spearman": 0.0,
        }, []
    doc_rows = {row.get("document_id", ""): row for row in read_csv_rows(doc_path)}
    single_rows = {row.get("document_id", ""): row for row in read_csv_rows(single_path)}
    joined = []
    for document_id in sorted(set(doc_rows) & set(single_rows)):
        left = doc_rows[document_id]
        right = single_rows[document_id]
        joined.append({"document_id": document_id, "left": left, "right": right})
    doc_rank = _rank_map(joined, "left")
    single_rank = _rank_map(joined, "right")
    rows = []
    cases = []
    for item in joined:
        document_id = item["document_id"]
        left = item["left"]
        right = item["right"]
        total_left = safe_float(left.get("total_score_100"))
        total_right = safe_float(right.get("total_score_100"))
        dim_diffs = [abs(safe_float(right.get(code)) - safe_float(left.get(code))) for code in dimensions]
        max_dim = max(dim_diffs) if dim_diffs else 0.0
        abs_total = abs(total_right - total_left)
        rank_change = safe_int(single_rank.get(document_id)) - safe_int(doc_rank.get(document_id))
        reasons = []
        if abs_total > 10:
            reasons.append("structure_abs_diff_gt_10")
        if max_dim >= 2:
            reasons.append("dimension_shift_review_required")
        row = {
            "document_id": document_id,
            "ticker": right.get("ticker") or left.get("ticker", ""),
            "company_name": right.get("company_name") or left.get("company_name", ""),
            "score_document_level": fmt(total_left),
            "score_single_dimension": fmt(total_right),
            "total_score_diff": fmt(total_right - total_left),
            "abs_total_score_diff": fmt(abs_total),
            "max_dimension_diff": fmt(max_dim),
            "rank_document_level": doc_rank.get(document_id, 0),
            "rank_single_dimension": single_rank.get(document_id, 0),
            "rank_change": rank_change,
            "structure_stability_flag": "stable" if abs_total <= 5 else "historical_reference_review" if abs_total > 10 else "moderate_difference",
            "review_required": "yes" if reasons else "no",
            "review_reasons": ";".join(reasons),
        }
        rows.append(row)
        if abs_total > 10 or max_dim >= 2:
            cases.append(
                {
                    "document_id": document_id,
                    "doc_type": doc_type,
                    "ticker": row["ticker"],
                    "company_name": row["company_name"],
                    "year_or_publish_date": right.get("report_year", right.get("publish_date", "")),
                    "issue_type": "structure_comparison_shift",
                    "severity": "medium",
                    "score_impact": fmt(abs_total),
                    "rank_impact": rank_change,
                    "dimension_affected": "",
                    "recommended_action": "manual_review",
                    "source_file": str(output_path),
                }
            )
    write_csv_rows(output_path, STRUCTURE_HEADERS, rows)
    doc_scores = [safe_float(row.get("score_document_level")) for row in rows]
    single_scores = [safe_float(row.get("score_single_dimension")) for row in rows]
    summary = {
        "status": "success" if rows else "empty-overlap",
        "doc_type": doc_type,
        "document_level_path": str(doc_path),
        "single_dimension_path": str(single_path),
        "output_path": str(output_path),
        "document_count": len(rows),
        "pearson": pearson(doc_scores, single_scores),
        "spearman": spearman(doc_scores, single_scores),
        "top_20_overlap": top_bottom_overlap(rows, "score_document_level", "score_single_dimension", top=True),
        "bottom_20_overlap": top_bottom_overlap(rows, "score_document_level", "score_single_dimension", top=False),
        "recommendation": "single-dimension remains the main system; document-level scoring is historical reference only.",
    }
    return summary, cases


def _rank_map(joined: list[dict[str, Any]], side: str) -> dict[str, int]:
    rows = [{"document_id": item["document_id"], "score": safe_float(item[side].get("total_score_100"))} for item in joined]
    ordered = sorted(rows, key=lambda row: (-safe_float(row.get("score")), row.get("document_id", "")))
    return {row["document_id"]: idx for idx, row in enumerate(ordered, start=1)}


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Structure Stability Report", "", "Compares historical document-level scores against single-dimension scores when optional inputs exist.", ""]
    for doc_type in ["mda", "news"]:
        item = summary.get(doc_type, {})
        lines.extend(
            [
                f"## {doc_type.upper()}",
                f"- status: {item.get('status', 'skipped')}",
                f"- document_count: {item.get('document_count', 0)}",
                f"- Pearson: {item.get('pearson', 0)}",
                f"- Spearman: {item.get('spearman', 0)}",
                f"- recommendation: {item.get('recommendation', '')}",
                f"- reason: {item.get('reason', '')}",
                "",
            ]
        )
    write_markdown(path, lines)


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run document-level vs single-dimension structure stability comparison.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--include-mda", type=parse_bool, default=True)
    parser.add_argument("--include-news", type=parse_bool, default=True)
    args = parser.parse_args()
    print(json.dumps(run_structure_comparison(args.root, args.include_mda, args.include_news), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
