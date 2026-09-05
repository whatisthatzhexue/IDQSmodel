from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from claim_utils import CLAIM_DOCUMENT_HEADERS, aggregate_document_claim_scores
from scoring_utils import SCORING_ROOT, ensure_directories, read_csv_rows, save_json, write_csv_rows


def aggregate_claim_scores(root: Path = SCORING_ROOT, source_type: str = "mda") -> dict[str, Any]:
    if source_type not in {"mda", "news"}:
        raise ValueError("source_type must be mda or news")
    ensure_directories(root)
    claims_dir = root / "10_claims" / f"{source_type}_claims"
    scores_dir = root / "10_claim_scoring" / f"{source_type}_claim_scores"
    output_dir = root / "10_claim_mapping"
    output_dir.mkdir(parents=True, exist_ok=True)

    claim_docs = _load_claim_documents(claims_dir)
    score_docs = _load_score_documents(scores_dir)
    document_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    for document_id, claims in claim_docs.items():
        scores = score_docs.get(document_id, [])
        if not claims:
            continue
        model_name = _first_nonempty(scores, "model_name", "deterministic_claim_proxy")
        llm_backend = _first_nonempty(scores, "llm_backend", "rules")
        document_row = aggregate_document_claim_scores(source_type, document_id, claims, scores, model_name, llm_backend)
        document_rows.append(document_row)
        for dimension_code in _dimension_codes(source_type):
            raw = document_row.get(dimension_code, "")
            dimension_rows.append(
                {
                    "document_id": document_id,
                    "source_type": source_type,
                    "dimension_code": dimension_code,
                    "raw_score": raw,
                    "std_score_100": document_row.get(f"{dimension_code}_std", ""),
                    "claim_count": len(claims),
                    "scored_claim_count": len(scores),
                    "aggregation_method": document_row["aggregation_method"],
                }
            )
    write_csv_rows(output_dir / f"{source_type}_document_scores_claim_level.csv", CLAIM_DOCUMENT_HEADERS, document_rows)
    write_csv_rows(
        output_dir / f"{source_type}_claim_dimension_scores.csv",
        ["document_id", "source_type", "dimension_code", "raw_score", "std_score_100", "claim_count", "scored_claim_count", "aggregation_method"],
        dimension_rows,
    )
    summary = {
        "source_type": source_type,
        "document_count": len(document_rows),
        "claim_count": sum(int(row.get("claim_count") or 0) for row in document_rows),
        "scored_claim_count": sum(int(row.get("scored_claim_count") or 0) for row in document_rows),
        "output_document_scores": str(output_dir / f"{source_type}_document_scores_claim_level.csv"),
        "output_dimension_scores": str(output_dir / f"{source_type}_claim_dimension_scores.csv"),
    }
    save_json(output_dir / f"{source_type}_claim_aggregation_summary.json", summary)
    return summary


def _load_claim_documents(path: Path) -> dict[str, list[dict[str, Any]]]:
    docs: dict[str, list[dict[str, Any]]] = {}
    for file_path in sorted(path.glob("*.json")):
        if file_path.name.endswith("_claims_error.json"):
            continue
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        docs[payload.get("document_id", file_path.stem)] = payload.get("claims", [])
    return docs


def _load_score_documents(path: Path) -> dict[str, list[dict[str, Any]]]:
    docs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for file_path in sorted(path.glob("*.json")):
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        docs[payload.get("document_id", file_path.stem)].extend(
            [row for row in payload.get("claim_scores", []) if str(row.get("score", "")).strip()]
        )
    return dict(docs)


def _first_nonempty(rows: list[dict[str, Any]], field: str, default: str) -> str:
    for row in rows:
        if row.get(field):
            return str(row[field])
    return default


def _dimension_codes(source_type: str) -> list[str]:
    prefix = "AR" if source_type == "mda" else "N"
    return [f"{prefix}{idx:02d}" for idx in range(1, 6)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate claim scores into document-level claim scores.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--source-type", choices=["mda", "news"], default="mda")
    args = parser.parse_args()
    print(json.dumps(aggregate_claim_scores(args.root, args.source_type), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
