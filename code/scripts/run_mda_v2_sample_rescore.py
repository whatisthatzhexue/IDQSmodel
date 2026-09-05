from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import RATINGS_LONG_HEADERS, REGISTRY_HEADERS, REVIEW_LOG_HEADERS, SCORING_ROOT, document_headers_for, read_csv_rows, write_csv_rows


def run_mda_v2_sample_rescore(
    root: Path = SCORING_ROOT,
    model: str = "qwen3:8b",
    mock: bool = False,
    limit: int | None = None,
    resume: bool = True,
    overwrite: bool = False,
) -> dict:
    sample_rows = read_csv_rows(root / "06_ratings" / "mda_v2_sample_rescore" / "mda_v2_rescore_sample.csv")
    sample_ids = [row["document_id"] for row in sample_rows if row.get("selected") in {"1", "True", "true"}]
    if limit:
        sample_ids = sample_ids[:limit]
    out_dir = root / "06_ratings" / "mda_v2_sample_rescore"
    out_dir.mkdir(parents=True, exist_ok=True)
    completed_ids = _completed_sample_ids(out_dir) if resume and not overwrite else set()
    run_ids = [doc_id for doc_id in sample_ids if doc_id not in completed_ids]
    if not run_ids:
        return {
            "doc_type": "mda",
            "mode": "full",
            "selected_count": len(sample_ids),
            "task_count": 0,
            "skipped_count": len(sample_ids),
            "success_count": 0,
            "failure_count": 0,
            "message": "All selected v2 sample documents already have terminal parsed records.",
        }
    registry = {row["document_id"]: row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv")}
    tmp_root = root / "_self_correction" / "mda_v2_rescore_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    for rel in ["04_prompts/scoring_output_schema.json", "04_prompts/mda_scoring_prompt.txt"]:
        target = tmp_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / rel, target)
    rows = []
    for doc_id in run_ids:
        if doc_id not in registry:
            continue
        src_text = root / "02_extracted_text" / "mda_clean_v2" / f"{doc_id}.txt"
        dst_text = tmp_root / "02_extracted_text" / "mda" / f"{doc_id}.txt"
        dst_text.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_text, dst_text)
        src_num = root / "03_numeric_checks" / "mda_v2" / f"{doc_id}_numeric_checks.json"
        dst_num = tmp_root / "03_numeric_checks" / "mda" / f"{doc_id}_numeric_checks.json"
        dst_num.parent.mkdir(parents=True, exist_ok=True)
        if src_num.exists():
            shutil.copy(src_num, dst_num)
        row = dict(registry[doc_id])
        row["text_path"] = f"02_extracted_text/mda/{doc_id}.txt"
        rows.append(row)
    write_csv_rows(tmp_root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], rows)
    summary = run_single_dimension_scoring(
        "mda",
        root=tmp_root,
        doc_ids=run_ids,
        output_name="mda_v2_sample",
        ratings_prefix="mda_v2",
        model=model,
        llm_workers=1,
        resume=False,
        overwrite=True,
        mock=mock,
    )
    source_dir = tmp_root / "06_ratings" / "mda_v2_sample"
    _merge_rows(out_dir / "mda_v2_ratings_long_sample.csv", source_dir / "mda_v2_ratings_long.csv", RATINGS_LONG_HEADERS, "rating_id")
    _merge_rows(out_dir / "mda_v2_document_scores_sample.csv", source_dir / "mda_v2_document_scores.csv", document_headers_for("mda"), "document_id")
    _merge_rows(out_dir / "mda_v2_sample_review_log.csv", source_dir / "mda_v2_review_log.csv", REVIEW_LOG_HEADERS, "__review__")

    raw_out = root / "05_raw_model_outputs" / "mda_v2_sample"
    if raw_out.exists() and overwrite:
        shutil.rmtree(raw_out)
    raw_src = tmp_root / "05_raw_model_outputs" / "mda_v2_sample"
    if raw_src.exists():
        raw_out.mkdir(parents=True, exist_ok=True)
        for path in raw_src.glob("*"):
            if path.is_file():
                shutil.copy2(path, raw_out / path.name)
    parsed_src = source_dir / "per_document"
    parsed_out = out_dir / "per_document"
    if parsed_out.exists() and overwrite:
        shutil.rmtree(parsed_out)
    if parsed_src.exists():
        parsed_out.mkdir(parents=True, exist_ok=True)
        for path in parsed_src.glob("*_parsed.json"):
            shutil.copy2(path, parsed_out / path.name)
    progress_src = source_dir / "mda_v2_single_dimension_progress_log.jsonl"
    if not progress_src.exists():
        progress_src = source_dir / "mda_v2_dimensionwise_progress_log.jsonl"
    if progress_src.exists():
        progress_dst = out_dir / "mda_v2_sample_scoring_progress_log.jsonl"
        if progress_dst.exists() and resume and not overwrite:
            with progress_dst.open("a", encoding="utf-8") as out_fh:
                out_fh.write(progress_src.read_text(encoding="utf-8"))
        else:
            shutil.copy2(progress_src, progress_dst)
    dim_src = source_dir / "per_dimension"
    dim_out = out_dir / "per_dimension"
    if dim_out.exists() and overwrite:
        shutil.rmtree(dim_out)
    if dim_src.exists():
        dim_out.mkdir(parents=True, exist_ok=True)
        for path in dim_src.glob("*_parsed.json"):
            shutil.copy2(path, dim_out / path.name)
    summary_src = source_dir / "mda_v2_single_dimension_summary.json"
    if not summary_src.exists():
        summary_src = source_dir / "mda_v2_dimensionwise_summary.json"
    if summary_src.exists():
        shutil.copy2(summary_src, out_dir / "mda_v2_sample_single_dimension_summary.json")
    summary["selected_count"] = len(sample_ids)
    summary["skipped_count"] = len(completed_ids & set(sample_ids))
    return summary


def _completed_sample_ids(out_dir: Path) -> set[str]:
    completed = set()
    parsed_dir = out_dir / "per_document"
    for path in parsed_dir.glob("*_parsed.json"):
        completed.add(path.name.removesuffix("_parsed.json"))
    for row in read_csv_rows(out_dir / "mda_v2_document_scores_sample.csv"):
        if row.get("document_id"):
            completed.add(row["document_id"])
    return completed


def _merge_rows(dst: Path, src: Path, headers: list[str], key_field: str) -> None:
    existing = {_row_key(row, key_field): row for row in read_csv_rows(dst) if _row_key(row, key_field)}
    for row in read_csv_rows(src):
        key = _row_key(row, key_field)
        if key:
            existing[key] = row
    write_csv_rows(dst, headers, list(existing.values()))


def _row_key(row: dict[str, str], key_field: str) -> str:
    if key_field != "__review__":
        return row.get(key_field, "")
    return "|".join(row.get(field, "") for field in ["document_id", "review_type", "review_reason", "dimension_code", "raw_output_path"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(run_mda_v2_sample_rescore(model=args.model, mock=args.mock, limit=args.limit, resume=args.resume, overwrite=args.overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
