from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import RATINGS_LONG_HEADERS, REGISTRY_HEADERS, REVIEW_LOG_HEADERS, SCORING_ROOT, document_headers_for, read_csv_rows, write_csv_rows


def run_mda_v2_full_rescore(
    root: Path = SCORING_ROOT,
    model: str = "qwen3:8b",
    llm_workers: int = 1,
    cpu_workers: int | None = None,
    resume: bool = True,
    overwrite: bool = False,
    limit: int | None = None,
    mock: bool = False,
    scoring_strategy: str = "single_dimension",
) -> dict[str, Any]:
    if scoring_strategy not in {"single_dimension", "dimensionwise"}:
        raise ValueError("MD&A single-call whole_document scoring is disabled; use single-dimension scoring.")
    out_dir = root / "06_ratings" / "mda_v2_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_rows = [row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    if limit:
        registry_rows = registry_rows[:limit]
    completed = _completed_ids(out_dir) if resume and not overwrite else set()
    rows_to_run = [row for row in registry_rows if row.get("document_id") not in completed]
    if not rows_to_run:
        return {
            "doc_type": "mda",
            "mode": "full",
            "selected_count": len(registry_rows),
            "task_count": 0,
            "skipped_count": len(registry_rows),
            "success_count": 0,
            "failure_count": 0,
            "scoring_strategy": "single_dimension",
            "message": "All selected v2_full documents already have terminal parsed records.",
        }

    tmp_root = root / "_self_correction" / "mda_v2_full_rescore_tmp"
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    _copy_static_files(root, tmp_root)
    prepared_rows = []
    for row in rows_to_run:
        doc_id = row.get("document_id", "")
        if not doc_id:
            continue
        src_text = root / "02_extracted_text" / "mda_clean_v2" / f"{doc_id}.txt"
        if not src_text.exists():
            continue
        dst_text = tmp_root / "02_extracted_text" / "mda" / f"{doc_id}.txt"
        dst_text.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_text, dst_text)
        src_num = root / "03_numeric_checks" / "mda_v2" / f"{doc_id}_numeric_checks.json"
        dst_num = tmp_root / "03_numeric_checks" / "mda" / f"{doc_id}_numeric_checks.json"
        dst_num.parent.mkdir(parents=True, exist_ok=True)
        if src_num.exists():
            shutil.copy2(src_num, dst_num)
        prepared = dict(row)
        prepared["text_path"] = f"02_extracted_text/mda/{doc_id}.txt"
        prepared_rows.append(prepared)
    write_csv_rows(tmp_root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], prepared_rows)
    summary = run_single_dimension_scoring(
        "mda",
        root=tmp_root,
        doc_ids=[row.get("document_id", "") for row in prepared_rows],
        output_name="mda_v2_full",
        ratings_prefix="mda_v2_full",
        model=model,
        llm_workers=llm_workers,
        resume=False,
        overwrite=True,
        mock=mock,
    )
    source_dir = tmp_root / "06_ratings" / "mda_v2_full"
    _merge_rows(out_dir / "mda_v2_full_ratings_long.csv", source_dir / "mda_v2_full_ratings_long.csv", RATINGS_LONG_HEADERS, "rating_id")
    _merge_rows(out_dir / "mda_v2_full_document_scores.csv", source_dir / "mda_v2_full_document_scores.csv", document_headers_for("mda"), "document_id")
    _merge_rows(out_dir / "mda_v2_full_review_log.csv", source_dir / "mda_v2_full_review_log.csv", REVIEW_LOG_HEADERS, "__review__")
    _copy_outputs(root, tmp_root, out_dir, overwrite=overwrite, source_output_name="mda_v2_full")
    _write_failed_cases(out_dir)
    _write_final_candidate(root, out_dir, model if not mock else "mock_qwen_unavailable")
    summary["selected_count"] = len(registry_rows)
    summary["skipped_count"] = len(completed & {row.get("document_id", "") for row in registry_rows})
    summary["scoring_strategy"] = "single_dimension"
    _write_full_report(root, summary, out_dir)
    return summary


def _copy_static_files(root: Path, tmp_root: Path) -> None:
    for rel in ["04_prompts/scoring_output_schema.json", "04_prompts/mda_scoring_prompt.txt"]:
        target = tmp_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)


def _completed_ids(out_dir: Path) -> set[str]:
    parsed_dir = out_dir / "per_document"
    ids = {path.name.removesuffix("_parsed.json") for path in parsed_dir.glob("*_parsed.json")}
    ids.update(row["document_id"] for row in read_csv_rows(out_dir / "mda_v2_full_document_scores.csv") if row.get("document_id"))
    return ids


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


def _copy_outputs(root: Path, tmp_root: Path, out_dir: Path, overwrite: bool, source_output_name: str | None = None) -> None:
    raw_out = root / "05_raw_model_outputs" / "mda_v2_full"
    if raw_out.exists() and overwrite:
        shutil.rmtree(raw_out)
    raw_src = tmp_root / "05_raw_model_outputs" / (source_output_name or "mda")
    if raw_src.exists():
        raw_out.mkdir(parents=True, exist_ok=True)
        for path in raw_src.glob("*"):
            if path.is_file():
                shutil.copy2(path, raw_out / path.name)
    parsed_out = out_dir / "per_document"
    if parsed_out.exists() and overwrite:
        shutil.rmtree(parsed_out)
    parsed_src = tmp_root / "06_ratings" / source_output_name / "per_document" if source_output_name else tmp_root / "06_ratings" / "per_document" / "mda"
    if parsed_src.exists():
        parsed_out.mkdir(parents=True, exist_ok=True)
        for path in parsed_src.glob("*_parsed.json"):
            shutil.copy2(path, parsed_out / path.name)
    dim_src = tmp_root / "06_ratings" / source_output_name / "per_dimension" if source_output_name else Path("")
    if source_output_name and dim_src.exists():
        dim_out = out_dir / "per_dimension"
        if dim_out.exists() and overwrite:
            shutil.rmtree(dim_out)
        dim_out.mkdir(parents=True, exist_ok=True)
        for path in dim_src.glob("*_parsed.json"):
            shutil.copy2(path, dim_out / path.name)
    progress_src = (
        tmp_root / "06_ratings" / source_output_name / "mda_v2_full_single_dimension_progress_log.jsonl"
        if source_output_name
        else tmp_root / "08_reports" / "scoring_progress_log.jsonl"
    )
    if source_output_name and not progress_src.exists():
        progress_src = tmp_root / "06_ratings" / source_output_name / "mda_v2_full_dimensionwise_progress_log.jsonl"
    if progress_src.exists():
        progress_dst = out_dir / "mda_v2_full_scoring_progress_log.jsonl"
        if progress_dst.exists() and not overwrite:
            with progress_dst.open("a", encoding="utf-8") as out_fh:
                out_fh.write(progress_src.read_text(encoding="utf-8"))
        else:
            shutil.copy2(progress_src, progress_dst)


def _write_failed_cases(out_dir: Path) -> None:
    rows = []
    for path in sorted((out_dir / "per_document").glob("*_parsed.json")):
        record = _load_json(path)
        if record.get("status") == "success":
            continue
        result = record.get("result", {}) if isinstance(record.get("result"), dict) else {}
        rows.append(
            {
                "document_id": record.get("document_id", path.name.removesuffix("_parsed.json")),
                "status": record.get("status", "failed"),
                "error_type": result.get("error_type", ""),
                "error_message": result.get("error_message", ""),
                "parse_status": result.get("parse_status", ""),
                "schema_validation_status": result.get("schema_validation_status", ""),
                "attempts_used": result.get("attempts_used", ""),
                "raw_output_path": result.get("raw_output_path", ""),
            }
        )
    write_csv_rows(
        out_dir / "mda_v2_full_failed_cases.csv",
        [
            "document_id",
            "status",
            "error_type",
            "error_message",
            "parse_status",
            "schema_validation_status",
            "attempts_used",
            "raw_output_path",
        ],
        rows,
    )


def _write_final_candidate(root: Path, out_dir: Path, model_name: str) -> None:
    source = out_dir / "mda_v2_full_document_scores.csv"
    rows = []
    for row in read_csv_rows(source):
        copied = dict(row)
        copied["score_version"] = "v2_full"
        copied["text_version"] = "mda_clean_v2"
        copied["model_name"] = model_name
        copied["llm_backend"] = "ollama" if model_name != "mock_qwen_unavailable" else "mock"
        rows.append(copied)
    headers = document_headers_for("mda") + ["score_version", "text_version"]
    write_csv_rows(root / "06_ratings" / "mda_document_scores_final_candidate.csv", headers, rows)


def _write_full_report(root: Path, summary: dict[str, Any], out_dir: Path) -> Path:
    failed_rows = read_csv_rows(out_dir / "mda_v2_full_failed_cases.csv")
    score_rows = read_csv_rows(out_dir / "mda_v2_full_document_scores.csv")
    path = root / "08_reports" / "mda_v2_full_scoring_report.md"
    lines = [
        "# MD&A v2 Full Scoring Report",
        "",
        f"- selected_count: {summary.get('selected_count', 0)}",
        f"- success_count: {len(score_rows)}",
        f"- failure_count: {len(failed_rows)}",
        f"- skipped_count: {summary.get('skipped_count', 0)}",
        f"- output_dir: {out_dir}",
        "- original mda_document_scores.csv overwritten: false",
        "- final candidate path: 06_scoring/06_ratings/mda_document_scores_final_candidate.csv",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "result": {"error_type": "json_decode_error"}}


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MD&A cleaned-text v2 full rescore into separate v2_full outputs.")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--cpu-workers", type=int)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--scoring-strategy", choices=["single_dimension", "dimensionwise", "whole_document"], default="single_dimension")
    args = parser.parse_args()
    print(
        run_mda_v2_full_rescore(
            model=args.model,
            llm_workers=args.llm_workers,
            cpu_workers=args.cpu_workers,
            resume=args.resume,
            overwrite=args.overwrite,
            limit=args.limit,
            mock=args.mock,
            scoring_strategy=args.scoring_strategy,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
