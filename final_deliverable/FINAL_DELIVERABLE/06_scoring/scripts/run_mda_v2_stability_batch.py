from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from run_single_dimension_scoring import run_single_dimension_scoring
from scoring_utils import REGISTRY_HEADERS, REVIEW_LOG_HEADERS, SCORING_ROOT, document_headers_for, read_csv_rows, write_csv_rows


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def should_recommend_full_rescore(summary: dict[str, Any]) -> bool:
    return float(summary.get("success_rate", 0)) >= 0.85 and int(summary.get("scope_violation_count", 0)) == 0


def summarize_stability_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    success = sum(1 for row in records if row.get("status") == "success")
    failed = total - success
    json_fail = sum(1 for row in records if _result(row).get("parse_status") == "parse_failed")
    schema_fail = sum(1 for row in records if _result(row).get("schema_validation_status") == "invalid")
    missing_evidence = sum(1 for row in records if "missing_evidence" in json.dumps(_result(row), ensure_ascii=False).lower())
    scope_violation = sum(1 for row in records if "scope_violation" in json.dumps(_result(row), ensure_ascii=False).lower())
    manual_required = sum(1 for row in records if row.get("status") != "success" or _result(row).get("needs_review"))
    durations = [_float(_result(row).get("duration_seconds", row.get("duration_seconds", 0))) for row in records]
    durations = [item for item in durations if item is not None]
    summary = {
        "total_batch_size": total,
        "success_count": success,
        "failure_count": failed,
        "success_rate": round(success / total, 3) if total else 0,
        "json_parse_failure_count": json_fail,
        "schema_failure_count": schema_fail,
        "missing_evidence_count": missing_evidence,
        "scope_violation_count": scope_violation,
        "manual_required_count": manual_required,
        "avg_duration_seconds": round(sum(durations) / len(durations), 3) if durations else 0,
    }
    summary["recommendation_for_full_rescore"] = (
        "recommend_full_rescore" if should_recommend_full_rescore(summary) else "continue_fixing_failures_do_not_full_rescore"
    )
    return summary


def run_mda_v2_stability_batch(
    root: Path = SCORING_ROOT,
    model: str = "qwen3:8b",
    llm_workers: int = 2,
    cpu_workers: int = 6,
    resume: bool = True,
    overwrite: bool = False,
    mock: bool = False,
    scoring_strategy: str = "single_dimension",
) -> dict[str, Any]:
    if scoring_strategy not in {"single_dimension", "dimensionwise"}:
        raise ValueError("MD&A single-call whole_document scoring is disabled; use single-dimension scoring.")
    out_dir = root / "06_ratings" / "mda_v2_stability_batch"
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = select_stability_batch(root)
    completed = _completed_ids(out_dir) if resume and not overwrite else set()
    run_ids = [doc_id for doc_id in selected_ids if doc_id not in completed]
    if run_ids:
        tmp_root = root / "_self_correction" / "mda_v2_stability_batch_tmp"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        _prepare_tmp_root(root, tmp_root, run_ids)
        run_single_dimension_scoring(
            "mda",
            root=tmp_root,
            doc_ids=run_ids,
            output_name="mda_v2_stability_batch",
            ratings_prefix="mda_v2_stability",
            model=model,
            llm_workers=llm_workers,
            resume=False,
            overwrite=True,
            mock=mock,
        )
        source_dir = tmp_root / "06_ratings" / "mda_v2_stability_batch"
        _merge_rows(out_dir / "mda_v2_stability_ratings_long.csv", source_dir / "mda_v2_stability_ratings_long.csv", None, "rating_id")
        _merge_rows(
            out_dir / "mda_v2_stability_document_scores.csv",
            source_dir / "mda_v2_stability_document_scores.csv",
            document_headers_for("mda"),
            "document_id",
        )
        _merge_rows(out_dir / "mda_v2_stability_review_log.csv", source_dir / "mda_v2_stability_review_log.csv", REVIEW_LOG_HEADERS, "__review__")
        _copy_terminal_outputs(
            root,
            tmp_root,
            out_dir,
            overwrite,
            "mda_v2_stability",
            source_output_name="mda_v2_stability_batch",
            raw_output_name="mda_v2_stability_batch",
        )
    write_csv_rows(out_dir / "mda_v2_stability_batch_selection.csv", ["document_id"], [{"document_id": doc_id} for doc_id in selected_ids])
    records = [_load_json(path) for path in sorted((out_dir / "per_document").glob("*_parsed.json")) if path.stem.removesuffix("_parsed") in selected_ids]
    summary = summarize_stability_results(records)
    summary["selected_ids"] = selected_ids
    summary["scoring_strategy"] = "single_dimension"
    _write_report(root / "08_reports" / "mda_v2_stability_batch_report.md", summary)
    return summary


def select_stability_batch(root: Path = SCORING_ROOT, seed: int = 42) -> list[str]:
    registry_ids = [row.get("document_id", "") for row in read_csv_rows(root / "01_registry" / "mda_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    rng = random.Random(seed)
    random_ids = rng.sample(registry_ids, min(10, len(registry_ids))) if registry_ids else []
    failed_ids = [
        path.name.removesuffix("_parsed.json")
        for path in sorted((root / "06_ratings" / "mda_v2_sample_rescore" / "per_document").glob("*_parsed.json"))
        if _load_json(path).get("status") != "success"
    ][:10]
    numeric_ids = [
        row.get("document_id", "")
        for row in read_csv_rows(root / "03_numeric_checks" / "mda_v2_numeric_checks_summary.csv")
        if _int(row.get("num_material_discrepancies")) > 0
    ][:10]
    text_quality_ids = [
        row.get("document_id", "")
        for row in read_csv_rows(root / "08_reports" / "mda_text_quality_audit.csv")
        if row.get("needs_review", "").lower() in {"1", "true", "yes"}
    ][:5]
    selected: list[str] = []
    for doc_id in random_ids + failed_ids + numeric_ids + text_quality_ids:
        if doc_id and doc_id not in selected:
            selected.append(doc_id)
    return selected


def _prepare_tmp_root(root: Path, tmp_root: Path, doc_ids: list[str]) -> None:
    for rel in ["04_prompts/scoring_output_schema.json", "04_prompts/mda_scoring_prompt.txt"]:
        target = tmp_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
    registry = {row.get("document_id", ""): row for row in read_csv_rows(root / "01_registry" / "mda_registry.csv")}
    rows = []
    for doc_id in doc_ids:
        if doc_id not in registry:
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
        row = dict(registry[doc_id])
        row["text_path"] = f"02_extracted_text/mda/{doc_id}.txt"
        rows.append(row)
    write_csv_rows(tmp_root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], rows)


def _completed_ids(out_dir: Path) -> set[str]:
    return {path.name.removesuffix("_parsed.json") for path in (out_dir / "per_document").glob("*_parsed.json")}


def _merge_rows(dst: Path, src: Path, headers: list[str] | None, key_field: str) -> None:
    source_rows = read_csv_rows(src)
    if headers is None:
        headers = list(source_rows[0].keys()) if source_rows else []
    existing = {_row_key(row, key_field): row for row in read_csv_rows(dst) if _row_key(row, key_field)}
    for row in source_rows:
        key = _row_key(row, key_field)
        if key:
            existing[key] = row
    write_csv_rows(dst, headers, list(existing.values()))


def _copy_terminal_outputs(
    root: Path,
    tmp_root: Path,
    out_dir: Path,
    overwrite: bool,
    prefix: str,
    source_output_name: str | None = None,
    raw_output_name: str | None = None,
) -> None:
    raw_out = root / "05_raw_model_outputs" / (raw_output_name or prefix)
    if raw_out.exists() and overwrite:
        shutil.rmtree(raw_out)
    raw_src = tmp_root / "05_raw_model_outputs" / (source_output_name or "mda")
    if raw_src.exists():
        raw_out.mkdir(parents=True, exist_ok=True)
        for path in raw_src.glob("*"):
            if path.is_file():
                shutil.copy2(path, raw_out / path.name)
    parsed_src = tmp_root / "06_ratings" / source_output_name / "per_document" if source_output_name else tmp_root / "06_ratings" / "per_document" / "mda"
    parsed_out = out_dir / "per_document"
    if parsed_out.exists() and overwrite:
        shutil.rmtree(parsed_out)
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
    progress = (
        tmp_root / "06_ratings" / source_output_name / f"{prefix}_single_dimension_progress_log.jsonl"
        if source_output_name
        else tmp_root / "08_reports" / "scoring_progress_log.jsonl"
    )
    if source_output_name and not progress.exists():
        progress = tmp_root / "06_ratings" / source_output_name / f"{prefix}_dimensionwise_progress_log.jsonl"
    if progress.exists():
        shutil.copy2(progress, out_dir / f"{prefix}_progress_log.jsonl")


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# MD&A v2 Stability Batch Report",
        "",
        *[f"- {key}: {value}" for key, value in summary.items() if key != "selected_ids"],
        "",
        "## Selected Documents",
        *[f"- {doc_id}" for doc_id in summary.get("selected_ids", [])],
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _result(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result", {})
    return result if isinstance(result, dict) else {}


def _row_key(row: dict[str, str], key_field: str) -> str:
    if key_field != "__review__":
        return row.get(key_field, "")
    return "|".join(row.get(field, "") for field in ["document_id", "review_type", "review_reason", "dimension_code", "raw_output_path"])


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "failed", "result": {"parse_status": "parse_failed"}}


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a risk-weighted MD&A v2 stability scoring batch.")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--llm-workers", type=int, default=2)
    parser.add_argument("--cpu-workers", type=int, default=6)
    parser.add_argument("--resume", type=parse_bool, default=True)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--scoring-strategy", choices=["single_dimension", "dimensionwise", "whole_document"], default="single_dimension")
    args = parser.parse_args()
    print(
        run_mda_v2_stability_batch(
            model=args.model,
            llm_workers=args.llm_workers,
            cpu_workers=args.cpu_workers,
            resume=args.resume,
            overwrite=args.overwrite,
            mock=args.mock,
            scoring_strategy=args.scoring_strategy,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
