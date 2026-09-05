from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path
from typing import Any

from parallel_scoring import run_parallel_pipeline
from scoring_utils import REGISTRY_HEADERS, SCORING_ROOT, read_csv_rows, registry_path, resolve_text_path, write_csv_rows


BENCHMARK_HEADERS = [
    "doc_type",
    "sample_size",
    "llm_workers",
    "success_count",
    "failure_count",
    "json_parse_failure_count",
    "schema_failure_count",
    "total_duration_seconds",
    "avg_duration_per_doc",
    "docs_per_minute",
    "avg_prompt_eval_count",
    "avg_eval_count",
    "avg_total_duration_from_ollama",
    "recommendation",
]


def run_benchmark(
    doc_type: str,
    sample_size: int,
    workers: list[int],
    root: Path = SCORING_ROOT,
    mock_mode: bool = False,
    timeout_seconds: int = 180,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bench_root = _prepare_benchmark_root(root, doc_type, sample_size)
    for worker_count in workers:
        started = time.perf_counter()
        summary = run_parallel_pipeline(
            doc_type,
            "pilot",
            root=bench_root,
            limit=sample_size,
            llm_workers=worker_count,
            overwrite=True,
            resume=False,
            mock_mode=mock_mode,
            timeout_seconds=timeout_seconds,
            aggregate=True,
        )
        duration = round(time.perf_counter() - started, 3)
        avg = round(duration / max(1, sample_size), 3)
        rows.append(
            {
                "doc_type": doc_type,
                "sample_size": sample_size,
                "llm_workers": worker_count,
                "success_count": summary.get("success_count", 0),
                "failure_count": summary.get("failure_count", 0),
                "json_parse_failure_count": summary.get("json_parse_failures", 0),
                "schema_failure_count": summary.get("schema_validation_failures", 0),
                "total_duration_seconds": duration,
                "avg_duration_per_doc": avg,
                "docs_per_minute": round((summary.get("success_count", 0) / duration) * 60, 3) if duration else 0,
                "avg_prompt_eval_count": 0,
                "avg_eval_count": 0,
                "avg_total_duration_from_ollama": 0,
                "recommendation": _recommendation(rows, worker_count, summary, duration),
            }
        )
    _write_outputs(root, rows)
    return rows


def _prepare_benchmark_root(root: Path, doc_type: str, sample_size: int) -> Path:
    bench_root = root / "_self_correction" / f"benchmark_tmp_{doc_type}"
    if bench_root.exists():
        shutil.rmtree(bench_root)
    for rel in [
        "04_prompts/scoring_output_schema.json",
        f"04_prompts/{doc_type}_scoring_prompt.txt",
        "04_prompts/mda_scoring_prompt.txt",
        "04_prompts/news_scoring_prompt.txt",
    ]:
        source = root / rel
        if source.exists():
            target = bench_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, target)
    source_rows = [
        row
        for row in read_csv_rows(registry_path(doc_type, root))
        if row.get("include_flag", "").strip().lower() == "yes"
    ][:sample_size]
    copied_rows = []
    for row in source_rows:
        copied = dict(row)
        source_text = resolve_text_path(row, doc_type, root)
        target_text = bench_root / "02_extracted_text" / doc_type / f"{row['document_id']}.txt"
        target_text.parent.mkdir(parents=True, exist_ok=True)
        if source_text.exists():
            shutil.copy(source_text, target_text)
        copied["text_path"] = f"02_extracted_text/{doc_type}/{row['document_id']}.txt"
        copied_rows.append(copied)
    if copied_rows:
        write_csv_rows(bench_root / "01_registry" / f"{doc_type}_registry.csv", REGISTRY_HEADERS[doc_type], copied_rows)
    return bench_root


def _recommendation(previous_rows: list[dict[str, Any]], workers: int, summary: dict[str, Any], duration: float) -> str:
    if summary.get("failure_count", 0) > 0:
        return "Lower workers or inspect failures before increasing concurrency."
    if workers == 1:
        return "Baseline."
    baseline = next((row for row in previous_rows if int(row["llm_workers"]) == 1), None)
    if not baseline:
        return "Compare with workers=1 baseline."
    base_duration = float(baseline["total_duration_seconds"])
    if workers == 2 and duration < base_duration * 0.9:
        return "Recommend 2 if local machine remains responsive."
    if workers > 2:
        return "Do not increase unless speed improves without failures or timeouts."
    return "Keep workers=1 unless benchmark shows stable improvement."


def _write_outputs(root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = root / "08_reports" / "parallel_benchmark_results.csv"
    write_csv_rows(csv_path, BENCHMARK_HEADERS, rows)
    md_path = root / "08_reports" / "parallel_benchmark_report.md"
    lines = ["# Parallel Benchmark Report", ""]
    for row in rows:
        lines.append(
            f"- doc_type={row['doc_type']} sample_size={row['sample_size']} "
            f"llm_workers={row['llm_workers']} success={row['success_count']} "
            f"failure={row['failure_count']} duration={row['total_duration_seconds']}s "
            f"docs_per_minute={row['docs_per_minute']} recommendation={row['recommendation']}"
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_workers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark local parallel qwen3:8b scoring.")
    parser.add_argument("--doc-type", choices=["mda", "news"], required=True)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--workers", default="1,2,3,4")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    rows = run_benchmark(
        args.doc_type,
        args.sample_size,
        _parse_workers(args.workers),
        mock_mode=args.mock,
        timeout_seconds=args.timeout_seconds,
    )
    for row in rows:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
