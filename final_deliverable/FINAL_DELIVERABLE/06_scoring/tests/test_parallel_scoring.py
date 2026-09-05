import csv
import json
import shutil
import time
from pathlib import Path

from aggregate_scores import aggregate_scores
from benchmark_parallel_scoring import run_benchmark
from build_review_log import build_review_log
from parallel_scoring import atomic_write_json, run_parallel_pipeline, warn_if_too_many_workers


def _copy_static_contracts(root: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for rel in [
        "04_prompts/scoring_output_schema.json",
        "04_prompts/mda_scoring_prompt.txt",
        "04_prompts/news_scoring_prompt.txt",
    ]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_root / rel, target)


def _write_mda_fixture(root: Path, count: int = 3) -> None:
    _copy_static_contracts(root)
    text_dir = root / "02_extracted_text" / "mda"
    text_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(count):
        document_id = f"MDA_000{idx}_TST_{2020 + idx}"
        text_path = text_dir / f"{document_id}.txt"
        text_path.write_text(
            "[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.\n\n"
            "[MDA_P002]\nManagement discusses costs, risks, and outlook.",
            encoding="utf-8",
        )
        rows.append(
            {
                "document_id": document_id,
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": f"000{idx}",
                "ticker": "TST",
                "company_name": "Test Food Berhad",
                "report_year": str(2020 + idx),
                "extraction_status": "success",
                "text_path": f"02_extracted_text/mda/{document_id}.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        )
    registry = root / "01_registry" / "mda_registry.csv"
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_parallel_mock_scoring_writes_one_parsed_json_per_document(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_fixture(root, count=3)

    summary = run_parallel_pipeline("mda", "full", root=root, llm_workers=2, mock_mode=True)

    parsed = sorted((root / "06_ratings" / "per_document" / "mda").glob("*_parsed.json"))
    assert summary["success_count"] == 3
    assert len(parsed) == 3
    assert len({p.name for p in parsed}) == 3


def test_resume_skips_completed_documents_without_duplicate_progress(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_fixture(root, count=2)
    run_parallel_pipeline("mda", "full", root=root, llm_workers=2, mock_mode=True)

    summary = run_parallel_pipeline("mda", "full", root=root, llm_workers=2, mock_mode=True, resume=True, overwrite=False)

    assert summary["skipped_count"] == 2
    log_rows = [
        json.loads(line)
        for line in (root / "08_reports" / "scoring_progress_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert {row["status"] for row in log_rows[-2:]} == {"skipped"}


def test_overwrite_true_regenerates_completed_document(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_fixture(root, count=1)
    run_parallel_pipeline("mda", "full", root=root, llm_workers=1, mock_mode=True)
    parsed = next((root / "06_ratings" / "per_document" / "mda").glob("*_parsed.json"))
    first_mtime = parsed.stat().st_mtime
    time.sleep(0.01)

    summary = run_parallel_pipeline("mda", "full", root=root, llm_workers=1, mock_mode=True, resume=True, overwrite=True)

    assert summary["success_count"] == 1
    assert parsed.stat().st_mtime > first_mtime


def test_aggregate_scores_keeps_document_ids_unique_and_total_scores_reproducible(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_fixture(root, count=2)
    run_parallel_pipeline("mda", "full", root=root, llm_workers=2, mock_mode=True)

    result = aggregate_scores("mda", root=root)
    rows = list(csv.DictReader((root / "06_ratings" / "mda_document_scores.csv").open(encoding="utf-8")))

    assert result["document_count"] == 2
    assert len({row["document_id"] for row in rows}) == 2
    assert all(float(row["total_score_100"]) >= 0 for row in rows)


def test_failed_per_document_output_enters_review_log(tmp_path):
    root = tmp_path / "06_scoring"
    failed_dir = root / "06_ratings" / "per_document" / "mda"
    failed_dir.mkdir(parents=True)
    atomic_write_json(
        failed_dir / "MDA_FAILED_parsed.json",
        {
            "document_id": "MDA_FAILED",
            "doc_type": "mda",
            "status": "failed",
            "result": {
                "review_reasons": ["json_parse_failed"],
                "error_type": "json_parse_failed",
                "error_message": "bad json",
            },
        },
    )

    build_review_log("mda", root=root)
    rows = list(csv.DictReader((root / "07_review" / "review_log.csv").open(encoding="utf-8")))

    assert rows[0]["document_id"] == "MDA_FAILED"
    assert rows[0]["review_type"] == "manual_required"


def test_atomic_write_json_replaces_tmp_with_complete_file(tmp_path):
    path = tmp_path / "payload.json"

    atomic_write_json(path, {"ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert not path.with_suffix(".json.tmp").exists()


def test_llm_workers_above_four_prints_warning(capsys):
    warn_if_too_many_workers(5)
    captured = capsys.readouterr()
    assert "Local qwen3:8b scoring may become slower or unstable" in captured.err


def test_benchmark_script_generates_results_table(tmp_path):
    root = tmp_path / "06_scoring"
    _write_mda_fixture(root, count=2)

    rows = run_benchmark("mda", sample_size=2, workers=[1, 2], root=root, mock_mode=True)

    output = root / "08_reports" / "parallel_benchmark_results.csv"
    assert len(rows) == 2
    assert output.exists()
    csv_rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert csv_rows[0]["llm_workers"] == "1"
