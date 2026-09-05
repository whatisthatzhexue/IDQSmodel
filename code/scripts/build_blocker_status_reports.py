from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from check_ollama import run_check
from paper_utils import load_json_safe, load_paper_status, real_news_rows
from scoring_utils import SCORING_ROOT, read_csv_rows
from stability_common import write_markdown


CANONICAL_NEWS_DIRS = ["01_raw_news", "news", "raw_news", "06_scoring/input_news", "data/news"]


def build_blocker_status_reports(root: Path = SCORING_ROOT, ollama_status: dict[str, Any] | None = None) -> dict[str, Any]:
    status = load_paper_status(root)
    metadata, _ = load_json_safe(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json")
    ollama = ollama_status if ollama_status is not None else run_check(root=root)
    canonical_news_files = _canonical_news_files(root)
    real_registry_rows = real_news_rows(root)
    news_text_count = _real_news_text_count(root, real_registry_rows)
    claim_links = read_csv_rows(root / "09_cross_validation" / "claim_links.csv")
    out_dir = root / "PAPER_OUTPUT" / "00_status"
    if not canonical_news_files:
        _write_real_news_missing_blocker(out_dir / "real_news_missing_blocker.md", real_registry_rows, news_text_count)
    _write_news_ready_report(out_dir / "news_ready_report.md", status, real_registry_rows, news_text_count, canonical_news_files, ollama)
    if not status.get("news_ready") or int(status.get("claim_link_count") or 0) == 0:
        _write_cross_validation_blocker(out_dir / "cross_validation_blocker_diagnosis.md", status, claim_links, real_registry_rows)
    _write_final_blocker_list(out_dir / "final_blocker_list.md", status, metadata, canonical_news_files, real_registry_rows, ollama)
    return {
        "paper_ready": status["paper_ready"],
        "freeze_allowed": status["freeze_allowed"],
        "mda_ready": status["mda_ready"],
        "news_ready": status["news_ready"],
        "cross_validation_ready": status["cross_validation_ready"],
        "reproducibility_ready": status["reproducibility_ready"],
        "canonical_news_file_count": len(canonical_news_files),
        "real_news_registry_count": len(real_registry_rows),
        "real_news_text_count": news_text_count,
        "claim_link_count": status["claim_link_count"],
        "ollama_available": ollama.get("ollama_available"),
        "blocking_reasons": status["blocking_reasons"],
    }


def _canonical_news_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for rel_dir in CANONICAL_NEWS_DIRS:
        path = root / rel_dir
        if path.exists():
            files.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in {".csv", ".xlsx", ".xls", ".txt", ".json"})
    return sorted(files)


def _real_news_text_count(root: Path, rows: list[dict[str, str]]) -> int:
    count = 0
    for row in rows:
        text_path = row.get("text_path", "")
        if text_path and (root / text_path).exists():
            count += 1
    return count


def _write_real_news_missing_blocker(path: Path, real_rows: list[dict[str, str]], text_count: int) -> None:
    write_markdown(
        path,
        [
            "# Real News Missing Blocker",
            "",
            "- canonical_raw_news_input_dirs_checked: 01_raw_news, news, raw_news, 06_scoring/input_news, data/news",
            "- canonical_raw_news_files_found: 0",
            f"- existing_real_news_registry_rows: {len(real_rows)}",
            f"- existing_real_news_text_count: {text_count}",
            "",
            "- news_ready remains false.",
            "- cross-validation cannot be interpreted empirically.",
            "- synthetic news is not used.",
            "",
            "Required user input: provide `news_raw.xlsx or news_raw.csv` with fields ticker, company_name, publish_date, source_name, title, url, article_text.",
            "The `url` field may be missing, but title/source/date/text must be retained.",
        ],
    )


def _write_news_ready_report(path: Path, status: dict[str, Any], real_rows: list[dict[str, str]], text_count: int, canonical_files: list[Path], ollama: dict[str, Any]) -> None:
    write_markdown(
        path,
        [
            "# News Ready Report",
            "",
            f"- news_ready: {str(status['news_ready']).lower()}",
            f"- news_final_dataset_size: {status['news_final_dataset_size']}",
            f"- synthetic_news_count: {status['synthetic_news_count']}",
            f"- canonical_raw_news_file_count: {len(canonical_files)}",
            f"- real_news_registry_count: {len(real_rows)}",
            f"- real_news_text_count: {text_count}",
            f"- ollama_available: {str(ollama.get('ollama_available')).lower()}",
            f"- qwen3_model_available: {str(ollama.get('model_available')).lower()}",
            f"- qwen3_errors: {'; '.join(ollama.get('errors', [])) or 'none'}",
            "",
            "Conclusion: News is not ready for empirical analysis until real News final scores and valid schema outputs exist.",
        ],
    )


def _write_cross_validation_blocker(path: Path, status: dict[str, Any], claim_links: list[dict[str, str]], real_rows: list[dict[str, str]]) -> None:
    write_markdown(
        path,
        [
            "# Cross-validation Blocker Diagnosis",
            "",
            f"- cross_validation_ready_flag: {str(status['cross_validation_ready']).lower()}",
            f"- news_ready: {str(status['news_ready']).lower()}",
            f"- claim_link_count: {status['claim_link_count']}",
            f"- claim_links_rows: {len(claim_links)}",
            f"- real_news_registry_count: {len(real_rows)}",
            "",
            "Interpretation: cross-validation must not be described as valid empirical evidence while news_ready=false.",
            "Checks to run after News becomes ready: ticker/company alignment, publish_date coverage, time window, claim extraction, matching strictness, same-source repetition handling, and topic overlap.",
        ],
    )


def _write_final_blocker_list(path: Path, status: dict[str, Any], metadata: dict[str, Any], canonical_files: list[Path], real_rows: list[dict[str, str]], ollama: dict[str, Any]) -> None:
    gates = metadata.get("quality_gates", {}) if isinstance(metadata, dict) else {}
    passed = [name for name, gate in gates.items() if gate.get("passed") is True]
    failed = [name for name, gate in gates.items() if gate.get("passed") is False]
    write_markdown(
        path,
        [
            "# Final Blocker List",
            "",
            "## Passed Gates",
            *[f"- {name}" for name in (passed or ["none"])],
            "",
            "## Failed Gates",
            *[f"- {name}" for name in (failed or status["blocking_reasons"] or ["none"])],
            "",
            "## Reasons",
            f"- MD&A stability insufficient: {status['mda_stability_success_rate']} < 0.85",
            f"- canonical real News raw files present: {len(canonical_files) > 0}",
            f"- real News registry rows: {len(real_rows)}",
            f"- news_final_dataset_size: {status['news_final_dataset_size']}",
            f"- qwen/Ollama available: {ollama.get('ollama_available')}",
            f"- qwen/Ollama errors: {'; '.join(ollama.get('errors', [])) or 'none'}",
            "",
            "## Can Write",
            "- methodology",
            "- scoring framework",
            "- pipeline validation",
            "- limitations",
            "- reproducibility",
            "",
            "## Cannot Write",
            "- final empirical results",
            "- real News empirical results",
            "- substantive cross-validation conclusion",
            "",
            "## Next Human Input",
            "- Start/fix local Ollama qwen3:8b so failed MD&A and real News scoring can run.",
            "- If canonical raw News input is required, provide news_raw.xlsx or news_raw.csv with ticker, company_name, publish_date, source_name, title, url, article_text.",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Write final blocker, News readiness, and cross-validation blocker reports.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_blocker_status_reports(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
