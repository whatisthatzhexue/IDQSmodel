from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import stat
from pathlib import Path
from typing import Any

from filter_news_common import FILTER_NEWS_CANDIDATES, SCORING_ROOT
from scoring_utils import read_csv_rows, write_csv_rows


DESKTOP_SCORE_DIR = Path("/Users/zhaojiahao/Desktop/score")

SCRIPT_FILES = [
    "filter_news_common.py",
    "import_filter_news_zip.py",
    "audit_filter_news_relevance.py",
    "build_news_registry_from_filter_news.py",
    "extract_filter_news_texts.py",
    "run_filter_news_final_scoring.py",
    "run_filter_news_stability.py",
    "finalize_filter_news_cross_validation.py",
    "score_package_checks.py",
    "scoring_utils.py",
    "llm_clients.py",
    "json_stabilization.py",
    "run_cross_validation.py",
    "check_ollama.py",
    "run_scoring.py",
    "mda_numeric_checker.py",
    "validate_model_scores.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_desktop_score_package(root: Path = SCORING_ROOT, target: Path = DESKTOP_SCORE_DIR) -> dict[str, Any]:
    if target.exists():
        shutil.rmtree(target)
    make_dirs(target)
    copy_code(root, target)
    copy_inputs(root, target)
    copy_results(root, target)
    write_manifest(target)
    ensure_no_absolute_paths(target)
    return package_summary(target)


def make_dirs(target: Path) -> None:
    for rel in [
        "code/scripts",
        "code/tests",
        "code/scorebooks",
        "code/prompts",
        "input_files/mda_texts",
        "input_files/mda_numeric_checks",
        "input_files/filter_news_original",
        "results/MDA",
        "results/NEWS",
        "results/CROSS_VALIDATION",
        "results/HUMAN_VALIDATION",
    ]:
        (target / rel).mkdir(parents=True, exist_ok=True)


def copy_code(root: Path, target: Path) -> None:
    scripts_dst = target / "code" / "scripts"
    for name in SCRIPT_FILES:
        src = root / "scripts" / name
        if src.exists():
            text = src.read_text(encoding="utf-8")
            text = sanitize_text(text)
            (scripts_dst / name).write_text(text, encoding="utf-8")
    scorebook_dir = root / "00_scorebook"
    if scorebook_dir.exists():
        for path in scorebook_dir.iterdir():
            if path.is_file():
                (target / "code" / "scorebooks" / path.name).write_text(sanitize_text(path.read_text(encoding="utf-8")), encoding="utf-8")
    prompts_dir = root / "04_prompts"
    if prompts_dir.exists():
        for path in prompts_dir.iterdir():
            if path.is_file():
                (target / "code" / "prompts" / path.name).write_text(sanitize_text(path.read_text(encoding="utf-8")), encoding="utf-8")
    (target / "code" / "requirements.txt").write_text("openpyxl\n", encoding="utf-8")
    run_script = target / "code" / "run_reproduce.sh"
    run_script.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        "cd \"$(dirname \"$0\")\"\n"
        "python3 scripts/score_package_checks.py\n"
        "python3 -m pytest -p no:cacheprovider tests -q\n",
        encoding="utf-8",
    )
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR)
    (target / "code" / "tests" / "test_score_package_checks.py").write_text(
        "from pathlib import Path\n"
        "import subprocess\n\n"
        "def test_score_package_checks_script_passes():\n"
        "    code_dir = Path(__file__).resolve().parents[1]\n"
        "    result = subprocess.run(['python3', 'scripts/score_package_checks.py'], cwd=code_dir, text=True, capture_output=True)\n"
        "    assert result.returncode == 0, result.stdout + result.stderr\n",
        encoding="utf-8",
    )


def copy_inputs(root: Path, target: Path) -> None:
    mda_rows = read_csv_rows(root / "01_registry" / "mda_registry.csv")
    sanitized_registry = []
    for row in mda_rows:
        document_id = row.get("document_id", "")
        text_src = root / row.get("text_path", "")
        if text_src.exists():
            shutil.copy2(text_src, target / "input_files" / "mda_texts" / f"{document_id}.txt")
        source_files = [Path(part.strip()).name for part in row.get("source_path", "").split(";") if part.strip()]
        new_row = dict(row)
        new_row["text_path"] = f"mda_texts/{document_id}.txt"
        new_row["source_path"] = "; ".join(source_files)
        sanitized_registry.append(new_row)
    if sanitized_registry:
        write_csv_rows(target / "input_files" / "mda_registry.csv", list(sanitized_registry[0].keys()), sanitized_registry)
    else:
        write_csv_rows(target / "input_files" / "mda_registry.csv", [], [])
    for name in ["mda_v2_numeric_checks_summary.csv", "mda_v2_numeric_discrepancy_audit.csv", "mda_numeric_checks_summary.csv"]:
        src = root / "03_numeric_checks" / name
        if src.exists():
            (target / "input_files" / "mda_numeric_checks" / name).write_text(sanitize_text(src.read_text(encoding="utf-8")), encoding="utf-8")
    raw_news = root / "input_news" / "news_raw_from_filter_news.csv"
    if raw_news.exists():
        shutil.copy2(raw_news, target / "input_files" / "news_raw_from_filter_news.csv")
    else:
        write_csv_rows(target / "input_files" / "news_raw_from_filter_news.csv", ["ticker", "company_name", "publish_date", "source_name", "title", "url", "article_text", "original_file", "original_file_hash"], [])
    source = next((path for path in FILTER_NEWS_CANDIDATES if path.exists()), None)
    if source:
        if source.is_file():
            shutil.copy2(source, target / "input_files" / "filter_news_original" / source.name)
        else:
            shutil.copytree(source, target / "input_files" / "filter_news_original" / source.name, dirs_exist_ok=True)


def copy_results(root: Path, target: Path) -> None:
    result_map = {
        root / "06_ratings" / "mda_final_v2" / "mda_final_document_scores.csv": target / "results" / "MDA" / "mda_final_document_scores.csv",
        root / "06_ratings" / "mda_final_v2" / "mda_final_ratings_long.csv": target / "results" / "MDA" / "mda_final_ratings_long.csv",
        root / "08_reports" / "mda_semantic_evidence_audit_v2.csv": target / "results" / "MDA" / "mda_semantic_evidence_audit_v2.csv",
        root / "07_review" / "mda_semantic_evidence_review_pool_v2.csv": target / "results" / "MDA" / "mda_semantic_evidence_review_pool_v2.csv",
        root / "06_ratings" / "mda_final_full" / "mda_final_failed_cases.csv": target / "results" / "MDA" / "mda_final_failed_cases.csv",
        root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv": target / "results" / "NEWS" / "news_final_document_scores.csv",
        root / "06_ratings" / "news_final_full" / "news_final_ratings_long.csv": target / "results" / "NEWS" / "news_final_ratings_long.csv",
        root / "06_ratings" / "news_final_full" / "news_final_failed_cases.csv": target / "results" / "NEWS" / "news_final_failed_cases.csv",
        root / "08_reports" / "filter_news_relevance_audit.csv": target / "results" / "NEWS" / "filter_news_relevance_audit.csv",
        root / "09_cross_validation" / "final_real_run" / "eligible_pairs.csv": target / "results" / "CROSS_VALIDATION" / "eligible_pairs.csv",
        root / "09_cross_validation" / "final_real_run" / "claim_links.csv": target / "results" / "CROSS_VALIDATION" / "claim_links.csv",
        root / "09_cross_validation" / "final_real_run" / "company_year_cross_validation_summary.csv": target / "results" / "CROSS_VALIDATION" / "company_year_cross_validation_summary.csv",
        root / "HUMAN_VALIDATION" / "mda_human_validation_sample_v2.xlsx": target / "results" / "HUMAN_VALIDATION" / "mda_human_validation_sample_v2.xlsx",
        root / "HUMAN_VALIDATION" / "mda_human_validation_instructions.md": target / "results" / "HUMAN_VALIDATION" / "mda_human_validation_instructions.md",
    }
    for src, dst in result_map.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            if src.suffix.lower() in {".csv", ".md", ".txt", ".json"}:
                dst.write_text(sanitize_text(src.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
            else:
                shutil.copy2(src, dst)
        elif dst.suffix.lower() == ".csv":
            write_csv_rows(dst, ["missing"], [])
        else:
            dst.write_text("", encoding="utf-8")


def write_manifest(target: Path) -> None:
    rows = []
    for path in sorted(target.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": path.relative_to(target).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_csv_rows(target / "input_files" / "source_file_manifest.csv", ["relative_path", "bytes", "sha256"], rows)


def sanitize_text(text: str) -> str:
    text = text.replace("/Users/zhaojiahao/Desktop/score", "<DESKTOP_SCORE>")
    text = text.replace("/Users/zhaojiahao/Desktop/马来西亚F&B年报（49）/06_scoring", "<PROJECT_ROOT>")
    text = text.replace("/Users/zhaojiahao/Desktop/马来西亚F&B年报（49）", "<PROJECT_PARENT>")
    text = text.replace("/Users/zhaojiahao", "<USER_HOME>")
    return text


def ensure_no_absolute_paths(target: Path) -> None:
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".md", ".txt", ".json", ".py", ".sh", ".yaml", ".yml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "/Users/zhaojiahao" in text:
            path.write_text(sanitize_text(text), encoding="utf-8")


def package_summary(target: Path) -> dict[str, Any]:
    def row_count(path: Path) -> int:
        with path.open(newline="", encoding="utf-8") as fh:
            return max(sum(1 for _ in csv.reader(fh)) - 1, 0)

    return {
        "score_folder": str(target),
        "mda_document_rows": row_count(target / "results" / "MDA" / "mda_final_document_scores.csv"),
        "mda_ratings_long_rows": row_count(target / "results" / "MDA" / "mda_final_ratings_long.csv"),
        "news_document_rows": row_count(target / "results" / "NEWS" / "news_final_document_scores.csv"),
        "news_ratings_long_rows": row_count(target / "results" / "NEWS" / "news_final_ratings_long.csv"),
        "eligible_pairs": row_count(target / "results" / "CROSS_VALIDATION" / "eligible_pairs.csv"),
        "claim_links": row_count(target / "results" / "CROSS_VALIDATION" / "claim_links.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Desktop score package with final MD&A and real Filter News outputs.")
    parser.add_argument("--target", type=Path, default=DESKTOP_SCORE_DIR)
    args = parser.parse_args()
    summary = build_desktop_score_package(target=args.target)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
