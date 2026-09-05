from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_cross_validation import CLAIM_LINK_HEADERS, COMPANY_YEAR_HEADERS, MDA_CLAIM_HEADERS, NEWS_CLAIM_HEADERS, PAIR_HEADERS
from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finalize_filter_news_cross_validation(root: Path = SCORING_ROOT, run: bool = True) -> dict[str, Any]:
    final_dir = root / "09_cross_validation" / "final_real_run"
    final_dir.mkdir(parents=True, exist_ok=True)
    mda_docs = read_csv_rows(root / "06_ratings" / "mda_final_v2" / "mda_final_document_scores.csv")
    news_docs = read_csv_rows(root / "06_ratings" / "news_final_full" / "news_final_document_scores.csv")
    news_registry = read_csv_rows(root / "01_registry" / "news_registry.csv")
    synthetic_count = sum(1 for row in news_registry if row.get("synthetic_flag", "").lower() == "true")
    mda_ready = len(mda_docs) == 149
    news_ready = bool(news_docs) and synthetic_count == 0
    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "mda_ready": mda_ready,
        "news_final_dataset_size": len(news_docs),
        "synthetic_count": synthetic_count,
        "empirical_ready": False,
        "cross_validation_pipeline_ready": False,
        "cross_validation_empirical_ready": False,
        "mock_mode": None,
        "cross_validation_executed": False,
        "claim_link_count": 0,
        "eligible_pair_count": 0,
        "notes": [],
        "hashes": {},
    }
    if not (mda_ready and news_ready):
        write_empty_cross_validation(final_dir, metadata, "Cross-validation skipped because MD&A ready/news ready/synthetic gates were not all true.")
        return metadata

    if run:
        completed = subprocess.run(
            ["python3", str(root / "scripts" / "run_cross_validation.py"), "--mode", "full"],
            cwd=root,
            text=True,
            capture_output=True,
        )
        metadata["cross_validation_command_returncode"] = completed.returncode
        metadata["cross_validation_stdout_tail"] = completed.stdout[-1000:]
        metadata["cross_validation_stderr_tail"] = completed.stderr[-1000:]
        if completed.returncode != 0:
            metadata["notes"].append("run_cross_validation.py failed; final_real_run written with empty templates.")
            write_empty_cross_validation(final_dir, metadata, "Cross-validation command failed.")
            return metadata

    copies = {
        root / "09_cross_validation" / "mda_claims.csv": final_dir / "mda_claims.csv",
        root / "09_cross_validation" / "news_claims.csv": final_dir / "news_claims.csv",
        root / "09_cross_validation" / "claim_links.csv": final_dir / "claim_links.csv",
        root / "09_cross_validation" / "mda_news_cross_validation_pairs.csv": final_dir / "eligible_pairs.csv",
        root / "09_cross_validation" / "company_year_cross_validation_summary.csv": final_dir / "company_year_cross_validation_summary.csv",
        root / "09_cross_validation" / "cross_validation_report.md": final_dir / "cross_validation_report.md",
    }
    for src, dst in copies.items():
        if src.exists():
            shutil.copy2(src, dst)
    links = read_csv_rows(final_dir / "claim_links.csv")
    pairs = read_csv_rows(final_dir / "eligible_pairs.csv")
    report_text = (final_dir / "cross_validation_report.md").read_text(encoding="utf-8", errors="replace") if (final_dir / "cross_validation_report.md").exists() else ""
    mock_mode = "mock mode: True" in report_text or "mock_mode=true" in report_text or "deterministic_claim_proxy" in report_text
    systemic_conflict = _systemic_conflict(links, pairs)
    empirical_ready = bool(links) and not mock_mode and not systemic_conflict
    metadata.update(
        {
            "empirical_ready": empirical_ready,
            "cross_validation_pipeline_ready": bool(links) or bool(pairs),
            "cross_validation_empirical_ready": empirical_ready,
            "mock_mode": mock_mode,
            "systemic_conflict_flag": systemic_conflict,
            "cross_validation_executed": True,
            "claim_link_count": len(links),
            "eligible_pair_count": len(pairs),
        }
    )
    if len(links) == 0:
        metadata["notes"].append("No semantic claim relation found under current matching rules.")
    if mock_mode:
        metadata["notes"].append("cross_validation_empirical_ready=false because claim scoring used deterministic_claim_proxy / mock_mode=true.")
    if systemic_conflict:
        metadata["notes"].append("cross_validation_empirical_ready=false because systemic conflict or contradiction gates require manual/qwen review.")
    for path in final_dir.glob("*"):
        if path.is_file():
            metadata["hashes"][path.name] = sha256_file(path)
    (final_dir / "cross_validation_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _systemic_conflict(links: list[dict[str, str]], pairs: list[dict[str, str]]) -> bool:
    contradictions = sum(str(row.get("contradiction_flag", "")).strip().lower() in {"1", "true", "yes"} for row in links)
    rate = contradictions / len(links) if links else 0
    pair_contradictions = 0
    for row in pairs:
        try:
            pair_contradictions += int(float(row.get("num_contradictions", "") or 0))
        except ValueError:
            continue
    return (contradictions >= 3 and rate > 0.20) or (pair_contradictions >= 3 and bool(pairs))


def write_empty_cross_validation(final_dir: Path, metadata: dict[str, Any], note: str) -> None:
    metadata["notes"].append(note)
    write_csv_rows(final_dir / "mda_claims.csv", MDA_CLAIM_HEADERS, [])
    write_csv_rows(final_dir / "news_claims.csv", NEWS_CLAIM_HEADERS, [])
    write_csv_rows(final_dir / "eligible_pairs.csv", PAIR_HEADERS, [])
    write_csv_rows(final_dir / "claim_links.csv", CLAIM_LINK_HEADERS, [])
    write_csv_rows(final_dir / "company_year_cross_validation_summary.csv", COMPANY_YEAR_HEADERS, [])
    report = [
        "# Final Real Cross-Validation Report",
        "",
        note,
        "",
        "No fabricated claim links were generated.",
    ]
    (final_dir / "cross_validation_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    for path in final_dir.glob("*"):
        if path.is_file() and path.name != "cross_validation_run_metadata.json":
            metadata["hashes"][path.name] = sha256_file(path)
    (final_dir / "cross_validation_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalize cross-validation outputs for real Filter News run.")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()
    metadata = finalize_filter_news_cross_validation(run=not args.skip_run)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
