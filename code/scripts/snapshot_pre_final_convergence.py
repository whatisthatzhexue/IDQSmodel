from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, save_json, write_csv_rows
from stability_common import write_markdown


SNAPSHOT_DIRS = [
    "PAPER_OUTPUT",
    "FINAL_OUTPUT",
    "06_ratings",
    "07_review",
    "11_stability_analysis",
    "09_cross_validation",
    "05_raw_model_outputs",
    "03_numeric_checks",
    "01_registry",
    "00_scorebook",
    "04_prompts",
]

HASH_HEADERS = ["relative_path", "size_bytes", "sha256"]


def snapshot_pre_final_convergence(root: Path = SCORING_ROOT, timestamp: str | None = None) -> dict[str, Any]:
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir = root / "archive" / f"pre_final_convergence_{timestamp}"
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True)
    copied_dirs: list[str] = []
    missing_dirs: list[str] = []
    for rel in SNAPSHOT_DIRS:
        src = root / rel
        dst = snapshot_dir / rel
        if src.exists():
            shutil.copytree(src, dst, symlinks=True)
            copied_dirs.append(rel)
        else:
            missing_dirs.append(rel)
    hashes = _hash_rows(snapshot_dir)
    write_csv_rows(snapshot_dir / "snapshot_file_hashes.csv", HASH_HEADERS, hashes)
    summary = {
        "snapshot_dir": str(snapshot_dir),
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "copied_dirs": copied_dirs,
        "missing_dirs": missing_dirs,
        "file_count": len(hashes),
    }
    save_json(snapshot_dir / "snapshot_manifest.json", summary)
    _write_summary(snapshot_dir / "snapshot_summary.md", summary)
    return summary


def _hash_rows(snapshot_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(item for item in snapshot_dir.rglob("*") if item.is_file()):
        if path.name in {"snapshot_manifest.json", "snapshot_file_hashes.csv", "snapshot_summary.md"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"relative_path": str(path.relative_to(snapshot_dir)), "size_bytes": path.stat().st_size, "sha256": digest})
    return rows


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Pre-final Convergence Snapshot",
        "",
        f"- snapshot_dir: {summary['snapshot_dir']}",
        f"- created_at: {summary['created_at']}",
        f"- file_count: {summary['file_count']}",
        f"- copied_dirs: {', '.join(summary['copied_dirs']) or 'none'}",
        f"- missing_dirs: {', '.join(summary['missing_dirs']) or 'none'}",
        "",
        "This snapshot freezes the pre-convergence state without overwriting earlier archives.",
    ]
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Snapshot current scoring state before final convergence.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    print(json.dumps(snapshot_pre_final_convergence(args.root, args.timestamp), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
