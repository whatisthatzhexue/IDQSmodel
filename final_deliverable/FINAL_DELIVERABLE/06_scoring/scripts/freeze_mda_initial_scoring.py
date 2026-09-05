from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_mda_initial_scoring(root: Path = SCORING_ROOT) -> dict[str, Any]:
    archive = root / "archive" / "v1_initial_mda_scoring"
    archive.mkdir(parents=True, exist_ok=True)
    candidates = [
        root / "06_ratings" / "mda_ratings_long.csv",
        root / "06_ratings" / "mda_document_scores.csv",
        root / "00_scorebook" / "mda_scorebook.yaml",
        root / "04_prompts" / "mda_scoring_prompt.txt",
        root / "08_reports" / "scoring_summary_report.md",
    ]
    raw_dir = root / "05_raw_model_outputs" / "mda"
    source_files: list[str] = []
    file_hashes: dict[str, str] = {}
    for source in candidates:
        if source.exists():
            target = archive / source.name
            shutil.copy2(source, target)
            source_files.append(str(source))
            file_hashes[source.name] = sha256_file(target)
    if raw_dir.exists():
        target_dir = archive / "raw_model_outputs"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(raw_dir, target_dir)
        for path in sorted(target_dir.rglob("*")):
            if path.is_file():
                file_hashes[str(path.relative_to(archive))] = sha256_file(path)
        source_files.append(str(raw_dir))
    doc_rows = read_csv_rows(root / "06_ratings" / "mda_document_scores.csv")
    metadata = {
        "scoring_version": "v1_initial_scoring",
        "text_version": "mda",
        "scorebook_version": _first_value(doc_rows, "scorebook_version", "unknown"),
        "model_name": _first_value(doc_rows, "model_name", "unknown"),
        "llm_backend": _first_value(doc_rows, "llm_backend", "unknown"),
        "frozen_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source_files": source_files,
        "file_hashes": file_hashes,
        "num_documents": len(doc_rows),
        "notes": "Frozen copy only. Original v1 files were not modified.",
    }
    save_json(archive / "v1_metadata.json", metadata)
    return metadata


def _first_value(rows: list[dict[str, str]], field: str, default: str) -> str:
    for row in rows:
        if row.get(field):
            return row[field]
    return default


def main() -> int:
    metadata = freeze_mda_initial_scoring()
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
