from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from scoring_utils import (
    DEFAULT_ENV,
    MDA_DOCUMENT_HEADERS,
    RATINGS_LONG_HEADERS,
    SCOREBOOK_VERSION,
    SCORING_ROOT,
    read_csv_rows,
    write_csv_rows,
)


MDA_PROMPT_VERSION = "mda_single_dimension_context_v2"
UNKNOWN_IF_ABSENT = {"num_predict", "think", "stream", "json_schema_mode", "scorebook_version", "prompt_version"}


def runtime_config_for(doc_type: str = "mda") -> dict[str, Any]:
    return {
        "model_name": os.environ.get("OLLAMA_MODEL", DEFAULT_ENV["OLLAMA_MODEL"]),
        "llm_backend": os.environ.get("SCORING_LLM_BACKEND", "ollama"),
        "temperature": _number("OLLAMA_TEMPERATURE", DEFAULT_ENV["OLLAMA_TEMPERATURE"], float),
        "seed": _number("OLLAMA_SEED", DEFAULT_ENV["OLLAMA_SEED"], int),
        "num_ctx": _number("OLLAMA_NUM_CTX", DEFAULT_ENV["OLLAMA_NUM_CTX"], int),
        "num_predict": _number("OLLAMA_NUM_PREDICT", DEFAULT_ENV["OLLAMA_NUM_PREDICT"], int),
        "think": False,
        "stream": False,
        "json_schema_mode": "ollama_format_schema",
        "scorebook_version": SCOREBOOK_VERSION,
        "prompt_version": MDA_PROMPT_VERSION if doc_type == "mda" else "news_single_dimension_v2_manual_preserve",
    }


def _number(name: str, default: str, cast):
    try:
        return cast(os.environ.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


def backfill_mda_runtime_metadata(root: Path = SCORING_ROOT) -> dict[str, Any]:
    config = runtime_config_for("mda")
    raw_updated = 0
    raw_unknown_fields = 0
    raw_root = root / "05_raw_model_outputs"
    for raw_dir in sorted(path for path in raw_root.glob("mda*") if path.is_dir()):
        for path in raw_dir.glob("*.json"):
            changed, unknown_count = _backfill_raw_json(path, config)
            if changed:
                raw_updated += 1
            raw_unknown_fields += unknown_count

    ratings_updated = 0
    ratings_unknown_fields = 0
    doc_scores_updated = 0
    doc_scores_unknown_fields = 0
    for path in sorted((root / "06_ratings").glob("mda*/**/*ratings_long*.csv")):
        rows = read_csv_rows(path)
        if not rows:
            continue
        new_rows = []
        changed_count = 0
        unknown_count = 0
        for row in rows:
            new_row, changed, row_unknown_count = _backfill_row(row, config)
            new_rows.append(new_row)
            changed_count += int(changed)
            unknown_count += row_unknown_count
        if changed_count:
            write_csv_rows(path, RATINGS_LONG_HEADERS, new_rows)
        ratings_updated += changed_count
        ratings_unknown_fields += unknown_count
    for path in sorted((root / "06_ratings").glob("mda*/**/*document_scores*.csv")):
        rows = read_csv_rows(path)
        if not rows:
            continue
        new_rows = []
        changed_count = 0
        unknown_count = 0
        for row in rows:
            new_row, changed, row_unknown_count = _backfill_row(row, config)
            new_rows.append(new_row)
            changed_count += int(changed)
            unknown_count += row_unknown_count
        if changed_count:
            write_csv_rows(path, MDA_DOCUMENT_HEADERS, new_rows)
        doc_scores_updated += changed_count
        doc_scores_unknown_fields += unknown_count

    report_path = root / "08_reports" / "mda_qwen_runtime_config.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _runtime_report(
            config,
            raw_updated,
            ratings_updated,
            doc_scores_updated,
            raw_unknown_fields,
            ratings_unknown_fields,
            doc_scores_unknown_fields,
        ),
        encoding="utf-8",
    )
    return {
        "raw_json_updated": raw_updated,
        "ratings_long_rows_updated": ratings_updated,
        "document_score_rows_updated": doc_scores_updated,
        "raw_unknown_fields_written": raw_unknown_fields,
        "ratings_unknown_fields_written": ratings_unknown_fields,
        "document_score_unknown_fields_written": doc_scores_unknown_fields,
        "report_path": str(report_path),
        "runtime_config": config,
    }


def _metadata_value(key: str, config: dict[str, Any]) -> Any:
    return "unknown" if key in UNKNOWN_IF_ABSENT else config[key]


def _backfill_raw_json(path: Path, config: dict[str, Any]) -> tuple[bool, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False, 0
    if not isinstance(payload, dict):
        return False, 0
    changed = False
    unknown_count = 0
    for key, value in config.items():
        if key not in payload or payload.get(key) in {"", None}:
            payload[key] = _metadata_value(key, config)
            changed = True
            unknown_count += int(key in UNKNOWN_IF_ABSENT)
    runtime = payload.get("runtime_config") if isinstance(payload.get("runtime_config"), dict) else {}
    for key, value in config.items():
        if key not in runtime or runtime.get(key) in {"", None}:
            runtime[key] = _metadata_value(key, config)
            changed = True
            unknown_count += int(key in UNKNOWN_IF_ABSENT)
    payload["runtime_config"] = runtime
    if changed:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return changed, unknown_count


def _backfill_row(row: dict[str, str], config: dict[str, Any]) -> tuple[dict[str, Any], bool, int]:
    out = dict(row)
    changed = False
    unknown_count = 0
    for key, value in config.items():
        if key in {"llm_backend", "temperature", "seed", "num_ctx", "num_predict", "think", "stream", "json_schema_mode", "scorebook_version", "prompt_version", "model_name"}:
            if not str(out.get(key, "")).strip():
                fill_value = _metadata_value(key, config)
                out[key] = str(fill_value).lower() if isinstance(fill_value, bool) else fill_value
                changed = True
                unknown_count += int(key in UNKNOWN_IF_ABSENT)
    if not str(out.get("scoring_timestamp", "")).strip():
        out["scoring_timestamp"] = out.get("finished_at", "") or out.get("started_at", "")
        changed = True
    return out, changed, unknown_count


def _runtime_report(
    config: dict[str, Any],
    raw_updated: int,
    ratings_updated: int,
    doc_scores_updated: int,
    raw_unknown_fields: int,
    ratings_unknown_fields: int,
    doc_scores_unknown_fields: int,
) -> str:
    lines = ["# MDA qwen3:8b Runtime Configuration", ""]
    for key in ["model_name", "llm_backend", "temperature", "seed", "num_ctx", "num_predict", "think", "stream", "json_schema_mode", "scorebook_version", "prompt_version"]:
        lines.append(f"- {key}: {config[key]}")
    lines.extend(
        [
            "",
            "## Backfill Summary",
            f"- raw_json_updated: {raw_updated}",
            f"- ratings_long_rows_updated: {ratings_updated}",
            f"- document_score_rows_updated: {doc_scores_updated}",
            f"- raw_unknown_fields_written: {raw_unknown_fields}",
            f"- ratings_unknown_fields_written: {ratings_unknown_fields}",
            f"- document_score_unknown_fields_written: {doc_scores_unknown_fields}",
            "- note: this metadata backfill does not change any MDA score, evidence locator, or reason text.",
            "- limitation: when historical raw JSON lacks runtime-only fields such as num_predict, think, stream, json_schema_mode, scorebook_version, or prompt_version, the backfill writes unknown rather than assuming a runtime value.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill qwen runtime metadata into existing MDA outputs without rescoring.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(backfill_mda_runtime_metadata(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
