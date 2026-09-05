from __future__ import annotations

import json
from pathlib import Path

from backfill_qwen_runtime_metadata import backfill_mda_runtime_metadata, runtime_config_for
from scoring_utils import MDA_DOCUMENT_HEADERS, RATINGS_LONG_HEADERS, read_csv_rows, write_csv_rows


def test_mda_runtime_backfill_adds_missing_fields_to_raw_json_and_score_tables(tmp_path: Path) -> None:
    root = tmp_path / "06_scoring"
    raw_dir = root / "05_raw_model_outputs" / "mda_final_full"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "MDA_TEST_AR01_attempt_1.json"
    raw_path.write_text(
        json.dumps(
            {
                "model_name": "qwen3:8b",
                "temperature": 0,
                "seed": 42,
                "num_ctx": 4096,
                "content": "{\"score\": 4}",
            }
        ),
        encoding="utf-8",
    )
    write_csv_rows(
        root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv",
        RATINGS_LONG_HEADERS,
        [
            {
                "rating_id": "MDA_TEST_AR01",
                "document_id": "MDA_TEST",
                "doc_type": "mda",
                "dimension_code": "AR01",
                "model_name": "qwen3:8b",
                "llm_backend": "ollama",
                "raw_output_path": "05_raw_model_outputs/mda_final_full/MDA_TEST_AR01_attempt_1.json",
            }
        ],
    )
    write_csv_rows(
        root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [
            {
                "document_id": "MDA_TEST",
                "doc_type": "mda",
                "model_name": "qwen3:8b",
                "llm_backend": "ollama",
            }
        ],
    )

    summary = backfill_mda_runtime_metadata(root)

    assert summary["raw_json_updated"] == 1
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    for field in ["num_predict", "think", "stream", "json_schema_mode", "scorebook_version", "prompt_version"]:
        assert field in payload
    ratings = read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_ratings_long.csv")
    docs = read_csv_rows(root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv")
    for field in ["num_predict", "think", "stream", "json_schema_mode", "scorebook_version", "prompt_version"]:
        assert ratings[0][field] != ""
        assert docs[0][field] != ""
    assert (root / "08_reports" / "mda_qwen_runtime_config.md").exists()


def test_runtime_config_defaults_match_qwen_contract() -> None:
    config = runtime_config_for("mda")
    assert config["model_name"] == "qwen3:8b"
    assert config["llm_backend"] == "ollama"
    assert config["temperature"] == 0
    assert config["seed"] == 42
    assert config["num_ctx"] == 4096
    assert config["num_predict"] == 192
    assert config["think"] is False
    assert config["stream"] is False
    assert config["json_schema_mode"]
    assert config["scorebook_version"]
    assert config["prompt_version"]
