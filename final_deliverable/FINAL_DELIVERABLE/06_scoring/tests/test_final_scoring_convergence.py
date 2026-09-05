import json
from pathlib import Path

from parallel_scoring import run_parallel_pipeline
from repair_json_pipeline import repair_then_validate_json
from run_scoring import run_pipeline
from run_single_dimension_scoring import _single_dimension_prompt, run_single_dimension_scoring
from scoring_stability_monitor import monitor_scoring_stability
from scoring_utils import REGISTRY_HEADERS, read_csv_rows, write_csv_rows
from validate_json_outputs import SIMPLE_JSON_FIELDS, validate_simple_dimension_json


def _write_dual_fixture(root: Path) -> None:
    mda_dir = root / "02_extracted_text" / "mda"
    news_dir = root / "02_extracted_text" / "news"
    mda_dir.mkdir(parents=True, exist_ok=True)
    news_dir.mkdir(parents=True, exist_ok=True)
    (mda_dir / "MDA_FINAL_2024.txt").write_text(
        "[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.\n\n"
        "[MDA_P002]\nManagement explains volume demand, cost pressure, risk controls, and cautious outlook.",
        encoding="utf-8",
    )
    (news_dir / "NEWS_FINAL_2024.txt").write_text(
        "[TITLE]\nFinal Food revenue improves\n\n"
        "[LEAD]\nThe company cited higher demand.\n\n"
        "[NEWS_P001]\nAccording to the company statement, revenue rose after stronger export demand and stable pricing.",
        encoding="utf-8",
    )
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        REGISTRY_HEADERS["mda"],
        [
            {
                "document_id": "MDA_FINAL_2024",
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": "8888",
                "ticker": "FINAL",
                "company_name": "Final Food Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": "02_extracted_text/mda/MDA_FINAL_2024.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        ],
    )
    write_csv_rows(
        root / "01_registry" / "news_registry.csv",
        REGISTRY_HEADERS["news"],
        [
            {
                "document_id": "NEWS_FINAL_2024",
                "doc_type": "news",
                "include_flag": "Yes",
                "company_name": "Final Food Berhad",
                "ticker": "FINAL",
                "source_name": "Synthetic Wire",
                "publish_date": "2024-06-16",
                "title": "Final Food revenue improves",
                "url": "",
                "text_path": "02_extracted_text/news/NEWS_FINAL_2024.txt",
                "notes": "",
            }
        ],
    )


def test_single_dimension_engine_scores_mda_and_news_with_simple_json_contract(tmp_path):
    root = tmp_path / "06_scoring"
    _write_dual_fixture(root)

    mda_summary = run_single_dimension_scoring(
        "mda",
        root=root,
        output_name="mda_final_single",
        ratings_prefix="mda_final",
        mock=True,
        overwrite=True,
    )
    news_summary = run_single_dimension_scoring(
        "news",
        root=root,
        output_name="news_final_single",
        ratings_prefix="news_final",
        mock=True,
        overwrite=True,
    )

    assert mda_summary["dimension_task_count"] == 5
    assert news_summary["dimension_task_count"] == 5
    assert mda_summary["success_count"] == 1
    assert news_summary["success_count"] == 1
    for output_name in ["mda_final_single", "news_final_single"]:
        raw_outputs = sorted((root / "05_raw_model_outputs" / output_name).glob("*.json"))
        assert len(raw_outputs) == 5
        raw_payload = json.loads(raw_outputs[0].read_text(encoding="utf-8"))
        assert set(raw_payload["parsed_simple"]) == SIMPLE_JSON_FIELDS
    mda_rows = read_csv_rows(root / "06_ratings" / "mda_final_single" / "mda_final_document_scores.csv")
    news_rows = read_csv_rows(root / "06_ratings" / "news_final_single" / "news_final_document_scores.csv")
    assert mda_rows[0]["dimension_count"] == "5"
    assert news_rows[0]["dimension_count"] == "5"
    assert float(mda_rows[0]["total_score_100"]) >= 0
    assert float(news_rows[0]["total_score_100"]) >= 0


def test_validation_layer_rejects_multidimension_or_missing_evidence():
    text = "[NEWS_P001]\nTraceable paragraph."
    issues = validate_simple_dimension_json(
        {"dimension_scores": [{"score": 4}], "score": 4, "evidence": "NEWS_P001", "reason": "ok"},
        "news",
        "N01",
        text,
    )
    assert "unexpected_field_dimension_scores" in issues
    assert "nested_value_dimension_scores" in issues
    assert validate_simple_dimension_json({"score": 4, "evidence": "NEWS_P999", "reason": "ok"}, "news", "N01", text) == [
        "evidence_locator_not_found"
    ]


def test_repair_pipeline_repairs_wrapped_json_and_manual_required_on_bad_output():
    repaired = repair_then_validate_json("```json\n{\"score\": 4, \"evidence\": \"MDA_P001\", \"reason\": \"ok\"}\n```", "mda", "AR01", "[MDA_P001]\nText")
    failed = repair_then_validate_json("not json", "mda", "AR01", "[MDA_P001]\nText")

    assert repaired["status"] == "valid"
    assert repaired["payload"] == {"score": 4, "evidence": "MDA_P001", "reason": "ok"}
    assert failed["status"] == "manual_required"
    assert failed["payload"] is None


def test_ar02_prompt_is_python_numeric_checker_only():
    prompt = _single_dimension_prompt(
        doc_type="mda",
        document_id="MDA_AR02",
        metadata={"document_id": "MDA_AR02"},
        dimension={"code": "AR02", "name": "MD&A internal numeric consistency and traceability"},
        text="[MDA_P001]\nRevenue increased.",
        numeric_result={"overall_numeric_consistency": "problematic"},
    )

    assert "numeric_check_summary" in prompt
    assert "LLM must not calculate" in prompt
    assert "Python numeric_checker" in prompt
    assert "Only explain" in prompt


def test_legacy_real_document_level_entrypoints_are_blocked_for_mda_and_news(tmp_path):
    root = tmp_path / "06_scoring"
    _write_dual_fixture(root)
    for doc_type in ["mda", "news"]:
        for runner in [
            lambda current=doc_type: run_pipeline(current, "full", root=root, mock_mode=False),
            lambda current=doc_type: run_parallel_pipeline(current, "full", root=root, mock_mode=False),
        ]:
            try:
                runner()
            except ValueError as exc:
                assert "single-dimension" in str(exc)
            else:
                raise AssertionError(f"{doc_type} real document-level scoring should be blocked")


def test_stability_monitor_flags_threshold_failures(tmp_path):
    root = tmp_path / "06_scoring"
    out_dir = root / "06_ratings" / "monitor_bad" / "per_dimension"
    out_dir.mkdir(parents=True)
    for idx, status in enumerate(["success"] * 4 + ["failed"], start=1):
        record = {
            "document_id": f"DOC_{idx}",
            "doc_type": "news",
            "dimension_code": "N01",
            "status": status,
            "result": {
                "parse_status": "parsed" if status == "success" else "parse_failed",
                "schema_validation_status": "valid" if status == "success" else "invalid",
                "error_type": "" if status == "success" else "manual_required",
            },
        }
        (out_dir / f"DOC_{idx}_N01_parsed.json").write_text(json.dumps(record), encoding="utf-8")

    summary = monitor_scoring_stability(root, "monitor_bad")

    assert summary["system_failure_rate"] == 0.2
    assert summary["passes_thresholds"] is False
    assert "json_success_rate_below_0.98" in summary["violations"]
