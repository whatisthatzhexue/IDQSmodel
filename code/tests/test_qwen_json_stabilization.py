import json
from pathlib import Path

from diagnose_qwen_json_failures import diagnose_qwen_json_failures
from json_stabilization import parse_model_json
from parallel_scoring import run_parallel_pipeline
from run_mda_dimensionwise_scoring import (
    _dimension_prompt,
    _dimension_schema,
    _expand_simple_dimension_payload,
    _validate_simple_dimension_payload,
    run_mda_dimensionwise_scoring,
)
from run_mda_v2_full_rescore import run_mda_v2_full_rescore
from run_mda_v2_stability_batch import run_mda_v2_stability_batch
from run_scoring import run_pipeline
from scoring_utils import MDA_DOCUMENT_HEADERS, REGISTRY_HEADERS, read_csv_rows, write_csv_rows
from validate_dimensionwise_outputs import validate_dimensionwise_outputs


def _copy_contracts(root: Path) -> None:
    source_root = Path(__file__).resolve().parents[1]
    for rel in ["04_prompts/scoring_output_schema.json", "04_prompts/mda_scoring_prompt.txt"]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((source_root / rel).read_text(encoding="utf-8"), encoding="utf-8")


def _write_clean_mda_rows(root: Path, doc_ids: list[str]) -> None:
    rows = []
    for idx, doc_id in enumerate(doc_ids, start=1):
        text_dir = root / "02_extracted_text" / "mda_clean_v2"
        text_dir.mkdir(parents=True, exist_ok=True)
        (text_dir / f"{doc_id}.txt").write_text(
            "[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.\n\n"
            "[MDA_P002]\nManagement describes raw material costs, supply risk, and cautious outlook.",
            encoding="utf-8",
        )
        rows.append(
            {
                "document_id": doc_id,
                "doc_type": "mda",
                "include_flag": "Yes",
                "stock_code": f"{idx:04d}",
                "ticker": "TST",
                "company_name": "Test Food Berhad",
                "report_year": "2024",
                "extraction_status": "success",
                "text_path": f"02_extracted_text/mda_clean_v2/{doc_id}.txt",
                "source_path": "",
                "source_file_sha256": "",
                "notes": "",
            }
        )
    write_csv_rows(root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], rows)


def test_parse_model_json_recovers_common_qwen_wrappers():
    raw = """<think>I should output only JSON.</think>
```json
{"dimension_code": "AR01", "raw_score": 3}
```
extra"""

    result = parse_model_json(raw)

    assert result["ok"] is True
    assert result["payload"] == {"dimension_code": "AR01", "raw_score": 3}
    assert result["has_markdown_fence"] is True
    assert result["has_think_block"] is True
    assert result["recommended_fix"] in {"strip_markdown_and_parse", "strip_think_block_and_parse", "extract_largest_json_object"}


def test_parse_model_json_classifies_empty_and_truncated_outputs():
    empty = parse_model_json("")
    truncated = parse_model_json('{"dimension_code": "AR02", "raw_score": 4')

    assert empty["ok"] is False
    assert empty["failure_type"] == "empty_response"
    assert truncated["ok"] is False
    assert truncated["failure_type"] == "truncated_json"
    assert truncated["missing_closing_brace"] is True
    assert truncated["recommended_fix"] == "repair_json"


def test_qwen_failure_diagnosis_reads_failed_runs_and_writes_report(tmp_path):
    root = tmp_path / "06_scoring"
    raw_dir = root / "05_raw_model_outputs" / "mda_v2_sample"
    parsed_dir = root / "06_ratings" / "mda_v2_sample_rescore" / "per_document"
    raw_dir.mkdir(parents=True)
    parsed_dir.mkdir(parents=True)
    raw_path = raw_dir / "MDA_BAD_JSON_attempt_1.json"
    raw_path.write_text(json.dumps({"content": "```json\n{\"document_id\":\"MDA_BAD_JSON\"}\n```"}), encoding="utf-8")
    (parsed_dir / "MDA_BAD_JSON_parsed.json").write_text(
        json.dumps(
            {
                "document_id": "MDA_BAD_JSON",
                "doc_type": "mda",
                "status": "failed",
                "result": {
                    "error_type": "scoring_failed",
                    "parse_status": "parse_failed",
                    "schema_validation_status": "invalid",
                    "raw_output_path": str(raw_path),
                },
            }
        ),
        encoding="utf-8",
    )

    summary = diagnose_qwen_json_failures(root)

    rows = read_csv_rows(root / "08_reports" / "qwen_json_failure_diagnosis.csv")
    assert summary["failure_count"] == 1
    assert summary["parser_repairable_count"] == 1
    assert rows[0]["failure_type"] == "markdown_wrapped_json"
    assert rows[0]["recommended_fix"] == "strip_markdown_and_parse"
    report = (root / "08_reports" / "qwen_json_failure_diagnosis.md").read_text(encoding="utf-8")
    assert "建议切换到 dimensionwise scoring" in report


def test_dimensionwise_mock_scoring_writes_aggregate_outputs(tmp_path):
    root = tmp_path / "06_scoring"
    _copy_contracts(root)
    _write_clean_mda_rows(root, ["MDA_DIM_2024"])

    summary = run_mda_dimensionwise_scoring(
        root=root,
        doc_ids=["MDA_DIM_2024"],
        output_name="mda_v2_dimensionwise_test",
        ratings_prefix="mda_v2_dimensionwise",
        mock=True,
        overwrite=True,
        llm_workers=1,
    )

    assert summary["success_count"] == 1
    assert summary["dimension_success_count"] == 5
    raw_outputs = sorted((root / "05_raw_model_outputs" / "mda_v2_dimensionwise_test").glob("*.json"))
    assert len(raw_outputs) == 5
    raw_payload = json.loads(raw_outputs[0].read_text(encoding="utf-8"))
    simple_payload = raw_payload["parsed_simple"]
    assert set(simple_payload) == {"score", "evidence", "reason"}
    assert isinstance(simple_payload["score"], int)
    assert simple_payload["evidence"].startswith("MDA_P")
    rows = read_csv_rows(root / "06_ratings" / "mda_v2_dimensionwise_test" / "mda_v2_dimensionwise_document_scores.csv")
    assert len(rows) == 1
    assert rows[0]["dimension_count"] == "5"
    assert all(rows[0][code] for code in ["AR01", "AR02", "AR03", "AR04", "AR05"])
    validation = validate_dimensionwise_outputs(root, output_name="mda_v2_dimensionwise_test", ratings_prefix="mda_v2_dimensionwise")
    assert validation["is_valid"] is True


def test_simple_dimension_schema_rejects_nested_or_five_dimension_outputs():
    schema = _dimension_schema({"code": "AR01", "name": "MD&A content completeness"})
    assert set(schema["required"]) == {"score", "evidence", "reason"}
    assert schema["additionalProperties"] is False
    assert _validate_simple_dimension_payload({"score": 4, "evidence": "MDA_P001", "reason": "clear"}, "AR01", "[MDA_P001] text") == []
    issues = _validate_simple_dimension_payload(
        {"dimension_scores": [{"score": 4}], "score": 4, "evidence": "MDA_P001", "reason": "clear"},
        "AR01",
        "[MDA_P001] text",
    )
    assert "unexpected_field_dimension_scores" in issues


def test_ar02_prompt_is_python_numeric_only_and_expansion_keeps_python_score():
    numeric = {"overall_numeric_consistency": "problematic", "num_material_discrepancies": 2}
    task = type(
        "Task",
        (),
        {
            "document_id": "MDA_AR02",
            "metadata": {"document_id": "MDA_AR02"},
            "dimension_code": "AR02",
            "dimension_definition": {"code": "AR02", "name": "MD&A internal numeric consistency and traceability"},
        },
    )()

    prompt = _dimension_prompt(task, "[MDA_P001] Revenue increased by 20%.", numeric)
    expanded = _expand_simple_dimension_payload(
        {"score": 5, "evidence": "MDA_P001", "reason": "model tried high score"},
        {"code": "AR02", "name": "MD&A internal numeric consistency and traceability"},
        numeric,
    )

    assert "numeric_check_summary" in prompt
    assert "LLM must not calculate percentages" in prompt
    assert "Do not perform arithmetic" in prompt
    assert expanded["raw_score"] == 1
    assert "Python numeric checker" in expanded["reason"]


def test_stability_and_full_scripts_default_to_single_dimension_without_overwriting_pilot(tmp_path):
    root = tmp_path / "06_scoring"
    _copy_contracts(root)
    _write_clean_mda_rows(root, ["MDA_STABLE_2024", "MDA_FULL_2024"])
    write_csv_rows(
        root / "06_ratings" / "mda_document_scores.csv",
        MDA_DOCUMENT_HEADERS,
        [{"document_id": "MDA_V1_ONLY", "doc_type": "mda", "total_score_100": "50"}],
    )
    before = (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8")

    stability_summary = run_mda_v2_stability_batch(root=root, mock=True, overwrite=True, llm_workers=1, cpu_workers=1)
    full_summary = run_mda_v2_full_rescore(root=root, mock=True, limit=1, overwrite=True, llm_workers=1, cpu_workers=1)

    assert stability_summary["scoring_strategy"] == "single_dimension"
    assert full_summary["scoring_strategy"] == "single_dimension"
    assert (root / "06_ratings" / "mda_document_scores.csv").read_text(encoding="utf-8") == before
    assert (root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_document_scores.csv").exists()
    assert (root / "06_ratings" / "mda_v2_full" / "mda_v2_full_document_scores.csv").exists()


def test_mda_v2_scripts_reject_whole_document_strategy(tmp_path):
    root = tmp_path / "06_scoring"
    _copy_contracts(root)
    _write_clean_mda_rows(root, ["MDA_REJECT_2024"])

    try:
        run_mda_v2_stability_batch(root=root, mock=True, scoring_strategy="whole_document")
    except ValueError as exc:
        assert "single-call" in str(exc)
    else:
        raise AssertionError("whole_document strategy should be rejected for MD&A stability scoring")

    try:
        run_mda_v2_full_rescore(root=root, mock=True, scoring_strategy="whole_document", limit=1)
    except ValueError as exc:
        assert "single-call" in str(exc)
    else:
        raise AssertionError("whole_document strategy should be rejected for MD&A full rescore")


def test_legacy_mda_whole_document_real_llm_entrypoints_are_blocked(tmp_path):
    root = tmp_path / "06_scoring"
    _copy_contracts(root)
    _write_clean_mda_rows(root, ["MDA_LEGACY_BLOCK_2024"])

    for runner in [
        lambda: run_pipeline("mda", "full", root=root, mock_mode=False),
        lambda: run_parallel_pipeline("mda", "full", root=root, mock_mode=False),
    ]:
        try:
            runner()
        except ValueError as exc:
            assert "single-call" in str(exc)
        else:
            raise AssertionError("real LLM MD&A whole-document entrypoint should be blocked")
