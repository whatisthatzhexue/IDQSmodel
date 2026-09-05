import json
from pathlib import Path

from recover_missing_mda_dimensions import recover_missing_mda_dimensions
from llm_clients import OllamaCallResult, OllamaClient
from run_mda_final_full_scoring import run_mda_final_full_scoring
from run_mda_dimensionwise_scoring import DimensionTask, score_one_dimension
from run_context_budget_smoke_test import run_context_budget_smoke_test, summarize_smoke_records
from run_single_dimension_scoring import SingleDimensionTask, _dimension_context_budget, _single_dimension_prompt_parts
from run_single_dimension_scoring import score_one_dimension as score_one_single_dimension
from scoring_utils import write_csv_rows
from select_mda_dimension_context import select_document_dimension_contexts
from validate_prompt_context_budget import PromptBudgetConfig, validate_prompt_context_budget


def _dimension(code: str = "AR02") -> dict:
    return {
        "code": code,
        "name": "MD&A internal numeric consistency and traceability",
        "weight": 0.25,
        "definition": "Numeric consistency and traceability.",
    }


def _sample_text() -> str:
    return (
        "[SECTION: Overview]\n\n"
        "[MDA_P001]\nManagement described the year in broad strategic terms.\n\n"
        "[MDA_P002]\nRevenue increased by 20% from RM100 million to RM120 million while profit margin improved.\n\n"
        "[MDA_P003]\nThe company donated food to communities and described staff events.\n\n"
        "[MDA_P004]\nOutlook remains cautious because raw material cost, supply chain risk, and FX uncertainty may persist.\n\n"
        "[MDA_P005]\nSales volume in the export segment grew due to demand from North America and price improvements.\n\n"
        "[MDA_P006]\nThe company sponsored a local festival and thanked long-serving staff.\n\n"
        "[MDA_P007]\nEmployees joined a community event during the year.\n\n"
        "[MDA_P008]\nThe chairman thanked the audience for attending the annual meeting.\n"
    )


def test_dimension_context_selector_writes_targeted_context_and_manifest(tmp_path):
    root = tmp_path / "06_scoring"
    text_path = root / "02_extracted_text" / "mda_clean_v2" / "MDA_TEST.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text(_sample_text(), encoding="utf-8")
    numeric_path = root / "03_numeric_checks" / "mda_v2" / "MDA_TEST_numeric_checks.json"
    numeric_path.parent.mkdir(parents=True)
    numeric_path.write_text(
        json.dumps(
            {
                "document_id": "MDA_TEST",
                "num_material_discrepancies": 1,
                "checks": [
                    {
                        "status": "material_discrepancy",
                        "evidence_locator": "[MDA_P002]",
                        "source_text": "Revenue increased by 50% from RM100 million to RM120 million.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    contexts = select_document_dimension_contexts(
        root=root,
        document_id="MDA_TEST",
        text_path=text_path,
        numeric_check_path=numeric_path,
        context_budget=700,
    )

    ar02 = contexts["AR02"]
    ar03 = contexts["AR03"]
    assert "MDA_P002" in ar02.selected_paragraph_ids
    assert "MDA_P008" not in ar02.selected_paragraph_ids
    assert "material_discrepancy" in ar02.context_text
    assert "MDA_P005" in ar03.selected_paragraph_ids
    assert (root / "02_extracted_text" / "mda_dimension_context" / "MDA_TEST_AR02.txt").exists()

    rows = (root / "02_extracted_text" / "mda_dimension_context" / "context_selection_manifest.csv").read_text(
        encoding="utf-8"
    )
    assert "document_id,dimension_code,source_paragraph_count,selected_paragraph_count" in rows
    assert "MDA_TEST,AR02" in rows
    assert ",True," in rows


def test_prompt_budget_rejects_full_text_before_model_call():
    config = PromptBudgetConfig(num_ctx=4096, reserved_output_tokens=384, safety_margin_tokens=256)
    instructions = "Score one dimension and return JSON only. " * 40
    schema = json.dumps({"type": "object", "required": ["score", "evidence", "reason"]})
    long_text = ("[MDA_P001] revenue sales profit cost margin cash flow segment RM % dividend. " * 900)
    short_text = "[MDA_P001] Revenue increased 20% from RM100 million to RM120 million."

    failed = validate_prompt_context_budget(
        instructions_text=instructions,
        schema_text=schema,
        selected_text=long_text,
        config=config,
    )
    passed = validate_prompt_context_budget(
        instructions_text=instructions,
        schema_text=schema,
        selected_text=short_text,
        config=config,
    )

    assert failed.budget_passed is False
    assert failed.total_expected_tokens > failed.input_token_budget
    assert passed.budget_passed is True
    assert passed.total_expected_tokens < passed.input_token_budget


def test_dimension_context_budget_reserves_prompt_overhead_for_ar02():
    context_budget = _dimension_context_budget(4096)
    overhead_tokens = 218 + 817 + 135

    assert context_budget + overhead_tokens <= 3456
    assert context_budget <= 1700


def test_single_dimension_prompt_uses_compact_metadata_not_full_registry():
    metadata = {
        "document_id": "MDA_META",
        "doc_type": "mda",
        "stock_code": "1234",
        "ticker": "META",
        "company_name": "Meta Food Berhad",
        "report_year": "2024",
        "source_path": "/very/long/path/" + ("x" * 2000),
        "notes": "long notes " * 500,
    }

    _prompt, _instructions, metadata_text = _single_dimension_prompt_parts(
        "mda",
        "MDA_META",
        metadata,
        _dimension("AR01"),
        "[MDA_P001]\nRevenue increased.",
        None,
    )

    assert "source_path" not in metadata_text
    assert "notes" not in metadata_text
    assert "company_name" in metadata_text


def test_single_dimension_prompt_uses_compact_numeric_summary_not_full_checks():
    numeric = {
        "document_id": "MDA_NUM",
        "overall_numeric_consistency": "problematic",
        "num_material_discrepancies": 1,
        "checks": [
            {
                "status": "material_discrepancy",
                "source_text": "Revenue increased by 50% from RM100 million to RM120 million.",
                "evidence_locator": "[MDA_P001]",
            }
        ],
    }

    _prompt, _instructions, metadata_text = _single_dimension_prompt_parts(
        "mda",
        "MDA_NUM",
        {"document_id": "MDA_NUM", "doc_type": "mda"},
        _dimension("AR02"),
        "[MDA_P001]\nRevenue increased.",
        numeric,
    )

    assert "num_material_discrepancies" in metadata_text
    assert "source_text" not in metadata_text
    assert "Revenue increased by 50%" not in metadata_text


def test_ollama_score_document_uses_chat_and_extracts_inner_json(monkeypatch):
    calls = []

    def fake_request(self, method, path, payload=None):
        calls.append((method, path, payload))
        return {
            "message": {"content": "{\"score\": \"4\", \"evidence\": \"MDA_P001\", \"reason\": \"ok\"}"},
            "prompt_eval_count": 100,
            "eval_count": 12,
        }

    monkeypatch.setattr(OllamaClient, "_request_json", fake_request)
    client = OllamaClient(model="qwen3:8b", num_ctx=4096, num_predict=192)
    result = client.score_document("prompt", {"type": "object"})

    assert calls[0][1] == "/api/chat"
    assert calls[0][2]["stream"] is False
    assert calls[0][2]["think"] is False
    assert calls[0][2]["options"]["num_ctx"] == 4096
    assert calls[0][2]["options"]["num_predict"] == 192
    assert result.parsed_json == {"score": "4", "evidence": "MDA_P001", "reason": "ok"}


def test_score_one_dimension_rejects_prompt_eval_near_context_limit(tmp_path, monkeypatch):
    root = tmp_path / "06_scoring"
    text_path = root / "02_extracted_text" / "mda_clean_v2" / "MDA_TEST.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.", encoding="utf-8")
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        ["document_id", "text_path"],
        [{"document_id": "MDA_TEST", "text_path": str(text_path)}],
    )

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def score_document(self, prompt, schema):
            return OllamaCallResult(
                ok=True,
                raw_response={
                    "message": {"content": "{\"score\": 3, \"evidence\": \"MDA_P001\", \"reason\": \"ok\"}"},
                    "prompt_eval_count": 4095,
                    "eval_count": 20,
                },
                content="{\"score\": 3, \"evidence\": \"MDA_P001\", \"reason\": \"ok\"}",
                parsed_json={"score": 3, "evidence": "MDA_P001", "reason": "ok"},
                fallback_used=False,
            )

        def repair_json(self, raw_text, schema):
            raise AssertionError("repair should not be called for a budget-invalid response")

    monkeypatch.setattr("run_mda_dimensionwise_scoring.OllamaClient", FakeClient)
    task = DimensionTask(
        document_id="MDA_TEST",
        dimension_code="AR02",
        dimension_name="MD&A internal numeric consistency and traceability",
        dimension_definition=_dimension("AR02"),
        metadata={"document_id": "MDA_TEST"},
        text_path=str(text_path),
        numeric_check_path="",
        output_name="mda_context_repair_test",
        root=str(root),
        model_name="qwen3:8b",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
        timeout_seconds=10,
        max_retries=1,
        mock=False,
    )

    result = score_one_dimension(task)

    assert result.status == "failed"
    assert result.error_type == "prompt_eval_too_high"
    parsed = json.loads((root / "06_ratings" / "mda_context_repair_test" / "per_dimension" / "MDA_TEST_AR02_parsed.json").read_text())
    assert parsed["status"] == "failed"
    assert parsed["result"]["error_type"] == "prompt_eval_too_high"


def test_single_dimension_mda_entry_rejects_prompt_eval_near_context_limit(tmp_path, monkeypatch):
    root = tmp_path / "06_scoring"
    text_path = root / "02_extracted_text" / "mda_clean_v2" / "MDA_TEST.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.", encoding="utf-8")

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def score_document(self, prompt, schema):
            return OllamaCallResult(
                ok=True,
                raw_response={
                    "message": {"content": "{\"score\": 3, \"evidence\": \"MDA_P001\", \"reason\": \"ok\"}"},
                    "prompt_eval_count": 4095,
                    "eval_count": 20,
                },
                content="{\"score\": 3, \"evidence\": \"MDA_P001\", \"reason\": \"ok\"}",
                parsed_json={"score": 3, "evidence": "MDA_P001", "reason": "ok"},
                fallback_used=False,
            )

    monkeypatch.setattr("run_single_dimension_scoring.OllamaClient", FakeClient)
    task = SingleDimensionTask(
        document_id="MDA_TEST",
        doc_type="mda",
        dimension_code="AR02",
        dimension_name="MD&A internal numeric consistency and traceability",
        dimension_definition=_dimension("AR02"),
        metadata={"document_id": "MDA_TEST"},
        text_path=str(text_path),
        numeric_check_path="",
        output_name="mda_context_repair_test",
        root=str(root),
        model_name="qwen3:8b",
        base_url="http://127.0.0.1:11434",
        temperature=0,
        seed=42,
        num_ctx=4096,
        timeout_seconds=10,
        max_retries=1,
        mock=False,
    )

    result = score_one_single_dimension(task)

    assert result.status == "failed"
    assert result.error_type == "prompt_eval_too_high"


def test_smoke_summary_blocks_recovery_when_any_contract_check_fails():
    records = [
        {
            "http_success": True,
            "inner_json_parse": True,
            "schema_success": True,
            "score_range_success": True,
            "evidence_match": True,
            "context_budget_pass": True,
        },
        {
            "http_success": True,
            "inner_json_parse": True,
            "schema_success": False,
            "score_range_success": True,
            "evidence_match": True,
            "context_budget_pass": True,
        },
    ]

    summary = summarize_smoke_records(records)

    assert summary["overall_status"] == "blocked"
    assert summary["allow_recover_55_dimensions"] is False
    assert summary["schema_success"] == "1/2"


def test_recover_missing_dimensions_requires_passed_context_budget_smoke(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "PAPER_OUTPUT" / "00_status" / "mda_stability_blocker_cases.csv",
        ["document_id", "failure_reason", "action_status"],
        [{"document_id": "MDA_FAIL", "failure_reason": "missing_dimension", "action_status": ""}],
    )

    result = recover_missing_mda_dimensions(
        root=root,
        runtime_status={"runtime_healthy": True},
        max_scoring_retries=1,
        llm_workers=1,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "context_budget_smoke_test_not_passed"


def test_final_full_scoring_requires_passed_context_budget_smoke(tmp_path):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        ["document_id", "include_flag", "text_path"],
        [{"document_id": "MDA_FULL", "include_flag": "Yes", "text_path": "02_extracted_text/mda_clean_v2/MDA_FULL.txt"}],
    )
    text_path = root / "02_extracted_text" / "mda_clean_v2" / "MDA_FULL.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("[MDA_P001]\nRevenue increased.", encoding="utf-8")

    result = run_mda_final_full_scoring(
        root=root,
        runtime_status={"runtime_healthy": True},
        metrics={
            "operational_completion_rate": 1.0,
            "repeatability_pass_rate": 1.0,
            "schema_success_rate": 1.0,
            "evidence_match_rate": 1.0,
        },
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "context_budget_smoke_test_not_passed"


def test_mock_smoke_test_does_not_unlock_recovery_or_full_scoring(tmp_path):
    root = tmp_path / "06_scoring"
    text_path = root / "02_extracted_text" / "mda_clean_v2" / "MDA_SMOKE.txt"
    text_path.parent.mkdir(parents=True)
    text_path.write_text("[MDA_P001]\nRevenue increased by 20% from RM100 million to RM120 million.", encoding="utf-8")
    write_csv_rows(
        root / "01_registry" / "mda_registry.csv",
        ["document_id", "include_flag", "text_path"],
        [{"document_id": "MDA_SMOKE", "include_flag": "Yes", "text_path": str(text_path)}],
    )

    summary = run_context_budget_smoke_test(root=root, repeat_count=1, matrix_doc_count=1, mock=True)

    assert summary["overall_status"] == "mock_passed"
    assert summary["allow_recover_55_dimensions"] is False
    assert summary["allow_full_scoring"] is False


def test_recover_missing_dimensions_writes_context_repair_outputs(tmp_path, monkeypatch):
    root = tmp_path / "06_scoring"
    write_csv_rows(
        root / "PAPER_OUTPUT" / "00_status" / "mda_stability_blocker_cases.csv",
        ["document_id", "failure_reason", "action_status"],
        [{"document_id": "MDA_FAIL", "failure_reason": "missing_dimension", "action_status": ""}],
    )
    (root / "08_reports").mkdir(parents=True)
    (root / "08_reports" / "context_budget_smoke_test_summary.json").write_text(
        json.dumps({"overall_status": "passed", "allow_recover_55_dimensions": True}),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_single_dimension_scoring(doc_type, **kwargs):
        captured.update(kwargs)
        return {
            "status": "completed",
            "dimension_success_count": 1,
            "dimension_task_count": 1,
            "success_count": 1,
            "json_validation_summary": {"evidence_match_rate": 1.0},
        }

    monkeypatch.setattr("recover_missing_mda_dimensions.run_single_dimension_scoring", fake_run_single_dimension_scoring)

    result = recover_missing_mda_dimensions(
        root=root,
        runtime_status={"runtime_healthy": True},
        max_scoring_retries=1,
        llm_workers=1,
    )

    assert captured["output_name"] == "mda_context_repair"
    assert captured["ratings_prefix"] == "mda_context_repair"
    assert result["status"] == "completed"
