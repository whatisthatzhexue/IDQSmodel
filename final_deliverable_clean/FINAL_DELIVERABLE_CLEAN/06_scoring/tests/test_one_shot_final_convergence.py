from __future__ import annotations

import json
from pathlib import Path

import run_one_shot_final_convergence as one_shot
from discover_real_news_across_project import discover_real_news_across_project
from rebuild_mda_stability_metrics import rebuild_mda_stability_metrics
from run_one_shot_final_convergence import run_one_shot_final_convergence
from snapshot_pre_final_convergence import snapshot_pre_final_convergence
from scoring_utils import REGISTRY_HEADERS, RATINGS_LONG_HEADERS, save_json, write_csv_rows


def test_snapshot_pre_final_convergence_copies_state_and_hashes(tmp_path):
    root = tmp_path / "06_scoring"
    (root / "PAPER_OUTPUT").mkdir(parents=True)
    (root / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json").write_text('{"paper_ready": false}', encoding="utf-8")
    (root / "06_ratings").mkdir(parents=True)
    (root / "06_ratings" / "mda_document_scores.csv").write_text("document_id\nMDA_A\n", encoding="utf-8")

    summary = snapshot_pre_final_convergence(root=root, timestamp="20260618T000000Z")

    archive = root / "archive" / "pre_final_convergence_20260618T000000Z"
    assert summary["snapshot_dir"] == str(archive)
    assert (archive / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json").exists()
    assert (archive / "snapshot_manifest.json").exists()
    assert "PAPER_OUTPUT/PAPER_READY_STATUS.json" in (archive / "snapshot_file_hashes.csv").read_text(encoding="utf-8")


def test_rebuild_mda_stability_metrics_separates_operational_from_repeatability(tmp_path):
    root = tmp_path / "06_scoring"
    batch_dir = root / "06_ratings" / "mda_v2_stability_batch"
    write_csv_rows(batch_dir / "mda_v2_stability_batch_selection.csv", ["document_id"], [{"document_id": "MDA_A"}, {"document_id": "MDA_B"}])
    write_csv_rows(batch_dir / "mda_v2_stability_document_scores.csv", ["document_id"], [{"document_id": "MDA_A"}])
    rows = []
    for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
        row = {field: "" for field in RATINGS_LONG_HEADERS}
        row.update({"document_id": "MDA_A", "dimension_code": code, "parse_status": "parsed", "schema_validation_status": "valid", "evidence_locator": "[MDA_P001]"})
        rows.append(row)
    write_csv_rows(batch_dir / "mda_v2_stability_ratings_long.csv", RATINGS_LONG_HEADERS, rows)
    (root / "02_extracted_text" / "mda").mkdir(parents=True)
    (root / "02_extracted_text" / "mda" / "MDA_A.txt").write_text("[MDA_P001] text", encoding="utf-8")

    metrics = rebuild_mda_stability_metrics(root=root)

    assert metrics["legacy_mda_stability_success_rate"] == 0.5
    assert metrics["operational_completion_rate"] == 0.5
    assert metrics["repeatability_pass_rate"] is None
    assert metrics["legacy_metric_note"].startswith("legacy_mda_stability_success_rate mixes")
    assert (root / "PAPER_OUTPUT" / "00_status" / "mda_stability_metrics_redefined.json").exists()


def test_rebuild_mda_stability_metrics_prefers_complete_stability_outputs(tmp_path):
    root = tmp_path / "06_scoring"
    batch_dir = root / "06_ratings" / "mda_v2_stability_batch"
    complete_dir = root / "06_ratings" / "mda_context_repair"
    selection = [{"document_id": "MDA_A"}, {"document_id": "MDA_B"}]
    write_csv_rows(batch_dir / "mda_v2_stability_batch_selection.csv", ["document_id"], selection)

    legacy_rows = []
    complete_rows = []
    for doc_id in ["MDA_A", "MDA_B"]:
        for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
            row = {field: "" for field in RATINGS_LONG_HEADERS}
            row.update(
                {
                    "document_id": doc_id,
                    "dimension_code": code,
                    "parse_status": "parsed",
                    "schema_validation_status": "valid",
                    "evidence_locator": "[MDA_P001]",
                }
            )
            complete_rows.append(row)
            if doc_id == "MDA_A":
                legacy_rows.append(row)
    write_csv_rows(batch_dir / "mda_v2_stability_ratings_long.csv", RATINGS_LONG_HEADERS, legacy_rows)
    write_csv_rows(complete_dir / "mda_stability_ratings_long_complete.csv", RATINGS_LONG_HEADERS, complete_rows)

    text_dir = root / "02_extracted_text" / "mda"
    text_dir.mkdir(parents=True)
    (text_dir / "MDA_A.txt").write_text("[MDA_P001] text", encoding="utf-8")
    (text_dir / "MDA_B.txt").write_text("[MDA_P001] text", encoding="utf-8")

    metrics = rebuild_mda_stability_metrics(root=root)

    assert metrics["legacy_mda_stability_success_rate"] == 0.5
    assert metrics["operational_completion_rate"] == 1.0
    assert metrics["operational_metrics_source"] == "mda_stability_ratings_long_complete"


def test_discover_real_news_across_project_standardizes_real_csv(tmp_path):
    project_root = tmp_path / "project"
    root = project_root / "06_scoring"
    source = project_root / "raw_news"
    source.mkdir(parents=True)
    (source / "news_raw.csv").write_text(
        "stock_code,company,date,publisher,headline,link,body\nAAA,Alpha,2024-01-01,Source,Alpha expands,http://example.com,Article body\n",
        encoding="utf-8",
    )

    summary = discover_real_news_across_project(project_root=project_root, root=root)

    assert summary["real_news_candidate_count"] == 1
    out_csv = root / "input_news" / "real_news_raw" / "real_news_raw.csv"
    assert out_csv.exists()
    text = out_csv.read_text(encoding="utf-8")
    assert "synthetic_flag" in text
    assert "false" in text


def test_one_shot_final_convergence_writes_blocker_deliverable_when_ollama_and_news_missing(tmp_path):
    project_root = tmp_path / "project"
    root = project_root / "06_scoring"
    save_json(
        root / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json",
        {
            "paper_ready": False,
            "freeze_allowed": False,
            "mda_stability_success_rate": 0.592593,
            "news_final_dataset_size": 0,
            "synthetic_news_count": 0,
            "claim_link_count": 0,
            "system_level_error_rate": 0.0,
        },
    )
    write_csv_rows(root / "01_registry" / "mda_registry.csv", REGISTRY_HEADERS["mda"], [])
    status = run_one_shot_final_convergence(
        project_root=project_root,
        model="qwen3:8b",
        max_runtime_repair_rounds=0,
        max_scoring_retries=1,
        llm_workers=1,
        cpu_workers=1,
        resume=True,
    )

    deliverable = root / "FINAL_DELIVERABLE"
    assert status["paper_ready"] is False
    assert status["freeze_allowed"] is False
    assert "real_news_missing" in status["blocking_reasons"]
    assert "mda_not_ready" in status["blocking_reasons"]
    assert (deliverable / "FINAL_STATUS.json").exists()
    assert (deliverable / "FINAL_REPORT.md").exists()
    assert json.loads((deliverable / "FINAL_STATUS.json").read_text(encoding="utf-8"))["paper_ready"] is False


def test_one_shot_refreshes_mda_metrics_after_full_scoring(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    root = project_root / "06_scoring"
    root.mkdir(parents=True)
    calls = {"metrics": 0}

    def fake_metrics(_root):
        calls["metrics"] += 1
        full_coverage = 1.0 if calls["metrics"] >= 3 else 0.0
        return {
            "expected_stability_documents": 27,
            "operational_completion_rate": 1.0,
            "repeatability_pass_rate": 1.0,
            "schema_success_rate": 1.0,
            "evidence_match_rate": 1.0,
            "unresolved_critical_numeric_conflicts": 0,
            "full_scoring_coverage_rate": full_coverage,
            "eligible_mda_count": 149,
            "complete_mda_count": 149 if full_coverage else 0,
        }

    monkeypatch.setattr(one_shot, "snapshot_pre_final_convergence", lambda root: {"status": "ok"})
    monkeypatch.setattr(one_shot, "rebuild_mda_stability_metrics", fake_metrics)
    monkeypatch.setattr(one_shot, "repair_ollama_runtime", lambda **_: {"runtime_healthy": True, "blocking_reasons": []})
    monkeypatch.setattr(one_shot, "recover_missing_mda_dimensions", lambda **_: {"status": "success", "missing_dimensions_still_failed": 0})
    monkeypatch.setattr(one_shot, "run_mda_final_repeatability_test", lambda **_: {"status": "success"})
    monkeypatch.setattr(one_shot, "run_mda_final_full_scoring", lambda **_: {"status": "success", "complete_mda_count": 149, "full_scoring_coverage_rate": 1.0})
    monkeypatch.setattr(one_shot, "discover_real_news_across_project", lambda **_: {"real_news_available": False, "real_news_candidate_count": 0})
    monkeypatch.setattr(one_shot, "_run_news_if_possible", lambda *_, **__: {"status": "blocked", "reason": "real_news_missing", "news_final_dataset_size": 0})
    monkeypatch.setattr(one_shot, "_run_cross_validation_if_possible", lambda *_, **__: {"cross_validation_pipeline_ready": True, "cross_validation_empirical_ready": False, "claim_link_count": 0})
    monkeypatch.setattr(one_shot, "run_final_data_freeze", lambda _root: {"system_level_error_rate": 0})
    monkeypatch.setattr(
        one_shot,
        "run_paper_readiness_check",
        lambda _root: {
            "mda_final_dataset_size": 149,
            "news_final_dataset_size": 0,
            "synthetic_news_count": 0,
            "reproducibility_ready": True,
            "blocking_reasons": ["news_not_ready", "cross_validation_not_ready"],
        },
    )
    monkeypatch.setattr(one_shot, "build_blocker_status_reports", lambda **_: None)
    monkeypatch.setattr(one_shot, "_build_deliverable", lambda *_: None)

    status = one_shot.run_one_shot_final_convergence(project_root=project_root, llm_workers=1, cpu_workers=1)

    assert calls["metrics"] == 3
    assert status["mda_ready"] is True
    assert status["mda_final_dataset_size"] == 149
    assert status["mda_full_scoring_coverage_rate"] == 1.0
    assert "mda_not_ready" not in status["blocking_reasons"]
