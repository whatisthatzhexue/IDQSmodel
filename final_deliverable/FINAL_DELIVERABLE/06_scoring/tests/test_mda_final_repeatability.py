import run_mda_final_repeatability_test as repeatability
from scoring_utils import write_csv_rows


def test_final_repeatability_reuses_complete_existing_pairs(tmp_path, monkeypatch):
    root = tmp_path / "06_scoring"
    selected = [{"document_id": "MDA_A"}, {"document_id": "MDA_B"}]
    write_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv", ["document_id"], selected)
    headers = ["document_id", "ticker", "company_name", "report_year", "AR01", "AR02", "AR03", "AR04", "AR05", "total_score_100"]
    rows = [
        {"document_id": "MDA_A", "ticker": "A", "company_name": "A Food", "report_year": "2023", "AR01": "3", "AR02": "3", "AR03": "3", "AR04": "3", "AR05": "3", "total_score_100": "60"},
        {"document_id": "MDA_B", "ticker": "B", "company_name": "B Food", "report_year": "2023", "AR01": "4", "AR02": "4", "AR03": "4", "AR04": "4", "AR05": "4", "total_score_100": "80"},
    ]
    write_csv_rows(root / "06_ratings" / "mda_repeatability_final" / "run_a" / "mda_repeatability_run_a_document_scores.csv", headers, rows)
    write_csv_rows(root / "06_ratings" / "mda_repeatability_final" / "run_b" / "mda_repeatability_run_b_document_scores.csv", headers, rows)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("existing complete repeatability outputs should be reused")

    monkeypatch.setattr(repeatability, "run_single_dimension_scoring", fail_if_called)

    summary = repeatability.run_mda_final_repeatability_test(root=root, runtime_status={"runtime_healthy": True})

    assert summary["reused_existing_outputs"] is True
    assert summary["valid_paired_documents"] == 2
    assert summary["repeatability_pass_rate"] == 1.0
    assert (root / "PAPER_OUTPUT" / "00_status" / "mda_final_repeatability_report.md").exists()


def test_final_repeatability_runs_model_when_existing_pairs_incomplete(tmp_path, monkeypatch):
    root = tmp_path / "06_scoring"
    selected = [{"document_id": "MDA_A"}, {"document_id": "MDA_B"}]
    write_csv_rows(root / "06_ratings" / "mda_v2_stability_batch" / "mda_v2_stability_batch_selection.csv", ["document_id"], selected)
    headers = ["document_id", "AR01", "AR02", "AR03", "AR04", "AR05", "total_score_100"]
    write_csv_rows(root / "06_ratings" / "mda_repeatability_final" / "run_a" / "mda_repeatability_run_a_document_scores.csv", headers, [{"document_id": "MDA_A", "AR01": "3", "AR02": "3", "AR03": "3", "AR04": "3", "AR05": "3", "total_score_100": "60"}])
    write_csv_rows(root / "06_ratings" / "mda_repeatability_final" / "run_b" / "mda_repeatability_run_b_document_scores.csv", headers, [{"document_id": "MDA_A", "AR01": "3", "AR02": "3", "AR03": "3", "AR04": "3", "AR05": "3", "total_score_100": "60"}])
    calls = []

    def fake_scoring(*_args, **kwargs):
        calls.append(kwargs["ratings_prefix"])
        rows = [
            {"document_id": "MDA_A", "AR01": "3", "AR02": "3", "AR03": "3", "AR04": "3", "AR05": "3", "total_score_100": "60"},
            {"document_id": "MDA_B", "AR01": "4", "AR02": "4", "AR03": "4", "AR04": "4", "AR05": "4", "total_score_100": "80"},
        ]
        out_name = kwargs["output_name"]
        prefix = kwargs["ratings_prefix"]
        write_csv_rows(root / "06_ratings" / out_name / f"{prefix}_document_scores.csv", headers, rows)
        return {"success_count": 2}

    monkeypatch.setattr(repeatability, "run_single_dimension_scoring", fake_scoring)

    summary = repeatability.run_mda_final_repeatability_test(root=root, runtime_status={"runtime_healthy": True})

    assert calls == ["mda_repeatability_run_a", "mda_repeatability_run_b"]
    assert "reused_existing_outputs" not in summary
    assert summary["valid_paired_documents"] == 2
