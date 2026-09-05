from __future__ import annotations

from build_paper_reproducibility_package import build_paper_reproducibility_package
from scoring_utils import read_csv_rows


def test_reproducibility_package_writes_setup_script_manifest_and_empty_csv_audit(tmp_path):
    root = tmp_path / "06_scoring"

    summary = build_paper_reproducibility_package(root=root)

    requirements = (root.parent / "requirements.txt").read_text(encoding="utf-8")
    assert "pytest" in requirements
    assert "matplotlib" in requirements
    assert "openpyxl" in requirements

    run_script = (root.parent / "run_reproduce.sh").read_text(encoding="utf-8")
    assert "SKIP_VENV" in run_script
    assert "python3 -m venv" in run_script
    assert "audit_filter_news_relevance.py --mode manual_preserve" in run_script
    assert "build_news_registry_from_filter_news.py" in run_script
    assert "run_final_data_freeze.py" in run_script

    repro_dir = root / "PAPER_OUTPUT" / "09_reproducibility"
    assert (repro_dir / "environment_summary.json").exists()
    assert (repro_dir / "command_log.md").exists()
    assert (repro_dir / "empty_csv_audit.csv").exists()
    assert (repro_dir / "empty_csv_explanation.md").exists()
    assert (root.parent / "FINAL_PACKAGE_MANIFEST.md").exists()
    assert summary["empty_csv_audit_path"].endswith("empty_csv_audit.csv")
    assert read_csv_rows(repro_dir / "empty_csv_audit.csv") == []
    assert "Allowed empty CSV categories" in (repro_dir / "empty_csv_explanation.md").read_text(encoding="utf-8")
