from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_utils import ensure_paper_output_dirs, load_json_safe, load_paper_status
from scoring_utils import SCORING_ROOT, read_csv_rows, save_json
from stability_common import write_markdown


DELIVERABLE_DIR_NAME = "PAPER_OUTPUT_FINAL_DELIVERABLE"


def run_overnight_final_delivery(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    commands = [
        ("paper_readiness_initial", [sys.executable, str(root / "scripts" / "run_paper_readiness_check.py")]),
        ("paper_convergence_loop", [sys.executable, str(root / "scripts" / "run_paper_convergence_loop.py"), "--max-rounds", "5"]),
        ("final_data_freeze", [sys.executable, str(root / "scripts" / "run_final_data_freeze.py")]),
        ("statistical_stability_analysis", [sys.executable, str(root / "scripts" / "run_statistical_stability_analysis.py")]),
        ("validate_stability_outputs", [sys.executable, str(root / "scripts" / "validate_stability_outputs.py")]),
        ("paper_ready_pipeline", [sys.executable, str(root / "scripts" / "run_paper_ready_pipeline.py"), "--max-convergence-rounds", "5"]),
    ]
    command_results = []
    command_log_dir = root / "PAPER_OUTPUT" / "09_reproducibility" / "overnight_command_logs"
    command_log_dir.mkdir(parents=True, exist_ok=True)
    for name, command in commands:
        completed = subprocess.run(command, cwd=root.parent, text=True, capture_output=True)
        log_path = command_log_dir / f"{name}.log"
        log_path.write_text("$ " + " ".join(command) + "\n\nSTDOUT:\n" + completed.stdout + "\nSTDERR:\n" + completed.stderr, encoding="utf-8")
        command_results.append(
            {
                "name": name,
                "command": " ".join(command),
                "returncode": completed.returncode,
                "log_path": str(log_path),
                "continued_after_failure": completed.returncode != 0,
            }
        )

    deliverable = build_final_deliverable(root, command_results)
    return deliverable


def build_final_deliverable(root: Path, command_results: list[dict[str, Any]]) -> dict[str, Any]:
    out = root / DELIVERABLE_DIR_NAME
    if out.exists():
        shutil.rmtree(out)
    for subdir in [
        "FINAL_DATASET",
        "STABILITY_REPORT",
        "CROSS_VALIDATION_REPORT",
        "QUALITY_CONTROL_REPORT",
        "REPRODUCIBILITY_PACKAGE",
        "TABLES",
        "FIGURES",
    ]:
        (out / subdir).mkdir(parents=True, exist_ok=True)
    status = load_paper_status(root)
    metadata, _ = load_json_safe(root / "FINAL_OUTPUT" / "dataset_frozen_metadata.json")
    stability_validation, _ = load_json_safe(root / "11_stability_analysis" / "logs" / "validate_stability_outputs.json")
    _copy_known_files(root, out)
    final_report_path = out / "FINAL_REPORT.md"
    write_markdown(final_report_path, _final_report_lines(status, metadata, stability_validation, command_results))
    save_json(out / "PAPER_READY_STATUS.json", status)
    save_json(out / "REPRODUCIBILITY_PACKAGE" / "overnight_command_results.json", command_results)
    manifest = _write_manifest(out)
    summary = {
        "deliverable_dir": str(out),
        "final_report": str(final_report_path),
        "paper_ready": status["paper_ready"],
        "freeze_allowed": status["freeze_allowed"],
        "mda_ready": status["mda_ready"],
        "news_ready": status["news_ready"],
        "cross_validation_ready": status["cross_validation_ready"],
        "reproducibility_ready": status["reproducibility_ready"],
        "blocking_reasons": status["blocking_reasons"],
        "advisory_warnings": status["advisory_warnings"],
        "stability_validation_passed": stability_validation.get("passed"),
        "command_failures": [item for item in command_results if item["returncode"] != 0],
        "manifest": str(manifest),
    }
    save_json(out / "DELIVERABLE_SUMMARY.json", summary)
    return summary


def _copy_known_files(root: Path, out: Path) -> None:
    copy_map = [
        ("FINAL_OUTPUT/mda_final_dataset.csv", "FINAL_DATASET/mda_final_dataset.csv"),
        ("FINAL_OUTPUT/news_final_dataset.csv", "FINAL_DATASET/news_final_dataset.csv"),
        ("FINAL_OUTPUT/cross_validation_final_summary.csv", "FINAL_DATASET/cross_validation_final_summary.csv"),
        ("FINAL_OUTPUT/dataset_frozen_metadata.json", "FINAL_DATASET/dataset_frozen_metadata.json"),
        ("FINAL_OUTPUT/final_quality_report.md", "QUALITY_CONTROL_REPORT/final_quality_report.md"),
        ("PAPER_OUTPUT/PAPER_READY_REPORT.md", "QUALITY_CONTROL_REPORT/PAPER_READY_REPORT.md"),
        ("PAPER_OUTPUT/00_status/paper_readiness_report.md", "QUALITY_CONTROL_REPORT/paper_readiness_report.md"),
        ("PAPER_OUTPUT/00_status/paper_convergence_log.md", "QUALITY_CONTROL_REPORT/paper_convergence_log.md"),
        ("08_reports/mda_stability_failure_root_cause.md", "QUALITY_CONTROL_REPORT/mda_stability_failure_root_cause.md"),
        ("08_reports/news_data_blocking_report.md", "QUALITY_CONTROL_REPORT/news_data_blocking_report.md"),
        ("08_reports/news_pipeline_status_report.md", "QUALITY_CONTROL_REPORT/news_pipeline_status_report.md"),
        ("11_stability_analysis/reports/statistical_stability_report.md", "STABILITY_REPORT/statistical_stability_report.md"),
        ("11_stability_analysis/reports/validate_stability_outputs.md", "STABILITY_REPORT/validate_stability_outputs.md"),
        ("11_stability_analysis/outputs/statistical_stability_summary.json", "STABILITY_REPORT/statistical_stability_summary.json"),
        ("11_stability_analysis/logs/validate_stability_outputs.json", "STABILITY_REPORT/validate_stability_outputs.json"),
        ("09_cross_validation/cross_validation_report.md", "CROSS_VALIDATION_REPORT/cross_validation_report.md"),
        ("09_cross_validation/fix_report.md", "CROSS_VALIDATION_REPORT/fix_report.md"),
        ("09_cross_validation/claim_links.csv", "CROSS_VALIDATION_REPORT/claim_links.csv"),
        ("09_cross_validation/mda_news_cross_validation_pairs.csv", "CROSS_VALIDATION_REPORT/mda_news_cross_validation_pairs.csv"),
        ("PAPER_OUTPUT/09_reproducibility/reproducibility_guide.md", "REPRODUCIBILITY_PACKAGE/reproducibility_guide.md"),
        ("PAPER_OUTPUT/09_reproducibility/command_log.md", "REPRODUCIBILITY_PACKAGE/command_log.md"),
        ("PAPER_OUTPUT/09_reproducibility/environment_summary.md", "REPRODUCIBILITY_PACKAGE/environment_summary.md"),
        ("PAPER_OUTPUT/09_reproducibility/file_manifest.csv", "REPRODUCIBILITY_PACKAGE/file_manifest.csv"),
    ]
    for source_rel, dest_rel in copy_map:
        _copy_file(root / source_rel, out / dest_rel)
    _copy_dir(root / "PAPER_OUTPUT" / "tables", out / "TABLES")
    _copy_dir(root / "PAPER_OUTPUT" / "figures", out / "FIGURES")
    _copy_dir(root / "PAPER_OUTPUT" / "09_reproducibility" / "overnight_command_logs", out / "REPRODUCIBILITY_PACKAGE" / "command_logs")


def _copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copy2(source, dest)
    else:
        dest.write_text(f"Missing source file: {source}\n", encoding="utf-8")


def _copy_dir(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        (dest / "MISSING_SOURCE.txt").write_text(f"Missing source directory: {source}\n", encoding="utf-8")
        return
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, dest / path.name)


def _final_report_lines(
    status: dict[str, Any],
    metadata: dict[str, Any],
    stability_validation: dict[str, Any],
    command_results: list[dict[str, Any]],
) -> list[str]:
    final_size = metadata.get("final_dataset_size", {})
    gates = metadata.get("quality_gates", {})
    command_failures = [item for item in command_results if item["returncode"] != 0]
    stability_met = bool(status.get("mda_stability_success_rate") is not None and status["mda_stability_success_rate"] >= 0.85)
    system_error_rate = status.get("system_level_error_rate")
    system_error_ok = bool(system_error_rate is not None and float(system_error_rate) < 0.05)
    news_true_available = bool(status.get("news_ready"))
    cross_trusted = bool(status.get("cross_validation_ready") and status.get("news_ready"))
    can_write = ["methodology", "scoring framework", "pipeline validation", "limitations", "reproducibility"]
    if status.get("paper_ready"):
        can_write.append("final empirical results")
    cannot_write = []
    if not status.get("paper_ready"):
        cannot_write.append("final empirical results")
    if not status.get("news_ready"):
        cannot_write.extend(["real News empirical results", "substantive cross-validation conclusion"])
    lines = [
        "# Overnight Final Delivery Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "## Executive Status",
        f"- paper_ready: {str(status['paper_ready']).lower()}",
        f"- freeze_allowed: {str(status['freeze_allowed']).lower()}",
        f"- mda_ready: {str(status['mda_ready']).lower()}",
        f"- news_ready: {str(status['news_ready']).lower()}",
        f"- cross_validation_ready: {str(status['cross_validation_ready']).lower()}",
        f"- reproducibility_ready: {str(status['reproducibility_ready']).lower()}",
        "",
        "## Required Answers",
        f"1. 当前系统是否 paper-ready: {str(status['paper_ready']).lower()}",
        f"2. freeze_allowed 是否 true: {str(status['freeze_allowed']).lower()}",
        f"3. MD&A 是否可以用于实证分析: {'yes' if status['mda_ready'] else 'no; MD&A stability gate is not satisfied'}",
        f"4. News 是否真实可用: {'yes' if news_true_available else 'no; real News final scores are unavailable'}",
        f"5. cross-validation 是否可信: {'yes' if cross_trusted else 'no for substantive interpretation; it remains auxiliary/technical only'}",
        f"6. 哪些结果可以写论文: {', '.join(can_write)}",
        f"7. 哪些不能写: {', '.join(cannot_write) if cannot_write else 'none'}",
        f"8. blocking reasons: {', '.join(status['blocking_reasons']) if status['blocking_reasons'] else 'none'}",
        f"9. system-level error rate: {status['system_level_error_rate']}",
        f"10. stability 是否达标: {'yes' if stability_met else 'no'}; mda_stability_success_rate={status['mda_stability_success_rate']}",
        f"11. 是否可以提交论文: {'yes' if status['paper_ready'] else 'no; methodology/pipeline package can be reviewed but final empirical conclusions are blocked'}",
        "",
        "## Dataset Status",
        f"- MD&A final dataset size: {final_size.get('mda', status['mda_final_dataset_size'])}",
        f"- News final dataset size: {final_size.get('news', status['news_final_dataset_size'])}",
        f"- synthetic_news_count: {status['synthetic_news_count']}",
        f"- claim_link_count: {status['claim_link_count']}",
        "",
        "## Freeze Logic Check",
        f"- mda_stability >= 0.85: {str(stability_met).lower()}",
        f"- news_exists/news_ready: {str(status['news_ready']).lower()}",
        f"- system_error_rate < 0.05: {str(system_error_ok).lower()}",
        f"- metadata freeze_allowed: {str(metadata.get('freeze_allowed', False)).lower()}",
        "",
        "## Quality Gates From Metadata",
    ]
    for name, gate in sorted(gates.items()):
        lines.append(f"- {name}: passed={str(gate.get('passed')).lower()}, value={gate.get('value')}, threshold={gate.get('threshold')}")
    lines.extend(
        [
            "",
            "## Stability Validation",
            f"- validate_stability_outputs passed: {stability_validation.get('passed')}",
            f"- stability validation issues: {len(stability_validation.get('issues', []))}",
        ]
    )
    for issue in stability_validation.get("issues", []):
        lines.append(f"- {issue}")
    lines.extend(["", "## Overnight Command Results"])
    for item in command_results:
        lines.append(f"- {item['name']}: returncode={item['returncode']}; log={item['log_path']}")
    if command_failures:
        lines.extend(["", "## Non-silent Failures"])
        for item in command_failures:
            lines.append(f"- {item['name']} failed with returncode={item['returncode']}; pipeline continued and preserved log.")
    lines.extend(
        [
            "",
            "## Final Conclusion",
            "The package is complete as an overnight deliverable. It is not a final empirical paper-ready dataset unless all gates above pass.",
        ]
    )
    return lines


def _write_manifest(out: Path) -> Path:
    rows = []
    for path in sorted(out.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(out)), "size_bytes": path.stat().st_size})
    manifest_path = out / "DELIVERABLE_MANIFEST.csv"
    from scoring_utils import write_csv_rows

    write_csv_rows(manifest_path, ["relative_path", "size_bytes"], rows)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run overnight final delivery mode and build final deliverable package.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(run_overnight_final_delivery(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
