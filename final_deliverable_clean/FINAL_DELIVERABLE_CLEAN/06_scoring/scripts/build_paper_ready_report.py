from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import PAPER_STATUS_FIELDS, ensure_paper_output_dirs, load_paper_status, paper_ready_allowed_text, write_status
from scoring_utils import SCORING_ROOT, save_json
from stability_common import write_markdown


def build_paper_ready_report(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    status = write_status(root, load_paper_status(root))
    complete = []
    blocked = []
    if status["mda_final_dataset_size"] > 0:
        complete.append("MD&A final dataset exists")
    if status["reproducibility_ready"]:
        complete.append("Reproducibility package is available")
    if status["claim_link_count"] > 0:
        complete.append("Cross-validation produced at least one claim link")
    if not status["mda_ready"]:
        blocked.append("MD&A readiness gate is not fully satisfied")
    if not status["news_ready"]:
        blocked.append("News readiness gate is not satisfied")
    if not status["cross_validation_ready"]:
        blocked.append("Cross-validation is not fully interpretable")
    if not status["reproducibility_ready"]:
        blocked.append("Reproducibility package is incomplete")
    empirical_allowed = status["paper_ready"]
    cross_interpretable = status["cross_validation_ready"] and status["news_ready"]
    lines = [
        "# Paper-ready Report",
        "",
        "## Overall Status",
        f"- paper_ready: {str(status['paper_ready']).lower()}",
        f"- freeze_allowed: {str(status['freeze_allowed']).lower()}",
        f"- empirical analysis allowed: {'yes' if empirical_allowed else 'no'}",
        f"- cross-validation interpretable: {'yes' if cross_interpretable else 'no'}",
        f"- mock_mode_or_deterministic_proxy: {str(status['mock_mode']).lower()}",
        f"- systemic_conflict_flag: {str(status['systemic_conflict_flag']).lower()}",
        f"- Results can be submitted: {'yes' if status['paper_ready'] else 'no'}",
        f"- Only methodology section can be written: {'no, methodology plus pipeline validation and limitations can be written' if not status['paper_ready'] else 'no'}",
        "",
        "## What Is Complete",
        *(f"- {item}" for item in (complete or ["No final empirical component is fully ready."])),
        "",
        "## What Is Blocked",
        *(f"- {item}" for item in (blocked or ["No blockers."])),
        "",
        "## What Can Be Used In Paper Now",
    ]
    if status["paper_ready"]:
        lines.append("- Final empirical results.")
    elif status["reproducibility_ready"]:
        lines.extend(["- Methodology.", "- Scoring framework.", "- Pipeline validation.", "- Limitations.", "- Reproducibility package."])
    else:
        lines.extend(["- Preliminary method drafts only.", "- Blocker report."])
    lines.extend(
        [
            "",
            "## What Cannot Be Used Yet",
            "- Final empirical conclusions are not allowed." if not status["paper_ready"] else "- No restriction beyond ordinary paper review.",
        ]
    )
    if not status["news_ready"]:
        lines.extend(["- Real News empirical results are not allowed.", "- Cross-validation should not be interpreted as substantive empirical evidence."])
    if status["mda_ready"] and not status["news_ready"]:
        lines.append("- MD&A-only preliminary analysis may be written only if clearly labeled as MD&A-only.")
    lines.extend(
        [
            "",
            "## Exact Next Steps",
            *(f"- {item}" for item in (status["recommended_next_actions"] or ["No action required."])),
            "",
            "## Interpretation Rule",
            paper_ready_allowed_text(status),
            "",
            "## Blocking Reasons",
            *(f"- {item}" for item in (status["blocking_reasons"] or ["none"])),
            "",
            "## Advisory Warnings",
            *(f"- {item}" for item in (status["advisory_warnings"] or ["none"])),
        ]
    )
    write_markdown(root / "PAPER_OUTPUT" / "PAPER_READY_REPORT.md", lines)
    save_json(root / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json", {field: status[field] for field in PAPER_STATUS_FIELDS})
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Build final paper-ready report.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_ready_report(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
