from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_news_registry import build_news_registry
from diagnose_mda_stability_failures import diagnose_mda_stability_failures
from extract_news_texts import extract_news_texts
from paper_utils import ensure_paper_output_dirs, real_news_rows
from run_cross_validation import run_cross_validation
from run_final_convergence_fix import run_final_convergence_fix
from run_final_data_freeze import run_final_data_freeze
from run_paper_readiness_check import run_paper_readiness_check
from scoring_utils import SCORING_ROOT, append_jsonl
from stability_common import write_markdown


def run_paper_convergence_loop(root: Path = SCORING_ROOT, max_rounds: int = 5) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    log_path = root / "PAPER_OUTPUT" / "00_status" / "paper_convergence_log.md"
    rounds_path = root / "PAPER_OUTPUT" / "00_status" / "paper_convergence_rounds.jsonl"
    if rounds_path.exists():
        rounds_path.unlink()
    log_lines = ["# Paper Convergence Log", ""]
    previous_signature = ""
    rounds: list[dict[str, Any]] = []
    stop_reason = "max_rounds reached"
    for round_number in range(1, max_rounds + 1):
        before = run_paper_readiness_check(root)
        actions: list[str] = ["run_paper_readiness_check"]
        blockers_before = list(before.get("blocking_reasons", []))
        if before["paper_ready"]:
            stop_reason = "paper_ready true"
            break
        if not before["mda_ready"]:
            diagnose_mda_stability_failures(root)
            actions.append("diagnose_mda_stability_failures")
        if not before["news_ready"]:
            real_news = real_news_rows(root)
            if not real_news:
                _write_real_news_required(root)
                actions.append("write_news_real_data_required")
                stop_reason = "real News data missing"
                after = run_paper_readiness_check(root)
                entry = _entry(round_number, before, after, actions, blockers_before, stop_reason)
                rounds.append(entry)
                append_jsonl(rounds_path, entry)
                break
            if not any((root / row.get("text_path", "")).exists() for row in real_news if row.get("text_path")):
                build_news_registry(root=root, overwrite=True)
                extract_news_texts(root=root, overwrite=True)
                actions.extend(["build_news_registry", "extract_news_texts"])
            run_final_convergence_fix(root=root, max_rounds=1)
            actions.append("run_final_convergence_fix")
        if not before["cross_validation_ready"] and real_news_rows(root):
            run_cross_validation("pilot", root=root, limit=10, mock_mode=False)
            actions.append("run_cross_validation")
        run_final_data_freeze(root)
        actions.append("run_final_data_freeze")
        after = run_paper_readiness_check(root)
        actions.append("run_paper_readiness_check")
        entry = _entry(round_number, before, after, actions, blockers_before, "continue")
        rounds.append(entry)
        append_jsonl(rounds_path, entry)
        signature = json.dumps({"paper_ready": after["paper_ready"], "blockers": after["blocking_reasons"]}, sort_keys=True)
        if after["paper_ready"]:
            stop_reason = "paper_ready true"
            break
        if signature == previous_signature:
            stop_reason = "no further automatic progress"
            break
        previous_signature = signature
    final_status = run_paper_readiness_check(root)
    log_lines.extend(_log_lines(rounds, final_status, stop_reason))
    write_markdown(log_path, log_lines)
    return {"paper_ready": final_status["paper_ready"], "rounds_run": len(rounds), "stop_reason": stop_reason, "blocking_reasons": final_status["blocking_reasons"]}


def _entry(round_number: int, before: dict[str, Any], after: dict[str, Any], actions: list[str], blockers_before: list[str], stop_reason: str) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "round": round_number,
        "actions": actions,
        "paper_ready_before": before.get("paper_ready"),
        "paper_ready_after": after.get("paper_ready"),
        "blockers_before": blockers_before,
        "blockers_after": after.get("blocking_reasons", []),
        "stop_reason": stop_reason,
    }


def _write_real_news_required(root: Path) -> None:
    write_markdown(
        root / "PAPER_OUTPUT" / "00_status" / "news_real_data_required.md",
        [
            "# Real News Data Required",
            "",
            "The system is engineering-ready but not paper-ready because real news data are missing.",
            "Synthetic news placeholders cannot be used for final empirical analysis.",
        ],
    )


def _log_lines(rounds: list[dict[str, Any]], final_status: dict[str, Any], stop_reason: str) -> list[str]:
    lines = [f"- stop_reason: {stop_reason}", f"- final_paper_ready: {str(final_status['paper_ready']).lower()}", ""]
    for item in rounds:
        lines.append(f"- round {item['round']}: actions={','.join(item['actions'])}; blockers_after={','.join(item['blockers_after']) or 'none'}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper convergence loop without bypassing gates.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run_paper_convergence_loop(args.root, args.max_rounds), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
