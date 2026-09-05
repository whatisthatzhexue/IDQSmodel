from __future__ import annotations

import argparse
from pathlib import Path

from run_scoring import write_summary_report
from scoring_utils import SCORING_ROOT, load_json


def build_summary_report(root: Path = SCORING_ROOT, summary_path: Path | None = None) -> Path:
    summary = {}
    if summary_path and summary_path.exists():
        summary = load_json(summary_path)
    return write_summary_report(root, summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build scoring summary report.")
    parser.add_argument("--summary-path")
    args = parser.parse_args()
    path = build_summary_report(summary_path=Path(args.summary_path) if args.summary_path else None)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
