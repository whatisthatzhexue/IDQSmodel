from __future__ import annotations

import argparse
from pathlib import Path

from scoring_utils import SCORING_ROOT, read_csv_rows


def generate_news_missing_next_steps(root: Path = SCORING_ROOT) -> Path:
    rows = [row for row in read_csv_rows(root / "01_registry" / "news_registry.csv") if row.get("include_flag", "").lower() == "yes"]
    path = root / "08_reports" / "news_data_missing_next_steps.md"
    lines = [
        "# News Data Missing Next Steps",
        "",
        f"- news_registry.csv is empty: {len(rows) == 0}",
        "- News scoring has not started for real article text.",
        "- Cross-validation currently has no real news claims and must not be treated as a research result.",
        "",
        "## Put News Text In One Of These Locations",
        "- `01_raw_news/`",
        "- `news/`",
        "- `raw_news/`",
        "- `06_scoring/input_news/`",
        "",
        "## Run After News Text Is Ready",
        "```bash",
        "python 06_scoring/scripts/build_news_registry.py",
        "python 06_scoring/scripts/extract_news_texts.py",
        "python 06_scoring/scripts/run_scoring.py --doc-type news --mode pilot --model qwen3:8b",
        "python 06_scoring/scripts/run_scoring.py --doc-type news --mode full --model qwen3:8b",
        "python 06_scoring/scripts/run_cross_validation.py --mode full",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate next-step report when news_registry.csv is empty.")
    parser.add_argument("--root", default=str(SCORING_ROOT))
    args = parser.parse_args()
    print(generate_news_missing_next_steps(Path(args.root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
