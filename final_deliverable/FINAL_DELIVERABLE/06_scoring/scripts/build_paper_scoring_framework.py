from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paper_utils import dimensions_table, ensure_paper_output_dirs, write_dual_language_sections, write_table
from scoring_utils import SCORING_ROOT


MDA_ROBUSTNESS = [
    {"scheme": "W1_baseline", "AR01": 0.20, "AR02": 0.25, "AR03": 0.25, "AR04": 0.20, "AR05": 0.10},
    {"scheme": "W2_main_recommended", "AR01": 0.15, "AR02": 0.25, "AR03": 0.30, "AR04": 0.20, "AR05": 0.10},
    {"scheme": "W3_conservative_financial", "AR01": 0.15, "AR02": 0.30, "AR03": 0.25, "AR04": 0.20, "AR05": 0.10},
]

NEWS_ROBUSTNESS = [
    {"scheme": "W1_baseline", "N01": 0.30, "N02": 0.20, "N03": 0.20, "N04": 0.15, "N05": 0.15},
    {"scheme": "W2_main_recommended", "N01": 0.30, "N02": 0.25, "N03": 0.15, "N04": 0.20, "N05": 0.10},
    {"scheme": "W3_source_reliability", "N01": 0.30, "N02": 0.30, "N03": 0.15, "N04": 0.15, "N05": 0.10},
]


def build_paper_scoring_framework(root: Path = SCORING_ROOT) -> dict[str, Any]:
    ensure_paper_output_dirs(root)
    mda_dims = dimensions_table("mda")
    news_dims = dimensions_table("news")
    write_table(root, "table_3_mda_dimensions", ["dimension", "name", "main_weight"], mda_dims)
    write_table(root, "table_4_news_dimensions", ["dimension", "name", "main_weight"], news_dims)
    weighting_rows = []
    for row in MDA_ROBUSTNESS:
        weighting_rows.append({"doc_type": "mda", **row})
    for row in NEWS_ROBUSTNESS:
        weighting_rows.append({"doc_type": "news", **row})
    write_table(root, "table_5_weighting_schemes", ["doc_type", "scheme", "AR01", "AR02", "AR03", "AR04", "AR05", "N01", "N02", "N03", "N04", "N05"], weighting_rows)
    en_lines = [
        "# Scoring Framework",
        "",
        "MD&A uses AR01-AR05 and News uses N01-N05. Scores are produced dimension by dimension and aggregated by Python.",
        "The paper output layer does not change the scoring dimensions, model, or weights.",
        "AR02 is limited to internal numeric consistency and traceability in MD&A text.",
        "Cross-validation is auxiliary triangulation and does not alter scores.",
    ]
    cn_lines = [
        "# 评分框架",
        "",
        "MD&A 使用 AR01-AR05，News 使用 N01-N05。评分按单维度执行，并由 Python 聚合。",
        "论文输出层不修改维度、模型或权重。",
        "AR02 仅限于 MD&A 文本内部数值一致性与逻辑可追踪性。",
        "Cross-validation 只是辅助交叉论证，不改变分数。",
    ]
    outputs = write_dual_language_sections(root, "01_methods", "scoring_framework_cn.md", "scoring_framework_en.md", cn_lines, en_lines)
    outputs.update({"weighting_table": str(root / "PAPER_OUTPUT" / "tables" / "table_5_weighting_schemes.csv")})
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper scoring framework and dimension tables.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    args = parser.parse_args()
    print(json.dumps(build_paper_scoring_framework(args.root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
