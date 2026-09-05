from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scoring_utils import SCORING_ROOT, read_csv_rows, save_json, weights_for
from stability_common import safe_float


def compute_human_model_agreement(root: Path = SCORING_ROOT, sample_path: Path | None = None) -> dict[str, Any]:
    path = sample_path or root / "HUMAN_VALIDATION" / "mda_human_validation_sample_v2.csv"
    if not path.is_absolute():
        path = root / path
    rows = read_csv_rows(path)
    filled = [row for row in rows if str(row.get("human_score_1_5", "")).strip()]
    ready = bool(rows) and len(filled) == len(rows)
    result = {
        "mda_human_validation_ready": ready,
        "sample_rows": len(rows),
        "filled_rows": len(filled),
        "weighted_kappa_by_dimension": {},
        "total_score_icc": "",
        "mean_absolute_difference": "",
        "large_difference_ge_2_ratio": "",
    }
    if ready:
        result["weighted_kappa_by_dimension"] = _weighted_kappas(filled)
        result["mean_absolute_difference"] = _mean_absolute_difference(filled)
        result["large_difference_ge_2_ratio"] = _large_diff_ratio(filled)
        result["total_score_icc"] = _total_score_icc(filled)
    save_json(root / "HUMAN_VALIDATION" / "human_model_agreement_v2.json", result)
    return result


def _weighted_kappas(rows: list[dict[str, str]]) -> dict[str, float]:
    out = {}
    for dimension in sorted({row.get("dimension_code", "") for row in rows if row.get("dimension_code")}):
        pairs = [
            (int(safe_float(row.get("model_raw_score_1_5"))), int(safe_float(row.get("human_score_1_5"))))
            for row in rows
            if row.get("dimension_code") == dimension
        ]
        out[dimension] = _weighted_kappa(pairs)
    return out


def _weighted_kappa(pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    max_distance = 4
    observed = sum(((model - human) / max_distance) ** 2 for model, human in pairs) / len(pairs)
    model_counts = defaultdict(int)
    human_counts = defaultdict(int)
    for model, human in pairs:
        model_counts[model] += 1
        human_counts[human] += 1
    expected = 0.0
    total = len(pairs)
    for model_score in range(1, 6):
        for human_score in range(1, 6):
            expected += (model_counts[model_score] / total) * (human_counts[human_score] / total) * (((model_score - human_score) / max_distance) ** 2)
    if expected == 0:
        return 1.0 if observed == 0 else 0.0
    return round(1 - observed / expected, 6)


def _mean_absolute_difference(rows: list[dict[str, str]]) -> float:
    diffs = [abs(safe_float(row.get("model_raw_score_1_5")) - safe_float(row.get("human_score_1_5"))) for row in rows]
    return round(sum(diffs) / len(diffs), 6) if diffs else 0.0


def _large_diff_ratio(rows: list[dict[str, str]]) -> float:
    diffs = [abs(safe_float(row.get("model_raw_score_1_5")) - safe_float(row.get("human_score_1_5"))) for row in rows]
    return round(sum(1 for diff in diffs if diff >= 2) / len(diffs), 6) if diffs else 0.0


def _total_score_icc(rows: list[dict[str, str]]) -> float:
    weights = weights_for("mda")
    by_doc: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        by_doc[row.get("document_id", "")][row.get("dimension_code", "")] = {
            "model": safe_float(row.get("model_raw_score_1_5")),
            "human": safe_float(row.get("human_score_1_5")),
        }
    model_totals = []
    human_totals = []
    for dim_scores in by_doc.values():
        if not all(code in dim_scores for code in weights):
            continue
        model_totals.append(sum((dim_scores[code]["model"] - 1) / 4 * 100 * weight for code, weight in weights.items()))
        human_totals.append(sum((dim_scores[code]["human"] - 1) / 4 * 100 * weight for code, weight in weights.items()))
    return _pearson(model_totals, human_totals)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(left) != len(right):
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denom_left = sum((a - mean_left) ** 2 for a in left) ** 0.5
    denom_right = sum((b - mean_right) ** 2 for b in right) ** 0.5
    if denom_left == 0 or denom_right == 0:
        return 0.0
    return round(numerator / (denom_left * denom_right), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute human-vs-model agreement for MD&A validation sample.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--sample-path", type=Path)
    args = parser.parse_args()
    print(json.dumps(compute_human_model_agreement(args.root, args.sample_path), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
