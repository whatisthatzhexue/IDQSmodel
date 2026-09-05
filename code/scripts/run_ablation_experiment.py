#!/usr/bin/env python3
"""
Ablation experiment: test score stability under prompt wording perturbation.

Per the group decision (docs/指示.txt):
- NO full ICC on ablation samples (ICC is for fixed-prompt test-retest)
- Instead: pick key sample points, run significance tests (paired T-test,
  proportion test), and argue qualitatively for robustness.

Design:
- Variant A (baseline): original prompt, results already exist (mda_final_full)
- Variant B (strict):   adds "Be strict..." tone instruction
- Variant C (lenient):  adds "Be lenient..." tone instruction
- Only the 25 selected sample documents are re-scored with variants B and C.

Outputs:
- 06_ratings/mda_ablation_strict/   (variant B results)
- 06_ratings/mda_ablation_lenient/  (variant C results)
- 08_reports/ablation_report.md     (statistical comparison)

Usage:
    python run_ablation_experiment.py --root . --doc-type mda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from scipy import stats
from scoring_utils import SCORING_ROOT, read_csv_rows, write_csv_rows


TONE_VARIANTS = {
    # single source of truth for tone texts lives in run_ablation_variant_scoring.py
    # (TONE_INSTRUCTIONS); this dict is kept only as a reference for report readers.
    "strict": (
        "Scoring tone instruction: Be STRICT. Only award high scores when the text provides "
        "explicit, concrete, verifiable evidence for the dimension. When in doubt, score LOWER.\n"
    ),
    "lenient": (
        "Scoring tone instruction: Be LENIENT. Focus on overall information quality rather than "
        "specific missing details. When in doubt, score HIGHER.\n"
    ),
    "mild_strict": (
        "Scoring tone instruction: Apply a rigorous, conservative evaluation standard. High "
        "scores require clear and explicit evidence in the text; indirect or ambiguous support "
        "should not be over-credited.\n"
    ),
    "mild_lenient": (
        "Scoring tone instruction: Apply a constructive, holistic evaluation standard. Credit "
        "substantive discussion and overall disclosure quality even when some specific details "
        "are missing.\n"
    ),
}

ABLATION_REPORT_HEADERS = [
    "document_id",
    "baseline_total",
    "variant_total",
    "total_diff",
    "grade_baseline",
    "grade_variant",
    "grade_changed",
    "dimensions_same",
    "dimensions_changed_count",
]


def load_ablation_sample(root: Path) -> list[str]:
    """Load the 25 selected sample document IDs."""
    path = root / "08_reports" / "ablation_sample.csv"
    if not path.exists():
        # Fallback: use all docs
        scores_path = root / "06_ratings" / "mda_repeatability_final" / "run_a" / "mda_repeatability_run_a_document_scores.csv"
        rows = read_csv_rows(scores_path)
        return [r["document_id"] for r in rows]
    with open(path, encoding="utf-8-sig") as f:
        return [r["document_id"] for r in csv.DictReader(f)]


def load_baseline_scores(root: Path) -> dict[str, dict[str, str]]:
    """Load baseline (original prompt) document scores."""
    path = root / "06_ratings" / "mda_final_full" / "mda_final_document_scores.csv"
    if not path.exists():
        path = root / "06_ratings" / "mda_repeatability_final" / "run_a" / "mda_repeatability_run_a_document_scores.csv"
    rows = read_csv_rows(path)
    return {r["document_id"]: r for r in rows}


def load_variant_scores(root: Path, variant_dir: str) -> dict[str, dict[str, str]]:
    """Load variant scoring results from its output directory."""
    path = root / "06_ratings" / variant_dir / "mda_final_document_scores.csv"
    if not path.exists():
        # Try alternate naming: ratings prefix may differ
        candidates = list((root / "06_ratings" / variant_dir).glob("*_document_scores.csv"))
        if candidates:
            path = candidates[0]
        else:
            return {}
    rows = read_csv_rows(path)
    return {r["document_id"]: r for r in rows}


def grade_label(total: float) -> str:
    if total >= 85:
        return "A"
    if total >= 70:
        return "B"
    if total >= 55:
        return "C"
    return "D"


def paired_t_test(diffs: list[float]) -> tuple[float, float]:
    """Paired T-test: H0: mean(diff) == 0. Returns (t_stat, two-tailed p via t-dist, df=n-1)."""
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    mean_diff = statistics.mean(diffs)
    sd_diff = statistics.stdev(diffs) if n > 1 else 0.0
    if sd_diff == 0:
        return 0.0, 1.0 if mean_diff == 0 else 0.0
    t_stat = mean_diff / (sd_diff / math.sqrt(n))
    # Two-tailed p from the t distribution (n=25 -> df=24; normal approx would be anti-conservative)
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)
    return t_stat, p_value


def proportion_test(success: int, total: int, p0: float = 0.9) -> tuple[float, float]:
    """One-sample proportion test: H0: p >= p0 (grade stability >= 90%)."""
    if total == 0:
        return 0.0, 1.0
    p_hat = success / total
    se = math.sqrt(p0 * (1 - p0) / total)
    if se == 0:
        return 0.0, 1.0
    z_stat = (p_hat - p0) / se
    from math import erf, sqrt
    p_value = 0.5 * (1 + erf(z_stat / sqrt(2)))
    return z_stat, p_value


def compare_variant(
    baseline: dict[str, dict[str, str]],
    variant: dict[str, dict[str, str]],
    sample_ids: list[str],
    variant_name: str,
) -> dict[str, Any]:
    """Compare baseline vs variant scores on the sample."""
    rows = []
    diffs = []
    grade_same = 0
    dims_same_total = 0
    dims_total = 0

    for doc_id in sample_ids:
        b = baseline.get(doc_id, {})
        v = variant.get(doc_id, {})
        if not b or not v:
            continue
        b_total = float(b.get("total_score_100", 0))
        v_total = float(v.get("total_score_100", 0))
        b_grade = b.get("grade_label", grade_label(b_total))
        v_grade = v.get("grade_label", grade_label(v_total))
        diff = v_total - b_total
        diffs.append(diff)

        dims_changed = 0
        dims_doc_total = 0
        for code in ["AR01", "AR02", "AR03", "AR04", "AR05"]:
            if b.get(code) and v.get(code):
                dims_doc_total += 1
                dims_total += 1
                if b.get(code) == v.get(code):
                    dims_same_total += 1
                else:
                    dims_changed += 1

        grade_changed = b_grade != v_grade
        if not grade_changed:
            grade_same += 1

        rows.append({
            "document_id": doc_id,
            "baseline_total": b_total,
            "variant_total": v_total,
            "total_diff": round(diff, 4),
            "grade_baseline": b_grade,
            "grade_variant": v_grade,
            "grade_changed": grade_changed,
            "dimensions_same": dims_doc_total - dims_changed,
            "dimensions_changed_count": dims_changed,
        })

    n = len(rows)
    t_stat, t_p = paired_t_test(diffs)
    grade_stability = grade_same / n if n else 0.0
    z_stat, z_p = proportion_test(grade_same, n)
    dim_stability = dims_same_total / dims_total if dims_total else 0.0

    return {
        "variant": variant_name,
        "sample_count": n,
        "mean_diff": round(statistics.mean(diffs), 4) if diffs else None,
        "max_abs_diff": round(max(abs(d) for d in diffs), 4) if diffs else None,
        "paired_t_stat": round(t_stat, 4),
        "paired_t_p": round(t_p, 6),
        "significant_at_0.05": t_p < 0.05,
        "grade_stability_rate": round(grade_stability, 4),
        "grade_proportion_z": round(z_stat, 4),
        "grade_proportion_p": round(z_p, 6),
        "dimension_agreement_rate": round(dim_stability, 4),
        "rows": rows,
    }


def write_ablation_report(
    root: Path,
    results: list[dict[str, Any]],
) -> None:
    """Write the ablation report markdown."""
    lines = [
        "# Ablation Experiment Report",
        "",
        "Prompt wording perturbation test on key sample documents.",
        "Per group decision: no ICC on ablation; significance tests only.",
        "",
    ]
    for r in results:
        lines.append(f"## Variant: {r['variant']}")
        lines.append(f"- sample documents: {r['sample_count']}")
        lines.append(f"- mean total-score difference: {r['mean_diff']}")
        lines.append(f"- max absolute difference: {r['max_abs_diff']}")
        lines.append(f"- paired T-test: t={r['paired_t_stat']}, p={r['paired_t_p']}")
        lines.append(f"- significant difference at 0.05: **{r['significant_at_0.05']}**")
        lines.append(f"- grade stability (unchanged): {r['grade_stability_rate']*100:.1f}%")
        lines.append(f"- grade proportion test vs 90%: z={r['grade_proportion_z']}, p={r['grade_proportion_p']}")
        lines.append(f"- dimension score agreement: {r['dimension_agreement_rate']*100:.1f}%")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- If paired T-test p > 0.05: no significant mean shift → prompt wording does not bias scores.")
    lines.append("- If grade stability >= 90%: rankings/grade labels are robust to wording.")
    lines.append("- If dimension agreement is high: per-dimension scores are stable.")
    lines.append("")
    lines.append("## Conclusion Template (for paper)")
    lines.append(
        "Prompt-wording perturbation on key boundary/typical samples produced no significant "
        "score shift (paired T-test p > 0.05) and high grade stability, indicating that the "
        "scoring framework is robust to prompt phrasing variations."
    )
    path = root / "08_reports" / "ablation_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAblation report written to: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ablation comparison (no LLM calls; requires variant results).")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--strict-dir", default="mda_ablation_strict")
    parser.add_argument("--lenient-dir", default="mda_ablation_lenient")
    parser.add_argument("--mild-strict-dir", default="mda_ablation_mild_strict")
    parser.add_argument("--mild-lenient-dir", default="mda_ablation_mild_lenient")
    args = parser.parse_args()
    root = args.root

    sample_ids = load_ablation_sample(root)
    print(f"Sample documents: {len(sample_ids)}")

    baseline = load_baseline_scores(root)
    print(f"Baseline scores loaded: {len(baseline)}")

    results = []
    variants = [
        ("strict", args.strict_dir),
        ("lenient", args.lenient_dir),
        ("mild_strict", args.mild_strict_dir),
        ("mild_lenient", args.mild_lenient_dir),
    ]
    for variant_name, variant_dir in variants:
        variant = load_variant_scores(root, variant_dir)
        print(f"Variant {variant_name}: {len(variant)} scores loaded from {variant_dir}")
        if variant:
            result = compare_variant(baseline, variant, sample_ids, variant_name)
            results.append(result)
            # Save comparison CSV
            csv_path = root / "08_reports" / f"ablation_{variant_name}_comparison.csv"
            write_csv_rows(csv_path, ABLATION_REPORT_HEADERS, result["rows"])
            print(f"  Comparison CSV: {csv_path}")
        else:
            print(f"  WARNING: variant scores not found. Run scoring with variant prompt first.")

    if results:
        write_ablation_report(root, results)
    else:
        print("No variant results available. Score variants first.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
