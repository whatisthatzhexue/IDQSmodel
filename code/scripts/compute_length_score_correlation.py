#!/usr/bin/env python3
"""
Compute text-length vs disclosure-quality-score correlation (anti-shortcut test).

For both MDA and News:
1. Reads raw text files and counts words/chars
2. Reads document scores from the specified scoring run
3. Computes Pearson and Spearman correlations between text length and total_score
4. Writes a correlation report

If text length is NOT significantly correlated with score, the model is not
biased toward longer texts ("写得长 ≠ 写得好").

Usage:
    python compute_length_score_correlation.py --root . --mda-run mda_repeatability_final/run_a --news-run news_repeatability_final/run_a
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from scoring_utils import SCORING_ROOT, read_csv_rows, resolve_text_path


def word_count(text: str) -> int:
    """Count words in text (English/whitespace-based)."""
    return len(text.split())


def char_count(text: str) -> int:
    """Count non-whitespace characters."""
    return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def pearson_r(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation coefficient."""
    if len(xs) < 3:
        return 0.0
    xm, ym = mean(xs), mean(ys)
    sx, sy = stdev(xs), stdev(ys)
    if sx == 0 or sy == 0:
        return 0.0
    n = len(xs)
    return sum((xs[i] - xm) * (ys[i] - ym) for i in range(n)) / ((n - 1) * sx * sy)


def spearman_r(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    if len(xs) < 3:
        return 0.0

    def rank(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda x: x[1])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(indexed):
            j = i
            while j < len(indexed) and indexed[j][1] == indexed[i][1]:
                j += 1
            avg = (i + j + 1) / 2.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg
            i = j
        return ranks

    return pearson_r(rank(xs), rank(ys))


def load_mda_texts(root: Path, registry_path: Path) -> dict[str, str]:
    """Load MDA texts keyed by document_id."""
    registry = read_csv_rows(registry_path)
    texts = {}
    for row in registry:
        if row.get("include_flag", "").lower() != "yes":
            continue
        doc_id = row.get("document_id", "")
        text_path = resolve_text_path(row, "mda", root)
        if text_path.exists():
            texts[doc_id] = text_path.read_text(encoding="utf-8", errors="replace")
    return texts


def load_news_texts(root: Path, registry_path: Path) -> dict[str, str]:
    """Load News texts keyed by document_id."""
    registry = read_csv_rows(registry_path)
    texts = {}
    for row in registry:
        if row.get("include_flag", "").lower() != "yes":
            continue
        doc_id = row.get("document_id", "")
        text_path = resolve_text_path(row, "news", root)
        if text_path.exists():
            texts[doc_id] = text_path.read_text(encoding="utf-8", errors="replace")
    return texts


def load_document_scores(scores_path: Path) -> dict[str, float]:
    """Load total_score_100 from document_scores.csv."""
    rows = read_csv_rows(scores_path)
    return {r["document_id"]: float(r["total_score_100"]) for r in rows if r.get("total_score_100")}


def compute_correlation(
    texts: dict[str, str],
    scores: dict[str, float],
    label: str,
) -> dict[str, Any]:
    """Compute length-score correlation for a set of documents."""
    common = sorted(set(texts) & set(scores))
    if not common:
        return {"label": label, "error": "no_common_documents"}

    words = [word_count(texts[doc_id]) for doc_id in common]
    chars = [char_count(texts[doc_id]) for doc_id in common]
    score_list = [scores[doc_id] for doc_id in common]

    return {
        "label": label,
        "document_count": len(common),
        "word_count": {
            "mean": round(mean(words), 1),
            "median": round(sorted(words)[len(words) // 2], 1),
            "min": min(words),
            "max": max(words),
            "std": round(stdev(words), 1),
        },
        "char_count": {
            "mean": round(mean(chars), 1),
            "median": round(sorted(chars)[len(chars) // 2], 1),
            "min": min(chars),
            "max": max(chars),
            "std": round(stdev(chars), 1),
        },
        "score": {
            "mean": round(mean(score_list), 2),
            "median": round(sorted(score_list)[len(score_list) // 2], 2),
            "min": min(score_list),
            "max": max(score_list),
            "std": round(stdev(score_list), 2),
        },
        "pearson_word_score": round(pearson_r(words, score_list), 4),
        "pearson_char_score": round(pearson_r(chars, score_list), 4),
        "spearman_word_score": round(spearman_r(words, score_list), 4),
        "spearman_char_score": round(spearman_r(chars, score_list), 4),
        "anti_shortcut_pass": abs(pearson_r(words, score_list)) < 0.3,
        "top5_longest_mean_score": round(
            mean([score_list[i] for i in sorted(range(len(words)), key=lambda i: words[i], reverse=True)[:5]]), 2
        ),
        "top5_shortest_mean_score": round(
            mean([score_list[i] for i in sorted(range(len(words)), key=lambda i: words[i])[:5]]), 2
        ),
    }


def write_report(results: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Text Length vs Score Correlation Report",
        "",
        "Purpose: verify the model is NOT scoring longer texts higher (anti-shortcut test).",
        "Pass condition: |Pearson r| < 0.3 (weak or no correlation between text length and score).",
        "",
    ]
    for r in results:
        lines.append(f"## {r['label']}")
        lines.append(f"- documents: {r.get('document_count', 0)}")
        if "error" in r:
            lines.append(f"- error: {r['error']}")
            continue
        wc = r["word_count"]
        sc = r["score"]
        lines.append(f"- word count: mean={wc['mean']}, median={wc['median']}, range=[{wc['min']}, {wc['max']}]")
        lines.append(f"- score: mean={sc['mean']}, median={sc['median']}, range=[{sc['min']}, {sc['max']}]")
        lines.append(f"- **Pearson(word, score): r={r['pearson_word_score']}**")
        lines.append(f"- **Spearman(word, score): r={r['spearman_word_score']}**")
        lines.append(f"- Pearson(char, score): r={r['pearson_char_score']}")
        lines.append(f"- anti_shortcut_pass: **{r['anti_shortcut_pass']}**")
        lines.append(f"- top 5 longest mean score: {r['top5_longest_mean_score']}")
        lines.append(f"- top 5 shortest mean score: {r['top5_shortest_mean_score']}")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("- |r| < 0.3: No meaningful length bias (passed)")
    lines.append("- 0.3 <= |r| < 0.5: Weak length bias (needs discussion)")
    lines.append("- |r| >= 0.5: Moderate/strong length bias (problematic for validity)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {path}")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute text-length vs score correlation.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--mda-run", default="mda_repeatability_final/run_a")
    parser.add_argument("--news-run", default="news_repeatability_final/run_a")
    args = parser.parse_args()
    root: Path = args.root

    results: list[dict[str, Any]] = []

    # MDA
    mda_registry = root / "01_registry" / "mda_registry.csv"
    mda_run_dir = root / "06_ratings" / args.mda_run
    mda_candidates = (
        list(mda_run_dir.glob("*_document_scores.csv")) if mda_run_dir.exists() else []
    )
    # Prefer canonical naming: mda_final_document_scores.csv
    canonical = mda_run_dir / "mda_final_document_scores.csv"
    if canonical.exists():
        mda_scores_csv = canonical
    elif mda_candidates:
        mda_scores_csv = mda_candidates[0]
    else:
        mda_scores_csv = canonical

    if mda_registry.exists() and mda_scores_csv.exists():
        print(f"Loading MDA texts...")
        mda_texts = load_mda_texts(root, mda_registry)
        mda_scores = load_document_scores(mda_scores_csv)
        print(f"  {len(mda_texts)} texts, {len(mda_scores)} scores")
        results.append(compute_correlation(mda_texts, mda_scores, "MDA"))
    else:
        print(f"MDA: registry={mda_registry.exists()}, scores={mda_scores_csv.exists()}")
        print(f"  scores path: {mda_scores_csv}")

    # News
    news_registry = root / "01_registry" / "news_registry.csv"
    news_run_dir = root / "06_ratings" / args.news_run
    news_candidates = (
        list(news_run_dir.glob("*_document_scores.csv")) if news_run_dir.exists() else []
    )
    canonical_news = news_run_dir / "news_final_document_scores.csv"
    if canonical_news.exists():
        news_scores_csv = canonical_news
    elif news_candidates:
        news_scores_csv = news_candidates[0]
    else:
        news_scores_csv = canonical_news

    if news_registry.exists() and news_scores_csv.exists():
        print(f"Loading News texts...")
        news_texts = load_news_texts(root, news_registry)
        news_scores = load_document_scores(news_scores_csv)
        print(f"  {len(news_texts)} texts, {len(news_scores)} scores")
        results.append(compute_correlation(news_texts, news_scores, "News"))
    else:
        print(f"News: registry={news_registry.exists()}, scores={news_scores_csv.exists()}")

    if not results:
        print("No results. Check paths.")
        return 1

    report_path = root / "08_reports" / "length_score_correlation_report.md"
    write_report(results, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
