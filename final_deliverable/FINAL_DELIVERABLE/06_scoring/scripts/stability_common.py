from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scoring_utils import (
    SCORING_ROOT,
    ensure_directories,
    read_csv_rows,
    save_json,
    standardize_score,
    write_csv_rows,
)


STABILITY_SUBDIRS = ["inputs", "outputs", "reports", "plots", "logs"]

HIGH_VOLATILITY_HEADERS = [
    "case_id",
    "document_id",
    "doc_type",
    "ticker",
    "company_name",
    "year_or_publish_date",
    "issue_type",
    "severity",
    "score_impact",
    "rank_impact",
    "dimension_affected",
    "recommended_action",
    "source_file",
]

INPUT_PRIORITIES = {
    "mda_scores": [
        "FINAL_OUTPUT/mda_final_dataset.csv",
        "06_ratings/mda_final_v2/mda_final_document_scores.csv",
        "06_ratings/mda_v2_full/mda_v2_full_document_scores.csv",
        "06_ratings/mda_document_scores_final_candidate.csv",
        "06_ratings/mda_document_scores.csv",
        "06_ratings/mda_v2_sample_rescore/mda_v2_document_scores_sample.csv",
    ],
    "mda_long": [
        "06_ratings/mda_final_v2/mda_final_ratings_long.csv",
        "06_ratings/mda_v2_full/mda_v2_full_ratings_long.csv",
        "06_ratings/mda_ratings_long.csv",
        "06_ratings/mda_v2_sample_rescore/mda_v2_ratings_long_sample.csv",
    ],
    "news_scores": [
        "FINAL_OUTPUT/news_final_dataset.csv",
        "06_ratings/news_final_full/news_final_document_scores.csv",
        "06_ratings/news_v2_full/news_v2_full_document_scores.csv",
        "06_ratings/news_document_scores_final_candidate.csv",
    ],
    "news_long": [
        "06_ratings/news_final_full/news_final_ratings_long.csv",
        "06_ratings/news_v2_full/news_v2_full_ratings_long.csv",
    ],
}

REPEAT_INPUTS = {
    "mda_run1": "06_ratings/repeat_runs/mda_repeat_run_1_document_scores.csv",
    "mda_run2": "06_ratings/repeat_runs/mda_repeat_run_2_document_scores.csv",
    "news_run1": "06_ratings/repeat_runs/news_repeat_run_1_document_scores.csv",
    "news_run2": "06_ratings/repeat_runs/news_repeat_run_2_document_scores.csv",
}

VERSION_INPUTS = {
    "mda_v1": "archive/v1_initial_mda_scoring/mda_document_scores.csv",
    "mda_v2": "06_ratings/mda_final_v2/mda_final_document_scores.csv",
}

STRUCTURE_INPUTS = {
    "mda_document_level": "06_ratings/document_level/mda_document_scores.csv",
    "mda_single_dimension": "06_ratings/single_dimension/mda_document_scores.csv",
    "news_document_level": "06_ratings/document_level/news_document_scores.csv",
    "news_single_dimension": "06_ratings/single_dimension/news_document_scores.csv",
}


def stability_root(root: Path = SCORING_ROOT) -> Path:
    return root / "11_stability_analysis"


def ensure_stability_dirs(root: Path = SCORING_ROOT) -> Path:
    ensure_directories(root)
    base = stability_root(root)
    for subdir in STABILITY_SUBDIRS:
        (base / subdir).mkdir(parents=True, exist_ok=True)
    return base


def discover_inputs(root: Path = SCORING_ROOT) -> dict[str, Any]:
    base = ensure_stability_dirs(root)
    selected: dict[str, str] = {}
    available: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for key, candidates in INPUT_PRIORITIES.items():
        found = None
        for rel_path in candidates:
            path = root / rel_path
            record = input_record(root, path, key)
            if path.exists():
                available.append(record)
                if found is None and record["row_count"] > 0:
                    found = path
            else:
                missing.append(record)
        selected[key] = str(found) if found else ""

    for group_name, mapping in [
        ("repeat", REPEAT_INPUTS),
        ("version", VERSION_INPUTS),
        ("structure", STRUCTURE_INPUTS),
    ]:
        for key, rel_path in mapping.items():
            path = root / rel_path
            record = input_record(root, path, f"{group_name}:{key}")
            if path.exists():
                available.append(record)
            else:
                missing.append(record)
            selected[key] = str(path) if path.exists() and record["row_count"] > 0 else ""

    registry_path = root / "01_registry" / "news_registry.csv"
    registry_record = input_record(root, registry_path, "news_registry")
    if registry_path.exists():
        available.append(registry_record)
        selected["news_registry"] = str(registry_path) if registry_record["row_count"] > 0 else ""
    else:
        missing.append(registry_record)
        selected["news_registry"] = ""

    manifest = {
        "selected_inputs": selected,
        "available_inputs": available,
        "missing_inputs": missing,
        "sha256_before": {record["path"]: record.get("sha256", "") for record in available},
        "sha256_after": {},
        "input_files_unchanged": True,
    }
    save_json(base / "inputs" / "input_manifest.json", manifest)
    return manifest


def finalize_input_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    after = {}
    unchanged = True
    for path_text, before in manifest.get("sha256_before", {}).items():
        path = Path(path_text)
        current = sha256_file(path) if path.exists() else ""
        after[path_text] = current
        if current != before:
            unchanged = False
    manifest["sha256_after"] = after
    manifest["input_files_unchanged"] = unchanged
    save_json(stability_root(root) / "inputs" / "input_manifest.json", manifest)
    return manifest


def input_record(root: Path, path: Path, key: str) -> dict[str, Any]:
    exists = path.exists()
    return {
        "key": key,
        "path": str(path),
        "relative_path": str(path.relative_to(root)) if _is_relative_to(path, root) else str(path),
        "exists": exists,
        "row_count": len(read_csv_rows(path)) if exists else 0,
        "sha256": sha256_file(path) if exists else "",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_empty_csv(path: Path, headers: list[str]) -> None:
    write_csv_rows(path, headers, [])


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_plot_skip(root: Path, plot_name: str, reason: str) -> None:
    path = stability_root(root) / "plots" / "plot_skips.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Plot Skips\n\n"
    existing += f"- {plot_name}: {reason}\n"
    path.write_text(existing, encoding="utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_number(value: Any) -> bool:
    try:
        if value is None or str(value).strip() == "":
            return False
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def fmt(value: float | int | str, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    rounded = round(float(value), digits)
    if rounded.is_integer():
        return str(int(rounded))
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")


def raw_to_100(raw_score: Any) -> float:
    return standardize_score(safe_float(raw_score, 1.0))


def weighted_total(row: dict[str, Any], weights: dict[str, float]) -> float:
    return round(sum(raw_to_100(row.get(code)) * weight for code, weight in weights.items()), 6)


def pearson(values_a: list[float], values_b: list[float]) -> float:
    pairs = [(float(a), float(b)) for a, b in zip(values_a, values_b) if math.isfinite(float(a)) and math.isfinite(float(b))]
    if len(pairs) < 2:
        return 0.0
    xs, ys = zip(*pairs)
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denominator_x == 0 or denominator_y == 0:
        return 0.0
    return max(-1.0, min(1.0, round(numerator / (denominator_x * denominator_y), 6)))


def ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    out = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            out[original_index] = rank
        index = end
    return out


def spearman(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return 0.0
    return pearson(ranks(values_a), ranks(values_b))


def descending_ranks(rows: list[dict[str, Any]], score_field: str) -> dict[str, int]:
    ordered = sorted(rows, key=lambda row: (-safe_float(row.get(score_field)), str(row.get("document_id", ""))))
    result: dict[str, int] = {}
    previous_score: float | None = None
    current_rank = 0
    for idx, row in enumerate(ordered, start=1):
        score = safe_float(row.get(score_field))
        if previous_score is None or score != previous_score:
            current_rank = idx
            previous_score = score
        result[str(row.get("document_id", ""))] = current_rank
    return result


def top_bottom_overlap(rows: list[dict[str, Any]], left_field: str, right_field: str, *, top: bool) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "denominator": 0, "ratio": 0.0}
    size = max(1, math.ceil(len(rows) * 0.20))
    reverse = top
    left_ids = {
        row["document_id"]
        for row in sorted(rows, key=lambda item: (safe_float(item.get(left_field)), str(item.get("document_id", ""))), reverse=reverse)[:size]
    }
    right_ids = {
        row["document_id"]
        for row in sorted(rows, key=lambda item: (safe_float(item.get(right_field)), str(item.get("document_id", ""))), reverse=reverse)[:size]
    }
    count = len(left_ids & right_ids)
    return {"count": count, "denominator": size, "ratio": round(count / size, 6)}


def dimension_stats(rows: list[dict[str, str]], dimensions: list[str]) -> list[dict[str, Any]]:
    output = []
    for dimension in dimensions:
        values = [safe_float(row.get(dimension)) for row in rows if is_number(row.get(dimension))]
        output.append(
            {
                "dimension": dimension,
                "mean": round(statistics.mean(values), 6) if values else 0.0,
                "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
                "count": len(values),
            }
        )
    return output


def dimension_correlation_matrix(rows: list[dict[str, str]], dimensions: list[str]) -> dict[str, dict[str, float]]:
    matrix: dict[str, dict[str, float]] = {}
    for left in dimensions:
        matrix[left] = {}
        for right in dimensions:
            left_values = [safe_float(row.get(left)) for row in rows if is_number(row.get(left)) and is_number(row.get(right))]
            right_values = [safe_float(row.get(right)) for row in rows if is_number(row.get(left)) and is_number(row.get(right))]
            matrix[left][right] = 1.0 if left == right and left_values else pearson(left_values, right_values)
    return matrix


def high_correlation_pairs(matrix: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    pairs = []
    dimensions = list(matrix.keys())
    for left_idx, left in enumerate(dimensions):
        for right in dimensions[left_idx + 1 :]:
            corr = matrix.get(left, {}).get(right, 0.0)
            if corr > 0.85:
                pairs.append({"dimension_a": left, "dimension_b": right, "correlation": corr, "warning": "possible redundancy"})
    return pairs


def stability_judgement(spearman_value: float) -> str:
    if spearman_value >= 0.90:
        return "stable"
    if spearman_value >= 0.80:
        return "moderately stable"
    return "weight-sensitive, needs review"


def valid_dimension_rows(rows: list[dict[str, str]], dimensions: list[str]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("document_id") and all(is_number(row.get(code)) for code in dimensions)]


def collect_correlation_metrics(summary: Any, analysis_name: str) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if isinstance(summary, dict):
        for key, value in summary.items():
            if isinstance(value, dict):
                metrics.extend(collect_correlation_metrics(value, f"{analysis_name}.{key}"))
            elif key in {"pearson", "spearman", "icc"} and isinstance(value, (int, float)):
                metrics.append({"analysis": analysis_name, "metric": key, "value": value})
        for item in summary.get("pairwise_metrics", []) if isinstance(summary.get("pairwise_metrics"), list) else []:
            if isinstance(item, dict):
                label = f"{analysis_name}:{item.get('scheme_a', '')}-{item.get('scheme_b', '')}"
                for metric in ["pearson", "spearman"]:
                    if metric in item:
                        metrics.append({"analysis": label, "metric": metric, "value": item[metric]})
    elif isinstance(summary, list):
        for idx, item in enumerate(summary):
            metrics.extend(collect_correlation_metrics(item, f"{analysis_name}[{idx}]"))
    return metrics


def write_correlation_metrics(root: Path, summaries: dict[str, Any]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for name, summary in summaries.items():
        metrics.extend(collect_correlation_metrics(summary, name))
    save_json(stability_root(root) / "outputs" / "correlation_metrics.json", metrics)
    return metrics


def write_high_volatility(root: Path, cases: list[dict[str, Any]]) -> Path:
    deduped: list[dict[str, Any]] = []
    seen = set()
    for idx, case in enumerate(cases, start=1):
        key = (
            case.get("document_id", ""),
            case.get("doc_type", ""),
            case.get("issue_type", ""),
            case.get("dimension_affected", ""),
            case.get("source_file", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        row = {header: case.get(header, "") for header in HIGH_VOLATILITY_HEADERS}
        row["case_id"] = row["case_id"] or f"HV_{len(deduped) + 1:04d}"
        deduped.append(row)
    path = stability_root(root) / "outputs" / "high_volatility_cases.csv"
    write_csv_rows(path, HIGH_VOLATILITY_HEADERS, deduped)
    return path


def try_make_scatter(root: Path, filename: str, rows: list[dict[str, Any]], x_field: str, y_field: str, title: str) -> None:
    if not rows:
        write_plot_skip(root, filename, "no data")
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # noqa: BLE001 - optional plotting dependency.
        write_plot_skip(root, filename, "matplotlib unavailable")
        return
    x_values = [safe_float(row.get(x_field)) for row in rows]
    y_values = [safe_float(row.get(y_field)) for row in rows]
    plt.figure(figsize=(6, 4))
    plt.scatter(x_values, y_values)
    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(stability_root(root) / "plots" / filename, dpi=150)
    plt.close()


def try_make_hist(root: Path, filename: str, values: list[float], title: str, xlabel: str) -> None:
    if not values:
        write_plot_skip(root, filename, "no data")
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:  # noqa: BLE001 - optional plotting dependency.
        write_plot_skip(root, filename, "matplotlib unavailable")
        return
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=min(20, max(3, len(values))))
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(stability_root(root) / "plots" / filename, dpi=150)
    plt.close()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
