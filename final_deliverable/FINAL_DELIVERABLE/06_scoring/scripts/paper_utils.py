from __future__ import annotations

import csv
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

from scoring_utils import DIMENSIONS, SCORING_ROOT, ensure_directories, read_csv_rows, save_json, write_csv_rows
from stability_common import safe_float, write_markdown


PAPER_OUTPUT_DIRS = [
    "PAPER_OUTPUT/00_status",
    "PAPER_OUTPUT/01_methods",
    "PAPER_OUTPUT/02_data",
    "PAPER_OUTPUT/03_results",
    "PAPER_OUTPUT/04_stability",
    "PAPER_OUTPUT/05_quality_control",
    "PAPER_OUTPUT/06_cross_validation",
    "PAPER_OUTPUT/07_limitations",
    "PAPER_OUTPUT/08_appendix",
    "PAPER_OUTPUT/09_reproducibility",
    "PAPER_OUTPUT/tables",
    "PAPER_OUTPUT/figures",
]

PAPER_STATUS_FIELDS = [
    "paper_ready",
    "freeze_allowed",
    "mda_structural_ready",
    "mda_semantic_ready",
    "mda_human_validation_ready",
    "mda_ready",
    "news_ready",
    "cross_validation_ready",
    "cross_validation_pipeline_ready",
    "cross_validation_empirical_ready",
    "mock_mode",
    "systemic_conflict_flag",
    "reproducibility_ready",
    "mda_final_dataset_size",
    "news_final_dataset_size",
    "news_company_focused_dataset_size",
    "synthetic_news_count",
    "claim_link_count",
    "eligible_company_year_pair_count",
    "mda_stability_success_rate",
    "system_level_error_rate",
    "blocking_reasons",
    "advisory_warnings",
    "recommended_next_actions",
]

REQUIRED_PAPER_FILES = [
    "PAPER_OUTPUT/00_status/paper_readiness_check.json",
    "PAPER_OUTPUT/01_methods/methods_section_en.md",
    "PAPER_OUTPUT/01_methods/scoring_framework_en.md",
    "PAPER_OUTPUT/02_data/data_section_en.md",
    "PAPER_OUTPUT/03_results/results_summary.md",
    "PAPER_OUTPUT/04_stability/stability_section_en.md",
    "PAPER_OUTPUT/05_quality_control/quality_control_section_en.md",
    "PAPER_OUTPUT/06_cross_validation/cross_validation_section_en.md",
    "PAPER_OUTPUT/07_limitations/limitations_en.md",
    "PAPER_OUTPUT/09_reproducibility/reproducibility_guide.md",
]


def ensure_paper_output_dirs(root: Path = SCORING_ROOT) -> Path:
    ensure_directories(root)
    for item in PAPER_OUTPUT_DIRS:
        (root / item).mkdir(parents=True, exist_ok=True)
    return root / "PAPER_OUTPUT"


def load_json_safe(path: Path) -> tuple[dict[str, Any], bool]:
    if not path.exists():
        return {}, False
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_invalid_json": str(path)}, False
    return loaded if isinstance(loaded, dict) else {"_wrong_root_type": str(path)}, True


def read_text_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "length": 0}
    text = path.read_text(encoding="utf-8", errors="replace")
    return {"path": str(path), "exists": True, "length": len(text)}


def paper_status_path(root: Path = SCORING_ROOT) -> Path:
    return root / "PAPER_OUTPUT" / "00_status" / "paper_readiness_check.json"


def load_paper_status(root: Path = SCORING_ROOT) -> dict[str, Any]:
    status, ok = load_json_safe(paper_status_path(root))
    if ok:
        return normalize_status(status)
    return normalize_status({})


def normalize_status(status: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "paper_ready": False,
        "freeze_allowed": False,
        "mda_structural_ready": False,
        "mda_semantic_ready": False,
        "mda_human_validation_ready": False,
        "mda_ready": False,
        "news_ready": False,
        "cross_validation_ready": False,
        "cross_validation_pipeline_ready": False,
        "cross_validation_empirical_ready": False,
        "mock_mode": False,
        "systemic_conflict_flag": False,
        "reproducibility_ready": False,
        "mda_final_dataset_size": 0,
        "news_final_dataset_size": 0,
        "news_company_focused_dataset_size": 0,
        "synthetic_news_count": 0,
        "claim_link_count": 0,
        "eligible_company_year_pair_count": 0,
        "mda_stability_success_rate": None,
        "system_level_error_rate": None,
        "blocking_reasons": [],
        "advisory_warnings": [],
        "recommended_next_actions": [],
    }
    for field in PAPER_STATUS_FIELDS:
        if field in status:
            normalized[field] = status[field]
    normalized["blocking_reasons"] = list(normalized.get("blocking_reasons") or [])
    normalized["advisory_warnings"] = list(normalized.get("advisory_warnings") or [])
    normalized["recommended_next_actions"] = list(normalized.get("recommended_next_actions") or [])
    return normalized


def write_status(root: Path, status: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_status(status)
    save_json(paper_status_path(root), normalized)
    save_json(root / "PAPER_OUTPUT" / "PAPER_READY_STATUS.json", normalized)
    return normalized


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    if not rows:
        rows = [{header: "" for header in headers}]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row.get(header, "")).replace("\n", " ") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_table(root: Path, name: str, headers: list[str], rows: list[dict[str, Any]], note: str = "") -> dict[str, str]:
    table_dir = ensure_paper_output_dirs(root) / "tables"
    csv_path = table_dir / f"{name}.csv"
    md_path = table_dir / f"{name}.md"
    write_csv_rows(csv_path, headers, rows)
    md_body = markdown_table(headers, rows)
    if note:
        md_body += "\n" + note.strip() + "\n"
    write_markdown(md_path, [md_body.rstrip()])
    return {"csv": str(csv_path), "markdown": str(md_path)}


def score_summary(rows: list[dict[str, str]], score_field: str = "total_score_100") -> dict[str, Any]:
    values = [safe_float(row.get(score_field, ""), math.nan) for row in rows]
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return {"N": 0, "mean": "", "median": "", "std": "", "min": "", "max": "", "p25": "", "p75": ""}
    sorted_values = sorted(values)
    return {
        "N": len(values),
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "std": round(statistics.pstdev(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "p25": round(percentile(sorted_values, 25), 6),
        "p75": round(percentile(sorted_values, 75), 6),
    }


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * pct / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def dimension_summary(rows: list[dict[str, str]], dimensions: list[str]) -> list[dict[str, Any]]:
    out = []
    for dimension in dimensions:
        values = [safe_float(row.get(dimension, ""), math.nan) for row in rows]
        values = [value for value in values if not math.isnan(value)]
        if not values:
            out.append({"dimension": dimension, "N": 0, "mean": "", "median": "", "std": "", "min": "", "max": ""})
        else:
            out.append(
                {
                    "dimension": dimension,
                    "N": len(values),
                    "mean": round(statistics.mean(values), 6),
                    "median": round(statistics.median(values), 6),
                    "std": round(statistics.pstdev(values), 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                }
            )
    return out


def grade_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for row in rows:
        label = row.get("grade_label", "").strip().upper()
        if label in counts:
            counts[label] += 1
            continue
        score = safe_float(row.get("total_score_100", ""), math.nan)
        if math.isnan(score):
            continue
        if score >= 85:
            counts["A"] += 1
        elif score >= 70:
            counts["B"] += 1
        elif score >= 55:
            counts["C"] += 1
        else:
            counts["D"] += 1
    return counts


def news_registry_path(root: Path = SCORING_ROOT) -> Path:
    rebuilt = root / "01_registry" / "news_registry_rebuilt.csv"
    canonical = root / "01_registry" / "news_registry.csv"
    rebuilt_rows = read_csv_rows(rebuilt)
    canonical_rows = read_csv_rows(canonical)
    rebuilt_real = _real_included_registry_rows(rebuilt_rows)
    canonical_real = _real_included_registry_rows(canonical_rows)
    if rebuilt.exists() and rebuilt_rows and rebuilt_real:
        selected = rebuilt
        reason = "selected rebuilt registry because it has real include_flag=Yes rows"
    elif rebuilt.exists():
        selected = canonical
        reason = "fallback to canonical registry because rebuilt registry is empty or has no real include_flag=Yes rows"
    else:
        selected = canonical
        reason = "selected canonical registry because rebuilt registry is absent"
    _write_news_registry_selection_report(
        root,
        selected=selected,
        reason=reason,
        rebuilt_rows=len(rebuilt_rows),
        rebuilt_real=len(rebuilt_real),
        canonical_rows=len(canonical_rows),
        canonical_real=len(canonical_real),
    )
    return selected


def read_news_registry_rows(root: Path = SCORING_ROOT) -> list[dict[str, str]]:
    return read_csv_rows(news_registry_path(root))


def _real_included_registry_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("include_flag", "").strip().lower() == "yes" and not _registry_row_is_synthetic(row)
    ]


def _write_news_registry_selection_report(
    root: Path,
    *,
    selected: Path,
    reason: str,
    rebuilt_rows: int,
    rebuilt_real: int,
    canonical_rows: int,
    canonical_real: int,
) -> None:
    report = root / "08_reports" / "news_registry_selection_report.md"
    lines = [
        "# News Registry Selection Report",
        "",
        f"- selected_registry: {selected.relative_to(root) if selected.is_relative_to(root) else selected}",
        f"- selection_reason: {reason}",
        f"- rebuilt_registry_rows: {rebuilt_rows}",
        f"- rebuilt_real_include_rows: {rebuilt_real}",
        f"- canonical_registry_rows: {canonical_rows}",
        f"- canonical_real_include_rows: {canonical_real}",
        "",
        "This guard prevents an empty rebuilt registry from freezing real News outputs to zero rows.",
    ]
    write_markdown(report, lines)


def _registry_row_is_synthetic(row: dict[str, str]) -> bool:
    synthetic_flag = row.get("synthetic_flag", "").strip().lower()
    notes = row.get("notes", "").lower()
    return synthetic_flag in {"1", "true", "yes"} or "synthetic_flag=true" in notes


def registry_synthetic_count(root: Path = SCORING_ROOT) -> int:
    return sum(_registry_row_is_synthetic(row) for row in read_news_registry_rows(root))


def real_news_rows(root: Path = SCORING_ROOT) -> list[dict[str, str]]:
    return [
        row
        for row in read_news_registry_rows(root)
        if row.get("include_flag", "").lower() == "yes" and not _registry_row_is_synthetic(row)
    ]


def source_fields_present(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    required_sets = [["title"], ["source_name"], ["publish_date"]]
    return all(any(row.get(field, "").strip() for field in fields) for row in rows for fields in required_sets)


def environment_summary() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def file_manifest(root: Path, output_path: Path) -> None:
    rows = []
    for path in sorted((root / "PAPER_OUTPUT").rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size})
    write_csv_rows(output_path, ["relative_path", "size_bytes"], rows)


def write_dual_language_sections(
    root: Path,
    subdir: str,
    cn_name: str,
    en_name: str,
    cn_lines: list[str],
    en_lines: list[str],
) -> dict[str, str]:
    ensure_paper_output_dirs(root)
    cn_path = root / "PAPER_OUTPUT" / subdir / cn_name
    en_path = root / "PAPER_OUTPUT" / subdir / en_name
    write_markdown(cn_path, cn_lines)
    write_markdown(en_path, en_lines)
    return {"cn": str(cn_path), "en": str(en_path)}


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = list(reader)
    return max(0, len(rows) - 1)


def paper_ready_allowed_text(status: dict[str, Any]) -> str:
    if status.get("paper_ready"):
        return "Final empirical results are allowed."
    return "Final empirical conclusions are not allowed; use methodology, pipeline validation, limitations, and reproducibility materials only."


def dimensions_table(doc_type: str) -> list[dict[str, Any]]:
    return [{"dimension": item["code"], "name": item["name"], "main_weight": item["weight"]} for item in DIMENSIONS[doc_type]]
