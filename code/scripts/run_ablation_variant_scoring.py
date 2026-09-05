#!/usr/bin/env python3
"""
Run ablation variant scoring: same pipeline, prompt wording perturbed.

Only the ablation sample documents (08_reports/ablation_sample.csv) are scored.

Usage:
    python run_ablation_variant_scoring.py --root . --tone strict
    python run_ablation_variant_scoring.py --root . --tone lenient

Output:
    06_ratings/mda_ablation_{tone}/
    05_raw_model_outputs/mda_ablation_{tone}/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from llm_clients import OllamaClient, OllamaClientError
from run_single_dimension_scoring import _save_raw, _single_dimension_prompt_parts
from run_scoring import _build_document_score_row, _build_rating_rows
from validate_json_outputs import simple_dimension_schema
from scoring_utils import (
    DEFAULT_ENV,
    RATINGS_LONG_HEADERS,
    SCORING_ROOT,
    dimensions_for,
    document_headers_for,
    ensure_directories,
    read_csv_rows,
    resolve_text_path,
    save_json,
    write_csv_rows,
)


TONE_INSTRUCTIONS = {
    # Extreme directive perturbations (explicit score-direction commands).
    "strict": (
        "Scoring tone instruction: Be STRICT. Only award high scores when the text provides "
        "explicit, concrete, verifiable evidence for the dimension. When in doubt, score LOWER.\n"
    ),
    "lenient": (
        "Scoring tone instruction: Be LENIENT. Focus on overall information quality rather than "
        "specific missing details. When in doubt, score HIGHER.\n"
    ),
    # Mild rephrasing perturbations: same strict/lenient spirit, but descriptive standards
    # WITHOUT explicit score-direction commands (realistic rubric wording variation).
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


def build_variant_prompt(
    doc_type: str,
    document_id: str,
    metadata: dict[str, str],
    dimension: dict[str, Any],
    text: str,
    tone: str,
    numeric_result: dict[str, Any] | None = None,
) -> str:
    """Build the PRODUCTION prompt via _single_dimension_prompt_parts and inject
    the tone perturbation at a fixed, documented position (right before the
    'Valid response example:' line). Everything else is byte-identical to the
    production scoring prompt."""
    prompt, _instructions, _metadata_text = _single_dimension_prompt_parts(
        doc_type, document_id, metadata, dimension, text, numeric_result)
    tone_text = TONE_INSTRUCTIONS.get(tone, "")
    marker = "Valid response example:"
    if marker in prompt:
        prompt = prompt.replace(marker, tone_text + marker, 1)
    return prompt


def score_variant(
    root: Path,
    tone: str,
    doc_ids: list[str],
    model: str = DEFAULT_ENV["OLLAMA_MODEL"],
    resume: bool = True,
) -> dict[str, Any]:
    """Score ablation sample documents with the tone-perturbed prompt."""
    output_name = f"mda_ablation_{tone}"
    ratings_prefix = f"mda_ablation_{tone}"
    ensure_directories(root)
    out_dir = root / "06_ratings" / output_name
    raw_dir = root / "05_raw_model_outputs" / output_name
    (out_dir / "per_dimension").mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    registry = {r["document_id"]: r for r in read_csv_rows(root / "01_registry" / "mda_registry.csv")}
    client = OllamaClient(model=model, temperature=0, seed=42, num_ctx=4096)

    if not client.ensure_model_available().get("available"):
        return {"status": "blocked", "reason": "ollama_unavailable"}

    dimensions = dimensions_for("mda")
    rating_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []

    total_calls = len(doc_ids) * len(dimensions)
    done = 0

    context_dir = root / "02_extracted_text" / "mda_dimension_context"
    for doc_id in doc_ids:
        row = registry.get(doc_id, {})
        text_path = resolve_text_path(row, "mda", root)
        if not text_path.exists():
            continue
        text = text_path.read_text(encoding="utf-8", errors="replace")

        doc_dim_scores: list[dict[str, Any]] = []
        for dimension in dimensions:
            done += 1
            code = dimension["code"]
            # Use the SAME budgeted per-dimension context as production scoring
            # (context files pre-built by the production pipeline).
            ctx_path = context_dir / f"{doc_id}_{code}.txt"
            scoring_text = (
                ctx_path.read_text(encoding="utf-8", errors="replace")
                if ctx_path.exists() else text)
            parsed_path = out_dir / "per_dimension" / f"{doc_id}_{code}_parsed.json"
            if resume and parsed_path.exists():
                try:
                    existing = json.loads(parsed_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "success" and existing.get("payload"):
                        print(f"[{done}/{total_calls}] {doc_id} {code}: skip (cached)")
                        doc_dim_scores.append(existing["payload"])
                        continue
                except Exception:
                    pass

            print(f"[{done}/{total_calls}] {doc_id} {code} ({tone})...", end=" ", flush=True)
            prompt = build_variant_prompt("mda", doc_id, row, dimension, scoring_text, tone)
            schema = simple_dimension_schema("mda")
            attempts_used = 0
            payload = None
            last_issues = []
            raw_path = None
            for attempt in range(1, 3):
                attempts_used = attempt
                try:
                    attempt_prompt = prompt
                    if attempt > 1:
                        attempt_prompt += (
                            "\n\nRetry: return exactly one JSON object with only score, evidence, reason.")
                    result = client.score_document(attempt_prompt, schema)
                    raw_path = _save_raw(root, output_name, doc_id, code, attempt, {
                        "model_name": model,
                        "temperature": 0,
                        "seed": 42,
                        "attempt": attempt,
                        "content": result.content,
                        "parsed_simple": result.parsed_json if isinstance(result.parsed_json, dict) else None,
                    })
                    if isinstance(result.parsed_json, dict):
                        simple = result.parsed_json
                        score = simple.get("score")
                        evidence = simple.get("evidence", "")
                        reason = simple.get("reason", "")
                        if isinstance(score, int) and 1 <= score <= 5 and reason:
                            payload = {
                                "dimension_code": code,
                                "dimension_name": dimension["name"],
                                "raw_score": score,
                                "reason": reason,
                                "evidence_locator": f"[{evidence}]" if not evidence.startswith("[") else evidence,
                                "evidence_text_short": reason[:240],
                                "confidence_level": "medium",
                                "missing_type": "none",
                                "comment_short": reason[:240],
                            }
                            break
                    last_issues = ["json_parse_failed"]
                except OllamaClientError as exc:
                    last_issues = [str(exc)]
            if payload is None:
                payload = {
                    "dimension_code": code,
                    "dimension_name": dimension["name"],
                    "raw_score": 3,
                    "reason": f"fallback after errors: {';'.join(last_issues)[:100]}",
                    "evidence_locator": "[MDA_P001]",
                    "evidence_text_short": "fallback",
                    "confidence_level": "low",
                    "missing_type": "none",
                    "comment_short": "fallback",
                }
                print(f"FALLBACK (score=3)")
            else:
                print(f"score={payload['raw_score']}")

            save_json(parsed_path, {
                "document_id": doc_id,
                "doc_type": "mda",
                "dimension_code": code,
                "status": "success",
                "payload": payload,
                "tone_variant": tone,
            })
            doc_dim_scores.append(payload)

        # Aggregate document-level score
        normalized = {
            "document_id": doc_id,
            "doc_type": "mda",
            "scoring_basis": "mda_disclosure_quality",
            "scorebook_version": "2026-06-15",
            "dimension_scores": doc_dim_scores,
            "flags": {},
            "overall_comment": f"ablation_{tone}",
            "needs_review": False,
            "review_reasons": [],
        }
        doc_score_row = _build_document_score_row(normalized, row, "mda", model)
        document_rows.append(doc_score_row)

        ts = "ablation_run"
        n_dim = len(doc_dim_scores)
        doc_rating_rows = _build_rating_rows(
            normalized, "mda", ts, ts, model,
            str(raw_path) if raw_path else "", "", "ok", "ok")
        # Mark ablation provenance (prompt variant) for traceability
        for rr in doc_rating_rows[-n_dim:] if n_dim else []:
            rr["review_round"] = "ablation"
            rr["reviewer_id"] = f"local_{model.replace(':', '_')}_{tone}"
        rating_rows.extend(doc_rating_rows)

    write_csv_rows(out_dir / f"{ratings_prefix}_ratings_long.csv", RATINGS_LONG_HEADERS, rating_rows)
    write_csv_rows(out_dir / f"{ratings_prefix}_document_scores.csv", document_headers_for("mda"), document_rows)
    save_json(out_dir / f"{ratings_prefix}_summary.json", {
        "tone": tone,
        "documents_scored": len(document_rows),
        "ratings": len(rating_rows),
    })
    return {"status": "success", "documents_scored": len(document_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score ablation samples with tone-perturbed prompts.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--tone",
                        choices=["strict", "lenient", "mild_strict", "mild_lenient"],
                        required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    # Load sample IDs
    import csv
    sample_path = args.root / "08_reports" / "ablation_sample.csv"
    with open(sample_path, encoding="utf-8-sig") as f:
        doc_ids = [r["document_id"] for r in csv.DictReader(f)]
    print(f"Sample: {len(doc_ids)} documents, tone={args.tone}")

    result = score_variant(args.root, args.tone, doc_ids, resume=not args.no_resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
