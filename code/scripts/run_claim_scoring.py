from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from claim_utils import CLAIM_SCORE_FIELDS, map_claim_to_dimensions, score_claim_with_rules
from json_stabilization import parse_model_json
from llm_clients import OllamaClient, OllamaClientError
from scoring_utils import DEFAULT_ENV, SCORING_ROOT, ensure_directories, save_json, write_csv_rows


def run_claim_scoring(
    root: Path = SCORING_ROOT,
    source_type: str = "mda",
    mock: bool = False,
    model: str = "qwen3:8b",
    limit: int | None = None,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    if source_type not in {"mda", "news"}:
        raise ValueError("source_type must be mda or news")
    ensure_directories(root)
    claims_dir = root / "10_claims" / f"{source_type}_claims"
    output_dir = root / "10_claim_scoring" / f"{source_type}_claim_scores"
    raw_dir = root / "10_claim_scoring" / "raw_model_outputs" / source_type
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(claims_dir.glob("*.json"))
    if limit is not None:
        paths = paths[:limit]
    client = None
    model_available = True
    if not mock:
        client = OllamaClient(model=model, timeout=timeout_seconds)
        status = client.ensure_model_available()
        model_available = bool(status.get("available") and status.get("model_available"))
        if not model_available:
            summary = {
                "source_type": source_type,
                "claim_count": 0,
                "success_count": 0,
                "failure_count": _count_claims(paths),
                "model_available": False,
                "mock_mode": False,
                "message": status.get("error", "Local qwen3:8b unavailable; no claim scores fabricated."),
            }
            save_json(root / "10_claim_scoring" / f"{source_type}_claim_scoring_summary.json", summary)
            return summary
    all_scores: list[dict[str, Any]] = []
    failure_count = 0
    started = time.perf_counter()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        document_id = payload.get("document_id", path.stem)
        claim_scores = []
        for claim in payload.get("claims", []):
            dimension = map_claim_to_dimensions(claim)[0]
            if mock:
                score = score_claim_with_rules(claim, dimension)
                raw_path = raw_dir / f"{claim['claim_id']}.json"
                save_json(raw_path, {"mock_mode": True, "claim": claim, "score": score})
                score.update(
                    {
                        "model_name": "deterministic_claim_proxy",
                        "llm_backend": "rules",
                        "parse_status": "parsed",
                        "schema_validation_status": "valid",
                        "raw_output_path": str(raw_path),
                    }
                )
            else:
                score, ok = _score_claim_with_llm(client, claim, dimension, raw_dir, model)
                failure_count += 0 if ok else 1
            claim_scores.append(score)
            all_scores.append(score)
        save_json(
            output_dir / f"{document_id}.json",
            {
                "document_id": document_id,
                "source_type": source_type,
                "claim_score_count": len(claim_scores),
                "claim_scores": claim_scores,
            },
        )
    write_csv_rows(root / "10_claim_scoring" / f"{source_type}_claim_scores_long.csv", CLAIM_SCORE_FIELDS, all_scores)
    duration = round(time.perf_counter() - started, 3)
    summary = {
        "source_type": source_type,
        "document_count": len(paths),
        "claim_count": len(all_scores),
        "success_count": sum(1 for row in all_scores if row.get("schema_validation_status") == "valid"),
        "failure_count": failure_count,
        "json_success_rate": round(sum(1 for row in all_scores if row.get("parse_status") == "parsed") / len(all_scores), 3) if all_scores else 0,
        "schema_validation_rate": round(sum(1 for row in all_scores if row.get("schema_validation_status") == "valid") / len(all_scores), 3) if all_scores else 0,
        "duration_seconds": duration,
        "model_available": None if mock else model_available,
        "mock_mode": mock,
        "model_name": "deterministic_claim_proxy" if mock else model,
    }
    save_json(root / "10_claim_scoring" / f"{source_type}_claim_scoring_summary.json", summary)
    return summary


def _score_claim_with_llm(client: OllamaClient | None, claim: dict[str, Any], dimension: str, raw_dir: Path, model: str) -> tuple[dict[str, Any], bool]:
    assert client is not None
    schema = {
        "type": "object",
        "required": ["claim_id", "dimension", "score", "reason", "evidence_locator", "confidence"],
        "properties": {
            "claim_id": {"type": "string"},
            "dimension": {"type": "string"},
            "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
            "reason": {"type": "string"},
            "evidence_locator": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        },
        "additionalProperties": False,
    }
    prompt = (
        "Score this one claim only. Return a compact JSON object.\n"
        f"Dimension: {dimension}\n"
        "For AR02, judge only internal numeric consistency and logical traceability inside the supplied claim.\n"
        f"Claim:\n{json.dumps(claim, ensure_ascii=False, indent=2)}"
    )
    raw_path = raw_dir / f"{claim['claim_id']}.json"
    try:
        result = client.score_document(prompt, schema)
        parsed = parse_model_json(result.content)
        save_json(raw_path, {"claim": claim, "content": result.content, "raw_response": result.raw_response, "parsed": parsed.get("payload")})
    except OllamaClientError as exc:
        save_json(raw_path, {"claim": claim, "error": str(exc)})
        return _failed_score_row(claim, dimension, raw_path, model, str(exc)), False
    payload = parsed.get("payload") if parsed.get("ok") else None
    if not isinstance(payload, dict):
        return _failed_score_row(claim, dimension, raw_path, model, "json_parse_failed"), False
    issues = _validate_score_payload(payload, claim, dimension)
    if issues:
        return _failed_score_row(claim, dimension, raw_path, model, ";".join(issues), parse_status="parsed"), False
    payload.update(
        {
            "document_id": claim["document_id"],
            "source_type": claim["source_type"],
            "model_name": model,
            "llm_backend": "ollama",
            "parse_status": "parsed",
            "schema_validation_status": "valid",
            "raw_output_path": str(raw_path),
        }
    )
    return payload, True


def _failed_score_row(
    claim: dict[str, Any],
    dimension: str,
    raw_path: Path,
    model: str,
    reason: str,
    parse_status: str = "parse_failed",
) -> dict[str, Any]:
    return {
        "claim_id": claim.get("claim_id", ""),
        "document_id": claim.get("document_id", ""),
        "source_type": claim.get("source_type", ""),
        "dimension": dimension,
        "score": "",
        "reason": reason,
        "evidence_locator": claim.get("evidence_locator", ""),
        "confidence": "low",
        "model_name": model,
        "llm_backend": "ollama",
        "parse_status": parse_status,
        "schema_validation_status": "invalid",
        "raw_output_path": str(raw_path),
    }


def _validate_score_payload(payload: dict[str, Any], claim: dict[str, Any], dimension: str) -> list[str]:
    issues = []
    if payload.get("claim_id") != claim.get("claim_id"):
        issues.append("claim_id_mismatch")
    if payload.get("dimension") != dimension:
        issues.append("dimension_mismatch")
    if payload.get("score") not in {1, 2, 3, 4, 5}:
        issues.append("score_out_of_range")
    if payload.get("confidence") not in {"high", "medium", "low"}:
        issues.append("invalid_confidence")
    return issues


def _count_claims(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += len(json.loads(path.read_text(encoding="utf-8")).get("claims", []))
        except json.JSONDecodeError:
            continue
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Score extracted claims one claim at a time.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--source-type", choices=["mda", "news"], default="mda")
    parser.add_argument("--model", default=DEFAULT_ENV["OLLAMA_MODEL"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_claim_scoring(args.root, args.source_type, args.mock, args.model, args.limit), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
