from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from llm_clients import OllamaClient, OllamaClientError
from scoring_utils import DEFAULT_ENV, SCORING_ROOT, ensure_directories


MINIMAL_SCHEMA = {
    "type": "object",
    "required": ["ok", "model"],
    "properties": {
        "ok": {"type": "boolean"},
        "model": {"type": "string"},
    },
    "additionalProperties": False,
}


def run_check(
    base_url: str = DEFAULT_ENV["OLLAMA_BASE_URL"],
    model: str = DEFAULT_ENV["OLLAMA_MODEL"],
    pull_if_missing: bool = False,
    root: Path = SCORING_ROOT,
) -> dict[str, Any]:
    ensure_directories(root)
    client = OllamaClient(base_url=base_url, model=model)
    health = client.health_check()
    report: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "ollama_available": health["available"],
        "ollama_version": health.get("version", ""),
        "model_available": False,
        "models": [],
        "minimal_json_test": "not_run",
        "errors": [],
        "manual_instruction": "",
    }
    if not health["available"]:
        report["errors"].append(health["error"])
        report["manual_instruction"] = f"Start Ollama, then run: ollama pull {model} or ollama run {model}"
        write_report(root, report)
        return report

    try:
        models = client.list_models()
        report["models"] = models
        report["model_available"] = model in models or any(item.split(":")[0] == model.split(":")[0] for item in models)
    except OllamaClientError as exc:
        report["errors"].append(str(exc))
        write_report(root, report)
        return report

    if not report["model_available"] and pull_if_missing:
        completed = subprocess.run(["ollama", "pull", model], text=True, capture_output=True)
        if completed.returncode == 0:
            report["models"] = client.list_models()
            report["model_available"] = model in report["models"]
        else:
            report["errors"].append(completed.stderr.strip() or completed.stdout.strip())

    if not report["model_available"]:
        report["manual_instruction"] = f"Run: ollama pull {model} or ollama run {model}"
        write_report(root, report)
        return report

    try:
        result = client.chat_json(
            [
                {
                    "role": "system",
                    "content": "Return JSON only.",
                },
                {
                    "role": "user",
                    "content": f'Return exactly this JSON shape with model "{model}": {{"ok": true, "model": "{model}"}}',
                },
            ],
            MINIMAL_SCHEMA,
            think=False,
        )
        parsed = result.parsed_json
        if isinstance(parsed, dict) and parsed.get("ok") is True and parsed.get("model"):
            report["minimal_json_test"] = "pass"
        else:
            report["minimal_json_test"] = "fail"
            report["errors"].append("Minimal JSON test did not return expected object.")
    except OllamaClientError as exc:
        report["minimal_json_test"] = "fail"
        report["errors"].append(str(exc))
    write_report(root, report)
    return report


def write_report(root: Path, report: dict[str, Any]) -> Path:
    path = root / "08_reports" / "ollama_check_report.md"
    lines = [
        "# Ollama Check Report",
        "",
        f"- base_url: {report['base_url']}",
        f"- model: {report['model']}",
        f"- Ollama available: {report['ollama_available']}",
        f"- Ollama version: {report.get('ollama_version', '')}",
        f"- qwen3:8b available: {report['model_available']}",
        f"- minimal JSON test: {report['minimal_json_test']}",
        f"- installed models: {', '.join(report.get('models', [])) or 'none'}",
        "",
    ]
    if report.get("errors"):
        lines.append("## Errors")
        lines.extend(f"- {item}" for item in report["errors"])
        lines.append("")
    if report.get("manual_instruction"):
        lines.append("## Next Step")
        lines.append(report["manual_instruction"])
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Check local Ollama/qwen3:8b availability.")
    parser.add_argument("--base-url", default=DEFAULT_ENV["OLLAMA_BASE_URL"])
    parser.add_argument("--model", default=DEFAULT_ENV["OLLAMA_MODEL"])
    parser.add_argument("--pull-if-missing", action="store_true")
    args = parser.parse_args()
    report = run_check(args.base_url, args.model, args.pull_if_missing)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
