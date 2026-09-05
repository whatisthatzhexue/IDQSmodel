from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scoring_utils import DEFAULT_ENV, SCORING_ROOT, save_json
from stability_common import write_markdown


BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:8b"
KEYWORDS = ["error", "panic", "runner", "memory", "metal", "out of memory", "failed to generate", "model failed", "502", "timeout", "signal", "crash"]
MINIMAL_SCHEMA = {
    "type": "object",
    "required": ["ok", "model"],
    "properties": {
        "ok": {"type": "boolean"},
        "model": {"type": "string"},
    },
    "additionalProperties": False,
}


def repair_ollama_runtime(
    root: Path = SCORING_ROOT,
    model: str = MODEL,
    base_url: str = BASE_URL,
    max_repair_rounds: int = 3,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "base_url": base_url,
        "model": model,
        "runtime_healthy": False,
        "api_chat_healthy": False,
        "openai_compatible_healthy": False,
        "attempts": [],
        "blocking_reasons": [],
    }
    diagnosis = _diagnose(root, base_url)
    report["diagnosis"] = diagnosis
    _copy_log_tails(root)

    for round_idx in range(max(0, max_repair_rounds) + 1):
        if round_idx > 0:
            _safe_repair_round(model, base_url, round_idx)
        chat = _run_three_chat_tests(base_url, model, timeout_seconds)
        report["attempts"].append({"round": round_idx, "endpoint": "/api/chat", **chat})
        if chat["passed"]:
            report["runtime_healthy"] = True
            report["api_chat_healthy"] = True
            break
        if _has_502(chat) and round_idx == 0 and max_repair_rounds > 0:
            time.sleep(5)
        if round_idx == max_repair_rounds:
            compat = _openai_compatible_test(base_url, model, timeout_seconds)
            report["attempts"].append({"round": round_idx, "endpoint": "/v1/chat/completions", **compat})
            if compat["passed"]:
                report["runtime_healthy"] = True
                report["openai_compatible_healthy"] = True

    if not report["runtime_healthy"]:
        report["blocking_reasons"].append("ollama_runtime_unhealthy")
    out_dir = root / "PAPER_OUTPUT" / "00_status"
    save_json(out_dir / "ollama_runtime_status.json", report)
    save_json(root / "08_reports" / "ollama_runtime_diagnosis.json", diagnosis)
    _write_diagnosis(root / "08_reports" / "ollama_runtime_diagnosis.md", diagnosis)
    _write_repair_report(out_dir / "ollama_repair_report.md", report)
    return report


def _diagnose(root: Path, base_url: str) -> dict[str, Any]:
    return {
        "ollama_version_command": _run_cmd(["ollama", "--version"]),
        "macos_version": _run_cmd(["sw_vers"]),
        "machine_architecture": platform.machine(),
        "available_memory": _run_cmd(["vm_stat"]),
        "ollama_list": _run_cmd(["ollama", "list"]),
        "ollama_ps_command": _run_cmd(["ollama", "ps"]),
        "api_version": _http_json(base_url, "/api/version", timeout=5),
        "api_tags": _http_json(base_url, "/api/tags", timeout=5),
        "api_ps": _http_json(base_url, "/api/ps", timeout=5),
        "log_keyword_hits": _log_keyword_hits(),
    }


def _run_three_chat_tests(base_url: str, model: str, timeout_seconds: int) -> dict[str, Any]:
    results = []
    for idx in range(3):
        results.append(_chat_test(base_url, model, timeout_seconds, idx + 1))
        if not results[-1]["http_200"] or not results[-1]["json_parse_success"] or not results[-1]["schema_success"]:
            break
    return {"passed": len(results) == 3 and all(item["http_200"] and item["json_parse_success"] and item["schema_success"] for item in results), "tests": results}


def _chat_test(base_url: str, model: str, timeout_seconds: int, attempt: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": f'Return exactly this JSON: {{"ok": true, "model": "{model}"}}'},
        ],
        "stream": False,
        "think": False,
        "format": MINIMAL_SCHEMA,
        "keep_alive": -1,
        "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 256},
    }
    status, data, raw = _post_json(f"{base_url}/api/chat", payload, timeout_seconds)
    content = data.get("message", {}).get("content", "") if isinstance(data, dict) else ""
    parsed = _parse_json(content)
    return {
        "attempt": attempt,
        "http_status": status,
        "http_200": status == 200,
        "json_parse_success": isinstance(parsed, dict),
        "schema_success": isinstance(parsed, dict) and parsed.get("ok") is True and parsed.get("model") == model,
        "raw_excerpt": raw[:500],
    }


def _openai_compatible_test(base_url: str, model: str, timeout_seconds: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": f'Return exactly this JSON: {{"ok": true, "model": "{model}"}}'},
        ],
        "stream": False,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "minimal_ollama_test", "strict": True, "schema": MINIMAL_SCHEMA},
        },
    }
    status, data, raw = _post_json(f"{base_url}/v1/chat/completions", payload, timeout_seconds)
    content = ""
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices:
            content = choices[0].get("message", {}).get("content", "")
    parsed = _parse_json(content)
    return {
        "passed": status == 200 and isinstance(parsed, dict) and parsed.get("ok") is True and parsed.get("model") == model,
        "tests": [
            {
                "http_status": status,
                "http_200": status == 200,
                "json_parse_success": isinstance(parsed, dict),
                "schema_success": isinstance(parsed, dict) and parsed.get("ok") is True and parsed.get("model") == model,
                "raw_excerpt": raw[:500],
            }
        ],
    }


def _safe_repair_round(model: str, base_url: str, round_idx: int) -> None:
    if round_idx == 1:
        time.sleep(5)
    elif round_idx == 2:
        time.sleep(15)
    elif round_idx >= 3:
        _run_cmd(["ollama", "stop", model], timeout=30)
        _wait_api(base_url, 60)
        _run_cmd(["open", "-a", "Ollama"], timeout=10)
        _wait_api(base_url, 60)


def _wait_api(base_url: str, seconds: int) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        payload = _http_json(base_url, "/api/version", timeout=3)
        if payload.get("ok"):
            return
        time.sleep(2)


def _has_502(result: dict[str, Any]) -> bool:
    return any(test.get("http_status") == 502 or "502" in test.get("raw_excerpt", "") for test in result.get("tests", []))


def _http_json(base_url: str, path: str, timeout: int) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return {"ok": True, "status": response.status, "payload": json.loads(raw) if raw.strip() else {}}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> tuple[int, dict[str, Any], str]:
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw) if raw.strip() else {}, raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, {}, raw
    except Exception as exc:  # noqa: BLE001
        return 0, {}, str(exc)


def _parse_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _run_cmd(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    if not shutil.which(cmd[0]):
        return {"ok": False, "cmd": cmd, "returncode": None, "stdout": "", "stderr": f"{cmd[0]} not found"}
    try:
        completed = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return {"ok": completed.returncode == 0, "cmd": cmd, "returncode": completed.returncode, "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]}
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "cmd": cmd, "returncode": None, "stdout": exc.stdout or "", "stderr": "timeout"}


def _copy_log_tails(root: Path) -> None:
    log_dir = Path.home() / ".ollama" / "logs"
    targets = {
        "server.log": root / "08_reports" / "ollama_server_log_tail.txt",
        "app.log": root / "08_reports" / "ollama_app_log_tail.txt",
    }
    for name, target in targets.items():
        source = log_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            target.write_text("\n".join(source.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]), encoding="utf-8")
        else:
            target.write_text(f"{source} not found\n", encoding="utf-8")


def _log_keyword_hits() -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for name in ["server.log", "app.log"]:
        source = Path.home() / ".ollama" / "logs" / name
        if not source.exists():
            hits[name] = []
            continue
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()[-500:]
        lowered_hits = [line for line in lines if any(keyword in line.lower() for keyword in KEYWORDS)]
        hits[name] = lowered_hits[-50:]
    return hits


def _write_diagnosis(path: Path, diagnosis: dict[str, Any]) -> None:
    lines = [
        "# Ollama Runtime Diagnosis",
        "",
        f"- api_version_ok: {str(diagnosis.get('api_version', {}).get('ok')).lower()}",
        f"- api_tags_ok: {str(diagnosis.get('api_tags', {}).get('ok')).lower()}",
        f"- api_ps_ok: {str(diagnosis.get('api_ps', {}).get('ok')).lower()}",
        f"- architecture: {diagnosis.get('machine_architecture', '')}",
        f"- server_log_keyword_hits: {len(diagnosis.get('log_keyword_hits', {}).get('server.log', []))}",
        f"- app_log_keyword_hits: {len(diagnosis.get('log_keyword_hits', {}).get('app.log', []))}",
    ]
    write_markdown(path, lines)


def _write_repair_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Ollama Repair Report",
        "",
        f"- runtime_healthy: {str(report['runtime_healthy']).lower()}",
        f"- api_chat_healthy: {str(report['api_chat_healthy']).lower()}",
        f"- openai_compatible_healthy: {str(report['openai_compatible_healthy']).lower()}",
        f"- blocking_reasons: {', '.join(report['blocking_reasons']) or 'none'}",
        "",
        "## Attempts",
    ]
    for attempt in report["attempts"]:
        lines.append(f"- round={attempt['round']} endpoint={attempt['endpoint']} passed={attempt.get('passed')}")
    if not report["runtime_healthy"]:
        lines.extend(["", "Scoring is blocked. No fake scores should be generated while qwen3:8b is unavailable."])
    write_markdown(path, lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose and repair local Ollama qwen3:8b runtime.")
    parser.add_argument("--root", type=Path, default=SCORING_ROOT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--max-repair-rounds", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(repair_ollama_runtime(args.root, args.model, args.base_url, args.max_repair_rounds, args.timeout_seconds), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
