from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from json_stabilization import parse_model_json
from scoring_utils import DEFAULT_ENV


class OllamaClientError(RuntimeError):
    pass


@dataclass
class OllamaCallResult:
    ok: bool
    raw_response: dict[str, Any]
    content: str
    parsed_json: Any | None
    fallback_used: bool
    error: str = ""


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int | float | None = None,
        temperature: int | float | None = None,
        seed: int | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
    ) -> None:
        self.base_url = (base_url or DEFAULT_ENV["OLLAMA_BASE_URL"]).rstrip("/")
        self.model = model or DEFAULT_ENV["OLLAMA_MODEL"]
        self.timeout = float(timeout if timeout is not None else DEFAULT_ENV["OLLAMA_TIMEOUT_SECONDS"])
        self.temperature = float(temperature if temperature is not None else DEFAULT_ENV["OLLAMA_TEMPERATURE"])
        self.seed = int(seed if seed is not None else DEFAULT_ENV["OLLAMA_SEED"])
        self.num_ctx = int(num_ctx if num_ctx is not None else DEFAULT_ENV["OLLAMA_NUM_CTX"])
        self.num_predict = int(num_predict if num_predict is not None else DEFAULT_ENV.get("OLLAMA_NUM_PREDICT", "256"))

    def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise OllamaClientError(f"HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise OllamaClientError(str(exc)) from exc
        if not body.strip():
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise OllamaClientError(f"Non-JSON Ollama response: {body[:500]}") from exc

    def health_check(self) -> dict[str, Any]:
        started = time.time()
        try:
            payload = self._request_json("GET", "/api/version")
            return {
                "available": True,
                "version": payload.get("version", ""),
                "error": "",
                "elapsed_seconds": round(time.time() - started, 3),
            }
        except OllamaClientError as exc:
            return {
                "available": False,
                "version": "",
                "error": str(exc),
                "elapsed_seconds": round(time.time() - started, 3),
            }

    def list_models(self) -> list[str]:
        payload = self._request_json("GET", "/api/tags")
        models = payload.get("models", [])
        names = []
        for item in models:
            name = item.get("name") or item.get("model")
            if name:
                names.append(name)
        return names

    def ensure_model_available(self) -> dict[str, Any]:
        health = self.health_check()
        if not health["available"]:
            return {"available": False, "model_available": False, "models": [], "error": health["error"]}
        try:
            models = self.list_models()
        except OllamaClientError as exc:
            return {"available": True, "model_available": False, "models": [], "error": str(exc)}
        model_available = self.model in models or any(name.split(":")[0] == self.model.split(":")[0] for name in models)
        return {"available": True, "model_available": model_available, "models": models, "error": ""}

    def _chat_payload(self, messages: list[dict[str, str]], schema: dict[str, Any], think: bool | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": schema,
            "keep_alive": -1,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if think is not None:
            payload["think"] = think
        return payload

    def runtime_config(self, *, think: bool = False, stream: bool = False, json_schema_mode: str = "ollama_format_schema") -> dict[str, Any]:
        return {
            "model_name": self.model,
            "llm_backend": "ollama",
            "base_url": self.base_url,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "think": think,
            "stream": stream,
            "json_schema_mode": json_schema_mode,
        }

    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        think: bool = False,
    ) -> OllamaCallResult:
        payload = self._chat_payload(messages, schema, think)
        fallback_used = False
        try:
            response = self._request_json("POST", "/api/chat", payload)
        except OllamaClientError as exc:
            if "think" not in str(exc).lower():
                raise
            payload = self._chat_payload(messages, schema, None)
            response = self._request_json("POST", "/api/chat", payload)
            fallback_used = True
        content = response.get("message", {}).get("content", "")
        parsed = _try_parse_json(content)
        return OllamaCallResult(
            ok=parsed is not None,
            raw_response=response,
            content=content,
            parsed_json=parsed,
            fallback_used=fallback_used,
            error="" if parsed is not None else "json_parse_failed",
        )

    def _generate_json(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> OllamaCallResult:
        prompt = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
        prompt += "\n\nReturn JSON only and match this JSON schema:\n" + json.dumps(schema, ensure_ascii=False)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
            },
        }
        response = self._request_json("POST", "/api/generate", payload)
        content = response.get("response", "")
        return OllamaCallResult(
            ok=_try_parse_json(content) is not None,
            raw_response=response,
            content=content,
            parsed_json=_try_parse_json(content),
            fallback_used=True,
            error="" if _try_parse_json(content) is not None else "json_parse_failed",
        )

    def repair_json(self, raw_text: str, schema: dict[str, Any]) -> OllamaCallResult:
        messages = [
            {
                "role": "system",
                "content": "Repair malformed JSON only. Do not change score meanings. Return JSON only.",
            },
            {
                "role": "user",
                "content": "Repair this text into JSON matching the schema.\n\nTEXT:\n"
                + raw_text
                + "\n\nSCHEMA:\n"
                + json.dumps(schema, ensure_ascii=False),
            },
        ]
        return self.chat_json(messages, schema, think=False)

    def score_document(self, prompt: str, schema: dict[str, Any]) -> OllamaCallResult:
        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON-only scoring assistant. Follow the research scope exactly.",
            },
            {"role": "user", "content": prompt},
        ]
        return self.chat_json(messages, schema, think=False)


def _try_parse_json(text: str) -> Any | None:
    parsed = parse_model_json(text or "")
    return parsed["payload"] if parsed.get("ok") else None


def extract_outer_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None
