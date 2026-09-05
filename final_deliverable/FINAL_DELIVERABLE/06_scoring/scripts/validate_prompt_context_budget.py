from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PromptBudgetConfig:
    num_ctx: int = 4096
    reserved_output_tokens: int = 384
    safety_margin_tokens: int = 256
    post_call_threshold_ratio: float = 0.90

    @property
    def input_token_budget(self) -> int:
        return max(0, self.num_ctx - self.reserved_output_tokens - self.safety_margin_tokens)

    @property
    def post_call_prompt_eval_limit(self) -> int:
        return min(
            self.num_ctx - self.reserved_output_tokens,
            int(math.floor(self.num_ctx * self.post_call_threshold_ratio)),
        )


@dataclass(frozen=True)
class PromptBudgetReport:
    instructions_tokens: int
    schema_tokens: int
    selected_text_tokens: int
    metadata_tokens: int
    reserved_output_tokens: int
    safety_margin_tokens: int
    total_input_tokens: int
    total_expected_tokens: int
    input_token_budget: int
    num_ctx: int
    budget_passed: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PostCallPromptEvalReport:
    prompt_eval_count: int | None
    num_ctx: int
    reserved_output_tokens: int
    post_call_prompt_eval_limit: int
    prompt_eval_passed: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_tokens(text: str) -> int:
    """Conservative local estimate used before Ollama returns prompt_eval_count."""
    if not text:
        return 0
    token_like = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[\u4e00-\u9fff]|[^\s]", text)
    word_piece_estimate = len(token_like)
    char_floor = math.ceil(len(text) / 4)
    return max(word_piece_estimate, char_floor)


def validate_prompt_context_budget(
    *,
    instructions_text: str,
    schema_text: str,
    selected_text: str,
    metadata_text: str = "",
    config: PromptBudgetConfig | None = None,
) -> PromptBudgetReport:
    cfg = config or PromptBudgetConfig()
    instructions_tokens = estimate_tokens(instructions_text)
    schema_tokens = estimate_tokens(schema_text)
    selected_text_tokens = estimate_tokens(selected_text)
    metadata_tokens = estimate_tokens(metadata_text)
    total_input_tokens = instructions_tokens + schema_tokens + selected_text_tokens + metadata_tokens
    total_expected_tokens = total_input_tokens + cfg.reserved_output_tokens + cfg.safety_margin_tokens
    budget_passed = total_input_tokens <= cfg.input_token_budget and total_expected_tokens <= cfg.num_ctx
    return PromptBudgetReport(
        instructions_tokens=instructions_tokens,
        schema_tokens=schema_tokens,
        selected_text_tokens=selected_text_tokens,
        metadata_tokens=metadata_tokens,
        reserved_output_tokens=cfg.reserved_output_tokens,
        safety_margin_tokens=cfg.safety_margin_tokens,
        total_input_tokens=total_input_tokens,
        total_expected_tokens=total_expected_tokens,
        input_token_budget=cfg.input_token_budget,
        num_ctx=cfg.num_ctx,
        budget_passed=budget_passed,
        status="passed" if budget_passed else "context_budget_failed",
    )


def validate_post_call_prompt_eval(
    raw_response: dict[str, Any] | None,
    *,
    config: PromptBudgetConfig | None = None,
) -> PostCallPromptEvalReport:
    cfg = config or PromptBudgetConfig()
    prompt_eval_count = _safe_int((raw_response or {}).get("prompt_eval_count"))
    passed = prompt_eval_count is not None and prompt_eval_count < cfg.post_call_prompt_eval_limit
    return PostCallPromptEvalReport(
        prompt_eval_count=prompt_eval_count,
        num_ctx=cfg.num_ctx,
        reserved_output_tokens=cfg.reserved_output_tokens,
        post_call_prompt_eval_limit=cfg.post_call_prompt_eval_limit,
        prompt_eval_passed=passed,
        status="passed" if passed else "prompt_eval_too_high",
    )


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
