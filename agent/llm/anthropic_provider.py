"""Stub for AnthropicProvider (Claude).

Reserved for a later phase. Construction always raises ``ModelUnavailable`` so
that misconfiguration surfaces immediately rather than silently routing to a
different backend.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from agent.llm.base import LLMMessage, LLMResponse, ModelUnavailable


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", **kwargs) -> None:
        raise ModelUnavailable(
            "AnthropicProvider is not implemented in MVP. "
            "Use provider='deepseek' or provider='mock'."
        )

    def complete(
        self,
        messages: List[LLMMessage],
        *,
        temperature: float = 0.3,
        max_output_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None,
    ) -> LLMResponse:  # pragma: no cover - never reached
        raise ModelUnavailable("AnthropicProvider is not implemented in MVP.")
